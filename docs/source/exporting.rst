.. _exporting:

Exporting Models
================

A trained :class:`~chemprop.models.model.MPNN` or :class:`~chemprop.models.multi.MulticomponentMPNN` can be exported with :func:`torch.export.export` or converted to ONNX with :func:`torch.onnx.export`, so it can run without Chemprop. This page covers the Chemprop-specific steps. It assumes you are familiar with exporting PyTorch modules in general. Not every model can be exported yet; see :ref:`export-limitations`.

Register ``BatchMolGraph``
--------------------------

``torch.export`` only accepts tensors and registered container types, so register ``BatchMolGraph`` once before exporting:

.. code-block:: python

    from chemprop.utils.export import register_batch_mol_graph_pytree

    register_batch_mol_graph_pytree()

Make an example batch
---------------------

Export traces the model on an example input. Put the model in evaluation mode and featurize a few molecules:

.. code-block:: python

    import torch

    from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
    from chemprop.models import MPNN

    model = MPNN.load_from_file("model.pt")
    model.eval()

    def make_batch(smiles):
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(smi) for smi in smiles])
        return collate_batch(dataset)[0]

    bmg = make_batch(["CCO", "c1ccccc1", "CC(=O)O"])

Note: Export fixes any dimension of size 0 or 1, so a sample batch of single atoms will not work.

Add checks for batch norm
-------------------------

If the model was trained with batch normalization, call :func:`~chemprop.utils.export.add_export_checks` before exporting. Batch normalization introduces a branch in the trace that needs to know the batch is not empty.

.. code-block:: python

    from chemprop.utils.export import add_export_checks

    add_export_checks(model)

Mark the graph size as dynamic
------------------------------

Mark the number of atoms and the number of edges as dynamic so the exported model accepts batches of any size:

.. code-block:: python

    num_atoms = torch.export.Dim("num_atoms")
    num_edges = torch.export.Dim("num_edges")
    dynamic_shapes = {
        "bmg": [
            {0: num_atoms},  # V
            {0: num_edges},  # E
            {1: num_edges},  # edge_index, shape 2 x num_edges
            {0: num_edges},  # rev_edge_index
            {0: num_atoms},  # batch
        ]
    }


Export with ``torch.export``
----------------------------

.. code-block:: python

    exported = torch.export.export(model, (bmg,), dynamic_shapes=dynamic_shapes, strict=False)

    new_bmg = make_batch(["C", "CCN", "c1ccncc1", "CC(C)O", "CCCC"])
    preds = exported.module()(new_bmg)


Export to ONNX
--------------

ONNX export needs the ``onnxscript`` package, and running the result needs ``onnxruntime``.

.. code-block:: python

    onnx_program = torch.onnx.export(model, (bmg,), dynamic_shapes=dynamic_shapes, dynamo=True)
    onnx_program.save("model.onnx")

The ONNX model's inputs are the five ``BatchMolGraph`` tensors, named ``bmg_v``, ``bmg_e``, ``bmg_edge_index``, ``bmg_rev_edge_index``, and ``bmg_batch``. Its output is the predictions, one row per molecule:

.. code-block:: python

    import onnxruntime as ort

    session = ort.InferenceSession("model.onnx")
    inputs = {
        "bmg_v": new_bmg.V.numpy(),
        "bmg_e": new_bmg.E.numpy(),
        "bmg_edge_index": new_bmg.edge_index.numpy(),
        "bmg_rev_edge_index": new_bmg.rev_edge_index.numpy(),
        "bmg_batch": new_bmg.batch.numpy(),
    }
    (preds,) = session.run(None, inputs)

Featurize new molecules with Chemprop, as above, or reproduce its featurization in your deployment environment.

Models with extra descriptors
-----------------------------

If the model uses extra atom descriptors (``V_d``, one row per atom) or molecule descriptors (``X_d``, one row per molecule), pass them to export after the ``BatchMolGraph`` and add them to ``dynamic_shapes``. Atom descriptors use the ``num_atoms`` dimension from above:

.. code-block:: python

    dynamic_shapes["V_d"] = {0: num_atoms}
    exported = torch.export.export(model, (bmg, V_d), dynamic_shapes=dynamic_shapes, strict=False)

Molecule descriptors need their own dimension. Pass ``None`` for ``V_d`` if the model has no atom descriptors:

.. code-block:: python

    dynamic_shapes["V_d"] = None
    dynamic_shapes["X_d"] = {0: torch.export.Dim("num_molecules")}
    exported = torch.export.export(
        model, (bmg, None, X_d), dynamic_shapes=dynamic_shapes, strict=False
    )

With molecule descriptors, give the program from ``torch.export`` batches of at least two molecules. A batch of one molecule raises ``Runtime assertion failed for expression u0 >= 2``. The ONNX model accepts batches of one molecule.

The ONNX model has one more input for each descriptor, named ``v_d`` and ``x_d``.

Multicomponent models
---------------------

A :class:`~chemprop.models.multi.MulticomponentMPNN` takes a list with one ``BatchMolGraph`` per component, for example a reaction and a solvent. Give each component its own dimensions, because the components have different numbers of atoms and edges:

.. code-block:: python

    def graph_dims(i):
        num_atoms = torch.export.Dim(f"num_atoms_{i}")
        num_edges = torch.export.Dim(f"num_edges_{i}")
        return [{0: num_atoms}, {0: num_edges}, {1: num_edges}, {0: num_edges}, {0: num_atoms}]

    dynamic_shapes = {"bmg": [graph_dims(0), graph_dims(1)]}
    exported = torch.export.export(
        model, ([bmg_0, bmg_1],), dynamic_shapes=dynamic_shapes, strict=False
    )

The ONNX model's inputs are the five tensors of each component in order, named ``bmg_0_v`` through ``bmg_0_batch``, then ``bmg_1_v`` through ``bmg_1_batch``.

.. _export-limitations:

Limitations
-----------

The following were checked by exporting with both ``torch.export`` and ONNX and comparing the exported model's predictions with the original model's on new batches, including a batch of one molecule:

* :class:`~chemprop.models.model.MPNN` with any predictor except ``BinaryDirichletFFN`` (see below), any aggregation (mean, sum, norm, attentive), atom or bond message passing, batch norm, atom or molecule descriptors, and reaction inputs. With molecule descriptors, the ``torch.export`` program needs at least two molecules per batch, as described above.
* :class:`~chemprop.models.multi.MulticomponentMPNN` with a reaction and a molecule component, with and without batch norm.

These cannot be exported yet:

* :class:`~chemprop.models.mol_atom_bond.MolAtomBondMPNN`. Its input is a :class:`~chemprop.data.collate.BatchMolAtomBondGraph`, which ``register_batch_mol_graph_pytree`` does not register. The model also expects an even number of edges, so its edge dimension must be declared as ``2 * torch.export.Dim("num_bonds")``, and ONNX conversion of a model with atom or bond constraints fails with ``Boolean index_put with multiple indices does not support None indices``.
* ``BinaryDirichletFFN`` (``classification-dirichlet``) with ``torch.export``. It fails with ``Could not extract specialized integer from data-dependent expression``. ONNX export of these models works.

Models that use ``cuik-molmaker`` featurization have not been checked.
