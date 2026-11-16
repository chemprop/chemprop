from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils._pytree import SUPPORTED_NODES, GetAttrKey, register_pytree_node
from torch.utils.hooks import RemovableHandle

from chemprop.data.collate import BatchMolGraph

_BMG_ATTRS = ("V", "E", "edge_index", "rev_edge_index", "batch")


def _flatten_bmg(bmg: BatchMolGraph) -> tuple[list[Tensor], None]:
    return [getattr(bmg, attr) for attr in _BMG_ATTRS], None


def _flatten_bmg_with_keys(bmg: BatchMolGraph) -> tuple[list[tuple[GetAttrKey, Tensor]], None]:
    children, context = _flatten_bmg(bmg)
    return list(zip(map(GetAttrKey, _BMG_ATTRS), children)), context


def _unflatten_bmg(children: Iterable[Tensor], context: None) -> BatchMolGraph:
    bmg = object.__new__(BatchMolGraph)
    for attr, child in zip(_BMG_ATTRS, children):
        setattr(bmg, attr, child)
    # size is only needed for loss/metric calculation during training
    bmg._BatchMolGraph__size = None
    return bmg


def register_batch_mol_graph_pytree() -> None:
    if BatchMolGraph in SUPPORTED_NODES:
        return

    register_pytree_node(
        BatchMolGraph,
        _flatten_bmg,
        _unflatten_bmg,
        flatten_with_keys_fn=_flatten_bmg_with_keys,
        serialized_type_name="chemprop.data.collate.BatchMolGraph",
    )


def _check_batch_nonempty(module: nn.Module, args: tuple[Tensor, ...]) -> None:
    torch._check(args[0].shape[0] > 0)


def add_export_checks(model: nn.Module) -> list[RemovableHandle]:
    """Tell batch norm the input is not empty so it knows how to branch"""
    return [
        module.register_forward_pre_hook(_check_batch_nonempty)
        for module in model.modules()
        if isinstance(module, nn.BatchNorm1d)
    ]
