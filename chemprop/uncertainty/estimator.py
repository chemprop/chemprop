from abc import ABC, abstractmethod
from typing import Iterable

from lightning import pytorch as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.models import MPNN, MolAtomBondMPNN
from chemprop.utils.registry import ClassRegistry


class UncertaintyEstimator(ABC):
    """A helper class for making model predictions and associated uncertainty predictions."""

    @abstractmethod
    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        """
        Calculate the uncalibrated predictions and uncertainties for the dataloader.

        dataloader: DataLoader
            the dataloader used for model predictions and uncertainty predictions
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN]
            the models used for model predictions and uncertainty predictions. If using
            MolAtomBondMPNN models, the uncertainty estimator will return preds and uncs for each of
            the mole, atom, and bond predictions and uncertainties.
        trainer: pl.Trainer
            an instance of the :class:`~lightning.pytorch.trainer.trainer.Trainer` used to manage model inference

        Returns
        -------
        preds : Tensor
            the model predictions, with shape varying by task type:

            * regression/binary classification: ``m x n x t``

            * multiclass classification: ``m x n x t x c``, where ``m`` is the number of models,
            ``n`` is the number of inputs, ``t`` is the number of tasks, and ``c`` is the number of classes.
        uncs : Tensor
            the predicted uncertainties, with shapes of ``m' x n x t``.

        .. note::
            The ``m`` and ``m'`` are different by definition. The ``m`` is the number of models,
            while the ``m'`` is the number of uncertainty estimations. For example, if two MVE
            or evidential models are provided, both ``m`` and ``m'`` are two. However, for an
            ensemble of two models, ``m'`` would be one (even though ``m = 2``).
        """


UncertaintyEstimatorRegistry = ClassRegistry[UncertaintyEstimator]()


@UncertaintyEstimatorRegistry.register("none")
class NoUncertaintyEstimator(UncertaintyEstimator):
    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            predss = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                predss.append(preds)
        else:
            mol_predss = []
            atom_predss = []
            bond_predss = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_predss.append(mol_preds)
                if atom_preds is not None:
                    atom_predss.append(atom_preds)
                if bond_preds is not None:
                    bond_predss.append(bond_preds)

        preds_tuple = (predss,) if not_mol_atom_bond else (mol_predss, atom_predss, bond_predss)
        processed_preds = []
        for raw_preds in preds_tuple:
            if raw_preds:
                processed_preds.append(torch.stack(raw_preds))
            else:
                processed_preds.append(None)
        if not_mol_atom_bond:
            return processed_preds[0], None
        return processed_preds, (None, None, None)


@UncertaintyEstimatorRegistry.register("mve")
class MVEEstimator(UncertaintyEstimator):
    """
    Class that estimates prediction means and variances (MVE). [nix1994]_

    References
    ----------
    .. [nix1994] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target
        probability distribution." Proceedings of 1994 IEEE International Conference on Neural
        Networks, 1994 https://doi.org/10.1109/icnn.1994.374138
    """

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            mves = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                mves.append(preds)
        else:
            mol_mves = []
            atom_mves = []
            bond_mves = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_mves.append(mol_preds)
                if atom_preds is not None:
                    atom_mves.append(atom_preds)
                if bond_preds is not None:
                    bond_mves.append(bond_preds)

        mves_tuple = (mves,) if not_mol_atom_bond else (mol_mves, atom_mves, bond_mves)
        means = []
        vars = []
        for raw_mves in mves_tuple:
            if raw_mves:
                mves = torch.stack(raw_mves, dim=0)
                mean, var = mves.unbind(dim=-1)
                means.append(mean)
                vars.append(var)
            else:
                means.append(None)
                vars.append(None)
        if not_mol_atom_bond:
            return means[0], vars[0]
        return means, vars


@UncertaintyEstimatorRegistry.register("ensemble")
class EnsembleEstimator(UncertaintyEstimator):
    """
    Class that predicts the uncertainty of predictions based on the variance in predictions among
    an ensemble's submodels.
    """

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        if len(models) <= 1:
            raise ValueError(
                "Ensemble method for uncertainty is only available when multiple models are provided."
            )
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            ensemble_preds = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                ensemble_preds.append(preds)
        else:
            mol_ensemble_preds = []
            atom_ensemble_preds = []
            bond_ensemble_preds = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_ensemble_preds.append(mol_preds)
                if atom_preds is not None:
                    atom_ensemble_preds.append(atom_preds)
                if bond_preds is not None:
                    bond_ensemble_preds.append(bond_preds)

        ensemble_preds_tuple = (
            (ensemble_preds,)
            if not_mol_atom_bond
            else (mol_ensemble_preds, atom_ensemble_preds, bond_ensemble_preds)
        )
        stacked_predss = []
        varss = []
        for ensemble_preds in ensemble_preds_tuple:
            if ensemble_preds:
                stacked_preds = torch.stack(ensemble_preds).float()
                vars = torch.var(stacked_preds, dim=0, correction=0).unsqueeze(0)
                stacked_predss.append(stacked_preds)
                varss.append(vars)
            else:
                stacked_predss.append(None)
                varss.append(None)
        if not_mol_atom_bond:
            return stacked_predss[0], varss[0]
        return stacked_predss, varss


@UncertaintyEstimatorRegistry.register("classification")
class ClassEstimator(UncertaintyEstimator):
    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            predss = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                predss.append(preds)
        else:
            mol_predss = []
            atom_predss = []
            bond_predss = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_predss.append(mol_preds)
                if atom_preds is not None:
                    atom_predss.append(atom_preds)
                if bond_preds is not None:
                    bond_predss.append(bond_preds)
        predss_tuple = (predss,) if not_mol_atom_bond else (mol_predss, atom_predss, bond_predss)
        processed_predss = []
        for raw_preds in predss_tuple:
            if raw_preds:
                processed_predss.append(torch.stack(raw_preds))
            else:
                processed_predss.append(None)
        if not_mol_atom_bond:
            return processed_predss[0], processed_predss[0]
        return processed_predss, processed_predss


@UncertaintyEstimatorRegistry.register("evidential-total")
class EvidentialTotalEstimator(UncertaintyEstimator):
    """
    Class that predicts the total evidential uncertainty based on hyperparameters of
    the evidential distribution [amini2020]_.

    References
    -----------
    .. [amini2020] Amini, A.; Schwarting, W.; Soleimany, A.; Rus, D. "Deep Evidential Regression".
        NeurIPS, 2020. https://arxiv.org/abs/1910.02600
    """

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            uncs = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                uncs.append(preds)
        else:
            mol_uncs = []
            atom_uncs = []
            bond_uncs = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_uncs.append(mol_preds)
                if atom_preds is not None:
                    atom_uncs.append(atom_preds)
                if bond_preds is not None:
                    bond_uncs.append(bond_preds)
        uncs_tuple = (uncs,) if not_mol_atom_bond else (mol_uncs, atom_uncs, bond_uncs)
        means = []
        total_uncss = []
        for raw_uncs in uncs_tuple:
            if raw_uncs:
                uncs = torch.stack(raw_uncs)
                mean, v, alpha, beta = uncs.unbind(-1)
                total_uncs = (1 + 1 / v) * (beta / (alpha - 1))
                means.append(mean)
                total_uncss.append(total_uncs)
            else:
                means.append(None)
                total_uncss.append(None)
        if not_mol_atom_bond:
            return means[0], total_uncss[0]
        return means, total_uncss


@UncertaintyEstimatorRegistry.register("evidential-epistemic")
class EvidentialEpistemicEstimator(UncertaintyEstimator):
    """
    Class that predicts the epistemic evidential uncertainty based on hyperparameters of
    the evidential distribution.
    """

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            uncs = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                uncs.append(preds)
        else:
            mol_uncs = []
            atom_uncs = []
            bond_uncs = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_uncs.append(mol_preds)
                if atom_preds is not None:
                    atom_uncs.append(atom_preds)
                if bond_preds is not None:
                    bond_uncs.append(bond_preds)
        uncs_tuple = (uncs,) if not_mol_atom_bond else (mol_uncs, atom_uncs, bond_uncs)
        means = []
        epistemic_uncss = []
        for raw_uncs in uncs_tuple:
            if raw_uncs:
                uncs = torch.stack(raw_uncs)
                mean, v, alpha, beta = uncs.unbind(-1)
                epistemic_uncs = (1 / v) * (beta / (alpha - 1))
                means.append(mean)
                epistemic_uncss.append(epistemic_uncs)
            else:
                means.append(None)
                epistemic_uncss.append(None)
        if not_mol_atom_bond:
            return means[0], epistemic_uncss[0]
        return means, epistemic_uncss


@UncertaintyEstimatorRegistry.register("evidential-aleatoric")
class EvidentialAleatoricEstimator(UncertaintyEstimator):
    """
    Class that predicts the aleatoric evidential uncertainty based on hyperparameters of
    the evidential distribution.
    """

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            uncs = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                uncs.append(preds)
        else:
            mol_uncs = []
            atom_uncs = []
            bond_uncs = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_uncs.append(mol_preds)
                if atom_preds is not None:
                    atom_uncs.append(atom_preds)
                if bond_preds is not None:
                    bond_uncs.append(bond_preds)
        uncs_tuple = (uncs,) if not_mol_atom_bond else (mol_uncs, atom_uncs, bond_uncs)
        means = []
        aleatoric_uncss = []
        for raw_uncs in uncs_tuple:
            if raw_uncs:
                uncs = torch.stack(raw_uncs)
                mean, v, alpha, beta = uncs.unbind(-1)
                aleatoric_uncs = beta / (alpha - 1)
                means.append(mean)
                aleatoric_uncss.append(aleatoric_uncs)
            else:
                means.append(None)
                aleatoric_uncss.append(None)
        if not_mol_atom_bond:
            return means[0], aleatoric_uncss[0]
        return means, aleatoric_uncss


@UncertaintyEstimatorRegistry.register("dropout")
class DropoutEstimator(UncertaintyEstimator):
    """
    A :class:`DropoutEstimator` creates a virtual ensemble of models via Monte Carlo dropout with
    the provided model [gal2016]_.

    Parameters
    ----------
    ensemble_size: int
        The number of samples to draw for the ensemble.
    dropout: float | None
        The probability of dropping out units in the dropout layers. If unspecified,
        the training probability is used, which is prefered but not possible if the model was not
        trained with dropout (i.e. p=0).

    References
    -----------
    .. [gal2016] Gal, Y.; Ghahramani, Z. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning."
        International conference on machine learning. PMLR, 2016. https://arxiv.org/abs/1506.02142
    """

    def __init__(self, ensemble_size: int, dropout: None | float = None):
        self.ensemble_size = ensemble_size
        self.dropout = dropout

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            meanss, varss = [], []
            for model in models:
                self._setup_model(model)
                individual_preds = []

                for _ in range(self.ensemble_size):
                    predss = trainer.predict(model, dataloader)
                    preds = torch.concat(predss, 0)
                    individual_preds.append(preds)

                stacked_preds = torch.stack(individual_preds, dim=0).float()
                means = torch.mean(stacked_preds, dim=0)
                vars = torch.var(stacked_preds, dim=0, correction=0)
                self._restore_model(model)
                meanss.append(means)
                varss.append(vars)
            return torch.stack(meanss), torch.stack(varss)
        else:
            mol_meanss, mol_varss = [], []
            atom_meanss, atom_varss = [], []
            bond_meanss, bond_varss = [], []
            for model in models:
                self._setup_model(model)
                mol_individual_preds = []
                atom_individual_preds = []
                bond_individual_preds = []

                for _ in range(self.ensemble_size):
                    MAB_predss = trainer.predict(model, dataloader)
                    mol_preds, atom_preds, bond_preds = (
                        torch.concat(preds, 0) if preds[0] is not None else None
                        for preds in zip(*MAB_predss)
                    )
                    if mol_preds is not None:
                        mol_individual_preds.append(mol_preds)
                    if atom_preds is not None:
                        atom_individual_preds.append(atom_preds)
                    if bond_preds is not None:
                        bond_individual_preds.append(bond_preds)

                if mol_individual_preds:
                    stacked_mol_preds = torch.stack(mol_individual_preds, dim=0).float()
                    mol_means = torch.mean(stacked_mol_preds, dim=0)
                    mol_vars = torch.var(stacked_mol_preds, dim=0, correction=0)
                    mol_meanss.append(mol_means)
                    mol_varss.append(mol_vars)

                if atom_individual_preds:
                    stacked_atom_preds = torch.stack(atom_individual_preds, dim=0).float()
                    atom_means = torch.mean(stacked_atom_preds, dim=0)
                    atom_vars = torch.var(stacked_atom_preds, dim=0, correction=0)
                    atom_meanss.append(atom_means)
                    atom_varss.append(atom_vars)

                if bond_individual_preds:
                    stacked_bond_preds = torch.stack(bond_individual_preds, dim=0).float()
                    bond_means = torch.mean(stacked_bond_preds, dim=0)
                    bond_vars = torch.var(stacked_bond_preds, dim=0, correction=0)
                    bond_meanss.append(bond_means)
                    bond_varss.append(bond_vars)

                self._restore_model(model)

            return (
                (
                    torch.stack(mol_meanss) if mol_meanss else None,
                    torch.stack(atom_meanss) if atom_meanss else None,
                    torch.stack(bond_meanss) if bond_meanss else None,
                ),
                (
                    torch.stack(mol_varss) if mol_varss else None,
                    torch.stack(atom_varss) if atom_varss else None,
                    torch.stack(bond_varss) if bond_varss else None,
                ),
            )

    def _setup_model(self, model):
        model._predict_step = model.predict_step
        model.predict_step = self._predict_step(model)
        model.apply(self._change_dropout)

    def _restore_model(self, model):
        model.predict_step = model._predict_step
        del model._predict_step
        model.apply(self._restore_dropout)

    def _predict_step(self, model):
        def _wrapped_predict_step(*args, **kwargs):
            model.apply(self._activate_dropout)
            return model._predict_step(*args, **kwargs)

        return _wrapped_predict_step

    def _activate_dropout(self, module):
        if isinstance(module, torch.nn.Dropout):
            module.train()

    def _change_dropout(self, module):
        if isinstance(module, torch.nn.Dropout):
            module._p = module.p
            if self.dropout:
                module.p = self.dropout

    def _restore_dropout(self, module):
        if isinstance(module, torch.nn.Dropout):
            if hasattr(module, "_p"):
                module.p = module._p
                del module._p


# TODO: Add in v2.1.x
# @UncertaintyEstimatorRegistry.register("spectra-roundrobin")
# class RoundRobinSpectraEstimator(UncertaintyEstimator):
#     def __call__(
#         self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
#     ) -> tuple[Tensor, Tensor]:
#         return


@UncertaintyEstimatorRegistry.register("classification-dirichlet")
class ClassificationDirichletEstimator(UncertaintyEstimator):
    """
    A :class:`ClassificationDirichletEstimator` predicts an amount of 'evidence' for both the
    negative class and the positive class as described in [sensoy2018]_. The class probabilities and
    the uncertainty are calculated based on the evidence.

    .. math::
        S = \sum_{i=1}^K \alpha_i
        p_i = \alpha_i / S
        u = K / S

    where :math:`K` is the number of classes, :math:`\alpha_i` is the evidence for class :math:`i`,
    :math:`p_i` is the probability of class :math:`i`, and :math:`u` is the uncertainty.

    References
    ----------
    .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
        classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
    """

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            uncs = []
            for model in models:
                preds = torch.concat(trainer.predict(model, dataloader), 0)
                uncs.append(preds)
        else:
            mol_uncs = []
            atom_uncs = []
            bond_uncs = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_uncs.append(mol_preds)
                if atom_preds is not None:
                    atom_uncs.append(atom_preds)
                if bond_preds is not None:
                    bond_uncs.append(bond_preds)
        uncs_tuple = (uncs,) if not_mol_atom_bond else (mol_uncs, atom_uncs, bond_uncs)
        ys = []
        us = []
        for raw_uncs in uncs_tuple:
            if raw_uncs:
                uncs = torch.stack(raw_uncs, dim=0)
                y, u = uncs.unbind(dim=-1)
                ys.append(y)
                us.append(u)
            else:
                ys.append(None)
                us.append(None)
        if not_mol_atom_bond:
            return ys[0], us[0]
        return ys, us


@UncertaintyEstimatorRegistry.register("multiclass-dirichlet")
class MulticlassDirichletEstimator(UncertaintyEstimator):
    """
    A :class:`MulticlassDirichletEstimator` predicts an amount of 'evidence' for each class as
    described in [sensoy2018]_. The class probabilities and the uncertainty are calculated based on
    the evidence.

    .. math::
        S = \sum_{i=1}^K \alpha_i
        p_i = \alpha_i / S
        u = K / S

    where :math:`K` is the number of classes, :math:`\alpha_i` is the evidence for class :math:`i`,
    :math:`p_i` is the probability of class :math:`i`, and :math:`u` is the uncertainty.

    References
    ----------
    .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
        classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
    """

    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            preds = []
            uncs = []
            for model in models:
                self._setup_model(model)
                output = torch.concat(trainer.predict(model, dataloader), 0)
                self._restore_model(model)
                preds.append(output[..., :-1])
                uncs.append(output[..., -1])
            preds = torch.stack(preds, 0)
            uncs = torch.stack(uncs, 0)

            return preds, uncs
        else:
            mol_preds = []
            atom_preds = []
            bond_preds = []
            mol_uncs = []
            atom_uncs = []
            bond_uncs = []
            for model in models:
                self._setup_model(model)
                MAB_preds = trainer.predict(model, dataloader)
                mol_output, atom_output, bond_output = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                self._restore_model(model)

                if mol_output is not None:
                    mol_preds.append(mol_output[..., :-1])
                    mol_uncs.append(mol_output[..., -1])
                if atom_output is not None:
                    atom_preds.append(atom_output[..., :-1])
                    atom_uncs.append(atom_output[..., -1])
                if bond_output is not None:
                    bond_preds.append(bond_output[..., :-1])
                    bond_uncs.append(bond_output[..., -1])

            mol_preds = torch.stack(mol_preds, 0) if mol_preds else None
            atom_preds = torch.stack(atom_preds, 0) if atom_preds else None
            bond_preds = torch.stack(bond_preds, 0) if bond_preds else None
            mol_uncs = torch.stack(mol_uncs, 0) if mol_uncs else None
            atom_uncs = torch.stack(atom_uncs, 0) if atom_uncs else None
            bond_uncs = torch.stack(bond_uncs, 0) if bond_uncs else None

            return (mol_preds, atom_preds, bond_preds), (mol_uncs, atom_uncs, bond_uncs)

    def _setup_model(self, model):
        model.predictor._forward = model.predictor.forward
        model.predictor.forward = self._forward.__get__(model.predictor, model.predictor.__class__)

    def _restore_model(self, model):
        model.predictor.forward = model.predictor._forward
        del model.predictor._forward

    def _forward(self, Z: Tensor) -> Tensor:
        alpha = self.train_step(Z)

        u = alpha.shape[2] / alpha.sum(-1, keepdim=True)
        Y = alpha / alpha.sum(-1, keepdim=True)

        return torch.concat([Y, u], -1)


@UncertaintyEstimatorRegistry.register("quantile-regression")
class QuantileRegressionEstimator(UncertaintyEstimator):
    def __call__(
        self,
        dataloader: DataLoader,
        models: Iterable[MPNN] | Iterable[MolAtomBondMPNN],
        trainer: pl.Trainer,
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor | None, Tensor | None, Tensor | None],
            tuple[Tensor | None, Tensor | None, Tensor | None],
        ]
    ):
        not_mol_atom_bond = isinstance(models[0], MPNN)
        if not_mol_atom_bond:
            individual_preds = []
            for model in models:
                predss = trainer.predict(model, dataloader)
                individual_preds.append(torch.concat(predss, 0))
        else:
            mol_individual_preds = []
            atom_individual_preds = []
            bond_individual_preds = []
            for model in models:
                MAB_preds = trainer.predict(model, dataloader)
                mol_preds, atom_preds, bond_preds = (
                    torch.concat(preds, 0) if preds[0] is not None else None
                    for preds in zip(*MAB_preds)
                )
                if mol_preds is not None:
                    mol_individual_preds.append(mol_preds)
                if atom_preds is not None:
                    atom_individual_preds.append(atom_preds)
                if bond_preds is not None:
                    bond_individual_preds.append(bond_preds)
        individual_preds_tuple = (
            (individual_preds,)
            if not_mol_atom_bond
            else (mol_individual_preds, atom_individual_preds, bond_individual_preds)
        )
        means = []
        half_intervals = []
        for individual_preds in individual_preds_tuple:
            if individual_preds:
                stacked_preds = torch.stack(individual_preds).float()
                mean, interval = stacked_preds.unbind(-1)
                means.append(mean)
                half_intervals.append(interval / 2)
            else:
                means.append(None)
                half_intervals.append(None)
        if not_mol_atom_bond:
            return means[0], half_intervals[0]
        return means, half_intervals
