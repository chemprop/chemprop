from __future__ import annotations

import logging

import torch
from torch import Tensor

from chemprop.data import BatchWeightedMolGraph
from chemprop.models.model import MPNN

logger = logging.getLogger(__name__)


class wMPNN(MPNN):
    r"""An :class:`wMPNN` is a sequence of message passing layers, an aggregation routine, and a
    predictor routine. Messages are weighted based on bond and atom weights. The final
    result is multiplied by the degree of polymerisation.

    The first two modules calculate learned fingerprints from an input molecule or
    reaction graph, and the final module takes these learned fingerprints as input to calculate a
    final prediction. I.e., the following operation:

    .. math::
        \mathtt{MPNN}(\mathcal{G}) =
            \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

    The full model is trained end-to-end.

    Parameters
    ----------
    message_passing : MessagePassing
        the message passing block to use to calculate learned fingerprints
    agg : Aggregation
        the aggregation operation to use during molecule-level prediction
    predictor : Predictor
        the function to use to calculate the final prediction
    batch_norm : bool, default=False
        if `True`, apply batch normalization to the output of the aggregation operation
    metrics : Iterable[Metric] | None, default=None
        the metrics to use to evaluate the model during training and evaluation
    warmup_epochs : int, default=2
        the number of epochs to use for the learning rate warmup
    init_lr : int, default=1e-4
        the initial learning rate
    max_lr : float, default=1e-3
        the maximum learning rate
    final_lr : float, default=1e-4
        the final learning rate

    Raises
    ------
    ValueError
        if the output dimension of the message passing block does not match the input dimension of
        the predictor function
    """

    def fingerprint(
        self, bmg: BatchWeightedMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch, V_w=bmg.V_w)
        H = self.weight(H, bmg.degree_of_poly)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def weight(self, H: Tensor, degree_of_poly: Tensor) -> Tensor:
        """weights the final molecular representation by the degree of polymerization"""
        return torch.mul(degree_of_poly.unsqueeze(1), H)
