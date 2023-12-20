from enum import auto

from torch import nn

from chemprop.utils.utils import EnumMapping


class Activation(EnumMapping):
    RELU = auto()
    LEAKYRELU = auto()
    PRELU = auto()
    TANH = auto()
    SELU = auto()
    ELU = auto()


def get_activation_function(activation: str | Activation) -> nn.Module:
    """Gets an activation function module given the name of the activation.

    See :class:`~chemprop.v2.models.utils.Activation` for available activations.

    Parameters
    ----------
    activation : str | Activation
        The name of the activation function.

    Returns
    -------
    nn.Module
        The activation function module.
    """
    match Activation.get(activation):
        case Activation.RELU:
            return nn.ReLU()
        case Activation.LEAKYRELU:
            return nn.LeakyReLU(0.1)
        case Activation.PRELU:
            return nn.PReLU()
        case Activation.TANH:
            return nn.Tanh()
        case Activation.SELU:
            return nn.SELU()
        case Activation.ELU:
            return nn.ELU()
        case _:
            raise RuntimeError("unreachable code reached!")
