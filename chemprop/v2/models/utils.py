from torch import nn


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    Parameters
    ----------
    activation : str
        The name of the activation function.

    Returns
    -------
    nn.Module
        The activation function module.
    """
    activation_ = activation.lower()
    if activation_ == "relu":
        return nn.ReLU()
    if activation_ == "leakyrelu":
        return nn.LeakyReLU(0.1)
    if activation_ == "prelu":
        return nn.PReLU()
    if activation_ == "tanh":
        return nn.Tanh()
    if activation_ == "selu":
        return nn.SELU()
    if activation_ == "elu":
        return nn.ELU()

    raise ValueError(
        f'Invalid activation! got: "{activation}". '
        f'expected one of: ("relu", "leakyrelu", "prelu", "tanh", "selu", "elu")'
    )
