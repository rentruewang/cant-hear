from torch import nn
from torch.nn import init as nn_init


def init(module):
    "Initialize the module"
    init_linear(module)
    init_conv(module)
    init_rnn(module)


def init_linear(module):
    "Initialize linear layers"
    if isinstance(module, nn.Linear):
        nn_init.xavier_uniform_(module.weight, gain=nn_init.calculate_gain("linear"))


def init_conv(module):
    "Initialize convolution layers"
    if isinstance(module, nn.modules.conv._ConvNd):
        nn_init.xavier_uniform_(module.weight, gain=nn_init.calculate_gain("conv1d"))


def init_rnn(module):
    "Initialize rnn layers"
    if isinstance(module, nn.RNNBase):
        nn_init.orthogonal_(module.weight_hh_l0, gain=nn_init.calculate_gain("linear"))
        nn_init.xavier_uniform_(
            module.weight_ih_l0, gain=nn_init.calculate_gain("linear")
        )
