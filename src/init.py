"""
This module provides several functions for reusable initialization on different `nn.Module`s
"""

from torch.nn import Linear, Module, RNNBase
from torch.nn import init as nn_init
from torch.nn.modules.conv import _ConvNd


def init(module: Module):
    "Initialize the module"
    init_linear(module)
    init_conv(module)
    init_rnn(module)


def init_linear(module: Module):
    "Initialize linear layers"
    if isinstance(module, Linear):
        nn_init.xavier_uniform_(module.weight, gain=nn_init.calculate_gain("linear"))


def init_conv(module: Module):
    "Initialize convolution layers"
    if isinstance(module, _ConvNd):
        nn_init.xavier_uniform_(module.weight, gain=nn_init.calculate_gain("conv1d"))


def init_rnn(module: Module):
    "Initialize rnn layers"
    if isinstance(module, RNNBase):
        nn_init.orthogonal_(module.weight_hh_l0, gain=nn_init.calculate_gain("linear"))
        nn_init.xavier_uniform_(
            module.weight_ih_l0, gain=nn_init.calculate_gain("linear")
        )
