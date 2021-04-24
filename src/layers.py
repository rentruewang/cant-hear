"""
This module provides some custom defined modules
"""

import torch
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    GRUCell,
    InstanceNorm1d,
    LeakyReLU,
    Linear,
    Module,
    Sequential,
)
from torch.nn import functional as F


class Squeeze(Module):
    "Performs squeezing on the tensor"

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        "Pass through"
        return x.squeeze(dim=self.dim)


class HighWay(Module):
    "Highway network"

    def __init__(self, features, ns):
        super().__init__()
        self.T = Linear(in_features=features, out_features=features)
        self.H = Linear(in_features=features, out_features=features)
        self.leaky_relu = LeakyReLU(negative_slope=ns)
        self.norm = InstanceNorm1d(num_features=features)

    def forward(self, x):
        "Pass through"
        T = torch.sigmoid(self.T(x))
        H = self.leaky_relu(self.H(x))
        return self.norm(T * H + (1 - T) * x)


class Conv1dNorm(Sequential):
    "Convolution1d and Norm"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        *args,
        activation=None,
        **kwargs
    ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs
        )
        self.norm = BatchNorm1d(num_features=out_channels)
        if activation:
            self.activation = activation


class TanhAttention(Module):
    "Attention using tanh"

    def __init__(self, features):
        super().__init__()
        self.query = Linear(in_features=features, out_features=features)
        self.value = Linear(in_features=features, out_features=1)

    def forward(self, query, memory):
        "Pass through"
        query = self.query(query)
        alignment = self.value(torch.tanh(query + memory))
        return alignment


class AttentionLayer(Module):
    "Attention layer"

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.attn = TanhAttention(hidden_size)

    def forward(self, x, state, memory):
        "Pass through"
        # x: batch, input_size
        # state: batch, hidden_size
        # memory: timesteps, batch, hidden_size
        # rnn_out: batch, 1, hidden_size
        # attention: batch, 1, hidden_size
        memory = memory.transpose(0, 1)
        rnn_out = self.rnn(x, state)
        rnn_out = rnn_out.unsqueeze(1)
        attention = self.attn(rnn_out, memory)
        attention = F.softmax(attention, dim=1)
        attention = torch.bmm(attention.transpose(1, 2), memory)
        return (rnn_out, attention)
