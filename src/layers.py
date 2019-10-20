import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nn_init


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


class HighWay(nn.Module):
    def __init__(self, features, ns):
        super().__init__()
        self.T = nn.Linear(in_features=features, out_features=features)
        self.H = nn.Linear(in_features=features, out_features=features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=ns)
        self.norm = nn.InstanceNorm1d(num_features=features)

    def forward(self, x):
        T = torch.sigmoid(self.T(x))
        H = self.leaky_relu(self.H(x))
        return self.norm(T * H + (1 - T) * x)


class Conv1dNorm(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs
        )
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        if activation:
            self.activation = activation


class TanhAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.query = nn.Linear(in_features=features, out_features=features)
        self.value = nn.Linear(in_features=features, out_features=1)

    def forward(self, query, memory):
        query = self.query(query)
        alignment = self.value(torch.tanh(query + memory))
        return alignment


class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.attn = TanhAttention(hidden_size)

    def forward(self, x, state, memory):
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
        return rnn_out, attention


if __name__ == "__main__":
    from torch import _C

    _C.set_grad_enabled(False)
    batch = 13
    in_size = 17
    hidden = 19
    timesteps = 43
    attn = AttentionLayer(in_size, hidden)
    qr = torch.randn(batch, in_size)
    mem = torch.randn(timesteps, batch, hidden)
    out, att = attn(qr, None, mem)
    print(out.shape, att.shape)
