from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Squeeze


def pad_layer(inp, layer, is_2d=False):
    "Pad the input and pass it through the layer"

    if isinstance(layer.kernel_size, Sequence):
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size // 2, kernel_size // 2 - 1)
        else:
            pad = (kernel_size // 2, kernel_size // 2)
    else:
        if kernel_size % 2 == 0:
            pad = (
                kernel_size // 2,
                kernel_size // 2 - 1,
                kernel_size // 2,
                kernel_size // 2 - 1,
            )
        else:
            pad = (
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
            )
    # padding
    inp = F.pad(inp, pad=pad, mode="reflect")
    out = layer(inp)
    return out


def pixel_shuffle_1d(inp, upscale_factor=2):
    (batch_size, channels, in_width) = inp.size()
    channels //= upscale_factor

    out_width = in_width * upscale_factor
    inp_view = inp.contiguous().view(batch_size, channels, upscale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor, mode="nearest")
    return x_up


def GLU(inp, layer, res=True):
    kernel_size = layer.kernel_size[0]
    channels = layer.out_channels // 2
    # padding
    out = F.pad(
        inp.unsqueeze(dim=3),
        pad=(0, 0, kernel_size // 2, kernel_size // 2),
        mode="constant",
        value=0.0,
    )
    out = out.squeeze(dim=3)
    out = layer(out)
    # gated
    A = out[:, :channels, :]
    B = F.sigmoid(out[:, channels:, :])
    if res:
        H = A * B + inp
    else:
        H = A * B
    return H


def highway(inp, layers, gates, act):
    # permute
    batch_size = inp.size(0)
    seq_len = inp.size(2)
    inp_permuted = inp.permute(0, 2, 1)
    # merge dim
    out_expand = inp_permuted.contiguous().view(
        batch_size * seq_len, inp_permuted.size(2)
    )
    for (l, g) in zip(layers, gates):
        H = l(out_expand)
        H = act(H)
        T = g(out_expand)
        T = F.sigmoid(T)
        out_expand = H * T + out_expand * (1.0 - T)
    out_permuted = out_expand.view(batch_size, seq_len, out_expand.size(1))
    out = out_permuted.permute(0, 2, 1)
    return out


def RNN(inp, layer):
    inp_permuted = inp.permute(2, 0, 1)
    (out_permuted, _) = layer(inp_permuted)
    out_rnn = out_permuted.permute(1, 2, 0)
    return out_rnn


def linear(inp, layer):
    batch_size = inp.size(0)
    hidden_dim = inp.size(1)
    seq_len = inp.size(2)
    inp_permuted = inp.permute(0, 2, 1)
    inp_expand = inp_permuted.contiguous().view(batch_size * seq_len, hidden_dim)
    out_expand = layer(inp_expand)
    out_permuted = out_expand.view(batch_size, seq_len, out_expand.size(1))
    out = out_permuted.permute(0, 2, 1)
    return out


def append_emb(emb, expand_size, output):
    emb = emb.unsqueeze(dim=2)
    emb_expand = emb.expand(emb.size(0), emb.size(1), expand_size)
    output = torch.cat([output, emb_expand], dim=1)
    return output


class PatchDiscriminator(nn.Module):
    def __init__(self, n_class=33, ns=0.2, dp=0.1):
        super().__init__()
        self.ns = ns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(512, 1, kernel_size=1)
        # self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(17, 4))
        self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(17, 4))
        self.drop1 = nn.Dropout2d(p=dp)
        self.drop2 = nn.Dropout2d(p=dp)
        self.drop3 = nn.Dropout2d(p=dp)
        self.drop4 = nn.Dropout2d(p=dp)
        self.drop5 = nn.Dropout2d(p=dp)
        self.ins_norm1 = nn.InstanceNorm2d(self.conv1.out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(self.conv2.out_channels)
        self.ins_norm3 = nn.InstanceNorm2d(self.conv3.out_channels)
        self.ins_norm4 = nn.InstanceNorm2d(self.conv4.out_channels)
        self.ins_norm5 = nn.InstanceNorm2d(self.conv5.out_channels)

    def conv_block(self, x, conv_layer, after_layers):
        out = pad_layer(x, conv_layer, is_2d=True)
        out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        return out

    def forward(self, x, classify=False):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv_block(x, self.conv1, [self.ins_norm1, self.drop1])
        out = self.conv_block(out, self.conv2, [self.ins_norm2, self.drop2])
        out = self.conv_block(out, self.conv3, [self.ins_norm3, self.drop3])
        out = self.conv_block(out, self.conv4, [self.ins_norm4, self.drop4])
        out = self.conv_block(out, self.conv5, [self.ins_norm5, self.drop5])
        # GAN output value
        val = pad_layer(out, self.conv6, is_2d=True)
        val = val.view(val.size(0), -1)
        mean_val = torch.mean(val, dim=1)
        if classify:
            # classify
            logits = self.conv_classify(out)
            print(logits.size())
            logits = logits.view(logits.size(0), -1)
            return (mean_val, logits)
        else:
            return mean_val


class SpeakerClassifier(nn.Module):
    def __init__(self, c_in=512, c_h=512, n_class=8, dp=0.1, ns=0.01):
        super().__init__()
        self.dp, self.ns = dp, ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv7 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv8 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv9 = nn.Conv1d(c_h, n_class, kernel_size=16)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h)

    def conv_block(self, x, conv_layers, after_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        out = self.conv_block(
            x, [self.conv1, self.conv2], [self.ins_norm1, self.drop1], res=False
        )
        out = self.conv_block(
            out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2], res=True
        )
        out = self.conv_block(
            out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3], res=True
        )
        out = self.conv_block(
            out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4], res=True
        )
        out = self.conv9(out)
        out = out.view(out.size()[0], -1)
        return out


class LatentDiscriminator(nn.Module):
    def __init__(self, c_in=1024, c_h=512, ns=0.2, dp=0.1):
        super().__init__()
        self.ns = ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv7 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv8 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv9 = nn.Conv1d(c_h, 1, kernel_size=16)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h)

    def conv_block(self, x, conv_layers, after_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        out = self.conv_block(
            x, [self.conv1, self.conv2], [self.ins_norm1, self.drop1], res=False
        )
        out = self.conv_block(
            out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2], res=True
        )
        out = self.conv_block(
            out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3], res=True
        )
        out = self.conv_block(
            out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4], res=True
        )
        out = self.conv9(out)
        out = out.view(out.size()[0], -1)
        mean_value = torch.mean(out, dim=1)
        return mean_value


class CBHG(nn.Module):
    def __init__(self, c_in=80, c_out=513):
        super().__init__()
        self.conv1s = nn.ModuleList(
            nn.Conv1d(c_in, 128, kernel_size=k) for k in range(1, 9)
        )
        self.bn1s = nn.ModuleList(nn.BatchNorm1d(128) for _ in range(1, 9))
        self.mp1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(len(self.conv1s) * 128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 80, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(80)
        # highway network
        self.linear1 = nn.Linear(80, 128)
        self.layers = nn.ModuleList(nn.Linear(128, 128) for _ in range(4))
        self.gates = nn.ModuleList(nn.Linear(128, 128) for _ in range(4))
        self.RNN = nn.GRU(
            input_size=128, hidden_size=128, num_layers=1, bidirectional=True
        )
        self.linear2 = nn.Linear(256, c_out)

    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            out = F.relu(out)
            outs.append(out)
        bn_outs = []
        for (out, bn) in zip(outs, self.bn1s):
            out = bn(out)
            bn_outs.append(out)
        out = torch.cat(bn_outs, dim=1)
        print(out.shape)
        out = pad_layer(out, self.mp1)
        print(out.shape)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        print(out.shape, x.shape)
        out = out + x
        out = linear(out, self.linear1)
        out = highway(out, self.layers, self.gates, F.relu)
        out_rnn = RNN(out, self.RNN)
        out = linear(out_rnn, self.linear2)
        out = F.sigmoid(out)
        return out


class Decoder(nn.Module):
    def __init__(self, c_in=512, c_out=513, c_h=512, emb_size=128, ns=0.2):
        super().__init__()
        self.ns = ns
        self.conv1 = nn.Conv1d(c_in + emb_size, 2 * c_h, kernel_size=3)
        self.conv2 = nn.Conv1d(c_h + emb_size, c_h, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h + emb_size, 2 * c_h, kernel_size=3)
        self.conv4 = nn.Conv1d(c_h + emb_size, c_h, kernel_size=3)
        self.conv5 = nn.Conv1d(c_h + emb_size, 2 * c_h, kernel_size=3)
        self.conv6 = nn.Conv1d(c_h + emb_size, c_h, kernel_size=3)
        self.dense1 = nn.Linear(c_h + emb_size, c_h)
        self.dense2 = nn.Linear(c_h + emb_size, c_h)
        self.dense3 = nn.Linear(c_h + emb_size, c_h)
        self.dense4 = nn.Linear(c_h + emb_size, c_h)
        self.RNN = nn.GRU(
            input_size=c_h + emb_size,
            hidden_size=c_h // 2,
            num_layers=1,
            bidirectional=True,
        )
        self.dense5 = nn.Linear(2 * c_h + emb_size, c_h)
        self.linear = nn.Linear(c_h, c_out)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h)
        self.ins_norm5 = nn.InstanceNorm1d(c_h)

    def conv_block(self, x, conv_layers, norm_layer, emb, res=True):
        # first layer
        x_append = append_emb(emb, x.size(2), x)
        out = pad_layer(x_append, conv_layers[0])
        out = F.leaky_relu(out, negative_slope=self.ns)
        # upsample by pixelshuffle
        out = pixel_shuffle_1d(out, upscale_factor=2)
        out = append_emb(emb, out.size(2), out)
        out = pad_layer(out, conv_layers[1])
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            x_up = upsample(x, scale_factor=2)
            out = out + x_up
        return out

    def dense_block(self, x, emb, layers, norm_layer, res=True):
        out = x
        for layer in layers:
            out = append_emb(emb, out.size(2), out)
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x, c):
        # emb = self.emb(c)
        (emb2, emb4, emb6, emb8, embd2, embd4, embrnn) = c
        # conv layer
        out = self.conv_block(
            x, [self.conv1, self.conv2], self.ins_norm1, embrnn, res=True
        )
        out = self.conv_block(
            out, [self.conv3, self.conv4], self.ins_norm2, embd4, res=True
        )
        out = self.conv_block(
            out, [self.conv5, self.conv6], self.ins_norm3, embd2, res=True
        )
        # dense layer
        out = self.dense_block(
            out, emb8, [self.dense1, self.dense2], self.ins_norm4, res=True
        )
        out = self.dense_block(
            out, emb6, [self.dense3, self.dense4], self.ins_norm5, res=True
        )
        out_appended = append_emb(emb4, out.size(2), out)
        # rnn layer
        out_rnn = RNN(out_appended, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = append_emb(emb2, out.size(2), out)
        out = linear(out, self.dense5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = linear(out, self.linear)
        out = out.exp()
        return out


class Encoder(nn.Module):
    def __init__(
        self, c_in=513, c_h1=128, c_h2=512, c_h3=128, ns=0.2, dp=0.3, emb_size=128
    ):
        super().__init__()
        self.ns = ns
        self.conv1s = nn.ModuleList(
            nn.Conv1d(c_in, c_h1, kernel_size=k) for k in range(1, 8)
        )
        self.conv2 = nn.Conv1d(len(self.conv1s) * c_h1 + c_in, c_h2, kernel_size=1)
        self.emb2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Squeeze(dim=-1),
            nn.Linear(c_h2, emb_size),
        )
        self.conv3 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.emb4 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Squeeze(dim=-1),
            nn.Linear(c_h2, emb_size),
        )
        self.conv5 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.emb6 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Squeeze(dim=-1),
            nn.Linear(c_h2, emb_size),
        )
        self.conv7 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv8 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.emb8 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Squeeze(dim=-1),
            nn.Linear(c_h2, emb_size),
        )
        self.dense1 = nn.Linear(c_h2, c_h2)
        self.dense2 = nn.Linear(c_h2, c_h2)
        self.embd2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Squeeze(dim=-1),
            nn.Linear(c_h2, emb_size),
        )
        self.dense3 = nn.Linear(c_h2, c_h2)
        self.dense4 = nn.Linear(c_h2, c_h2)
        self.embd4 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Squeeze(dim=-1),
            nn.Linear(c_h2, emb_size),
        )
        self.RNN = nn.GRU(
            input_size=c_h2, hidden_size=c_h3, num_layers=1, bidirectional=True
        )
        self.embrnn = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            Squeeze(dim=-1),
            nn.Linear(c_h2, emb_size),
        )
        self.linear = nn.Linear(c_h2 + 2 * c_h3, c_h2)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h2)
        self.ins_norm2 = nn.InstanceNorm1d(c_h2)
        self.ins_norm3 = nn.InstanceNorm1d(c_h2)
        self.ins_norm4 = nn.InstanceNorm1d(c_h2)
        self.ins_norm5 = nn.InstanceNorm1d(c_h2)
        self.ins_norm6 = nn.InstanceNorm1d(c_h2)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.drop5 = nn.Dropout(p=dp)
        self.drop6 = nn.Dropout(p=dp)

    def conv_block(self, x, conv_layers, norm_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            x_pad = F.pad(x, pad=(0, x.size(2) % 2), mode="reflect")
            x_down = F.avg_pool1d(x_pad, kernel_size=2)
            out = x_down + out
        return out

    def dense_block(self, x, layers, norm_layers, res=True):
        out = x
        for layer in layers:
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = self.conv_block(
            out, [self.conv2], [self.ins_norm1, self.drop1], res=False
        )
        emb2 = self.emb2(out)
        out = self.conv_block(
            out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2]
        )
        emb4 = self.emb4(out)
        out = self.conv_block(
            out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3]
        )
        emb6 = self.emb6(out)
        out = self.conv_block(
            out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4]
        )
        emb8 = self.emb8(out)
        # dense layer
        out = self.dense_block(
            out, [self.dense1, self.dense2], [self.ins_norm5, self.drop5], res=True
        )
        embd2 = self.embd2(out)
        out = self.dense_block(
            out, [self.dense3, self.dense4], [self.ins_norm6, self.drop6], res=True
        )
        embd4 = self.embd4(out)
        out_rnn = RNN(out, self.RNN)
        embrnn = self.embrnn(out)
        out = torch.cat([out, out_rnn], dim=1)
        out = linear(out, self.linear)
        out = F.leaky_relu(out, negative_slope=self.ns)
        return (out, (emb2, emb4, emb6, emb8, embd2, embd4, embrnn))


class Model(nn.Module):
    def __init__(
        self,
        connection: str,
        enc_kwargs: dict = {},
        dec_kwargs: dict = {},
        emb_size: tuple = (128,),
    ):
        super().__init__()
        self.enc = Encoder(**enc_kwargs)
        self.dec = Decoder(**dec_kwargs)
        self.connection = connection
        self.emb = nn.Parameter(torch.zeros(emb_size), requires_grad=True)

    def forward(self, x):
        (e, c) = self.enc(x)

        if self.connection == "normal":
            pass
        elif self.connection == "emb":
            c = tuple(torch.stack([self.emb] * len(c[0]), dim=0)) * len(c)
        elif self.connection == "zero":
            c = (_con.new_zeros(_con.shape) for _con in c)
        else:
            raise ValueError

        return self.dec(e, c)
