import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as nn_init

from .layers import AttentionLayer, Conv1dNorm, HighWay


class CBHG(nn.Module):
    def __init__(self, features, proj_size, rnn_size, rnn_layers, K, ns, max_pooling):
        super().__init__()
        conv_banks = [
            Conv1dNorm(
                in_channels=features,
                out_channels=features,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                activation=nn.LeakyReLU(negative_slope=ns),
            )
            for k in range(1, 1 + K)
        ]
        self.conv_banks = nn.ModuleList(conv_banks)
        if max_pooling:
            self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        conv_proj = [
            Conv1dNorm(
                in_channels=in_size,
                out_channels=out_size,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=activ,
            )
            for in_size, out_size, activ in zip(
                (K * features, *proj_size[:-1]),
                proj_size,
                [nn.LeakyReLU(negative_slope=ns) for _ in range(len(proj_size) - 1)]
                + [None],
            )
        ]
        self.conv_proj = nn.ModuleList(conv_proj)
        self.pre_highway = nn.Linear(in_features=proj_size[-1], out_features=features)
        self.highways = nn.ModuleList([HighWay(features, ns=ns) for _ in range(4)])
        self.rnn = nn.GRU(
            input_size=features,
            hidden_size=rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
        )

    def forward(self, x):
        # x : batch, features, timesteps
        timesteps = x.size(-1)
        outs = [conv(x)[..., :timesteps] for conv in self.conv_banks]
        outs = torch.cat(outs, dim=1)
        try:
            outs = self.maxpool1d(outs)[..., :timesteps]
        except AttributeError:
            pass
        for conv in self.conv_proj:
            outs = conv(outs)
        outs = outs.permute(2, 0, 1)
        outs = self.pre_highway(outs)
        for highway in self.highways:
            outs = highway(outs)
        return self.rnn(outs)


class PreNet(nn.Module):
    def __init__(self, features, ns):
        super().__init__()
        in_features = features[:-1]
        out_features = features[1:]
        layers = sum(
            (
                (
                    nn.Linear(in_features=in_size, out_features=out_size),
                    nn.InstanceNorm1d(num_features=out_size),
                    nn.LeakyReLU(negative_slope=ns),
                    nn.Dropout(0.5),
                )
                for in_size, out_size in zip(in_features, out_features)
            ),
            (),
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        n_channels,
        prenet_units,
        proj_size,
        cbhg_maxpool,
        rnn_size,
        rnn_layers,
        K,
        ns,
    ):
        super().__init__()
        self.prenet = PreNet(features=[n_channels] + list(prenet_units), ns=ns)
        self.cbhg = CBHG(
            features=prenet_units[-1],
            proj_size=proj_size,
            rnn_size=rnn_size,
            rnn_layers=rnn_layers,
            K=K,
            ns=ns,
            max_pooling=cbhg_maxpool,
        )

    def forward(self, signal):
        # signal: (batch, features, timesteps)
        signal = signal.transpose(1, 2)
        processed = self.prenet(signal)
        processed.transpose_(1, 2)
        return self.cbhg(processed)


class Decoder(nn.Module):
    def __init__(
        self,
        prenet_units,
        hidden_sizes,
        cbhg_maxpool,
        rnn_layers,
        proj_size,
        cbhg_rnn,
        n_channels,
        K,
        R,
        ns,
    ):
        super().__init__()
        prev_rnn, self_rnn = hidden_sizes
        self.prenet_enc = PreNet(features=[2 * prev_rnn] + list(prenet_units), ns=ns)
        self.prenet_dec = PreNet(features=[self_rnn] + list(prenet_units), ns=ns)
        self.prenet_dec_transform = nn.Linear(
            in_features=prenet_units[-1], out_features=2 * prev_rnn
        )
        self.prenet_enc_transform = nn.Linear(
            in_features=prenet_units[-1], out_features=2 * prev_rnn
        )
        self.attn_enc = AttentionLayer(
            input_size=prenet_units[-1], hidden_size=2 * prev_rnn
        )
        self.attn_dec = AttentionLayer(
            input_size=prenet_units[-1], hidden_size=2 * prev_rnn
        )
        self.pre_decoder_rnn = nn.Linear(
            in_features=4 * 2 * prev_rnn, out_features=self_rnn
        )
        self.decoder_rnn = nn.GRU(
            input_size=self_rnn, hidden_size=self_rnn, num_layers=rnn_layers
        )
        # self.decoder_attention = AttentionLayer()
        cbhg_rnn_size, cbhg_rnn_layers = cbhg_rnn
        self.cbhg = CBHG(
            features=self_rnn,
            proj_size=proj_size,
            rnn_size=cbhg_rnn_size,
            rnn_layers=cbhg_rnn_layers,
            K=K,
            ns=ns,
            max_pooling=cbhg_maxpool,
        )
        self.out = nn.Linear(in_features=2 * cbhg_rnn_size, out_features=n_channels)
        self.multiple_frames = nn.Linear(
            in_features=self_rnn, out_features=R * self_rnn
        )
        self.R = R
        self.leaky_relu = nn.LeakyReLU(negative_slope=ns)

    def forward(self, encoded, state_enc, start_token, starting_states, max_len):
        assert max_len % self.R == 0
        decoded = [start_token]
        state_dec, _state_dec = starting_states
        state_enc = state_enc.view(state_enc.size(1), -1)
        for _ in range(max_len // self.R):
            current_decoded = torch.cat(decoded, dim=0)
            reduced_encoded = self.prenet_enc(encoded)
            reduced_decoded = self.prenet_dec(current_decoded)
            transformed_encoded = self.prenet_enc_transform(reduced_encoded)
            state_enc, alignment_enc = self.attn_enc(
                reduced_encoded[-1], state_enc, transformed_encoded
            )
            transformed_decoded = self.prenet_dec_transform(reduced_decoded)
            state_dec, alignment_dec = self.attn_dec(
                reduced_decoded[-1], state_dec, transformed_decoded
            )
            concat = torch.cat(
                [state_enc, alignment_enc, state_dec, alignment_dec], dim=-1
            )
            concat = self.leaky_relu(concat)
            concat = self.pre_decoder_rnn(concat)
            concat = self.leaky_relu(concat)
            concat = concat.transpose(0, 1)
            dec_out, _state_dec = self.decoder_rnn(concat, _state_dec)
            dec_out = dec_out + concat
            sizes = dec_out.size()
            dec_out = self.leaky_relu(dec_out)
            dec_out = self.multiple_frames(dec_out)
            dec_out = dec_out.view(self.R, *sizes[1:])
            decoded.append(dec_out)
            state_enc, state_dec = state_enc.squeeze(1), state_dec.squeeze(1)
        decoded = torch.cat(decoded[1:], dim=0)
        decoded = decoded.permute(1, 2, 0)
        dec_out, _ = self.cbhg(decoded)
        dec_out = self.leaky_relu(dec_out)
        out = self.out(dec_out)
        return out
