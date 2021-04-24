"""
This module contains a tacotron model that can be reused.
"""

import torch
from torch.nn import (
    GRU,
    Dropout,
    InstanceNorm1d,
    LeakyReLU,
    Linear,
    MaxPool1d,
    Module,
    ModuleList,
)

from .layers import AttentionLayer, Conv1dNorm, HighWay


class CBHG(Module):
    "CBHG network"

    def __init__(self, features, proj_size, rnn_size, rnn_layers, K, ns, max_pooling):
        super().__init__()
        conv_banks = [
            Conv1dNorm(
                in_channels=features,
                out_channels=features,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                activation=LeakyReLU(negative_slope=ns),
            )
            for k in range(1, 1 + K)
        ]
        self.conv_banks = ModuleList(conv_banks)
        if max_pooling:
            self.maxpool1d = MaxPool1d(kernel_size=2, stride=1, padding=1)
        conv_proj = [
            Conv1dNorm(
                in_channels=in_size,
                out_channels=out_size,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=activ,
            )
            for (in_size, out_size, activ) in zip(
                (K * features, *proj_size[:-1]),
                proj_size,
                (LeakyReLU(negative_slope=ns) for _ in range(len(proj_size) - 1))
                + [None],
            )
        ]
        self.conv_proj = ModuleList(conv_proj)
        self.pre_highway = Linear(in_features=proj_size[-1], out_features=features)
        self.highways = ModuleList(HighWay(features, ns=ns) for _ in range(4))
        self.rnn = GRU(
            input_size=features,
            hidden_size=rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
        )

    def forward(self, x):
        "Pass through"
        # x : batch, features, timesteps
        timesteps = x.size(-1)
        outs = torch.cat([conv(x)[..., :timesteps] for conv in self.conv_banks], dim=1)
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


class PreNet(Module):
    "Encoder's door step"

    def __init__(self, features, ns):
        super().__init__()
        in_features = features[:-1]
        out_features = features[1:]
        layers = sum(
            (
                (
                    Linear(in_features=in_size, out_features=out_size),
                    InstanceNorm1d(num_features=out_size),
                    LeakyReLU(negative_slope=ns),
                    Dropout(0.5),
                )
                for (in_size, out_size) in zip(in_features, out_features)
            ),
            (),
        )
        self.layers = ModuleList(layers)

    def forward(self, x):
        "Pass through"
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(Module):
    "Encoder of tactron"

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
        "Pass through"
        # signal: (batch, features, timesteps)
        signal = signal.transpose(1, 2)
        processed = self.prenet(signal)
        processed.transpose_(1, 2)
        return self.cbhg(processed)


class Decoder(Module):
    "Decoder of tactron"

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
        (prev_rnn, self_rnn) = hidden_sizes
        self.prenet_enc = PreNet(features=[2 * prev_rnn] + list(prenet_units), ns=ns)
        self.prenet_dec = PreNet(features=[self_rnn] + list(prenet_units), ns=ns)
        self.prenet_dec_transform = Linear(
            in_features=prenet_units[-1], out_features=2 * prev_rnn
        )
        self.prenet_enc_transform = Linear(
            in_features=prenet_units[-1], out_features=2 * prev_rnn
        )
        self.attn_enc = AttentionLayer(
            input_size=prenet_units[-1], hidden_size=2 * prev_rnn
        )
        self.attn_dec = AttentionLayer(
            input_size=prenet_units[-1], hidden_size=2 * prev_rnn
        )
        self.pre_decoder_rnn = Linear(
            in_features=4 * 2 * prev_rnn, out_features=self_rnn
        )
        self.decoder_rnn = GRU(
            input_size=self_rnn, hidden_size=self_rnn, num_layers=rnn_layers
        )
        # self.decoder_attention = AttentionLayer()
        (cbhg_rnn_size, cbhg_rnn_layers) = cbhg_rnn
        self.cbhg = CBHG(
            features=self_rnn,
            proj_size=proj_size,
            rnn_size=cbhg_rnn_size,
            rnn_layers=cbhg_rnn_layers,
            K=K,
            ns=ns,
            max_pooling=cbhg_maxpool,
        )
        self.out = Linear(in_features=2 * cbhg_rnn_size, out_features=n_channels)
        self.multiple_frames = Linear(in_features=self_rnn, out_features=R * self_rnn)
        self.R = R
        self.leaky_relu = LeakyReLU(negative_slope=ns)

    def forward(self, encoded, state_enc, start_token, starting_states, max_len):
        "Pass through"
        assert max_len % self.R == 0
        decoded = [start_token]
        (state_dec, _state_dec) = starting_states
        state_enc = state_enc.view(state_enc.size(1), -1)
        for _ in range(max_len // self.R):
            current_decoded = torch.cat(decoded, dim=0)
            reduced_encoded = self.prenet_enc(encoded)
            reduced_decoded = self.prenet_dec(current_decoded)
            transformed_encoded = self.prenet_enc_transform(reduced_encoded)
            (state_enc, alignment_enc) = self.attn_enc(
                reduced_encoded[-1], state_enc, transformed_encoded
            )
            transformed_decoded = self.prenet_dec_transform(reduced_decoded)
            (state_dec, alignment_dec) = self.attn_dec(
                reduced_decoded[-1], state_dec, transformed_decoded
            )
            concat = torch.cat(
                [state_enc, alignment_enc, state_dec, alignment_dec], dim=-1
            )
            concat = self.leaky_relu(concat)
            concat = self.pre_decoder_rnn(concat)
            concat = self.leaky_relu(concat)
            concat = concat.transpose(0, 1)
            (dec_out, _state_dec) = self.decoder_rnn(concat, _state_dec)
            dec_out = dec_out + concat
            sizes = dec_out.size()
            dec_out = self.leaky_relu(dec_out)
            dec_out = self.multiple_frames(dec_out)
            dec_out = dec_out.view(self.R, *sizes[1:])
            decoded.append(dec_out)
            (state_enc, state_dec) = (state_enc.squeeze(1), state_dec.squeeze(1))
        decoded = torch.cat(decoded[1:], dim=0)
        decoded = decoded.permute(1, 2, 0)
        (dec_out, _) = self.cbhg(decoded)
        dec_out = self.leaky_relu(dec_out)
        out = self.out(dec_out)
        return out


class Tacotron(Module):
    "Tacotron 2 model"

    def __init__(
        self,
        n_channels,
        r,
        ns,
        enc_prenet_units,
        enc_proj_size,
        enc_rnn_size,
        enc_rnn_layers,
        enc_k,
        dec_prenet_units,
        dec_proj_size,
        dec_rnn_size,
        dec_rnn_layers,
        dec_k,
        main_rnn_size,
        main_rnn_layers,
        enc_cbhg_maxpool,
        dec_cbhg_maxpool,
    ):
        super().__init__()
        self.enc = Encoder(
            n_channels=n_channels,
            prenet_units=enc_prenet_units,
            proj_size=enc_proj_size,
            cbhg_maxpool=enc_cbhg_maxpool,
            rnn_size=enc_rnn_size,
            rnn_layers=enc_rnn_layers,
            K=enc_k,
            ns=ns,
        )
        self.dec = Decoder(
            prenet_units=dec_prenet_units,
            hidden_sizes=(enc_rnn_size, main_rnn_size),
            rnn_layers=main_rnn_layers,
            proj_size=dec_proj_size,
            cbhg_maxpool=dec_cbhg_maxpool,
            cbhg_rnn=(dec_rnn_size, dec_rnn_layers),
            n_channels=n_channels,
            K=dec_k,
            R=r,
            ns=ns,
        )

    def forward(self, x, max_len, start_token=None, starting_states=None):
        "Pass through"
        if start_token is None:
            start_token = x.new_zeros(
                size=[1, len(x), self.dec.decoder_rnn.hidden_size]
            )
        if starting_states is None:
            starting_states = [
                x.new_zeros(size=[len(x), 2 * self.enc.cbhg.rnn.hidden_size]),
                x.new_zeros(
                    size=[
                        self.dec.decoder_rnn.num_layers,
                        len(x),
                        self.dec.decoder_rnn.hidden_size,
                    ]
                ),
            ]
        (encoded, state) = self.enc(x)
        decoded = self.dec(
            encoded=encoded,
            state_enc=state,
            start_token=start_token,
            starting_states=starting_states,
            max_len=max_len,
        )
        return decoded.permute(1, 2, 0)
