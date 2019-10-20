import torch
from torch import nn
from torch.nn import functional as F

from layers import Conv1dNorm, TanhAttention
from models import Decoder, Encoder


class Tacotron(nn.Module):
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
        if start_token == None:
            start_token = x.new_zeros(
                size=[1, len(x), self.dec.decoder_rnn.hidden_size]
            )
        if starting_states == None:
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
        encoded, state = self.enc(x)
        decoded = self.dec(
            encoded=encoded,
            state_enc=state,
            start_token=start_token,
            starting_states=starting_states,
            max_len=max_len,
        )
        return decoded.permute(1, 2, 0)
