import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # optional predictor+corrector heads
        self.enable_corrector = getattr(configs, 'enable_corrector', False)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if self.enable_corrector:
            self.projector_pred = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            self.projector_corr = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        else:
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        if self.enable_corrector:
            dec_pred = self.projector_pred(enc_out).permute(0, 2, 1)[:, :, :N]
            dec_corr = self.projector_corr(enc_out).permute(0, 2, 1)[:, :, :N]
            dec_out = dec_pred + dec_corr
        else:
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            scale = (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            shift = (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            if self.enable_corrector:
                dec_pred = dec_pred * scale + shift
                dec_corr = dec_corr * scale  # residual in value scale
                dec_out = dec_pred + dec_corr
            else:
                dec_out = dec_out * scale + shift

        if self.enable_corrector:
            extra = {
                'out': dec_out,
                'pred': dec_pred,
                'corr': dec_corr,
                'attn': attns,
                'feat': enc_out,
            }
            return extra
        else:
            return dec_out, attns, enc_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.enable_corrector:
            # Align horizon to last pred_len for safety
            for k in ['out', 'pred', 'corr']:
                if out[k].shape[1] > self.pred_len:
                    out[k] = out[k][:, -self.pred_len:, :]
            return out
        else:
            dec_out, attns, enc_out = out

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns, enc_out
            else:
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
