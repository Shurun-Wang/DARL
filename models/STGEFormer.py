# ICASSP2025 STGE-Former: Spatial-Temporal Graph-Enhanced Transformer for EEG-Based Major Depressive Disorder Detection

from einops import repeat
import torch.nn.functional as F
import numpy as np
from math import sqrt
from einops import rearrange

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = x + self.position_embedding(x)
        return self.dropout(x), n_vars



class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nwl,vw->nvl', (x, A))
        return x.contiguous()


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout=0.2, alpha=0.1):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        d = adj.sum(1)
        a = adj / (d.view(-1, 1)+1e-8)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = F.dropout(h, self.dropout)
            h = self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=2)
        ho = self.mlp(ho)
        return ho


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=1, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, node_emb):
        nodevec1 = F.gelu(self.alpha * self.lin1(node_emb))
        nodevec2 = F.gelu(self.alpha * self.lin2(node_emb))
        adj = F.relu(torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0)))

        if self.k < node_emb.shape[0]:
            n_nodes = node_emb.shape[0]
            mask = torch.zeros(n_nodes, n_nodes).to(node_emb.device)
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
        return adj


class STGEFormer(nn.Module):
    def get_size(self, channel, timepoint):
        data = torch.ones((1, channel, timepoint))
        data = self.feature(data)
        return data.size()

    def __init__(self, chans, cfg):
        super().__init__()
        enc_in = chans
        self.patch_embedding = PatchEmbedding(cfg['patch_embed_d_model'], cfg['patch_len'], cfg['patch_stride'], cfg['patch_drop'])
        self.cls_token = nn.Parameter(torch.randn(1, 3, cfg['patch_embed_d_model']))
        self.value_embedding = nn.Linear(2000, 512)

        self.encoder_layer_1 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['encoder_1_drop'], output_attention=False),
                           cfg['encoder_d_model'], cfg['encoder_1_n_heads']),
            cfg['encoder_d_model'], cfg['encoder_1_diff'], dropout=cfg['encoder_1_dropout'], activation="gelu"
        )
        self.encoder_layer_2 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['encoder_2_drop'], output_attention=False),
                           cfg['encoder_d_model'], cfg['encoder_2_n_heads']),
            cfg['encoder_d_model'], cfg['encoder_2_diff'], dropout=cfg['encoder_2_dropout'], activation="gelu"
        )
        self.encoder_layer_3 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['encoder_3_drop'], output_attention=False),
                           cfg['encoder_d_model'], cfg['encoder_3_n_heads']),
            cfg['encoder_d_model'], cfg['encoder_3_diff'], dropout=cfg['encoder_3_dropout'], activation="gelu"
        )
        self.encoder_layer_4 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['encoder_4_drop'], output_attention=False),
                           cfg['encoder_d_model'], cfg['encoder_4_n_heads']),
            cfg['encoder_d_model'], cfg['encoder_4_diff'], dropout=cfg['encoder_4_dropout'], activation="gelu"
        )

        self.mixprop_1 = mixprop(cfg['encoder_d_model'], cfg['encoder_d_model'], gdep=3)
        self.mixprop_2 = mixprop(cfg['encoder_d_model'], cfg['encoder_d_model'], gdep=3)
        self.mixprop_3 = mixprop(cfg['encoder_d_model'], cfg['encoder_d_model'], gdep=3)

        self.graph_learning = graph_constructor(enc_in, cfg['knn'], cfg['graph_embed_dim'], alpha=1)
        self.node_embs = nn.Parameter(torch.randn(enc_in, cfg['graph_embed_dim']), requires_grad=True)

        # Norm layer
        self.encoder_norm_1 = nn.LayerNorm(cfg['encoder_d_model'])
        self.encoder_norm_2 = nn.LayerNorm(cfg['encoder_d_model'])
        self.encoder_norm_3 = nn.LayerNorm(cfg['encoder_d_model'])
        self.encoder_norm_4 = nn.LayerNorm(cfg['encoder_d_model'])

        self.spatial_layer_1 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['spatial_1_drop'], output_attention=False),
                           enc_in, cfg['spatial_1_n_heads']),
            enc_in, cfg['spatial_1_diff'], dropout=cfg['spatial_1_dropout'], activation="gelu"
        )
        self.spatial_layer_2 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['spatial_2_drop'], output_attention=False),
                           enc_in, cfg['spatial_2_n_heads']),
            enc_in, cfg['spatial_2_diff'], dropout=cfg['spatial_2_dropout'], activation="gelu"
        )
        self.spatial_layer_3 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['spatial_3_drop'], output_attention=False),
                           enc_in, cfg['spatial_3_n_heads']),
            enc_in, cfg['spatial_3_diff'], dropout=cfg['spatial_3_dropout'], activation="gelu"
        )
        self.spatial_layer_4 = EncoderLayer(
            AttentionLayer(FullAttention(attention_dropout=cfg['spatial_4_drop'], output_attention=False),
                           enc_in, cfg['spatial_4_n_heads']),
            enc_in, cfg['spatial_4_diff'], dropout=cfg['spatial_4_dropout'], activation="gelu"
        )
        self.spatial_norm = torch.nn.LayerNorm(enc_in)
        self.cls_len = 3
        self.act = F.gelu
        self.dropout = nn.Dropout(cfg['dropout'])
        self.linear_size = self.get_size(chans, 2000)[1]
        self.classifer_head = nn.Sequential(
            nn.Linear(self.linear_size, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def feature(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_enc = self.value_embedding(x_enc)

        enc_out, n_vars = self.patch_embedding(x_enc)
        patch_len = enc_out.shape[1]
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=enc_out.shape[0])
        enc_out = torch.cat([cls_tokens, enc_out], dim=1)

        adj = self.graph_learning(self.node_embs)

        enc_out, _ = self.encoder_layer_1(enc_out)

        g = enc_out[:, :self.cls_len]
        g = rearrange(g, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])
        g = self.mixprop_1(g, adj) + g
        g = rearrange(g, '(b p) n d -> (b n) p d', p=self.cls_len)
        enc_out = torch.cat([g, enc_out[:, self.cls_len:]], dim=1)
        enc_out = self.encoder_norm_1(enc_out)

        enc_out, _ = self.encoder_layer_2(enc_out)
        g = enc_out[:, :self.cls_len]
        g = rearrange(g, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])
        g = self.mixprop_2(g, adj) + g
        g = rearrange(g, '(b p) n d -> (b n) p d', p=self.cls_len)
        enc_out = torch.cat([g, enc_out[:, self.cls_len:]], dim=1)
        enc_out = self.encoder_norm_2(enc_out)

        enc_out, _ = self.encoder_layer_3(enc_out)

        g = enc_out[:, :self.cls_len]
        g = rearrange(g, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])
        g = self.mixprop_3(g, adj) + g
        g = rearrange(g, '(b p) n d -> (b n) p d', p=self.cls_len)
        enc_out = torch.cat([g, enc_out[:, self.cls_len:]], dim=1)
        enc_out = self.encoder_norm_3(enc_out)

        enc_out, _ = self.encoder_layer_4(enc_out)
        enc_out = self.encoder_norm_4(enc_out)

        enc_out = enc_out[:, -patch_len:, :]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        enc_out = torch.reshape(enc_out, (enc_out.shape[0], -1))

        enc_out_spatial = x_enc
        enc_out_spatial = enc_out_spatial.permute(0, 2, 1)

        enc_out_spatial, _ = self.spatial_layer_1(enc_out_spatial)
        enc_out_spatial, _ = self.spatial_layer_2(enc_out_spatial)
        enc_out_spatial, _ = self.spatial_layer_3(enc_out_spatial)
        enc_out_spatial, _ = self.spatial_layer_4(enc_out_spatial)
        enc_out_spatial = self.spatial_norm(enc_out_spatial)

        enc_out_spatial = torch.reshape(enc_out_spatial, (enc_out_spatial.shape[0], -1))

        enc_out = torch.cat([enc_out, enc_out_spatial], dim=1)
        enc_out = self.act(enc_out)
        enc_out = self.dropout(enc_out)

        return enc_out

    def forward(self, x):
        dec_out = self.feature(x)
        x = self.classifer_head(dec_out)
        return x

