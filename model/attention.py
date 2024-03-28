import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, embed_dim: int, heads: int, dropout: float, num_space_patches, num_time_patches, attn_type=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = nn.Dropout(dropout)
        assert (
            self.head_dim * heads == embed_dim
        ), "Embedding size needs to be divisible by heads"

        assert (
            attn_type in ['space', 'time']
        ), "Attention type should be either spacial or temporal"

        self.attn_type = attn_type
        self.num_space_patches = num_space_patches
        self.num_time_patches = num_time_patches

        self.values = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x):
        b, tn, d = x.shape
        x = x.reshape(b, self.num_time_patches, self.num_space_patches, d)
        if self.attn_type == 'space':
            x = self.forward_space(x)
            return x
        x = self.forward_time(x)
        return x

    def forward_space(self, x):
        b, t, n, d = x.shape
        x = x.reshape(b*t, n, d)
        x, _ = self.forward_attn(x, x, x)
        x = x.reshape(b, t, n, d)
        x = x.reshape(b, t*n, d)
        return x

    def forward_time(self, x):
        b, t, n, d = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b*n, t, d)
        x, _ = self.forward_attn(x, x, x)
        x = x.reshape(b, n, t, d)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, t*n, d)
        return x

    def forward_attn(self, query, keys, values, attn_mask=None, key_padding_mask=None):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if attn_mask is None:
            energy = energy / (self.embed_dim ** (1 / 2))
        else:
            energy = (energy + attn_mask) / (self.embed_dim ** (1 / 2))

        if key_padding_mask is None:
            attention = torch.softmax(energy, dim=3)
        else:
            attention = torch.softmax(energy + key_padding_mask, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        avg_attn = attention.sum(dim=1)
        avg_attn /= self.heads

        return self.dropout(out), avg_attn
