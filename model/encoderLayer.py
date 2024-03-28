import torch
import torch.nn as nn
import torchvision
from model.attention import Attention
from model.mlp import MLPBlock


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, dropout, mlp_size, num_space_patches, num_time_patches):
        super().__init__()

        self.num_space_patches = num_space_patches
        self.num_time_patches = num_time_patches

        half_heads = int(heads/2.0)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention_space = Attention(
            embed_dim, half_heads, dropout, num_space_patches, num_time_patches, 'space')

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.attention_time = Attention(
            embed_dim, half_heads, dropout, num_space_patches, num_time_patches, 'time')

        self.linear = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.Dropout(dropout)
        )

        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(mlp_size, embed_dim, dropout)

    def forward(self, x):

        x_space = self.attention_space(self.layer_norm1(x))
        x_time = self.attention_time(self.layer_norm2(x))
        total_attn = torch.cat([x_space, x_time], dim=2)
        total_attn = self.linear(total_attn)
        x = total_attn + x

        out = self.mlp(x)
        out = out + x

        return out
