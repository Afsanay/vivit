import torch
import torch.nn as nn
from model.encoderLayer import EncoderLayer
from copy import deepcopy


class Encoder(nn.Module):
    def __init__(self, embed_dim, mlp_size, num_heads, num_layers, dropout, num_space_patches, num_time_patches):
        super().__init__()
        encoder_layer = EncoderLayer(
            embed_dim, num_heads, dropout, mlp_size, num_space_patches, num_time_patches)
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
