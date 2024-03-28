import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, mlp_size, embedding_dim, p_dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
