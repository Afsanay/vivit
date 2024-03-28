import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
