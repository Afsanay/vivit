import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            in_channels, embed_dim, patch_size, patch_size)
        self.flatten = nn.Flatten(2, 4)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)


if __name__ == "__main__":
    pe = PatchEmbedding(3, 256, 16)
    x = torch.randn(1, 3, 32, 112, 112)
    print("Patch Embedder:", pe(x).shape)
