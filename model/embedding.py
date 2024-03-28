import torch
import torch.nn as nn
from model.patchEmbedding import PatchEmbedding

device = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_dim):
        super().__init__()
        self.position = nn.Embedding(patch_size**2, embedding_dim)

    def forward(self, x):
        batch_size, max_len, _ = x.shape
        positions = torch.arange(0, max_len).expand(
            batch_size, max_len).to(device)
        pos = self.position(positions)
        x = x + pos
        return x


class Embedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels, embed_dim, patch_size)
        self.positional_embed = PositionalEmbedding(
            patch_size, embed_dim)

    def forward(self, x):
        patch_embed = self.patch_embed(x)
        total_embed = self.positional_embed(patch_embed)

        return total_embed


if __name__ == "__main__":
    embed = Embedding(3, 256, 16)
    x = torch.randn(1, 3, 32, 224, 224)
    label = torch.LongTensor([[1]])
    print(embed(x).shape)
