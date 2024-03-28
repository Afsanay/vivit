import torch
import torch.nn as nn
from model.encoder import Encoder
from model.classifier import Classifier
from model.embedding import Embedding
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"


class ViVIT(nn.Module):
    def __init__(self, in_channels, image_size, clip_size, num_classes, embed_dim, patch_size, mlp_size, num_heads, num_layers, dropout):
        super().__init__()

        num_space_patches = (image_size // patch_size)**2
        num_time_patches = clip_size // patch_size
        self.embedder = Embedding(
            in_channels, embed_dim, patch_size)
        self.encoder = Encoder(embed_dim, mlp_size, num_heads,
                               num_layers, dropout, num_space_patches, num_time_patches)
        self.classfier = Classifier(embed_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.embedder(x))
        x = self.encoder(x)
        x = self.classfier(x[:, 0])
        return x


if __name__ == "__main__":
    vivit = ViVIT(3, 224, 32, 10, 256, 16, 256*4, 8, 8, 0.1).to(device)
    torch.cuda.empty_cache()
    summary(model=vivit,
            input_size=[
                (32, 3, 32, 224, 224)
            ], col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20, row_settings=["var_names"])
