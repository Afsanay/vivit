import torch.nn as nn
from model.transformer import ViVIT
from data import VideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
import os
import torch

NUM_WORKERS = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(num_epochs):

    vivit = ViVIT(3, 16, 16, 101, 256, 16, 256*4, 8, 4, 0.1).to(device)
    data = VideoDataset()
    dataloader = DataLoader(data, 32, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)
    optim = SGD(vivit.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        vivit.train()
        iterator = tqdm(dataloader, desc=f"Processing Epoch: {epoch}")
        for batch in iterator:
            video = batch["video"].to(device)
            cls = batch["class"].to(device)
            output = vivit(video)

            loss = loss_fn(cls, output)
            iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        torch.save(vivit.state_dict(), f'saved_models/model_{epoch}.pth')


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train(60)
