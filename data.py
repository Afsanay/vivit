from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import torch
import cv2
from torchvision import transforms

IMG_SIZE = 16
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class VideoDataset(Dataset):
    def __init__(self):
        super().__init__()
        file_path = "data.txt"
        self.dir_path = "/UCF101/UCF-101/"
        self.pairs = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                video, cls = line.rsplit()
                self.pairs.append([video, int(cls)-1])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_path, cls = self.pairs[idx]
        frames = []

        cap = cv2.VideoCapture(os.path.join(self.dir_path, video_path))
        i = 0
        while (cap.isOpened() and i < 16):
            i += 1
            success, frame = cap.read()
            if not success:
                break
            frames.append(transform(frame))
        video = torch.stack(frames).permute(1, 0, 2, 3)
        return {
            "video": video,
            "class": torch.tensor(cls)
        }


if __name__ == "__main__":
    data = VideoDataset()
    dl = DataLoader(data, 32)
    print(next(iter(dl))["video"].shape)
    # print(data.__getitem__(10)["video"].shape)
