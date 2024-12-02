import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# Custom Dataset class to load data from CSV
class MURADataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, sep=",", header=None, names=['image_path', 'label'])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        # Load image
        image = Image.open(img_path).convert("RGB")
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, int(label)