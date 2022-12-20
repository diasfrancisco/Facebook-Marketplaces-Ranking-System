import os
import pandas as pd
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.img_dir = "./data/cleaned_images/"
        self.img_labels = pd.read_csv("./data/Labels.csv")

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

    def __len__(self):
        return len(self.img_labels)