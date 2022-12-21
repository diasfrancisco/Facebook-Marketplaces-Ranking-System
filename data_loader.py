import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class ImageDataset(Dataset):
    """
    Creates a custom dataset for the cleaned images
    """
    def __init__(self, img_dir, transform=None):
        # Initialises the parent class, allowing us to use the methods it contains
        super().__init__()
        self.img_labels = pd.read_csv("./data/LabelledImages.csv")
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        # Grabs the image at a certain index
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1] + '.jpg')
        # Converts the image into a tensor using torchvision's read_image() function
        image = read_image(img_path)
        # Grabs the label corresponding to the index provided
        label = self.img_labels.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.img_labels)
    
image_dataset = ImageDataset(img_dir="./data/cleaned_images/",
                             transform=None)
random_sample = image_dataset.__getitem__(idx=113)
print(random_sample)