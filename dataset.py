import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Creates a custom image dataset

    Args:
        Dataset (class): An abstract class provided by PyTorch
    """    
    def __init__(self, img_dir, transform=None, ft_extraction=False):
        # Initialises the parent class, allowing us to use the methods it contains as well as any attributes
        super().__init__()
        self.img_labels = pd.read_csv("./data/LabelledImages.csv")
        self.img_dir = img_dir
        self.transform = transforms.PILToTensor()
        self.ft_extraction = ft_extraction

    def __getitem__(self, idx):
        # Grabs the image at a certain index
        img_id = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        # Converts the image into a tensor using torchvision's read_image() function
        image = Image.open(img_path)
        # Grabs the label corresponding to the index provided
        label = self.img_labels.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        if self.ft_extraction:
            return image, label, str(img_id)
        else:
            return image, label

    def __len__(self):
        return len(self.img_labels)