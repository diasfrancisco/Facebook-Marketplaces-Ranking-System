import os
import torchvision
from PIL import Image


def img_loader():
    path = './data/images/'
    dirs = os.listdir(path)
    for item in dirs:
        img = Image.open(path+item)
        img = transform(img)

transform = torchvision.transforms.PILToTensor()
img_loader()