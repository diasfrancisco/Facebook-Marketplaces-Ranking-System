import os
import pandas as pd
from PIL import Image

import config


def clean_text_data():
    df = pd.read_csv("./data/Products.csv", lineterminator="\n")
    df["price"].str.replace("£", "")
    
def clean_image_data(im):
    size = im.size
    ratio = float(config.FINAL_SIZE)/max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (config.FINAL_SIZE, config.FINAL_SIZE))
    new_im.paste(im, ((config.FINAL_SIZE-new_image_size[0])//2, (config.FINAL_SIZE-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':
    # Create a directory called 'cleaned_images' if it doesn't exist
    if os.path.isdir("./data/cleaned_images"):
        pass
    else:
        os.mkdir("./data/cleaned_images")

    # Set the path, directory, and final size of the images
    path = './data/images/' 
    dirs = os.listdir(path)

    # Loops through all the files in the directory and runs the clean_image_data()
    # function on each. The cleaned images are saved in the cleaned_images directory
    for item in dirs:
        im = Image.open(path + item)
        new_im = clean_image_data(config.FINAL_SIZE, im)
        new_im.save(f'./data/cleaned_images/{item}')
        
    clean_text_data()