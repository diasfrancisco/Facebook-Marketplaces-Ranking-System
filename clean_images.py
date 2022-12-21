from PIL import Image
import os


def clean_image_data(final_size, im):
    size = im.size
    ratio = float(final_size)/max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
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
    final_size = 512
    # Loops through all the files in the directory and runs the clean_image_data()
    # function on each. The cleaned images are saved in the cleaned_images directory
    for item in dirs:
        im = Image.open(path + item)
        new_im = clean_image_data(final_size, im)
        new_im.save(f'./data/cleaned_images/{item}')