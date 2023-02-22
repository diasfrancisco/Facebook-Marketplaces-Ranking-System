from torchvision import transforms
from clean_data import clean_image_data
from PIL import Image
import io


def img_loader(user_input):
    transform = transforms.PILToTensor()
    img = Image.open(io.BytesIO(user_input))
    img = clean_image_data(img)
    img = transform(img)
    img = img[None, :]
    
    return img

if __name__ == "__main__":
    img_loader()