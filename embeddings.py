import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset
from image_cnn import CNN
import config

def product_embeddings():
    embeddings_dict = {}
    train_data = ImageDataset(img_dir="./data/cleaned_images/", transform=True, ft_extraction=True)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    model = CNN().to(device=config.DEVICE)
    checkpoint = torch.load('model_evaluation/21_Feb_2023_18_26_20_807442/weights/epoch19_weights.pt', map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    
    for batch in train_loader:
        x, y, img_id = batch
        x, y, img_id = x.to(device=config.DEVICE), y.to(device=config.DEVICE), img_id
        emb = model(x.float())
        embeddings_dict[img_id[0]] = emb
    
    torch.save(embeddings_dict, 'image_embeddings.json')
    
product_embeddings()