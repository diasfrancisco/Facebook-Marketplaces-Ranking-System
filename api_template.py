from fastapi import FastAPI, HTTPException, UploadFile
from image_processor import img_loader
from image_cnn import CNN, FtCNN
import pandas as pd
import torch

api = FastAPI()

def load_model(fp: str, model: str):
    if model == 'clf':
        model = CNN()
    else:
        model = FtCNN()
    checkpoint = torch.load(fp)
    model.load_state_dict(checkpoint)
    
    return model.eval()

@api.get('/')
def root():
    welcome = {'message': 'Welcome to the Facebook MarketPlace Search Ranking System'}
    
    return welcome['message']

@api.post('/classifier')
def img_clf(user_input: UploadFile):
    if not user_input:
        raise HTTPException(status_code=400, detail='Please provide a valid image')
    elif user_input.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(status_code=415, detail='Unsupported media type. Please use JPEG/PNG')
    
    labels_csv = pd.read_csv('./data/Labels.csv').rename(columns={'Unnamed: 0' : 'idx'}).to_dict()
    encoder_dict = dict((k,v) for k,v in labels_csv['Category'].items())
    
    file_contents = user_input.file.read()
    img = img_loader(file_contents)
    
    clf_model = load_model(fp='./final_models/clf_model.pt', model='clf')
    pred = clf_model(img)
    _, out_class = torch.max(pred, dim=1)
    
    return encoder_dict[out_class.item()]
    
@api.post('/ranker')
def ranking_sys(user_input: UploadFile):
    if not user_input:
        raise HTTPException(status_code=400, detail='Please provide a valid image')
    elif user_input.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(status_code=415, detail='Unsupported media type. Please use JPEG/PNG')
    
    file_contents = user_input.file.read()
    img = img_loader(file_contents)
    emb_store = torch.load('image_embeddings.json')
    
    ft_model = load_model(fp='./final_models/ft_model.pt', model='ft')
    fts = ft_model(img)
    emb = fts[0]
    
    return emb.shape