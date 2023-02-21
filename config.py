import torch
from datetime import datetime


# Data preprocessing
FINAL_SIZE = 512

# Training
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIMESTAMP = datetime.now().strftime('%d_%b_%Y_%H_%M_%S_%f')
EPOCHS = 20
SEED = 5