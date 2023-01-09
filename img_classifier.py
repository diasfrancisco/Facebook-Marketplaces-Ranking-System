import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR

class ImageDataset(Dataset):
    """Creates a custom image dataset

    Args:
        Dataset (class): An abstract class provided by PyTorch
    """    
    def __init__(self, img_dir, transform=None):
        # Initialises the parent class, allowing us to use the methods it contains as well as any attributes
        super().__init__()
        self.img_labels = pd.read_csv("./data/LabelledImages.csv")
        self.img_dir = img_dir
        self.transform = transforms.PILToTensor()

    def __getitem__(self, idx):
        # Grabs the image at a certain index
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1] + '.jpg')
        # Converts the image into a tensor using torchvision's read_image() function
        image = Image.open(img_path)
        # Grabs the label corresponding to the index provided
        label = self.img_labels.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.img_labels)

class CNN(torch.nn.Module):
    """A convolution neural network based upon the RESNET50 CNN

    Args:
        torch.nn.Module (class): The base class for all neural networks
    """    
    def __init__(self):
        # Initialise parameters
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13, bias=True)
    
    def forward(self, features):
        # Use the layers to make a prediction
        return self.resnet50(features.float())

def train(cnn_model, epochs=20):
    min_valid_loss = np.inf
    batch_idx = 0
    # Sets the optimiser and scheduler to be used for the model
    optimiser = torch.optim.SGD(cnn_model.parameters() ,lr=0.1)
    scheduler = OneCycleLR(optimiser, max_lr=0.02, steps_per_epoch=len(train_dataloader), epochs=epochs)
    # Creates a instance of the summary writer used by TensorBoard
    writer = SummaryWriter()
    
    for epoch in range(epochs):
        print("\nCurrent Epoch:", f'{epoch+1}/{epochs}')
        
        cnn_model.train()
        
        running_train_loss = 0
        correct_train = 0
        
        for batch in train_dataloader:
            # Clears the gradients
            optimiser.zero_grad()
            # Unpacks the batch with the features and labels
            X_train, y_train = batch
            X_train, y_train = X_train.to(device), y_train.to(device)
            # Makes a prediction on the features using the CNN model
            prediction = cnn_model(X_train)
            # Calculates the loss using the cross entropy function
            loss = F.cross_entropy(prediction, y_train.long())
            # Differentiates the loss wrt the model parameters
            loss.backward()
            # Optimisation step
            optimiser.step()
            scheduler.step()
            # Writes the loss against the batch number in TensorBoard
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
            # Calculates the training loss and the accuracy at each epoch
            running_train_loss += loss.item()
            _, predicted = torch.max(prediction, dim=1)
            correct_train += torch.sum(predicted == y_train)
        
        # Calculate the training loss and accuracy of the model at each epoch    
        train_loss = running_train_loss/len(train_dataloader)
        train_acc = correct_train.float()/len(train_dataset)
        
        print('Train Loss: %.3f | Train Accuracy: %.3f'%(train_loss, train_acc*100))
        
        # Saves all models at each epoch
        torch.save(cnn_model.state_dict(), f'./model_evaluation/{timestamp}/weights/epoch{epoch}_weights.pt')
        with open(f'./model_evaluation/{timestamp}/weights/epoch{epoch}_loss_and_acc.txt', 'w') as f:
            f.writelines(['Train Loss: %.3f'%train_loss, '\nTrain Accuracy: %.3f'%train_acc])

        cnn_model.eval()
        
        running_val_loss = 0
        correct_val = 0
        
        for batch in validation_dataloader:
            # Unpacks the batch with the features and labels
            X_val, y_val = batch
            X_val, y_val = X_val.to(device), y_val.to(device)
            # Makes a prediction on the features using the CNN model
            prediction = cnn_model(X_val)
            # Calculates the loss using the cross entropy function
            loss = F.cross_entropy(prediction, y_val.long())
            running_val_loss += loss.item()
            _, predicted = torch.max(prediction, dim=1)
            correct_val += torch.sum(predicted == y_val)
        
        # Calculate the validation loss and accuracy of the model at each epoch    
        val_loss = running_val_loss/len(validation_dataloader)
        val_acc = correct_val.float()/len(validation_dataset)
        
        print('Validation Loss: %.3f | Validation Accuracy: %.3f'%(val_loss, val_acc*100))
        
        if min_valid_loss > val_loss:
            min_valid_loss = val_loss
            torch.save(cnn_model.state_dict(), f'./model_evaluation/{timestamp}/min_valid_loss_model.pt')
        
        running_test_loss = 0
        test_correct = 0
        
        for batch in test_dataloader:
            X_test, y_test = batch
            X_test, y_test = X_test.to(device), y_test.to(device)
            prediction = cnn_model(X_test)
            loss = F.cross_entropy(prediction, y_test.long())
            running_test_loss += loss.item()
            _, predicted = torch.max(prediction, dim=1)
            test_correct += torch.sum(predicted == y_test)
        
        test_loss = running_val_loss/len(test_dataloader)
        test_acc = test_correct.float()/len(test_dataset)
        
        print('Test Loss: %.3f | Test Accuracy: %.3f'%(test_loss, test_acc*100))
            
            
if __name__=="__main__":
    timestamp = datetime.now().strftime('%d_%b_%Y_%H_%M_%S_%f')
    
    # Sets the device to either cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialises an instance of the custom image dataset
    image_dataset = ImageDataset(img_dir="./data/cleaned_images/",
                                transform=True)
    # Splits the dataset into a training, validation and testing dataset
    train_dataset, validation_dataset, test_dataset = random_split(image_dataset, [0.6, 0.2, 0.2])

    if os.path.isdir('./model_evaluation'):
        os.makedirs(f'./model_evaluation/{timestamp}/weights', exist_ok=True)
    else:
        os.makedirs(f'./model_evaluation/{timestamp}/weights', exist_ok=True)

    # Loads the training, validation and testing data in shuffled "minibatches"
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    # Initialises an instance of the model on the device set
    cnn_model = CNN().to(device=device)
    
    # Release any unoccupied memory held by the caching allocator
    torch.cuda.empty_cache()
    
    train(cnn_model)
    
    # Save my model's weights and biases
    torch.save(cnn_model.state_dict(), f'./model_evaluation/{timestamp}/end_model.pt')