import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR

import config
from dataset import ImageDataset
from image_cnn import CNN

def train(
    cnn_model,
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    train_dataset,
    validation_dataset,
    test_dataset,
    optimiser,
    scheduler,
    writer,
    min_valid_loss,
    batch_idx,
    epoch
):
    running_train_loss = 0
    correct_train = 0
    
    for batch in train_dataloader:
        # Clears the gradients
        optimiser.zero_grad()
        # Unpacks the batch with the features and labels
        X_train, y_train = batch
        X_train, y_train = X_train.to(config.DEVICE), y_train.to(config.DEVICE)
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
    torch.save(cnn_model.state_dict(), f'./model_evaluation/{config.TIMESTAMP}/weights/epoch{epoch}_weights.pt')
    with open(f'./model_evaluation/{config.TIMESTAMP}/weights/epoch{epoch}_loss_and_acc.txt', 'w') as f:
        f.writelines(['Train Loss: %.3f'%train_loss, '\nTrain Accuracy: %.3f'%train_acc])

    cnn_model.eval()
    
    running_val_loss = 0
    correct_val = 0
    
    for batch in validation_dataloader:
        # Unpacks the batch with the features and labels
        X_val, y_val = batch
        X_val, y_val = X_val.to(config.DEVICE), y_val.to(config.DEVICE)
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
        torch.save(cnn_model.state_dict(), f'./model_evaluation/{config.TIMESTAMP}/min_valid_loss_model.pt')
    
    running_test_loss = 0
    test_correct = 0
    
    for batch in test_dataloader:
        X_test, y_test = batch
        X_test, y_test = X_test.to(config.DEVICE), y_test.to(config.DEVICE)
        prediction = cnn_model(X_test)
        loss = F.cross_entropy(prediction, y_test.long())
        running_test_loss += loss.item()
        _, predicted = torch.max(prediction, dim=1)
        test_correct += torch.sum(predicted == y_test)
    
    test_loss = running_val_loss/len(test_dataloader)
    test_acc = test_correct.float()/len(test_dataset)
    
    print('Test Loss: %.3f | Test Accuracy: %.3f'%(test_loss, test_acc*100))
    
    cnn_model.train()

def main():
    # Initialises an instance of the custom image dataset
    image_dataset = ImageDataset(img_dir="./data/cleaned_images/", transform=True)
    
    # Splits the dataset into a training, validation and testing dataset
    train_dataset, validation_dataset, test_dataset = random_split(image_dataset, [0.6, 0.2, 0.2])

    # Loads the training, validation and testing data in shuffled "minibatches"
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    # Initialises an instance of the model on the device set
    cnn_model = CNN().to(device=config.DEVICE)
    
    # Release any unoccupied memory held by the caching allocator
    torch.cuda.empty_cache()
    
    min_valid_loss = np.inf
    batch_idx = 0
    # Sets the optimiser and scheduler to be used for the model
    optimiser = torch.optim.SGD(cnn_model.parameters() ,lr=0.1)
    scheduler = OneCycleLR(optimiser, max_lr=0.02, steps_per_epoch=len(train_dataloader), epochs=config.EPOCHS)
    
    # Creates a instance of the summary writer used by TensorBoard
    writer = SummaryWriter()
    
    for epoch in range(config.EPOCHS):
        print("\nCurrent Epoch:", f'{epoch+1}/{config.EPOCHS}')
        
        train(cnn_model,
              train_dataloader,
              validation_dataloader,
              test_dataloader,
              train_dataset,
              validation_dataset,
              test_dataset,
              optimiser,
              scheduler,
              writer,
              min_valid_loss,
              batch_idx,
              epoch
        )
    
    # Save my model's weights and biases
    torch.save(cnn_model.state_dict(), f'./model_evaluation/{config.TIMESTAMP}/end_model.pt')

            
if __name__=="__main__":
    main()