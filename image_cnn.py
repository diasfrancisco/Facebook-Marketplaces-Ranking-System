import torch


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
    
class FtCNN(torch.nn.Module):
    """A convolution neural network based upon the RESNET50 CNN

    Args:
        torch.nn.Module (class): The base class for all neural networks
    """    
    def __init__(self):
        # Initialise parameters
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)

        for counter, child in enumerate(self.resnet50.children()):
            if counter < 6:
                for param in child.parameters():
                    param.requires_grad = False # Freeze layers 1-5
    
    def forward(self, features):
        # Use the layers to make a prediction
        return self.resnet50(features.float())