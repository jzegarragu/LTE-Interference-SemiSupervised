"""
Deep Learning Architectures Definitions
---------------------------------------
This module defines the neural network architectures used for LTE interference classification.
It includes the custom baseline CNN and ResNet backbones (ResNet18, ResNet50) 
explicitly adapted for single-channel (grayscale) spectrogram inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNNPaper(nn.Module):
    """
    Custom Baseline CNN Architecture.
    
    Structure tailored for 100x100 input spectrograms based on the paper's proposal.
    Layers: 3 Convolutional blocks -> Flatten -> Dense Layers.
    """
    def __init__(self, num_classes):
        super().__init__()
        # Block 1: Input (1, 100, 100) -> Output (32, 50, 50)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        # Block 2: Output (64, 25, 25)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        # Block 3: Output (128, 12, 12)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        # Flatten calculation: 128 channels * 12 * 12 spatial dimensions
        flatten_dim = 128 * 12 * 12
        
        self.fc1 = nn.Linear(flatten_dim, 256)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

def build_resnet50_1ch(num_classes, pretrained=True):
    """
    Constructs a ResNet50 model adapted for 1-channel input (grayscale).
    
    Adaptation strategy:
    - Replaces the first Conv2d layer (3 channels) with a 1-channel layer.
    - If pretrained: Initializes the new weights by averaging the original RGB weights.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet50(weights=weights)
    
    # Adapt the first convolutional layer for 1 channel
    w_old = m.conv1.weight.data
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        if pretrained:
            # Average the weights of the 3 channels (RGB) to initialize the single channel
            m.conv1.weight[:] = w_old.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
            
    # Adapt the final classification layer
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_resnet18_1ch(num_classes, pretrained=True):
    """
    Constructs a ResNet18 model adapted for 1-channel input (grayscale).
    
    Adaptation strategy:
    - Replaces the first Conv2d layer (3 channels) with a 1-channel layer.
    - If pretrained: Initializes the new weights by averaging the original RGB weights.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    
    # Adapt the first convolutional layer for 1 channel
    w_old = m.conv1.weight.data
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        if pretrained:
            # Average the weights of the 3 channels (RGB)
            m.conv1.weight[:] = w_old.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
            
    # Adapt the final classification layer
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_model(model_name, num_classes, resnet_use_pretrain=True):
    """
    Factory to create the specified model architecture.
    """
    model_name = model_name.lower().strip()
    
    if model_name in ("cnn_paper", "cnn_paper_l2"):
        return CNNPaper(num_classes)
    elif model_name == "resnet50":
        return build_resnet50_1ch(num_classes, pretrained=resnet_use_pretrain)
    elif model_name == "resnet18":
        return build_resnet18_1ch(num_classes, pretrained=resnet_use_pretrain)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

