import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class CustomNetwork(nn.Module):
    def __init__(self, num_classes: int, internal_features: int = 512):
        super(CustomNetwork, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, internal_features)

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(internal_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)  
        x = self.relu(x) 
        x = self.classifier(x)  
        return x


class CustomNetworkMetric(nn.Module):
    def __init__(self, num_classes: int, internal_features: int = 512):
        super(CustomNetworkMetric, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, internal_features)

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(internal_features, num_classes)

    def forward(self, x):
        l = self.resnet(x) 
        x = self.relu(l) 
        x = self.classifier(x) 
        return x, l