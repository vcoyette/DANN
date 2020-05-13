"""Define models for office domain."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FeatureExtractor(nn.Module):
    """Feature Extractor module."""

    def __init__(self):
        """Initialize Feature extractor for Office."""
        super(FeatureExtractor, self).__init__()

        # Load pretrained alexnet
        alexnet = torchvision.models.alexnet(pretrained=True)

        # Remove last fc layer from alexnet
        self.conv = alexnet.features
        self.avgpool = alexnet.avgpool
        self.linear = alexnet.classifier[:5]

        # Add a fc bottleneck layer
        self.bottleneck = nn.Linear(4096, 256)

    def forward(self, x):
        """Forward pass x in the feature extractor."""
        x = self.conv(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.linear(x)
        x = F.relu(self.bottleneck(x))

        return x


class Classifier(nn.Module):
    """Classifier on the image classes."""

    def __init__(self):
        """Initialize the classifier."""
        nn.Module.__init__(self)

        self.fc = nn.Linear(256, 31)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Forward pass x and return probability of each class."""
        x = self.logsoftmax(self.fc(x))

        return x


class DomainRegressor(nn.Module):
    """Domain Regressor between source and domain."""

    def __init__(self):
        """Initialize DomainRegressor."""
        super(DomainRegressor, self).__init__()

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Forward pass X and return probabilities of source and domain."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))
        return x
