"""Define models for SVHN adaptation."""

import torch.nn as nn
import torch.nn.functional as F


class DomainRegressor(nn.Module):
    """Domain Regressor between source and domain."""

    def __init__(self):
        """Initialize DomainRegressor."""
        super(DomainRegressor, self).__init__()

        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Forward pass X and return probabilities of source and domain."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))
        return x


class Classifier(nn.Module):
    """Classifier on the images classes."""

    def __init__(self):
        """Initialize classifier."""
        super(Classifier, self).__init__()

        # Fully-connected
        self.fc1 = nn.Linear(128 * 2 * 2, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 10)

        # Activation
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Forward pass X and return probability of each class."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))

        return x


class FeatureExtractor(nn.Module):
    """Feature Extractor from images to the representational space."""

    def __init__(self):
        """Initialize the feature extractor."""
        super(FeatureExtractor, self).__init__()

        # Conv
        self.conv1 = nn.Conv2d(3, 64, 5, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        """Forward pass X and return its representation."""
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 128 * 2 * 2)

        return x
