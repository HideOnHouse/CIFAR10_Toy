import torch
import torch.nn as nn
from torch import flatten


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(3072, 768),
            nn.CELU(),
            nn.Linear(768, 192),
            nn.CELU(),
            nn.Linear(192, 48),
            nn.CELU(),
            nn.Linear(48, 10)
        )
        self.classifier = nn.Sequential()

    def forward(self, x):
        x = flatten(x, 1)
        x = self.features(x)
        out = self.classifier(x)
        return out


class RBF(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))

        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, x):
        size = (x.shape[0], self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        out = torch.exp(-1 * distances)
        return out


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 18, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(18, 48, 6),
            nn.Tanh(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 84),
            nn.Tanh(),
            RBF(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        out = self.classifier(x)
        return out


# CNN adjusted due to image resolution difference
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LocalResponseNorm(3),
            nn.Conv2d(12, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LocalResponseNorm(3),
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 32, 1, 1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        out = self.classifier(x)
        return out


# CNN adjusted due to image resolution difference
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        out = self.classifier(x)
        return out
