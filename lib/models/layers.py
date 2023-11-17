import torch
import torch.nn as nn
from torchvision.models import resnet18


class DummyDecoder(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        model_size_in_gb: int = 1,  # 1000 MB
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 1, 1),
            nn.MaxPool2d(4, 4),
            nn.Flatten(1),
        )
        # calculate the dimension of the input mlp
        x = torch.randn(1, 3, image_size, image_size)
        flatten_dim = self.cnn(x).shape[-1]

        bottle_dim = 1000

        self.up = nn.Sequential(nn.Linear(flatten_dim, 1), nn.Linear(1, bottle_dim))

        layers = []
        for _ in range(model_size_in_gb):
            layers.append(nn.Linear(bottle_dim, bottle_dim))
        self.mlp = nn.Sequential(*layers)

        self.down = nn.Linear(bottle_dim, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.up(x)
        x = self.mlp(x)
        x = self.down(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 1, 1),
            nn.Flatten(1),
        )
        # calculate the dimension of the input mlp
        x = torch.randn(1, 3, image_size, image_size)
        flatten_dim = self.cnn(x).shape[-1]

        self.mlp = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, embedding_size),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, embedding_size: int = 64):
        super().__init__()
        self.resnet18 = resnet18()
        self.resnet18.fc = torch.nn.Linear(in_features=512, out_features=embedding_size)

    def forward(self, batch):
        return self.resnet18(batch)
