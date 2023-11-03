import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        embedding_size: int = 32,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 4, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5),
            nn.MaxPool2d(2, 2),
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
