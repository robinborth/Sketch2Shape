import torch
from lightning import LightningModule
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class ResNet18(LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.fc = torch.nn.Identity()

    def forward(self, batch):
        return self.resnet18(batch)
