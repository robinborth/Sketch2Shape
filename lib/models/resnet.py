import torch
from lightning import LightningModule
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class ResNet18(LightningModule):
    def __init__(
        self,
        head: str = "none",  # none, linear, mlp
        embedding_size: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = torch.nn.Identity()

        if head == "none":
            self.head = torch.nn.Sequential(torch.nn.Identity())
        elif head == "linear":
            self.head = torch.nn.Sequential(torch.nn.Linear(512, embedding_size))
        elif head == "mlp":
            self.head = torch.nn.Sequential(
                torch.nn.Linear(512, embedding_size),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_size, embedding_size),
            )
        else:
            raise NotImplementedError()

    def forward(self, images, *args, **kwargs):
        """Embbedd the images with the resnet model.

        Args:
            images (torch.Tensor): The images are of dim (B, C, W, H)

        Returns:
            torch.Tensor: The embeddings of dim (B, D)
        """
        x = self.backbone(images)
        return self.head(x)
