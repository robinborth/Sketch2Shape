import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from transformers import CLIPModel, CLIPProcessor


class EvalResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.fc = torch.nn.Identity()

    def forward(self, batch):
        return self.resnet18(batch)


class EvalCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, batch):
        inputs = self.processor(
            text=[""],
            images=batch * 255,
            return_tensors="pt",
            padding=True,
        )
        inputs.to(batch.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.image_embeds
