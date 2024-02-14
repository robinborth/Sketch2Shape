import torch
from lightning import LightningModule
from transformers import CLIPModel, CLIPProcessor


class CLIP(LightningModule):
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
