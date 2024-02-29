import torch
from lightning import LightningModule
from transformers import CLIPModel, CLIPProcessor


class CLIP(LightningModule):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, batch, *args, **kwargs):
        inputs = self.processor(
            text=[""],
            images=batch,
            return_tensors="pt",
            padding=True,
        )
        inputs.to(batch.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.image_embeds
