import torch
from torchmetrics import Metric

from lib.models.clip import CLIP


class CLIPScore(Metric):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIP(model_name=model_name)
        self.add_state("clip_score", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, img1: torch.tensor, img2: torch.tensor):
        emb1 = self.clip(img1)
        emb2 = self.clip(img2)
        emb1 = torch.nn.functional.normalize(emb1)
        emb2 = torch.nn.functional.normalize(emb2)
        self.clip_score += ((emb1 @ emb2.T) * 100).item()
        self.total += 1

    def compute(self):
        return self.clip_score / self.total
