import torch.nn as nn


class CLIPScoreMetric(nn.Module):
    def forward(self):
        raise NotImplementedError()
