import torch


class Coarse2FineScheduler:
    """
    Assumes no downsampling / coarse rays by default. Every milestone that is added
    means that all epochs before the milestone will be downsampled by a factor of two.
    """

    def __init__(
        self,
        resolution: int = 256,
        max_epochs: int = 1,
        milestones: list[int] = [],
    ):
        self.reducer = {
            "max": torch.nn.MaxPool2d(2),
            "avg": torch.nn.MaxPool2d(2),
        }
        self.current_epoch = 0
        self.milestones = milestones
        self._resolution = resolution

    def update(self, current_epoch):
        self.current_epoch = current_epoch

    @property
    def resolution(self):
        return self._resolution // (2 ** (self.num_downsample(self.current_epoch)))

    def num_downsample(self, current_epoch: int = 0):
        return sum(m >= current_epoch for m in self.milestones)

    def downsample(
        self,
        x: torch.Tensor,
        reducer: str = "max",
        current_epoch: int | None = None,
    ) -> torch.Tensor:
        # skip if after the final milestone
        current_epoch = current_epoch or self.current_epoch
        if not (num_downsample := self.num_downsample(current_epoch)):
            return x
        # apply the downsampling
        channel = x.shape[-1]
        x = x.reshape(self._resolution, self._resolution, -1).transpose(0, 2)
        for _ in range(num_downsample):
            x = self.reducer[reducer](x)
        return x.transpose(0, 2).reshape(-1, channel)

    def downsample_mask(self, x: torch.Tensor, current_epoch: int | None = None):
        x = x.to(torch.float32).unsqueeze(2)
        x = self.downsample(x, current_epoch=current_epoch, reducer="max")
        return x.to(torch.bool).squeeze()
