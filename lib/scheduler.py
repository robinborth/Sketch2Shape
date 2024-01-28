from typing import List

import numpy as np

"""Schedulers for various reasons used"""


class Coarse2FineScheduler:
    """
    Assumes no downsampling / coarse rays by default.
    Every milestone that is added means that all epochs
    before the milestone will be downsampled by a factor of two.
    """

    def __init__(self, milestones: List[int], max_epochs: int):
        self.downsample_list = np.zeros(max_epochs, dtype=int)
        for milestone in milestones:
            self.downsample_list[:milestone] += 1

    def __getitem__(self, epoch):
        return self.downsample_list[epoch]
