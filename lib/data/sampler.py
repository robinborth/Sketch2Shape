import numpy as np
from torch.utils.data.sampler import Sampler


class ChunkSampler(Sampler):
    """
    At every iteration, this will return chunk_size samples per label class. But each
    label is only fetched once through one epoch. For example, if chunk_size = 2 and
    there are 100 unique labels, and 50 labels per unique label,
    hence len(labels) == 100 * 50, we have still 200 labels per epoch.

    When sample_steps=-1 we sample the full labels.
    """

    def __init__(
        self,
        labels: np.ndarray,
        chunk_size: int = 2,
        sample_steps: int = 1,
    ):
        self.labels = np.unique(labels, return_inverse=True)[1]
        self.chunk_size = chunk_size
        self.unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.num_images = counts[0]
        # ensure that each label idx is only retrieved once
        if sample_steps == -1:  # full samples
            sample_steps = self.num_images // self.chunk_size
        self.sample_steps = min(sample_steps, self.num_images // self.chunk_size)
        # check that the labels are correct and sorted
        assert np.all(counts == self.num_images)
        assert np.all(self.labels[: self.num_images] == 0)

    def __len__(self):
        return self.chunk_size * len(self.unique_labels) * self.sample_steps

    def __iter__(self):
        num_images, chunk_size = self.num_images, self.chunk_size
        idx_list = []
        for step in range(self.sample_steps):
            labels = self.unique_labels.copy()
            np.random.shuffle(labels)
            idxs = []
            for _ in range(self.num_unique_labels):
                _idxs = np.random.choice(num_images, size=num_images, replace=False)
                idxs.append(_idxs)
            iter_idxs = np.stack(idxs)[
                :, step * chunk_size : (step + 1) * chunk_size
            ].flatten()
            iter_labels = np.stack([labels] * chunk_size).T.flatten()
            assert len(iter_idxs) == len(iter_labels)
            _idx_list = (iter_labels * num_images) + iter_idxs
            idx_list.append(_idx_list)
        idx_list = np.concatenate(idx_list)
        return iter(idx_list)
