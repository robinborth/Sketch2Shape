import faiss
import numpy as np
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.aggregation import MeanMetric

from lib.eval.utils import get_all_vectors_from_faiss_index


class SiameseTester(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        index_mode: str = "image",  # image, sketch, all
        query_mode: str = "sketch",  # image, sketch, all
    ):
        super().__init__()
        self.model = model

        self.index = faiss.IndexFlatL2(self.model.embedding_size)
        self.normalized_index = faiss.IndexFlatIP(self.model.embedding_size)
        self._labels: list[int] = []

        self.index_mode = index_mode
        self.query_mode = query_mode

        self.l2_distance_at_1_object = MeanMetric()
        self.cosine_similarity_at_1_object = MeanMetric()
        self.recall_at_1_object = MeanMetric()
        self.recall_at_2_object = MeanMetric()
        self.recall_at_5_object = MeanMetric()
        self.recall_at_10_object = MeanMetric()
        self.recall_at_20_object = MeanMetric()
        self.recall_at_1_percent = MeanMetric()
        self.recall_at_5_percent = MeanMetric()
        self.recall_at_20_percent = MeanMetric()

    @property
    def labels(self):
        return np.array(self._labels)

    @property
    def num_views_per_object(self):
        _, counts = np.unique(self.labels, return_counts=True)
        assert len(counts) > 0
        min_count = np.min(counts)
        max_count = np.max(counts)
        assert min_count == max_count
        return min_count

    @property
    def num_unique_objects(self):
        return len(np.unique(self.labels))

    def k_for_total_percent(self, percent: float = 0.01):
        return int(self.num_unique_objects * self.num_views_per_object * percent)

    def k_for_num_objects(self, num_objects: int = 1):
        return int(num_objects * self.num_views_per_object)

    @property
    def index_vectors(self):
        return get_all_vectors_from_faiss_index(self.index)

    def normalize(self, embedding):
        l2_distance = np.linalg.norm(embedding, axis=-1)
        return embedding / l2_distance.reshape(-1, 1)

    def forward(self, batch):
        return self.model(batch).detach().cpu().numpy()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        index_emb = self.forward(batch[self.index_mode])
        self.index.add(index_emb)

        normalized_emb = self.normalize(index_emb)
        self.normalized_index.add(normalized_emb)

        self._labels.extend(batch["label"].detach().cpu().numpy())

    def test_step(self, batch, batch_idx):
        gt_labels = batch["label"].cpu().numpy().reshape(-1, 1)
        query_emb = self.forward(batch[self.query_mode])
        normalized_query_emb = self.normalize(query_emb)

        k_at_1_object = self.k_for_num_objects(num_objects=1)
        k_at_2_object = self.k_for_num_objects(num_objects=2)
        k_at_5_object = self.k_for_num_objects(num_objects=5)
        k_at_10_object = self.k_for_num_objects(num_objects=10)
        k_at_20_object = self.k_for_num_objects(num_objects=20)
        k_at_1_percent = self.k_for_total_percent(percent=0.01)
        k_at_5_percent = self.k_for_total_percent(percent=0.05)
        k_at_20_percent = self.k_for_total_percent(percent=0.20)
        k = max(k_at_20_object, k_at_20_percent)
        dist, idx = self.index.search(query_emb, k=k)
        cos_sim, _ = self.normalized_index.search(normalized_query_emb, k=k_at_1_object)

        dist_at_1_object = dist[:, :k_at_1_object]
        idx_at_1_object = idx[:, :k_at_1_object]
        idx_at_2_object = idx[:, :k_at_2_object]
        idx_at_5_object = idx[:, :k_at_5_object]
        idx_at_10_object = idx[:, :k_at_10_object]
        idx_at_20_object = idx[:, :k_at_20_object]
        idx_at_1_percent = idx[:, :k_at_1_percent]
        idx_at_5_percent = idx[:, :k_at_5_percent]
        idx_at_20_percent = idx[:, :k_at_20_percent]

        self.l2_distance_at_1_object.update(dist_at_1_object)
        self.log(
            f"l2_distance_at_1_object_k_{k_at_1_object}",
            self.l2_distance_at_1_object,
        )

        self.cosine_similarity_at_1_object.update(cos_sim)
        self.log(
            f"cosine_similarity_at_1_object_k_{k_at_1_object}",
            self.cosine_similarity_at_1_object,
        )

        labels = self.labels[idx_at_1_object]
        retrievals_matrix = labels == gt_labels
        recall = retrievals_matrix.sum(axis=1) / self.num_views_per_object
        self.recall_at_1_object.update(recall)
        self.log(f"recall_at_1_object_k_{k_at_1_object}", self.recall_at_1_object)

        labels = self.labels[idx_at_2_object]
        retrievals_matrix = labels == gt_labels
        recall = retrievals_matrix.sum(axis=1) / self.num_views_per_object
        self.recall_at_2_object.update(recall)
        self.log(f"recall_at_2_object_k_{k_at_2_object}", self.recall_at_2_object)

        labels = self.labels[idx_at_5_object]
        retrievals_matrix = labels == gt_labels
        recall = retrievals_matrix.sum(axis=1) / self.num_views_per_object
        self.recall_at_5_object.update(recall)
        self.log(f"recall_at_5_object_k_{k_at_5_object}", self.recall_at_5_object)

        labels = self.labels[idx_at_10_object]
        retrievals_matrix = labels == gt_labels
        recall = retrievals_matrix.sum(axis=1) / self.num_views_per_object
        self.recall_at_10_object.update(recall)
        self.log(f"recall_at_10_object_k_{k_at_10_object}", self.recall_at_10_object)

        labels = self.labels[idx_at_20_object]
        retrievals_matrix = labels == gt_labels
        recall = retrievals_matrix.sum(axis=1) / self.num_views_per_object
        self.recall_at_20_object.update(recall)
        self.log(f"recall_at_20_object_k_{k_at_20_object}", self.recall_at_20_object)

        labels = self.labels[idx_at_1_percent]
        retrievals_matrix = labels == gt_labels
        num_labels = min(self.num_views_per_object, k_at_1_percent)
        recall = retrievals_matrix.sum(axis=1) / num_labels
        self.recall_at_1_percent.update(recall)
        self.log(f"recall_at_1_percent_k_{k_at_1_percent}", self.recall_at_1_percent)

        labels = self.labels[idx_at_5_percent]
        retrievals_matrix = labels == gt_labels
        num_labels = min(self.num_views_per_object, k_at_5_percent)
        recall = retrievals_matrix.sum(axis=1) / num_labels
        self.recall_at_5_percent.update(recall)
        self.log(f"recall_at_5_percent_k_{k_at_5_percent}", self.recall_at_5_percent)

        labels = self.labels[idx_at_20_percent]
        retrievals_matrix = labels == gt_labels
        num_labels = min(self.num_views_per_object, k_at_20_percent)
        recall = retrievals_matrix.sum(axis=1) / num_labels
        self.recall_at_20_percent.update(recall)
        self.log(f"recall_at_20_percent_k_{k_at_20_percent}", self.recall_at_20_percent)
