import random
from typing import Any

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from lightning import LightningModule
from torchmetrics.aggregation import CatMetric, MeanMetric

from lib.eval.utils import (
    get_all_vectors_from_faiss_index,
    image_grid,
    plot_single_image,
    transform_to_plot,
)


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
        self._image_ids: list[int] = []

        self.index_mode = index_mode
        self.query_mode = query_mode

        # TODO clean that up
        self.l2_distance_at_1_object = MeanMetric()
        self.cosine_similarity_at_1_object = MeanMetric()
        self.recall_at_1_object = MeanMetric()
        self.recall_at_2_object = MeanMetric()
        self.recall_at_5_object = MeanMetric()
        self.recall_at_10_object = MeanMetric()
        self.recall_at_20_object = MeanMetric()
        self.recall_at_1_percent = MeanMetric()
        self.recall_at_5_percent = MeanMetric()

        self.recall_heatmap = CatMetric()

    @property
    def labels(self):
        return np.array(self._labels)

    @property
    def image_ids(self):
        return np.array(self._image_ids)

    @property
    def images(self):
        return np.stack(self._images)

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
        self._image_ids.extend(batch["image_id"].detach().cpu().numpy())

    def _calc_recall(self, k, idx, gt_labels):
        labels = self.labels[idx[:, :k]]
        retrievals_matrix = labels == gt_labels
        return retrievals_matrix.sum(axis=1) / self.num_views_per_object

    def predict_step(self, batch, batch_idx) -> Any:
        out = {}

        gt_labels = batch["label"].cpu().numpy().reshape(-1, 1)
        query_emb = self.forward(batch[self.query_mode])
        normalized_query_emb = self.normalize(query_emb)

        k = max(
            self.k_for_total_percent(percent=0.05),
            self.k_for_num_objects(num_objects=20),
        )
        dist, idx = self.index.search(query_emb, k=k)
        cos_sim, cos_idx = self.normalized_index.search(normalized_query_emb, k=k)

        k_at_1_object = self.k_for_num_objects(num_objects=1)
        out["recall_at_1_object"] = self._calc_recall(k_at_1_object, idx, gt_labels)
        out["k_at_1_object"] = k_at_1_object

        k_at_2_object = self.k_for_num_objects(num_objects=2)
        out["recall_at_2_object"] = self._calc_recall(k_at_2_object, idx, gt_labels)
        out["k_at_2_object"] = k_at_2_object

        k_at_5_object = self.k_for_num_objects(num_objects=5)
        out["recall_at_5_object"] = self._calc_recall(k_at_5_object, idx, gt_labels)
        out["k_at_5_object"] = k_at_5_object

        k_at_10_object = self.k_for_num_objects(num_objects=10)
        out["recall_at_10_object"] = self._calc_recall(k_at_10_object, idx, gt_labels)
        out["k_at_10_object"] = k_at_10_object

        k_at_20_object = self.k_for_num_objects(num_objects=20)
        out["recall_at_20_object"] = self._calc_recall(k_at_20_object, idx, gt_labels)
        out["k_at_20_object"] = k_at_20_object

        k_at_1_percent = self.k_for_total_percent(percent=0.01)
        out["recall_at_1_percent"] = self._calc_recall(k_at_1_percent, idx, gt_labels)
        out["k_at_1_percent"] = k_at_1_percent

        k_at_5_percent = self.k_for_total_percent(percent=0.05)
        out["recall_at_5_percent"] = self._calc_recall(k_at_5_percent, idx, gt_labels)
        out["k_at_5_percent"] = k_at_5_percent

        out["l2_distance_at_1_object"] = dist[:, :k_at_1_object]
        out["cosine_similarity_at_1_object"] = cos_sim[:, :k_at_1_object]

        out["labels_at_1_object"] = self.labels[idx[:, :k_at_1_object]]
        out["image_ids_at_1_object"] = self.image_ids[idx[:, :k_at_1_object]]
        out["labels"] = batch["label"].cpu().numpy()
        out["image_ids"] = batch["image_id"].cpu().numpy()

        sorted_indices = np.argsort(out["image_ids"])
        out["recall_heatmap"] = np.take_along_axis(
            out["recall_at_1_object"], sorted_indices, axis=0
        )

        return out

    def test_step(self, batch, batch_idx):
        out = self.predict_step(batch, batch_idx=batch_idx)
        self.l2_distance_at_1_object.update(out["l2_distance_at_1_object"])
        self.log(
            f"l2_distance_at_1_object_k_{out['k_at_1_object']}",
            self.l2_distance_at_1_object,
        )

        self.cosine_similarity_at_1_object.update(out["cosine_similarity_at_1_object"])
        self.log(
            f"cosine_similarity_at_1_object_k_{out['k_at_1_object']}",
            self.cosine_similarity_at_1_object,
        )

        self.recall_at_1_object.update(out["recall_at_1_object"])
        self.log(
            f"recall_at_1_object_k_{out['k_at_1_object']}", self.recall_at_1_object
        )

        self.recall_at_2_object.update(out["recall_at_2_object"])
        self.log(
            f"recall_at_2_object_k_{out['k_at_2_object']}", self.recall_at_2_object
        )

        self.recall_at_5_object.update(out["recall_at_5_object"])
        self.log(
            f"recall_at_5_object_k_{out['k_at_5_object']}", self.recall_at_5_object
        )

        self.recall_at_10_object.update(out["recall_at_10_object"])
        self.log(
            f"recall_at_10_object_k_{out['k_at_10_object']}", self.recall_at_10_object
        )

        self.recall_at_20_object.update(out["recall_at_20_object"])
        self.log(
            f"recall_at_20_object_k_{out['k_at_20_object']}", self.recall_at_20_object
        )

        self.recall_at_1_percent.update(out["recall_at_1_percent"])
        self.log(
            f"recall_at_1_percent_k_{out['k_at_1_percent']}", self.recall_at_1_percent
        )

        self.recall_at_5_percent.update(out["recall_at_5_percent"])
        self.log(
            f"recall_at_5_percent_k_{out['k_at_5_percent']}", self.recall_at_5_percent
        )

        self.recall_heatmap.update(out["recall_heatmap"])

        if batch_idx % 8 == 0:
            metainfo = self.model.metainfo
            dataset = self.trainer.test_dataloaders.dataset
            image_id = 21
            if batch_idx % 16 == 0:
                image_id = random.randint(0, 31)
            idx = torch.where(batch["image_id"] == image_id)[0][0]

            gt_label = batch["label"][idx].item()
            gt_image_id = batch["image_id"][idx].item()
            labels_at_1_object = out["labels_at_1_object"][idx]
            image_ids_at_1_object = out["image_ids_at_1_object"][idx]

            index_images = {}
            obj_id = metainfo.label_to_obj_id(gt_label)
            image_id = str(gt_image_id).zfill(5)
            sketch = dataset._fetch("sketches", obj_id, image_id)
            plt.clf()
            plot_single_image(sketch)
            index_images["sketch"] = wandb.Image(plt)

            images = []
            for label, image_id in zip(labels_at_1_object, image_ids_at_1_object):
                is_false = gt_label != label
                obj_id = metainfo.label_to_obj_id(label)
                image_id = str(image_id).zfill(5)
                image = dataset._fetch("images", obj_id, image_id)
                if is_false:
                    image[image > 0.95] = 0
                images.append(image)
            image_data = transform_to_plot(images, batch=True)
            plt.clf()
            image_grid(image_data, 4, 8)
            index_images["rendered_images"] = wandb.Image(plt)
            self.logger.log_metrics(index_images)  # type: ignore

    def on_test_end(self) -> None:
        recall_heatmap = self.recall_heatmap.compute()
        recall_heatmap = recall_heatmap.view(-1, 1, 32).mean(dim=0)
        recall_heatmap = recall_heatmap.detach().cpu().numpy()
        plt.clf()
        c = plt.imshow(recall_heatmap, cmap="viridis", interpolation="nearest")
        plt.colorbar(c)
        image = wandb.Image(c)
        self.logger.log_metrics({"recall_heatmap": image})  # type: ignore
