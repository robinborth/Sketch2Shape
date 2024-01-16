import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from lightning import LightningModule
from torchmetrics.aggregation import CatMetric, MeanMetric

from lib.data.metainfo import MetaInfo
from lib.models.decoder import EvalCLIP, EvalResNet18
from lib.models.siamese import Siamese
from lib.visualize.image import image_grid, plot_single_image, transform_to_plot


class SiameseTester(LightningModule):
    def __init__(
        self,
        ckpt_path: str = ".ckpt",
        data_dir: str = "/data",
        index_mode: str = "image",  # image, sketch, all
        query_mode: str = "sketch",  # image, sketch, all
    ):
        super().__init__()

        self.model = self.load_model(ckpt_path=ckpt_path)
        self.metainfo = MetaInfo(data_dir=data_dir)

        self._index: list = []
        self._labels: list[int] = []
        self._image_ids: list[int] = []

        self.index_mode = index_mode
        self.query_mode = query_mode

        self.cosine_similarity_at_1_object = MeanMetric()
        self.recall_at_5_object = MeanMetric()
        self.recall_at_1_percent = MeanMetric()
        self.heatmap_at_1_object = CatMetric()

    def load_model(self, ckpt_path: str) -> nn.Module:
        if ckpt_path == "resnet18":
            return EvalResNet18()
        if ckpt_path == "clip":
            return EvalCLIP()
        return Siamese.load_from_checkpoint(ckpt_path).decoder

    @property
    def max_k(self):
        return max(
            self.k_for_total_percent(percent=0.01),
            self.k_for_num_objects(num_objects=5),
        )

    @property
    def index(self):
        return np.stack(self._index)

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

    def normalize(self, embedding):
        l2_distance = np.linalg.norm(embedding, axis=-1)
        return embedding / l2_distance.reshape(-1, 1)

    def forward(self, batch):
        output = self.model(batch).detach().cpu().numpy()
        return self.normalize(output)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        index_emb = self.forward(batch[self.index_mode])
        self._index.append(index_emb)
        self._labels.extend(batch["label"].detach().cpu().numpy())
        self._image_ids.extend(batch["image_id"].detach().cpu().numpy())

    def calculate_recall(self, recall_type, idx, gt_labels):
        out = {}
        _, _, n, _type = recall_type.split("_")
        k = self.k_for_total_percent(percent=n)
        if _type == "object":
            k = self.k_for_num_objects(num_objects=n)
        labels = self.labels[idx[:, :k]]
        retrievals_matrix = labels == gt_labels
        out[recall_type] = retrievals_matrix.sum(axis=1) / self.num_views_per_object
        out[f"recall_at_{n}_{_type}"] = k
        return out

    def search(self, query_emb: np.ndarray, k: int = 1):
        similarity = query_emb @ self.index.T
        idx = np.argsort(similarity)[:, :k]
        return similarity.take(idx), idx

    def predict_step(self, batch, batch_idx) -> Any:
        # get the similarity
        gt_labels = batch["label"].cpu().numpy().reshape(-1, 1)
        query_emb = self.forward(batch[self.query_mode])
        k_at_1_object = self.k_for_num_objects(num_objects=1)
        similarity, idx = self.search(query_emb, k=self.max_k)
        # get the output
        out = {}
        out.update(self.calculate_recall("recall_at_1_object", idx, gt_labels))
        out.update(self.calculate_recall("recall_at_5_object", idx, gt_labels))
        out.update(self.calculate_recall("recall_at_1_percent", idx, gt_labels))
        out["cosine_similarity_at_1_object"] = similarity[:, :k_at_1_object]
        out["labels_at_1_object"] = self.labels[idx[:, :k_at_1_object]]
        out["image_ids_at_1_object"] = self.image_ids[idx[:, :k_at_1_object]]
        out["labels"] = batch["label"].cpu().numpy()
        out["image_ids"] = batch["image_id"].cpu().numpy()
        out["heatmap_at_1_object"] = np.take_along_axis(
            arr=out["recall_at_1_object"],
            indices=np.argsort(out["image_ids"]),
            axis=0,
        )
        return out

    def test_step(self, batch, batch_idx):
        out = self.predict_step(batch, batch_idx=batch_idx)

        self.recall_heatmap.update(out["recall_heatmap"])
        for metric_name in [
            "cosine_similarity_at_1_object",
            "recall_at_1_object",
            "recall_at_5_object",
            "recall_at_1_percent",
        ]:
            _, _, n, _type = metric_name.split("_")
            metric = getattr(self, metric_name)
            metric.update(out[metric_name])
            k = out[f"k_at_{n}_{_type}"]
            self.log(f"{metric_name}_k_{k}")

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
