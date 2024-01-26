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
from lib.models.siamese import Siamese
from lib.models.siamese_eval import EvalCLIP, EvalResNet18
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

        self.cosine_similarity = MeanMetric()
        self.recall_at_1_object = MeanMetric()
        self.recall_at_5_object = MeanMetric()
        self.recall_at_1_percent = MeanMetric()
        self.heatmap = CatMetric()

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
        return np.concatenate(self._index)

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
        _, _, n, _type = recall_type.split("_")
        k = self.k_for_num_objects(num_objects=int(n))
        if _type == "percent":
            percent = int(n) * 0.01
            k = self.k_for_total_percent(percent=percent)
        labels = self.labels[idx[:, :k]]
        retrievals_matrix = labels == gt_labels
        return retrievals_matrix.sum(axis=1) / self.num_views_per_object

    def search(self, query_emb: np.ndarray, k: int = 1):
        similarity = query_emb @ self.index.T
        idx = np.argsort(similarity)[:, ::-1][:, :k]
        return similarity.take(idx), idx

    def test_step(self, batch, batch_idx):
        # extract from the batch
        gt_labels = batch["label"].cpu().numpy().reshape(-1, 1)

        # get the similarity
        query_emb = self.forward(batch[self.query_mode])
        k_at_1_object = self.k_for_num_objects(num_objects=1)
        self.log("k_at_1_object", k_at_1_object)
        similarity, idx = self.search(query_emb, k=self.max_k)

        # calculate the metrics
        for metric_name in [
            "recall_at_1_object",
            "recall_at_5_object",
            "recall_at_1_percent",
        ]:
            metric = getattr(self, metric_name)
            recall = self.calculate_recall(metric_name, idx, gt_labels)
            metric.update(recall)
            self.log(metric_name, metric)

        self.cosine_similarity.update(similarity[:, :k_at_1_object])
        self.log("cosine_similarity", self.cosine_similarity)

        # get the top labels and image_ids
        # gt_image_ids = batch["image_id"].cpu().numpy()
        # labels = self.labels[idx[:, :k_at_1_object]]
        # image_ids = self.image_ids[idx[:, :k_at_1_object]]
        # recall_at_1_object=self.calculate_recall("recall_at_1_object", idx, gt_labels)

        # TODO we want to have the recall per view
        # heatmap = np.take_along_axis(
        #     arr=recall_at_1_object,
        #     indices=np.argsort(image_ids),
        #     axis=0,
        # )
        # self.heatmap.update(heatmap)

        # if batch_idx % 8 == 0:
        #     image_id = 21
        #     if batch_idx % 16 == 0:
        #         image_id = random.randint(0, 31)
        #     idx = torch.where(gt_image_ids == image_id)[0][0]

        #     gt_label = batch["label"][idx].item()
        #     gt_image_id = batch["image_id"][idx].item()
        #     labels_at_1_object = out["labels_at_1_object"][idx]
        #     image_ids_at_1_object = out["image_ids_at_1_object"][idx]

        #     index_images = {}
        #     obj_id = self.metainfo.label_to_obj_id(gt_label)
        #     sketch = self.metainfo.load_sketch(obj_id, image_id=f"{gt_image_id:05}")
        #     plot_single_image(sketch)
        #     index_images["sketch"] = wandb.Image(plt)

        #     images = []
        #     for label, image_id in zip(labels_at_1_object, image_ids_at_1_object):
        #         is_false = gt_label != label
        #         obj_id = metainfo.label_to_obj_id(label)
        #         image_id = str(image_id).zfill(5)
        #         image = dataset._fetch("images", obj_id, image_id)
        #         if is_false:
        #             image[image > 0.95] = 0
        #         images.append(image)
        #     image_data = transform_to_plot(images, batch=True)
        #     image_grid(image_data, 4, 8)
        #     index_images["rendered_images"] = wandb.Image(plt)
        #     self.logger.log_metrics(index_images)  # type: ignore

    # def on_test_end(self) -> None:
    #     heatmap = self.heatmap.compute()
    #     heatmap = heatmap.view(-1, 1, 32).mean(dim=0).detach().cpu().numpy()
    #     plt.clf()
    #     c = plt.imshow(heatmap, cmap="viridis", interpolation="nearest")
    #     plt.colorbar(c)
    #     self.logger.log_metrics({"heatmap": wandb.Image(c)})  # type: ignore
