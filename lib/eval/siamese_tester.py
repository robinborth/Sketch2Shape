import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import wandb
from lightning import LightningModule
from torchmetrics.aggregation import CatMetric, MeanMetric
from torchvision.transforms import ToTensor

from lib.data.metainfo import MetaInfo
from lib.models.siamese import Siamese
from lib.models.siamese_eval import EvalCLIP, EvalResNet18
from lib.visualize.image import image_grid, plot_single_image, transform_to_plot


class SiameseTester(LightningModule):
    def __init__(
        self,
        ckpt_path: str = ".ckpt",
        data_dir: str = "/data",
        index_mode: str = "normal",  # normal, sketch
        query_mode: str = "sketch",  # normal, sketch
        obj_capture_image_id: int = 11,  # (azims=40, elev=-30)
        obj_capture_rate: int = 16,
    ):
        super().__init__()

        self.model = self.load_model(ckpt_path=ckpt_path)
        self.metainfo = MetaInfo(data_dir=data_dir)

        self._index: list = []
        self._labels: list[int] = []
        self._image_ids: list[int] = []

        self.index_mode = index_mode
        self.index_image_type = self.metainfo.image_type_2_type_idx[self.index_mode]
        self.query_mode = query_mode
        self.query_image_type = self.metainfo.image_type_2_type_idx[self.query_mode]
        self.obj_capture_rate = obj_capture_rate
        self.obj_capture_image_id = obj_capture_image_id

        self.cosine_similarity = MeanMetric()
        self.recall_at_1_object = MeanMetric()
        self.recall_at_5_object = MeanMetric()
        self.recall_at_1_percent = MeanMetric()
        self.heatmap = CatMetric()

        self.transform = ToTensor()

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
        index_mask = batch["image_type"] == self.index_image_type
        index_emb = self.forward(batch["image"][index_mask])
        self._index.append(index_emb)
        self._labels.extend(batch["label"][index_mask].detach().cpu().numpy())
        self._image_ids.extend(batch["image_id"][index_mask].detach().cpu().numpy())

    def calculate_recall(self, recall_type, idx, gt_labels):
        _, _, n, _type = recall_type.split("_")
        k = self.k_for_num_objects(num_objects=int(n))
        if _type == "percent":
            percent = int(n) * 0.01
            k = self.k_for_total_percent(percent=percent)
        labels = self.labels[idx[:, :k]]
        retrievals_matrix = labels == gt_labels.reshape(-1, 1)
        return retrievals_matrix.sum(axis=1) / self.num_views_per_object

    def search(self, query_emb: np.ndarray, k: int = 1):
        similarity = query_emb @ self.index.T
        idx = np.argsort(similarity)[:, ::-1][:, :k]
        return similarity.take(idx), idx

    def test_step(self, batch, batch_idx):
        # extract from the batch
        query_mask = batch["image_type"] == self.query_image_type
        gt_labels = batch["label"][query_mask].cpu().numpy()
        gt_image_ids = batch["image_id"][query_mask].cpu().numpy()

        # get the similarity
        query_emb = self.forward(batch["image"][query_mask])
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

        # get the heatmap based on the image_ids
        labels_at_1_object = self.labels[idx[:, :k_at_1_object]]
        image_ids_at_1_object = self.image_ids[idx[:, :k_at_1_object]]
        recall_at_1_object = self.calculate_recall("recall_at_1_object", idx, gt_labels)
        heatmap = np.take_along_axis(
            arr=recall_at_1_object[..., None],
            indices=np.argsort(image_ids_at_1_object),
            axis=0,
        )
        self.heatmap.update(heatmap)

        # every self.obj_capture_rate log on retrieved normals, sketch pairs in wadnb
        if (batch_idx % self.obj_capture_rate) == 0:
            _idx = np.where(gt_image_ids == self.obj_capture_image_id)
            gt_label = gt_labels[_idx].item()
            gt_image_id = gt_image_ids[_idx].item()
            _labels_at_1_object = labels_at_1_object[_idx].squeeze()
            _image_ids_at_1_object = image_ids_at_1_object[_idx].squeeze()

            index_images = {}
            # fetch the sketch from the current batch
            obj_id = self.metainfo.label_to_obj_id(gt_label)
            sketch = self.metainfo.load_sketch(obj_id, f"{gt_image_id:05}")
            plot_single_image(sketch)
            index_images["query/sketch"] = wandb.Image(plt)
            # fetch the top k retrieved normal images from the dataset
            images = []
            for label, image_id in zip(_labels_at_1_object, _image_ids_at_1_object):
                obj_id = self.metainfo.label_to_obj_id(label)
                normal = self.metainfo.load_normal(obj_id, f"{image_id:05}")
                normal = self.transform(normal)
                if gt_label != label:  # color background back of wrong labels
                    background = normal.mean(0) > 0.99
                    normal[:, background] = 0
                images.append(normal)
            image_data = transform_to_plot(images, batch=True)
            image_grid(image_data, rows=5, cols=18)
            index_images["index/normals"] = wandb.Image(plt)
            # log the sketch and normals
            self.logger.log_metrics(index_images)  # type: ignore

    def on_test_end(self) -> None:
        # heatmap based on: query sketch and how good are the different views
        heatmap_angles = self.heatmap.compute().mean(0).reshape(18, 5)
        heatmap_angles = heatmap_angles.detach().cpu().numpy()
        plt.clf()
        c = plt.imshow(heatmap_angles, cmap="viridis", interpolation="nearest")
        plt.colorbar(c)
        self.logger.log_metrics({"heatmap_angles": wandb.Image(c)})  # type: ignore

        # heatmap for both normals and sketch as query hence 2D matrix
        size = self.num_views_per_object
        heatmap_full = self.heatmap.compute()
        heatmap_full = heatmap_full.reshape(-1, size, size).mean(dim=0)
        heatmap_full = heatmap_full.detach().cpu().numpy()
        plt.clf()
        c = plt.imshow(heatmap_full, cmap="viridis", interpolation="nearest")
        plt.colorbar(c)
        self.logger.log_metrics({"heatmap_full": wandb.Image(c)})  # type: ignore
