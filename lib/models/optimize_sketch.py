import torch
from torch.nn.functional import cosine_similarity

from lib.models.optimize_latent import LatentOptimizer
from lib.models.siamese import Siamese


class DeepSDFSketchRender(LatentOptimizer):
    def __init__(
        self,
        siamese_ckpt_path: str,
        siamese_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.siamese = Siamese.load_from_checkpoint(siamese_ckpt_path)
        self.siamese.freeze()
        self.siamese.eval()

        self.chair_0_idx = [
            2930,
            189,
            94,
            2946,
            1604,
            2844,
            2543,
            2086,
            2556,
            1431,
            940,
            837,
            177,
            1129,
            934,
            2627,
            502,
            2536,
            2317,
            3732,
            2595,
            623,
            126,
            1264,
            2733,
            1070,
            4020,
            2144,
            2785,
            2013,
            4023,
            1225,
            2124,
            3928,
            3868,
            713,
            3021,
            1370,
            3252,
            742,
            364,
            3093,
            4047,
            2650,
            1486,
            2597,
            1162,
            1138,
            2749,
            1198,
            1532,
            2730,
            3687,
            3433,
            2686,
            1868,
            2306,
            2326,
            2427,
            1726,
            3722,
            893,
            231,
            2900,
            485,
            1186,
            1353,
            2793,
            2846,
            2168,
            3276,
            389,
            524,
            627,
            3400,
            1019,
            3062,
            1511,
            2295,
            165,
            527,
            3476,
            2206,
            2628,
            3074,
            572,
            3793,
            846,
            1320,
            2362,
            2473,
            3360,
            3805,
            2906,
            2532,
            855,
            1705,
            1755,
            810,
            3769,
            3539,
            3058,
            2039,
            3491,
            949,
            2257,
            422,
            3136,
            2504,
            1374,
            3410,
            112,
            2933,
            2828,
            1034,
            777,
            3576,
            1065,
            2015,
            1516,
            3688,
            3176,
            3002,
            932,
            274,
            825,
            3996,
            1622,
        ]
        self.couch_3_idx = [
            755,
            1898,
            782,
            3863,
            1962,
            962,
            2801,
            3385,
            1383,
            3196,
            1058,
            1637,
            2633,
            1248,
            667,
            3094,
        ]
        self.latent = self.deepsdf.lat_vecs.weight[self.chair_0_idx].mean(0)

    def training_step(self, batch, batch_idx):
        self.siamese.eval()

        # get the gt image and normals
        sketch = batch["sketch"]  # dim (1, 3, H, W) and values are (-1, 1)
        sketch_emb = self.siamese(sketch)  # (1, D)

        # calculate the normals map and embedding
        points, surface_mask = self.deepsdf.sphere_tracing(
            latent=self.latent,
            points=batch["points"].squeeze(),
            rays=batch["rays"].squeeze(),
            mask=batch["mask"].squeeze(),
        )
        rendered_normal = self.deepsdf.render_normals(
            latent=self.latent,
            points=points,
            mask=surface_mask,
        )  # (H, W, 3)
        normal = self.deepsdf.normal_to_siamese(rendered_normal)  # (1, 3, H, W)
        normal_emb = self.siamese(normal)  # (1, D)

        siamese_loss = 1 - cosine_similarity(sketch_emb, normal_emb).clone()
        siamese_loss *= self.hparams["siamese_weight"]
        self.log("optimize/siamese_loss", siamese_loss)

        reg_loss = torch.tensor(0).to(siamese_loss)
        if self.hparams["reg_loss"]:
            std = self.deepsdf.lat_vecs.weight[self.chair_0_idx].std(0)
            mean = self.deepsdf.lat_vecs.weight[self.chair_0_idx].mean(0)
            reg_loss = ((self.latent.clone() - mean) / std).pow(2)
            self.log("optimize/reg_loss_abs_max", torch.abs(reg_loss).max())

            # reg_loss = torch.norm(self.latent, dim=-1).clone()
            reg_loss *= self.hparams["reg_weight"]
            self.log("optimize/reg_loss", reg_loss.mean())

        loss = reg_loss.mean() + siamese_loss.mean()
        self.log("optimize/loss", loss, prog_bar=True)

        latent_norm = torch.norm(self.latent, dim=-1)
        self.log("optimize/latent_norm", latent_norm)

        # visualize the different images
        self.log_image("normal", self.deepsdf.siamese_input_to_image(normal))
        self.log_image("sketch", self.deepsdf.siamese_input_to_image(sketch))

        return loss
