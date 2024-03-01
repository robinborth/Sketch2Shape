from lib.models.deepsdf import DeepSDF


class VideoCamera:
    def __init__(
        self,
        # base settings
        deepsdf_ckpt_path: str = "deepsdf.ckpt",
        # video settings
        latent_dir: str = "/latent_dir",
        keystones: dict = {},
        # rendering settings
        n_render_steps: int = 100,
        clamp_sdf: float = 0.1,
        step_scale: float = 1.0,
        surface_eps: float = 1e-03,
        sphere_eps: float = 1e-01,
        normal_eps: float = 5e-03,
    ):
        # init deepsdf
        self.deepsdf = DeepSDF.load_from_checkpoint(
            deepsdf_ckpt_path,
            strict=True,
            # rendering settings
            n_render_steps=n_render_steps,
            clamp_sdf=clamp_sdf,
            step_scale=step_scale,
            surface_eps=surface_eps,
            sphere_eps=sphere_eps,
            normal_eps=normal_eps,
        )
        self.deepsdf.freeze()
        self.deepsdf.eval()

        self.latent_dir = latent_dir
        self.keystones = keystones

    def create(self) -> list:
        return []
