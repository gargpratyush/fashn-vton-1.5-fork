import torch

from fashn_vton.pipeline import TryOnPipeline


class DummyTryOnModel:
    channels_in = 3
    input_shape = (12, 12)

    def forward_for_cfg(self, images, _timesteps, **_kwargs):
        return {"v_c": torch.zeros_like(images), "v_u": torch.zeros_like(images)}


def test_sample_returns_cpu_timing_result():
    pipeline = TryOnPipeline.__new__(TryOnPipeline)
    pipeline.tryon_model = DummyTryOnModel()
    image = torch.zeros((1, 3, 12, 12))
    pose = torch.zeros((1, 1, 12, 12))

    images, gpu_seconds = pipeline._sample(
        ca_images=image,
        garment_images=image,
        person_poses=pose,
        garment_poses=pose,
        garment_categories=torch.ones(1, dtype=torch.long),
        num_timesteps=1,
        use_tqdm=False,
    )

    assert len(images) == 1
    assert images[0].size == (12, 12)
    assert gpu_seconds is None
