from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy

from nemo.utils.trainer_utils import resolve_trainer_cfg


def test_resolve_trainer_cfg_strategy():
    cfg = OmegaConf.create({"strategy": "ddp"})
    ans = resolve_trainer_cfg(cfg)
    assert isinstance(ans, dict)
    assert ans["strategy"] == "ddp"

    cfg = OmegaConf.create(
        {"strategy": {"_target_": "pytorch_lightning.strategies.DDPStrategy", "gradient_as_bucket_view": True}}
    )
    ans = resolve_trainer_cfg(cfg)
    assert isinstance(ans, dict)
    assert isinstance(ans["strategy"], DDPStrategy)
    assert "gradient_as_bucket_view" in ans["strategy"]._ddp_kwargs
    assert ans["strategy"]._ddp_kwargs["gradient_as_bucket_view"] == True
