from typing import Mapping

_HAS_HYDRA = True

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:
    DictConfig = Mapping
    OmegaConf = None
    _HAS_HYDRA = False


def resolve_trainer_cfg(trainer_cfg: DictConfig) -> DictConfig:
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True)
    if not _HAS_HYDRA:
        return trainer_cfg
    if (strategy := trainer_cfg.get("strategy", None)) is not None and isinstance(strategy, Mapping):
        trainer_cfg["strategy"] = hydra.utils.instantiate(strategy)
    return trainer_cfg
