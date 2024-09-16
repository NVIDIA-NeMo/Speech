import pytorch_lightning as pl
from megatron.core import ModelParallelConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy


class MegatronTokenDropCallback(Callback):
    """
    A PyTorch Lightning callback to enable token drop for MOEs. Token drop improves performance by better
    balancing work across experts, but may affect convergence.

    Args:
        moe_expert_capacity_factor (float): The capacity factor for all experts
        moe_pad_expert_input_to_capacity (bool): Pad the input for each expert to the expert capacity lengt

    Example:
        >>> callback = MegatronCommOverlapCallback()
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        moe_expert_capacity_factor: float = 1.0,
        moe_pad_expert_input_to_capacity: bool = True,
    ):

        if moe_expert_capacity_factor < 0:
            moe_expert_capacity_factor = None
        self.moe_expert_capacity_factor = moe_expert_capacity_factor
        self.moe_pad_expert_input_to_capacity = moe_pad_expert_input_to_capacity

    def _set_cfgs(self, cfg):
        cfg.moe_expert_capacity_factor = self.moe_expert_capacity_factor
        cfg.moe_pad_expert_input_to_capacity = self.moe_pad_expert_input_to_capacity

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        assert isinstance(trainer.strategy, MegatronStrategy), "MegatronTokenDrop requires MegatronStrategy"
        if hasattr(trainer.model, "config") and isinstance(trainer.model.config, ModelParallelConfig):
            assert trainer.model.config.moe_token_dispatcher_type in [
                "alltoall",
                "alltoall_seq",
            ], 'moe_expert_capacity_factor only works with alltoall token dispatcher'
            assert trainer.model.config.moe_router_load_balancing_type in [
                "aux_loss",
                "none",
            ], 'moe_expert_capacity_factor only works with aux_loss or none load balancing'

            if self.moe_pad_expert_input_to_capacity:
                if self.moe_expert_capacity_factor is None:
                    raise ValueError('moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity')

            self._set_cfgs(trainer.model.config)
            if hasattr(trainer.model, '__io__'):
                self._set_cfgs(trainer.model.__io__.config)
