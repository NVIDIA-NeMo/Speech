from dataclasses import asdict, dataclass, fields
import pytorch_lightning as pl

from megatron.core import ModelParallelConfig
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import TransformerLayerTPOverlapCfg
from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy, ParallelismConfig
from nemo.utils import logging

try:
    from megatron.core.num_microbatches_calculator import get_micro_batch_size
except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_micro_batch_size

try:
    import transformer_engine

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass
class _CommOverlapConfig:
    # Tensor parallel communication overlap (experimental)
    tp_comm_overlap: bool = None
    tp_comm_overlap_cfg: dict = None
    # Pipeline parallel communication overlap
    overlap_p2p_comm: bool = None
    batch_p2p_comm: bool = None
    # Data parallel communication overlap
    overlap_grad_reduce: bool = None
    overlap_param_gather: bool = None
    overlap_param_gather_with_optimizer_step: bool = None
    align_param_gather: bool = None
    bucket_size: int = None
    # Pipeline bubble overlap
    defer_embedding_wgrad_compute: bool = None
    wgrad_deferral_limit: int = None


class MegatronCommOverlapCallback(Callback):
    """
    A PyTorch Lightning callback to enable communication compute overlap.
    This callback enables the following:
        - tensor parallel communication overlap
        - pipeline parallel communication overlap
        - data parallel communication overlap
        - pipeline bubble overlap

    Args:
        tp_comm_overlap (bool): Enable tensor parallel overlap (experimental)
        tp_comm_overlap_cfg (TransformerLayerTPOverlapCfg): Tensor parallel overlap config
        overlap_p2p_comm (bool): Enable pipeline parallel overlap
        batch_p2p_comm (bool): Batch pipeline parallel send/recv into a single op
        overlap_grad_reduce (bool): Overlap data parallel gradient reduction with compute
        overlap_param_gather (bool): Overlap data parallel parameter gather with compute
        overlap_param_gather_with_optimizer_step (bool): Overlap data parallel parameter gather optimizer step
        align_param_gather (bool): Align data parallel parameter gather across virtual pipeline chunks
        bucket_size (int): The DDP bucket size, controls the data parallel overlap granularity
        defer_embedding_wgrad_compute (bool): Overlap wgrads with the pipeline drain bubble for the last pipeline stage
        wgrad_deferral_limit (int): Limit of how many outstanding wgrads may be overlapped with the pipeline drain bubble

    Example:
        >>> callback = MegatronCommOverlapCallback(tp_comm_overlap=True)
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        tp_comm_overlap: bool = None,
        tp_comm_overlap_cfg: TransformerLayerTPOverlapCfg = None,
        overlap_p2p_comm: bool = None,
        batch_p2p_comm: bool = None,
        overlap_grad_reduce: bool = None,
        overlap_param_gather: bool = None,
        overlap_param_gather_with_optimizer_step: bool = None,
        align_param_gather: bool = None,
        bucket_size: int = None,
        defer_embedding_wgrad_compute: bool = None,
        wgrad_deferral_limit: int = None,
    ):

        self.user_comm_overlap_cfg = _CommOverlapConfig(
            tp_comm_overlap=tp_comm_overlap,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
            align_param_gather=align_param_gather,
            bucket_size=bucket_size,
            defer_embedding_wgrad_compute=defer_embedding_wgrad_compute,
            wgrad_deferral_limit=wgrad_deferral_limit,
        )

        self.tp_comm_overlap_cfg = None
        self.need_tp_overlap_ub_init = False

    def _get_model_comm_overlap_cfgs(
        self,
        parallelism_cfg: ParallelismConfig,
    ) -> _CommOverlapConfig:
        comm_overlap_cfg = _CommOverlapConfig()

        vp_size = parallelism_cfg.virtual_pipeline_model_parallel_size
        if vp_size is None:
            vp_size = 1

        # Optimizations disabled by default, can be overriden by user
        comm_overlap_cfg.tp_comm_overlap = False
        comm_overlap_cfg.tp_comm_overlap_cfg = None
        comm_overlap_cfg.defer_embedding_wgrad_compute = False
        comm_overlap_cfg.wgrad_deferral_limit = -1

        # Check if TP overlap can be safely enabled
        if self.user_comm_overlap_cfg.tp_comm_overlap is True:
            if parallelism_cfg.tensor_model_parallel_size < 2:
                logging.warning("Disabling tensor parallel communication overlap due to TP size < 2.")
                self.user_comm_overlap_cfg.tp_comm_overlap = False
            elif not parallelism_cfg.sequence_parallel:
                logging.warning("Disabling tensor parallel communication overlap due to sequence_parallel=False.")
                self.user_comm_overlap_cfg.tp_comm_overlap = False
            elif not HAVE_TE:
                logging.warning("Disabling tensor parallel communication overlap due to TE not detected.")
                self.user_comm_overlap_cfg.tp_comm_overlap = False

        # PP overlap
        if parallelism_cfg.pipeline_model_parallel_size > 1:
            if vp_size > 1:
                comm_overlap_cfg.overlap_p2p_comm = True
                comm_overlap_cfg.batch_p2p_comm = False
            else:
                comm_overlap_cfg.overlap_p2p_comm = False
                comm_overlap_cfg.batch_p2p_comm = True
        else:
            comm_overlap_cfg.overlap_p2p_comm = False
            comm_overlap_cfg.batch_p2p_comm = False

        comm_overlap_cfg = self._override_user_cfgs(comm_overlap_cfg)
        return comm_overlap_cfg

    def _get_optimizer_overlap_cfgs(self, parallelism_cfg: ParallelismConfig) -> _CommOverlapConfig:
        from nemo.utils import AppState

        app_state = AppState()
        data_parallel_size = app_state.data_parallel_size

        vp_size = parallelism_cfg.virtual_pipeline_model_parallel_size
        if vp_size is None:
            vp_size = 1

        comm_overlap_cfg = _CommOverlapConfig()
        comm_overlap_cfg.bucket_size = None
        comm_overlap_cfg.overlap_grad_reduce = False
        comm_overlap_cfg.overlap_param_gather = False
        comm_overlap_cfg.overlap_param_gather_with_optimizer_step = False
        comm_overlap_cfg.align_param_gather = False

        if data_parallel_size > 1:
            comm_overlap_cfg.bucket_size = 128 * 1024 * 1024
            comm_overlap_cfg.overlap_grad_reduce = True
            comm_overlap_cfg.overlap_param_gather = True
            if parallelism_cfg.pipeline_model_parallel_size > 1 and vp_size > 1:
                comm_overlap_cfg.overlap_param_gather_with_optimizer_step = True
                comm_overlap_cfg.align_param_gather = True

        comm_overlap_cfg = self._override_user_cfgs(comm_overlap_cfg)
        return comm_overlap_cfg

    def _apply_cfgs(self, src_cfg, dest_cfg):
        # apply optimizations into dest_cfg
        for field in fields(src_cfg):
            if hasattr(dest_cfg, field.name):
                setattr(dest_cfg, field.name, getattr(src_cfg, field.name))

    def _override_user_cfgs(self, comm_overlap_cfg):
        # override default configs with any user provided configs
        if isinstance(self.user_comm_overlap_cfg, _CommOverlapConfig):
            for field in fields(self.user_comm_overlap_cfg):
                user_value = getattr(self.user_comm_overlap_cfg, field.name)
                if user_value is not None:
                    setattr(comm_overlap_cfg, field.name, user_value)

        return comm_overlap_cfg

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        assert isinstance(trainer.strategy, MegatronStrategy), "MegatronCommOverlapCallback requires MegatronStrategy"
        parallelism_cfg = trainer.strategy.parallelism

        if hasattr(trainer.model, "config") and isinstance(trainer.model.config, ModelParallelConfig):
            comm_overlap_cfg = self._get_model_comm_overlap_cfgs(parallelism_cfg)
            self._apply_cfgs(comm_overlap_cfg, trainer.model.config)
            if hasattr(trainer.model, '__io__'):
                self._apply_cfgs(comm_overlap_cfg, trainer.model.__io__.config)

            if trainer.model.config.tp_comm_overlap:
                self.tp_comm_overlap_cfg = comm_overlap_cfg.tp_comm_overlap_cfg
                self.need_tp_overlap_ub_init = True

        # Data parallel overlap is only available with the Megatron DDP and Distributed optimizer
        if (
            hasattr(trainer.model.optim, "config")
            and isinstance(trainer.model.optim.config, OptimizerConfig)
            and isinstance(trainer.strategy.ddp_config, DistributedDataParallelConfig)
            and trainer.strategy.ddp_config.use_distributed_optimizer
        ):
            comm_overlap_cfg = self._get_optimizer_overlap_cfgs(parallelism_cfg)
            self._apply_cfgs(comm_overlap_cfg, trainer.model.optim.config)
            self._apply_cfgs(comm_overlap_cfg, trainer.strategy.ddp_config)
            if hasattr(trainer.model, '__io__'):
                self._apply_cfgs(comm_overlap_cfg, trainer.model.__io__.optim.config)

    def _init_te_userbuffers(self, model_parallel_cfg: ModelParallelConfig):
        from megatron.core import parallel_state

        if self.tp_comm_overlap_cfg is None:
            logging.warning(
                "Tensor parallel overlap: No overlap config provided. Initializing TP comm overlap with the default config."
            )
        else:
            # ub_cfgs is a dataclass, however TE needs a dict, so convert here
            self.tp_comm_overlap_cfg = asdict(self.tp_comm_overlap_cfg)

        micro_batch_size = get_micro_batch_size()
        hidden_size = model_parallel_cfg.hidden_size
        sequence_length = model_parallel_cfg.seq_length
        fp8 = model_parallel_cfg.fp8 is not None

        input_shape = [
            sequence_length * micro_batch_size // parallel_state.get_context_parallel_world_size(),
            hidden_size,
        ]

        try:
            transformer_engine.pytorch.module.base.initialize_ub(
                shape=input_shape,
                tp_size=parallel_state.get_tensor_model_parallel_world_size(),
                use_fp8=fp8,
                ub_cfgs=self.tp_comm_overlap_cfg,
            )
        except Exception as error:
            raise Exception(f"Tensor parallel overlap: userbuffer initialization failed with {error}")

        self.need_tp_overlap_ub_init = False

    # _init_te_userbuffers must run once before any stages, however there isnt such a
    # unified callback, so add a hook for every stage
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.need_tp_overlap_ub_init:
            self._init_te_userbuffers(trainer.model.config)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.need_tp_overlap_ub_init:
            self._init_te_userbuffers(trainer.model.config)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.need_tp_overlap_ub_init:
            self._init_te_userbuffers(trainer.model.config)

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.need_tp_overlap_ub_init:
            self._init_te_userbuffers(trainer.model.config)
