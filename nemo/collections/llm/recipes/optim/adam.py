from megatron.core.optimizer import OptimizerConfig

from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule, OptimizerModule


def distributed_fused_adam_with_cosine_annealing(max_lr: float = 1e-4) -> Config[OptimizerModule]:
    opt_cfg = Config(
        OptimizerConfig,
        optimizer="adam",
        lr=max_lr,
        weight_decay=0.1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
    )

    sched = Config(
        CosineAnnealingScheduler,
        warmup_steps=2000,
        constant_steps=0,
        min_lr=0.1 * max_lr,
    )

    return Config(
        MegatronOptimizerModule,
        config=opt_cfg,
        lr_scheduler=sched,
    )
