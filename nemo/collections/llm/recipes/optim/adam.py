from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections.llm.utils import factory


@factory
def adam_with_cosine_annealing() -> nl.OptimizerModule:
    return nl.MegatronOptimizerModule(
        config=OptimizerConfig(optimizer="adam", lr=0.001, use_distributed_optimizer=True),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(),
    )


# TODO: Fix the name-arg inside the factory-function so we don't need to do this
with_cosine_annealing = adam_with_cosine_annealing
