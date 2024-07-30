import pytorch_lightning as pl

from nemo import lightning as nl
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.api import squad
from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel
from nemo.collections.llm.recipes.log.default import default_log
from nemo.collections.llm.recipes.optim.adam import adam_with_cosine_annealing
from nemo.collections.llm.utils import Partial, factory

NAME = "llama3_8b_16k"


@factory(name=NAME)
def model() -> pl.LightningModule:
    return LlamaModel(Llama3Config8B(seq_length=16384))


@factory(name=NAME)
def trainer(devices=8) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
        context_parallel_size=2,
        sequence_parallel=True,
    )

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )


@factory(name=NAME, for_task="llm.pretrain")
def pretrain_recipe() -> Partial:
    return Partial(
        pretrain,
        model=model,
        trainer=trainer,
        data=squad,
        log=default_log,
        optim=adam_with_cosine_annealing,
    )
