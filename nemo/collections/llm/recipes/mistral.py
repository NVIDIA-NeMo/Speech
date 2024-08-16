import pytorch_lightning as pl

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.api import squad
from nemo.collections.llm.gpt.model.mistral import MistralConfig7B, MistralModel
from nemo.collections.llm.peft.api import gpt_lora
from nemo.collections.llm.recipes.log.default import default_log
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.utils import Partial, factory

NAME = "mistral"


@factory(name=NAME)
def model() -> pl.LightningModule:
    return MistralModel(MistralConfig7B())


@factory(name=NAME)
def trainer(devices=8) -> nl.Trainer:
    strategy = nl.MegatronStrategy(tensor_model_parallel_size=2)

    return nl.Trainer(
        devices=devices,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )


@factory(name=NAME + "_hf")
def hf_resume() -> nl.AutoResume:
    return nl.AutoResume(import_path="hf://mistralai/Mistral-7B-v0.3")


@factory(name=NAME, for_task="llm.pretrain")
def pretrain_recipe() -> Partial:
    return Partial(
        pretrain,
        model=model,
        trainer=trainer,
        data=squad,
        log=default_log,
        optim=distributed_fused_adam_with_cosine_annealing(),
    )


@factory(name=NAME, for_task="llm.finetune")
def finetune_recipe() -> Partial:
    return Partial(
        finetune,
        model=model,
        trainer=trainer,
        data=squad,
        log=default_log,
        optim=distributed_fused_adam_with_cosine_annealing(),
        peft=gpt_lora,
        resume=hf_resume,
    )
