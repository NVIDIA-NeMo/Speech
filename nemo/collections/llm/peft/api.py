# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Tuple, Union

import pytorch_lightning as pl
from megatron.core import dist_checkpointing
from pytorch_lightning.trainer.states import TrainerFn

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.peft.lora import LoRA, LoRAMerge
from nemo.collections.llm.utils import factory
from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib, io
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME, ckpt_to_context_subdir
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.lightning.pytorch.callbacks import PEFT
from nemo.lightning.pytorch.callbacks.peft import PEFT
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils import logging


@factory
def gpt_lora() -> PEFT:
    return LoRA()


def merge_lora(
    lora_checkpoint_path: str,
    output_path: str,
) -> None:
    """
    Merges the LoRA adapter weights into the base model's weights.

    Python Usage:
    ```python
    if __name__ == '__main__':
        llm.peft.merge_lora(
            lora_checkpoint_path=your_lora_checkpoint_path,
            output_path=your_output_path,
        )
    ```

    Args:
        lora_checkpoint_path: The path to the LoRA checkpoint.
        output_path: The path to save the merged checkpoint.

    """
    from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed

    trainer = Trainer(
        devices=1,
        accelerator="cpu",
        strategy=MegatronStrategy(ddp="pytorch", setup_optimizers=False, plugins=bf16_mixed()),
    )

    model, lora = _load_base_model_and_lora(lora_checkpoint_path)
    _setup_trainer_and_restore_model_and_adapter(Path(lora_checkpoint_path), trainer, model, lora)

    lora_merge = LoRAMerge()
    merged_model = lora_merge(trainer.strategy.megatron_parallel)
    merged_weights = {k: v for k, v in merged_model.sharded_state_dict().items() if ".adapter." not in k}
    _save_merged_weight(output_path, merged_weights, model, trainer)


def _load_base_model_and_lora(lora_checkpoint_path: Path) -> Tuple[pl.LightningModule, LoRA]:
    model = io.load_context(ckpt_to_context_subdir(lora_checkpoint_path), "model")
    model.model_transform, model.__io__.model_transform = None, None
    model.config.bf16 = False
    lora: Union[io.TrainerContext, LoRA] = io.load_context(
        ckpt_to_context_subdir(lora_checkpoint_path), "model.model_transform"
    )
    assert isinstance(lora, LoRA), "LoRA config not found in checkpoint"
    return model, lora


def _setup_trainer_and_restore_model_and_adapter(
    lora_checkpoint_path: Path, trainer: Trainer, model: pl.LightningModule, lora: LoRA
) -> None:
    if (
        adapter_meta_path := ckpt_to_weights_subdir(lora_checkpoint_path, is_saving=False) / ADAPTER_META_FILENAME
    ).exists():
        with open(adapter_meta_path, "r") as f:
            metadata = json.load(f)
        restore_config = RestoreConfig(
            path=metadata["model_ckpt_path"],
            load_model_state=True,
            load_optim_state=False,
        )
    else:
        raise ValueError(f"Cannot find adapter meta file in {lora_checkpoint_path}")

    trainer.strategy.restore_config = restore_config
    trainer.strategy._setup_optimizers = False
    trainer.ckpt_path = None
    trainer.strategy.connect(model)
    trainer.strategy.setup_environment()

    if not model.state_dict():
        with _strategy_lib.megatron_cpu_init_context(model.config):
            model.configure_model()

    trainer.strategy.setup(trainer)  # load base model ckpt
    trainer.state.fn = TrainerFn.TESTING
    trainer.strategy.setup_megatron_parallel(trainer=trainer)
    trainer.strategy.trainer = trainer
    model.trainer = trainer

    lora(model)
    adapter_sharded_state_dict = {
        k: v for k, v in trainer.strategy.megatron_parallel.sharded_state_dict().items() if ".adapter." in k
    }
    adapter_state = trainer.strategy.checkpoint_io.load_checkpoint(
        ckpt_to_weights_subdir(lora_checkpoint_path, is_saving=False), sharded_state_dict=adapter_sharded_state_dict
    )
    trainer.strategy.load_model_state_dict(adapter_state, strict=False)


def _save_merged_weight(output_path: str, merged_weights: dict, model: pl.LightningModule, trainer: Trainer):
    weight_path = ckpt_to_weights_subdir(output_path, is_saving=True)
    Path(weight_path).mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(merged_weights, str(ckpt_to_weights_subdir(output_path, is_saving=True)))
    if hasattr(model.tokenizer, "save_pretrained"):
        model.tokenizer.save_pretrained("/tmp/nemo_tokenizer")
        model.tokenizer = AutoTokenizer("/tmp/nemo_tokenizer")
    if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, '__io__'):
        trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
    TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(output_path), yaml_attrs=["model"])
    logging.info(f"Merged checkpoint saved to {output_path}")


__all__ = ["gpt_lora", "merge_lora"]
