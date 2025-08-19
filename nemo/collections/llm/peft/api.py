# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
import torch
from lightning.pytorch.trainer.states import TrainerFn
from megatron.core import dist_checkpointing
from rich.console import Console

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.peft.lora import LoRA, LoRAMerge
from nemo.collections.llm.utils import factory
from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib, io
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME, ckpt_to_context_subdir
from nemo.lightning.io import api
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.lightning.pytorch.callbacks import PEFT
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils import logging


@factory
def gpt_lora() -> PEFT:
    """ """
    return LoRA()


def export_lora(
    lora_checkpoint_path: str,
    output_path: str,
):
    """
    Export the LoRA adapter weights to HF format. Requires an implementation of HF PEFT exporter class.
    See HFLlamaPEFTExporter for an example.

    Python Usage:
    ```python
    if __name__ == '__main__':
        llm.peft.export_lora(
            lora_checkpoint_path=your_lora_checkpoint_path,
            output_path=your_output_path,
        )
    ```

    Args:
        lora_checkpoint_path: The path to the LoRA checkpoint.
        output_path: The path to save the HF checkpoint.

    """
    output = api.export_ckpt(
        path=Path(lora_checkpoint_path),
        target="hf-peft",
        output_path=Path(output_path),
    )

    console = Console()
    console.print(f"[green]✓ LoRA checkpoint exported to {output}[/green]")
    return output


def merge_lora(
    lora_checkpoint_path: str,
    output_path: str,
    legacy_ckpt: bool = False,
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

    # Load ckpt saved with TE < 1.14
    if legacy_ckpt:
        trainer.strategy.ckpt_load_strictness = False

    model, lora = _load_base_model_and_lora(lora_checkpoint_path)
    _setup_trainer_and_restore_model_and_adapter(Path(lora_checkpoint_path), trainer, model, lora)

    lora_merge = LoRAMerge()
    merged_model = lora_merge(trainer.strategy.megatron_parallel)
    merged_weights = {k: v for k, v in merged_model.sharded_state_dict().items() if ".adapter." not in k}
    _save_merged_weight(output_path, merged_weights, model, trainer)

    console = Console()
    console.print(f"[green]✓ LoRA checkpoint merged and saved to {output_path}[/green]")


def _load_base_model_and_lora(lora_checkpoint_path: Path) -> Tuple[pl.LightningModule, LoRA]:
    model = io.load_context(ckpt_to_context_subdir(lora_checkpoint_path), "model")
    model.model_transform, model.__io__.model_transform = None, None
    model.config.bf16 = True
    model.config.params_dtype = torch.bfloat16
    model.config.perform_initialization = False
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
    weights_dir = ckpt_to_weights_subdir(lora_checkpoint_path, is_saving=False)
    sharded_sd_metadata = trainer.strategy.unwrapped_checkpoint_io.load_content_metadata(weights_dir)
    adapter_sharded_state_dict = {
        k: v
        for k, v in trainer.strategy.megatron_parallel.sharded_state_dict(metadata=sharded_sd_metadata).items()
        if ".adapter." in k
    }
    adapter_state = trainer.strategy.checkpoint_io.load_checkpoint(
        weights_dir, sharded_state_dict=adapter_sharded_state_dict
    )
    trainer.strategy.load_model_state_dict(adapter_state, strict=False)


def _save_merged_weight(output_path: str, merged_weights: dict, model: pl.LightningModule, trainer: Trainer):
    weight_path = ckpt_to_weights_subdir(output_path, is_saving=True)
    Path(weight_path).mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(
        merged_weights,
        str(ckpt_to_weights_subdir(output_path, is_saving=True)),
        content_metadata=trainer.strategy.sharded_state_dict_metadata,
    )
    if hasattr(model.tokenizer, "save_pretrained"):
        model.tokenizer.save_pretrained("/tmp/nemo_tokenizer")
        model.tokenizer = AutoTokenizer("/tmp/nemo_tokenizer")
    if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, '__io__'):
        trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
    TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(output_path), yaml_attrs=["model"])
    logging.info(f"Merged checkpoint saved to {output_path}")


__all__ = ["gpt_lora", "merge_lora"]
