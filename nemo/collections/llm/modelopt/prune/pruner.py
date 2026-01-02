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

from dataclasses import dataclass
from functools import partial

import modelopt.torch.prune as mtp
import pytorch_lightning as pl
from megatron.core import dist_checkpointing

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

SUPPORTED_PRUNING_HPARAMS = {
    # Width pruning
    "ffn_hidden_size",
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "mamba_num_heads",
    "mamba_head_dim",
    # Depth pruning
    "num_layers",
}


@dataclass
class PruningConfig:
    """Pruning parameters. None means no pruning of the corresponding dimension.

    Args:
        target_ffn_hidden_size (int, optional): Target size of MLP FFN hidden dimension.
        target_hidden_size (int, optional): Target size of embedding hidden dimension.
        target_num_attention_heads (int, optional): Target number of attention heads.
            Required if `target_num_query_groups` is provided.
        target_num_query_groups (int, optional): Target number of query groups for grouped-query attention.
            Required if `target_num_attention_heads` is provided.
        target_mamba_num_heads (int, optional): Target number of Mamba attention heads.
        target_mamba_head_dim (int, optional): Target dimension of Mamba attention heads.
        target_num_layers (int, optional): Target number of transformer layers using importance metric.
        drop_layers (list[int], optional): List of specific layer indices (1-indexed) to drop from the model.
            Cannot be used with other pruning parameters.
    """

    target_ffn_hidden_size: int | None = None
    target_hidden_size: int | None = None
    target_num_attention_heads: int | None = None
    target_num_query_groups: int | None = None
    target_mamba_num_heads: int | None = None
    target_mamba_head_dim: int | None = None
    target_num_layers: int | None = None
    drop_layers: list[int] | None = None

    def __post_init__(self):
        if self.drop_layers:
            other_params = [
                self.target_ffn_hidden_size,
                self.target_hidden_size,
                self.target_num_attention_heads,
                self.target_num_query_groups,
                self.target_mamba_num_heads,
                self.target_mamba_head_dim,
                self.target_num_layers,
            ]
            if any(p is not None for p in other_params):
                raise ValueError("drop_layers cannot be used with other pruning parameters")


def prune_language_model(
    model: llm.GPTModel,
    pruning_config: PruningConfig,
    data_module: pl.LightningDataModule | None = None,
    trainer: nl.Trainer | None = None,
) -> llm.GPTModel:
    """Prune a GPT / Mamba (sub-class of GPT) model in-place based on the provided pruning configuration.

    Args:
        model (llm.GPTModel): The model to prune.
        pruning_config (PruningConfig): The pruning configuration.
        data_module (pl.LightningDataModule, optional): The data module to use for pruning.
            Required if not dropping layers.
        trainer (nl.Trainer, optional): The trainer to use for pruning.
            Required if not dropping layers.

    Returns:
        llm.GPTModel: The pruned model.
    """
    if pruning_config.drop_layers:
        mtp.plugins.drop_mcore_language_model_layers(model, layers_to_drop=pruning_config.drop_layers)
    else:
        assert data_module is not None, "data_module is required to prune the model."
        assert trainer is not None, "trainer is required to prune the model."
        # Overwrite val dataloader to use train dataloader with llm.validate
        data_module.val_dataloader = data_module.train_dataloader

        logging.info("Pruning model...")
        export_config = {
            k: getattr(pruning_config, f"target_{k}")
            for k in SUPPORTED_PRUNING_HPARAMS
            if getattr(pruning_config, f"target_{k}") is not None
        }
        mtp.prune(
            model,
            mode="mcore_minitron",
            constraints={"export_config": export_config},
            dummy_input=None,  # Not used
            config={"forward_loop": partial(llm.validate, data=data_module, trainer=trainer, tokenizer="model")},
        )

    return model


def save_pruned_model(trainer: nl.Trainer, save_path: str) -> None:
    """Save pruned model nemo checkpoint."""
    logging.info(f"Saving pruned model to {save_path}...")
    # Make sure pruned hparams are saved
    for k in SUPPORTED_PRUNING_HPARAMS | {"kv_channels"}:
        setattr(trainer.model.__io__.config, k, getattr(trainer.model.config, k))

    # TODO: trainer.save_checkpoint(save_path) doesnt seem to save metadata.json or .metadata files!
    weight_path = ckpt_to_weights_subdir(save_path, is_saving=True)
    weight_path.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(
        trainer.strategy.megatron_parallel.sharded_state_dict(),
        weight_path,
        content_metadata=trainer.strategy.sharded_state_dict_metadata,
    )

    if is_global_rank_zero():
        TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(save_path), yaml_attrs=["model"])

    logging.info(f"Pruned model saved to {save_path}\n")
