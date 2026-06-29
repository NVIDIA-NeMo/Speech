# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
    seed_everything(cfg.data.train_ds.seed)
    torch.set_float32_matmul_precision("medium")

    # Pin the process-global default dtype to match bf16-true precision. Under
    # bf16-true, the default dtype is bf16 around the forward, but activation-
    # checkpoint recompute runs in backward where it has reverted to fp32. Any
    # @torch.compile'd kernel whose Dynamo GLOBAL_STATE guard includes
    # default_dtype (e.g. Automodel's Float32RMSNorm) is then traced under bf16
    # in the forward and re-entered under fp32 in recompute -> it recompiles every
    # step until Dynamo's recompile_limit (8) raises, killing training ~33 min in
    # (2904 mtp4r-v2 BSHD recipe, 2026-06-29). Pinning the global default makes the
    # forward and the recompute agree so the guard never flips. This protects all
    # compiled kernels (not just RMSNorm) and is numerically inert: bf16-true
    # already trains in bf16, and fp32-critical modules/buffers are created with an
    # explicit dtype (RMSNorm upcasts via .float(); RoPE/A_log/router bias are
    # explicitly fp32), so they are unaffected by the default.
    if cfg.trainer.get("precision") == "bf16-true":
        torch.set_default_dtype(torch.bfloat16)

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    model_cls = SALM
    if cfg.model.get("use_nemo_automodel", False):
        from nemo.collections.speechlm2 import SALMAutomodel

        model_cls = SALMAutomodel

    with trainer.init_module():
        model = model_cls(OmegaConf.to_container(cfg.model, resolve=True))

    dataset = SALMDataset(tokenizer=model.tokenizer, multispeaker_cfg=cfg.data.get("multispeaker_cfg", None))
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    # Evaluation-only path: run the Lightning validation loop without any
    # training (e.g. to measure MTP per-head token acceptance on a checkpoint
    # loaded via model.init_from_checkpoint). configure_model() still loads the
    # checkpoint weights for validate, so this exercises the val metrics on the
    # restored model without touching its weights.
    if cfg.get("run_validate_only", False):
        trainer.validate(model, datamodule)
    else:
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
