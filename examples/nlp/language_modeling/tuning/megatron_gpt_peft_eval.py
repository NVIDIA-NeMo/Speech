# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


import os

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from torch.utils.data import DataLoader

from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import (
    MegatronGPTAdapterModel,
    MegatronGPTAdapterPTuningModel,
    MegatronGPTIA3Model,
    MegatronGPTLoRAModel,
    MegatronGPTPEFTModel,
    MegatronGPTPTuningModel,
)
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PEFTSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

mp.set_start_method("spawn", force=True)

"""
This is the script to train an Adapter infused GPT Model for text generation.
A base GPT Model is required as a starting point. This script will then insert
Adapters into each Transformer layer and will train/update only these adapters
during training. The base GPT Model weights will remain frozen.

During training this script will only save the newly trained Adapter weights
in checkpoints. At the end of training a .nemo file of Adapter weights will 
be saved.

Usage:
    Assuming the base model is a 125m GPT Model, with TP=1, PP=1:
    a. run a training run for a base gpt nemo file:
        python megatron_gpt_adapter_tuning.py \
            "model.data.train_ds=[PATH TO TRAINING JSONL FILE]",
            "model.data.validation_ds=[PATH TO VALIDATION JSONL FILE]",
            model.language_model_path="PATH TO BASE GPT MODEL .nemo FILE"
            name="NAME OF TRAINING RUN"
            exp_manager.exp_dir="DIR TO SAVE CHECKPOINTS and .nemo FILE",
            trainer.max_epochs=2
"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_peft_eval_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    assert cfg.model.restore_from_path is not None
    assert cfg.model.peft.restore_from_path is not None
    megatron_amp_o2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = False

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, "bf16"]:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get("native_amp_init_scale", 2 ** 32),
                growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                hysteresis=cfg.model.get("hysteresis", 2),
                enabled=False
                if cfg.model.pipeline_model_parallel_size > 1
                else True,  # turn off the grad scale for pipeline parallel LM model
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))

    if cfg.get("cluster_type", None) == "BCP":
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    peft_model_cfg = MegatronGPTPEFTModel.restore_from(
        restore_path=cfg.model.peft.restore_from_path, trainer=trainer, return_config=True,
    )

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(peft_model_cfg):
        # update the model config of the trained model with params we want to set at inference time.
        peft_model_cfg.precision = cfg.trainer.precision
        peft_model_cfg.data.test_ds = cfg.model.data.test_ds

    with open_dict(cfg):
        # update the config with the trained model config
        # required for hydra interpolation to work inside cfg.inference
        cfg.inference.add_BOS = peft_model_cfg.data.test_ds.add_bos
        cfg.inference.tokens_to_generate = peft_model_cfg.data.test_ds.tokens_to_generate

    save_restore_connector = PEFTSaveRestoreConnector(
        peft_model_nemo_path=cfg.model.peft.restore_from_path, peft_model_ckpt_path=None,
    )
    if os.path.isdir(peft_model_cfg.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
    # peft_cls = _get_peft_scheme(peft_model_cfg)
    model = NLPModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=peft_model_cfg,
        save_restore_connector=save_restore_connector,
    )

    model.freeze()
    _test_ds = model._build_dataset(peft_model_cfg.data.test_ds, is_train=False)
    request_dl = DataLoader(
        dataset=_test_ds[0],
        batch_size=peft_model_cfg.data.test_ds.global_batch_size,
        collate_fn=_test_ds[0].collate_fn,
    )
    config = OmegaConf.to_container(cfg.inference, resolve=True)
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)
    if model.global_rank == 0:
        print("***************************")
        if cfg.inference.outfile_path is not None:
            with open(cfg.inference.outfile_path, "w", encoding="utf-8") as f:
                for batch in response:
                    for sentence in batch["sentences"]:
                        s = " ".join(sentence.split("\n"))
                        f.write(s + "\n")
            print("predictions saved to {}".format(cfg.inference.outfile_path))
        else:
            print(response)
    print("***************************")


if __name__ == "__main__":
    main()
