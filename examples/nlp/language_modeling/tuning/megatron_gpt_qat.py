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

from itertools import islice

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.export.quantize import Quantizer
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)

"""
This is a modified version of `megatron_gpt_finetuning.py` to perform PTQ and QAT on a SFT Model like Llama2-7b.
Please see docs/source/nlp/quantization.rst for more details on the usage.
"""


def get_forward_loop(fwd_bwd_step, dataloader, num_batches):
    if len(dataloader) < num_batches:
        logging.warning(
            f"Dataloader has fewer batches ({len(dataloader)}) than required ({num_batches}) for calibration."
        )
        num_batches = len(dataloader)

    def forward_loop(model):
        data_iter = islice(iter(dataloader), num_batches)
        for _ in tqdm(range(num_batches), desc="Calibrating"):
            fwd_bwd_step(data_iter, forward_only=True)

    return forward_loop


@hydra_runner(config_path="conf", config_name="megatron_gpt_qat_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    quantizer = Quantizer(cfg.quantization, cfg.export)

    model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
    model_cfg = quantizer.modify_model_config(model_cfg)

    model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    assert model.mcore_gpt, "Only MCoreGPTModel is supported with nvidia-modelopt for QAT."

    # Setup dataloaders
    model.setup()

    # Perform PTQ on the SFT Model
    if cfg.quantization.algorithm is not None:
        model_module_list = model.get_model_module_list()
        assert len(model_module_list) == 1
        unwrapped_model = model_module_list[0]

        num_batches = cfg.quantization.num_calib_size // cfg.model.global_batch_size
        forward_loop = get_forward_loop(model.fwd_bwd_step, model.train_dataloader(), num_batches)
        quantizer.quantize(unwrapped_model, forward_loop)

        logging.info("Validating model after PTQ...")
        trainer.validate(model)

    # Perform QAT on the PTQ Model
    trainer.fit(model)

    # Export the quantized model for TensorRT-LLM inference
    # INT4 export is not supported yet
    if cfg.quantization.algorithm != "int4":
        quantizer.export(model)


if __name__ == '__main__':
    main()
