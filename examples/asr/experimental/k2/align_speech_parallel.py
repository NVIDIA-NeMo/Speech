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

"""
# Based on examples/asr/transcribe_speech_parallel.py
# ASR alignment with multi-GPU/multi-node support for large datasets
# It supports both tarred and non-tarred datasets
# Arguments
#    model: path to a nemo/PTL checkpoint file or name of a pretrained model
#    predict_ds: config of the dataset/dataloader
#    aligner_args: aligner config
#    output_path: path to store the predictions
#    model_stride: model downsampling factor, 8 for Citrinet models and 4 for Conformer models
#
# Results of each GPU/worker is written into a file named 'predictions_{rank}.json, and aggregated results of all workers are written into 'predictions_all.json'

Example for non-tarred datasets:

python align_speech_parallel.py \
    model=stt_en_conformer_ctc_large \
    predict_ds.manifest_filepath=/dataset/manifest_file.json \
    predict_ds.batch_size=16 \
    output_path=/tmp/

Example for tarred datasets:

python align_speech_parallel.py \
    predict_ds.is_tarred=true \
    predict_ds.manifest_filepath=/tarred_dataset/tarred_audio_manifest.json \
    predict_ds.tarred_audio_filepaths=/tarred_dataset/audio__OP_0..127_CL_.tar \
    ...

By default the trainer uses all the GPUs available and default precision is FP32.
By setting the trainer config you may control these configs. For example to do the predictions with AMP on just two GPUs:

python align_speech_parallel.py \
    trainer.precision=16 \
    trainer.devices=2 \
    ...

You may control the dataloader's config by setting the predict_ds:

python align_speech_parallel.py \
    predict_ds.num_workers=8 \
    predict_ds.min_duration=2.0 \
    predict_ds.sample_rate=16000 \
    model=stt_en_conformer_ctc_small \
    ...

You may control the aligner's config by setting the aligner_args:
    aligner_args.alignment_type=argmax \
    aligner_args.word_output=False \
    aligner_args.cpu_decoding=True \
    aligner_args.decode_batch_size=8 \
    aligner_args.ctc_cfg.prob_suppress_index=-1 \
    aligner_args.ctc_cfg.prob_suppress_value=0.5 \
    aligner_args.rnnt_cfg.predictor_window_size=10 \
    aligner_args.decoder_module_cfg.intersect_pruned=true \
    aligner_args.decoder_module_cfg.intersect_conf.search_beam=40 \
    ...

"""


import os
from dataclasses import dataclass, field, is_dataclass
from typing import Optional

import pytorch_lightning as ptl
import torch
from omegaconf import MISSING, OmegaConf

from nemo.collections.asr.data.audio_to_ctm_dataset import ASRCTMPredictionWriter
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.configs.aligner_config import K2AlignerWrapperModelConfig
from nemo.collections.asr.models.configs.asr_models_config import ASRDatasetConfig
from nemo.collections.asr.models.k2_aligner_model import AlignerWrapperModel
from nemo.core.config import TrainerConfig, hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@dataclass
class ParallelAlignmentConfig:
    model: Optional[str] = None  # name
    predict_ds: ASRDatasetConfig = field(
        default_factory=lambda: ASRDatasetConfig(return_sample_id=True, num_workers=4)
    )
    aligner_args: K2AlignerWrapperModelConfig = field(default_factory=lambda: K2AlignerWrapperModelConfig())
    output_path: str = MISSING
    model_stride: int = 8

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(devices=-1, accelerator="ddp"))

    # there arguments will be ignored
    return_predictions: bool = False
    use_cer: bool = False


def match_train_config(predict_ds, train_ds):
    # It copies the important configurations from the train dataset of the model
    # into the predict_ds to be used for prediction. It is needed to match the training configurations.
    if train_ds is None:
        return

    predict_ds.sample_rate = train_ds.get("sample_rate", 16000)
    cfg_name_list = [
        "int_values",
        "use_start_end_token",
        "blank_index",
        "unk_index",
        "normalize",
        "parser",
        "eos_id",
        "bos_id",
        "pad_id",
    ]

    if is_dataclass(predict_ds):
        predict_ds = OmegaConf.structured(predict_ds)
    for cfg_name in cfg_name_list:
        if hasattr(train_ds, cfg_name):
            setattr(predict_ds, cfg_name, getattr(train_ds, cfg_name))

    return predict_ds


@hydra_runner(config_name="AlignmentConfig", schema=ParallelAlignmentConfig)
def main(cfg: ParallelAlignmentConfig):
    if cfg.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = ASRModel.restore_from(restore_path=cfg.model, map_location="cpu")
    elif cfg.model.endswith(".ckpt"):
        logging.info("Attempting to initialize from .ckpt file")
        model = ASRModel.load_from_checkpoint(checkpoint_path=cfg.model, map_location="cpu")
    else:
        logging.info(
            "Attempting to initialize from a pretrained model as the model name does not have the extension of .nemo or .ckpt"
        )
        model = ASRModel.from_pretrained(model_name=cfg.model, map_location="cpu")

    trainer = ptl.Trainer(**cfg.trainer)

    cfg.predict_ds.return_sample_id = True
    cfg.return_predictions = False
    cfg.use_cer = False
    cfg.predict_ds = match_train_config(predict_ds=cfg.predict_ds, train_ds=model._cfg.train_ds)
    data_loader = model._setup_dataloader_from_config(cfg.predict_ds)

    os.makedirs(cfg.output_path, exist_ok=True)
    # trainer.global_rank is not valid before predict() is called. Need this hack to find the correct global_rank.
    global_rank = trainer.node_rank * trainer.num_devices + int(os.environ.get("LOCAL_RANK", 0))
    output_file = os.path.join(cfg.output_path, f"predictions_{global_rank}.json")
    output_ctm_dir = os.path.join(cfg.output_path, "ctm")
    predictor_writer = ASRCTMPredictionWriter(
        dataset=data_loader.dataset,
        output_file=output_file,
        output_ctm_dir=output_ctm_dir,
        time_per_frame=cfg.model_stride * model._cfg.preprocessor['window_stride'],
    )
    trainer.callbacks.extend([predictor_writer])

    aligner_wrapper = AlignerWrapperModel(model=model, cfg=cfg.aligner_args)
    trainer.predict(model=aligner_wrapper, dataloaders=data_loader, return_predictions=cfg.return_predictions)
    samples_num = predictor_writer.close_output_file()

    logging.info(
        f"Prediction on rank {global_rank} is done for {samples_num} samples and results are stored in {output_file}."
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    samples_num = 0
    if is_global_rank_zero():
        output_file = os.path.join(cfg.output_path, f"predictions_all.json")
        logging.info(f"Prediction files are being aggregated in {output_file}.")
        with open(output_file, 'tw', encoding="utf-8") as outf:
            for rank in range(trainer.world_size):
                input_file = os.path.join(cfg.output_path, f"predictions_{rank}.json")
                with open(input_file, 'r', encoding="utf-8") as inpf:
                    lines = inpf.readlines()
                    samples_num += len(lines)
                    outf.writelines(lines)
        logging.info(
            f"Prediction is done for {samples_num} samples in total on all workers and results are aggregated in {output_file}."
        )


if __name__ == '__main__':
    main()
