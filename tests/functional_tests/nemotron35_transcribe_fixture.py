# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Stage the training inputs and record batches from the real training entry point."""

import json
import shutil
import sys
import tarfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import torch
from lhotse.indexing import create_jsonl_index
from lightning.pytorch import Callback
from omegaconf import OmegaConf
from scripts.dataloading.convert_indexes_to_idxpack import main as convert_indexes
from torch.distributed.tensor import DTensor

from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index

SPEAKER_COUNT = 8
TRAINING_STEPS = 4
RESUME_STEP = 2
SERVING_FIXTURE = "speechlm/nemotron35-transcribe-ci-v1"
ENCODER_FIXTURE = "speechlm/nemotron35-transcribe-encoder-ci-v1/parallel-expert-encoder.nemo"


def prepare_training_inputs(work_dir: Path, test_data: Path):
    serving = work_dir / "fixture"
    shutil.copytree(test_data / SERVING_FIXTURE, serving)
    shutil.copyfile(test_data / ENCODER_FIXTURE, work_dir / "encoder.nemo")
    speech = _speech_source(work_dir, serving)
    text = _text_source(work_dir)

    cfg = OmegaConf.load("examples/speechlm2/conf/salm_nemotron.yaml")
    tiny = OmegaConf.load(serving / "model/config.json")
    cfg.model.pretrained_llm = str(serving / "model/llm_backbone")
    cfg.model.tokenizer_path = str(serving / "model")
    cfg.model.pretrained_asr = tiny.pretrained_asr
    cfg.model.pretrained_weights = False
    cfg.model.pe_encoder_path = str(work_dir / "encoder.nemo")
    cfg.model.prompt_format = "nemotron3p5"
    cfg.model.perception.preprocessor = tiny.perception.preprocessor
    cfg.model.perception.preprocessor.normalize = "per_feature"
    cfg.model.perception.encoder = tiny.perception.encoder
    cfg.model.perception.modality_adapter.d_model = tiny.perception.modality_adapter.d_model
    cfg.model.perception.output_dim = tiny.perception.output_dim
    cfg.model.moe_metrics.enabled = False
    cfg.model.lr_scheduler.warmup_steps = 1
    cfg.model.lr_scheduler.max_steps = TRAINING_STEPS

    cfg.trainer.devices = 1
    cfg.trainer.num_nodes = 1
    cfg.trainer.strategy.ep_size = 1
    cfg.trainer.strategy.moe_config = None
    cfg.trainer.max_steps = TRAINING_STEPS
    cfg.trainer.limit_train_batches = 8
    cfg.trainer.log_every_n_steps = 1
    cfg.trainer.enable_progress_bar = False
    for dataset in (cfg.data.train_ds, cfg.data.validation_ds):
        dataset.indexes_root = str(work_dir)
        dataset.index_pack_root = str(work_dir)
    cfg.data.train_ds.num_workers = 1
    cfg.data.train_ds.prefetch_factor = 2
    cfg.data.train_ds.batch_tokens = 768
    cfg.data.train_ds.max_tokens = 512
    cfg.data.train_ds.packing_buffer_size = 8
    cfg.data.train_ds.input_cfg = [speech, text]
    cfg.data.validation_ds.datasets = {
        "dev_asr": {"input_cfg": speech["input_cfg"], "index_pack": speech["index_pack"]}
    }

    cfg.exp_manager.explicit_log_dir = str(work_dir / "continuous")
    cfg.exp_manager.max_time_per_run = None
    cfg.exp_manager.create_wandb_logger = False
    cfg.exp_manager.resume_if_exists = False
    cfg.exp_manager.checkpoint_callback_params.every_n_train_steps = RESUME_STEP
    cfg.exp_manager.checkpoint_callback_params.every_n_epochs = 0
    return cfg


class RecordTraining(Callback):
    """Save the small batches and measurements used by the pytest assertions."""

    def __init__(self, output: str):
        self.output = output
        self.steps = []

    def on_train_start(self, trainer, pl_module):
        self.initial_step = trainer.global_step
        self.projection = _local_tensor(pl_module.perception.proj.weight).detach().clone()
        optimizer = trainer.optimizers[0]
        self.optimizer = type(optimizer).__name__
        self.restored_adam_state = any("exp_avg" in state for state in optimizer.state.values())
        self.diarization_frozen = all(
            not parameter.requires_grad for parameter in pl_module.perception.encoder.diarization_model.parameters()
        )
        self.mel_normalization = pl_module.perception.preprocessor.featurizer.normalize

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.max_gradients = {
            "asr": _max_gradient(pl_module.perception.encoder.asr_encoder),
            "mtp": _max_gradient(pl_module.llm.mtp),
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_keys = (
            "input_ids",
            "loss_mask",
            "text_cu_seqlens",
            "packed_audio_samples",
            "audio_lens",
            "audio_cu_seqlens",
            "spk_targets",
            "spk_target_cu_seqlens",
        )
        self.steps.append(
            {
                "step": trainer.global_step,
                "lr": trainer.optimizers[0].param_groups[0]["lr"],
                "batch": {key: batch[key].detach().cpu().clone() for key in batch_keys if key in batch},
                "max_gradients": self.max_gradients,
                "losses": {
                    name: float(trainer.callback_metrics[name])
                    for name in ("loss", "mtp_loss", "mtp_loss_unscaled/head_1", "mtp_loss_unscaled/head_2")
                },
            }
        )

    def on_train_end(self, trainer, pl_module):
        torch.save(
            {
                "start_step": self.initial_step,
                "optimizer": self.optimizer,
                "restored_adam_state": self.restored_adam_state,
                "projection_changed": not torch.equal(
                    self.projection, _local_tensor(pl_module.perception.proj.weight)
                ),
                "diarization_frozen": self.diarization_frozen,
                "mel_normalization": self.mel_normalization,
                "steps": self.steps,
            },
            self.output,
        )


def _speech_source(work_dir, audio_dir):
    short_wave = audio_dir / "short.wav"
    long_wave = audio_dir / "long.wav"
    rttm = work_dir / "speakers.rttm"
    rttm.write_text(
        "".join(
            f"SPEAKER long 1 {speaker}.000 20.000 <NA> <NA> <spk:{speaker}> <NA> <NA>\n"
            for speaker in range(SPEAKER_COUNT)
        )
    )
    short = {"audio_filepath": "short.wav", "answer": "because you were sleeping"}
    long = {
        "audio_filepath": "long.wav",
        "answer": " ".join(f"<spk:{speaker}> hello" for speaker in range(SPEAKER_COUNT)),
        "rttm_filepath": str(rttm),
    }
    contextual = {
        "audio_filepath": "context.wav",
        "answer": "sleeping",
        "preference_instructions": [
            {
                "prompt": "Transcribe the audio. Names: Rose Princess.",
                "target": "contextual",
                "tags": {"type": "context_asr-fine_context", "target_lang": "en"},
            }
        ],
    }
    # Tar order differs from manifest order to exercise the v3 member ordinals.
    tar_path = work_dir / "speech.tar"
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for row, audio_path in ((contextual, short_wave), (long, long_wave), (short, short_wave)):
            info = sf.info(audio_path)
            row.update(duration=info.duration, sampling_rate=info.samplerate, target_lang="en")
            archive.add(audio_path, arcname=row["audio_filepath"])
    create_tar_index(tar_path, Path(f"{tar_path}.idx"))
    rows = [short, long, contextual]
    return {
        **_index_source(work_dir, "speech", rows, {"type": "nemo_tarred", "tarred_audio_filepaths": str(tar_path)}),
        "type": "lhotse_as_conversation",
        "weight": 0.8,
        "preference_sampling": {"weights": {"context_asr-fine_context": 1.0}, "skip_invalid_target": True},
        "tags": {"system_prompt": "", "override_system_prompt": True},
    }


def _text_source(work_dir):
    row = {
        "id": "multi-turn",
        "conversation": [
            {"sender": "User", "fragments": ["first"]},
            {"sender": "Assistant", "fragments": ["earlier"]},
            {"sender": "User", "fragments": ["second"]},
            {"sender": "Assistant", "fragments": ["later"]},
        ],
    }
    return {
        **_index_source(work_dir, "text", [row], {"type": "nemotron_text_converation"}, path_key="paths"),
        "type": "group",
        "weight": 0.2,
    }


def _index_source(work_dir, name, rows, config, path_key="manifest_filepath"):
    manifest = work_dir / f"{name}.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    create_jsonl_index(manifest)
    blend = work_dir / f"{name}.yaml"
    OmegaConf.save([{**config, path_key: str(manifest)}], blend)
    pack = work_dir / f"{name}.idxpack"
    convert_indexes(["--output", str(pack), str(blend)], standalone_mode=False)
    return {"input_cfg": str(blend), "index_pack": pack.name, "force_finite": False}


def _local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _max_gradient(module):
    maxima = [
        _local_tensor(parameter.grad).detach().float().abs().amax()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return torch.stack(maxima).amax().item() if maxima else 0.0


if __name__ == "__main__":
    from examples.speechlm2 import salm_train

    config = Path(sys.argv[1])
    recorder = RecordTraining(config.with_suffix(".pt"))
    # Attach the test callback directly while exercising the real training entry point.
    with patch.object(salm_train, "Trainer", partial(salm_train.Trainer, callbacks=[recorder])):
        salm_train.train(OmegaConf.load(config))
