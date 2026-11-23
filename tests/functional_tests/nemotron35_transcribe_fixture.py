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
"""Build small offline inputs for Nemotron 3.5 Transcribe CI."""

import argparse
import gzip
import io
import json
import tarfile
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from lhotse.indexing import create_jsonl_index
from omegaconf import OmegaConf
from scripts.dataloading.convert_indexes_to_idxpack import main as convert_indexes
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import NemotronHConfig, PreTrainedTokenizerFast

from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index
from tests.collections.asr.test_parallel_expert_encoder import (
    bundle_config,
    toy_packed_diarization_model_cfg,
    toy_transformer_asr_encoder_cfg,
)

AUDIO_TOKEN = "<|audio|>"
SPEAKER_COUNT = 8
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
BACKBONE_WIDTH = 128
ENCODER_WIDTH = 32
LIBRISPEECH_CUTS = "speechlm/lhotse/libri/librispeech_cuts_lower_dev-clean-2-first10.jsonl.gz"
EARLIER_ANSWER = "earlier"
FINAL_ANSWER = "later"
CONTEXT_ANSWER = "contextual"


def prepare_training_inputs(work_dir: Path, test_data: Path, world_size: int) -> None:
    torch.manual_seed(43)
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_backbone(work_dir / "backbone")
    preprocessor, encoder = _write_encoder_bundle(work_dir / "encoder.nemo")
    speech = _write_speech_source(work_dir, test_data)
    text = _write_text_source(work_dir)

    cfg = OmegaConf.load("examples/speechlm2/conf/salm_nemotron.yaml")
    cfg.model.pretrained_llm = str(work_dir / "backbone")
    # Keep the ASR source field for the serving config. The inline mel frontend avoids a checkpoint read.
    cfg.model.pretrained_asr = "nvidia/canary-1b-flash"
    cfg.model.pretrained_weights = False
    cfg.model.pe_encoder_path = str(work_dir / "encoder.nemo")
    cfg.model.prompt_format = "nemotron3p5"
    cfg.model.perception.preprocessor = preprocessor
    cfg.model.perception.encoder = encoder
    cfg.model.perception.modality_adapter.d_model = ENCODER_WIDTH
    cfg.model.perception.output_dim = BACKBONE_WIDTH
    cfg.model.moe_metrics.enabled = False
    cfg.model.lr_scheduler.warmup_steps = 1
    cfg.model.lr_scheduler.max_steps = 4

    cfg.trainer.devices = world_size
    cfg.trainer.num_nodes = 1
    cfg.trainer.strategy.ep_size = world_size
    # These options apply to expert-parallel training. A single rank still uses TE and grouped GEMM.
    cfg.trainer.strategy.moe_config = (
        {"ignore_router_for_ac": True, "reshard_after_forward": True} if world_size > 1 else None
    )
    cfg.trainer.max_steps = 4
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
    cfg.exp_manager.explicit_log_dir = str(work_dir / "train")
    cfg.exp_manager.max_time_per_run = None
    cfg.exp_manager.create_wandb_logger = False
    cfg.exp_manager.resume_if_exists = False
    cfg.exp_manager.checkpoint_callback_params.every_n_train_steps = 2
    cfg.exp_manager.checkpoint_callback_params.every_n_epochs = 0
    OmegaConf.save(cfg, work_dir / "train.yaml")


def _write_backbone(path: Path) -> None:
    special = ["<unk>", "<s>", "<|im_end|>", AUDIO_TOKEN, "<|im_start|>", "<think>", "</think>"]
    words = [
        "system",
        "user",
        "assistant",
        "Transcribe",
        "the",
        "audio.",
        "Names:",
        "Rose",
        "Princess.",
        "because",
        "you",
        "were",
        "sleeping",
        "of",
        "conquering",
        "hello",
        "again",
        "first",
        "second",
        CONTEXT_ANSWER,
        EARLIER_ANSWER,
        FINAL_ANSWER,
    ] + [f"<spk:{speaker}>" for speaker in range(SPEAKER_COUNT)]
    vocab = {word: index for index, word in enumerate(special + words)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<unk>",
        bos_token="<s>",
        eos_token="<|im_end|>",
        extra_special_tokens=special[3:],
        chat_template=(
            "{% for message in messages %}{{ '<|im_start|>' + message.role + '\\n' + message.content + "
            "'<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n<think></think>' }}{% endif %}"
        ),
    )
    tokenizer.save_pretrained(path)
    NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        vocab_size=128,
        hidden_size=BACKBONE_WIDTH,
        intermediate_size=128,
        num_hidden_layers=3,
        layers_block_type=["mamba", "attention", "moe"],
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        mamba_num_heads=4,
        mamba_head_dim=64,
        ssm_state_size=128,
        chunk_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        n_group=1,
        n_groups=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=128,
        max_position_embeddings=2048,
        dtype="bfloat16",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    ).save_pretrained(path)


def _write_encoder_bundle(path: Path):
    encoder = toy_transformer_asr_encoder_cfg()
    encoder.rotary_fraction = 0.5
    diarizer = toy_packed_diarization_model_cfg()
    diarizer.streaming_mode = True
    diarizer.high_resolution = True
    diarizer.output_subsampling_factor = 1
    diarizer.max_num_of_spks = SPEAKER_COUNT
    diarizer.sortformer_modules.num_spks = SPEAKER_COUNT
    cfg = bundle_config(
        asr_encoder_type="transformer",
        asr_encoder_cfg=encoder,
        diarization_model_cfg=diarizer,
        diar_normalize_type="NA",
        freeze_asr=False,
        freeze_diar=True,
        max_speaker_count=SPEAKER_COUNT,
    )
    ParallelExpertEncoderPT(cfg).save_to(str(path))
    preprocessor = diarizer.preprocessor.copy()
    preprocessor.normalize = "per_feature"
    preprocessor.pad_to = 0
    preprocessor.dither = 0.0
    return preprocessor, encoder


def _write_speech_source(work_dir: Path, test_data: Path) -> dict:
    cuts = test_data / LIBRISPEECH_CUTS
    with gzip.open(cuts, "rt") as stream:
        entries = list(islice(map(json.loads, stream), 2))
    audio = []
    for entry in entries:
        source = entry["recording"]["sources"][0]["source"]
        waveform, sample_rate = sf.read(cuts.parent / source, dtype="float32")
        assert sample_rate == SAMPLE_RATE and waveform.ndim == 1
        audio.append(waveform)

    # Cross the production 30 s encoder window with a short second window.
    long_audio = np.concatenate((audio[0], audio[1], audio[0]))[: (CHUNK_SECONDS * SAMPLE_RATE + 1281)]
    rttm = work_dir / "speakers.rttm"
    speaker_tags = [f"<spk:{speaker}>" for speaker in range(SPEAKER_COUNT)]
    # Synthetic permutation-resolved RTTM labels exercise every speaker column with overlap.
    rttm.write_text(
        "".join(
            f"SPEAKER long 1 {speaker}.000 20.000 <NA> <NA> {tag} <NA> <NA>\n"
            for speaker, tag in enumerate(speaker_tags)
        )
    )
    rows = [
        {
            "audio_filepath": "short.wav",
            "answer": "because you were sleeping",
        },
        {
            "audio_filepath": "long.wav",
            "answer": " ".join(f"{tag} hello" for tag in speaker_tags),
            "rttm_filepath": str(rttm),
        },
        {
            "audio_filepath": "context.wav",
            "answer": "because you were sleeping",
            "preference_instructions": [
                {"prompt": "Transcribe the audio.", "target": "sleeping", "tags": {"type": "tn-pnc"}},
                {
                    "prompt": "Transcribe the audio. Names: Rose Princess.",
                    "target": CONTEXT_ANSWER,
                    "tags": {"type": "context_asr-fine_context", "target_lang": "en"},
                },
            ],
        },
    ]
    tar_path = work_dir / "speech.tar"
    # Different order from the manifest exercises the idxpack v3 ordinal map.
    with tarfile.open(tar_path, "w") as archive:
        for row, waveform in zip(reversed(rows), (audio[1], long_audio, audio[0])):
            row["duration"] = len(waveform) / SAMPLE_RATE
            row["sampling_rate"] = SAMPLE_RATE
            row["target_lang"] = "en"
            buffer = io.BytesIO()
            sf.write(buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            member = tarfile.TarInfo(row["audio_filepath"])
            member.size = buffer.tell()
            buffer.seek(0)
            archive.addfile(member, buffer)
    create_tar_index(tar_path, Path(f"{tar_path}.idx"))
    # Also keep audio for the serving smoke without unpacking the training tar.
    sf.write(work_dir / "short.wav", audio[0], SAMPLE_RATE)
    sf.write(work_dir / "long.wav", long_audio, SAMPLE_RATE)
    return {
        **_index_source(work_dir, "speech", rows, {"type": "nemo_tarred", "tarred_audio_filepaths": str(tar_path)}),
        "type": "lhotse_as_conversation",
        "weight": 0.8,
        "preference_sampling": {
            "weights": {"context_asr-fine_context": 1.0},
            "skip_invalid_target": True,
        },
        "tags": {"system_prompt": "", "override_system_prompt": True},
    }


def _write_text_source(work_dir: Path) -> dict:
    return {
        **_index_source(
            work_dir,
            "text",
            [
                {
                    "id": "multi-turn",
                    "conversation": [
                        {"sender": "User", "fragments": ["first"]},
                        {"sender": "Assistant", "fragments": [EARLIER_ANSWER]},
                        {"sender": "User", "fragments": ["second"]},
                        {"sender": "Assistant", "fragments": [FINAL_ANSWER]},
                    ],
                }
            ],
            {"type": "nemotron_text_converation"},
            path_key="paths",
        ),
        "type": "group",
        "weight": 0.2,
    }


def _index_source(work_dir: Path, name: str, rows: list[dict], config: dict, path_key="manifest_filepath") -> dict:
    manifest = work_dir / f"{name}.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    create_jsonl_index(manifest)
    blend = work_dir / f"{name}.yaml"
    OmegaConf.save(OmegaConf.create([{**config, path_key: str(manifest)}]), blend)
    pack = work_dir / f"{name}.idxpack"
    convert_indexes(["--output", str(pack), str(blend)], standalone_mode=False)
    return {"input_cfg": str(blend), "index_pack": pack.name, "force_finite": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("test_data", type=Path)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    prepare_training_inputs(args.work_dir, args.test_data, args.world_size)
