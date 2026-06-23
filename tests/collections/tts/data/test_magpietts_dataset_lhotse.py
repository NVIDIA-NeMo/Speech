# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import OmegaConf

from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset


pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000
CODEC_MODEL_SAMPLES_PER_FRAME = 480
CODEC_MODEL_INPUT_SAMPLE_RATE = 24000
FRAME_STACKING_FACTOR = 1
NUM_AUDIO_CODEBOOKS = 8

BPE_TOKENIZER_NAME = "nemotron_bpe"
BPE_TOKENIZER_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
BPE_TOKENIZER_CACHED_PATH = Path("/home/TestData/nvidia--NVIDIA-Nemotron-Nano-9B-v2/")
if BPE_TOKENIZER_CACHED_PATH.exists():
    BPE_TOKENIZER_MODEL = str(BPE_TOKENIZER_CACHED_PATH)


def _seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def _tokenizer_config():
    return OmegaConf.create(
        {
            BPE_TOKENIZER_NAME: {
                "_target_": "AutoTokenizer",
                "pretrained_model": BPE_TOKENIZER_MODEL,
            }
        }
    )


def _memory_temporal_array(values, frame_shift=0.02):
    buffer = BytesIO()
    np.save(buffer, values)
    return TemporalArray(
        array=Array(
            storage_type="memory_npy",
            storage_path="",
            storage_key=buffer.getvalue(),
            shape=list(values.shape),
        ),
        temporal_dim=-1,
        frame_shift=frame_shift,
        start=0,
    )


def _cached_codes(num_codebooks=NUM_AUDIO_CODEBOOKS, num_frames=5, offset=0):
    codes = np.arange(num_codebooks * num_frames, dtype=np.int32).reshape(num_codebooks, num_frames)
    return (codes + offset) % 16


def _single_turn_cutset():
    cut = dummy_cut(
        0,
        duration=0.5,
        recording=dummy_recording(0, duration=0.5, with_data=True, sampling_rate=SAMPLE_RATE),
    )
    cut.target_audio = dummy_recording(10, duration=0.5, with_data=True, sampling_rate=SAMPLE_RATE)
    cut.supervisions = [
        SupervisionSegment(
            id="single-turn",
            recording_id=cut.recording_id,
            start=0.0,
            duration=0.2,
            text="hello",
            language="en",
            speaker="| Language:en Dataset:Unit Speaker:spk |",
            custom={"context_text": "speaker prompt", "normalized_text": "hello"},
        )
    ]
    cut.custom = {
        **(cut.custom or {}),
        "target_codes": _memory_temporal_array(_cached_codes(num_frames=4)),
        "context_codes": _memory_temporal_array(_cached_codes(num_frames=3, offset=3)),
        "tokenizer_names": [BPE_TOKENIZER_NAME],
        "lang": "en",
    }
    return CutSet.from_cuts([cut])


def _multiturn_cutset():
    cut = dummy_cut(
        1,
        duration=0.8,
        recording=dummy_recording(1, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE),
    )
    cut.target_audio = dummy_recording(11, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE)
    cut.supervisions = [
        SupervisionSegment(
            id="turn-user-0",
            recording_id=cut.recording_id,
            start=0.0,
            duration=0.2,
            text="hi",
            language="en",
            speaker="user",
            custom={"context_text": "chat prompt"},
        ),
        SupervisionSegment(
            id="turn-agent-0",
            recording_id=cut.recording_id,
            start=0.3,
            duration=0.2,
            text="hello",
            language="en",
            speaker="assistant",
        ),
        SupervisionSegment(
            id="turn-user-1",
            recording_id=cut.recording_id,
            start=0.55,
            duration=0.1,
            text="ok",
            language="en",
            speaker="user",
        ),
        SupervisionSegment(
            id="turn-agent-1",
            recording_id=cut.recording_id,
            start=0.68,
            duration=0.1,
            text="okay",
            language="en",
            speaker="assistant",
            custom={"reward": 0.5},
        ),
    ]
    cut.custom = {
        **(cut.custom or {}),
        "task": "dialog",
        "target_codes": _memory_temporal_array(_cached_codes(num_frames=8)),
        "source_codes": _memory_temporal_array(_cached_codes(num_frames=8, offset=1)),
        "context_codes": _memory_temporal_array(_cached_codes(num_frames=4, offset=2)),
        "tokenizer_names": [BPE_TOKENIZER_NAME],
        "lang": "en",
    }
    return CutSet.from_cuts([cut])


def _dataset_kwargs():
    return {
        "sample_rate": SAMPLE_RATE,
        "volume_norm": False,
        "codec_model_samples_per_frame": CODEC_MODEL_SAMPLES_PER_FRAME,
        "num_audio_codebooks": NUM_AUDIO_CODEBOOKS,
        "prior_scaling_factor": None,
        "load_cached_codes_if_available": True,
        "dataset_type": "train",
        "load_16khz_audio": False,
        "pad_context_text_to_max_duration": False,
        "context_duration_min": 0.04,
        "context_duration_max": 0.04,
        "use_text_conditioning_tokenizer": True,
        "text_conditioning_tokenizer_name": BPE_TOKENIZER_NAME,
        "tokenizer_config": _tokenizer_config(),
    }


class TestMagpieTTSLhotseDatasets:
    def test_single_turn_dataset_uses_bpe_and_cached_codes(self):
        _seed_everything()
        dataset = MagpieTTSLhotseDataset(**_dataset_kwargs())

        batch = dataset[_single_turn_cutset()]

        assert batch["dataset_names"] == ["Unit"]
        assert batch["languages"] == ["en"]
        assert batch["raw_texts"] == ["hello"]
        assert "audio" not in batch
        assert "context_audio" not in batch
        assert batch["audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 4)
        assert batch["audio_codes_lens"].tolist() == [4]
        assert batch["context_audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 2)
        assert batch["context_audio_codes_lens"].tolist() == [2]
        assert batch["text"].shape[0] == 1
        assert batch["text_lens"].item() > 0
        assert batch["context_text_tokens"].shape[0] == 1
        assert batch["context_text_tokens_lens"].item() > 0
        assert batch["has_text_context"].tolist() == [True]

    def test_multiturn_dataset_uses_bpe_and_cached_codes(self):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "codec_model_input_sample_rate": CODEC_MODEL_INPUT_SAMPLE_RATE,
                "frame_stacking_factor": FRAME_STACKING_FACTOR,
                "source_sample_rate": SAMPLE_RATE,
                "input_roles": ["user"],
                "output_roles": ["assistant"],
                "add_text_bos": False,
            }
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)

        batch = dataset[_multiturn_cutset()]

        assert batch["sample_id"] == ["dummy-mono-cut-0001"]
        assert batch["dataset_names"] == ["unknown"]
        assert batch["languages"] == ["en"]
        assert batch["raw_texts"] == ["hello okay"]
        assert batch["task"] == ["dialog"]
        assert batch["audio"].shape[0] == 1
        assert batch["source_audio"].shape[0] == 1
        assert batch["audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 8)
        assert batch["audio_codes_lens"].tolist() == [8]
        assert batch["source_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 8)
        assert batch["source_codes_lens"].tolist() == [8]
        assert batch["context_audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 2)
        assert batch["context_audio_codes_lens"].tolist() == [2]
        assert batch["source_tokens"].shape[0] == 1
        assert batch["source_token_lens"].item() > 0
        assert batch["text"].shape[0] == 1
        assert batch["text_lens"].item() > 0
        assert batch["agent_mask"].shape == batch["user_mask"].shape
        assert batch["agent_mask_lens"].tolist() == batch["user_mask_lens"].tolist()
        assert batch["agent_mask"].sum().item() > 0
        assert batch["user_mask"].sum().item() > 0
        assert batch["user_audio_turn_splitted"].shape[0] == 2
        assert batch["user_audio_turn_splitted_lens"].tolist() == [4800, 2400]
        assert batch["user_audio_turn_splitted_indices"].shape == (2, 3)
        assert batch["context_text_tokens"].shape[0] == 1
        assert batch["context_text_tokens_lens"].item() > 0
        assert batch["has_text_context"].tolist() == [True]
        torch.testing.assert_close(batch["rewards"], torch.tensor([0.5], device=batch["rewards"].device))
