# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
import os
import random
from io import BytesIO

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from lhotse import CutSet, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import OmegaConf

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import (
    MagpieTTSLhotseMultiturnDataset,
    collate_phoneme_channel,
    collate_token_channel,
)
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p


pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000
CODEC_MODEL_SAMPLES_PER_FRAME = 480
CODEC_MODEL_INPUT_SAMPLE_RATE = 24000
FRAME_STACKING_FACTOR = 1
NUM_AUDIO_CODEBOOKS = 8

BPE_TOKENIZER_NAME = "nemotron_bpe"
BPE_TOKENIZER_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
try:
    # Attempt to resolve local cache for CI tests
    BPE_TOKENIZER_MODEL = snapshot_download(BPE_TOKENIZER_MODEL, local_files_only=True)
except LocalEntryNotFoundError:
    # For local tests, can call into HF servers to download as needed
    pass


class _FakeIPATokenizer:
    pad = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text):
        return [7, 8] if text else []


class _FakeTextTokenizer:
    tokens = list(range(100))
    pad = 0

    def __init__(self):
        self.encoded_texts = []

    def encode(self, text, tokenizer_name):
        self.encoded_texts.append(text)
        return [10 + len(text)]


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


def _single_turn_cutset(text="hello", normalized_text="hello"):
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
            text=text,
            language="en",
            speaker="| Language:en Dataset:Unit Speaker:spk |",
            custom={"context_text": "speaker prompt", "normalized_text": normalized_text},
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
            custom={"ipa": "həloʊ", "ipa_alignment": [[0, 5, "hello", "həloʊ"]]},
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


def _raw_audio_cutset(target_audio_duration, text="hello"):
    # No "target_codes" custom field, so __getitem__ takes the raw-audio branch regardless of
    # load_cached_codes_if_available.
    cut = dummy_cut(
        2,
        duration=target_audio_duration,
        recording=dummy_recording(2, duration=target_audio_duration, with_data=True, sampling_rate=SAMPLE_RATE),
    )
    cut.target_audio = dummy_recording(20, duration=target_audio_duration, with_data=True, sampling_rate=SAMPLE_RATE)
    cut.supervisions = [
        SupervisionSegment(
            id="raw-audio",
            recording_id=cut.recording_id,
            start=0.0,
            duration=0.2,
            text=text,
            language="en",
            speaker="| Language:en Dataset:Unit Speaker:spk |",
            custom={"normalized_text": text},
        )
    ]
    cut.custom = {
        **(cut.custom or {}),
        "context_codes": _memory_temporal_array(_cached_codes(num_frames=3, offset=3)),
        "tokenizer_names": [BPE_TOKENIZER_NAME],
        "lang": "en",
    }
    return CutSet.from_cuts([cut])


def _multiturn_fallback_cut(with_target_codes: bool):
    cut = dummy_cut(
        2,
        duration=0.8,
        recording=dummy_recording(2, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE),
    )
    cut.target_audio = dummy_recording(12, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE)
    cut.supervisions = [
        SupervisionSegment(
            id="fallback-agent-turn",
            recording_id=cut.recording_id,
            start=0.2,
            duration=0.3,
            text="fallback context",
            language="en",
            speaker="assistant",
        )
    ]
    if with_target_codes:
        cut.custom = {
            **(cut.custom or {}),
            "target_codes": _memory_temporal_array(_cached_codes(num_frames=40)),
        }
    return cut


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


class TestMagpieTTSDataset:
    @pytest.mark.parametrize(
        "load_normalized_text_percent,expected_text",
        [(1.0, "july fifteenth"), (0.0, "July 15th")],
    )
    def test_text_selection(self, tmp_path, load_normalized_text_percent, expected_text):
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": "unused.wav",
                    "duration": 1.0,
                    "text": "July 15th",
                    "normalized_text": "july fifteenth",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = MagpieTTSDataset(
            dataset_meta={"test": {"manifest_path": str(manifest_path), "audio_dir": str(tmp_path)}},
            sample_rate=22050,
            codec_model_samples_per_frame=1024,
            eos_id=1,
            num_audio_codebooks=8,
            load_normalized_text_percent=load_normalized_text_percent,
        )

        assert dataset.data_samples[0].text == expected_text

    @pytest.mark.parametrize("load_normalized_text_percent", [-0.1, 1.1])
    def test_load_normalized_text_percent_validation(self, load_normalized_text_percent):
        with pytest.raises(ValueError, match="load_normalized_text_percent"):
            MagpieTTSDataset(
                dataset_meta={},
                sample_rate=22050,
                codec_model_samples_per_frame=1024,
                eos_id=1,
                num_audio_codebooks=8,
                load_normalized_text_percent=load_normalized_text_percent,
            )


def _multiturn_dataset_kwargs():
    kwargs = _dataset_kwargs()
    kwargs.update(
        {
            "codec_model_input_sample_rate": CODEC_MODEL_INPUT_SAMPLE_RATE,
            "frame_stacking_factor": FRAME_STACKING_FACTOR,
        }
    )
    return kwargs


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

    @pytest.mark.parametrize(
        "target_audio_duration,expected_audio_len",
        [(0.5, 12000), (0.51, 12480)],
    )
    def test_raw_audio_target_padding_is_frame_aligned(self, target_audio_duration, expected_audio_len):
        """MagpieTTSLhotseDataset's raw-audio branch must not pad an already frame-aligned target."""
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update({"load_cached_codes_if_available": False, "use_text_conditioning_tokenizer": False})
        dataset = MagpieTTSLhotseDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.pad_id = dataset.text_tokenizer.pad

        batch = dataset[_raw_audio_cutset(target_audio_duration)]

        assert batch["audio_lens"].tolist() == [expected_audio_len]

    def test_multiturn_pronunciation_control_only_changes_target_turns(self):
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
                "use_text_conditioning_tokenizer": False,
                "enable_phoneme_text_input": True,
                "partial_phoneme_text_prob": 1.0,
                "partial_phoneme_portion_min": 1.0,
                "partial_phoneme_portion_max": 1.0,
            }
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.cfg_unk_token_id = dataset.bos_id + 2
        dataset.interruption_token_id = dataset.bos_id + 3
        dataset.pad_id = dataset.text_tokenizer.pad
        dataset.text_phoneme_token_offset = dataset.bos_id + 4

        batch = dataset[_multiturn_cutset()]

        target_tokens = batch["text"][0]
        source_tokens = batch["source_tokens"][0]
        assert target_tokens[15:17].tolist() == [
            dataset.text_phoneme_token_offset + 7,
            dataset.text_phoneme_token_offset + 8,
        ]
        assert torch.all(source_tokens < dataset.text_phoneme_token_offset)
        assert target_tokens[25].item() == dataset.interruption_token_id

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

    @pytest.mark.parametrize(
        "load_normalized_text_percent,expected_text",
        [(1.0, "july fifteenth"), (0.0, "July 15th")],
    )
    def test_single_turn_text_selection(self, load_normalized_text_percent, expected_text):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs["load_normalized_text_percent"] = load_normalized_text_percent
        dataset = MagpieTTSLhotseDataset(**kwargs)

        batch = dataset[_single_turn_cutset(text="July 15th", normalized_text="july fifteenth")]

        assert batch["raw_texts"] == [expected_text]

    @pytest.mark.parametrize(
        "load_normalized_text_percent,expected_texts",
        [
            (
                1.0,
                [
                    "normalized assistant greeting",
                    "normalized assistant acknowledgement",
                    "normalized user greeting",
                    "normalized user acknowledgement",
                ],
            ),
            (0.0, ["hello", "okay", "hi", "ok"]),
        ],
    )
    def test_multiturn_text_selection(self, load_normalized_text_percent, expected_texts):
        _seed_everything()
        cuts = _multiturn_cutset()
        normalized_texts = [
            "normalized user greeting",
            "normalized assistant greeting",
            "normalized user acknowledgement",
            "normalized assistant acknowledgement",
        ]
        for supervision, normalized_text in zip(next(iter(cuts)).supervisions, normalized_texts):
            supervision.custom = {**(supervision.custom or {}), "normalized_text": normalized_text}

        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "codec_model_input_sample_rate": CODEC_MODEL_INPUT_SAMPLE_RATE,
                "frame_stacking_factor": FRAME_STACKING_FACTOR,
                "source_sample_rate": SAMPLE_RATE,
                "input_roles": ["user"],
                "output_roles": ["assistant"],
                "add_text_bos": False,
                "use_text_conditioning_tokenizer": False,
                "load_normalized_text_percent": load_normalized_text_percent,
            }
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.cfg_unk_token_id = dataset.bos_id + 2
        dataset.interruption_token_id = dataset.bos_id + 3
        dataset.pad_id = dataset.text_tokenizer.pad

        batch = dataset[cuts]

        assert batch["text"].shape[0] == 1
        assert dataset.text_tokenizer.encoded_texts == expected_texts

    @pytest.mark.parametrize("load_normalized_text_percent", [-0.1, 1.1])
    @pytest.mark.parametrize("dataset_class", [MagpieTTSLhotseDataset, MagpieTTSLhotseMultiturnDataset])
    def test_load_normalized_text_percent_validation(self, dataset_class, load_normalized_text_percent):
        kwargs = _dataset_kwargs()
        kwargs["load_normalized_text_percent"] = load_normalized_text_percent

        with pytest.raises(ValueError, match="load_normalized_text_percent"):
            dataset_class(**kwargs)

    def test_multiturn_fallback_uses_cached_target_turn_codes_when_available(self):
        dataset = MagpieTTSLhotseMultiturnDataset(**_multiturn_dataset_kwargs())
        cut = _multiturn_fallback_cut(with_target_codes=True)

        context_audio, context_audio_codes = dataset._load_fallback_context(cut, context_text=None)

        assert context_audio is None
        expected = torch.from_numpy(_cached_codes(num_frames=40))[:, 10:25].T
        torch.testing.assert_close(context_audio_codes, expected)

    def test_multiturn_fallback_uses_target_turn_audio_when_cached_codes_are_unavailable(self):
        dataset = MagpieTTSLhotseMultiturnDataset(**_multiturn_dataset_kwargs())
        cut = _multiturn_fallback_cut(with_target_codes=False)

        context_audio, context_audio_codes = dataset._load_fallback_context(cut, context_text=None)

        expected = (
            cut.resample(SAMPLE_RATE, recording_field="target_audio")
            .truncate(offset=0.2, duration=0.3)
            .load_custom("target_audio")
            .squeeze(0)
        )
        assert context_audio_codes is None
        torch.testing.assert_close(context_audio, torch.from_numpy(expected))


def _dataset_with_fake_tokenizers(dataset_class):
    kwargs = _dataset_kwargs()
    kwargs.update(
        {
            "use_text_conditioning_tokenizer": False,
            "context_duration_min": 0.1,
            "context_duration_max": 0.6,
        }
    )
    if dataset_class is MagpieTTSLhotseMultiturnDataset:
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
    dataset = dataset_class(**kwargs)
    dataset.enable_checkpointed_rng()
    dataset.text_tokenizer = _FakeTextTokenizer()
    dataset.bos_id = len(dataset.text_tokenizer.tokens)
    dataset.eos_id = dataset.bos_id + 1
    dataset.cfg_unk_token_id = dataset.bos_id + 2
    dataset.interruption_token_id = dataset.bos_id + 3
    dataset.pad_id = dataset.text_tokenizer.pad
    return dataset


def _cutset_with_long_context(dataset_class):
    cuts = _multiturn_cutset() if dataset_class is MagpieTTSLhotseMultiturnDataset else _single_turn_cutset()
    cut = next(iter(cuts))
    # 40 codec frames (0.8 s) of context, so that context slicing has to pick a random duration and offset.
    cut.context_codes = _memory_temporal_array(_cached_codes(num_frames=40))
    return cuts


def _assert_batches_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, msg=key)
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize("dataset_class", [MagpieTTSLhotseDataset, MagpieTTSLhotseMultiturnDataset])
class TestMagpieTTSLhotseDatasetRNG:
    def test_restored_rng_state_reproduces_batches(self, dataset_class):
        cuts = _cutset_with_long_context(dataset_class)
        dataset = _dataset_with_fake_tokenizers(dataset_class)
        dataset.rng.seed(0)
        _ = dataset[cuts]
        state = dataset.state_dict()
        expected = [dataset[cuts] for _ in range(3)]

        # A resumed process does not share the global RNG state of the original one.
        random.seed(1234)
        np.random.seed(1234)
        restored = _dataset_with_fake_tokenizers(dataset_class)
        restored.load_state_dict(state)

        for expected_batch in expected:
            _assert_batches_equal(restored[cuts], expected_batch)

    def test_context_slice_differs_when_cut_is_read_again(self, dataset_class):
        cuts = _cutset_with_long_context(dataset_class)
        dataset = _dataset_with_fake_tokenizers(dataset_class)
        dataset.rng.seed(0)

        contexts = [tuple(dataset[cuts]["context_audio_codes"].flatten().tolist()) for _ in range(5)]

        assert len(set(contexts)) > 1

    def test_global_rng_is_used_unless_checkpointed_rng_is_enabled(self, dataset_class):
        # Legacy behaviour: without use_stateful_dataloader the draws follow the global ``random`` module.
        cuts = _cutset_with_long_context(dataset_class)
        dataset = _dataset_with_fake_tokenizers(dataset_class)
        dataset._checkpointed_rng_enabled = False

        random.seed(0)
        first = dataset[cuts]["context_audio_codes"]
        random.seed(0)
        second = dataset[cuts]["context_audio_codes"]

        assert dataset.rng is random
        torch.testing.assert_close(first, second)


def _normalized_multiturn_cutset():
    cuts = _multiturn_cutset()
    for supervision in next(iter(cuts)).supervisions:
        supervision.custom = {**(supervision.custom or {}), "normalized_text": f"normalized {supervision.text}"}
    return cuts


def _collate_tokens(**kwargs):
    tokens, lens = collate_token_channel(
        _normalized_multiturn_cutset(),
        _FakeTextTokenizer(),
        0.04,
        roles={"assistant"},
        pad_id=0,
        eos_id=1,
        bos_id=2,
        interruption_token_id=3,
        load_normalized_text_percent=0.5,
        **kwargs,
    )
    return tokens.tolist(), lens.tolist()


def _collate_phonemes(**kwargs):
    tokens, lens, dropout = collate_phoneme_channel(
        _multiturn_cutset(),
        _FakeIPATokenizer(),
        0.04,
        roles={"assistant"},
        ignore_phoneme_languages=[],
        phoneme_turn_dropout_batch_prob=0.5,
        phoneme_turn_dropout_turn_prob=0.5,
        apply_turn_dropout=True,
        **kwargs,
    )
    return tokens.tolist(), lens.tolist(), dropout.tolist()


@pytest.mark.parametrize("collate", [_collate_tokens, _collate_phonemes])
def test_multiturn_channel_helpers_default_to_the_global_rng(collate):
    random.seed(0)
    without_rng = [collate() for _ in range(8)]
    random.seed(0)
    with_global_module = [collate(rng=random) for _ in range(8)]

    assert without_rng == with_global_module


@pytest.mark.parametrize("collate", [_collate_tokens, _collate_phonemes])
def test_multiturn_channel_helpers_draw_from_injected_rng(collate):
    random.seed(0)
    first = [collate(rng=random.Random(5)) for _ in range(8)]
    random.seed(1)
    second = [collate(rng=random.Random(5)) for _ in range(8)]
    many = [collate(rng=random.Random(seed)) for seed in range(16)]

    assert first == second
    # The draws matter: different generators give different channels.
    assert len({str(out) for out in many}) > 1


G2P_PHONEME_DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g2p", "phoneme_dict", "test_dict_en.txt"
)
G2P_TEXT = " ".join(["hello world nvidia jones airport"] * 4)


def _g2p_tokenizer():
    """A real IPA tokenizer whose G2P keeps each word as graphemes with probability 0.5."""
    g2p = IpaG2p(
        G2P_PHONEME_DICT_PATH,
        locale="en-US",
        use_chars=True,
        phoneme_probability=0.5,
        apply_to_oov_word=lambda x: x,
    )
    return AggregatedTTSTokenizer([IPATokenizer(g2p=g2p, locale="en-US")], ["english_phoneme"])


def _single_turn_dataset_with_g2p():
    dataset = _dataset_with_fake_tokenizers(MagpieTTSLhotseDataset)
    dataset.text_tokenizer = _g2p_tokenizer()
    dataset.bos_id = len(dataset.text_tokenizer.tokens)
    dataset.eos_id = dataset.bos_id + 1
    dataset.pad_id = dataset.text_tokenizer.pad
    return dataset


def _g2p_cutset(dataset_class=MagpieTTSLhotseDataset):
    cuts = _cutset_with_long_context(dataset_class)
    cut = next(iter(cuts))
    cut.tokenizer_names = ["english_phoneme"]
    # The multiturn token channel is frame-aligned, so its turns only fit a short text.
    text = G2P_TEXT if dataset_class is MagpieTTSLhotseDataset else "hello jones"
    for supervision in cut.supervisions:
        supervision.text = text
        supervision.custom = {k: v for k, v in (supervision.custom or {}).items() if k not in ("ipa", "ipa_alignment")}
        supervision.normalized_text = text
    return cuts


def _multiturn_dataset_with_g2p():
    dataset = _dataset_with_fake_tokenizers(MagpieTTSLhotseMultiturnDataset)
    dataset.text_tokenizer = _g2p_tokenizer()
    dataset.bos_id = len(dataset.text_tokenizer.tokens)
    dataset.eos_id = dataset.bos_id + 1
    dataset.cfg_unk_token_id = dataset.bos_id + 2
    dataset.interruption_token_id = dataset.bos_id + 3
    dataset.pad_id = dataset.text_tokenizer.pad
    return dataset


_G2P_DATASETS = {
    MagpieTTSLhotseDataset: _single_turn_dataset_with_g2p,
    MagpieTTSLhotseMultiturnDataset: _multiturn_dataset_with_g2p,
}


@pytest.mark.parametrize("dataset_class", [MagpieTTSLhotseDataset, MagpieTTSLhotseMultiturnDataset])
class TestMagpieTTSDatasetG2pRNG:
    def test_lhotse_dataset_restores_g2p_draws(self, dataset_class):
        cuts = _g2p_cutset(dataset_class)
        make_dataset = _G2P_DATASETS[dataset_class]
        dataset = make_dataset()
        dataset.rng.seed(0)  # a fixed sequence that picks both pronunciations
        _ = dataset[cuts]
        state = dataset.state_dict()
        expected = [dataset[cuts]["text"] for _ in range(3)]

        # A resumed process starts with different global and torch RNG states.
        random.seed(1234)
        torch.manual_seed(1234)
        restored = make_dataset()
        restored.load_state_dict(state)

        for expected_text in expected:
            torch.testing.assert_close(restored[cuts]["text"], expected_text)
        # phoneme_probability=0.5 makes the tokens vary between reads.
        assert len({tuple(text.flatten().tolist()) for text in expected}) > 1

    def test_g2p_keeps_its_own_generator_unless_checkpointed_rng_is_enabled(self, dataset_class):
        cuts = _g2p_cutset(dataset_class)
        dataset = _G2P_DATASETS[dataset_class]()
        dataset._checkpointed_rng_enabled = False
        g2p = dataset.text_tokenizer.tokenizers["english_phoneme"].g2p
        own_rng = g2p.rng

        _ = dataset[cuts]

        assert g2p.rng is own_rng


def _g2p_tokenizer_config():
    """Config for a lazily constructed real IPA tokenizer whose G2P keeps words as graphemes with probability 0.5."""
    return OmegaConf.create(
        {
            "english_phoneme": {
                "_target_": "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer",
                "locale": "en-US",
                "g2p": {
                    "_target_": "nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p",
                    "phoneme_dict": G2P_PHONEME_DICT_PATH,
                    "locale": "en-US",
                    "use_chars": True,
                    "phoneme_probability": 0.5,
                },
            }
        }
    )


def _g2p_loader_cuts(dataset_class, num_cuts=6):
    from lhotse.utils import fastcopy

    template = next(iter(_g2p_cutset(dataset_class)))
    return CutSet.from_cuts(fastcopy(template, id=f"g2p-{i}") for i in range(num_cuts))


def _g2p_loader(dataset_class, mode):
    from unittest.mock import patch

    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

    kwargs = _dataset_kwargs()
    kwargs.update(
        {
            "use_text_conditioning_tokenizer": False,
            "context_duration_min": 0.1,
            "context_duration_max": 0.6,
            "tokenizer_config": _g2p_tokenizer_config(),
        }
    )
    if dataset_class is MagpieTTSLhotseMultiturnDataset:
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
    config = OmegaConf.create(
        {
            "batch_size": 2,
            "shuffle": True,
            "seed": 0,
            "shard_seed": 0,
            "num_workers": 0,
            "use_stateful_dataloader": True,
            "force_map_dataset": mode == "map",
            "force_iterable_dataset": mode == "iterable",
        }
    )
    # Note: the tokenizer is not set here; the dataset builds it lazily on its first read.
    with patch(
        "nemo.collections.common.data.lhotse.dataloader.read_cutset_from_config",
        return_value=(_g2p_loader_cuts(dataset_class), False),
    ):
        return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=dataset_class(**kwargs))


def _summarize_g2p_batch(batch):
    return (batch["text"].tolist(), batch["text_lens"].tolist(), batch["context_audio_codes"].tolist())


@pytest.mark.parametrize("restore_into", ["fresh_loader", "initialized_loader"])
@pytest.mark.parametrize("mode", ["map", "iterable"])
@pytest.mark.parametrize("dataset_class", [MagpieTTSLhotseDataset, MagpieTTSLhotseMultiturnDataset])
def test_stateful_loader_restores_lazily_built_g2p_draws(dataset_class, mode, restore_into):
    torch.manual_seed(0)
    uninterrupted = [_summarize_g2p_batch(batch) for batch in _g2p_loader(dataset_class, mode)]
    loader = _g2p_loader(dataset_class, mode)
    it = iter(loader)
    _ = next(it)
    state = loader.state_dict()

    # A resumed process starts with different global and torch RNG states.
    random.seed(1234)
    torch.manual_seed(1234)
    if restore_into == "fresh_loader":
        # The checkpoint is restored before the dataset has built its tokenizer.
        resumed_loader = _g2p_loader(dataset_class, mode)
    else:
        # The checkpoint is restored into a loader whose dataset already built (and bound) its tokenizer.
        resumed_loader = _g2p_loader(dataset_class, mode)
        _ = next(iter(resumed_loader))
    resumed_loader.load_state_dict(state)
    resumed = [_summarize_g2p_batch(batch) for batch in resumed_loader]

    assert len(uninterrupted) == 3
    assert resumed == uninterrupted[1:]
    # phoneme_probability=0.5 chooses different pronunciations for the identical texts.
    texts = {tuple(row) for text, _, _ in uninterrupted for row in text}
    assert len(texts) > 1
