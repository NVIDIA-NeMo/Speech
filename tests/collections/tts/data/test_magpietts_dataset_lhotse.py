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

from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
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
    tokenizer_pad_ids = {BPE_TOKENIZER_NAME: 0}

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

    @pytest.mark.parametrize('normalized_text', [None, 'one thousand'])
    def test_map_getitem_and_collate_keep_normalized_cer_reference(self, tmp_path, normalized_text):
        codes_path = tmp_path / 'codes.pt'
        torch.save(torch.from_numpy(_cached_codes(num_frames=4)), codes_path)
        entry = {
            'audio_filepath': 'unused.wav',
            'target_audio_codes_path': str(codes_path),
            'context_audio_codes_path': str(codes_path),
            'duration': 0.08,
            'text': '1000',
        }
        if normalized_text is not None:
            entry['normalized_text'] = normalized_text
        manifest_path = tmp_path / 'manifest.jsonl'
        manifest_path.write_text(json.dumps(entry) + '\n', encoding='utf-8')
        dataset = MagpieTTSDataset(
            dataset_meta={'test': {'manifest_path': str(manifest_path), 'audio_dir': str(tmp_path)}},
            sample_rate=SAMPLE_RATE,
            codec_model_samples_per_frame=CODEC_MODEL_SAMPLES_PER_FRAME,
            eos_id=101,
            num_audio_codebooks=NUM_AUDIO_CODEBOOKS,
            load_normalized_text_percent=0.0,
            load_16khz_audio=False,
            context_duration_min=0.08,
            context_duration_max=0.08,
        )
        dataset.text_tokenizer = _FakeTextTokenizer()

        example = dataset[0]
        legacy_example = {key: value for key, value in example.items() if key != 'cer_reference_text'}
        batch = dataset.collate_fn([example, legacy_example])

        assert example['raw_text'] == '1000' and example['tokens'].tolist() == [14, 101]
        assert example['cer_reference_text'] == (normalized_text or '1000')
        assert batch['raw_texts'] == ['1000', '1000']
        assert batch['cer_reference_texts'] == [normalized_text or '1000', '1000']

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

    @pytest.mark.parametrize("default,override", [(1.0, 0.0), (0.0, 1.0)])
    @pytest.mark.parametrize("dataset_type", ["train", "test"])
    def test_multiturn_yaml_text_selection_override(self, default, override, dataset_type):
        from nemo.collections.common.data.lhotse.cutset import attach_tags

        _seed_everything()
        normal = next(iter(_multiturn_cutset()))
        tagged = next(iter(_multiturn_cutset()))
        tagged.id += "-tagged"
        tags = OmegaConf.create({"task": "duplex", "load_normalized_text_percent": override, "acoustic_loss": False})
        attach_tags(tagged, tags)
        for cut in (normal, tagged):
            for supervision in cut.supervisions:
                supervision.normalized_text = f"normalized {supervision.text}"
                supervision.ipa = "test ipa"

        kwargs = _multiturn_dataset_kwargs()
        kwargs.update(
            input_roles=["user"],
            output_roles=["assistant"],
            add_text_bos=False,
            use_text_conditioning_tokenizer=False,
            load_normalized_text_percent=default,
            dataset_type=dataset_type,
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.cfg_unk_token_id = dataset.bos_id + 2
        dataset.interruption_token_id = dataset.bos_id + 3
        dataset.pad_id = dataset.text_tokenizer.pad
        batch = dataset[CutSet.from_cuts([normal, tagged])]

        raw = ["hello", "okay", "hi", "ok"]
        # Target turns for both cuts are encoded before source turns.
        expected = []
        for role_texts in (raw[:2], raw[2:]):
            for probability in (default, override):
                expected.extend(f"normalized {text}" if probability else text for text in role_texts)
        assert dataset.text_tokenizer.encoded_texts == expected
        assert batch["acoustic_loss_mask"].tolist() == [True, False]

    @pytest.mark.parametrize("override", [-0.1, 1.1, float("nan"), float("inf"), "0.0", None, True])
    def test_multiturn_invalid_text_selection_override(self, override):
        from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import build_token_channel

        cut = next(iter(_multiturn_cutset()))
        cut.custom["load_normalized_text_percent"] = override
        with pytest.raises(ValueError, match="load_normalized_text_percent"):
            build_token_channel(cut, _FakeTextTokenizer(), 0.1, roles={"assistant"})

    @pytest.mark.parametrize("draw,expected", [(0.2, "normalized hello"), (0.4, "hello")])
    def test_multiturn_fractional_text_selection_override(self, monkeypatch, draw, expected):
        from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import build_token_channel

        cut = next(iter(_multiturn_cutset()))
        cut.load_normalized_text_percent = 0.25
        for supervision in cut.supervisions:
            supervision.normalized_text = f"normalized {supervision.text}"
        monkeypatch.setattr("nemo.collections.tts.parts.utils.tts_dataset_utils.random.random", lambda: draw)
        tokenizer = _FakeTextTokenizer()
        build_token_channel(cut, tokenizer, 0.1, roles={"assistant"}, add_text_bos=False)
        assert tokenizer.encoded_texts[0] == expected

    def test_multiturn_missing_normalized_text_falls_back_to_raw(self):
        from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import build_token_channel

        cut = next(iter(_multiturn_cutset()))
        cut.load_normalized_text_percent = 1.0
        tokenizer = _FakeTextTokenizer()
        build_token_channel(cut, tokenizer, 0.1, roles={"assistant"}, add_text_bos=False)
        assert tokenizer.encoded_texts == ["hello", "okay"]

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


class TestMagpieTTSLhotseChallengingText:
    @staticmethod
    def _dataset(**overrides):
        kwargs = _dataset_kwargs()
        kwargs.update(context_duration_min=0.06, context_duration_max=0.06)
        kwargs.update(overrides)
        dataset = MagpieTTSLhotseDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.pad_id = dataset.text_tokenizer.pad
        dataset.text_phoneme_token_offset = dataset.eos_id + 2
        return dataset

    def test_disabled_replacement_preserves_behavior_and_worker_rng(self, tmp_path):
        path = tmp_path / 'challenging.txt'
        path.write_text('repeat one two three\n', encoding='utf-8')
        baseline = self._dataset()
        augmented = self._dataset(challenging_texts_path=str(path), challenging_text_replacement_prob=0.0)
        cuts = _single_turn_cutset(text='invoice $10', normalized_text='invoice ten dollars')
        random.seed(19)
        expected = baseline[cuts]
        expected_rng = random.getstate()
        random.seed(19)
        actual = augmented[cuts]

        assert random.getstate() == expected_rng
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected[key])
            else:
                assert value == expected[key]

    def test_replacement_changes_input_and_reward_reference_preserving_context(self, tmp_path):
        replacement = 'Read A-1000, then repeat A-1000.'
        path = tmp_path / 'challenging.txt'
        path.write_text(replacement + '\n', encoding='utf-8')
        cuts = _single_turn_cutset(text='invoice $10', normalized_text='invoice ten dollars')
        baseline = self._dataset()[cuts]
        dataset = self._dataset(challenging_texts_path=str(path), challenging_text_replacement_prob=1.0)

        batch = dataset[cuts]

        assert batch['raw_texts'] == [replacement]
        assert batch['cer_reference_texts'] == [replacement]
        assert batch['text'][0].tolist() == [10 + len(replacement), dataset.eos_id]
        assert replacement in dataset.text_tokenizer.encoded_texts
        assert 'invoice ten dollars' not in dataset.text_tokenizer.encoded_texts
        for key in ('audio_codes', 'audio_codes_lens', 'context_audio_codes', 'context_audio_codes_lens'):
            torch.testing.assert_close(batch[key], baseline[key])
        torch.testing.assert_close(batch['context_text_tokens'], baseline['context_text_tokens'])
        assert batch['dataset_names'] == baseline['dataset_names'] == ['Unit']
        assert batch['languages'] == baseline['languages'] == ['en']
        supervision = next(iter(cuts)).supervisions[0]
        assert supervision.text == 'invoice $10' and supervision.normalized_text == 'invoice ten dollars'
        assert supervision.context_text == 'speaker prompt'

    def test_replacement_uses_seeded_worker_rng(self, tmp_path):
        path = tmp_path / 'challenging.txt'
        prompts = ['repeat ten', 'read twenty twice', 'say thirty three three']
        path.write_text('\n'.join(prompts) + '\n', encoding='utf-8')
        dataset = self._dataset(challenging_texts_path=str(path), challenging_text_replacement_prob=1.0)
        cuts = _single_turn_cutset()
        random.seed(42)
        expected = [dataset[cuts]['raw_texts'][0] for _ in range(8)]
        random.seed(42)
        actual = [dataset[cuts]['raw_texts'][0] for _ in range(8)]

        assert actual == expected
        assert set(actual).issubset(prompts) and len(set(actual)) > 1

    def test_replacement_discards_original_phoneme_supervision(self, tmp_path, monkeypatch):
        replacement = 'read 1234 twice'
        path = tmp_path / 'challenging.txt'
        path.write_text(replacement + '\n', encoding='utf-8')
        cuts = _single_turn_cutset(text='old text', normalized_text='old text')
        supervision = next(iter(cuts)).supervisions[0]
        supervision.custom.update(ipa='oʊld tɛkst', ipa_alignment=[[0, 8, 'old text', 'oʊld tɛkst']])
        dataset = self._dataset(
            challenging_texts_path=str(path),
            challenging_text_replacement_prob=1.0,
            enable_phoneme_text_input=True,
            partial_phoneme_text_prob=1.0,
        )
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        monkeypatch.setattr(
            'nemo.collections.tts.data.text_to_speech_dataset_lhotse.partially_phonemize_text',
            lambda **kwargs: pytest.fail('Replaced prompts must not reuse the original IPA alignment'),
        )

        batch = dataset[cuts]

        assert batch['raw_texts'] == [replacement]
        assert batch['text'][0].tolist() == [10 + len(replacement), dataset.eos_id]
        assert batch['phoneme_tokens'][0].tolist() == [1, 2]
        assert batch['phoneme_tokens_lens'].tolist() == [2]
        assert supervision.ipa == 'oʊld tɛkst'

    def test_validation_does_not_replace_prompts(self, tmp_path):
        path = tmp_path / 'challenging.txt'
        path.write_text('replacement text\n', encoding='utf-8')
        dataset = self._dataset(
            challenging_texts_path=str(path), challenging_text_replacement_prob=1.0, dataset_type='test'
        )

        batch = dataset[_single_turn_cutset(text='raw input', normalized_text='normalized input')]

        assert batch['raw_texts'] == ['normalized input']
        assert 'replacement text' not in dataset.text_tokenizer.encoded_texts

    def test_batch_replaces_only_one_english_tts_prompt(self, tmp_path):
        replacement = 'read one two again'
        path = tmp_path / 'challenging.txt'
        path.write_text(replacement + '\n', encoding='utf-8')
        cuts = []
        for index, (language, task) in enumerate([('en', 'tts'), ('en', 'tts'), ('es', 'tts'), ('en', 'dialog')]):
            cut = next(iter(_single_turn_cutset(text=f'original {index}', normalized_text=f'original {index}')))
            cut.id = f'prompt-{index}'
            cut.lang = language
            cut.task = task
            cuts.append(cut)
        dataset = self._dataset(challenging_texts_path=str(path), challenging_text_replacement_prob=1.0)

        batch = dataset[CutSet.from_cuts(cuts)]

        assert batch['raw_texts'].count(replacement) == 1
        assert replacement in batch['raw_texts'][:2]
        assert batch['raw_texts'][2:] == ['original 2', 'original 3']

    def test_replacement_respects_supervision_language_without_cut_language(self, tmp_path):
        path = tmp_path / 'challenging.txt'
        path.write_text('English replacement\n', encoding='utf-8')
        normalized = 'Ruf eins zwei drei vier an'
        cuts = _single_turn_cutset(text='Ruf 1234 an', normalized_text=normalized)
        cut = next(iter(cuts))
        del cut.custom['lang']
        cut.supervisions[0].language = 'de'
        dataset = self._dataset(challenging_texts_path=str(path), challenging_text_replacement_prob=1.0)

        batch = dataset[cuts]

        assert batch['languages'] == ['de']
        assert batch['raw_texts'] == batch['cer_reference_texts'] == [normalized]
        assert 'English replacement' not in dataset.text_tokenizer.encoded_texts

    def test_replacement_skips_candidates_exceeding_codec_frame_budget(self, tmp_path):
        fitting = 'say ten'
        too_long = 'one two three four five six seven eight nine ten'
        path = tmp_path / 'challenging.txt'
        path.write_text(too_long + '\n' + fitting + '\n', encoding='utf-8')
        dataset = self._dataset(challenging_texts_path=str(path), challenging_text_replacement_prob=1.0)

        class WordTokenizer(_FakeTextTokenizer):
            def encode(self, text, tokenizer_name):
                self.encoded_texts.append(text)
                return list(range(10, 10 + len(text.split())))

        dataset.text_tokenizer = WordTokenizer()
        cuts = _single_turn_cutset(text='old text', normalized_text='old text')
        cut = next(iter(cuts))
        cut.duration = 4 * CODEC_MODEL_SAMPLES_PER_FRAME / SAMPLE_RATE
        cut.supervisions[0].duration = cut.duration
        batch = dataset[cuts]

        assert batch['raw_texts'] == [fitting]
        assert batch['text'][0].tolist() == [10, 11, dataset.eos_id]

    @pytest.mark.parametrize('normalized_text', [None, '', 'Call one two three four.'])
    def test_cer_reference_uses_normalization_independently_of_raw_input(self, normalized_text):
        raw = 'Call 1234.'
        cuts = _single_turn_cutset(text=raw, normalized_text=normalized_text)
        if normalized_text is None:
            del next(iter(cuts)).supervisions[0].custom['normalized_text']
        dataset = self._dataset(load_normalized_text_percent=0.0)

        batch = dataset[cuts]

        assert batch['raw_texts'] == [raw]
        assert batch['text'][0].tolist() == [10 + len(raw), dataset.eos_id]
        assert batch['cer_reference_texts'] == [normalized_text or raw]

    @pytest.mark.parametrize('global_percent,cut_percent', [(1.0, 0.0), (0.0, 1.0)])
    def test_cut_normalization_override_keeps_normalized_cer_reference(self, global_percent, cut_percent):
        raw = 'Call 1234.'
        normalized = 'Call one two three four.'
        cuts = _single_turn_cutset(text=raw, normalized_text=normalized)
        next(iter(cuts)).load_normalized_text_percent = cut_percent
        dataset = self._dataset(load_normalized_text_percent=global_percent)

        batch = dataset[cuts]

        expected_input = normalized if cut_percent else raw
        assert batch['raw_texts'] == [expected_input]
        assert batch['text'][0].tolist() == [10 + len(expected_input), dataset.eos_id]
        assert batch['cer_reference_texts'] == [normalized]

    @pytest.mark.parametrize('cut_percent', [-0.1, 1.1])
    def test_invalid_cut_normalization_override_is_rejected(self, cut_percent):
        cuts = _single_turn_cutset()
        next(iter(cuts)).load_normalized_text_percent = cut_percent
        dataset = self._dataset(load_normalized_text_percent=1.0)

        with pytest.raises(ValueError, match='load_normalized_text_percent'):
            dataset[cuts]
