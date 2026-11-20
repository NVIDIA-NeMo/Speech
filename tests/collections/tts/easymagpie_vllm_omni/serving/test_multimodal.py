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
from types import SimpleNamespace

import numpy as np
import pytest

from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.multimodal import (
    EasyMagpieAudioParser,
    EasyMagpieDummyInputsBuilder,
    EasyMagpieMultiModalProcessor,
    _infer_audio_output_kinds,
)
from vllm.multimodal.processing import ProcessorInputs


def test_audio_parser_requires_explicit_codec_format():
    parser = EasyMagpieAudioParser(sample_rate=16000, expected_hidden_size=None)
    waveform = np.zeros(16, dtype=np.float32)

    parsed, sample_rate = parser._get_audio_with_sr((waveform, 16000))
    assert parsed is waveform
    assert sample_rate is None

    with pytest.raises(ValueError, match="16000 Hz; received 22050 Hz"):
        parser._get_audio_with_sr((waveform, 22050))
    with pytest.raises(ValueError, match="must be mono"):
        parser._get_audio_with_sr((np.zeros((2, 16), dtype=np.float32), 16000))
    with pytest.raises(ValueError, match="must be passed as a"):
        parser._get_audio_with_sr(waveform)


def test_dummy_inputs_profile_reference_and_user_layout(monkeypatch):
    builder = object.__new__(EasyMagpieDummyInputsBuilder)
    builder.info = SimpleNamespace(
        arch=SimpleNamespace(audio_input_token_id=1),
        parse_mm_data=lambda data, validate: data,
    )
    monkeypatch.setattr(builder, "get_dummy_mm_data", lambda seq_len, mm_counts, mm_options: {"audio": []})

    inputs = builder.get_dummy_processor_inputs(
        seq_len=0,
        mm_counts={"audio": 2},
        mm_options={},
    )

    assert inputs.prompt == [1, 0, 1]
    assert inputs.hf_processor_mm_kwargs == {}


class _AudioItems:
    def __init__(self, lengths):
        self.lengths = lengths

    def get_audio_length(self, item_idx):
        return self.lengths[item_idx]

    def get_count(self):
        return len(self.lengths)


def _processor_for_lengths(lengths):
    processor = EasyMagpieMultiModalProcessor.__new__(EasyMagpieMultiModalProcessor)
    processor.info = SimpleNamespace(arch=EasyMagpieOmniArch(codec_encoder_bundled=True))
    return processor, {"audio": _AudioItems(lengths)}


def test_strict_audio_parser_bypasses_vllm_preprocessing_cache(monkeypatch):
    processor = EasyMagpieMultiModalProcessor.__new__(EasyMagpieMultiModalProcessor)
    expected = object()
    processor.info = SimpleNamespace(arch=SimpleNamespace(audio_input_token_id=1))
    captured = []

    def apply_processor(inputs, timing_ctx):
        captured.append(inputs)
        return expected

    monkeypatch.setattr(processor, "_apply_hf_processor", apply_processor)

    inputs = ProcessorInputs(prompt=[1, 0, 1], mm_data_items={"audio": _AudioItems([100, 100])})

    assert processor._cached_apply_hf_processor(inputs, None) is expected
    assert captured[0].hf_processor_mm_kwargs["_easymagpie_audio_outputs"] == [1, 0]


def test_reference_audio_placeholder_uses_actual_codec_row_count():
    processor, mm_items = _processor_for_lengths([32_000])
    update = processor._get_prompt_updates(
        mm_items,
        {},
        None,
    )[
        0
    ].resolve(0)

    assert update.content.full == [1] * 27


def test_capable_checkpoint_expands_inferred_reference_and_user_outputs():
    processor = EasyMagpieMultiModalProcessor.__new__(EasyMagpieMultiModalProcessor)
    processor.info = SimpleNamespace(
        arch=EasyMagpieOmniArch(
            codec_encoder_bundled=True,
            use_multiturn_dataset=True,
            condition_on_user_speech=True,
            use_user_speaking_token=True,
            use_user_speaking_end_token=True,
            streaming_phonemes_delay=3,
            streaming_speech_delay=5,
        )
    )
    mm_items = {"audio": _AudioItems([32_000, 100])}

    replacement = processor._get_prompt_updates(
        mm_items,
        {"_easymagpie_audio_outputs": [1, 0]},
        None,
    )[0]

    assert replacement.resolve(0).content.full == [1] * 27
    assert replacement.resolve(1).content.full == [1] * 6


@pytest.mark.parametrize(
    "prompt,audio_count,expected",
    [
        pytest.param([1, 0], 1, [1], id="reference"),
        pytest.param([1], 1, [0], id="user"),
        pytest.param([1, 0, 1], 2, [1, 0], id="reference-and-user"),
    ],
)
def test_audio_markers_classify_items_by_prompt_position(prompt, audio_count, expected):
    assert _infer_audio_output_kinds(prompt, audio_count, marker_id=1) == expected


def test_audio_marker_count_must_match_audio_item_count():
    with pytest.raises(ValueError, match="1 audio markers for 2 audio items"):
        _infer_audio_output_kinds([1, 0], audio_count=2, marker_id=1)


@pytest.mark.parametrize("output_kind", [0, 1], ids=["user", "reference"])
def test_raw_audio_limit_rejects_reference_and_user_items(output_kind):
    processor = EasyMagpieMultiModalProcessor.__new__(EasyMagpieMultiModalProcessor)
    arch = EasyMagpieOmniArch(
        codec_encoder_bundled=True,
        max_audio_seconds=0.001,
        use_multiturn_dataset=True,
        condition_on_user_speech=True,
        use_user_speaking_token=True,
    )
    max_samples = 16
    processor.info = SimpleNamespace(
        arch=arch,
        get_tokenizer=lambda: SimpleNamespace(encode=lambda *args, **kwargs: []),
        get_max_audio_samples=lambda: max_samples,
    )
    processor_kwargs = {"_easymagpie_audio_outputs": [output_kind]}
    audio_at_limit = np.zeros(max_samples, dtype=np.float32)
    output = processor._call_hf_processor("", {"audios": [audio_at_limit]}, processor_kwargs, {})

    assert output["audio_lens"].tolist() == [max_samples]

    with pytest.raises(ValueError, match=rf"raw audio has {max_samples + 1} samples"):
        processor._call_hf_processor(
            "",
            {"audios": [np.zeros(max_samples + 1, dtype=np.float32)]},
            processor_kwargs,
            {},
        )
