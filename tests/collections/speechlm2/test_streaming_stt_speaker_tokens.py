# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""`<spk:N>` token registration and decode round-tripping."""

import pytest

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import decode_with_blank
from nemo.collections.speechlm2.models import streaming_stt_model as M

BASE = {
    "blank_token": "<blank>",
    "compact_template": False,
    "prepend_write_token": False,
    "write_token": "<|write|>",
    "end_of_audio_token": "<|im_start|>",
}


def _config_defaults() -> dict:
    """Every default from the real config dataclass.

    ``BASE`` only names the handful of fields these tests care about. Building the stub from that
    alone means any NEW config field added elsewhere makes `_register_special_tokens` raise
    ``AttributeError`` here, in a test that has nothing to do with it -- which is how a merge that
    added ``register_audio_token`` broke this file. Starting from the dataclass keeps the stub
    honest without listing fields by hand.
    """
    from dataclasses import MISSING, fields

    defaults = {}
    for field in fields(M.StreamingSTTModelConfig):
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[misc]
            defaults[field.name] = field.default_factory()  # type: ignore[misc]
    return defaults


class _Stub:
    """Exercises `_register_special_tokens` without building a 1.7B LLM."""

    def __init__(self, cfg):
        self.core_cfg = type("C", (), {**_config_defaults(), **cfg})()
        self.tokenizer = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
        self.resized = 0

    def _resize_llm_embeddings(self):
        self.resized += 1

    @property
    def blank_token_id(self):
        return self.tokenizer.tokenizer.convert_tokens_to_ids(self.blank_token)


def _register(**speaker_tokens):
    stub = _Stub({**BASE, "speaker_tokens": speaker_tokens or None})
    M.StreamingSTTModel._register_special_tokens(stub)
    return stub


class TestSpeakerTokenRegistration:
    @pytest.mark.unit
    def test_tags_become_single_tokens(self):
        # Stock Qwen3 splits `<spk:0>` into SIX tokens; unregistered, every speaker change would
        # cost six emissions and the loss would be dominated by tag fragments.
        stub = _register(enable=True, template="<spk:{i}>", max_speakers=4)
        assert len(stub.speaker_token_ids) == 4
        assert all(M.token_in_vocab(f"<spk:{i}>", stub.tokenizer) for i in range(4))

    @pytest.mark.unit
    def test_ids_are_contiguous(self):
        stub = _register(enable=True, template="<spk:{i}>", max_speakers=4)
        first = stub.speaker_token_ids[0]
        assert stub.speaker_token_ids == list(range(first, first + 4))

    @pytest.mark.unit
    @pytest.mark.parametrize("cfg", [{}, {"enable": False, "max_speakers": 4}])
    def test_disabled_is_inert(self, cfg):
        assert _register(**cfg).speaker_token_ids == []

    @pytest.mark.unit
    def test_base_token_id_mismatch_is_rejected(self):
        # Guards the patched-tokenizer layout the SALM/phPEE reference expects.
        with pytest.raises(ValueError, match="base_token_id"):
            _register(enable=True, template="<spk:{i}>", max_speakers=4, base_token_id=100)


class TestSpeakerTokenDecoding:
    @pytest.mark.unit
    def test_tags_survive_decoding(self):
        # `<spk:N>` are registered as *special* tokens, and `tokens_to_text` filters
        # `all_special_tokens` — without the id map they vanish and cpWER would compare tagged
        # references against untagged hypotheses.
        nt = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
        hf = nt.tokenizer
        hf.add_special_tokens({"additional_special_tokens": ["<blank>"] + [f"<spk:{i}>" for i in range(2)]})
        blank_id = hf.convert_tokens_to_ids("<blank>")
        spk_map = {hf.convert_tokens_to_ids(f"<spk:{i}>"): f"<spk:{i}>" for i in range(2)}
        ids = (
            hf.encode("<spk:0> hello there", add_special_tokens=False)
            + [blank_id]
            + hf.encode("<spk:1> hi", add_special_tokens=False)
        )

        assert "<spk:" not in decode_with_blank(ids, "<blank>", nt), "baseline: tags are stripped"
        assert decode_with_blank(ids, "<blank>", nt, speaker_token_ids=spk_map) == "<spk:0> hello there <spk:1> hi"

    @pytest.mark.unit
    def test_empty_map_preserves_legacy_behaviour(self):
        nt = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
        hf = nt.tokenizer
        hf.add_special_tokens({"additional_special_tokens": ["<blank>"]})
        ids = hf.encode("hello world", add_special_tokens=False)
        assert decode_with_blank(ids, "<blank>", nt, speaker_token_ids={}) == decode_with_blank(ids, "<blank>", nt)
