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

import numpy as np
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
# Rows of Qwen3-1.7B's LM head (`config.vocab_size`): more than its tokenizer has ids.
QWEN3_LM_HEAD_ROWS = 151936


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

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "<spk:0> hello diarization<lang:en> <spk:1> hi"),
            (
                {"replace_blank": "|", "collapse_whitespace": False},
                "<spk:0>  hello diarization<lang:en> | <spk:1>  hi | |",
            ),
        ],
        ids=["default", "kept_blanks_and_spacing"],
    )
    def test_ids_beyond_the_tokenizer_decode_as_if_absent(self, kwargs, expected):
        """With ``allow_shrink_embedding: false`` the LLM keeps the rows of its vocabulary beyond the tokenizer, so a
        decoder can emit an id that the tokenizer does not have. ``ids_to_tokens`` maps it to None, on which
        ``tokens_to_text`` raised TypeError. It is dropped, as ``ids_to_text`` drops it: wherever it falls, the
        text is that of the ids without it. The tokenizer's last id, an added content token, is kept."""
        nt = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
        hf = nt.tokenizer
        hf.add_special_tokens({"additional_special_tokens": ["<blank>"] + [f"<spk:{i}>" for i in range(2)]})
        # The tokenizer's last id, right below the first spare id. A bound below len(tokenizer) would drop it: HF's
        # vocab_size (which leaves out added tokens), a hard-coded size, or len(tokenizer) - 1.
        hf.add_tokens(["<lang:en>"])
        lang = hf.convert_tokens_to_ids("<lang:en>")
        assert lang == len(hf) - 1
        blank_id = hf.convert_tokens_to_ids("<blank>")
        spk0, spk1 = (hf.convert_tokens_to_ids(f"<spk:{i}>") for i in range(2))
        spk_map = {spk0: "<spk:0>", spk1: "<spk:1>"}
        hello = hf.encode(" hello", add_special_tokens=False)
        di, ar, ization = hf.encode(" diarization", add_special_tokens=False)  # one word, three pieces
        hi = hf.encode(" hi", add_special_tokens=False)
        spare = [len(hf), QWEN3_LM_HEAD_ROWS - 1]
        ids = [spk0, *hello, di, ar, ization, lang, blank_id, spk1, *hi, blank_id, blank_id]
        # Before the first tag, inside a word, before a tag, as all there is between two blanks, and at the end.
        with_spare = [spare[0], spk0, *hello, di, spare[1], ar, ization, lang, blank_id, spare[0], spk1, *hi, blank_id]
        with_spare += [*spare, blank_id, spare[1]]

        assert decode_with_blank(ids, "<blank>", nt, speaker_token_ids=spk_map, **kwargs) == expected
        assert decode_with_blank(with_spare, "<blank>", nt, speaker_token_ids=spk_map, **kwargs) == expected

    @pytest.mark.unit
    def test_ids_beyond_the_tokenizer_are_dropped_without_its_size(self):
        """Whether the tokenizer has an id is found from that id alone. ``len()`` of an HF fast tokenizer, which is
        also NeMo's ``vocab_size``, builds its whole vocabulary on every call: tens of ms for Qwen3, against well
        under 1 ms to decode a text. A tokenizer that refuses its size and its vocabulary decodes as the real one,
        NumPy ids as Python ones."""

        class SizeRefused:
            """Forwards to ``inner``, but refuses its size and its vocabulary."""

            def __init__(self, inner, **attributes):
                self.inner = inner
                vars(self).update(attributes)

            def __getattr__(self, name):
                if name in ("vocab_size", "vocab", "get_vocab"):
                    raise AssertionError(f"decode_with_blank asked for the tokenizer's {name}")
                return getattr(self.inner, name)

            def __len__(self):
                raise AssertionError("decode_with_blank asked for len(tokenizer)")

        nt = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
        hf = nt.tokenizer
        hf.add_special_tokens({"additional_special_tokens": ["<blank>"]})
        blank_id = hf.convert_tokens_to_ids("<blank>")
        hello = hf.encode("hello", add_special_tokens=False)
        di, ar, ization = hf.encode(" diarization", add_special_tokens=False)
        there = hf.encode(" there", add_special_tokens=False)
        ids = [*hello, di, ar, ization, blank_id, *there]
        with_spare = [*hello, di, len(hf), ar, ization, blank_id, *there, QWEN3_LM_HEAD_ROWS - 1]
        refused = SizeRefused(nt, tokenizer=SizeRefused(hf))

        assert decode_with_blank(ids, "<blank>", nt) == "hello diarization there"
        assert decode_with_blank(with_spare, "<blank>", refused) == "hello diarization there"
        # NumPy ids too: HF converts a lone id only if it is a Python int.
        assert decode_with_blank(list(np.array(with_spare)), "<blank>", refused) == "hello diarization there"
