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

import copy
import warnings

import numpy as np
import pytest
from omegaconf import OmegaConf

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


# The required fields of the model config, for building the real dataclass.
_REQUIRED = {
    "pretrained_llm": "Qwen/Qwen3-1.7B",
    "pretrained_asr": "unused",
    "load_llm_weights": False,
    "blank_token": "<blank>",
    "load_asr_weights": False,
    "freeze_speech_encoder": False,
    "freeze_modality_adapter": False,
    "freeze_modality_proj": False,
    "freeze_llm_model": True,
    "freeze_llm_head": False,
    "freeze_embed_tokens": False,
    "chunk_size": 2,
}
# The speaker tokens of the suffix + switch-token arm trained before the rename: four tags and `<spk_switch>`, under
# the deprecated key. Its tokenizer has `<blank>` at 151669, the tags at 151670-151673 and `<spk_switch>` at 151674.
ARM_B_SPEAKER_TOKENS = {
    "enable": True,
    "template": "<spk:{i}>",
    "max_speakers": 4,
    "base_token_id": None,
    "switch_token": "<spk_switch>",
}
TAG_IDS = [151670, 151671, 151672, 151673]


def _model_config(speaker_tokens):
    from nemo.collections.speechlm2.parts.utils import to_dataclass

    return to_dataclass(M.StreamingSTTModelConfig, {**_REQUIRED, "speaker_tokens": speaker_tokens})


def _register_config(speaker_tokens):
    """`_register_special_tokens` on the real config dataclass, so the keys are read as the model reads them."""
    stub = _Stub(BASE)
    stub.core_cfg = _model_config(speaker_tokens)
    M.StreamingSTTModel._register_special_tokens(stub)
    return stub


def _with(speaker_tokens=ARM_B_SPEAKER_TOKENS, drop=(), **keys):
    out = {k: v for k, v in speaker_tokens.items() if k not in drop}
    out.update(keys)
    return out


_MODEL_ALIASES = {
    "new_only": ({"turn_start_token": "<|turn_start|>"}, "<|turn_start|>", False),
    "old_only": ({"switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "both_equal": ({"turn_start_token": "<spk_switch>", "switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "old_null": ({"turn_start_token": "<|turn_start|>", "switch_token": None}, "<|turn_start|>", False),
    "old_empty": ({"turn_start_token": "<|turn_start|>", "switch_token": ""}, "<|turn_start|>", False),
    "new_null": ({"turn_start_token": None, "switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "new_empty": ({"turn_start_token": "", "switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "both_empty": ({"turn_start_token": "", "switch_token": ""}, None, False),
}


class TestTurnStartTokenModelConfig:
    """``speaker_tokens.turn_start_token`` replaces ``speaker_tokens.switch_token``, which is still read."""

    @pytest.mark.unit
    @pytest.mark.parametrize("case", list(_MODEL_ALIASES))
    def test_alias_rules(self, case):
        keys, resolved, warns = _MODEL_ALIASES[case]
        given = _with(drop=("switch_token",), **keys)
        before = copy.deepcopy(given)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _model_config(given)
        deprecations = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
        assert cfg.speaker_tokens == {**_with(drop=("switch_token",)), "turn_start_token": resolved}
        assert deprecations == (
            [
                "model.speaker_tokens.switch_token is deprecated: it was renamed to "
                f"model.speaker_tokens.turn_start_token. Using {resolved!r} as model.speaker_tokens.turn_start_token."
            ]
            if warns
            else []
        )
        assert given == before  # resolved on a copy: the config the model saves keeps its keys

    @pytest.mark.unit
    def test_different_values_under_both_keys_are_an_error(self):
        with pytest.raises(ValueError, match=r"turn_start_token='<\|turn_start\|>' and .*switch_token='<spk_switch>'"):
            _model_config(_with(turn_start_token="<|turn_start|>"))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "speaker_tokens", [None, {}, {"enable": True, "max_speakers": 2}], ids=["none", "empty", "no_token"]
    )
    def test_speaker_tokens_without_the_token_are_unchanged(self, speaker_tokens):
        assert _model_config(speaker_tokens).speaker_tokens == speaker_tokens

    @pytest.mark.unit
    @pytest.mark.parametrize("key", ["detected_token", "turn_start", "switch", "Enable"])
    def test_an_unknown_key_is_rejected(self, key):
        # A misspelled or renamed key used to be ignored: the token was then silently not registered.
        with pytest.raises(ValueError, match=rf"model.speaker_tokens has unknown keys \['{key}'\]"):
            _model_config(_with(**{key: "<|turn_start|>"}))


class TestTurnStartTokenRegistration:
    """The turn-start token is registered in the same call as the tags, after them, under either key."""

    @pytest.mark.unit
    def test_the_deprecated_key_keeps_the_ids_of_models_trained_with_it(self):
        with pytest.warns(FutureWarning):
            stub = _register_config(ARM_B_SPEAKER_TOKENS)
        hf = stub.tokenizer.tokenizer
        assert stub.speaker_token_ids == TAG_IDS
        assert stub.turn_start_token_id == hf.convert_tokens_to_ids("<spk_switch>") == 151674
        assert len(hf) == 151675
        assert M.StreamingSTTModel.speaker_switch_token_id.fget(stub) == 151674  # the attribute's old name

    @pytest.mark.unit
    def test_the_new_key_takes_the_same_slot(self):
        stub = _register_config(_with(drop=("switch_token",), turn_start_token="<|turn_start|>"))
        hf = stub.tokenizer.tokenizer
        assert stub.speaker_token_ids == TAG_IDS
        assert stub.turn_start_token_id == 151674
        assert hf.encode("hi<|turn_start|><spk:1> there", add_special_tokens=False)[1:3] == [151674, TAG_IDS[1]]
        # A special token, so decoding drops it; the tags are put back by the speaker-token map.
        ids = hf.encode("<|turn_start|><spk:0> hi there <|turn_start|><spk:1> yes", add_special_tokens=False)
        spk_map = M.StreamingSTTModel.speaker_token_map.fget(stub)
        decoded = decode_with_blank(ids, "<blank>", stub.tokenizer, speaker_token_ids=spk_map, strip_whitespace=True)
        assert decoded == "<spk:0> hi there <spk:1> yes"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "speaker_tokens", [{"enable": True, "max_speakers": 4}, _with(enable=False)], ids=["no_token", "disabled"]
    )
    def test_no_token_is_registered_without_one(self, speaker_tokens):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            stub = _register_config(speaker_tokens)
        assert stub.turn_start_token_id is None
        assert M.StreamingSTTModel.speaker_switch_token_id.fget(stub) is None
        assert len(stub.tokenizer.tokenizer) == (151674 if speaker_tokens.get("enable", True) else 151670)


def _data_cfg(enable=True, **keys):
    return OmegaConf.create({"words_per_group": 1, "multispeaker_cfg": {"enable": enable, "num_speakers": 4}, **keys})


def _assert_matches(speaker_tokens, data_cfg, val_data_cfg=None):
    from types import SimpleNamespace

    stub = SimpleNamespace(core_cfg=_model_config(speaker_tokens))
    M.StreamingSTTModel._assert_speaker_config_matches_data(stub, data_cfg, val_data_cfg)


_NEW = _with(drop=("switch_token",), turn_start_token="<|turn_start|>")
_NONE = _with(drop=("switch_token",))


class TestTurnStartTokenMatchesTheData:
    """The model registers the token and the dataset writes it: built with a dataset config, the two must agree."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "speaker_tokens,data_keys",
        [
            pytest.param(_NEW, {}, id="model_only"),
            pytest.param(_NONE, {"turn_start_token": "<|turn_start|>"}, id="data_only"),
            pytest.param(_with(enable=False), {"speaker_switch_token": "<spk_switch>"}, id="model_disabled"),
            pytest.param(_NEW, {"turn_start_token": "<spk_switch>"}, id="different"),
            pytest.param(ARM_B_SPEAKER_TOKENS, {"turn_start_token": "<|turn_start|>"}, id="different_old_model_key"),
            pytest.param(ARM_B_SPEAKER_TOKENS, {}, id="old_model_key_only"),
        ],
    )
    def test_a_mismatch_is_an_error(self, speaker_tokens, data_keys):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(ValueError, match=r"model.speaker_tokens.turn_start_token=.* but data.dataset"):
                _assert_matches(speaker_tokens, _data_cfg(**data_keys))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "speaker_tokens,data_keys",
        [
            pytest.param(ARM_B_SPEAKER_TOKENS, {"speaker_switch_token": "<spk_switch>"}, id="old_keys"),
            pytest.param(_NEW, {"turn_start_token": "<|turn_start|>"}, id="new_keys"),
            pytest.param(ARM_B_SPEAKER_TOKENS, {"turn_start_token": "<spk_switch>"}, id="old_model_new_data"),
            pytest.param(_NONE, {}, id="neither"),
            pytest.param(_NONE, {"turn_start_token": None, "speaker_switch_token": ""}, id="neither_null_and_empty"),
            pytest.param(_with(enable=False), {}, id="model_disabled_data_none"),
            pytest.param(None, {}, id="no_speaker_tokens"),
        ],
    )
    def test_equal_tokens_pass(self, speaker_tokens, data_keys):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _assert_matches(speaker_tokens, _data_cfg(**data_keys))

    @pytest.mark.unit
    def test_a_config_without_multispeaker_is_not_compared(self):
        # An ASR-only validation set: the dataset ignores the token there.
        _assert_matches(_NEW, _data_cfg(turn_start_token="<|turn_start|>"), _data_cfg(enable=False))
        _assert_matches(_NEW, OmegaConf.create({"words_per_group": 1}))

    @pytest.mark.unit
    def test_the_validation_config_is_compared_too(self):
        with pytest.raises(ValueError, match=r"but the validation dataset config.turn_start_token=None"):
            _assert_matches(_NEW, _data_cfg(turn_start_token="<|turn_start|>"), _data_cfg())

    @pytest.mark.unit
    def test_the_deprecated_data_key_messages_name_the_keys_as_the_dataset_does(self):
        # The validation config is described in prose, which must not be glued to the key names.
        old = _data_cfg(speaker_switch_token="<|turn_start|>")
        with pytest.warns(FutureWarning) as record:
            _assert_matches(_NEW, old, old)
        assert {str(w.message) for w in record} == {
            "speaker_switch_token is deprecated: it was renamed to turn_start_token. "
            "Using '<|turn_start|>' as turn_start_token."
        }
        new = _data_cfg(turn_start_token="<|turn_start|>")
        both = _data_cfg(turn_start_token="<|turn_start|>", speaker_switch_token="<spk_switch>")
        for configs, name in (((both,), "data.dataset"), ((new, both), "the validation dataset config")):
            with pytest.raises(ValueError) as err:
                _assert_matches(_NEW, *configs)
            assert str(err.value) == (
                f"{name}: turn_start_token='<|turn_start|>' and speaker_switch_token='<spk_switch>' are both set and "
                "differ. speaker_switch_token is the deprecated name of turn_start_token: set only turn_start_token."
            )

    @pytest.mark.unit
    def test_a_data_key_interpolated_from_the_deprecated_model_key(self):
        root = OmegaConf.create(
            {
                "model": {"speaker_tokens": ARM_B_SPEAKER_TOKENS},
                "data": {"dataset": _data_cfg(speaker_switch_token="${model.speaker_tokens.switch_token}")},
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _assert_matches(OmegaConf.to_container(root.model.speaker_tokens), root.data.dataset)

    @pytest.mark.unit
    def test_a_dataset_config_dataclass(self):
        from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataConfig

        data = StreamingSTTDataConfig(
            sample_rate=16000,
            frame_length_in_secs=0.08,
            chunk_size=2,
            turn_start_token="<|turn_start|>",
            multispeaker_cfg={"enable": True, "num_speakers": 4},
        )
        _assert_matches(_NEW, data)
        with pytest.raises(ValueError, match="turn_start_token=None but data.dataset.turn_start_token"):
            _assert_matches(_NONE, data)


def _build_model(monkeypatch, speaker_tokens, data_cfg, **overrides):
    """A StreamingSTTModel with a tiny Qwen3 (the real tokenizer and vocabulary size) and a tiny encoder."""
    import torch

    from tests.collections.speechlm2.test_streaming_stt_automodel import make_cfg
    from tests.collections.speechlm2.test_streaming_stt_dynamic_diarizer import _tiny_llm

    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    _tiny_llm(monkeypatch)
    try:
        cfg = make_cfg(use_nemo_automodel=False, speaker_tokens=speaker_tokens, **overrides)
        return M.StreamingSTTModel(cfg, data_cfg=data_cfg)
    finally:
        torch.set_default_device(previous)


class TestTurnStartTokenModel:
    """Constructing the model, as training and ``load_from_checkpoint`` do."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "speaker_tokens,data_keys,shrink",
        [
            pytest.param(ARM_B_SPEAKER_TOKENS, {"speaker_switch_token": "<spk_switch>"}, True, id="deprecated_keys"),
            pytest.param(_NEW, {"turn_start_token": "<|turn_start|>"}, True, id="new_keys"),
            # With the embedding table kept at its size, nothing would fail if the token were dropped.
            pytest.param(
                ARM_B_SPEAKER_TOKENS, {"speaker_switch_token": "<spk_switch>"}, False, id="deprecated_keys_no_shrink"
            ),
        ],
    )
    def test_the_token_takes_the_slot_after_the_tags(self, monkeypatch, speaker_tokens, data_keys, shrink):
        data_cfg = _data_cfg(speaker_tag_placement="suffix", **data_keys)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model = _build_model(monkeypatch, speaker_tokens, data_cfg, allow_shrink_embedding=shrink)
        assert model.speaker_token_ids == TAG_IDS
        assert model.turn_start_token_id == model.speaker_switch_token_id == 151674
        assert len(model.tokenizer.tokenizer) == 151675
        assert model.embed_tokens.num_embeddings == (151675 if shrink else QWEN3_LM_HEAD_ROWS)
        assert OmegaConf.to_container(model.cfg.speaker_tokens) == speaker_tokens  # saved as given

    @pytest.mark.unit
    def test_a_token_the_dataset_does_not_write_is_an_error(self, monkeypatch):
        with pytest.raises(ValueError, match=r"turn_start_token='<\|turn_start\|>' but data.dataset.turn_start_token"):
            _build_model(monkeypatch, _NEW, _data_cfg())
