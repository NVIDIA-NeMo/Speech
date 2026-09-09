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
"""
Tests for ``StreamingSTTModel._resize_llm_embeddings``.

The embedding table must never *shrink* to ``len(tokenizer)``.  Backbones ship
spare rows (Qwen3: ``config.vocab_size=151936`` vs ``len(tokenizer)=151669``),
and shrinking makes the parameter shape a function of how many special tokens a
config happens to register — so adding one more special token would break
loading every checkpoint trained without it.

Uses lightweight stubs rather than a real LLM: the method only reads
``len(tokenizer)`` and the input-embedding row count.
"""

from types import SimpleNamespace

import pytest

from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel, StreamingSTTModelConfig

# Qwen3-1.7B: 267 spare embedding rows before any special token is added.
QWEN3_VOCAB_ROWS = 151936
QWEN3_TOKENIZER_LEN = 151669


class _StubTokenizer:
    def __init__(self, n_tokens: int):
        self._n = n_tokens

    def __len__(self) -> int:
        return self._n


class _StubLLM:
    """Records ``resize_token_embeddings`` calls instead of performing them."""

    def __init__(self, n_rows: int):
        self._embed = SimpleNamespace(weight=SimpleNamespace(shape=(n_rows, 2048)))
        self.resize_calls: list[int] = []

    def get_input_embeddings(self):
        return self._embed

    def resize_token_embeddings(self, target: int):
        self.resize_calls.append(target)
        self._embed = SimpleNamespace(weight=SimpleNamespace(shape=(target, 2048)))


def _make_mock_self(n_rows: int, n_tokens: int, allow_shrink: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        llm=_StubLLM(n_rows),
        tokenizer=SimpleNamespace(tokenizer=_StubTokenizer(n_tokens)),
        core_cfg=SimpleNamespace(allow_shrink_embedding=allow_shrink),
    )


@pytest.mark.unit
@pytest.mark.parametrize("n_added_special_tokens", [0, 2, 3, 4])
def test_spare_rows_are_reused_without_resizing(n_added_special_tokens):
    """Regression: added special tokens must occupy spare rows, not shrink the table.

    Fails before the fix — the old implementation called
    ``resize_token_embeddings(len(tokenizer))`` unconditionally, shrinking
    151936 -> 151669 + n and making the shape depend on how many tokens the
    config registered.
    """
    mock_self = _make_mock_self(QWEN3_VOCAB_ROWS, QWEN3_TOKENIZER_LEN + n_added_special_tokens, allow_shrink=False)

    StreamingSTTModel._resize_llm_embeddings(mock_self)

    assert mock_self.llm.resize_calls == []
    assert mock_self.llm.get_input_embeddings().weight.shape[0] == QWEN3_VOCAB_ROWS


@pytest.mark.unit
def test_shape_is_independent_of_how_many_tokens_are_registered():
    """The whole point: two configs registering different token counts agree on shape."""
    shapes = []
    for n_added in (2, 3):  # e.g. blank+write, then blank+write+audio_placeholder
        mock_self = _make_mock_self(QWEN3_VOCAB_ROWS, QWEN3_TOKENIZER_LEN + n_added, allow_shrink=False)
        StreamingSTTModel._resize_llm_embeddings(mock_self)
        shapes.append(mock_self.llm.get_input_embeddings().weight.shape[0])

    assert shapes[0] == shapes[1], "embedding shape must not depend on the number of added tokens"


@pytest.mark.unit
def test_table_grows_when_tokenizer_exceeds_rows():
    """No spare rows left — the table must still grow to cover every token ID."""
    mock_self = _make_mock_self(n_rows=100, n_tokens=105, allow_shrink=False)

    StreamingSTTModel._resize_llm_embeddings(mock_self)

    assert mock_self.llm.resize_calls == [105]
    assert mock_self.llm.get_input_embeddings().weight.shape[0] == 105


@pytest.mark.unit
def test_exact_fit_is_a_noop():
    mock_self = _make_mock_self(n_rows=105, n_tokens=105, allow_shrink=False)

    StreamingSTTModel._resize_llm_embeddings(mock_self)

    assert mock_self.llm.resize_calls == []


@pytest.mark.unit
def test_allow_shrink_embedding_restores_legacy_resize():
    """The escape hatch for loading checkpoints trained under the old behaviour.

    Those checkpoints have a table shrunk to ``len(tokenizer)``, so a
    freshly-built model that keeps the backbone's spare rows would fail to load
    them with an ``embed_tokens.weight`` shape mismatch.
    """
    mock_self = _make_mock_self(QWEN3_VOCAB_ROWS, QWEN3_TOKENIZER_LEN + 2, allow_shrink=True)

    StreamingSTTModel._resize_llm_embeddings(mock_self)

    assert mock_self.llm.resize_calls == [QWEN3_TOKENIZER_LEN + 2]
    assert mock_self.llm.get_input_embeddings().weight.shape[0] == QWEN3_TOKENIZER_LEN + 2


@pytest.mark.unit
def test_allow_shrink_embedding_still_grows_when_needed():
    """Legacy mode must also cover every token ID, not only shrink."""
    mock_self = _make_mock_self(n_rows=100, n_tokens=105, allow_shrink=True)

    StreamingSTTModel._resize_llm_embeddings(mock_self)

    assert mock_self.llm.resize_calls == [105]


@pytest.mark.unit
def test_missing_field_matches_the_dataclass_default():
    """A config object without the field must behave like a real one.

    ``_resize_llm_embeddings`` reads the flag with ``getattr(..., True)``; if that
    fallback disagreed with ``StreamingSTTModelConfig.allow_shrink_embedding``,
    lightweight config objects would silently take the opposite branch from a
    real config.
    """
    assert StreamingSTTModelConfig.allow_shrink_embedding is True, "fallback below must track this"

    mock_self = SimpleNamespace(
        llm=_StubLLM(QWEN3_VOCAB_ROWS),
        tokenizer=SimpleNamespace(tokenizer=_StubTokenizer(QWEN3_TOKENIZER_LEN + 2)),
        core_cfg=SimpleNamespace(),  # no allow_shrink_embedding attribute at all
    )

    StreamingSTTModel._resize_llm_embeddings(mock_self)

    assert mock_self.llm.resize_calls == [QWEN3_TOKENIZER_LEN + 2]


@pytest.mark.unit
def test_default_preserves_legacy_checkpoint_shape():
    """The default must reproduce the shape every existing checkpoint was written with.

    Those were trained under an unconditional ``resize_token_embeddings``, so a
    freshly-built model has to land on ``len(tokenizer)`` — not on the backbone's
    larger spare-row count — for ``embed_tokens.weight`` to load.
    """
    n_tokens = QWEN3_TOKENIZER_LEN + 2  # blank + write
    mock_self = _make_mock_self(QWEN3_VOCAB_ROWS, n_tokens)  # no explicit flag → default

    StreamingSTTModel._resize_llm_embeddings(mock_self)

    assert mock_self.llm.get_input_embeddings().weight.shape[0] == n_tokens


@pytest.mark.unit
def test_registering_the_audio_token_is_shape_neutral_with_spare_rows():
    """M2/R8: the placeholder must fit in the backbone's spare rows.

    Qwen3 ships 151936 embedding rows for a 151669-token tokenizer, so blank,
    write and the audio placeholder occupy three of the 267 spares. Under
    ``allow_shrink_embedding=False`` the table is untouched; under the default
    ``True`` it tracks ``len(tokenizer)``, which is why registering the token is
    NOT safe for a checkpoint trained without it.
    """
    with_audio = QWEN3_TOKENIZER_LEN + 3  # blank + write + audio placeholder

    never_shrink = _make_mock_self(QWEN3_VOCAB_ROWS, with_audio, allow_shrink=False)
    StreamingSTTModel._resize_llm_embeddings(never_shrink)
    assert never_shrink.llm.get_input_embeddings().weight.shape[0] == QWEN3_VOCAB_ROWS
    assert never_shrink.llm.resize_calls == []

    legacy = _make_mock_self(QWEN3_VOCAB_ROWS, with_audio, allow_shrink=True)
    StreamingSTTModel._resize_llm_embeddings(legacy)
    assert legacy.llm.get_input_embeddings().weight.shape[0] == with_audio


@pytest.mark.unit
def test_audio_token_registration_is_off_by_default():
    """No recipe sets model.audio_tag, so the default must not register anything."""
    assert StreamingSTTModelConfig.register_audio_token is False
    assert StreamingSTTModelConfig.audio_tag == "<audio>"
