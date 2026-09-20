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
Tests for StreamingSTTModel inference helpers:  `_ensure_inference_cache`.

Uses the real Qwen3 tokenizer to ensure the template is correct for the
actual model used in training.
"""

from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX
from nemo.collections.speechlm2.models.streaming_stt_model import (
    StreamingSTTModel,
    _duration_for_frames,
    _repr_chunk_size,
)

PRETRAINED_LLM = "Qwen/Qwen3-1.7B"
BLANK_TOKEN = "<blank>"
CHUNK_SIZE = 2

# Qwen3 special token IDs
IM_START = 151644
IM_END = 151645
NEWLINE = 198
THINK_START = 151667
THINK_END = 151668


@pytest.fixture(scope="module")
def hf_tok():
    return AutoTokenizer.from_pretrained(PRETRAINED_LLM)


def _make_mock_self(hf_tok, chunk_size=CHUNK_SIZE, blank_token=BLANK_TOKEN):
    """Build a minimal namespace that satisfies ``_ensure_inference_cache``."""
    mock = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenizer=hf_tok),
        core_cfg=SimpleNamespace(
            chunk_size=chunk_size,
            audio_tag="<audio>",
            compact_template=False,
            write_token="<|im_start|>",
        ),
        blank_token=blank_token,
        # blank_token_id is a model property; the mock just needs the value the
        # cache's logging line reads.
        blank_token_id=hf_tok.convert_tokens_to_ids(blank_token),
    )
    # ``_ensure_inference_cache`` builds the turn template through this method so
    # that it and ``_generate_chunked_streaming`` cannot drift apart. Delegate to
    # the real implementation rather than reimplementing it here.
    mock._build_turn_template_ids = lambda cs: StreamingSTTModel._build_turn_template_ids(mock, cs)
    return mock


def _run_ensure_cache(hf_tok, chunk_size=CHUNK_SIZE):
    """Call ``_ensure_inference_cache`` on a mock self and return it."""
    mock = _make_mock_self(hf_tok, chunk_size=chunk_size)
    StreamingSTTModel._ensure_inference_cache(mock)
    return mock


# ===========================================================================
# Tests: _ensure_inference_cache  (real Qwen3 tokenizer)
# ===========================================================================
class TestEnsureInferenceCache:

    def test_audio_slot_count(self, hf_tok):
        """Turn template must contain exactly chunk_size AUDIO_TOKEN_IDX markers."""
        mock = _run_ensure_cache(hf_tok, chunk_size=2)
        assert mock._turn_template_ids.count(AUDIO_TOKEN_IDX) == 2

    def test_audio_slot_count_different_chunk_sizes(self, hf_tok):
        for cs in (1, 2, 4, 8):
            mock = _run_ensure_cache(hf_tok, chunk_size=cs)
            assert mock._turn_template_ids.count(AUDIO_TOKEN_IDX) == cs

    def test_no_think_tokens_in_template(self, hf_tok):
        """The turn template must NOT contain Qwen3 <think>/<\/think> tokens."""
        mock = _run_ensure_cache(hf_tok)
        assert THINK_START not in mock._turn_template_ids
        assert THINK_END not in mock._turn_template_ids

    def test_template_starts_with_user_header(self, hf_tok):
        """Template must start with <|im_start|> user \\n."""
        mock = _run_ensure_cache(hf_tok)
        ids = mock._turn_template_ids
        assert ids[0] == IM_START
        # "user" token followed by newline
        user_id = hf_tok.encode("user", add_special_tokens=False)[0]
        assert ids[1] == user_id
        assert ids[2] == NEWLINE

    def test_template_ends_with_assistant_header(self, hf_tok):
        """Template must end with <|im_start|> assistant \\n."""
        mock = _run_ensure_cache(hf_tok)
        ids = mock._turn_template_ids
        asst_id = hf_tok.encode("assistant", add_special_tokens=False)[0]
        assert ids[-3:] == [IM_START, asst_id, NEWLINE]

    def test_audio_block_is_contiguous(self, hf_tok):
        """AUDIO_TOKEN_IDX markers should form a contiguous block."""
        mock = _run_ensure_cache(hf_tok, chunk_size=4)
        ids = mock._turn_template_ids
        first = ids.index(AUDIO_TOKEN_IDX)
        block = ids[first : first + 4]
        assert block == [AUDIO_TOKEN_IDX] * 4
        # No stray markers outside the block
        assert ids[:first].count(AUDIO_TOKEN_IDX) == 0
        assert ids[first + 4 :].count(AUDIO_TOKEN_IDX) == 0

    def test_template_matches_training_format(self, hf_tok):
        """The turn template should exactly match what the dataset produces
        for a single non-last user+assistant turn (no think tags)."""
        mock = _run_ensure_cache(hf_tok, chunk_size=2)
        ids = mock._turn_template_ids

        # Manually construct the expected template
        user_header = [IM_START] + hf_tok.encode("user", add_special_tokens=False) + [NEWLINE]
        audio = [AUDIO_TOKEN_IDX] * 2
        user_footer = [IM_END, NEWLINE]
        asst_header = [IM_START] + hf_tok.encode("assistant", add_special_tokens=False) + [NEWLINE]
        expected = user_header + audio + user_footer + asst_header
        assert ids == expected

    def test_assistant_footer_ids(self, hf_tok):
        """Assistant footer should be [<|im_end|>, \\n]."""
        mock = _run_ensure_cache(hf_tok)
        assert mock._asst_footer_ids == [IM_END, NEWLINE]

    def test_eos_id(self, hf_tok):
        """EOS ID must be set from the tokenizer."""
        mock = _run_ensure_cache(hf_tok)
        assert mock._eos_id == hf_tok.eos_token_id

    def test_eos_in_footer(self, hf_tok):
        """For Qwen3, eos_token_id (<|im_end|>) is the first token of the footer."""
        mock = _run_ensure_cache(hf_tok)
        assert mock._eos_in_footer is True
        assert mock._eos_id == mock._asst_footer_ids[0]

    def test_blank_id(self):
        """Blank token resolves to a real (non-UNK) id once added as a special token."""
        # Add <blank> as a special token (as the model __init__ does)
        hf_tok_copy = AutoTokenizer.from_pretrained(PRETRAINED_LLM)
        hf_tok_copy.add_special_tokens({"additional_special_tokens": [BLANK_TOKEN]})
        mock = _make_mock_self(hf_tok_copy)
        StreamingSTTModel._ensure_inference_cache(mock)
        blank_id = hf_tok_copy.convert_tokens_to_ids(BLANK_TOKEN)
        assert mock.blank_token_id == blank_id
        assert blank_id != hf_tok_copy.unk_token_id

    def test_idempotent(self, hf_tok):
        """Calling _ensure_inference_cache twice should not change results."""
        mock = _make_mock_self(hf_tok)
        StreamingSTTModel._ensure_inference_cache(mock)
        first_template = list(mock._turn_template_ids)
        first_footer = list(mock._asst_footer_ids)
        first_eos = mock._eos_id
        first_eos_in_footer = mock._eos_in_footer

        StreamingSTTModel._ensure_inference_cache(mock)
        assert mock._turn_template_ids == first_template
        assert mock._asst_footer_ids == first_footer
        assert mock._eos_id == first_eos
        assert mock._eos_in_footer == first_eos_in_footer


# ===========================================================================
# Tests: multi chunk-size config (list) + backward compatibility (scalar int)
# ===========================================================================
class TestRepresentativeChunkSize:
    """``_repr_chunk_size`` collapses a config value to a single scalar."""

    def test_scalar_passthrough(self):
        # Backward compatible: plain ints are returned unchanged.
        assert _repr_chunk_size(2) == 2
        assert _repr_chunk_size(0) == 0
        assert _repr_chunk_size(-1) == -1

    def test_list_returns_longest(self):
        assert _repr_chunk_size([2, 6, 13]) == 13
        assert _repr_chunk_size([13, 2, 6]) == 13
        assert _repr_chunk_size((4, 8)) == 8
        assert _repr_chunk_size([5]) == 5


class TestResolveInferenceChunkSize:
    """``_resolve_inference_chunk_size``: override wins, else the config repr."""

    def test_default_uses_repr(self):
        mock = SimpleNamespace(_chunk_size_repr=13)
        assert StreamingSTTModel._resolve_inference_chunk_size(mock, None) == 13

    def test_override_wins(self):
        mock = SimpleNamespace(_chunk_size_repr=13)
        assert StreamingSTTModel._resolve_inference_chunk_size(mock, 6) == 6
        # Override even applies when it exceeds the configured sizes.
        assert StreamingSTTModel._resolve_inference_chunk_size(mock, 20) == 20


class TestBuildTurnTemplateIds:
    """``_build_turn_template_ids`` embeds exactly ``chunk_size`` audio slots."""

    def test_audio_slot_count(self):
        mock = SimpleNamespace(_user_header_ids=[1, 2], _user_footer_and_asst_header_ids=[9])
        for cs in (1, 2, 6, 13):
            t = StreamingSTTModel._build_turn_template_ids(mock, cs)
            assert t.count(AUDIO_TOKEN_IDX) == cs
            # header ... audio block ... footer, contiguous
            assert t[:2] == [1, 2]
            assert t[2 : 2 + cs] == [AUDIO_TOKEN_IDX] * cs
            assert t[2 + cs :] == [9]


class TestEnsureInferenceCacheListChunkSize:
    """A list ``chunk_size`` builds the default template for the longest size,
    while a scalar ``chunk_size`` behaves exactly as before (backward compat)."""

    def test_list_template_uses_longest(self, hf_tok):
        mock = _run_ensure_cache(hf_tok, chunk_size=[2, 6, 13])
        assert mock._turn_template_ids.count(AUDIO_TOKEN_IDX) == 13

    def test_scalar_matches_singleton_list(self, hf_tok):
        scalar = _run_ensure_cache(hf_tok, chunk_size=6)
        as_list = _run_ensure_cache(hf_tok, chunk_size=[6])
        assert scalar._turn_template_ids == as_list._turn_template_ids


class TestEncoderAttContext:
    """``_set_encoder_att_context`` sets right-context = chunk_size - 1, fixed left."""

    def _mock(self, att_context_size=(70, 99)):
        class Enc:
            def __init__(self):
                self.att_context_size = list(att_context_size)
                self.recomputed = False

            def setup_streaming_params(self):
                self.recomputed = True

        enc = Enc()
        s = SimpleNamespace(
            perception=SimpleNamespace(encoder=enc),
            core_cfg=SimpleNamespace(att_context_size=[70, 12]),
        )
        return s, enc

    def test_training_path_sets_lookahead(self):
        s, enc = self._mock()
        StreamingSTTModel._set_encoder_att_context(s, 6)
        assert enc.att_context_size == [70, 5]
        assert enc.recomputed is False  # training does not recompute streaming cfg

    def test_inference_path_recomputes_streaming(self):
        s, enc = self._mock()
        StreamingSTTModel._set_encoder_att_context(s, 13, recompute_streaming=True)
        assert enc.att_context_size == [70, 12]
        assert enc.recomputed is True

    def test_noop_for_dynamic_offline_and_none(self):
        s, enc = self._mock()
        enc.att_context_size = [70, 7]
        for cs in (0, -1, None):
            StreamingSTTModel._set_encoder_att_context(s, cs)
            assert enc.att_context_size == [70, 7]

    def test_noop_when_att_context_unset(self):
        s, enc = self._mock()
        enc.att_context_size = [70, 7]
        s.core_cfg.att_context_size = None
        StreamingSTTModel._set_encoder_att_context(s, 6)
        assert enc.att_context_size == [70, 7]


# ===========================================================================
# Tests: write_token refactor — unified emit gate + dedicated end_of_audio_token
# ===========================================================================
class TestWriteTokenRefactor:
    """``content_score_mode`` / ``_content_score_token_id`` / boundary marker
    behavior after unifying ``write_token`` (emit gate) and introducing
    ``end_of_audio_token`` (compact scaffold anchor, default ``<|im_start|>``)."""

    WRITE_TOKEN = "<|write|>"
    END_OF_AUDIO_TOKEN = "<|im_start|>"

    def _core_cfg(self, **overrides):
        cfg = dict(
            chunk_size=CHUNK_SIZE,
            audio_tag="<audio>",
            compact_template=True,
            write_token=self.WRITE_TOKEN,
            end_of_audio_token=self.END_OF_AUDIO_TOKEN,
            blank_token=BLANK_TOKEN,
            prepend_write_token=True,
            use_chunk_classifier=False,
        )
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def _mock(self, hf_tok, **cfg_overrides):
        return SimpleNamespace(
            tokenizer=SimpleNamespace(tokenizer=hf_tok),
            core_cfg=self._core_cfg(**cfg_overrides),
            blank_token=cfg_overrides.get("blank_token", BLANK_TOKEN),
            blank_token_id=hf_tok.convert_tokens_to_ids(BLANK_TOKEN),
        )

    def test_content_score_mode_binary_in_compact(self, hf_tok):
        """Fixed + compact + prepend_write_token + blank!='' → 'binary' emit gate."""
        mock = self._mock(hf_tok, chunk_size=CHUNK_SIZE, compact_template=True, prepend_write_token=True)
        mode = StreamingSTTModel.content_score_mode.func(mock)
        assert mode == "binary"

    def test_content_score_mode_blank_only_without_gate(self, hf_tok):
        """Fixed + compact + no gate + blank!='' → 'blank_only' (unchanged fallback)."""
        mock = self._mock(hf_tok, chunk_size=CHUNK_SIZE, prepend_write_token=False)
        assert StreamingSTTModel.content_score_mode.func(mock) == "blank_only"

    def test_content_score_token_id_binary_is_write_token(self, hf_tok):
        """'binary' mode reads the write_token id (the emit gate)."""
        mock = self._mock(hf_tok)
        mock.content_score_mode = "binary"
        tid = StreamingSTTModel._content_score_token_id.func(mock)
        assert tid == hf_tok.convert_tokens_to_ids(self.WRITE_TOKEN)

    def test_content_score_token_id_marker_compact_is_end_of_audio(self, hf_tok):
        """Dynamic + compact 'marker' mode reads the end_of_audio_token id, NOT write_token."""
        mock = self._mock(hf_tok, chunk_size=0)
        mock.content_score_mode = "marker"
        tid = StreamingSTTModel._content_score_token_id.func(mock)
        assert tid == hf_tok.convert_tokens_to_ids(self.END_OF_AUDIO_TOKEN)
        assert tid != hf_tok.convert_tokens_to_ids(self.WRITE_TOKEN)

    def test_dynamic_compact_boundary_marker_is_end_of_audio(self, hf_tok):
        """_ensure_inference_cache: dynamic compact boundary == end_of_audio id."""
        mock = _make_mock_self(hf_tok, chunk_size=0)
        mock.core_cfg.compact_template = True
        mock.core_cfg.end_of_audio_token = self.END_OF_AUDIO_TOKEN
        mock.core_cfg.write_token = self.WRITE_TOKEN
        StreamingSTTModel._ensure_inference_cache(mock)
        assert mock._user_footer_first_id == hf_tok.convert_tokens_to_ids(self.END_OF_AUDIO_TOKEN)


class TestStreamingBufferFrameArithmetic:
    """Frame/sample counts for the streaming audio buffer must be exact integer arithmetic.

    ``BatchedCacheFeatureBufferer`` takes durations in float seconds and recovers integer frame and
    sample counts from them by truncation, so any duration accumulated in seconds can silently lose
    a frame. Two concrete regressions this guards:
      * ``pre_encode*0.01 + chunk*0.08`` -> ``int(secs/0.01)`` undercounts (chunk 29/58/59 at
        pre_encode 0; chunk 6/9 at pre_encode 8), misaligning every frame of every chunk.
      * ``ceil(chunk*0.08*16000)`` overshoots by one sample (chunk 35/41/47/57), which exceeds the
        audio buffer and raises AudioBufferer's "Frame size exceeds buffer size".
    """

    WS, SR, FRAME_LEN = 0.01, 16000, 0.08

    def _mock(self, frame_length_in_secs=None, window_stride=None):
        mock = SimpleNamespace(
            core_cfg=SimpleNamespace(frame_length_in_secs=frame_length_in_secs or self.FRAME_LEN, sample_rate=self.SR),
            perception=SimpleNamespace(
                cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=window_stride or self.WS))
            ),
        )
        # _samples_per_encoder_frame calls this on self; bind it for the namespace stand-in.
        mock._mel_frames_per_encoder_frame = lambda: StreamingSTTModel._mel_frames_per_encoder_frame(mock)
        return mock

    def test_duration_for_frames_round_trips(self):
        """The duration must read back as exactly the frame count it was built from."""
        for n in range(1, 2001):
            secs = _duration_for_frames(n, self.WS)
            assert int(secs / self.WS) == n, f"{n} frames -> {secs!r} -> {int(secs / self.WS)}"

    def test_duration_for_frames_fixes_naive_multiplication(self):
        """Frame counts where the naive product truncates low — the reason the helper exists."""
        for n in (29, 58, 59, 116, 232):
            assert int((n * self.WS) / self.WS) < n  # naive is wrong ...
            assert int(_duration_for_frames(n, self.WS) / self.WS) == n  # ... helper is not

    def test_duration_for_frames_does_not_move_sample_count(self):
        """The ULP nudge is far below one sample, so derived sample counts are untouched."""
        for n in range(1, 501):
            secs = _duration_for_frames(n, self.WS)
            assert int(secs * self.SR) == n * round(self.WS * self.SR)

    def test_mel_frames_per_encoder_frame(self):
        assert StreamingSTTModel._mel_frames_per_encoder_frame(self._mock()) == 8
        assert StreamingSTTModel._mel_frames_per_encoder_frame(self._mock(frame_length_in_secs=0.04)) == 4

    def test_non_integral_frame_ratio_raises(self):
        """Encoder frames that do not land on whole mel frames are a config error, not a rounding job."""
        with pytest.raises(ValueError, match="whole multiple"):
            StreamingSTTModel._mel_frames_per_encoder_frame(self._mock(frame_length_in_secs=0.085))

    def test_samples_per_encoder_frame_is_exact(self):
        mock = self._mock()
        assert StreamingSTTModel._samples_per_encoder_frame(mock) == 1280
        # ceil() overshot by one sample at these chunk sizes; exact arithmetic must not.
        for cs in (35, 41, 47, 57):
            assert cs * StreamingSTTModel._samples_per_encoder_frame(mock) == cs * 1280

    @pytest.mark.parametrize("pre_encode", [0, 8, 16])
    @pytest.mark.parametrize("chunk_size", [1, 6, 9, 14, 28, 29, 35, 41, 47, 57, 58, 59, 64])
    def test_buffer_frames_and_samples_agree(self, pre_encode, chunk_size):
        """The buffer must hold exactly the intended frames, and a chunk must fit inside it."""
        mock = self._mock()
        want_frames = pre_encode + chunk_size * StreamingSTTModel._mel_frames_per_encoder_frame(mock)
        secs = _duration_for_frames(want_frames, self.WS)
        assert int(secs / self.WS) == want_frames
        capacity_samples = int(secs * self.SR)
        assert capacity_samples == want_frames * round(self.WS * self.SR)
        chunk_samples = chunk_size * StreamingSTTModel._samples_per_encoder_frame(mock)
        assert chunk_samples <= capacity_samples


class TestContextFlagGuards:
    """M6: the context knobs must never apply silently or half-apply.

    Both failure modes are invisible at runtime. A model/dataset mismatch trains the
    emit gate at one position and reads it at another, so the model just stops
    emitting; and the fast inference path would feed the unmodified context while
    reporting numbers that look valid.
    """

    @staticmethod
    def _mock(drop=False, collapse=False, blank="<blank>", compact=True, chunk_size=2):
        return SimpleNamespace(
            core_cfg=SimpleNamespace(
                drop_blank_from_context=drop,
                collapse_silent_audio=collapse,
                compact_template=compact,
                chunk_size=chunk_size,
            ),
            blank_token=blank,
            # ``has_blank`` is a property on the real class, derived from blank_token;
            # mirror that derivation rather than pinning it independently.
            has_blank=blank != "",
        )

    def test_flags_resolve_when_preconditions_hold(self):
        m = self._mock(drop=True)
        assert StreamingSTTModel._resolve_context_flags(m, 2) == (True, False)

    def test_collapse_implies_drop(self):
        m = self._mock(collapse=True)
        assert StreamingSTTModel._resolve_context_flags(m, 2) == (True, True)

    def test_off_resolves_to_off_without_warning(self):
        """R2: with the flags off nothing may change, including no spurious warnings."""
        import warnings as _w

        m = self._mock()
        with _w.catch_warnings():
            _w.simplefilter("error")
            assert StreamingSTTModel._resolve_context_flags(m, 2) == (False, False)

    @pytest.mark.parametrize(
        "kwargs,chunk_size,reason",
        [
            ({"drop": True}, 0, "not fixed chunking"),
            ({"drop": True, "blank": ""}, 2, "blank_token is empty"),
            ({"drop": True, "compact": False}, 2, "compact_template=False"),
        ],
    )
    def test_downgrade_is_loud(self, kwargs, chunk_size, reason):
        """A silent downgrade would run the unmodified format while the config says otherwise."""
        m = self._mock(**kwargs)
        with pytest.warns(UserWarning, match="cannot apply"):
            assert StreamingSTTModel._resolve_context_flags(m, chunk_size) == (False, False)

    @pytest.mark.parametrize("flag", ["drop_blank_from_context", "collapse_silent_audio"])
    def test_model_dataset_mismatch_raises(self, flag):
        m = SimpleNamespace(core_cfg=SimpleNamespace(drop_blank_from_context=False, collapse_silent_audio=False))
        setattr(m.core_cfg, flag, True)
        with pytest.raises(ValueError, match="must match"):
            StreamingSTTModel._assert_context_flags_match_data(m, {flag: False})

    def test_matching_flags_are_accepted(self):
        m = SimpleNamespace(core_cfg=SimpleNamespace(drop_blank_from_context=True, collapse_silent_audio=False))
        StreamingSTTModel._assert_context_flags_match_data(
            m, {"drop_blank_from_context": True, "collapse_silent_audio": False}
        )

    def test_no_data_cfg_is_not_an_error(self):
        """Inference-only construction passes no dataset config."""
        m = SimpleNamespace(core_cfg=SimpleNamespace(drop_blank_from_context=True, collapse_silent_audio=False))
        StreamingSTTModel._assert_context_flags_match_data(m, None)
