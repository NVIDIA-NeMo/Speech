# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Integration tests for StreamingSALM model (Phase 4 — T8)."""

import os

import pytest
import torch
import torch.nn as nn

from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.speechlm2.parts.interleaving import WordAlignment

if torch.cuda.is_available():
    torch.set_default_device("cuda")


# ---------------------------------------------------------------------------
# Mocking helpers — avoid downloading real pretrained models
# ---------------------------------------------------------------------------

class FakeEncoderOutput:
    def __init__(self, codes):
        self.audio_codes = codes


class FakeMimiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def encode(self, audio, padding_mask=None):
        B, channels, T = audio.shape  # HF Mimi expects (B, channels, T)
        num_frames = max(1, T // 1920)
        codes = torch.randint(0, 2048, (B, 8, num_frames))
        return FakeEncoderOutput(codes)


class FakeFeatureExtractor:
    pass


class FakeAlignmentResult:
    def __init__(self, text, start, end):
        self.text = text
        self.start_time = start
        self.end_time = end


class FakeAligner:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def align(self, audio, text, language):
        results = []
        for t in text:
            words = t.split()
            time = 0.0
            word_results = []
            for w in words:
                duration = len(w) * 0.1
                word_results.append(FakeAlignmentResult(w, time, time + duration))
                time += duration + 0.05
            results.append(word_results)
        return results


def resolve_pretrained_models():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1",
        }
    else:
        return {
            "pretrained_llm": "TinyLlama/TinyLlama_v1.1",
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model(tmp_path_factory, monkeypatch_session):
    """Build StreamingSALM with mocked Mimi + QFA and real TinyLlama."""
    # Mock Mimi
    monkeypatch_session.setattr(
        "transformers.MimiModel.from_pretrained",
        staticmethod(lambda *a, **kw: FakeMimiModel()),
    )
    monkeypatch_session.setattr(
        "transformers.AutoFeatureExtractor.from_pretrained",
        staticmethod(lambda *a, **kw: FakeFeatureExtractor()),
    )
    # Mock QFA
    import qwen_asr
    monkeypatch_session.setattr(qwen_asr, "Qwen3ForcedAligner", FakeAligner)

    from nemo.collections.speechlm2.models.streaming_salm import StreamingSALM

    pretrained = resolve_pretrained_models()
    cfg = {
        "pretrained_llm": pretrained["pretrained_llm"],
        "pretrained_mimi": "fake_mimi",
        "pretrained_forced_aligner": "fake_qfa",
        "pretrained_weights": True,
        "blank_token": "<blank>",
        "num_codebooks": 2,
        "min_latency": 1,
        "max_latency": 3,
        "context_biasing_prob": 0.5,
        "cache_sink_size": 8,
        "cache_window_size": 32,
        "freeze_params": [],
        "prevent_freeze_params": [],
    }
    m = StreamingSALM(cfg)
    if torch.cuda.is_available():
        m = m.to("cuda")
    return m


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def mock_batch(model):
    """Minimal batch dict that prepare_inputs expects."""
    device = model.device
    B = 2
    # 1 second of audio at 16kHz
    audio_len = 16000
    return {
        "audios": torch.randn(B, audio_len, device=device),
        "audio_lens": torch.tensor([audio_len, audio_len], device=device),
        "transcripts": ["hello world", "good morning"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamingSALMInit:
    def test_model_has_required_components(self, model):
        assert hasattr(model, "llm")
        assert hasattr(model, "embed_tokens")
        assert hasattr(model, "mimi")
        assert hasattr(model, "forced_aligner")
        assert hasattr(model, "audio_embeddings")

    def test_blank_token_in_tokenizer(self, model):
        assert model.blank_token_id is not None
        assert model.blank_token_id > 0

    def test_audio_embeddings_count(self, model):
        assert len(model.audio_embeddings) == model.num_codebooks

    def test_audio_embedding_dimensions(self, model):
        for emb in model.audio_embeddings:
            assert emb.num_embeddings == model.audio_codebook_size + 1
            assert emb.embedding_dim == model.llm.config.hidden_size


class TestEmbedAudioCodes:
    def test_output_shape(self, model):
        B, K, T = 2, model.num_codebooks, 10
        codes = torch.randint(0, model.audio_codebook_size, (B, K, T), device=model.device)
        embs = model.embed_audio_codes(codes)
        assert embs.shape == (B, T, model.llm.config.hidden_size)

    def test_padding_idx_produces_zeros(self, model):
        pad = model.audio_codebook_size
        codes = torch.full((1, model.num_codebooks, 3), pad, device=model.device)
        embs = model.embed_audio_codes(codes)
        assert (embs == 0).all()


class TestPrepareInputs:
    def test_output_keys(self, model, mock_batch):
        inputs = model.prepare_inputs(mock_batch)
        assert "input_embeds" in inputs
        assert "attention_mask" in inputs
        assert "labels" in inputs

    def test_shapes_consistent(self, model, mock_batch):
        inputs = model.prepare_inputs(mock_batch)
        B, T, H = inputs["input_embeds"].shape
        assert inputs["attention_mask"].shape == (B, T)
        assert inputs["labels"].shape == (B, T)

    def test_labels_contain_blanks_and_text(self, model, mock_batch):
        inputs = model.prepare_inputs(mock_batch)
        labels = inputs["labels"]
        valid = labels[labels != -100]
        assert (valid == model.blank_token_id).any()


class TestTrainingStep:
    def test_loss_is_finite(self, model, mock_batch):
        batch = move_data_to_device(mock_batch, model.device)
        result = model.training_step(batch, batch_idx=0)
        assert torch.isfinite(result["loss"])
        assert result["loss"] > 0

    def test_backward_pass(self, model, mock_batch):
        batch = move_data_to_device(mock_batch, model.device)
        model.zero_grad()
        result = model.training_step(batch, batch_idx=0)
        result["loss"].backward()
        for emb in model.audio_embeddings:
            assert emb.weight.grad is not None


class TestGenerate:
    def test_offline_generation_returns_strings(self, model):
        audio = torch.randn(1, 16000, device=model.device)
        audio_lens = torch.tensor([16000], device=model.device)
        results = model.generate(audio, audio_lens, latency=1)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_offline_generation_with_context(self, model):
        audio = torch.randn(1, 16000, device=model.device)
        audio_lens = torch.tensor([16000], device=model.device)
        results = model.generate(audio, audio_lens, latency=2, context="hello")
        assert isinstance(results, list)

    def test_offline_generation_flush_runs(self, model):
        """With high latency, generate() should run flush steps after audio ends."""
        audio = torch.randn(1, 16000, device=model.device)
        audio_lens = torch.tensor([16000], device=model.device)
        # Both latency values should produce valid output (not crash);
        # the flush ensures high-latency doesn't silently drop trailing tokens.
        result_k1 = model.generate(audio, audio_lens, latency=1)
        result_k5 = model.generate(audio, audio_lens, latency=5)
        assert isinstance(result_k1[0], str)
        assert isinstance(result_k5[0], str)


class TestStreamingGenerate:
    def test_new_session_initialization(self, model):
        from nemo.collections.speechlm2.models.streaming_salm import StreamingState

        codes = torch.randint(0, 2048, (1, model.num_codebooks, 5), device=model.device)
        tokens, state = model.generate_streaming(codes, state=None, latency=2)
        assert isinstance(state, StreamingState)
        assert state.latency == 2
        assert state.num_processed_frames == 5

    def test_incremental_processing(self, model):
        state = None
        total_frames = 0
        for chunk_size in [3, 5, 2]:
            codes = torch.randint(0, 2048, (1, model.num_codebooks, chunk_size), device=model.device)
            tokens, state = model.generate_streaming(codes, state=state, latency=1)
            total_frames += chunk_size
            assert state.num_processed_frames == total_frames

    def test_flush_returns_list(self, model):
        codes = torch.randint(0, 2048, (1, model.num_codebooks, 3), device=model.device)
        _, state = model.generate_streaming(codes, state=None, latency=1)
        tokens, state2 = model.generate_streaming(None, state=state)
        # Flush should return a list of lists (possibly with tokens from latency buffer)
        assert isinstance(tokens, list)
        assert isinstance(tokens[0], list)
        # abs_position should advance (flush does decoding steps)
        assert state2.abs_position >= state.abs_position

    def test_cache_eviction_during_streaming(self, model):
        state = None
        for _ in range(100):
            codes = torch.randint(0, 2048, (1, model.num_codebooks, 1), device=model.device)
            _, state = model.generate_streaming(codes, state=state, latency=1)
        for k, v in state.kv_cache:
            assert k.shape[2] <= model.cache_sink_size + model.cache_window_size + 2

    def test_multiple_sessions_independent(self, model):
        codes_a = torch.randint(0, 2048, (1, model.num_codebooks, 5), device=model.device)
        codes_b = torch.randint(0, 2048, (1, model.num_codebooks, 3), device=model.device)
        tokens_a, state_a = model.generate_streaming(codes_a, None, latency=1)
        tokens_b, state_b = model.generate_streaming(codes_b, None, latency=3)
        assert state_a.latency == 1
        assert state_b.latency == 3
        assert state_a.num_processed_frames == 5
        assert state_b.num_processed_frames == 3

    def test_abs_position_tracks_correctly(self, model):
        """Verify abs_position increments by total tokens processed (audio + text feedback)."""
        codes = torch.randint(0, 2048, (1, model.num_codebooks, 3), device=model.device)
        _, state = model.generate_streaming(codes, state=None, latency=1)
        # abs_position = prompt_len + (audio frames + any text feedback steps)
        # It should be at least prompt_len + 3 (audio frames) and at most prompt_len + 6 (if every frame emitted text)
        prompt_len = state.sink_size  # sink_size is set to prompt_len
        assert state.abs_position >= prompt_len + 3
        assert state.abs_position <= prompt_len + 6


class TestUnshiftedLoss:
    """Verify the unshifted loss invariant: logits[i] predicts labels[i]."""

    def test_labels_and_inputs_same_length(self, model, mock_batch):
        """The number of label positions must equal the number of input positions."""
        inputs = model.prepare_inputs(mock_batch)
        B, T, H = inputs["input_embeds"].shape
        assert inputs["labels"].shape == (B, T), (
            f"Labels shape {inputs['labels'].shape} != input shape ({B}, {T}). "
            "This indicates a shift mismatch."
        )

    def test_prompt_positions_masked_in_labels(self, model, mock_batch):
        """Prompt positions in labels should be -100 (ignore_index)."""
        inputs = model.prepare_inputs(mock_batch)
        labels = inputs["labels"]
        attn = inputs["attention_mask"]
        # For each sample, the first non-padding position is where the prompt starts.
        # All prompt positions should have label = -100.
        for b in range(labels.shape[0]):
            valid_start = attn[b].long().argmax().item()  # first True position
            # There must be at least some -100 labels at the start (the prompt)
            prompt_labels = labels[b, valid_start:]
            leading_masked = 0
            for v in prompt_labels:
                if v.item() == -100:
                    leading_masked += 1
                else:
                    break
            assert leading_masked > 0, "Prompt positions should be masked with -100"

    def test_valid_labels_are_text_or_blank(self, model, mock_batch):
        """Valid (non-masked) labels should be either text token IDs or blank_id."""
        inputs = model.prepare_inputs(mock_batch)
        labels = inputs["labels"]
        valid = labels[labels != -100]
        assert len(valid) > 0, "No valid labels found"
        for v in valid:
            tok_id = v.item()
            assert tok_id >= 0, f"Label {tok_id} is negative but not -100"
            assert tok_id < model.text_vocab_size or tok_id == model.blank_token_id, (
                f"Label {tok_id} is out of vocabulary range"
            )

    def test_interleaved_pattern_has_blanks_after_text_feedback(self, model, mock_batch):
        """After a text token label, the next label should be blank (for the fed-back text position)."""
        inputs = model.prepare_inputs(mock_batch)
        labels = inputs["labels"]
        blank_id = model.blank_token_id

        for b in range(labels.shape[0]):
            seq = labels[b]
            valid_mask = seq != -100
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) < 2:
                continue

            # Find text tokens (non-blank, non-masked)
            for i in range(len(valid_indices) - 1):
                idx = valid_indices[i].item()
                next_idx = valid_indices[i + 1].item()
                tok = seq[idx].item()
                next_tok = seq[next_idx].item()

                # If this is a text token (not blank), the next valid label should be blank
                # (it's the fed-back text position)
                if tok != blank_id and next_idx == idx + 1:
                    assert next_tok == blank_id, (
                        f"After text token {tok} at position {idx}, "
                        f"expected blank at position {next_idx} but got {next_tok}"
                    )


class TestSampleRateValidation:
    def test_wrong_sample_rate_raises(self, model):
        """Batch with wrong sample_rate should raise AssertionError."""
        batch = {
            "audios": torch.randn(1, 16000, device=model.device),
            "audio_lens": torch.tensor([16000], device=model.device),
            "transcripts": ["hello"],
            "sample_rate": 16000,
        }
        with pytest.raises(AssertionError, match="24000"):
            model.prepare_inputs(batch)

    def test_correct_sample_rate_passes(self, model, mock_batch):
        """Batch with correct sample_rate should not raise."""
        mock_batch["sample_rate"] = 24000
        # Should not raise
        inputs = model.prepare_inputs(mock_batch)
        assert "input_embeds" in inputs
