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

"""Tests for the cache-aware streaming variant of the parallel expert encoder.

The load-bearing claim is that the subclass adds **no behaviour**: its streaming step is its ASR
branch's step plus fusion, bit for bit, and its offline forward is the base class's.
"""

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder, StreamingParallelExpertEncoder
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from tests.collections.asr.test_parallel_expert_encoder import (
    _MEL_FEATURES,
    _N_SPK,
    _SUBSAMPLING_FACTOR,
    toy_asr_encoder_cfg,
    toy_diarization_model_cfg,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Pin this module's default device to CPU, and restore it afterwards.

    Several ``tests/collections/speechlm2`` modules call ``torch.set_default_device('cuda')`` at
    import time and never restore it, so in a full-suite run this file would otherwise inherit a
    CUDA default and mix devices.
    """
    previous = torch.get_default_device()
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(previous)


def streaming_asr_encoder_cfg() -> DictConfig:
    """Tiny *cache-aware* ConformerEncoder config, so the streaming interface is exercisable."""
    cfg = toy_asr_encoder_cfg()
    # [12, 3] -> a 32-mel chunk. [8, 1] gives 16, which the toy Sortformer CANNOT pre-encode
    # (feature_stacking x8 over a 16-frame chunk minus the 9-frame cache leaves nothing), so every
    # streaming test had to stub the diarizer out and no real diarizer step was ever exercised.
    # That is why several streaming-diarization bugs shipped. `chunked_limited` additionally
    # requires left % (right + 1) == 0, and 12 % 4 == 0.
    cfg.att_context_size = [12, 3]
    cfg.att_context_style = 'chunked_limited'
    cfg.causal_downsampling = True
    cfg.conv_context_size = 'causal'
    return cfg


def build_toy_streaming_pe_encoder(**overrides):
    """A real StreamingParallelExpertEncoder over the tiny cache-aware ASR + diar configs."""
    kwargs = dict(
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type=None,
        online_inference_length=500,
    )
    kwargs.update(overrides)
    return StreamingParallelExpertEncoder(**kwargs)


@pytest.mark.unit
def test_only_the_streaming_subclass_advertises_the_streaming_interface():
    """The base class must NOT claim streaming capability -- that is the point of the split."""
    base = ParallelExpertEncoder(
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type=None,
    )
    assert not isinstance(base, StreamingEncoder)
    assert not hasattr(base, "cache_aware_stream_step")

    enc = build_toy_streaming_pe_encoder()
    assert isinstance(enc, StreamingEncoder)
    # ...while still satisfying the `isinstance(..., ParallelExpertEncoder)` gate in salm_automodel.
    assert isinstance(enc, ParallelExpertEncoder)
    enc.setup_streaming_params()
    # streaming_cfg is the ASR branch's, not a copy that can drift out of sync.
    assert enc.streaming_cfg is enc.asr_encoder.streaming_cfg


@pytest.mark.unit
def test_streaming_step_delegates_exactly_to_asr_branch():
    """PE's cache-aware step must be its ASR branch's step plus fusion -- nothing else. Any extra
    transform of the signal (e.g. a stray re-normalization) shows up here as a nonzero diff."""
    enc = build_toy_streaming_pe_encoder(speaker_activity_threshold=0.5).eval()
    enc._fuse_diar_and_asr = lambda asr_encoded, spk_targets: asr_encoded
    bare = enc.asr_encoder
    enc.setup_streaming_params()

    torch.manual_seed(0)
    mel = torch.randn(1, _MEL_FEATURES, 512)
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    shift = enc.streaming_cfg.shift_size
    shift = shift[1] if isinstance(shift, (list, tuple)) else shift

    pe_state = list(enc.get_initial_cache_state(batch_size=1, dtype=mel.dtype, device=mel.device))
    bare_state = list(bare.get_initial_cache_state(batch_size=1, dtype=mel.dtype, device=mel.device))
    assert all(torch.equal(a, b) for a, b in zip(pe_state, bare_state))

    n_steps = 0
    with torch.no_grad():
        for step in range(3):
            chunk = mel[:, :, step * shift : step * shift + chunk_size]
            if chunk.shape[-1] < chunk_size:
                break
            kwargs = dict(
                processed_signal=chunk,
                processed_signal_length=torch.tensor([chunk.shape[-1]]),
                keep_all_outputs=False,
                drop_extra_pre_encoded=0 if step == 0 else enc.streaming_cfg.drop_extra_pre_encoded,
            )
            pe_out = enc.cache_aware_stream_step(
                cache_last_channel=pe_state[0],
                cache_last_time=pe_state[1],
                cache_last_channel_len=pe_state[2],
                spk_targets=torch.zeros(1, chunk_size // _SUBSAMPLING_FACTOR, _N_SPK),
                **kwargs,
            )
            bare_out = bare.cache_aware_stream_step(
                cache_last_channel=bare_state[0],
                cache_last_time=bare_state[1],
                cache_last_channel_len=bare_state[2],
                **kwargs,
            )
            assert torch.equal(pe_out[0], bare_out[0]), f"encoder output diverged at step {step}"
            for pe_cache, bare_cache in zip(pe_out[2:4], bare_out[2:4]):
                assert torch.equal(pe_cache, bare_cache), f"cache diverged at step {step}"
            pe_state, bare_state = list(pe_out[2:]), list(bare_out[2:])
            n_steps += 1
    assert n_steps > 0, "test exercised no streaming steps"


@pytest.mark.unit
def test_offline_forward_is_inherited_unchanged():
    """Mounting the streaming subclass must not perturb the offline path."""
    torch.manual_seed(0)
    common = dict(
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type=None,
        online_inference_length=500,
    )
    torch.manual_seed(0)
    base = ParallelExpertEncoder(**common).eval()
    torch.manual_seed(0)
    streaming = StreamingParallelExpertEncoder(**common).eval()
    streaming.load_state_dict(base.state_dict())

    mel = torch.randn(1, _MEL_FEATURES, 128)
    length = torch.tensor([mel.shape[-1]])
    spk_targets = torch.rand(1, mel.shape[-1] // _SUBSAMPLING_FACTOR, _N_SPK)
    with torch.no_grad():
        base_out, base_len = base(audio_signal=mel, length=length, spk_targets=spk_targets.clone())
        strm_out, strm_len = streaming(audio_signal=mel, length=length, spk_targets=spk_targets.clone())
    assert torch.equal(base_out, strm_out)
    assert torch.equal(base_len, strm_len)


@pytest.mark.unit
def test_diarizer_drops_the_same_pre_encode_frames_as_the_asr_branch():
    """Both branches must consume the same `drop_extra_pre_encoded`, or the fusion goes stale.

    The ASR branch drops N cache frames from its output; if the diarizer does not, it emits N+2
    frames per N ASR frames and `_align_diar_frames` keeps the OLDEST ones -- so the speaker signal
    ends up one chunk behind the ASR frames it is added to. `perception.forward` does not pass
    `drop_extra_pre_encoded`, so PE must fall back to the ASR branch's streaming config, not to 0.
    """
    enc = build_toy_streaming_pe_encoder().eval()
    enc.setup_streaming_params()
    state = enc.get_initial_cache_state(batch_size=1, dtype=torch.float32, device=torch.device("cpu"))
    cache_last_channel, cache_last_time, cache_last_channel_len = state

    seen = []

    original = enc._stream_diarizer

    def spy(processed_signal, processed_signal_length, align_target, drop_extra_pre_encoded):
        seen.append(drop_extra_pre_encoded)
        return original(processed_signal, processed_signal_length, align_target, drop_extra_pre_encoded)

    enc._stream_diarizer = spy
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    mel = torch.randn(1, _MEL_FEATURES, chunk_size)
    with torch.no_grad():
        # No `drop_extra_pre_encoded` argument -- exactly how perception.forward calls it.
        enc.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=torch.tensor([chunk_size]),
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
        )
    assert seen == [enc.streaming_cfg.drop_extra_pre_encoded], f"diarizer got drop={seen}, not the ASR branch's"


@pytest.mark.unit
def test_stream_step_without_initial_cache_state_raises_actionable_error():
    """`nn.Module.__getattr__` would raise AttributeError and mask the real instruction."""
    enc = build_toy_streaming_pe_encoder().eval()
    enc.setup_streaming_params()
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    with pytest.raises(RuntimeError, match="get_initial_cache_state"):
        enc.cache_aware_stream_step(
            processed_signal=torch.randn(1, _MEL_FEATURES, chunk_size),
            processed_signal_length=torch.tensor([chunk_size]),
        )


@pytest.mark.unit
def test_subset_stepping_reports_the_batch_mismatch_clearly():
    """Stepping a SUBSET of streams must fail with an actionable message, not a tensor-size error.

    `_generate_dynamic_streaming` slices the ASR cache to the streams needing a refill and scatters
    the result back. The diarizer keeps ONE batched state on the module, so it cannot follow --
    previously this surfaced as `RuntimeError: Sizes of tensors must match except in dimension 1`
    from deep inside the Sortformer, which says nothing about the cause.

    This pins the diagnosis. It should be replaced by a real subset-stepping test if the state is
    ever made sliceable (see PLAN section 0.0, next step 1).
    """
    enc = build_toy_streaming_pe_encoder().eval()
    enc.setup_streaming_params()
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size

    batch = 3
    cache_last_channel, cache_last_time, cache_last_channel_len = enc.get_initial_cache_state(
        batch_size=batch, dtype=torch.float32, device=torch.device("cpu")
    )
    with torch.no_grad():
        _, _, cache_last_channel, cache_last_time, cache_last_channel_len = enc.cache_aware_stream_step(
            processed_signal=torch.randn(batch, _MEL_FEATURES, chunk_size),
            processed_signal_length=torch.tensor([chunk_size] * batch),
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
        )

    subset = torch.tensor([0, 2])  # stream 1 is still generating, as in the dynamic FSM
    with pytest.raises(RuntimeError, match="steps a subset of streams"):
        with torch.no_grad():
            enc.cache_aware_stream_step(
                processed_signal=torch.randn(len(subset), _MEL_FEATURES, chunk_size),
                processed_signal_length=torch.tensor([chunk_size] * len(subset)),
                cache_last_channel=cache_last_channel.index_select(1, subset),
                cache_last_time=cache_last_time.index_select(1, subset),
                cache_last_channel_len=cache_last_channel_len[subset],
                keep_all_outputs=False,
            )


@pytest.mark.unit
def test_an_empty_diarizer_step_is_named_not_left_to_the_fusion():
    """A chunk too short for the diarizer must fail here, with the cause in the message.

    ``_align_diar_frames`` pads by repeating the last frame, and repeating a zero-width tensor
    gives another zero-width tensor. Without this guard an empty diarizer step would flow into
    ``_fuse_diar_and_asr`` and surface as a shape mismatch deep in the fusion, naming neither the
    diarizer nor the chunk size that caused it.

    Unreachable in every measured configuration -- the diarizer and the ASR branch agree
    frame-for-frame -- which is exactly when a guard is cheap to add and cheap to keep.
    """
    enc = build_toy_streaming_pe_encoder().eval()
    enc.setup_streaming_params()
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    batch = 2
    cache_last_channel, cache_last_time, cache_last_channel_len = enc.get_initial_cache_state(
        batch_size=batch, dtype=torch.float32, device=torch.device("cpu")
    )

    # Make the diarizer return without adding frames, which is what a too-short chunk does.
    def _no_new_frames(processed_signal, processed_signal_length, streaming_state, total_preds, **kwargs):
        return streaming_state, total_preds

    enc.diarization_model.forward_streaming_step = _no_new_frames

    with pytest.raises(RuntimeError, match="produced no new frames"), torch.no_grad():
        enc.cache_aware_stream_step(
            processed_signal=torch.randn(batch, _MEL_FEATURES, chunk_size),
            processed_signal_length=torch.tensor([chunk_size] * batch),
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
        )


class TestDiarizationGating:
    """The diarizer's run/skip decision must not depend on what data a rank received.

    Skipping when no row carries the missing-RTTM sentinel is a sound optimisation for one
    process. Under DDP/FSDP it is a hang: mix an RTTM-backed corpus with one that has none, and a
    rank whose batch happens to be entirely RTTM-backed skips a module forward that its peers run.
    Collectives then desynchronise, and the job stalls instead of failing.
    """

    @staticmethod
    def _encoder():
        return build_toy_streaming_pe_encoder().eval()

    @staticmethod
    def _targets(n_missing, batch=3, frames=6, n_spk=_N_SPK):
        targets = torch.zeros(batch, frames, n_spk)
        targets[:n_missing] = -1.0  # the missing-RTTM sentinel
        return targets

    @pytest.mark.unit
    def test_skips_when_every_row_has_real_targets(self):
        """The fast path this optimisation exists for: single process, nothing to infer."""
        assert self._encoder()._should_run_diarization(self._targets(n_missing=0)) is False

    @pytest.mark.unit
    def test_runs_when_any_row_carries_the_sentinel(self):
        assert self._encoder()._should_run_diarization(self._targets(n_missing=1)) is True

    @pytest.mark.unit
    def test_runs_when_there_are_no_targets_at_all(self):
        assert self._encoder()._should_run_diarization(None) is True

    @pytest.mark.unit
    def test_training_always_runs_it(self):
        """Otherwise one rank's batch composition decides, and ranks diverge."""
        enc = self._encoder().train()
        assert enc._should_run_diarization(self._targets(n_missing=0)) is True

    @pytest.mark.unit
    def test_distributed_always_runs_it(self, monkeypatch):
        """Even in eval: inference with world_size > 1 collects across ranks too."""
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        assert self._encoder()._should_run_diarization(self._targets(n_missing=0)) is True

    @pytest.mark.unit
    def test_single_rank_distributed_keeps_the_fast_path(self, monkeypatch):
        """world_size == 1 has no peer to desynchronise from."""
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 1)
        assert self._encoder()._should_run_diarization(self._targets(n_missing=0)) is False
