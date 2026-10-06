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

from nemo.collections.asr.modules.parallel_expert_encoder import (
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    StreamingParallelExpertEncoder,
    StreamingParallelExpertEncoderPT,
)
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from tests.collections.asr.test_parallel_expert_encoder import (
    _MEL_FEATURES,
    _N_SPK,
    _SUBSAMPLING_FACTOR,
    _UNSET_SENTINEL,
    WINDOWED_PE_KWARGS,
    _capture_pe_warnings,
    build_toy_pe_encoder,
    capture_diarizer_inputs,
    diarizer_cfg_with_normalize,
    hf_exported_pe_config,
    toy_asr_encoder_cfg,
    toy_bundle_config,
    toy_diarization_model_cfg,
    write_toy_branch_checkpoints,
    write_toy_bundle,
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
@pytest.mark.parametrize("with_targets", [True, False], ids=["oracle_targets", "diarizer_predictions"])
def test_streaming_bundle_shell_restores_the_same_weights_and_offline_outputs(tmp_path, with_targets):
    """Compatibility pin for mounting bundles as the streaming class (StreamingSTT, P-2).

    The same ``.nemo`` strict-loads through both shells with an identical state dict, and the offline
    forward is the same, so the class change only adds the streaming interface. Both classes replay the
    toy diarizer's own ``per_feature`` on its input, over the whole utterance.
    """
    torch.manual_seed(0)
    bundle = write_toy_bundle(
        tmp_path / 'pe.nemo',
        encoder=build_toy_streaming_pe_encoder(),
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        asr_normalize_type=None,
    )
    base = ParallelExpertEncoderPT.load_from_nemo(bundle, strict=True).eval()
    streaming = StreamingParallelExpertEncoderPT.load_from_nemo(bundle, strict=True).eval()
    assert isinstance(base, ParallelExpertEncoder) and not isinstance(base, StreamingEncoder)
    assert isinstance(streaming, StreamingParallelExpertEncoder)

    base_state, streaming_state = base.state_dict(), streaming.state_dict()
    assert list(base_state) == list(streaming_state)
    assert all(torch.equal(base_state[key], streaming_state[key]) for key in base_state)

    mel = torch.randn(2, _MEL_FEATURES, 128)
    length = torch.tensor([128, 96])
    spk_targets = torch.rand(2, 128 // _SUBSAMPLING_FACTOR, _N_SPK) if with_targets else None
    with torch.no_grad():
        base_out, base_len = base(audio_signal=mel, length=length, spk_targets=spk_targets)
        strm_out, strm_len = streaming(audio_signal=mel, length=length, spk_targets=spk_targets)
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


# ==============================================================================================
# Oracle targets with missing-RTTM sentinel rows (per-row diarizer fallback in the streaming step)
# ==============================================================================================

_SENTINEL = -1.0
_N_CHUNKS = 3


def _stream(enc, mel, targets=None, sentinel_rows=(), sentinel_chunks=range(_N_CHUNKS)):
    """Step ``enc`` over the first ``_N_CHUNKS`` chunks of ``mel`` and return each chunk's output.

    ``targets`` is a ``(B, T, n_spk)`` oracle tensor on the encoder-frame grid, sliced per chunk the
    way StreamingSTT slices it. The last slice is a single frame, shorter than the chunk's output, as
    at the end of a stream. ``sentinel_rows`` are overwritten with the missing-RTTM sentinel on ``sentinel_chunks``.
    """
    batch = mel.shape[0]
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    shift = enc.streaming_cfg.shift_size
    shift = shift[1] if isinstance(shift, (list, tuple)) else shift
    frames = chunk_size // _SUBSAMPLING_FACTOR
    state = list(enc.get_initial_cache_state(batch_size=batch, dtype=mel.dtype, device=mel.device))
    outputs = []
    with torch.no_grad():
        for step in range(_N_CHUNKS):
            chunk = mel[:, :, step * shift : step * shift + chunk_size]
            assert chunk.shape[-1] == chunk_size, "the toy mel is too short for _N_CHUNKS chunks"
            kwargs = {}
            if targets is not None:
                width = 1 if step == _N_CHUNKS - 1 else frames
                chunk_targets = targets[:, step * frames : step * frames + width].clone()
                if step in sentinel_chunks:
                    for row in sentinel_rows:
                        chunk_targets[row] = _SENTINEL
                kwargs["spk_targets"] = chunk_targets
            out = enc.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=torch.tensor([chunk_size] * batch),
                cache_last_channel=state[0],
                cache_last_time=state[1],
                cache_last_channel_len=state[2],
                keep_all_outputs=False,
                drop_extra_pre_encoded=0 if step == 0 else enc.streaming_cfg.drop_extra_pre_encoded,
                **kwargs,
            )
            outputs.append(out[0])
            state = list(out[2:])
    return outputs


def _sentinel_fixture(batch=3):
    """A seeded toy streaming encoder, mel batch and random oracle targets that differ across speakers.

    Speaker-distinct values matter: ``diar_norm`` is a LayerNorm over the speakers, so a
    speaker-constant row (all zeros, all ones, or the sentinel itself) maps to its bias, and a
    comparison built on such rows would not tell diarizer predictions from silence. Seeded for the
    same reason: some initialisations of the toy diarizer predict every speaker above the threshold.
    """
    torch.manual_seed(0)
    enc = build_toy_streaming_pe_encoder().eval()
    enc.setup_streaming_params()
    generator = torch.Generator().manual_seed(0)
    mel = torch.randn(batch, _MEL_FEATURES, 512, generator=generator)
    targets = torch.rand(batch, 512 // _SUBSAMPLING_FACTOR, _N_SPK, generator=generator)
    return enc, mel, targets


@pytest.mark.unit
def test_streaming_step_substitutes_sentinel_rows():
    """A sentinel row must get the diarizer's predictions, and a real row keep its oracle targets.

    The sentinel means "no RTTM for this row, use your own diarizer". The offline path honours it
    per row; the streaming step used to fuse the sentinel itself, which ``diar_norm`` maps to the
    same output as all-zero targets: the row was decoded as if nobody spoke.
    """
    enc, mel, targets = _sentinel_fixture()
    diarizer = _stream(enc, mel)  # no targets: every row gets the diarizer's predictions
    oracle = _stream(enc, mel, targets)
    silence = _stream(enc, mel, torch.zeros_like(targets))
    mixed = _stream(enc, mel, targets, sentinel_rows=(1,))

    for step in range(_N_CHUNKS):
        # The toy must be able to tell the two apart, or the assertions below prove nothing.
        assert not torch.equal(diarizer[step][1], silence[step][1]), f"degenerate toy at chunk {step}"
        assert not torch.equal(mixed[step][1], silence[step][1]), f"sentinel row fused as silence at chunk {step}"
        assert torch.equal(mixed[step][1], diarizer[step][1]), f"sentinel row is not the diarizer's at chunk {step}"
        for row in (0, 2):
            assert torch.equal(mixed[step][row], oracle[step][row]), f"real row {row} changed at chunk {step}"


@pytest.mark.unit
def test_all_sentinel_stream_equals_no_targets():
    """A batch without any RTTM decodes exactly as if no targets had been passed."""
    enc, mel, targets = _sentinel_fixture()
    diarizer = _stream(enc, mel)
    all_sentinel = _stream(enc, mel, targets, sentinel_rows=(0, 1, 2))
    for step in range(_N_CHUNKS):
        assert torch.equal(all_sentinel[step], diarizer[step]), f"chunk {step} differs"


@pytest.mark.unit
def test_full_coverage_stream_is_unchanged_and_never_steps_the_diarizer():
    """Pin: when every row has real targets, the step is the ASR branch's step plus the fusion of
    those targets, bit for bit, and the diarizer is never stepped (no cost, no state, no RNG draw)."""
    enc, mel, targets = _sentinel_fixture()
    calls = []
    original = enc.diarization_model.forward_streaming_step

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    enc.diarization_model.forward_streaming_step = spy
    rng_before = torch.get_rng_state()
    outputs = _stream(enc, mel, targets)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert calls == [], "the diarizer was stepped for a batch whose rows all have real targets"
    assert enc._diar_total_preds.shape[1] == 0

    # The same stream through the bare ASR branch, fused with the same slices.
    bare = enc.asr_encoder
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    shift = enc.streaming_cfg.shift_size
    shift = shift[1] if isinstance(shift, (list, tuple)) else shift
    frames = chunk_size // _SUBSAMPLING_FACTOR
    batch = mel.shape[0]
    state = list(bare.get_initial_cache_state(batch_size=batch, dtype=mel.dtype, device=mel.device))
    with torch.no_grad():
        for step in range(_N_CHUNKS):
            out = bare.cache_aware_stream_step(
                processed_signal=mel[:, :, step * shift : step * shift + chunk_size],
                processed_signal_length=torch.tensor([chunk_size] * batch),
                cache_last_channel=state[0],
                cache_last_time=state[1],
                cache_last_channel_len=state[2],
                keep_all_outputs=False,
                drop_extra_pre_encoded=0 if step == 0 else enc.streaming_cfg.drop_extra_pre_encoded,
            )
            width = 1 if step == _N_CHUNKS - 1 else frames
            expected = enc._fuse_diar_and_asr(out[0], targets[:, step * frames : step * frames + width])
            assert torch.equal(outputs[step], expected), f"chunk {step} differs"
            state = list(out[2:])


@pytest.mark.unit
def test_a_stream_that_steps_the_diarizer_keeps_stepping_it():
    """The diarizer's state is batched and follows the audio, so once a stream has stepped it, it is
    stepped on every later chunk, also on chunks whose rows all have real targets."""
    enc, mel, targets = _sentinel_fixture()
    calls = []
    original = enc.diarization_model.forward_streaming_step

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    oracle = _stream(enc, mel, targets)
    enc.diarization_model.forward_streaming_step = spy
    # The sentinel on chunk 0 only: not what the dataset produces, but the state must still follow.
    stepped = _stream(enc, mel, targets, sentinel_rows=(1,), sentinel_chunks={0})
    assert len(calls) == _N_CHUNKS, f"the diarizer was stepped on {len(calls)} of {_N_CHUNKS} chunks"
    for step in range(1, _N_CHUNKS):
        assert torch.equal(stepped[step], oracle[step]), f"chunk {step} has no sentinel row but changed"


@pytest.mark.unit
def test_a_sentinel_row_after_skipped_chunks_fails_closed():
    """A diarizer that skipped the start of a stream has no state for it; fusing its predictions
    would be silently wrong, so the step raises. A new stream starts clean."""
    enc, mel, targets = _sentinel_fixture()
    with pytest.raises(RuntimeError, match="has not seen the start"):
        _stream(enc, mel, targets, sentinel_rows=(1,), sentinel_chunks=range(1, _N_CHUNKS))
    # `get_initial_cache_state` resets the decision: a stream with sentinel rows from chunk 0 works.
    diarizer = _stream(enc, mel)
    mixed = _stream(enc, mel, targets, sentinel_rows=(1,))
    assert all(torch.equal(m[1], d[1]) for m, d in zip(mixed, diarizer))


# ==============================================================================================
# Grad mode: a trainable branch follows the caller's, so torch.no_grad() / inference_mode() reach it
# ==============================================================================================
_PATHS = ["offline", "online_inference", "stream_step"]
# The branch a case leaves trainable; the other one is frozen.
_FREEZE_FLAGS = {
    "asr": dict(freeze_asr=False, freeze_diar=True),
    "diar": dict(freeze_asr=True, freeze_diar=False),
}


class _SavesInputForBackward(torch.autograd.Function):
    """An identity that saves its input for backward, as the autograd Function of a fused kernel does.

    PyTorch refuses to save an inference tensor while autograd records, so on CPU this fails the way the
    fused Triton subsampling of a ``dw_striding`` encoder fails on GPU when autograd is turned back on under
    ``torch.inference_mode()``.
    """

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _grad_mode_encoder(path, trainable, **overrides):
    """The plain class for the offline forward and online inference, the streaming class for its step."""
    encoder_cls = StreamingParallelExpertEncoder if path == "stream_step" else ParallelExpertEncoder
    torch.manual_seed(0)
    kwargs = dict(
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type=None,
        **WINDOWED_PE_KWARGS,
        **_FREEZE_FLAGS[trainable],
    )
    kwargs.update(overrides)
    enc = encoder_cls(**kwargs)
    enc._suppress_online_pbar = True
    return enc


def _spy_on_branch_graphs(enc, save_input_for_backward=False) -> dict:
    """Record, per branch and per call, whether autograd recorded the output of the branch's ``pre_encode``.

    Every path (offline forward, online inference, streaming step) runs ``pre_encode`` in both branches. With
    ``save_input_for_backward``, its output also passes through :class:`_SavesInputForBackward`, tied to the
    ``pre_encode`` weights, so it is saved for backward exactly when those weights are recorded.
    """
    seen = {"asr": [], "diar": []}

    def spy(branch, pre_encode):
        weight = next(pre_encode.parameters())

        def hook(module, args, output):
            x, *rest = output
            if save_input_for_backward:
                x = _SavesInputForBackward.apply(x, weight)
            seen[branch].append(x.requires_grad)
            return (x, *rest)

        pre_encode.register_forward_hook(hook)

    spy("asr", enc.asr_encoder.pre_encode)
    spy("diar", enc.diarization_model.encoder.pre_encode)
    return seen


def _run_path(enc, path) -> list:
    """Run ``path`` without ``spk_targets``, so both branches run, in the caller's grad mode. Returns its outputs."""
    generator = torch.Generator().manual_seed(0)
    if path != "stream_step":
        mels = torch.randn(1, _MEL_FEATURES, 160, generator=generator)
        with enc.online_inference(path == "online_inference"):
            return [enc(mels, torch.tensor([160]))[0]]
    enc.setup_streaming_params()
    chunk_size, shift = _chunk_size(enc), _shift_size(enc)
    mels = torch.randn(1, _MEL_FEATURES, 512, generator=generator)
    state = list(enc.get_initial_cache_state(batch_size=1))
    outputs = []
    for step in range(2):
        out = enc.cache_aware_stream_step(
            processed_signal=mels[:, :, step * shift : step * shift + chunk_size],
            processed_signal_length=torch.tensor([chunk_size]),
            cache_last_channel=state[0],
            cache_last_time=state[1],
            cache_last_channel_len=state[2],
            keep_all_outputs=False,
            drop_extra_pre_encoded=0 if step == 0 else enc.streaming_cfg.drop_extra_pre_encoded,
        )
        outputs.append(out[0])
        state = list(out[2:])
    return outputs


@pytest.mark.unit
@pytest.mark.parametrize("trainable", sorted(_FREEZE_FLAGS))
@pytest.mark.parametrize("path", _PATHS)
def test_a_trainable_branch_builds_no_graph_under_no_grad(path, trainable):
    """Under an outer ``torch.no_grad()`` (validation, generation), a trainable branch keeps autograd off.

    Otherwise it builds a graph nobody uses and holds its activations for it. ``freeze_asr`` is false by default,
    so this is the ASR branch of every validation and generation step. The fused output never requires grad either
    way, since the fusion runs outside the branches.
    """
    enc = _grad_mode_encoder(path, trainable).eval()
    seen = _spy_on_branch_graphs(enc)
    with torch.no_grad():
        outputs = _run_path(enc, path)
    assert seen["asr"] and seen["diar"], f"a branch did not run: {seen}"
    assert not any(seen["asr"] + seen["diar"]), f"a branch built a graph under torch.no_grad(): {seen}"
    assert not any(output.requires_grad for output in outputs)


@pytest.mark.unit
@pytest.mark.parametrize("trainable", sorted(_FREEZE_FLAGS))
@pytest.mark.parametrize("path", _PATHS)
def test_a_trainable_branch_runs_under_inference_mode(path, trainable):
    """Under ``torch.inference_mode()``, Lightning's default for ``trainer.validate``, every path runs.

    A trainable branch that turns autograd back on makes each autograd Function in it save inference tensors for
    backward, which raises ("Inference tensors cannot be saved for backward"). On GPU, the fused Triton
    subsampling of a ``dw_striding`` branch is such a Function; :class:`_SavesInputForBackward` stands in for it.
    """
    enc = _grad_mode_encoder(path, trainable).eval()
    seen = _spy_on_branch_graphs(enc, save_input_for_backward=True)
    with torch.inference_mode():
        outputs = _run_path(enc, path)
    assert seen["asr"] and seen["diar"], f"a branch did not run: {seen}"
    assert not any(seen["asr"] + seen["diar"]), f"a branch built a graph under torch.inference_mode(): {seen}"
    assert not any(output.requires_grad for output in outputs)


@pytest.mark.unit
@pytest.mark.parametrize("trainable", sorted(_FREEZE_FLAGS))
@pytest.mark.parametrize("path", _PATHS)
def test_with_grad_enabled_only_the_trainable_branch_builds_a_graph(path, trainable):
    """Pin for training: with grad enabled, the trainable branch builds its graph and the frozen one does not.

    Every parameter is made to require grad first, as an outer blanket unfreeze would, so only the frozen
    branch's grad mode keeps it out of the graph. On the offline (training) forward the gradients are checked
    end to end; continuous speaker features let them reach the diarizer through the fusion.
    """
    enc = _grad_mode_encoder(
        path, trainable, speaker_feature_mode="continuous", speaker_activity_threshold=None
    ).train()
    for parameter in enc.parameters():
        parameter.requires_grad_(True)
    seen = _spy_on_branch_graphs(enc)
    outputs = _run_path(enc, path)
    frozen = "diar" if trainable == "asr" else "asr"
    assert seen[trainable] and all(seen[trainable]), f"the trainable branch built no graph: {seen}"
    assert seen[frozen] and not any(seen[frozen]), f"the frozen branch built a graph: {seen}"
    if path == "offline":
        sum(output.sum() for output in outputs).backward()
        branches = {"asr": enc.asr_encoder, "diar": enc.diarization_model}
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in branches[trainable].parameters())
        assert all(p.grad is None for p in branches[frozen].parameters())


# ==============================================================================================
# The diarizer input's normalization (`diar_normalize_type`): as the base class, per chunk in a stream
# ==============================================================================================
def _chunk_size(enc) -> int:
    chunk_size = enc.streaming_cfg.chunk_size
    return chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size


def _shift_size(enc) -> int:
    shift = enc.streaming_cfg.shift_size
    return shift[1] if isinstance(shift, (list, tuple)) else shift


@pytest.mark.unit
@pytest.mark.parametrize("path", ["offline", "online_inference", "stream_step"])
@pytest.mark.parametrize(
    "diarizer_normalize, passed, expected",
    [
        ("per_feature", _UNSET_SENTINEL, "per_feature"),
        ("all_features", _UNSET_SENTINEL, "all_features"),
        ("NA", _UNSET_SENTINEL, None),
        ("per_feature", "NA", None),
        ("NA", "per_feature", "per_feature"),
        ("NA", "all_features", "all_features"),
    ],
    ids=[
        "unset_follows_per_feature",
        "unset_follows_all_features",
        "unset_follows_NA",
        "NA_wins",
        "per_feature_wins",
        "all_features_wins",
    ],
)
def test_streaming_encoder_normalizes_the_diarizer_input_as_resolved(path, diarizer_normalize, passed, expected):
    """The streaming encoder resolves ``diar_normalize_type`` as the base class does: unset follows the diarizer's
    own normalization, and an explicit value wins either way. Its offline (training) forward and windowed online
    inference normalise the whole utterance, as the base class does; its streaming step normalises each chunk on
    its own, as it does the ASR input. The diarizer's streaming knobs are set explicitly, so their overrides apply."""
    overrides = {} if passed is _UNSET_SENTINEL else {"diar_normalize_type": passed}
    enc = build_toy_streaming_pe_encoder(
        diarization_model_cfg=diarizer_cfg_with_normalize(diarizer_normalize), **WINDOWED_PE_KWARGS, **overrides
    ).eval()
    enc._suppress_online_pbar = True
    enc.setup_streaming_params()
    seen = capture_diarizer_inputs(enc)
    generator = torch.Generator().manual_seed(0)

    if path == "stream_step":
        mels = 4.0 * torch.randn(2, _MEL_FEATURES, 512, generator=generator) + 17.0
        _stream(enc, mels)
        chunk_size, shift = _chunk_size(enc), _shift_size(enc)
        lengths = torch.tensor([chunk_size] * mels.shape[0])
        # Each chunk with its own statistics: a stream has no others.
        given = [(mels[:, :, step * shift : step * shift + chunk_size], lengths) for step in range(_N_CHUNKS)]
    else:
        mels = 4.0 * torch.randn(2, _MEL_FEATURES, 160, generator=generator) + 17.0
        lengths = torch.tensor([160, 113])
        with torch.no_grad(), enc.online_inference(path == "online_inference"):
            enc(mels, lengths)
        # The whole utterance, whose first frames are the diarizer's first input on both paths.
        given = [(mels, lengths)]

    assert len(seen) >= len(given), f"the diarizer was run {len(seen)} times"
    for got, (signal, length) in zip(seen, given):
        want = normalize_batch(signal, length, normalize_type=expected)[0] if expected else signal
        torch.testing.assert_close(got, want[:, :, : got.shape[-1]])
    assert enc.diar_normalize_type == expected


def _streaming_encoder_via(route, tmp_path, diar_normalize_type=_UNSET_SENTINEL, diarizer_normalize="per_feature"):
    """Build a StreamingParallelExpertEncoder on ``route`` over a toy diarizer trained with ``diarizer_normalize``,
    with ``diar_normalize_type`` unless it is unset."""
    stated = {} if diar_normalize_type is _UNSET_SENTINEL else {"diar_normalize_type": diar_normalize_type}
    diar_cfg = diarizer_cfg_with_normalize(diarizer_normalize)
    if route == "constructor":
        return build_toy_streaming_pe_encoder(diarization_model_cfg=diar_cfg, **stated)
    if route == "inline_config":
        config = toy_bundle_config(
            asr_encoder_cfg=streaming_asr_encoder_cfg(),
            asr_normalize_type=None,
            diarization_model_cfg=diar_cfg,
            **stated,
        )
        return StreamingParallelExpertEncoderPT.from_inline_config(config)
    if route == "config_overrides":
        bundle = write_toy_bundle(
            tmp_path / "pe.nemo",
            encoder=build_toy_streaming_pe_encoder(),
            asr_encoder_cfg=streaming_asr_encoder_cfg(),
            asr_normalize_type=None,
            diarization_model_cfg=diar_cfg,
        )
        return StreamingParallelExpertEncoderPT.load_from_nemo(bundle, config_overrides=stated)
    asr, diar = write_toy_branch_checkpoints(tmp_path, diar_normalize=diarizer_normalize)
    return StreamingParallelExpertEncoder.from_checkpoints(asr, diar, asr_normalize_type=None, **stated)


_STREAMING_ROUTES = ["constructor", "inline_config", "config_overrides", "from_checkpoints"]


@pytest.mark.unit
@pytest.mark.parametrize("route", _STREAMING_ROUTES)
@pytest.mark.parametrize(
    "diarizer_normalize, value, expected",
    [
        ("per_feature", _UNSET_SENTINEL, "per_feature"),
        ("NA", _UNSET_SENTINEL, None),
        ("per_feature", None, None),
        ("per_feature", "NA", None),
        ("NA", "per_feature", "per_feature"),
        ("NA", "all_features", "all_features"),
    ],
    ids=[
        "unset_follows_per_feature",
        "unset_follows_NA",
        "null_wins",
        "NA_wins",
        "per_feature_wins",
        "all_features_wins",
    ],
)
def test_streaming_routes_resolve_the_diarizer_normalization(tmp_path, route, diarizer_normalize, value, expected):
    """Every route that builds the streaming class resolves ``diar_normalize_type`` as the base class does: unset
    follows the diarizer's own normalization, and an explicit value overrides it either way, without an error. A
    bundle shell records the result in the bundle config it keeps, ``'NA'`` for none."""
    enc = _streaming_encoder_via(route, tmp_path, value, diarizer_normalize)
    assert isinstance(enc, StreamingParallelExpertEncoder)
    assert enc.diar_normalize_type == expected
    if route in ("inline_config", "config_overrides"):
        assert enc._bundle_config.diar_normalize_type == (expected or "NA")


def _per_chunk_normalization_warnings(warnings) -> list:
    return [line for line in warnings if "normalizes each chunk on its own" in line]


@pytest.mark.unit
@pytest.mark.parametrize("first_path", ["offline", "online_inference", "stream_step"])
@pytest.mark.parametrize(
    "diarizer_normalize, passed, expected",
    [
        ("per_feature", _UNSET_SENTINEL, "per_feature"),
        ("all_features", _UNSET_SENTINEL, "all_features"),
        ("NA", _UNSET_SENTINEL, None),
        ("per_feature", "NA", None),
        ("NA", "per_feature", "per_feature"),
    ],
    ids=["unset_per_feature", "unset_all_features", "unset_NA", "explicit_NA", "explicit_per_feature"],
)
def test_streaming_encoder_warns_once_when_it_normalizes_the_diarizer_input(
    monkeypatch, first_path, diarizer_normalize, passed, expected
):
    """The first time the streaming encoder normalises the diarizer input, on whichever path, it warns that its
    streaming step normalises each chunk on its own while the offline (training) forward uses the whole utterance.
    It warns once: later forwards, chunks and online windows add nothing. Without a normalization there is
    nothing to warn about, and the plain encoder, which has no streaming step, never warns."""
    warnings = _capture_pe_warnings(monkeypatch)
    overrides = {} if passed is _UNSET_SENTINEL else {"diar_normalize_type": passed}
    diar_cfg = diarizer_cfg_with_normalize(diarizer_normalize)
    enc = build_toy_streaming_pe_encoder(diarization_model_cfg=diar_cfg, **WINDOWED_PE_KWARGS, **overrides).eval()
    enc._suppress_online_pbar = True
    enc.setup_streaming_params()
    assert _per_chunk_normalization_warnings(warnings) == [], "warned before normalising anything"
    mels = torch.randn(1, _MEL_FEATURES, 512, generator=torch.Generator().manual_seed(0))
    lengths = torch.tensor([mels.shape[-1]])

    with torch.no_grad():
        if first_path == "stream_step":
            _stream(enc, mels)
        else:
            with enc.online_inference(first_path == "online_inference"):
                enc(mels, lengths)
        lines = _per_chunk_normalization_warnings(warnings)
        assert len(lines) == (1 if expected else 0), lines
        assert all(repr(expected) in line for line in lines), lines

        with enc.online_inference(False):
            enc(mels, lengths)
        _stream(enc, mels)
        with enc.online_inference():
            enc(mels, lengths)
        assert _per_chunk_normalization_warnings(warnings) == lines, "warned again"

        plain = build_toy_pe_encoder(diarization_model_cfg=diar_cfg, **WINDOWED_PE_KWARGS, **overrides).eval()
        plain._suppress_online_pbar = True
        with plain.online_inference(False):
            plain(mels, lengths)
        with plain.online_inference():
            plain(mels, lengths)
    assert _per_chunk_normalization_warnings(warnings) == lines, "the plain encoder warned"


@pytest.mark.unit
@pytest.mark.parametrize(
    "source_states, template_states, expected",
    [(_UNSET_SENTINEL, "NA", "per_feature"), ("NA", "per_feature", None)],
    ids=["unset_follows_per_feature", "NA_over_per_feature"],
)
def test_streaming_export_and_save_keep_the_diar_normalization(tmp_path, source_states, template_states, expected):
    """The streaming encoder's effective ``diar_normalize_type`` is what its HF export and a bundle it saves state,
    so a reload into either class (StreamingSTT, or vLLM and SALM, which mount the plain class) normalises the
    diarizer input as the encoder did. ``save_to_nemo`` writes it over what the template states."""
    stated = {} if source_states is _UNSET_SENTINEL else {"diar_normalize_type": source_states}
    streaming = StreamingParallelExpertEncoderPT.from_inline_config(
        toy_bundle_config(asr_encoder_cfg=streaming_asr_encoder_cfg(), asr_normalize_type=None, **stated)
    )
    assert streaming.diar_normalize_type == expected
    exported = hf_exported_pe_config(streaming)
    assert exported["diar_normalize_type"] == expected

    template = write_toy_bundle(
        tmp_path / "template.nemo",
        encoder=build_toy_streaming_pe_encoder(),
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        asr_normalize_type=None,
        diar_normalize_type=template_states,
    )
    saved = str(tmp_path / "saved.nemo")
    StreamingParallelExpertEncoderPT.save_to_nemo(streaming, saved, template_bundle_path=template)
    for loader in (ParallelExpertEncoderPT, StreamingParallelExpertEncoderPT):
        assert loader.from_inline_config(exported).diar_normalize_type == expected
        assert loader.load_from_nemo(saved).diar_normalize_type == expected


@pytest.mark.unit
@pytest.mark.parametrize("normalize", ["per_feature", "all_features"])
def test_streaming_step_normalizes_the_diarizer_chunk_as_it_does_the_asr_chunk(normalize):
    """The streaming step hands the diarizer the chunk normalised exactly as the ASR branch's: the same tensor,
    with the same per-row valid lengths, so a padded row is normalised over its valid frames only."""
    enc = build_toy_streaming_pe_encoder(asr_normalize_type=normalize, diar_normalize_type=normalize).eval()
    enc.setup_streaming_params()
    asr_inputs, diar_inputs = [], []
    asr_step = enc.asr_encoder.cache_aware_stream_step
    diar_step = enc.diarization_model.forward_streaming_step

    def spy_asr_step(**kwargs):
        asr_inputs.append(kwargs["processed_signal"].detach().clone())
        return asr_step(**kwargs)

    def spy_diar_step(**kwargs):
        diar_inputs.append(kwargs["processed_signal"].detach().transpose(1, 2).clone())
        return diar_step(**kwargs)

    enc.asr_encoder.cache_aware_stream_step = spy_asr_step
    enc.diarization_model.forward_streaming_step = spy_diar_step
    chunk_size, shift = _chunk_size(enc), _shift_size(enc)
    mels = 3.0 * torch.randn(2, _MEL_FEATURES, 512, generator=torch.Generator().manual_seed(0)) + 11.0
    state = list(enc.get_initial_cache_state(batch_size=2))
    with torch.no_grad():
        for step in range(_N_CHUNKS):
            chunk = mels[:, :, step * shift : step * shift + chunk_size]
            # The second row's last chunk is padded, as at the end of a shorter stream.
            lengths = torch.tensor([chunk_size, chunk_size - 5 if step == _N_CHUNKS - 1 else chunk_size])
            out = enc.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=lengths,
                cache_last_channel=state[0],
                cache_last_time=state[1],
                cache_last_channel_len=state[2],
                keep_all_outputs=False,
                drop_extra_pre_encoded=0 if step == 0 else enc.streaming_cfg.drop_extra_pre_encoded,
            )
            state = list(out[2:])
            torch.testing.assert_close(diar_inputs[-1], asr_inputs[-1], rtol=0, atol=0)
            torch.testing.assert_close(diar_inputs[-1], normalize_batch(chunk, lengths, normalize_type=normalize)[0])
