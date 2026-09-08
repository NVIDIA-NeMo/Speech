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

import io
import tarfile

import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.transformer_encoder import StreamingTransformerEncoder
from nemo.collections.asr.modules.parallel_expert_encoder import (
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
)

# ``@experimental`` wraps the class in a wrapt proxy, so ``__new__`` (used to build
# bare instances that skip the heavy real ``__init__``) must target the underlying
# class. Attribute access / isinstance still go through the proxy name.
_PEE = getattr(ParallelExpertEncoder, "__wrapped__", ParallelExpertEncoder)


# ----------------------------------------------------------------------------- #
# Module-level context managers / helpers
# ----------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Pin this module's default device to CPU, and restore it afterwards.

    Several ``tests/collections/speechlm2`` modules call ``torch.set_default_device('cuda')`` at
    import time and never restore it, so in a full-suite run this file would otherwise inherit a
    CUDA default and mix devices (freshly built modules on CUDA, some framework-internal tensors on
    CPU). Tests that actually want GPU opt in explicitly via ``.cuda()`` / ``device="cuda"``.
    """
    previous = torch.get_default_device()
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(previous)


@pytest.mark.unit
def test_clone_config_is_deep_and_handles_none():
    cfg = OmegaConf.create({"a": {"b": 1}})
    clone = _clone_config(cfg)
    assert clone == cfg
    clone.a.b = 2
    assert cfg.a.b == 1  # original untouched
    assert _clone_config(None) is None


@pytest.mark.unit
@pytest.mark.parametrize("target_dtype", [torch.float64, torch.float16])
def test_default_dtype_sets_and_restores(target_dtype):
    prev = torch.get_default_dtype()
    with _default_dtype(target_dtype):
        assert torch.get_default_dtype() == target_dtype
    assert torch.get_default_dtype() == prev


@pytest.mark.unit
@pytest.mark.parametrize("noop_dtype", [torch.get_default_dtype(), torch.int32])
def test_default_dtype_noop_paths(noop_dtype):
    # Same-dtype and non-floating dtype are both no-ops.
    prev = torch.get_default_dtype()
    with _default_dtype(noop_dtype):
        assert torch.get_default_dtype() == prev
    assert torch.get_default_dtype() == prev


@pytest.mark.unit
def test_disable_dist_feature_sync_noop_when_uninitialized():
    assert not dist.is_initialized()
    orig = dist.is_initialized
    with _disable_dist_feature_sync():
        pass
    assert dist.is_initialized is orig  # nothing patched when dist is down


# ----------------------------------------------------------------------------- #
# Static pure helpers on ParallelExpertEncoder
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("max_pos, dim", [(4, 8), (1, 16), (10, 4)])
def test_build_sinusoid_position_encoding(max_pos, dim):
    pe = ParallelExpertEncoder._build_sinusoid_position_encoding(max_pos, dim)
    assert pe.shape == (max_pos, dim)
    # row 0: sin(0)=0 on even indices, cos(0)=1 on odd indices
    assert torch.allclose(pe[0, 0::2], torch.zeros(dim // 2))
    assert torch.allclose(pe[0, 1::2], torch.ones(dim // 2))


@pytest.mark.unit
@pytest.mark.parametrize(
    "cur_len, target_len",
    [(3, 6), (6, 3), (5, 5), (1, 4)],
)
def test_align_diar_frames_length_and_padding(cur_len, target_len):
    n_spk = 3
    diar = torch.arange(cur_len * n_spk, dtype=torch.float32).reshape(1, cur_len, n_spk)
    out = ParallelExpertEncoder._align_diar_frames(diar, target_len)
    assert out.shape == (1, target_len, n_spk)
    if target_len <= cur_len:
        # truncation keeps the leading frames unchanged
        assert torch.equal(out, diar[:, :target_len, :])
    else:
        # padding repeats the last frame
        assert torch.equal(out[:, :cur_len, :], diar)
        for t in range(cur_len, target_len):
            assert torch.equal(out[:, t, :], diar[:, -1, :])


@pytest.mark.unit
@pytest.mark.parametrize("param_dtype", [torch.float64, torch.float16])
def test_match_module_io_casts_to_param_dtype(param_dtype):
    module = nn.Linear(4, 4).to(param_dtype)
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    out = ParallelExpertEncoder._match_module_io(tensor, module)
    assert out.dtype == param_dtype


@pytest.mark.unit
def test_match_module_io_paramless_module_unchanged():
    module = nn.Identity()  # no parameters
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    out = ParallelExpertEncoder._match_module_io(tensor, module)
    assert out.dtype == torch.float32
    assert out is tensor


# ----------------------------------------------------------------------------- #
# forward() offline/online dispatch
# ----------------------------------------------------------------------------- #
def dispatch_stub(online_inference_length, chunk_feat_len, training):
    """Build a bare ParallelExpertEncoder with stubbed branch methods."""
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.online_inference_length = online_inference_length
    enc.chunk_feat_len = chunk_feat_len
    enc.training = training
    enc._forward = lambda **kw: "offline"
    enc._forward_online = lambda **kw: "online"
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "online_len, chunk_feat_len, training, n_frames, expected",
    [
        (500, 100, False, 200, "online"),  # eval + long enough -> online
        (500, 100, False, 50, "offline"),  # eval but shorter than one window
        (500, 100, True, 200, "offline"),  # training always offline
        (0, 100, False, 200, "offline"),  # online disabled
        (500, 100, False, 100, "offline"),  # exactly one window (not strictly greater)
    ],
)
def test_forward_dispatch(online_len, chunk_feat_len, training, n_frames, expected):
    enc = dispatch_stub(online_len, chunk_feat_len, training)
    audio = torch.zeros(1, 8, n_frames)
    length = torch.tensor([n_frames])
    assert enc.forward(audio, length) == expected


# ----------------------------------------------------------------------------- #
# _forward_online orchestration (stubbed ASR encoder, provided spk_targets)
# ----------------------------------------------------------------------------- #
class _FakeASR(nn.Module):
    """Minimal stand-in for the wrapped ConformerEncoder."""

    def __init__(self, d_model: int, sf: int):
        super().__init__()
        self.subsampling_factor = sf
        self.d_model = d_model
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, audio_signal, length):
        b, _, t = audio_signal.shape
        # generous frame count so the trim logic never clamps
        t_out = (t + self.subsampling_factor - 1) // self.subsampling_factor + 8
        out = torch.randn(b, self.d_model, t_out)
        return out, length // self.subsampling_factor


def online_stub(d_model, n_spk, sf, win, lc, rc):
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.asr_encoder = _FakeASR(d_model, sf)
    enc.asr_normalize_type = None
    enc.online_inference_length = win
    enc.chunk_left_context = lc
    enc.chunk_right_context = rc
    enc.chunk_feat_len = win * sf
    enc.left_ctx_feat_len = lc * sf
    enc.right_ctx_feat_len = rc * sf
    enc.freeze_asr = True
    enc.freeze_diar = False  # The stub has no `diarization_model`, so `freeze_diar` must be False to keep
    enc.asr_norm = nn.LayerNorm(d_model)
    enc.diar_norm = nn.LayerNorm(n_spk)
    enc.register_buffer("diar_kernel", torch.randn(n_spk, d_model))
    # Fusion knobs the real ctor sets; `__new__` bypasses it, so mirror the ctor defaults here.
    enc.speaker_activity_threshold = 0.5
    enc.spk_kernel_scale = 1.0
    enc._suppress_online_pbar = True
    enc.eval()
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "sf, win, lc, rc, n_frames",
    [
        (8, 10, 2, 2, 240),  # 3 full chunks
        (8, 10, 0, 0, 200),  # partial last chunk, no context
        (4, 5, 1, 1, 64),  # 4 chunks, small subsampling
        (8, 50, 5, 5, 160),  # single chunk (n_frames < window)
    ],
)
def test_forward_online_output_length_telescopes(sf, win, lc, rc, n_frames):
    d_model, n_spk, b = 16, 4, 2
    enc = online_stub(d_model, n_spk, sf, win, lc, rc)

    mels = torch.randn(b, 80, n_frames)
    length = torch.tensor([n_frames] * b)
    spk_targets = torch.rand(b, 5, n_spk)  # arbitrary; aligned internally

    outputs, encoded_len = enc._forward_online(audio_signal=mels, length=length, spk_targets=spk_targets)

    expected_t = round(n_frames / sf)
    assert outputs.shape == (b, d_model, expected_t)
    assert encoded_len.tolist() == [expected_t] * b


# ----------------------------------------------------------------------------- #
# ParallelExpertEncoderPT.is_pe_nemo
# ----------------------------------------------------------------------------- #
def write_nemo(path, *, target=None, include_cfg=True):
    with tarfile.open(path, "w") as tf:
        if include_cfg:
            data = (f"target: {target}\n" if target is not None else "foo: bar\n").encode()
            info = tarfile.TarInfo(name="model_config.yaml")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        else:
            data = b"not a config"
            info = tarfile.TarInfo(name="weights.ckpt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.mark.unit
@pytest.mark.parametrize(
    "target, expected",
    [
        ("nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT", True),
        ("ParallelExpertEncoderPT", True),
        ("nemo.collections.asr.models.SomethingElse", False),
        (None, False),  # model_config.yaml present but no `target`
    ],
)
def test_is_pe_nemo_by_target(tmp_path, target, expected):
    nemo_path = str(tmp_path / "bundle.nemo")
    write_nemo(nemo_path, target=target)
    assert ParallelExpertEncoderPT.is_pe_nemo(nemo_path) is expected


@pytest.mark.unit
def test_is_pe_nemo_without_model_config(tmp_path):
    nemo_path = str(tmp_path / "no_cfg.nemo")
    write_nemo(nemo_path, include_cfg=False)
    assert ParallelExpertEncoderPT.is_pe_nemo(nemo_path) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_path",
    [None, 123, "missing.nemo", "not_a_nemo.txt"],
)
def test_is_pe_nemo_rejects_bad_paths(tmp_path, bad_path):
    # a real-but-non-.nemo file to exercise the suffix check
    if bad_path == "not_a_nemo.txt":
        p = tmp_path / "not_a_nemo.txt"
        p.write_text("hello")
        bad_path = str(p)
    assert ParallelExpertEncoderPT.is_pe_nemo(bad_path) is False


# ----------------------------------------------------------------------------- #
# ParallelExpertEncoderPT.save_to_nemo guard rails
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_save_to_nemo_rejects_non_encoder(tmp_path):
    with pytest.raises(TypeError):
        ParallelExpertEncoderPT.save_to_nemo(
            nn.Linear(2, 2), str(tmp_path / "out.nemo"), template_bundle_path=str(tmp_path / "tpl.nemo")
        )


@pytest.mark.unit
def test_save_to_nemo_missing_template(tmp_path):
    # __new__ produces a real ParallelExpertEncoder instance (passes isinstance)
    # without running the heavy __init__, so we reach the template existence check.
    fake_encoder = _PEE.__new__(_PEE)
    with pytest.raises(FileNotFoundError):
        ParallelExpertEncoderPT.save_to_nemo(
            fake_encoder,
            str(tmp_path / "out.nemo"),
            template_bundle_path=str(tmp_path / "does_not_exist.nemo"),
        )


# ----------------------------------------------------------------------------- #
# End-to-end fusion with real toy encoders
#
# ParallelExpertEncoder loads two real sub-encoders and fuses them:
#   * an ASR ConformerEncoder (cf. tests/collections/asr/test_conformer_encoder.py)
#   * a Sortformer diarizer    (cf. tests/collections/speaker_tasks/test_diar_sortformer_models.py)
# These tests build tiny-but-real instances of both and run the wrapper end to end.
# ----------------------------------------------------------------------------- #
_MEL_FEATURES = 128
_UNSET_SENTINEL = object()
_ASR_D_MODEL = 32
_DIAR_FC_D_MODEL = 32
_DIAR_TF_D_MODEL = 16
_N_SPK = 4
_SUBSAMPLING_FACTOR = 8


def toy_asr_encoder_cfg() -> DictConfig:
    """Tiny ConformerEncoder config the PE encoder mounts as its ASR branch."""
    return DictConfig(
        {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': _MEL_FEATURES,
            'feat_out': -1,
            'n_layers': 1,
            'd_model': _ASR_D_MODEL,
            'subsampling': 'dw_striding',
            'subsampling_factor': _SUBSAMPLING_FACTOR,
            'subsampling_conv_channels': 16,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 4,
            'att_context_size': [-1, -1],
            'conv_kernel_size': 9,
            'dropout': 0.0,
            'dropout_pre_encoder': 0.0,
            'dropout_emb': 0.0,
            'dropout_att': 0.0,
        }
    )


def toy_diarization_model_cfg() -> DictConfig:
    """Tiny SortformerEncLabelModel config the PE encoder mounts as its diar branch."""
    model_defaults = {'fc_d_model': _DIAR_FC_D_MODEL, 'tf_d_model': _DIAR_TF_D_MODEL}
    return DictConfig(
        {
            'target': 'nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel',
            'sample_rate': 16000,
            'pil_weight': 0.5,
            'ats_weight': 0.5,
            'max_num_of_spks': _N_SPK,
            'streaming_mode': False,
            'async_streaming': False,
            'model_defaults': DictConfig(model_defaults),
            'preprocessor': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                    'normalize': 'per_feature',
                    'window_size': 0.025,
                    'sample_rate': 16000,
                    'window_stride': 0.01,
                    'window': 'hann',
                    'features': _MEL_FEATURES,
                    'n_fft': 512,
                    'frame_splicing': 1,
                    'dither': 0.00001,
                }
            ),
            'encoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                    'feat_in': _MEL_FEATURES,
                    'feat_out': -1,
                    'n_layers': 1,
                    'd_model': _DIAR_FC_D_MODEL,
                    'subsampling': 'dw_striding',
                    'subsampling_factor': _SUBSAMPLING_FACTOR,
                    'subsampling_conv_channels': 16,
                    'causal_downsampling': False,
                    'ff_expansion_factor': 4,
                    'self_attention_model': 'rel_pos',
                    'n_heads': 4,
                    'att_context_size': [-1, -1],
                    'conv_kernel_size': 9,
                    'conv_norm_type': 'batch_norm',
                    'dropout': 0.0,
                    'dropout_pre_encoder': 0.0,
                    'dropout_emb': 0.0,
                    'dropout_att': 0.0,
                }
            ),
            'transformer_encoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder',
                    'num_layers': 1,
                    'hidden_size': _DIAR_TF_D_MODEL,
                    'inner_size': 32,
                    'num_attention_heads': 4,
                    'attn_score_dropout': 0.0,
                    'attn_layer_dropout': 0.0,
                    'ffn_dropout': 0.0,
                    'hidden_act': 'relu',
                    'pre_ln': False,
                    'pre_ln_final_layer_norm': True,
                }
            ),
            'sortformer_modules': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.sortformer_modules.SortformerModules',
                    'num_spks': _N_SPK,
                    'dropout_rate': 0.0,
                    'fc_d_model': _DIAR_FC_D_MODEL,
                    'tf_d_model': _DIAR_TF_D_MODEL,
                }
            ),
            'loss': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.losses.bce_loss.BCELoss',
                    'weight': None,
                    'reduction': 'mean',
                }
            ),
        }
    )


def build_toy_pe_encoder(**overrides) -> ParallelExpertEncoder:
    """Construct a real ParallelExpertEncoder from the tiny ASR + diar configs."""
    kwargs = dict(
        asr_encoder_cfg=toy_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type='per_feature',
        # Keep the input far below one window so forward() stays on the offline path.
        online_inference_length=500,
    )
    kwargs.update(overrides)
    return ParallelExpertEncoder(**kwargs)


@pytest.mark.unit
def test_pe_encoder_builds_and_wires_both_real_encoders():
    enc = build_toy_pe_encoder()
    # The two fused sub-encoders are the real classes, not stubs.
    assert isinstance(enc.asr_encoder, ConformerEncoder)
    assert isinstance(enc.diarization_model, SortformerEncLabelModel)
    # ConformerEncoder-compatible drop-in properties come from the ASR branch.
    assert enc.d_model == _ASR_D_MODEL
    assert enc.subsampling_factor == _SUBSAMPLING_FACTOR
    # Speaker count + fusion kernel come from the diar branch.
    assert enc.n_spk == _N_SPK
    assert enc.diar_kernel.shape == (_N_SPK, _ASR_D_MODEL)
    # freeze_diar defaults to True -> diar params are frozen, ASR params remain trainable.
    assert all(not p.requires_grad for p in enc.diarization_model.parameters())
    assert any(p.requires_grad for p in enc.asr_encoder.parameters())


@pytest.mark.unit
def test_pe_encoder_accepts_a_non_conformer_cache_aware_asr_branch():
    """The ASR branch is duck-typed, so a StreamingTransformerEncoder mounts just like a Conformer.

    It differs from the Conformer in ways the PE never touches (`feature_stacking` instead of
    convolutional subsampling, `rope` instead of `rel_pos`), and having no convolutions is exactly
    why it is worth trying: nothing has to be reconstructed from `cache_last_time` across chunks.
    """
    asr_cfg = DictConfig(
        {
            '_target_': 'nemo.collections.asr.modules.transformer_encoder.StreamingTransformerEncoder',
            'feat_in': _MEL_FEATURES,
            'n_layers': 1,
            'd_model': _ASR_D_MODEL,
            # head_dim = d_model / n_heads must be >= 16 for the flex-attention CUDA backend.
            'n_heads': 2,
            'subsampling': 'feature_stacking',
            'subsampling_factor': _SUBSAMPLING_FACTOR,
            'self_attention_model': 'rope',
            'att_context_style': 'chunked_limited',
            'att_context_size': [70, 1],
            'drop_rate': 0.0,
        }
    )
    enc = build_toy_pe_encoder(asr_encoder_cfg=asr_cfg)

    assert isinstance(enc.asr_encoder, StreamingTransformerEncoder)
    # The drop-in properties the PE re-exports still resolve off the branch.
    assert enc.d_model == _ASR_D_MODEL
    assert enc.subsampling_factor == _SUBSAMPLING_FACTOR
    assert enc.diar_kernel.shape == (_N_SPK, _ASR_D_MODEL)


@pytest.mark.unit
def test_pe_encoder_rejects_an_asr_branch_that_cannot_stream():
    """A branch missing the cache-aware API must fail at construction, naming what is missing."""
    asr_cfg = DictConfig({'_target_': 'torch.nn.Linear', 'in_features': 4, 'out_features': 4})

    with pytest.raises(TypeError, match="cannot serve as a ParallelExpertEncoder ASR branch"):
        build_toy_pe_encoder(asr_encoder_cfg=asr_cfg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "high_resolution, requested_diar_subsampling_factor, expected_asr_aligned_factor",
    [(True, 1, _SUBSAMPLING_FACTOR)],
)
def test_pe_encoder_matches_diar_output_resolution_to_asr_encoder(
    high_resolution, requested_diar_subsampling_factor, expected_asr_aligned_factor
):
    diar_cfg = toy_diarization_model_cfg()
    diar_cfg.high_resolution = high_resolution
    diar_cfg.output_subsampling_factor = requested_diar_subsampling_factor

    enc = build_toy_pe_encoder(diarization_model_cfg=diar_cfg)

    assert (
        enc.diarization_model.output_subsampling_factor
        == enc.asr_encoder.subsampling_factor
        == expected_asr_aligned_factor
    )
    assert (
        enc.diarization_model._cfg.output_subsampling_factor
        == enc.asr_encoder.subsampling_factor
        == expected_asr_aligned_factor
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "asr_subsampling_factor, error_match",
    [(4, "requires the diarization output subsampling factor")],
)
def test_pe_encoder_rejects_incompatible_diar_output_resolution(asr_subsampling_factor, error_match):
    asr_cfg = toy_asr_encoder_cfg()
    asr_cfg.subsampling_factor = asr_subsampling_factor

    with pytest.raises(ValueError, match=error_match):
        build_toy_pe_encoder(asr_encoder_cfg=asr_cfg)


@pytest.mark.unit
@pytest.mark.parametrize("batch_size, n_frames", [(1, 160), (2, 200)])
def test_pe_encoder_offline_forward_runs_internal_diarizer(batch_size, n_frames):
    enc = build_toy_pe_encoder().eval()
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)  # spk_targets=None -> Sortformer runs internally

    expected_t = int(encoded_len[0].item())
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
    assert encoded_len.tolist() == [expected_t] * batch_size


@pytest.mark.unit
def test_pe_encoder_offline_forward_accepts_diar_override_and_fuses_it():
    enc = build_toy_pe_encoder().eval()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    # Arbitrary diar frame count: PE aligns it to the ASR frame count internally.
    dp1 = torch.rand(batch_size, 7, _N_SPK)
    dp2 = torch.rand(batch_size, 7, _N_SPK)

    with torch.no_grad():
        out1, len1 = enc(mels, length, spk_targets=dp1)
        out2, len2 = enc(mels, length, spk_targets=dp2)

    expected_t = int(len1[0].item())
    assert out1.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.equal(len1, len2)
    assert torch.isfinite(out1).all()
    # Same audio + same (dropout-free, eval) ASR branch, but different speaker
    # predictions must change the fused output -> proves the diar branch is fused in.
    assert not torch.allclose(out1, out2)


@pytest.mark.unit
def test_pe_encoder_online_forward_matches_conformer_io_with_real_encoders():
    # Small window so a modest input crosses onto the long-form online path.
    enc = build_toy_pe_encoder(
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    enc._suppress_online_pbar = True

    batch_size, n_frames = 1, 320  # > online_inference_length * subsampling_factor (=80)
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()


# ----------------------------------------------------------------------------- #
# GPU end-to-end fusion with real toy encoders
#
# These mirror the CPU end-to-end tests but run on CUDA. They additionally
# exercise the device/dtype-bridging machinery the wrapper exists for: fp32 mels
# fed into (optionally) bf16 experts on the GPU, handled by `_match_module_io`
# (offline) and `_default_dtype` / `_disable_dist_feature_sync` (online).
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
@pytest.mark.parametrize("batch_size, n_frames", [(1, 160), (2, 200)])
def test_pe_encoder_offline_forward_on_gpu(batch_size, n_frames):
    enc = build_toy_pe_encoder().eval().cuda()
    # Mels arrive un-normalised in fp32 (the SALM perception contract).
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)  # spk_targets=None -> Sortformer runs internally

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
    assert encoded_len.tolist() == [expected_t] * batch_size


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="PEE bf16 GPU test requires CUDA with bf16 support",
)
def test_pe_encoder_offline_forward_bf16_experts_on_gpu():
    # Experts run in bf16 while mels stay fp32 -> exercises `_match_module_io`
    # device/dtype bridging on both branches before their conv subsampling.
    enc = build_toy_pe_encoder().eval().cuda().to(torch.bfloat16)
    batch_size, n_frames = 2, 200
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.dtype == torch.bfloat16
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.isfinite(outputs).all()


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
def test_pe_encoder_offline_forward_accepts_diar_override_on_gpu():
    enc = build_toy_pe_encoder().eval().cuda()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    dp1 = torch.rand(batch_size, 7, _N_SPK, device="cuda")
    dp2 = torch.rand(batch_size, 7, _N_SPK, device="cuda")

    with torch.no_grad():
        out1, len1 = enc(mels, length, spk_targets=dp1)
        out2, len2 = enc(mels, length, spk_targets=dp2)

    expected_t = int(len1[0].item())
    assert out1.is_cuda
    assert out1.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.equal(len1, len2)
    assert torch.isfinite(out1).all()
    # Different speaker predictions must change the fused output.
    assert not torch.allclose(out1, out2)


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
def test_pe_encoder_online_forward_on_gpu():
    enc = (
        build_toy_pe_encoder(
            online_inference_length=10,
            chunk_left_context=2,
            chunk_right_context=2,
            diar_fifo_len=10,
            diar_spkcache_update_period=20,
            diar_spkcache_len=20,
        )
        .eval()
        .cuda()
    )
    enc._suppress_online_pbar = True

    batch_size, n_frames = 1, 320  # > online_inference_length * subsampling_factor (=80)
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "passed, expected",
    [
        (_UNSET_SENTINEL, 'per_feature'),  # key absent -> reference default
        (None, None),  # explicitly disabled
        ('NA', None),  # checkpoint-contract spelling of "no normalization"
        ('per_feature', 'per_feature'),
        ('all_features', 'all_features'),
    ],
)
def test_asr_normalize_type_resolution(passed, expected):
    """`None`/`'NA'` must genuinely disable normalization; only an *absent* key defaults to
    `per_feature`. A plain `or 'per_feature'` fallback silently re-normalized mels coming from a
    preprocessor that is already `normalize: NA`."""
    overrides = {} if passed is _UNSET_SENTINEL else {"asr_normalize_type": passed}
    enc = build_toy_pe_encoder(**overrides)
    assert enc.asr_normalize_type == expected


@pytest.mark.unit
@pytest.mark.parametrize("normalize_type, should_match", [(None, True), ('NA', True), ('per_feature', False)])
def test_disabled_normalization_feeds_asr_branch_untouched(normalize_type, should_match):
    """With normalization off, the ASR branch inside PE sees exactly the mels it would see as a
    standalone encoder (PLAN 1.7.1)."""
    enc = build_toy_pe_encoder(asr_normalize_type=normalize_type).eval()
    enc._fuse_diar_and_asr = lambda asr_encoded, spk_targets: asr_encoded  # isolate the ASR path

    torch.manual_seed(0)
    mel = torch.randn(1, _MEL_FEATURES, 128)
    length = torch.tensor([mel.shape[-1]])
    spk_targets = torch.zeros(1, mel.shape[-1] // _SUBSAMPLING_FACTOR, _N_SPK)

    with torch.no_grad():
        fused, _ = enc(audio_signal=mel, length=length, spk_targets=spk_targets)
        standalone, _ = enc.asr_encoder(audio_signal=mel, length=length)

    assert torch.equal(fused, standalone) is should_match


@pytest.mark.unit
def test_apply_internal_freeze_reasserts_the_branch_split():
    """A blanket outer unfreeze must not put the frozen diarizer back in the optimizer.

    speechlm2's `freeze_speech_encoder: false` calls `unfreeze_module(perception.encoder)`, which
    walks the whole tree and cannot know a branch is meant to stay frozen -- so the encoder has to
    be able to re-assert its own policy afterwards.
    """
    enc = build_toy_pe_encoder(freeze_diar=True, freeze_asr=False)
    diar_params = list(enc.diarization_model.parameters())
    assert not any(p.requires_grad for p in diar_params)

    for param in enc.parameters():  # simulate the outer blanket unfreeze
        param.requires_grad = True
    assert all(p.requires_grad for p in diar_params)

    enc.apply_internal_freeze()
    assert not any(p.requires_grad for p in diar_params), "freeze_diar was not re-asserted"
    assert all(p.requires_grad for p in enc.asr_encoder.parameters()), "freeze_asr=False must stay trainable"
    assert not enc.diarization_model.training, "frozen branch must also be back in eval mode"


@pytest.mark.unit
def test_att_context_size_forwards_to_the_asr_branch():
    """`att_context_size` must reach the branch that actually implements attention.

    speechlm2's `_set_encoder_att_context` assigns `encoder.att_context_size = [left, right]` per
    batch (streaming_stt_model.py:957, :1205, :3198). On a plain nn.Module that silently creates a
    dead attribute, leaving the look-ahead at whatever the bundle was built with -- which corrupts
    training look-ahead for fixed chunking, not just streaming decode.
    """
    enc = build_toy_pe_encoder()
    assert enc.att_context_size == enc.asr_encoder.att_context_size

    enc.att_context_size = [70, 5]
    assert enc.asr_encoder.att_context_size == [70, 5], "assignment did not reach the ASR branch"
    assert enc.att_context_size == [70, 5]
    # and it must not have shadowed the property with a plain instance attribute
    assert "att_context_size" not in enc.__dict__


@pytest.mark.unit
def test_setting_att_context_size_collapses_the_multi_context_choice_set():
    """Pinning the look-ahead must survive `ConformerEncoder.forward`'s training-time sampling.

    A multi-context checkpoint (the streaming Conformer ships
    `[[70, 13], [70, 6], [70, 1], [70, 0]]`) leaves `att_context_size_all` longer than one, and
    the encoder then samples a context at random on every *training* forward
    (conformer_encoder.py:663) -- silently overriding whatever `att_context_size` was set to.
    """
    cfg = toy_asr_encoder_cfg()
    cfg.att_context_size = [[8, 7], [8, 3], [8, 1], [8, 0]]  # chunked_limited needs left % (right+1) == 0
    cfg.att_context_style = 'chunked_limited'
    enc = build_toy_pe_encoder(asr_encoder_cfg=cfg)
    assert len(enc.asr_encoder.att_context_size_all) == 4, "fixture no longer exercises multi-context"

    enc.att_context_size = [8, 1]
    assert enc.asr_encoder.att_context_size == [8, 1]
    assert enc.asr_encoder.att_context_size_all == [[8, 1]], "choice set not collapsed -- training would resample"
    if getattr(enc.asr_encoder, "att_context_probs", None) is not None:
        assert enc.asr_encoder.att_context_probs == [1.0]


@pytest.mark.unit
def test_att_context_size_ctor_arg_pins_the_branch_at_construction():
    """`_set_encoder_att_context` early-returns for dynamic chunking, so construction must pin it."""
    cfg = toy_asr_encoder_cfg()
    cfg.att_context_size = [[8, 7], [8, 3], [8, 1], [8, 0]]  # chunked_limited needs left % (right+1) == 0
    cfg.att_context_style = 'chunked_limited'

    unpinned = build_toy_pe_encoder(asr_encoder_cfg=toy_asr_encoder_cfg())
    assert unpinned.att_context_size == [-1, -1], "unpinned should keep the checkpoint's own setting"

    pinned = build_toy_pe_encoder(asr_encoder_cfg=cfg, att_context_size=[8, 1])
    assert pinned.att_context_size == [8, 1]
    assert pinned.asr_encoder.att_context_size_all == [[8, 1]]


@pytest.mark.unit
@pytest.mark.parametrize("member_prefix", ["", "./"])
def test_nemo_member_lookup_tolerates_both_naming_conventions(tmp_path, member_prefix):
    """``.nemo`` archives disagree on member naming, so lookup must go by basename.

    The ASR checkpoint stores ``./model_config.yaml`` while the Sortformer one stores a bare
    ``model_config.yaml``; an exact-name lookup silently works for one and raises for the other.
    """
    import tarfile

    from nemo.collections.asr.modules.parallel_expert_encoder import _nemo_member

    payload = tmp_path / "model_config.yaml"
    payload.write_text("target: dummy\n")
    archive_path = tmp_path / "bundle.nemo"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(payload, arcname=f"{member_prefix}model_config.yaml")

    with tarfile.open(archive_path) as archive:
        member = _nemo_member(archive, "model_config.yaml")
        assert archive.extractfile(member).read().decode() == "target: dummy\n"


@pytest.mark.unit
def test_nemo_member_raises_when_absent(tmp_path):
    import tarfile

    from nemo.collections.asr.modules.parallel_expert_encoder import _nemo_member

    other = tmp_path / "something_else.txt"
    other.write_text("x")
    archive_path = tmp_path / "bundle.nemo"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(other, arcname="something_else.txt")

    with tarfile.open(archive_path) as archive:
        with pytest.raises(FileNotFoundError, match="model_weights.ckpt"):
            _nemo_member(archive, "model_weights.ckpt")


@pytest.mark.unit
def test_constructing_pe_does_not_retune_the_frozen_diarizer():
    """PE must inherit the diarizer checkpoint's streaming knobs, not impose its own.

    The diarizer is frozen during SpeechLM training, so its streaming behaviour should be exactly
    what it was trained with. The old defaults (fifo 40 / update 300 / cache 188) came from the
    placeholder bundle's model card, which pairs a *different* Sortformer -- and `chunk_len` was
    taken from `online_inference_length`, which is PE's ASR long-form window, a different quantity.
    """
    diar_cfg = toy_diarization_model_cfg()
    reference = SortformerEncLabelModel.from_config_dict(diar_cfg).sortformer_modules
    baseline = {k: getattr(reference, k) for k in ("chunk_len", "fifo_len", "spkcache_len", "spkcache_update_period")}

    enc = build_toy_pe_encoder(diarization_model_cfg=diar_cfg)
    sm = enc.diarization_model.sortformer_modules
    assert {k: getattr(sm, k) for k in baseline} == baseline, "construction retuned the frozen diarizer"

    enc.setup_streaming_params() if hasattr(enc, "setup_streaming_params") else None
    assert {k: getattr(sm, k) for k in baseline} == baseline, "setup_streaming_params retuned it"


@pytest.mark.unit
def test_explicit_diar_streaming_override_is_honored():
    """An explicit knob still wins, and only that knob changes."""
    diar_cfg = toy_diarization_model_cfg()
    reference = SortformerEncLabelModel.from_config_dict(diar_cfg).sortformer_modules
    baseline = {k: getattr(reference, k) for k in ("chunk_len", "spkcache_len", "spkcache_update_period")}

    enc = build_toy_pe_encoder(diarization_model_cfg=diar_cfg, diar_fifo_len=7)
    sm = enc.diarization_model.sortformer_modules
    assert sm.fifo_len == 7
    assert {k: getattr(sm, k) for k in baseline} == baseline, "an explicit override leaked into other knobs"


@pytest.mark.unit
def test_missing_rttm_rows_are_detected_per_row():
    """The mask must be per-row: a batch normally mixes cuts with and without an RTTM."""
    enc = build_toy_pe_encoder()
    real = torch.rand(4, _N_SPK)
    sentinel = torch.full((4, _N_SPK), -1.0)
    batch = torch.stack([real, sentinel, real, sentinel])
    mask = enc.missing_rttm_rows(batch)
    assert mask.tolist() == [False, True, False, True]
    assert enc.missing_rttm_rows(None) is None


@pytest.mark.unit
def test_partially_sentinel_row_is_not_treated_as_missing():
    """Only a row that is ENTIRELY sentinel counts -- `.all`, not `.any`.

    A row that merely dips below the sentinel somewhere still carries real supervision; treating it
    as missing would silently discard a whole cut's RTTM.
    """
    enc = build_toy_pe_encoder()
    row = torch.zeros(1, 4, _N_SPK)
    row[0, 0, 0] = -1.0
    assert enc.missing_rttm_rows(row).tolist() == [False]


@pytest.mark.unit
def test_sentinel_rows_get_diarizer_predictions_and_others_keep_rttm():
    """End to end: in one batch, a sentinel row must be fused from diarizer output while a real
    row keeps its own targets. Before this, an all-ones placeholder was thresholded into
    "every speaker active at every frame" and fused as if it were ground truth."""
    enc = build_toy_pe_encoder().eval()
    seen = {}
    original = enc._fuse_diar_and_asr

    def spy(asr_encoded, spk_targets):
        seen["targets"] = spk_targets.detach().clone()
        return original(asr_encoded, spk_targets)

    enc._fuse_diar_and_asr = spy

    mel = torch.randn(2, _MEL_FEATURES, 256)
    length = torch.tensor([256, 256])
    n_frames = 256 // _SUBSAMPLING_FACTOR
    real = torch.rand(n_frames, _N_SPK)
    targets = torch.stack([real, torch.full((n_frames, _N_SPK), -1.0)])

    with torch.no_grad():
        enc(audio_signal=mel, length=length, spk_targets=targets.clone())

    used = seen["targets"]
    assert torch.allclose(used[0], real, atol=1e-5), "row 0 should keep its RTTM targets"
    assert not bool((used[1] <= -1.0).all()), "row 1 should have been replaced by diarizer output"
