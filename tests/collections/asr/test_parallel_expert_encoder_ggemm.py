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

from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder
from nemo.collections.asr.modules.parallel_expert_encoder_ggemm import (
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
)
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder

# ``@experimental`` wraps the class in a wrapt proxy, so ``__new__`` (used to build
# bare instances that skip the heavy real ``__init__``) must target the underlying
# class. Attribute access / isinstance still go through the proxy name.
_PEE = getattr(ParallelExpertEncoder, "__wrapped__", ParallelExpertEncoder)
# @experimental wraps these in a wrapt proxy; isinstance needs the real class.
_MOE_ENCODER_CLS = getattr(MoETransformerEncoder, "__wrapped__", MoETransformerEncoder)
_TF_ENCODER_CLS = getattr(TransformerEncoder, "__wrapped__", TransformerEncoder)


# ----------------------------------------------------------------------------- #
# Module-level context managers / helpers
# ----------------------------------------------------------------------------- #
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
# Static pure helpers
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
def test_match_module_io_casts_to_expert_dtype(param_dtype):
    """Mels arrive fp32; `_match_module_io` moves them onto the experts' device/dtype.

    It reads the dtype off the container's own parameters, so it is an instance
    method rather than the free function it used to be.
    """
    enc = build_toy_pe_encoder().to(param_dtype)
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    assert enc._match_module_io(tensor).dtype == param_dtype


@pytest.mark.unit
def test_match_module_io_paramless_container_unchanged():
    enc = build_toy_pe_encoder()
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    # Stub out the parameter source: with nothing to match, the tensor passes through.
    enc.pee = nn.Identity()
    out = enc._match_module_io(tensor)
    assert out is tensor and out.dtype == torch.float32


# ----------------------------------------------------------------------------- #
# forward() offline/online dispatch
# ----------------------------------------------------------------------------- #
def dispatch_stub(online_inference_length, enabled):
    """Bare encoder with both branch methods stubbed.

    forward() dispatches purely on `online_inference_enabled` (set by the
    `online_inference()` context manager) AND a positive window -- NOT on the audio
    length. Long-form splitting happens one level down, inside _forward_online.
    """
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.online_inference_length = online_inference_length
    enc.online_inference_enabled = enabled
    enc._forward = lambda **kw: ("offline", None, {})
    enc._forward_online = lambda **kw: ("online", None, {})
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "online_len, enabled, expected",
    [
        (500, True, "online"),  # context manager open + positive window
        (500, False, "offline"),  # not opened -> training/validation path
        (0, True, "offline"),  # windowing disabled by online_inference_length=0
        (0, False, "offline"),
    ],
)
def test_forward_dispatch(online_len, enabled, expected):
    enc = dispatch_stub(online_len, enabled)
    audio = torch.zeros(1, 8, 200)
    length = torch.tensor([200])
    assert enc.forward(audio, length)[0] == expected


@pytest.mark.unit
def test_online_inference_context_toggles_dispatch():
    """The context manager is the only thing that turns the windowed path on."""
    enc = dispatch_stub(500, False)
    audio, length = torch.zeros(1, 8, 200), torch.tensor([200])
    assert enc.forward(audio, length)[0] == "offline"
    with _PEE.online_inference(enc):
        assert enc.forward(audio, length)[0] == "online"
    assert enc.forward(audio, length)[0] == "offline"  # restored on exit


# ----------------------------------------------------------------------------- #
# _forward_online orchestration (stubbed ASR encoder, provided spk_targets)
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "win, n_frames",
    [
        (10, 240),  # 3 full windows
        (10, 200),  # partial last window
        (5, 64),  # many small windows
        (50, 160),  # single window -> fast path delegates to _forward
    ],
)
def test_forward_online_output_length_telescopes(win, n_frames):
    """Windowed output must telescope back to the same frame count as one offline pass.

    Uses the real encoder: the old hand-built stub could not express the three-expert
    container, and a stub that drifts from the module is how this suite went stale.
    """
    b = 2
    enc = build_toy_pe_encoder(
        online_inference_length=win,
        diar_spkcache_update_period=win,
    )
    enc._suppress_online_pbar = True
    enc.eval()

    mels = torch.randn(b, _MEL_FEATURES, n_frames)
    length = torch.tensor([n_frames] * b)

    with torch.no_grad():
        off_out, off_len = enc(mels, length)
        with enc.online_inference():
            on_out, on_len = enc(mels, length)

    assert on_out.shape[0] == b and on_out.shape[1] == enc.d_model
    # Both paths cover the same audio, so they must agree on the frame count.
    assert on_out.shape[2] == off_out.shape[2], (on_out.shape, off_out.shape)
    assert on_len.tolist() == [on_out.shape[2]] * b
    assert torch.isfinite(on_out).all()


# ----------------------------------------------------------------------------- #
# Bundle checkpoint helpers
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_extract_encoder_state_dict_selects_requested_expert_encoder():
    state = {
        "encoder.layers.0.weight": torch.tensor([1.0]),
        "transformer_encoder.layers.0.weight": torch.tensor([2.0]),
        "decoder.weight": torch.tensor([3.0]),
    }

    asr_encoder = ParallelExpertEncoderPT.extract_encoder_state_dict(state)
    speaker_encoder = ParallelExpertEncoderPT.extract_encoder_state_dict(state, "transformer_encoder")

    assert set(asr_encoder) == {"layers.0.weight"}
    assert asr_encoder["layers.0.weight"] is state["encoder.layers.0.weight"]
    assert set(speaker_encoder) == {"layers.0.weight"}
    assert speaker_encoder["layers.0.weight"] is state["transformer_encoder.layers.0.weight"]


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
        ("nemo.collections.asr.modules.parallel_expert_encoder_ggemm.ParallelExpertEncoderPT", True),
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
# Bundle save guard rails
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_save_to_nemo_rejects_non_encoder(tmp_path):
    with pytest.raises(TypeError):
        ParallelExpertEncoderPT.save_to_nemo(
            nn.Linear(2, 2), str(tmp_path / "out.nemo"), template_bundle_path=str(tmp_path / "tpl.nemo")
        )


@pytest.mark.unit
def test_save_to_nemo_missing_template(tmp_path):
    # __new__ skips the heavy initializer so the test reaches the template check.
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
# The wrapper loads real sub-encoders and fuses them:
#   * an ASR ConformerEncoder (cf. tests/collections/asr/test_conformer_encoder.py)
#   * a Sortformer diarizer    (cf. tests/collections/speaker_tasks/test_diar_sortformer_models.py)
# These tests build tiny-but-real instances of both and run the wrapper end to end.
# ----------------------------------------------------------------------------- #
_MEL_FEATURES = 128
_ASR_D_MODEL = 32  # speech + sound expert width
_DIAR_FC_D_MODEL = 16  # half-width speaker expert
_DIAR_TF_D_MODEL = 16
_N_SPK = 4
_SUBSAMPLING_FACTOR = 8
_N_LAYERS = 1
_N_HEADS_WIDE = 2  # head_dim 16, shared by all three experts
_N_HEADS_NARROW = 1
_CHUNK_LEN = 500  # enc frames per online window
_SPKCACHE_LEN = 16


def _toy_expert_cfg(target: str, d_model: int, n_heads: int, **extra) -> DictConfig:
    """One tiny flex-encoder expert config.

    All three experts use the flex TransformerEncoder family with a shared
    front-end (same feat_in / subsampling / frame rate) and rope attention, which is
    what lets forward_packed batch them into one attention group.
    """
    cfg = {
        '_target_': target,
        'feat_in': _MEL_FEATURES,
        'feat_out': -1,
        'n_layers': _N_LAYERS,
        'd_model': d_model,
        'n_heads': n_heads,
        'subsampling': 'feature_stacking',
        'subsampling_factor': _SUBSAMPLING_FACTOR,
        'ff_expansion': 1.0,
        'self_attention_model': 'rope',
        'pos_emb_max_len': 5000,
        'xscaling': False,
        'qkv_bias': False,
        'qk_norm': False,
        'pre_block_norm': True,
        'attn_mode': 'full',
        'drop_rate': 0.0,
        'dropout_pre_encoder': 0.0,
        'dropout_emb': 0.0,
    }
    cfg.update(extra)
    return DictConfig(cfg)


def toy_speech_expert_cfg() -> DictConfig:
    """Speech expert: the MoE backbone the speaker kernel and sound merge fuse into."""
    return _toy_expert_cfg(
        'nemo.collections.asr.modules.MoETransformerEncoder',
        d_model=_ASR_D_MODEL,
        n_heads=_N_HEADS_WIDE,
        moe_num_experts=4,
        moe_top_k=2,
    )


def toy_speaker_expert_cfg() -> DictConfig:
    """Speaker expert with half the width of the speech and sound experts."""
    return _toy_expert_cfg(
        'nemo.collections.asr.modules.TransformerEncoder',
        d_model=_DIAR_FC_D_MODEL,
        n_heads=_N_HEADS_NARROW,
    )


def toy_sound_expert_cfg() -> DictConfig:
    """Sound expert: same width as speech, since the merge is an elementwise add."""
    return _toy_expert_cfg(
        'nemo.collections.asr.modules.TransformerEncoder',
        d_model=_ASR_D_MODEL,
        n_heads=_N_HEADS_WIDE,
    )


def toy_sound_ctc_head_cfg() -> DictConfig:
    """Tiny sound CTC head with both event and style tags."""
    vocabulary = [
        'ordinary_piece',
        '<ev:laughter>',
        '<ev:music>',
        '<sty:stt:anger>',
        '<sty:end:anger>',
    ]
    return DictConfig(
        {
            '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
            'feat_in': _ASR_D_MODEL,
            'num_classes': len(vocabulary),
            'vocabulary': vocabulary,
            'add_blank': True,
        }
    )


def toy_sortformer_modules_cfg() -> DictConfig:
    """Sortformer head + streaming cache logic, loaded separately from the encoder."""
    return DictConfig(
        {
            '_target_': 'nemo.collections.asr.modules.sortformer_modules.SortformerModules',
            'num_spks': _N_SPK,
            'dropout_rate': 0.0,
            'fc_d_model': _DIAR_FC_D_MODEL,
            'tf_d_model': _DIAR_TF_D_MODEL,
            'subsampling_factor': _SUBSAMPLING_FACTOR,
            'spkcache_len': _SPKCACHE_LEN,
            'fifo_len': 0,
            'chunk_len': _CHUNK_LEN,
            'spkcache_update_period': _CHUNK_LEN,
            'chunk_left_context': 0,
            'chunk_right_context': 0,
            'spkcache_sil_frames_per_spk': 1,
        }
    )


def build_toy_pe_encoder(**overrides) -> ParallelExpertEncoder:
    """Construct a real encoder from the tiny three-expert configs."""
    kwargs = dict(
        speech_expert_cfg=toy_speech_expert_cfg(),
        speaker_expert_cfg=toy_speaker_expert_cfg(),
        sound_expert_cfg=toy_sound_expert_cfg(),
        sortformer_modules_cfg=toy_sortformer_modules_cfg(),
        asr_normalize_type='per_feature',
        # Keep the input far below one window so forward() stays on the offline path.
        online_inference_length=_CHUNK_LEN,
        chunk_left_context=0,
        chunk_right_context=0,
        diar_fifo_len=0,
        diar_spkcache_update_period=_CHUNK_LEN,
        diar_spkcache_len=_SPKCACHE_LEN,
        # The existing fixture exercises the legacy encoder-state route. Tests for the
        # new default SoundToken route provide a CTC head explicitly below.
        merge_sound_expert_to_asr=True,
    )
    kwargs.update(overrides)
    return ParallelExpertEncoder(**kwargs)


@pytest.mark.unit
def test_pe_encoder_builds_and_wires_all_three_experts():
    enc = build_toy_pe_encoder()
    # All three experts are real flex encoders inside one GGEMM container.
    assert set(enc.pee.expert_names) == {"speech", "speaker", "sound"}
    assert isinstance(enc.pee.experts["speech"], _MOE_ENCODER_CLS)
    assert isinstance(enc.pee.experts["speaker"], _TF_ENCODER_CLS)
    assert isinstance(enc.pee.experts["sound"], _TF_ENCODER_CLS)
    # Decoder/head metadata belongs to the owning model, not the GGEMM container.
    assert enc.get_expert_task("speech") == "asr_encoder"
    assert enc.get_expert_task("speaker") == "diarization"
    assert enc.get_expert_task("sound") == "sound_ctc"
    assert not hasattr(enc.pee, "expert_tasks")
    # The speech expert is the backbone: it drives the drop-in ConformerEncoder props.
    assert enc.d_model == _ASR_D_MODEL
    assert enc.subsampling_factor == _SUBSAMPLING_FACTOR
    # Speaker count + fusion kernel come from the Sortformer head.
    assert enc.n_spk == _N_SPK
    assert enc.diar_kernel.shape == (_N_SPK, _ASR_D_MODEL)
    # The sound merge is an elementwise add, so sound must match the speech width.
    assert enc.pee.experts["sound"].d_model == enc.d_model
    # Defaults: the auxiliary speaker and sound experts are frozen; only the speech
    # backbone and fusion layers train.
    assert all(not p.requires_grad for p in enc.pee.experts["speaker"].parameters())
    assert all(not p.requires_grad for p in enc.sortformer_modules.parameters())
    assert any(p.requires_grad for p in enc.pee.experts["speech"].parameters())
    assert all(not p.requires_grad for p in enc.pee.experts["sound"].parameters())


@pytest.mark.unit
def test_pe_encoder_train_and_unfreeze_preserve_auxiliary_freezing():
    enc = build_toy_pe_encoder()

    enc.train()
    enc.unfreeze()

    assert enc.pee.experts["speech"].training
    assert any(p.requires_grad for p in enc.pee.experts["speech"].parameters())
    for role in ("speaker", "sound"):
        assert not enc.pee.experts[role].training
        assert all(not p.requires_grad for p in enc.pee.experts[role].parameters())
    assert not enc.sortformer_modules.training
    assert all(not p.requires_grad for p in enc.sortformer_modules.parameters())


@pytest.mark.unit
def test_ggemm_packed_matches_per_expert_reference():
    """The packed inference kernel must preserve every expert's native output."""
    enc = build_toy_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 64)
    lengths = torch.tensor([64, 48])
    signal, signal_lengths = enc._prepare_input(mels, lengths)

    with torch.no_grad():
        reference = enc.pee.forward_all(signal, signal_lengths)
        packed = enc.pee.forward_packed(signal, signal_lengths)

    assert set(packed) == set(reference) == {"speech", "speaker", "sound"}
    for role in reference:
        reference_states, reference_lengths = reference[role]
        packed_states, packed_lengths = packed[role]
        torch.testing.assert_close(packed_states, reference_states, atol=1e-5, rtol=1e-4)
        assert torch.equal(packed_lengths, reference_lengths)


@pytest.mark.unit
def test_pe_encoder_uses_separate_training_and_inference_compute_paths(monkeypatch):
    """Training uses native expert forwards; eval uses the fused GGEMM path."""
    enc = build_toy_pe_encoder()
    mels = torch.randn(1, _MEL_FEATURES, 64)
    lengths = torch.tensor([64])
    spk_targets = torch.zeros(1, 8, _N_SPK)
    calls = []

    original_forward_all = enc.pee.forward_all
    original_forward_packed = enc.pee.forward_packed

    def record_forward_all(*args, **kwargs):
        calls.append("forward_all")
        return original_forward_all(*args, **kwargs)

    def record_forward_packed(*args, **kwargs):
        calls.append("forward_packed")
        return original_forward_packed(*args, **kwargs)

    monkeypatch.setattr(enc.pee, "forward_all", record_forward_all)
    monkeypatch.setattr(enc.pee, "forward_packed", record_forward_packed)

    enc.train()
    with torch.no_grad():
        train_output, _ = enc(mels, lengths, spk_targets=spk_targets)
    assert calls == ["forward_all"]
    assert train_output.requires_grad is False

    calls.clear()
    enc.eval()
    with torch.no_grad():
        eval_output, _ = enc(mels, lengths, spk_targets=spk_targets)
    assert calls == ["forward_packed"]
    assert eval_output.requires_grad is False


@pytest.mark.unit
def test_pe_encoder_activation_checkpointing_is_pee_local(monkeypatch):
    """The owning model controls checkpointing without changing native experts."""
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    enc = build_toy_pe_encoder().train()
    checkpointed = []

    def record_checkpoint(function, *args, **kwargs):
        checkpointed.append(function)
        return torch_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(
        "nemo.collections.asr.modules.parallel_expert_encoder_ggemm.checkpoint",
        record_checkpoint,
    )
    for expert in enc.pee.experts.values():
        parameter = next(expert.parameters())

        def lightweight_forward(audio_signal, length, bypass_pre_encode=False, *, _expert=expert, _p=parameter):
            del bypass_pre_encode
            encoded_length = torch.div(length, _SUBSAMPLING_FACTOR, rounding_mode="floor")
            encoded = _p.reshape(-1)[0] * audio_signal.new_ones(
                audio_signal.shape[0], _expert.d_model, int(encoded_length.max())
            )
            return encoded, encoded_length

        monkeypatch.setattr(expert, "forward", lightweight_forward)
    enc.set_activation_checkpointing(True)

    assert enc.activation_checkpointing is True
    assert all(not hasattr(expert, "activation_checkpointing") for expert in enc.pee.experts.values())

    mels = torch.randn(1, _MEL_FEATURES, 64)
    lengths = torch.tensor([64])
    outputs = enc._forward_all_training(mels, lengths)

    assert checkpointed == [enc.pee.experts["speech"]]
    assert set(outputs) == {"speech", "speaker", "sound"}
    assert all(torch.isfinite(output).all() for output, _ in outputs.values())
    sum(output.float().sum() for output, _ in outputs.values()).backward()
    assert any(parameter.grad is not None for parameter in enc.pee.experts["speech"].parameters())
    assert all(parameter.grad is None for parameter in enc.pee.experts["sound"].parameters())
    assert all(parameter.grad is None for parameter in enc.pee.experts["speaker"].parameters())


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE checkpoint test requires CUDA")
def test_pe_encoder_activation_checkpointing_real_cuda_backward():
    """Checkpoint real native expert/FlexAttention forwards without modifying them."""
    enc = build_toy_pe_encoder().train().cuda()
    enc.set_activation_checkpointing(True)

    mels = torch.randn(1, _MEL_FEATURES, 64, device="cuda")
    lengths = torch.tensor([64], device="cuda")
    spk_targets = torch.zeros(1, 8, _N_SPK, device="cuda")
    output, _ = enc(mels, lengths, spk_targets=spk_targets)
    output.float().sum().backward()

    assert torch.isfinite(output).all()
    assert any(parameter.grad is not None for parameter in enc.pee.experts["speech"].parameters())
    assert all(parameter.grad is None for parameter in enc.pee.experts["sound"].parameters())
    assert all(parameter.grad is None for parameter in enc.pee.experts["speaker"].parameters())


@pytest.mark.unit
def test_pe_encoder_sound_token_route_uses_frozen_ctc_tags():
    enc = build_toy_pe_encoder(
        merge_sound_expert_to_asr=False,
        sound_ctc_head_cfg=toy_sound_ctc_head_cfg(),
        freeze_sound=True,
    ).eval()

    assert enc.sound_event_tokens == ('<ev:laughter>', '<ev:music>')
    assert enc.sound_style_tokens == ('<sty:stt:anger>', '<sty:end:anger>')
    assert enc.n_sound_events == 2
    assert enc.n_sound_styles == 2
    assert all(not param.requires_grad for param in enc.sound_ctc_head.parameters())

    mels = torch.randn(1, _MEL_FEATURES, 160)
    length = torch.tensor([160])
    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    assert outputs.shape == (1, _ASR_D_MODEL, int(encoded_len[0]))
    assert torch.isfinite(outputs).all()


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
