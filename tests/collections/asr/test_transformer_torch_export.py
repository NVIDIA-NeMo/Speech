# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from examples.asr.export.build_torch_trt_encoder import capture_encoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder


@pytest.mark.unit
@pytest.mark.parametrize("position", ["rel_pos", "rope", "abs_pos", "no_pos"])
@pytest.mark.parametrize("subsampling", ["feature_stacking", "stacking", "stacking_norm", None])
def test_dynamic_transformer_torch_export(position: str, subsampling: str | None) -> None:
    torch.manual_seed(42)
    model = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        feat_out=24,
        subsampling=subsampling,
        subsampling_factor=4,
        self_attention_model=position,
        rotary_fraction=0.5,
        qk_norm=True,
        qkv_bias=True,
        drop_rate=0.0,
        pos_emb_max_len=128,
        sync_max_audio_length=False,
    ).eval()
    program = capture_encoder(model, min_frames=8, opt_frames=16, max_frames=128, opt_batch=2, max_batch=4)
    restored = program.module()
    for batch, time, lengths in [
        (2, 16, [16, 9]),
        (1, 9, [9]),
        (3, 23, [23, 13, 0]),
        (2, 80, [65, 0]),
        (4, 128, [128, 65, 1, 0]),
    ]:
        features = torch.randn(batch, 8, time)
        lengths = torch.tensor(lengths)
        with torch.no_grad():
            expected = model(features, lengths)
            actual = restored(features, lengths)
        torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("frames", [1, 2, 3, 8])
def test_relative_bias_export(frames: int) -> None:
    model = TransformerEncoder(feat_in=8, d_model=32, n_heads=2, n_layers=1, drop_rate=0).double()
    attention = model.layers[0].attn
    query = torch.randn(2, 2, frames, 16, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(1, 2 * frames - 1, 32, dtype=torch.float64, requires_grad=True)
    native, native_query = attention._build_rel_pos_score_mod(query, positions)
    exported, exported_query = attention._build_rel_pos_score_mod(query, positions, for_export=True)
    torch.testing.assert_close(native_query, exported_query, atol=0, rtol=0)
    torch.testing.assert_close(native._relative_position_bias, exported._relative_position_bias, atol=0, rtol=0)
    variables = (query, positions, attention.linear_pos.weight, attention.pos_bias_v)
    native_gradients = torch.autograd.grad(native._relative_position_bias.sum(), variables)
    exported_gradients = torch.autograd.grad(exported._relative_position_bias.sum(), variables)
    for expected, actual in zip(native_gradients, exported_gradients):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.unit
def test_default_transformer_torch_export() -> None:
    model = TransformerEncoder(feat_in=8, d_model=32, n_heads=2, n_layers=1, drop_rate=0).eval()
    program = capture_encoder(model)
    with torch.no_grad():
        for batch, frames in ((1, 17), (3, 127)):
            features = torch.randn(batch, 8, frames)
            lengths = torch.tensor([frames] + [0] * (batch - 1))
            expected = model(features, lengths)
            actual = program.module()(features, lengths)
            torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "position,subsampling",
    [
        ("rope", "feature_stacking"),
        ("rel_pos", "feature_stacking"),
        ("rel_pos", "stacking"),
        ("rel_pos", "stacking_norm"),
        ("rel_pos", None),
    ],
)
def test_dynamic_transformer_tensorrt(tmp_path, monkeypatch, position: str, subsampling: str | None):
    if not torch.cuda.is_available():
        pytest.skip("TensorRT execution requires CUDA.")
    trt = pytest.importorskip("tensorrt")
    pytest.importorskip("torch_tensorrt")
    from examples.asr.export.build_torch_trt_encoder import build_encoder

    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    model = (
        TransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            feat_out=24,
            subsampling=subsampling,
            subsampling_factor=4,
            self_attention_model=position,
            rotary_fraction=0.5,
            qk_norm=True,
            qkv_bias=True,
            drop_rate=0.0,
            pos_emb_max_len=128,
            sync_max_audio_length=False,
        )
        .eval()
        .cuda()
    )
    program = capture_encoder(model, min_frames=8, opt_frames=16, max_frames=128, opt_batch=2, max_batch=4)
    plan = build_encoder(program, 8, 16, 128, 2, 4, 8, workspace_bytes=128 * 1024**2)
    saved = tmp_path / "encoder.plan"
    saved.write_bytes(plan)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(saved.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "audio_signal",
        "length",
        "outputs",
        "encoded_lengths",
    }
    dtypes = {trt.float32: torch.float32, trt.int64: torch.int64}
    with torch.inference_mode():
        for batch, frames, lengths in [(2, 16, [16, 9]), (1, 9, [9]), (4, 23, [23, 13, 1, 0]), (2, 128, [128, 0])]:
            features = torch.randn(batch, 8, frames, device="cuda")
            lengths = torch.tensor(lengths, device="cuda")
            expected = model(features, lengths)
            buffers = {"audio_signal": features, "length": lengths}
            for name, value in buffers.items():
                assert context.set_input_shape(name, tuple(value.shape))
                assert context.set_tensor_address(name, value.data_ptr())
            assert not context.infer_shapes()
            for name in ("outputs", "encoded_lengths"):
                buffers[name] = torch.empty(
                    tuple(context.get_tensor_shape(name)), device="cuda", dtype=dtypes[engine.get_tensor_dtype(name)]
                )
                assert context.set_tensor_address(name, buffers[name].data_ptr())
            assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            torch.testing.assert_close(buffers["outputs"], expected[0], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(buffers["encoded_lengths"], expected[1], atol=0, rtol=0)
