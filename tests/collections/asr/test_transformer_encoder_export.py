# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX parity for the offline Transformer encoder, including dynamic padded batches."""

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder


@pytest.mark.unit
@pytest.mark.parametrize("position", ["rope", "rel_pos", "abs_pos", "no_pos"])
@pytest.mark.parametrize("mask", ["full", "causal", "tail"])
def test_transformer_onnx_parity(tmp_path, position, mask):
    _check_export(tmp_path, position=position, mask=mask)


@pytest.mark.unit
@pytest.mark.parametrize("subsampling", ["stacking", "stacking_norm", None])
def test_transformer_onnx_subsampling(tmp_path, subsampling):
    _check_export(tmp_path, subsampling=subsampling)


def _check_export(tmp_path, position="rope", mask="full", subsampling="feature_stacking"):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(42)
    model = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        feat_out=24,
        subsampling=subsampling,
        subsampling_factor=4 if subsampling else 1,
        self_attention_model=position,
        rotary_fraction=0.5,
        qk_norm=True,
        qkv_bias=True,
        drop_rate=0.0,
        xscaling=True,
        pos_emb_max_len=128,
        attn_mode="causal" if mask == "causal" else "full",
        causal_tail_len=3 if mask == "tail" else 0,
        causal_tail_block_size=2,
        sync_max_audio_length=False,
    ).eval()
    # Nonzero Transformer-XL biases exercise both content and relative-position terms.
    if position == "rel_pos":
        for layer in model.layers:
            torch.nn.init.normal_(layer.attn.pos_bias_u, std=0.1)
            torch.nn.init.normal_(layer.attn.pos_bias_v, std=0.1)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    example = (torch.randn(2, 8, 16), torch.tensor([16, 9]))
    path = tmp_path / "encoder.onnx"
    with torch.no_grad():
        expected = model(*example)
        model.export(str(path), input_example=example, onnx_opset_version=17)
        # Export must not leave the regular forward using the export implementation.
        actual = model(*example)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert state.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    assert [v.name for v in session.get_inputs()] == ["audio_signal", "length"]
    assert [v.name for v in session.get_outputs()] == ["outputs", "encoded_lengths"]
    assert isinstance(session.get_inputs()[0].shape[0], str)
    assert isinstance(session.get_inputs()[0].shape[2], str)
    # Change batch size, length, padding remainder and mask values without re-export.
    for batch, frames, lengths in [
        (2, 16, [16, 9]),
        (1, 7, [7]),
        (3, 23, [23, 13, 1]),
        (2, 68, [65, 0]),
    ]:
        audio = torch.randn(batch, 8, frames)
        length = torch.tensor(lengths, dtype=torch.int64)
        with torch.no_grad():
            expected = model(audio, length)
            export_output = model.forward_for_export(audio, length)
        torch.testing.assert_close(export_output[0], expected[0], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(export_output[1], expected[1], rtol=0, atol=0)
        actual = session.run(None, {"audio_signal": audio.numpy(), "length": length.numpy()})
        assert np.isfinite(actual[0]).all()
        np.testing.assert_allclose(actual[0], expected[0].numpy(), rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(actual[1], expected[1].numpy())
