# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import types
from unittest import mock

import pytest
import torch
from easymagpie_vllm_omni import watermark
from easymagpie_vllm_omni.watermark import (
    AudioChunk,
    DisabledAudioWatermarker,
    TaperedPerthWatermarker,
)


class _FakeAudioProcessor:
    def signal_to_magphase(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return signal, torch.zeros_like(signal)

    def magphase_to_signal(self, magnitude: torch.Tensor, _phase: torch.Tensor) -> torch.Tensor:
        return magnitude


class _FakeEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, magnitude: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.batch_sizes.append(int(magnitude.shape[0]))
        return magnitude + 0.125, None


class _FakePerthNet:
    def __init__(self) -> None:
        self.hp = types.SimpleNamespace(sample_rate=22_050, n_fft=8)
        self.device = torch.device("cpu")
        self.ap = _FakeAudioProcessor()
        self.encoder = _FakeEncoder()


def _chunks(*waveforms: torch.Tensor) -> list[AudioChunk]:
    return [
        AudioChunk(request_id=f"req-{index}", is_final=False, audio=waveform)
        for index, waveform in enumerate(waveforms)
    ]


def test_unindexed_cuda_device_is_normalized() -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=2),
    ):
        assert watermark.normalized_device("cuda") == "cuda:2"


def test_equal_length_waveforms_are_batched_and_length_is_preserved() -> None:
    perth_net = _FakePerthNet()
    watermarker = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)
    result = watermarker.apply(_chunks(torch.zeros(16), torch.full((16,), 0.5), torch.zeros(12)))

    assert perth_net.encoder.batch_sizes == [2, 1]
    assert [tensor.numel() for tensor in result] == [16, 16, 12]
    ramp = torch.sin(torch.linspace(0, torch.pi / 2, 4)).square()
    envelope = torch.ones(16)
    envelope[:4] = ramp
    envelope[-4:] = ramp.flip(0)
    torch.testing.assert_close(result[0], 0.125 * envelope)
    torch.testing.assert_close(result[1], 0.5 + 0.125 * envelope)


def test_watermark_delta_taper_preserves_chunk_edge_samples() -> None:
    original = torch.linspace(-0.5, 0.5, 16).repeat(2, 1)
    marked = original + 0.125

    result = watermark.taper_watermark_delta(marked, original, edge_samples=4)

    torch.testing.assert_close(result[:, 0], original[:, 0])
    torch.testing.assert_close(result[:, -1], original[:, -1])
    torch.testing.assert_close(result[:, 4:-4], marked[:, 4:-4])


def test_short_waveform_is_left_unchanged() -> None:
    perth_net = _FakePerthNet()
    watermarker = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)
    original = torch.tensor([0.1, -0.2, 0.3, -0.4])

    result = watermarker.apply(_chunks(original.clone()))

    assert perth_net.encoder.batch_sizes == []
    torch.testing.assert_close(result[0], original)


def test_disabled_watermarker_is_identity() -> None:
    waveforms = [torch.zeros(16), torch.ones(8)]
    chunks = _chunks(*waveforms)
    result = DisabledAudioWatermarker().apply(chunks)
    assert result[0] is waveforms[0]
    assert result[1] is waveforms[1]


def test_explicit_disable_bypasses_perth_load() -> None:
    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "off"}),
        mock.patch.object(watermark, "_load_perth_net") as load_perth,
    ):
        created = watermark.create_audio_watermarker("cpu", sample_rate=22_050)

    load_perth.assert_not_called()
    assert isinstance(created, DisabledAudioWatermarker)


def test_enabled_initialization_failure_is_fatal() -> None:
    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "1"}),
        mock.patch.object(watermark, "_load_perth_net", side_effect=ValueError("bad checkpoint")),
        pytest.raises(RuntimeError, match="could not be initialized"),
    ):
        TaperedPerthWatermarker("cpu", sample_rate=22_050)


def test_encoder_is_compiled_on_cuda() -> None:
    encoder = object()
    perth_net = types.SimpleNamespace(
        hp=types.SimpleNamespace(sample_rate=32_000, n_fft=8),
        encoder=encoder,
        device="cuda:0",
        ap=_FakeAudioProcessor(),
    )
    compiled = mock.Mock(name="compiled_encoder", side_effect=lambda mag: (mag, None))

    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "1"}),
        mock.patch.object(watermark, "normalized_device", return_value="cuda:0"),
        mock.patch.object(torch, "compile", return_value=compiled) as compile_fn,
        mock.patch.object(torch.cuda, "is_available", return_value=False),
    ):
        result = TaperedPerthWatermarker("cuda:0", sample_rate=22_050, perth_net=perth_net)

    compile_fn.assert_called_once_with(encoder, mode="default", dynamic=True)
    assert result.perth_net.encoder is compiled


def test_encoder_is_not_compiled_on_cpu() -> None:
    encoder = object()
    perth_net = types.SimpleNamespace(
        hp=types.SimpleNamespace(sample_rate=32_000, n_fft=8),
        encoder=encoder,
        device="cpu",
    )
    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "1"}),
        mock.patch.object(torch, "compile") as compile_fn,
    ):
        result = TaperedPerthWatermarker("cpu", sample_rate=22_050, perth_net=perth_net)

    compile_fn.assert_not_called()
    assert result.perth_net.encoder is encoder


def test_load_perth_net_imports_perthnet_without_sys_modules_shim() -> None:
    source = inspect.getsource(watermark._load_perth_net)
    assert "_install_perth_net_namespace" not in source
    assert "sys.modules" not in source
    assert "perth.perth_net.perth_net_implicit.model.perth_net" in source


def test_create_audio_watermarker_forwards_models_dir() -> None:
    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "1"}),
        mock.patch.object(watermark, "_load_perth_net") as load_perth,
    ):
        fake = mock.MagicMock()
        fake.hp.sample_rate = 32_000
        fake.device = "cpu"
        fake.encoder = object()
        load_perth.return_value = fake
        watermark.create_audio_watermarker("cpu", sample_rate=22_050, models_dir="/ckpt/watermark/perth")

    load_perth.assert_called_once_with("cpu", models_dir="/ckpt/watermark/perth")
