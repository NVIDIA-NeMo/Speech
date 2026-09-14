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

import os
import random
import subprocess
import sys

import pytest
import torch

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor as NemoAudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.audio_preprocessing_standalone import (
    AudioToMelSpectrogramPreprocessor as StandaloneAudioToMelSpectrogramPreprocessor,
)


def _make_inputs(max_length=4096):
    torch.manual_seed(2026)
    signal = torch.randn(4, max_length, dtype=torch.float32)
    time = torch.linspace(0, 1, max_length, dtype=torch.float32)
    signal[0] += 0.05 * torch.sin(2 * torch.pi * 220 * time)
    signal[1] += 0.03 * torch.sin(2 * torch.pi * 440 * time)
    signal[2] += torch.linspace(-0.1, 0.1, max_length)
    lengths = torch.tensor([max_length, max_length - 317, max_length // 2 + 129, max_length // 4 + 73])
    return signal, lengths


def _make_benchmark_inputs(batch_size=64, max_length=16000, device="cuda"):
    torch.manual_seed(2026)
    signal = torch.randn(batch_size, max_length, dtype=torch.float32, device=device)
    offsets = (torch.arange(batch_size, device=device, dtype=torch.long) * 37) % (max_length // 2)
    lengths = max_length - offsets
    return signal, lengths


def _compare_preprocessors(config, *, training=False, dtype=None, atol=2e-5, rtol=2e-5):
    signal, lengths = _make_inputs()
    nemo_preprocessor = NemoAudioToMelSpectrogramPreprocessor(**config)
    standalone_preprocessor = StandaloneAudioToMelSpectrogramPreprocessor(**config)

    if training:
        nemo_preprocessor.train()
        standalone_preprocessor.train()
    else:
        nemo_preprocessor.eval()
        standalone_preprocessor.eval()

    if dtype is not None:
        nemo_preprocessor = nemo_preprocessor.to(dtype=dtype)
        standalone_preprocessor = standalone_preprocessor.to(dtype=dtype)

    torch.manual_seed(12345)
    nemo_features, nemo_lengths = nemo_preprocessor(input_signal=signal.clone(), length=lengths.clone())
    torch.manual_seed(12345)
    standalone_features, standalone_lengths = standalone_preprocessor(
        input_signal=signal.clone(), length=lengths.clone()
    )

    assert standalone_lengths.equal(nemo_lengths)
    assert standalone_features.shape == nemo_features.shape
    assert standalone_features.dtype == nemo_features.dtype
    torch.testing.assert_close(standalone_features.float(), nemo_features.float(), atol=atol, rtol=rtol)


def _benchmark_cuda_forward(preprocessor, signal, lengths, *, warmup=10, iterations=50):
    with torch.inference_mode():
        for _ in range(warmup):
            preprocessor(input_signal=signal, length=lengths)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            preprocessor(input_signal=signal, length=lengths)
        end.record()
        torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations, torch.cuda.max_memory_allocated()


def _run_cuda_benchmark_case(config, *, batch_size, max_length, warmup, iterations):
    signal, lengths = _make_benchmark_inputs(batch_size=batch_size, max_length=max_length)
    nemo_preprocessor = NemoAudioToMelSpectrogramPreprocessor(**config).cuda().eval()
    standalone_preprocessor = StandaloneAudioToMelSpectrogramPreprocessor(**config).cuda().eval()

    with torch.inference_mode():
        nemo_features, nemo_lengths = nemo_preprocessor(input_signal=signal, length=lengths)
        standalone_features, standalone_lengths = standalone_preprocessor(input_signal=signal, length=lengths)

    assert nemo_features.is_cuda
    assert standalone_features.is_cuda
    assert standalone_lengths.equal(nemo_lengths)
    torch.testing.assert_close(standalone_features.float(), nemo_features.float(), atol=1e-4, rtol=1e-4)

    nemo_ms, nemo_peak_memory = _benchmark_cuda_forward(
        nemo_preprocessor, signal, lengths, warmup=warmup, iterations=iterations
    )
    standalone_ms, standalone_peak_memory = _benchmark_cuda_forward(
        standalone_preprocessor, signal, lengths, warmup=warmup, iterations=iterations
    )

    return {
        "batch_size": batch_size,
        "max_length": max_length,
        "iterations": iterations,
        "nemo_ms": nemo_ms,
        "standalone_ms": standalone_ms,
        "speed_ratio": nemo_ms / standalone_ms,
        "nemo_peak_memory_gb": nemo_peak_memory / 1024**3,
        "standalone_peak_memory_gb": standalone_peak_memory / 1024**3,
    }


@pytest.mark.unit
def test_standalone_audio_to_mel_imports_from_modules_path():
    code = """
from nemo.collections.asr.modules.audio_preprocessing_standalone import AudioToMelSpectrogramPreprocessor
AudioToMelSpectrogramPreprocessor(dither=0)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"dither": 0, "pad_to": 0, "normalize": None},
        {
            "dither": 0,
            "pad_to": 16,
            "normalize": "per_feature",
            "features": 80,
            "lowfreq": 20,
            "highfreq": 7600,
        },
        {
            "sample_rate": 8000,
            "window_size": None,
            "window_stride": None,
            "n_window_size": 200,
            "n_window_stride": 80,
            "window": "hamming",
            "normalize": "all_features",
            "n_fft": 512,
            "preemph": None,
            "features": 40,
            "lowfreq": 50,
            "highfreq": 3600,
            "log_zero_guard_type": "clamp",
            "log_zero_guard_value": "eps",
            "dither": 0,
            "pad_to": 8,
            "mag_power": 1.0,
        },
        {
            "window_size": None,
            "window_stride": None,
            "n_window_size": 320,
            "n_window_stride": 160,
            "window": "blackman",
            "normalize": None,
            "n_fft": 512,
            "preemph": 0.95,
            "features": 32,
            "log": False,
            "dither": 0,
            "pad_to": 0,
            "frame_splicing": 3,
            "exact_pad": True,
            "pad_value": -11.0,
        },
        {
            "window": "bartlett",
            "normalize": {
                "fixed_mean": [[-14.0] * 24, [-13.5] * 24, [-13.0] * 24, [-12.5] * 24],
                "fixed_std": [[2.0] * 24, [2.1] * 24, [2.2] * 24, [2.3] * 24],
            },
            "features": 24,
            "lowfreq": 10,
            "highfreq": 7400,
            "mel_norm": None,
            "dither": 0,
            "pad_to": 4,
        },
    ],
)
def test_standalone_audio_to_mel_matches_nemo_outputs(config):
    _compare_preprocessors(config, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"features": 64, "n_fft": None, "lowfreq": 0, "highfreq": None, "mel_norm": "slaney"},
        {
            "sample_rate": 8000,
            "window_size": None,
            "window_stride": None,
            "n_window_size": 200,
            "n_window_stride": 80,
            "features": 40,
            "n_fft": 512,
            "lowfreq": 50,
            "highfreq": 3600,
            "mel_norm": None,
        },
    ],
)
def test_standalone_audio_to_mel_filter_banks_match_nemo(config):
    nemo_preprocessor = NemoAudioToMelSpectrogramPreprocessor(dither=0, **config)
    standalone_preprocessor = StandaloneAudioToMelSpectrogramPreprocessor(dither=0, **config)

    torch.testing.assert_close(
        standalone_preprocessor.filter_banks,
        nemo_preprocessor.filter_banks,
        atol=1e-7,
        rtol=1e-6,
    )


@pytest.mark.unit
def test_standalone_audio_to_mel_matches_nemo_dtype_conversion():
    _compare_preprocessors({"dither": 0, "pad_to": 0, "normalize": None}, dtype=torch.float16, atol=1e-3, rtol=1e-3)


@pytest.mark.unit
def test_standalone_audio_to_mel_matches_nemo_training_dither():
    _compare_preprocessors(
        {"dither": 1e-4, "pad_to": 0, "normalize": None},
        training=True,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.unit
def test_standalone_audio_to_mel_matches_nemo_training_narrowband_augmentation():
    config = {
        "dither": 0,
        "pad_to": 0,
        "normalize": None,
        "n_fft": 512,
        "nb_augmentation_prob": 1.0,
        "nb_max_freq": 3000,
        "rng": random.Random(7),
    }
    _compare_preprocessors(config, training=True, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_standalone_audio_to_mel_cuda_benchmark_matches_nemo(capsys):
    config = {
        "dither": 0,
        "pad_to": 0,
        "normalize": None,
        "features": 80,
    }
    batch_size = 64
    max_length = 16000
    iterations = 50

    result = _run_cuda_benchmark_case(
        config,
        batch_size=batch_size,
        max_length=max_length,
        warmup=10,
        iterations=iterations,
    )

    with capsys.disabled():
        print(
            "\nLogMel CUDA benchmark "
            f"(device={torch.cuda.get_device_name(0)}, batch_size={batch_size}, samples={max_length}, "
            f"iterations={iterations})"
        )
        print(f"NeMo AudioToMelSpectrogramPreprocessor: {result['nemo_ms']:.3f} ms/iter")
        print(f"Standalone AudioToMelSpectrogramPreprocessor: {result['standalone_ms']:.3f} ms/iter")
        print(f"Standalone speed vs NeMo: {result['speed_ratio']:.2f}x")


@pytest.mark.skipduringci
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(
    os.environ.get("NEMO_RUN_LONG_LOGMEL_BENCHMARK") != "1",
    reason="Set NEMO_RUN_LONG_LOGMEL_BENCHMARK=1 to run the long CUDA LogMel benchmark",
)
def test_standalone_audio_to_mel_cuda_long_benchmark_matches_nemo(batch_size, capsys):
    """Compare NeMo and standalone LogMel extraction on long CUDA inputs.

    This benchmark is opt-in via ``NEMO_RUN_LONG_LOGMEL_BENCHMARK=1`` because it allocates large tensors.
    On an NVIDIA H100 80GB HBM3 with 10-minute 16 kHz inputs, 1 warmup, and 3 measured iterations,
    standalone was consistently faster while using the same peak memory:

    +------------+---------------+-------------------+---------+-------------+
    | Batch size | NeMo ms/iter  | Standalone ms/iter| Speedup | Peak memory |
    +------------+---------------+-------------------+---------+-------------+
    | 1          | 1.138         | 1.067             | 1.07x   | 0.41 GiB    |
    | 4          | 4.072         | 3.869             | 1.05x   | 1.56 GiB    |
    | 8          | 7.840         | 7.501             | 1.05x   | 3.08 GiB    |
    | 16         | 15.453        | 14.780            | 1.05x   | 6.13 GiB    |
    +------------+---------------+-------------------+---------+-------------+

    The standalone speedup is smaller for 10-minute audio than for the 1-second batch-64 benchmark,
    where standalone was about 1.28x faster.
    """
    config = {
        "dither": 0,
        "pad_to": 0,
        "normalize": None,
        "features": 80,
    }
    sample_rate = 16000
    duration_seconds = int(os.environ.get("NEMO_LONG_LOGMEL_BENCHMARK_SECONDS", "600"))
    max_length = sample_rate * duration_seconds
    warmup = int(os.environ.get("NEMO_LONG_LOGMEL_BENCHMARK_WARMUP", "1"))
    iterations = int(os.environ.get("NEMO_LONG_LOGMEL_BENCHMARK_ITERATIONS", "3"))

    result = _run_cuda_benchmark_case(
        config,
        batch_size=batch_size,
        max_length=max_length,
        warmup=warmup,
        iterations=iterations,
    )

    with capsys.disabled():
        print(
            "\nLong LogMel CUDA benchmark "
            f"(device={torch.cuda.get_device_name(0)}, batch_size={batch_size}, "
            f"duration_seconds={duration_seconds}, samples={max_length}, iterations={iterations})"
        )
        print(
            f"NeMo AudioToMelSpectrogramPreprocessor: {result['nemo_ms']:.3f} ms/iter, "
            f"peak={result['nemo_peak_memory_gb']:.2f} GiB"
        )
        print(
            f"Standalone AudioToMelSpectrogramPreprocessor: {result['standalone_ms']:.3f} ms/iter, "
            f"peak={result['standalone_peak_memory_gb']:.2f} GiB"
        )
        print(f"Standalone speed vs NeMo: {result['speed_ratio']:.2f}x")


def _make_30s_inputs(sample_rate=16000):
    """Create test inputs for 30-second audio clips."""
    max_length = 30 * sample_rate  # 480,000 samples at 16kHz
    torch.manual_seed(2026)
    signal = torch.randn(32, max_length, dtype=torch.float32)
    time = torch.linspace(0, 30, max_length, dtype=torch.float32)
    # Add some structured signals
    signal[0] += 0.05 * torch.sin(2 * torch.pi * 220 * time)
    signal[1] += 0.03 * torch.sin(2 * torch.pi * 440 * time)
    signal[2] += torch.linspace(-0.1, 0.1, max_length)
    # Variable lengths simulating different clip durations
    lengths = torch.tensor([max_length - i * 10000 for i in range(32)])
    lengths = torch.clamp(lengths, min=max_length // 4)
    return signal, lengths


def _make_5min_inputs(sample_rate=16000):
    """Create test inputs for 5-minute audio clips."""
    max_length = 5 * 60 * sample_rate  # 4,800,000 samples at 16kHz
    torch.manual_seed(2026)
    signal = torch.randn(8, max_length, dtype=torch.float32)
    time = torch.linspace(0, 300, max_length, dtype=torch.float32)  # 300 seconds = 5 minutes
    # Add some structured signals
    signal[0] += 0.05 * torch.sin(2 * torch.pi * 220 * time)
    signal[1] += 0.03 * torch.sin(2 * torch.pi * 440 * time)
    signal[2] += torch.linspace(-0.1, 0.1, max_length)
    # Variable lengths simulating different recording durations
    lengths = torch.tensor([max_length - i * 50000 for i in range(8)])
    lengths = torch.clamp(lengths, min=max_length // 4)
    return signal, lengths


@pytest.mark.unit
def test_standalone_audio_to_mel_matches_nemo_30s_clips():
    """Test with realistic 30-second audio clips (32 samples)."""
    config = {
        "dither": 0,
        "pad_to": 0,
        "normalize": "per_feature",
        "features": 80,
        "lowfreq": 0,
        "highfreq": None,
    }

    signal, lengths = _make_30s_inputs()
    nemo_preprocessor = NemoAudioToMelSpectrogramPreprocessor(**config)
    standalone_preprocessor = StandaloneAudioToMelSpectrogramPreprocessor(**config)

    nemo_preprocessor.eval()
    standalone_preprocessor.eval()

    torch.manual_seed(12345)
    nemo_features, nemo_lengths = nemo_preprocessor(input_signal=signal.clone(), length=lengths.clone())
    torch.manual_seed(12345)
    standalone_features, standalone_lengths = standalone_preprocessor(
        input_signal=signal.clone(), length=lengths.clone()
    )

    assert standalone_lengths.equal(nemo_lengths)
    assert standalone_features.shape == nemo_features.shape
    assert standalone_features.dtype == nemo_features.dtype
    torch.testing.assert_close(standalone_features.float(), nemo_features.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_standalone_audio_to_mel_matches_nemo_5min_recordings():
    """Test with realistic 5-minute audio recordings (8 samples)."""
    config = {
        "dither": 0,
        "pad_to": 0,
        "normalize": "all_features",
        "features": 64,
        "lowfreq": 0,
        "highfreq": None,
    }

    signal, lengths = _make_5min_inputs()
    nemo_preprocessor = NemoAudioToMelSpectrogramPreprocessor(**config)
    standalone_preprocessor = StandaloneAudioToMelSpectrogramPreprocessor(**config)

    nemo_preprocessor.eval()
    standalone_preprocessor.eval()

    torch.manual_seed(12345)
    nemo_features, nemo_lengths = nemo_preprocessor(input_signal=signal.clone(), length=lengths.clone())
    torch.manual_seed(12345)
    standalone_features, standalone_lengths = standalone_preprocessor(
        input_signal=signal.clone(), length=lengths.clone()
    )

    assert standalone_lengths.equal(nemo_lengths)
    assert standalone_features.shape == nemo_features.shape
    assert standalone_features.dtype == nemo_features.dtype
    torch.testing.assert_close(standalone_features.float(), nemo_features.float(), atol=1e-4, rtol=1e-4)
