# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Check packed perception features against a fixed synthetic encoder fixture."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from nemo.collections.speechlm2.modules.perception import AudioPerceptionModule
from nemo.collections.speechlm2.parts.pretrained import setup_parallel_expert_encoder


def packed_encoder_features(fixture_dir: Path) -> dict[str, torch.Tensor]:
    """Load the synthetic encoder bundle with stochastic training layers disabled."""
    cfg = OmegaConf.create(
        {
            "pe_encoder_path": str(fixture_dir / "parallel-expert-encoder.nemo"),
            "pe_encoder_overrides": {
                "sync_max_audio_length": False,
                "speaker_feature_config_version": 1,
                "speaker_feature_mode": "thresholded",
                "speaker_activity_threshold": 0.5,
                "diar_normalize_type": "NA",
                "asr_normalize_type": "per_feature",
            },
            "spk_kernel_scale": 1.0,
            "packed_encoder_sequences": True,
            "encoder_chunk_size_seconds": 30.0,
            "perception": {
                "preprocessor": {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "sample_rate": 16000,
                    "normalize": "per_feature",
                    "window_size": 0.025,
                    "window_stride": 0.01,
                    "window": "hann",
                    "features": 128,
                    "n_fft": 512,
                    "log": True,
                    "frame_splicing": 1,
                    "dither": 0.0,
                    "pad_to": 0,
                },
                "encoder": {"_target_": "torch.nn.Identity"},
                "modality_adapter": {"_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector"},
            },
        }
    )
    model = SimpleNamespace(cfg=cfg, perception=AudioPerceptionModule(cfg.perception))
    setup_parallel_expert_encoder(model)
    perception = model.perception.to(device="cuda", dtype=torch.bfloat16).eval()

    waveforms = []
    for name in ("short.wav", "long.wav"):
        wave, sample_rate = sf.read(fixture_dir / name, dtype="float32")
        assert sample_rate == 16000 and wave.ndim == 1, name
        waveforms.append(torch.from_numpy(wave))
    lengths = torch.tensor([wave.numel() for wave in waveforms], device="cuda")
    cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    audio = torch.cat(waveforms).cuda()

    # The long clip crosses the 30-second post-stacking chunk boundary. Supply
    # all eight speaker slots at the ASR frame rate (80 ms).
    silent = torch.zeros((2, 377, 8), device="cuda")
    missing = torch.full_like(silent, -1.0)
    mixed = silent.clone()
    mixed[0] = -1.0
    mixed[1, 40:100, 7] = 1.0

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    outputs = {}
    with torch.inference_mode():
        for name, targets in (("missing", missing), ("silent", silent), ("mixed", mixed)):
            encoded = perception.forward_sequence_packed(
                input_signal=audio,
                input_signal_length=lengths,
                input_signal_cu_seqlens=cu_seqlens,
                spk_targets=targets,
            )
            outputs[f"{name}.features"] = encoded.data.float().cpu()
            outputs[f"{name}.lengths"] = encoded.lengths.cpu()
            outputs[f"{name}.cu_seqlens"] = encoded.cu_seqlens.cpu()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    args = parser.parse_args()
    expected = load_file(args.fixture_dir / "encoder-reference.safetensors")
    actual = packed_encoder_features(args.fixture_dir)
    for case in ("missing", "silent", "mixed"):
        lengths = expected[f"{case}.lengths"]
        for name in (f"{case}.lengths", f"{case}.cu_seqlens"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        actual_rows = actual[f"{case}.features"].split(lengths.tolist())
        expected_rows = expected[f"{case}.features"].split(lengths.tolist())
        for row, (features, reference) in enumerate(zip(actual_rows, expected_rows, strict=True)):
            assert features.shape == reference.shape, f"{case} utterance {row}"
            relative_rms_error = ((features - reference).norm() / reference.norm()).item()
            # Allow BF16 kernel roundoff across GPUs.
            assert relative_rms_error <= 0.02, f"{case} utterance {row}: relative RMS error {relative_rms_error:.2%}"
            print(f"{case} utterance {row}: {features.shape[0]} frames, relative RMS error {relative_rms_error:.2%}")


if __name__ == "__main__":
    main()
