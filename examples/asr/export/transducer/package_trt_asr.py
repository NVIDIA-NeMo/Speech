# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Package existing TensorRT plans as an offline waveform-to-text Triton model."""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


def package(
    model: ASRModel,
    output: Path,
    preprocessor: Path,
    encoder: Path,
    decoder: Path,
    max_batch: int = 16,
    max_samples: int = 639999,
    queue_delay_us: int = 1000,
) -> None:
    """Check engine profiles on the current GPU and package plans from the same checkpoint.

    Profile checks cannot establish weight identity; validate recognition parity before deployment.
    """
    if not isinstance(model.tokenizer, SentencePieceTokenizer) or model.tokenizer.legacy:
        raise ValueError("The waveform serving example requires a non-legacy SentencePiece tokenizer.")
    if max_batch < 1 or max_samples < 1 or queue_delay_us < 0:
        raise ValueError("Invalid batching or waveform profile.")
    feat = model.preprocessor.featurizer
    if feat.exact_pad or feat.pad_to != 0:
        raise ValueError("The waveform serving example currently requires center padding and pad_to=0.")
    decoder_config = json.loads((decoder / "config.json").read_text())
    if max_batch > decoder_config["max_batch"]:
        raise ValueError("Serving batch exceeds the decoder profile.")
    min_samples, min_frames = _profile_limits(preprocessor, encoder, decoder_config, max_batch, max_samples)
    version = output / "1"
    version.mkdir(parents=True, exist_ok=False)
    shutil.copy2(preprocessor, version / "preprocessor.plan")
    shutil.copy2(encoder, version / "encoder.plan")
    shutil.copytree(decoder, version / "decoder")
    (version / "tokenizer.model").write_bytes(model.tokenizer.tokenizer.serialized_model_proto())
    config = {
        "format_version": 1,
        "sample_rate": model.cfg.preprocessor.sample_rate,
        "max_batch": max_batch,
        "max_samples": max_samples,
        "min_samples": min_samples,
        "min_feature_frames": min_frames,
        "min_valid_samples": 2 * feat.hop_length if feat.normalize == "per_feature" else 1,
        "punctuation_pattern": (
            model.decoding.space_before_punct_pattern.pattern if model.decoding.supported_punctuation else None
        ),
        "strip_lang_tags": model.decoding.strip_lang_tags,
        "language_pattern": model.decoding.lang_tag_pattern.pattern if model.decoding.strip_lang_tags else None,
    }
    (version / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(Path(__file__).with_name("waveform_triton_model.py"), version / "model.py")
    (output / "config.pbtxt").write_text(
        'backend: "python"\n'
        f'max_batch_size: {max_batch}\n'
        'input [\n'
        '  { name: "AUDIO" data_type: TYPE_FP32 dims: [-1] },\n'
        '  { name: "AUDIO_LENGTH" data_type: TYPE_INT64 dims: [1] }\n'
        ']\noutput [\n'
        '  { name: "TEXT" data_type: TYPE_STRING dims: [1] },\n'
        '  { name: "TOKENS" data_type: TYPE_INT64 dims: [-1] },\n'
        '  { name: "TOKEN_LENGTH" data_type: TYPE_INT64 dims: [1] },\n'
        '  { name: "TIMESTAMPS" data_type: TYPE_INT64 dims: [-1] },\n'
        '  { name: "TOKEN_DURATIONS" data_type: TYPE_INT64 dims: [-1] }\n'
        ']\ninstance_group [{ kind: KIND_GPU count: 1 }]\n'
        f'dynamic_batching {{ max_queue_delay_microseconds: {queue_delay_us} }}\n'
        'parameters { key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: { string_value: "no" } }\n'
        'parameters { key: "use_cuda_graphs" value: { string_value: "true" } }\n'
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=639999)
    parser.add_argument("--queue-delay-us", type=int, default=1000)
    args = parser.parse_args()
    model = ASRModel.restore_from(str(args.model), map_location="cpu").float().eval()
    package(
        model,
        args.output,
        args.preprocessor,
        args.encoder,
        args.decoder,
        args.max_batch,
        args.max_samples,
        args.queue_delay_us,
    )


def _profile_limits(
    preprocessor: Path, encoder: Path, decoder_config: dict, max_batch: int, max_samples: int
) -> tuple[int, int]:
    from nemo.collections.asr.parts.submodules.tensorrt_plan import TensorRTPlan

    frontend = TensorRTPlan(preprocessor)
    encoder_plan = TensorRTPlan(encoder)
    if set(frontend.inputs) != {"input_signal", "length"} or set(frontend.outputs) != {
        "processed_signal",
        "processed_length",
    }:
        raise ValueError("Expected a waveform preprocessing plan.")
    if set(encoder_plan.inputs) != {"audio_signal", "length"} or set(encoder_plan.outputs) != {
        "outputs",
        "encoded_lengths",
    }:
        raise ValueError("Expected an offline encoder plan.")
    import torch

    if frontend.dtypes["input_signal"] != torch.float32 or frontend.dtypes["length"] != torch.int64:
        raise ValueError("Preprocessing requires FP32 waveform and INT64 length inputs.")
    low, _, high = frontend.engine.get_tensor_profile_shape("input_signal", 0)
    if len(low) != 2:
        raise ValueError("Expected a rank-two waveform profile.")
    min_samples = low[1]
    if min_samples < 1 or not min_samples <= max_samples <= high[1]:
        raise ValueError("Requested waveform range exceeds the preprocessing profile.")
    low, _, _ = encoder_plan.engine.get_tensor_profile_shape("audio_signal", 0)
    if len(low) != 3 or low[2] < 1:
        raise ValueError("Expected a rank-three encoder feature profile.")
    min_frames = low[2]
    # Log-mel frame counts and offline encoder output sizes grow monotonically.
    # Check both ends of the advertised range before writing a usable bundle.
    for batch in {1, max_batch}:
        for samples in {min_samples, max_samples}:
            features = _output_shapes(frontend, {"input_signal": (batch, samples), "length": (batch,)})
            shape = features["processed_signal"]
            if len(shape) != 3 or shape[0] != batch or features["processed_length"] != (batch,):
                raise ValueError("Unexpected preprocessing output shapes.")
            shape = (batch, shape[1], max(shape[2], min_frames))
            encoded = _output_shapes(encoder_plan, {"audio_signal": shape, "length": (batch,)})
            shape = encoded["outputs"]
            if (
                len(shape) != 3
                or shape[0] != batch
                or shape[1] != decoder_config["encoder_hidden"]
                or not 1 <= shape[2] <= decoder_config["max_frames"]
                or encoded["encoded_lengths"] != (batch,)
            ):
                raise ValueError("Encoder outputs exceed the decoder's channel or frame profile.")
    return min_samples, min_frames


def _output_shapes(plan, inputs: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    context = plan.engine.create_execution_context()
    if context is None:
        raise RuntimeError("Cannot create a TensorRT context for profile validation.")
    for name, shape in inputs.items():
        low, _, high = plan.engine.get_tensor_profile_shape(name, 0)
        if len(shape) != len(low) or any(n < lo or n > hi for n, lo, hi in zip(shape, low, high)):
            raise ValueError(f"{name} shape {shape} exceeds its engine profile.")
        if not context.set_input_shape(name, shape):
            raise ValueError(f"TensorRT rejected {name} shape {shape}.")
    if context.infer_shapes():
        raise ValueError("The engine must infer output shapes without executing.")
    return {name: tuple(context.get_tensor_shape(name)) for name in plan.outputs}


if __name__ == "__main__":
    main()
