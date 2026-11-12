# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a standalone TensorRT encoder plan directly from a NeMo checkpoint."""

from argparse import ArgumentParser
from pathlib import Path

import torch


class EncoderExport(torch.nn.Module):
    """Use the encoder's portable attention path without producing an ONNX graph."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_length = self.encoder.forward_for_export(audio_signal, length)
        # Without subsampling, lengths can alias the input; TensorRT needs a distinct output binding.
        return encoded, encoded_length.clone()


def capture_encoder(
    encoder: torch.nn.Module,
    min_frames: int = 16,
    opt_frames: int = 1000,
    max_frames: int = 4000,
    opt_batch: int = 16,
    max_batch: int = 16,
) -> torch.export.ExportedProgram:
    """Capture an offline encoder with shared dynamic batch and padded feature-frame axes.

    The encoder must provide ``forward_for_export`` and ``input_example``.
    Position buffers are provisioned before capture, without distributed synchronization.
    This function does not build an engine or require TensorRT.
    """
    if not 1 <= min_frames < opt_frames < max_frames:
        raise ValueError("Require 1 <= min_frames < opt_frames < max_frames.")
    if not 2 <= opt_batch <= max_batch:
        raise ValueError("Require 2 <= opt_batch <= max_batch for dynamic batch capture.")
    encoder.eval()
    encoder.sync_max_audio_length = False
    encoder.set_max_audio_length(max_frames)
    example = encoder.input_example(max_batch=opt_batch, max_dim=opt_frames)
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    frames = torch.export.Dim("frames", min=min_frames, max=max_frames)
    with torch.no_grad():
        return torch.export.export(EncoderExport(encoder), example, dynamic_shapes=({0: batch, 2: frames}, {0: batch}))


def build_encoder(
    program: torch.export.ExportedProgram,
    min_frames: int,
    opt_frames: int,
    max_frames: int,
    opt_batch: int,
    max_batch: int,
    features: int,
    workspace_bytes: int = 2 * 1024**3,
) -> bytes:
    """Convert the complete exported graph to an FP32 TensorRT plan, with TF32 disabled.

    Requires a compatible Torch-TensorRT installation and CUDA device.
    Unsupported operations raise an error; there is no PyTorch fallback in the saved plan.
    Binding names match NeMo's offline ONNX encoder export.
    """
    import torch_tensorrt

    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, features, min_frames),
            opt_shape=(opt_batch, features, opt_frames),
            max_shape=(max_batch, features, max_frames),
            dtype=torch.float32,
        ),
        torch_tensorrt.Input(min_shape=(1,), opt_shape=(opt_batch,), max_shape=(max_batch,), dtype=torch.int64),
    ]
    return torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        program,
        inputs=inputs,
        disable_tf32=True,
        workspace_size=workspace_bytes,
        immutable_weights=True,
        arg_input_binding_names=("audio_signal", "length"),
        output_binding_names=("outputs", "encoded_lengths"),
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--opt-frames", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.engine.exists():
        raise FileExistsError(args.engine)
    if args.workspace_gib <= 0:
        parser.error("workspace-gib must be positive.")
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.restore_from(str(args.model), map_location="cpu").float().eval().cuda()
    program = capture_encoder(
        model.encoder, args.min_frames, args.opt_frames, args.max_frames, args.opt_batch, args.max_batch
    )
    plan = build_encoder(
        program,
        args.min_frames,
        args.opt_frames,
        args.max_frames,
        args.opt_batch,
        args.max_batch,
        model.encoder.input_example()[0].shape[1],
        int(args.workspace_gib * 1024**3),
    )
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    with args.engine.open("xb") as stream:
        stream.write(plan)


if __name__ == "__main__":
    main()
