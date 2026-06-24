#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Stage 1/2: export the EasyMagpieTTS (25 fps spectral) codec decoder to ONNX.

The exported graph takes the model's **raw stacked codes** and produces a
waveform, baking the two stateless model->codec glue steps into the graph so the
serving side (e.g. a Triton python backend) needs no NeMo:

    input   audio_codes  : int64  (batch, frames, num_stacked_codebooks)
    output  audio_values : float  (batch, frames * output_samples_per_frame)

    audio_codes -> clamp(specials) -> unstack -> index-convert -> codec.decode

``num_stacked_codebooks = num_audio_codebooks * frame_stacking_factor`` (e.g.
``8 * 2 = 16``). With ``--nemo_file`` the wrapper:

* **clamps** out-of-range special tokens (audio bos/eos/mask) to valid indices,
* **unstacks** ``(B, T, C*S) -> (B, C, T*S)`` (inverse of ``stack_codes``), and
* **index-converts** the model's regrouped FSQ space (e.g. 8 codebooks of 1024)
  to the codec's native ``GroupFiniteScalarQuantizer`` space (e.g. 5 codebooks of
  4^8) via ``VectorQuantizerIndexConverter.convert_new_to_original`` -- a lossless
  per-frame index remap, read straight from the EasyMagpie ``.nemo``.

Without ``--nemo_file`` it falls back to the codec's *native* decode (input
``(batch, frames, num_codebooks)``, no unstack / convert).

For ``25fps_spectral_codec_with_bandwidth_extension.nemo`` the codec emits 882
output samples / frame (decode emits 22050 Hz; encoder runs at 16000 Hz / 640
samples per frame); one model frame unstacks to ``frame_stacking_factor`` codec
frames.

The frame axis is exported as **static** (``--frames``) and only ``batch`` is
dynamic -- this matches the streaming decode usage (a fixed chunk size) and lets
TensorRT pick efficient tactics. Build several engines if you need several chunk
sizes, or pass a frames profile to the TRT builder for a dynamic frame axis.

Stage 2 (TRT engine build) lives in ``export_codec_decoder_trt.py``.

Example:
    python examples/tts/easymagpie_vllm_omni/scripts/export_codec_decoder_onnx.py \\
        --codec_model_path /path/to/25fps_spectral_codec_with_bandwidth_extension.nemo \\
        --nemo_file /path/to/easymagpie.nemo \\
        --onnx-path codec/codec_decoder.onnx \\
        --frames 15 --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Match ORT's full-FP32 matmul; PyTorch on Ampere+ uses TF32 by default and would
# otherwise diverge from the ONNX/ORT reference during the parity check.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    import onnx
except ImportError as exc:  # pragma: no cover
    raise ImportError("`onnx` is required. Install with: pip install onnx onnxruntime") from exc

from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging


class CodecDecoderWrapper(torch.nn.Module):
    """Wrap ``AudioCodecModel`` so a single ``(B, T, C)`` int tensor decodes to ``(B, T_audio)``.

    With ``converter``/``stacking`` set, the input is the model's *stacked* codes
    ``(B, T, C*S)`` and the wrapper clamps special tokens, unstacks to ``(B, C, T*S)``
    and index-converts to the codec's native space before decoding. Otherwise the
    input is the codec's *native* codes ``(B, T, num_codebooks)``.

    The codec's conv layers mask out-of-range positions using a per-batch length.
    We bake a *full-length* length tensor (all frames valid) so the mask folds to a
    constant at export time and disappears from the graph.
    """

    def __init__(
        self,
        codec_model: AudioCodecModel,
        converter: torch.nn.Module = None,
        stacking: int = 1,
        clamp_max: int = None,
    ):
        super().__init__()
        self.codec_model = codec_model
        self.converter = converter
        self.stacking = int(stacking)
        self.clamp_max = clamp_max

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        # audio_codes: (B, T, C) -> codec expects (B, C, T)
        tokens = audio_codes.transpose(1, 2).contiguous()
        bsz = tokens.shape[0]

        if self.stacking > 1:
            # Unstack (B, C*S, T) -> (B, C, T*S): inverse of EasyMagpie stack_codes.
            cs, t = tokens.shape[1], tokens.shape[2]
            c = cs // self.stacking
            tokens = tokens.view(bsz, c, self.stacking, t).permute(0, 1, 3, 2).reshape(bsz, c, t * self.stacking)

        if self.clamp_max is not None:
            # Drop special tokens (audio bos/eos/mask live above the codebook).
            tokens = tokens.clamp(0, self.clamp_max)

        tokens = tokens.contiguous()
        frames = tokens.shape[2]
        tokens_len = torch.full((bsz,), frames, dtype=torch.long, device=tokens.device)

        if self.converter is not None:
            tokens = self.converter.convert_new_to_original(audio_tokens=tokens, audio_lens=tokens_len)

        audio, _ = self.codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        return audio


def check_onnx_parity(wrapper, onnx_path, audio_codes, device, atol=1e-3):
    try:
        import onnxruntime as ort
    except ImportError:
        logging.warning("onnxruntime not installed -- skipping parity check")
        return True

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    with torch.inference_mode():
        ref = wrapper(audio_codes).detach().cpu().float().numpy()
    ort_out = sess.run(None, {"audio_codes": audio_codes.cpu().numpy()})[0]
    max_diff = float(np.abs(ref - ort_out).max())
    ok = max_diff <= atol
    logging.info(
        f"ONNX parity ({sess.get_providers()[0]}): max_abs_diff={max_diff:.6f} "
        f"atol={atol} {'PASSED' if ok else 'FAILED'}"
    )
    return ok


def load_codec_decoder(codec_model_path: str, device: torch.device) -> AudioCodecModel:
    """Restore the codec in FP32/eval and strip the (unused at inference) discriminator."""
    codec_cfg = AudioCodecModel.restore_from(codec_model_path, return_config=True)
    if "use_scl_loss" in codec_cfg:
        codec_cfg.use_scl_loss = False
    codec = AudioCodecModel.restore_from(codec_model_path, strict=False, override_config_path=codec_cfg)
    if hasattr(codec, "discriminator"):
        del codec.discriminator
    codec = codec.to(device).eval().float()
    codec.freeze()
    # Fuse weight-norm reparameterizations into plain conv weights for a clean graph.
    if hasattr(codec, "audio_decoder") and hasattr(codec.audio_decoder, "remove_weight_norm"):
        codec.audio_decoder.remove_weight_norm()
    return codec


def load_index_converter(codec: AudioCodecModel, nemo_file: str, device: torch.device):
    """Build the model->codec index converter + stacking factor from an EasyMagpie .nemo.

    Reads only the EasyMagpie config (no weights): the ``vector_quantizer`` override
    the model was trained with and its ``frame_stacking_factor``. Returns
    ``(converter_or_None, stacking, new_codebook_size)``. ``converter`` is None when
    the model and codec already share the same FSQ grouping.
    """
    from hydra.utils import instantiate

    from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
    from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter

    em_cfg = EasyMagpieTTSInferenceModel.restore_from(nemo_file, return_config=True)
    stacking = int(em_cfg.get("frame_stacking_factor", 1))
    vq_cfg = em_cfg.get("vector_quantizer")
    if vq_cfg is None:
        return None, stacking, None

    vq_new = instantiate(vq_cfg).to(device).eval()
    new_codebook_size = int(vq_new.codebook_size)
    if vq_new.num_codebooks == codec.vector_quantizer.num_codebooks:
        return None, stacking, new_codebook_size

    converter = VectorQuantizerIndexConverter(
        vector_quantizer_original=codec.vector_quantizer,
        vector_quantizer_new=vq_new,
    ).to(device).eval()
    return converter, stacking, new_codebook_size


def parse_args():
    p = argparse.ArgumentParser(description="Export the EasyMagpieTTS codec decoder to ONNX")
    p.add_argument("--codec_model_path", required=True, help="Path to the audio codec .nemo checkpoint")
    p.add_argument(
        "--nemo_file",
        default=None,
        help="EasyMagpie .nemo: bakes unstack + index conversion in (input becomes stacked model codes). "
        "Omit to export the codec's native decode.",
    )
    p.add_argument("--onnx-path", default="codec_decoder.onnx")
    p.add_argument("--frames", type=int, default=30, help="Static frame count baked into the graph (chunk size)")
    p.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used for export/parity")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--atol", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    codec = load_codec_decoder(args.codec_model_path, device)

    converter, stacking, new_codebook_size = (None, 1, None)
    if args.nemo_file is not None:
        converter, stacking, new_codebook_size = load_index_converter(codec, args.nemo_file, device)

    if args.nemo_file is not None:
        # Input is the model's stacked codes; clamp specials, unstack, convert.
        model_codebooks = (
            converter.vector_quantizer_new.num_codebooks if converter is not None else int(codec.num_codebooks)
        )
        codebook_size = new_codebook_size if new_codebook_size is not None else int(codec.codebook_size)
        nq = model_codebooks * stacking  # num_stacked_codebooks (e.g. 16)
        clamp_max = codebook_size - 1
    else:
        # Input is the codec's native codes.
        nq = int(codec.num_codebooks)
        codebook_size = int(codec.codebook_size)
        clamp_max = None

    wrapper = CodecDecoderWrapper(codec, converter=converter, stacking=stacking, clamp_max=clamp_max).to(device).eval()

    logging.info(
        f"codec: sample_rate={codec.sample_rate} output_sample_rate={codec.output_sample_rate} "
        f"samples_per_frame={codec.samples_per_frame} native_codebooks={int(codec.num_codebooks)} "
        f"| input num_codebooks={nq} stacking={stacking} convert={converter is not None}"
    )

    dummy = torch.randint(0, codebook_size, (args.batch_size, args.frames, nq), dtype=torch.long, device=device)

    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["audio_codes"],
            output_names=["audio_values"],
            dynamic_axes={
                "audio_codes": {0: "batch"},
                "audio_values": {0: "batch"},
            },
        )
    logging.info(f"ONNX exported to {onnx_path}")

    onnx.checker.check_model(str(onnx_path))

    if not check_onnx_parity(wrapper, onnx_path, dummy, device, atol=args.atol):
        raise RuntimeError("ONNX vs PyTorch parity failed -- export is broken.")


if __name__ == "__main__":
    main()
