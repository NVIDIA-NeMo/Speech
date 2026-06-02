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
"""
Standalone speaker-encoder output extractor for EasyMagpieTTS.

Pre-computes ONLY the speaker-encoded context-audio embedding so it can be fed
to a separate (e.g. vLLM) backbone implementation. Context-text / task
embeddings are intentionally NOT included here -- the caller is expected to
prepend/append those (e.g. inside the vLLM model's ``preprocess``).

This reproduces the audio branch of
``EasyMagpieTTSInferenceModel.prepare_context_tensors``::

    audio -> codec codes -> (codec convert) -> add BOS/EOS -> frame stacking
          -> per-codebook embedding -> speaker encoder

and saves the resulting ``(T_audio, embedding_dim)`` tensor to disk.

Example:
    python examples/tts/easy_magpietts_extract_speaker_encoding.py \\
        --nemo_file /path/to/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay.nemo \\
        --codec_model_path /path/to/25fps_spectral_codec_with_bandwidth_extension.nemo \\
        --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json \\
        --context_audio /path/to/reference_voice.wav \\
        --out_file ./speaker_encoding.pt
"""
from __future__ import annotations

import argparse

import torch

from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_easy_magpie_model
from nemo.collections.tts.modules.magpietts_modules import add_special_tokens
from nemo.utils import logging


def main():
    parser = argparse.ArgumentParser(description="Extract EasyMagpieTTS speaker-encoder output")
    parser.add_argument("--nemo_file", required=True, help="Path to the EasyMagpieTTS .nemo checkpoint")
    parser.add_argument("--codec_model_path", required=True, help="Path to the audio codec .nemo checkpoint")
    parser.add_argument(
        "--phoneme_tokenizer_path",
        default=None,
        help="Override the phoneme (IPA BPE) tokenizer path baked into the checkpoint. "
        "Required if the path stored in the .nemo does not exist locally.",
    )
    parser.add_argument("--context_audio", required=True, help="Reference/context wav for voice cloning")
    parser.add_argument(
        "--disable_cas_for_context_text",
        action="store_true",
        help="Set for legacy checkpoints trained without CAS embeddings on context text",
    )
    parser.add_argument("--context_audio_duration", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out_file",
        default="./speaker_encoding.pt",
        help="Output path. A torch .pt file (dict) is written; if it ends with .npy the "
        "speaker-encoding tensor is saved as a NumPy array instead.",
    )

    args = parser.parse_args()

    model, ckpt_name = load_easy_magpie_model(
        ModelLoadConfig(
            nemo_file=args.nemo_file,
            codecmodel_path=args.codec_model_path,
            phoneme_tokenizer_path=args.phoneme_tokenizer_path,
            disable_cas_for_context_text=args.disable_cas_for_context_text,
        ),
        device=args.device,
    )
    logging.info(f"Loaded EasyMagpieTTS checkpoint: {ckpt_name}")
    logging.info(f"use_speaker_encoder={getattr(model, 'use_speaker_encoder', False)}")

    device = next(model.parameters()).device

    with torch.inference_mode():
        # Load + trim context audio exactly like EasyMagpieTTSInferenceModel.do_tts.
        context_audio = model._load_audio_for_inference(args.context_audio, model.sample_rate)
        context_audio = model._adjust_audio_to_duration_for_inference(
            context_audio,
            model.sample_rate,
            args.context_audio_duration,
            model.codec_model_samples_per_frame,
        )
        context_audio = context_audio.to(device)
        context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long, device=device)
        context_audio_codes, context_audio_codes_lens = model._codec_helper.audio_to_codes(
            context_audio, context_audio_lens
        )

        # --- Audio branch of prepare_context_tensors (no context text / task embedding) ---
        if model._codec_converter is not None:
            context_audio_codes = model._codec_converter.convert_original_to_new(
                audio_tokens=context_audio_codes, audio_lens=context_audio_codes_lens
            ).long()

        context_audio_codes, context_audio_codes_lens = add_special_tokens(
            codes=context_audio_codes,
            codes_len=context_audio_codes_lens,
            bos_id=model.context_audio_bos_id,
            eos_id=model.context_audio_eos_id,
        )

        context_audio_codes, context_audio_codes_lens = model.stack_codes(
            context_audio_codes,
            context_audio_codes_lens,
            model.context_audio_bos_id,
            model.context_audio_eos_id,
            model.frame_stacking_factor,
            model.num_audio_codebooks,
        )

        context_audio_embedded = model.embed_audio_tokens(context_audio_codes)  # (B, T_audio, E)

        if getattr(model, "use_speaker_encoder", False):
            context_audio_embedded = model.encode_context_audio_embeddings(
                context_audio_embedded=context_audio_embedded,
                context_audio_lens=context_audio_codes_lens,
            )
        else:
            logging.warning(
                "Checkpoint has use_speaker_encoder=False; saving raw per-codebook audio embeddings "
                "(no speaker encoder applied)."
            )

    # Strip batch dim (B == 1) -> (T_audio, embedding_dim).
    audio_len = int(context_audio_codes_lens[0].item())
    speaker_encoding = context_audio_embedded[0, :audio_len].contiguous().float().detach().cpu()
    logging.info(f"Extracted speaker-encoder output: {tuple(speaker_encoding.shape)}")

    if args.out_file.endswith(".npy"):
        import numpy as np

        np.save(args.out_file, speaker_encoding.numpy())
    else:
        torch.save(
            {
                "speaker_encoding": speaker_encoding,
                "context_audio": args.context_audio,
                "embedding_dim": int(speaker_encoding.size(-1)),
                "num_frames": int(speaker_encoding.size(0)),
                "checkpoint": ckpt_name,
            },
            args.out_file,
        )
    logging.info(f"Wrote speaker encoding of shape {tuple(speaker_encoding.shape)} to {args.out_file}")


if __name__ == "__main__":
    main()
