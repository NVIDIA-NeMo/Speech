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
Minimal pure-PyTorch single-utterance inference for EasyMagpieTTS.

No vLLM, no manifest, no evalset config. Just: one context wav + one text -> one wav.

Example:
    python examples/tts/easy_magpietts_single_infer.py \\
        --nemo_file /path/to/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay.nemo \\
        --codec_model_path /path/to/25fps_spectral_codec_with_bandwidth_extension.nemo \\
        --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json \\
        --context_audio /path/to/reference_voice.wav \\
        --text "Hello, this is a test of the EasyMagpie text to speech model." \\
        --out_wav ./out.wav
"""
from __future__ import annotations

import argparse

import soundfile as sf
import torch

from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_easy_magpie_model
from nemo.utils import logging


def main():
    parser = argparse.ArgumentParser(description="EasyMagpieTTS single-utterance pure-torch inference")
    parser.add_argument("--nemo_file", required=True, help="Path to the EasyMagpieTTS .nemo checkpoint")
    parser.add_argument("--codec_model_path", required=True, help="Path to the audio codec .nemo checkpoint")
    parser.add_argument(
        "--phoneme_tokenizer_path",
        default=None,
        help="Override the phoneme (IPA BPE) tokenizer path baked into the checkpoint. "
        "Required if the path stored in the .nemo does not exist locally.",
    )
    parser.add_argument("--context_audio", default=None, help="Reference/context wav for voice cloning")
    parser.add_argument(
        "--context_text",
        default=None,
        help="Optional style/context text tag. The voice is cloned from --context_audio; this is a "
        "separate style/language conditioning string. If omitted, the correct in-distribution "
        '"no text context" placeholder is auto-selected to match how the checkpoint was trained '
        "(language tag like [EN] if add_language_to_context_text=True, else [NO TEXT CONTEXT]). "
        "Do NOT pass a free-form sentence unless you want it spoken/styled.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language of --text; used to build the [LANG] context-text placeholder for checkpoints "
        "trained with add_language_to_context_text=True (e.g. en, de, es, fr, it, hi, zh, vi, ko-KR, pt-BR, ar)",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out_wav", default="./out.wav", help="Output wav path")

    # Tokenizer selection: defaults to the first text tokenizer in the checkpoint config
    # (e.g. nemotron_nano_30b). Override only if your checkpoint has multiple.
    parser.add_argument("--main_tokenizer_name", default=None)

    # The legacy Qwen EasyMagpie checkpoint was trained without CAS embeddings on context text.
    parser.add_argument(
        "--disable_cas_for_context_text",
        action="store_true",
        help="Set for legacy checkpoints trained without CAS embeddings on context text",
    )

    # Sampling / decoding parameters (defaults mirror the InferEvaluate functional test).
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--use_cfg", action="store_true", default=True)
    parser.add_argument("--no_cfg", dest="use_cfg", action="store_false")
    parser.add_argument("--cfg_scale", type=float, default=2.5)
    parser.add_argument("--no_local_transformer", dest="use_local_transformer", action="store_false", default=True)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--context_audio_duration", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")

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
    logging.info(f"Available text tokenizers: {list(model.tokenizer.tokenizers.keys())}")

    # Resolve the context-text placeholder to match the training-time convention.
    # The dataset uses "[<LANG>]" when add_language_to_context_text=True, else "[NO TEXT CONTEXT]".
    # Passing the wrong placeholder is out-of-distribution and the model may literally speak it
    # (e.g. starting the audio with the word "context").
    context_text = args.context_text
    if context_text is None:
        if getattr(model, "add_language_to_context_text", False):
            context_text = f"[{args.language.upper()}]"
        else:
            context_text = "[NO TEXT CONTEXT]"
    logging.info(f"Using context_text={context_text!r}")

    with torch.inference_mode():
        audio, audio_lens = model.do_tts(
            transcript=args.text,
            context_audio_file_path=args.context_audio,
            context_text=context_text,
            main_tokenizer_name=args.main_tokenizer_name,
            context_audio_duration=args.context_audio_duration,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            use_local_transformer=args.use_local_transformer,
            temperature=args.temperature,
            topk=args.topk,
            max_steps=args.max_steps,
        )

    audio_len = int(audio_lens[0].item())
    audio_np = audio[0, :audio_len].float().detach().cpu().numpy()
    sf.write(args.out_wav, audio_np, model.output_sample_rate)
    logging.info(
        f"Wrote {audio_len / model.output_sample_rate:.2f}s of audio "
        f"({model.output_sample_rate} Hz) to {args.out_wav}"
    )


if __name__ == "__main__":
    main()
