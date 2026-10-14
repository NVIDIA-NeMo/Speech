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

"""Persistent JSON-lines worker for process-isolated reward ASR backends."""

import argparse
import contextlib
import json
import re
import sys


QWEN_LANGUAGE_MAP = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
LANGUAGE_TAG_PATTERN = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Za-z]{2,4})?>\s*")


def emit(payload):
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def load_qwen(args):
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    def transcribe(audio_paths, languages, attention_context):
        del attention_context
        qwen_languages = [QWEN_LANGUAGE_MAP.get(language, language) for language in languages]
        return [result.text for result in model.transcribe(audio=audio_paths, language=qwen_languages)]

    return transcribe


def load_nemo(args, language_map):
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
        EncDecHybridRNNTCTCBPEModelWithPrompt,
        HybridRNNTCTCPromptTranscribeConfig,
    )
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
        EncDecRNNTBPEModelWithPrompt,
        RNNTPromptTranscribeConfig,
    )
    from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig

    if args.model.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(restore_path=args.model)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    model.to(args.device)
    model.freeze()
    prompted = isinstance(model, (EncDecHybridRNNTCTCBPEModelWithPrompt, EncDecRNNTBPEModelWithPrompt))
    configured_attention_context = None

    def transcribe(audio_paths, languages, attention_context):
        nonlocal configured_attention_context
        if attention_context is not None and attention_context != configured_attention_context:
            model.encoder.set_default_att_context_size(list(attention_context))
            configured_attention_context = list(attention_context)

        if not prompted:
            results = model.transcribe(
                audio_paths,
                batch_size=len(audio_paths),
                override_config=TranscribeConfig(use_lhotse=False, batch_size=len(audio_paths), num_workers=0),
            )
            return [result.text if hasattr(result, "text") else str(result) for result in results]

        transcripts = [""] * len(audio_paths)
        grouped = {}
        for index, (audio_path, language) in enumerate(zip(audio_paths, languages)):
            grouped.setdefault(language, []).append((index, audio_path))
        for language, items in grouped.items():
            target_lang = language_map[language]
            config_cls = (
                HybridRNNTCTCPromptTranscribeConfig
                if isinstance(model, EncDecHybridRNNTCTCBPEModelWithPrompt)
                else RNNTPromptTranscribeConfig
            )
            for chunk_start in range(0, len(items), args.batch_size):
                chunk = items[chunk_start : chunk_start + args.batch_size]
                config = config_cls(
                    use_lhotse=False,
                    batch_size=len(chunk),
                    return_hypotheses=False,
                    num_workers=0,
                    verbose=False,
                    target_lang=target_lang,
                )
                results = model.transcribe([path for _, path in chunk], batch_size=len(chunk), override_config=config)
                for (index, _), result in zip(chunk, results):
                    text = result.text if hasattr(result, "text") else str(result)
                    transcripts[index] = LANGUAGE_TAG_PATTERN.sub(" ", text).strip()
        return transcripts

    return transcribe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nemo", "qwen"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--language-map", type=json.loads, default={})
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        transcribe = load_qwen(args) if args.backend == "qwen" else load_nemo(args, args.language_map)
    emit({"status": "ready", "backend": args.backend, "model": args.model, "device": args.device})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                emit({"status": "stopped"})
                return
            if request.get("command") != "transcribe":
                raise ValueError(f"Unknown command: {request.get('command')!r}")
            audio_paths = request["audio_paths"]
            languages = request["languages"]
            if len(audio_paths) != len(languages):
                raise ValueError(
                    f"audio_paths and languages must have equal lengths, got {len(audio_paths)} and {len(languages)}"
                )
            with contextlib.redirect_stdout(sys.stderr):
                transcripts = transcribe(audio_paths, languages, request.get("attention_context"))
            emit({"status": "ok", "transcripts": transcripts})
        except Exception as error:
            emit({"status": "error", "error": repr(error)})


if __name__ == "__main__":
    main()
