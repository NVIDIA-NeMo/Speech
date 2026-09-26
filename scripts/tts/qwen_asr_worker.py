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

"""Persistent JSON-lines Qwen3-ASR worker for reward computation."""

import argparse
import contextlib
import json
import sys

import torch
from qwen_asr import Qwen3ASRModel


LANGUAGE_MAP = {
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


def emit(payload):
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        model = Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map=args.device,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    emit({"status": "ready", "model": args.model, "device": args.device})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                emit({"status": "stopped"})
                return
            if request.get("command") != "transcribe":
                raise ValueError(f"Unknown command: {request.get('command')!r}")

            audio_paths = request["audio_paths"]
            languages = [LANGUAGE_MAP.get(language, language) for language in request["languages"]]
            if len(audio_paths) != len(languages):
                raise ValueError(
                    f"audio_paths and languages must have equal lengths, got {len(audio_paths)} and {len(languages)}"
                )
            with contextlib.redirect_stdout(sys.stderr):
                results = model.transcribe(audio=audio_paths, language=languages)
            emit(
                {
                    "status": "ok",
                    "transcripts": [result.text for result in results],
                    "detected_languages": [result.language for result in results],
                }
            )
        except Exception as error:
            emit({"status": "error", "error": repr(error)})


if __name__ == "__main__":
    main()
