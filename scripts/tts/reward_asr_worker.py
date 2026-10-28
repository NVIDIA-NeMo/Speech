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

"""Persistent JSON-lines worker for process-isolated reward ASR backends.

``ProcessRewardASRBackend`` in ``nemo/collections/tts/parts/utils/reward_asr.py`` starts one worker per
training rank and exchanges one JSON request and response per line over stdin/stdout.

Qwen3-ASR requires this separate interpreter: ``qwen-asr`` pins an exact ``transformers`` 4.x release,
which cannot be installed alongside the ``transformers`` 5.x used by NeMo. It therefore runs from its own
virtual environment (the backend's ``python_executable``). The ``nemo`` backend uses the same protocol to
keep NeMo ASR out of the TTS model's CUDA context.

NeMo is not installed in the Qwen environment, so this script must not import NeMo at module level.
Backend dependencies are imported inside ``load_qwen`` and ``load_nemo``.
"""

import argparse
import base64
import contextlib
import json
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


TranscribeFn = Callable[[Sequence[np.ndarray], int, Sequence[str], Optional[List[int]]], List[str]]


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


def load_qwen(args: argparse.Namespace) -> TranscribeFn:
    """Load Qwen ASR and return a waveform transcription callable.

    Args:
        args: Worker CLI settings containing model, device, batch size, and
            maximum generated token count.

    Returns:
        Callable accepting waveforms, sample rate in Hz, language codes, and an
        optional attention context, and returning ordered transcript strings.
    """
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    def transcribe(
        waveforms: Sequence[np.ndarray],
        sample_rate: int,
        languages: Sequence[str],
        attention_context: Optional[List[int]],
    ) -> List[str]:
        """Transcribe a waveform batch with Qwen's array/rate interface.

        Args:
            waveforms: True-length mono float32 NumPy arrays with shapes ``[T_i]``.
            sample_rate: Actual sample rate in Hz, shared by all waveforms.
            languages: One request language code per waveform.
            attention_context: Unused by Qwen; accepted for worker-interface compatibility.

        Returns:
            One transcript per waveform in the original input order.
        """
        del attention_context
        qwen_languages = [QWEN_LANGUAGE_MAP.get(language, language) for language in languages]
        return [
            result.text
            for result in model.transcribe(audio=[(wav, sample_rate) for wav in waveforms], language=qwen_languages)
        ]

    return transcribe


def load_nemo(args: argparse.Namespace, language_map: Dict[str, str]) -> TranscribeFn:
    """Load NeMo ASR and return a waveform transcription callable.

    Args:
        args: Worker CLI settings containing the model, device, and batch size.
        language_map: Mapping from request language codes to NeMo prompt locales.

    Returns:
        Callable accepting waveforms, sample rate in Hz, language codes, and an
        optional attention context, and returning ordered transcript strings.
    """
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

    def transcribe(
        waveforms: Sequence[np.ndarray],
        sample_rate: int,
        languages: Sequence[str],
        attention_context: Optional[List[int]],
    ) -> List[str]:
        """Resample and transcribe a waveform batch with NeMo ASR.

        Args:
            waveforms: True-length mono float32 NumPy arrays with shapes ``[T_i]``.
            sample_rate: Actual sample rate in Hz, shared by all waveforms.
            languages: One request language code per waveform, used by prompted models.
            attention_context: Optional encoder attention-context settings applied
                before transcription.

        Returns:
            One transcript per waveform in the original input order, with NeMo
            language tags removed for prompted models.
        """
        import torch
        from nemo.collections.audio.parts.utils.transforms import resample

        nonlocal configured_attention_context
        target_rate = int(model.cfg.sample_rate)
        audio = [torch.from_numpy(wav) for wav in waveforms]
        if sample_rate != target_rate:
            audio = [resample(wav, sample_rate, target_rate) for wav in audio]
        if attention_context is not None and attention_context != configured_attention_context:
            model.encoder.set_default_att_context_size(list(attention_context))
            configured_attention_context = list(attention_context)

        if not prompted:
            results = model.transcribe(
                audio,
                batch_size=len(audio),
                override_config=TranscribeConfig(use_lhotse=False, batch_size=len(audio), num_workers=0),
            )
            return [result.text if hasattr(result, "text") else str(result) for result in results]

        transcripts = [""] * len(audio)
        grouped = {}
        for index, (waveform, language) in enumerate(zip(audio, languages)):
            grouped.setdefault(language, []).append((index, waveform))
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
            sample_rate = int(request['sample_rate'])
            if sample_rate <= 0:
                raise ValueError('sample_rate must be positive.')
            audio = [
                np.frombuffer(base64.b64decode(value, validate=True), dtype='<f4').copy()
                for value in request['waveforms']
            ]
            if any(wav.size == 0 for wav in audio):
                raise ValueError('Expected nonempty waveforms.')
            languages = request["languages"]
            if len(audio) != len(languages):
                raise ValueError(f"audio and languages must have equal lengths, got {len(audio)} and {len(languages)}")
            with contextlib.redirect_stdout(sys.stderr):
                transcripts = transcribe(audio, sample_rate, languages, request.get("attention_context"))
            emit({"status": "ok", "transcripts": transcripts})
        except Exception as error:
            emit({"status": "error", "error": repr(error)})
        finally:
            # Do not retain the last batch while waiting for the next request.
            request = audio = line = None


if __name__ == "__main__":
    main()
