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

"""Configurable ASR backends used to score generated TTS audio."""

import abc
import json
import os
import re
import selectors
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

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
from nemo.collections.tts.parts.utils.helpers import transcribe_with_whisper_from_filepaths
from nemo.utils import logging


DEFAULT_NEMOTRON_LANGUAGE_MAP = {
    "ar": "ar-AR",
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "hi": "hi-IN",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "pt": "pt-BR",
    "vi": "vi-VN",
    "zh": "zh-CN",
}
NEMOTRON_LANGUAGE_TAG_PATTERN = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Za-z]{2,4})?>\s*")


class RewardASRBackend(abc.ABC):
    """Minimal interface for reward transcription implementations."""

    @abc.abstractmethod
    def transcribe(self, audio_paths: Sequence[str], languages: Sequence[str]) -> List[str]:
        """Return one transcript for each audio path."""

    def close(self) -> None:
        """Release backend resources."""


class WhisperRewardASRBackend(RewardASRBackend):
    def __init__(self, cfg: Mapping, device_getter: Callable[[], torch.device]):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_name = cfg.get("model_name", "openai/whisper-large-v3")
        self.device_getter = device_getter
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def transcribe(self, audio_paths: Sequence[str], languages: Sequence[str]) -> List[str]:
        device = self.device_getter()
        self.model.to(device)
        return transcribe_with_whisper_from_filepaths(
            audio_filepaths=audio_paths,
            language=languages,
            whisper_processor=self.processor,
            whisper_model=self.model,
            device=device,
            normalizer=None,
        )


class NemoRewardASRBackend(RewardASRBackend):
    """NeMo ASR backend with optional multilingual prompt routing."""

    def __init__(self, cfg: Mapping, device_getter: Callable[[], torch.device]):
        self.device_getter = device_getter
        model_name = cfg.get("model_name", "nvidia/parakeet-ctc-0.6b")
        if str(model_name).endswith(".nemo"):
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_name)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.model.freeze()

        self.language_map = dict(cfg.get("language_map", DEFAULT_NEMOTRON_LANGUAGE_MAP))
        self.prompted = isinstance(
            self.model, (EncDecHybridRNNTCTCBPEModelWithPrompt, EncDecRNNTBPEModelWithPrompt)
        )
        if self.prompted:
            prompt_dictionary = self.model.cfg.model_defaults.get("prompt_dictionary", {})
            missing = sorted(set(self.language_map.values()) - set(prompt_dictionary))
            if missing:
                raise ValueError(f"NeMo reward ASR does not support configured locales: {missing}")

            attention_context = cfg.get("attention_context")
            if attention_context is not None and hasattr(self.model.encoder, "att_context_size_all"):
                attention_context = list(attention_context)
                available = [list(context) for context in self.model.encoder.att_context_size_all]
                if attention_context not in available:
                    raise ValueError(
                        f"Unsupported ASR attention context {attention_context}; available contexts: {available}"
                    )
                self.model.encoder.set_default_att_context_size(attention_context)

    def transcribe(self, audio_paths: Sequence[str], languages: Sequence[str]) -> List[str]:
        self.model.to(self.device_getter())
        if not self.prompted:
            results = self.model.transcribe(
                list(audio_paths),
                batch_size=len(audio_paths),
                override_config=TranscribeConfig(
                    use_lhotse=False, batch_size=len(audio_paths), num_workers=0
                ),
            )
            return [result.text if hasattr(result, "text") else str(result) for result in results]

        transcripts = [""] * len(audio_paths)
        grouped: Dict[str, List[tuple[int, str]]] = {}
        for index, (audio_path, language) in enumerate(zip(audio_paths, languages)):
            grouped.setdefault(language, []).append((index, audio_path))

        for language, items in grouped.items():
            if language not in self.language_map:
                raise ValueError(f"No locale configured for reward ASR language {language!r}")
            target_lang = self.language_map[language]
            config_cls = (
                HybridRNNTCTCPromptTranscribeConfig
                if isinstance(self.model, EncDecHybridRNNTCTCBPEModelWithPrompt)
                else RNNTPromptTranscribeConfig
            )
            config = config_cls(
                use_lhotse=False,
                batch_size=len(items),
                return_hypotheses=False,
                num_workers=0,
                verbose=False,
                target_lang=target_lang,
            )
            results = self.model.transcribe(
                [path for _, path in items], batch_size=len(items), override_config=config
            )
            for (index, _), result in zip(items, results):
                text = result.text if hasattr(result, "text") else str(result)
                transcripts[index] = NEMOTRON_LANGUAGE_TAG_PATTERN.sub(" ", text).strip()
        return transcripts


class QwenRewardASRBackend(RewardASRBackend):
    """Persistent Qwen worker isolated in its supported Python environment."""

    def __init__(self, cfg: Mapping, device_getter: Callable[[], torch.device]):
        self.device_getter = device_getter
        self.python_executable = str(cfg.get("python_executable", "/opt/qwen_asr/bin/python"))
        default_worker = Path(__file__).resolve().parents[5] / "scripts" / "tts" / "qwen_asr_worker.py"
        self.worker_script = str(cfg.get("worker_script", default_worker))
        self.model_name = str(cfg.get("model_name", "Qwen/Qwen3-ASR-0.6B"))
        self.batch_size = max(int(cfg.get("batch_size", 4)), 1)
        self.max_new_tokens = max(int(cfg.get("max_new_tokens", 256)), 1)
        self.timeout_seconds = max(float(cfg.get("timeout_seconds", 300.0)), 1.0)
        self.process: Optional[subprocess.Popen] = None
        self.stderr_handle = None

        if not os.path.isfile(self.python_executable):
            raise RuntimeError(
                "The Qwen reward backend requires the Qwen-enabled training container "
                f"with {self.python_executable}. Use that container or select a NeMo/Whisper ASR backend."
            )
        if not os.path.isfile(self.worker_script):
            raise FileNotFoundError(f"Qwen ASR worker not found: {self.worker_script}")

    def _read_response(self) -> Dict:
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(self.timeout_seconds):
                raise TimeoutError(f"Qwen ASR worker timed out after {self.timeout_seconds:.1f} seconds")
            line = self.process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise RuntimeError("Qwen ASR worker exited without returning a response")
        return json.loads(line)

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        device = self.device_getter()
        device_index = device.index if device.index is not None else 0
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(device_index)
        stderr_path = os.path.join(
            tempfile.gettempdir(), f"qwen_asr_worker_{os.getpid()}_{device_index}.log"
        )
        self.stderr_handle = open(stderr_path, "a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                self.python_executable,
                self.worker_script,
                "--model",
                self.model_name,
                "--device",
                "cuda:0",
                "--batch-size",
                str(self.batch_size),
                "--max-new-tokens",
                str(self.max_new_tokens),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        ready = self._read_response()
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected Qwen ASR worker startup response: {ready}")
        logging.info(
            f"Qwen ASR worker ready on {ready.get('device')} with model {ready.get('model')} "
            f"(stderr: {stderr_path})"
        )

    def _request(self, audio_paths: Sequence[str], languages: Sequence[str]) -> List[str]:
        self._start()
        request = {
            "command": "transcribe",
            "audio_paths": list(audio_paths),
            "languages": list(languages),
        }
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok":
            raise RuntimeError(f"Qwen ASR worker failed: {response}")
        transcripts = response.get("transcripts", [])
        if len(transcripts) != len(audio_paths):
            raise RuntimeError(f"Qwen ASR returned {len(transcripts)} transcripts for {len(audio_paths)} inputs")
        return transcripts

    def transcribe(self, audio_paths: Sequence[str], languages: Sequence[str]) -> List[str]:
        try:
            return self._request(audio_paths, languages)
        except (BrokenPipeError, TimeoutError, RuntimeError):
            # A single restart handles transient worker failures without hiding a
            # persistent configuration or model error.
            self.close()
            return self._request(audio_paths, languages)

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if self.stderr_handle is not None:
            self.stderr_handle.close()
            self.stderr_handle = None


BACKEND_TYPES = {
    "nemo": NemoRewardASRBackend,
    "qwen": QwenRewardASRBackend,
    "whisper": WhisperRewardASRBackend,
}


class RewardASRRouter:
    """Route each language to a configured ASR backend while preserving order.

    Example configuration::

        reward_asr:
          default_backend: qwen
          language_routes: {hi: whisper}
          backends:
            qwen: {type: qwen, model_name: Qwen/Qwen3-ASR-0.6B}
            whisper: {type: whisper, model_name: openai/whisper-large-v3}
    """

    def __init__(
        self,
        cfg: Mapping,
        device_getter: Callable[[], torch.device],
        backend_types: Optional[Mapping[str, type[RewardASRBackend]]] = None,
    ):
        self.default_backend = str(cfg.get("default_backend", "nemo"))
        self.language_routes = dict(cfg.get("language_routes", {}))
        backend_cfgs = cfg.get("backends", {})
        backend_types = dict(backend_types or BACKEND_TYPES)

        required_names = {self.default_backend, *self.language_routes.values()}
        self.backends: Dict[str, RewardASRBackend] = {}
        for name in required_names:
            if name not in backend_cfgs:
                raise ValueError(f"Missing configuration for reward ASR backend {name!r}")
            backend_cfg = backend_cfgs[name]
            backend_type = str(backend_cfg.get("type", name))
            if backend_type not in backend_types:
                raise ValueError(
                    f"Unknown reward ASR backend type {backend_type!r}; available: {sorted(backend_types)}"
                )
            self.backends[name] = backend_types[backend_type](backend_cfg, device_getter)

    def transcribe(self, audio_paths: Sequence[str], languages: Sequence[str]) -> List[str]:
        if len(audio_paths) != len(languages):
            raise ValueError(
                f"audio_paths and languages must have equal lengths, got {len(audio_paths)} and {len(languages)}"
            )
        transcripts = [""] * len(audio_paths)
        grouped: Dict[str, List[tuple[int, str, str]]] = {}
        for index, (audio_path, language) in enumerate(zip(audio_paths, languages)):
            backend_name = self.language_routes.get(language, self.default_backend)
            grouped.setdefault(backend_name, []).append((index, audio_path, language))

        for backend_name, items in grouped.items():
            backend = self.backends[backend_name]
            results = backend.transcribe(
                [path for _, path, _ in items], [language for _, _, language in items]
            )
            for (index, _, _), transcript in zip(items, results):
                transcripts[index] = transcript
        return transcripts

    def close(self) -> None:
        for backend in self.backends.values():
            backend.close()
