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

"""Validation inference and quality evaluation for EasyMagpieTTS CFG distillation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.tts.parts.utils.helpers import (
    get_speaker_embeddings_from_filepaths,
    process_text_for_cer,
    transcribe_with_whisper,
    transcribe_with_whisper_from_filepaths,
)
from nemo.utils import logging

try:
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    _HAVE_UTMOSV2 = True
except (ImportError, ModuleNotFoundError):
    _HAVE_UTMOSV2 = False


@dataclass
class ValidationAudioPaths:
    """Generated and context audio file paths for validation."""

    generated: list[Path] = field(default_factory=list)
    context: list[Path] = field(default_factory=list)


@dataclass
class ValidationSpeakerEmbeddings:
    """Generated and context speaker embeddings for validation."""

    generated: Optional[np.ndarray]
    context: Optional[np.ndarray]


@dataclass
class ValidationInferenceParams:
    """Configuration for validation-time inference and quality evaluation."""

    run_val_inference: bool
    use_multilingual_asr: bool
    use_utmos: bool
    whisper_model_name: str
    nemo_asr_model_name: str
    ssim_model_name: str

    @classmethod
    def create_from_config(
        cls,
        cfg: DictConfig,
    ) -> "ValidationInferenceParams":
        """Create validation-inference parameters from a model configuration.

        Args:
            cfg: Model configuration containing validation-inference settings.

        Returns:
            Validation-inference parameter container.
        """
        obj = cls(
            run_val_inference=cfg.get("run_val_inference", False),
            use_multilingual_asr=cfg.get("use_multilingual_asr", False),
            use_utmos=cfg.get("use_utmos", False),
            whisper_model_name=cfg.get("whisper_model_name", "openai/whisper-large-v3"),
            nemo_asr_model_name=cfg.get("nemo_asr_model_name", "nvidia/parakeet-ctc-0.6b"),
            ssim_model_name=cfg.get("ssim_model_name", "titanet_large"),
        )
        return obj


class ValidationInferenceEngine:
    """Validation-time inference and quality-evaluation engine."""

    def __init__(self, params: ValidationInferenceParams) -> None:
        self.params = params
        self._is_active = False

        self.default_device: torch.device = torch.device("cpu")

        self.whisper_processor: Optional[WhisperProcessor] = None
        self.whisper_model: Optional[WhisperForConditionalGeneration] = None
        self.nemo_asr_model: Optional[nemo_asr.models.EncDecRNNTBPEModel] = None
        self.speaker_verification_model: Optional[nemo_asr.models.EncDecSpeakerLabelModel] = None
        self.utmos_calculator = None

        if self.params.use_utmos and not _HAVE_UTMOSV2:
            raise ImportError(
                "UTMOSv2 is required for UTMOS scoring but is not installed. "
                "Install it with: `pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1`."
            )

    @property
    def is_active(self) -> bool:
        """Get whether validation inference is initialized and active."""
        return self._is_active

    @property
    def use_multilingual_asr(self) -> bool:
        """Get whether validation uses the multilingual ASR model."""
        return self.params.use_multilingual_asr

    def _setup_asr_model(self) -> None:
        """Initialize the ASR model used for validation transcription."""
        if self.use_multilingual_asr:
            self.whisper_processor = WhisperProcessor.from_pretrained(
                pretrained_model_name_or_path=self.params.whisper_model_name
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.params.whisper_model_name
            )
            self.whisper_model.eval()
            self.whisper_model.to(self.default_device)
            self.nemo_asr_model = None

            for p in self.whisper_model.parameters():
                p.requires_grad = False

        else:
            self.nemo_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=self.params.nemo_asr_model_name
            )
            self.nemo_asr_model.freeze()
            self.nemo_asr_model.to(self.default_device)
            self.whisper_processor = None
            self.whisper_model = None

    def _setup_ssim_model(self) -> None:
        """Initialize the speaker-verification model used for similarity scoring."""
        self.speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.params.ssim_model_name
        )
        self.speaker_verification_model.freeze()
        self.speaker_verification_model.to(self.default_device)

    def _setup_utmos_model(self) -> None:
        """Initialize the UTMOS calculator used for audio-quality scoring."""
        self.utmos_calculator = UTMOSv2Calculator(device="cpu")

    def setup(self) -> None:
        """Initialize validation inference and quality-evaluation models when enabled."""
        if not self.params.run_val_inference:
            return

        logging.info("Loading eval models for validation inference...")
        self._setup_asr_model()
        logging.info("ASR model initialized.")
        self._setup_ssim_model()
        logging.info("Speaker verification model initialized.")

        if self.params.use_utmos:
            self._setup_utmos_model()
            logging.info("UTMOSv2 calculator initialized.")

        logging.info("Eval models loaded successfully.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_active = True

    def switch_device(self, device: torch.device | str) -> None:
        """Move loaded validation models to a target device.

        Args:
            device: Target device for the loaded ASR and speaker-verification models.
        """
        target = torch.device(device)

        if self.whisper_model is not None:
            self.whisper_model.to(target)

        if self.nemo_asr_model is not None:
            self.nemo_asr_model.to(target)

        if self.speaker_verification_model is not None:
            self.speaker_verification_model.to(target)

        if target.type == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _transcribe_using_whisper(
        self,
        paths: list[Path],
        languages: list[str],
    ) -> list[str | None]:
        """Transcribe audio files with the multilingual Whisper model.

        Args:
            paths: Audio file paths to transcribe.
            languages: Language codes corresponding to the audio files.

        Returns:
            Normalized transcripts, with `None` for files that could not be transcribed.
        """
        if self.whisper_processor is None:
            return [None] * len(paths)

        device = self.whisper_model.device

        try:
            transcripts = transcribe_with_whisper_from_filepaths(
                audio_filepaths=paths,
                language=languages,
                whisper_processor=self.whisper_processor,
                whisper_model=self.whisper_model,
                device=device,
                normalizer=None,
            )
            transcripts = [process_text_for_cer(x) for x in transcripts]

        except Exception as e:
            logging.warning(f"Batched transcription failed, falling back to per-file mode: {e}.")

            transcripts = []

            for i, p in enumerate(paths):
                lang = languages[i] if i < len(languages) else "en"

                try:
                    transcript = transcribe_with_whisper(
                        audio_filepath=p,
                        language=lang,
                        whisper_processor=self.whisper_processor,
                        whisper_model=self.whisper_model,
                        device=device,
                        normalizer=None,
                    )
                    transcripts.append(process_text_for_cer(transcript))

                except Exception as inner_e:
                    logging.warning(f"Transcription failed for {p}: {inner_e}.")
                    transcripts.append(None)

        return transcripts

    def _transcribe_using_nemo(
        self,
        paths: list[Path],
    ) -> list[str | None]:
        """Transcribe audio files with the configured NeMo ASR model.

        Args:
            paths: Audio file paths to transcribe.

        Returns:
            Normalized transcripts, or `None` values when the ASR model is not available.
        """
        if self.nemo_asr_model is None:
            return [None] * len(paths)

        config = TranscribeConfig(
            use_lhotse=False,
            batch_size=len(paths),
            num_workers=0,
        )
        transcripts = self.nemo_asr_model.transcribe(
            audio=paths,
            batch_size=len(paths),
            override_config=config,
        )
        transcripts = [process_text_for_cer(t.text) for t in transcripts]

        return transcripts

    @torch.no_grad()
    def transcribe_files(
        self,
        paths: list[Path],
        languages: Optional[list[str]],
    ) -> list[str | None]:
        """Transcribe audio files with the configured validation ASR backend.

        Args:
            paths: Audio file paths to transcribe.
            languages: Optional language codes corresponding to the audio files.
                English is used when no languages are supplied.

        Returns:
            Normalized transcripts, with `None` for files that could not be transcribed.
        """
        if self.use_multilingual_asr:
            if languages is None:
                languages = ["en"] * len(paths)
            transcripts = self._transcribe_using_whisper(paths, languages)
        else:
            transcripts = self._transcribe_using_nemo(paths)

        return transcripts

    @torch.no_grad()
    def verify_speakers(self, paths: ValidationAudioPaths):
        """Extract speaker embeddings for generated and context audio.

        Args:
            paths: Generated and context audio file paths.

        Returns:
            A container with generated and context speaker embeddings or `None`
            values when speaker verification is unavailable or fails.
        """
        generated_embeds, context_embeds = None, None

        if self.speaker_verification_model is None:
            return ValidationSpeakerEmbeddings(generated=generated_embeds, context=context_embeds)

        device = self.speaker_verification_model.device

        try:
            generated_embeds = get_speaker_embeddings_from_filepaths(
                filepaths=paths.generated,
                speaker_verification_model=self.speaker_verification_model,
                device=device,
            )
            generated_embeds = generated_embeds.cpu().float().numpy()

            context_embeds = get_speaker_embeddings_from_filepaths(
                filepaths=paths.context,
                speaker_verification_model=self.speaker_verification_model,
                device=device,
            )
            context_embeds = context_embeds.cpu().float().numpy()

        except Exception as e:
            logging.warning(f"Speaker verification failed: {e}.")

        return ValidationSpeakerEmbeddings(generated=generated_embeds, context=context_embeds)

    @torch.no_grad()
    def get_utmos_scores(
        self,
        input_dir: Path,
        filenames: list[str],
    ) -> Optional[list[float]]:
        """Calculate UTMOS quality scores for generated audio files.

        Args:
            input_dir: Directory containing the audio files.
            filenames: Audio filenames to score.

        Returns:
            UTMOS scores in filename order or `None` when UTMOS scoring is not configured.

        Raises:
            RuntimeError: If UTMOS scoring fails.
        """
        scores = None

        if self.utmos_calculator is None:
            return scores

        try:
            res = self.utmos_calculator.process_directory(
                input_dir=input_dir.as_posix(),
                batch_size=len(filenames),
                num_workers=0,
                val_list=filenames,
            )
            scores = [float(item["predicted_mos"]) for item in res]

        except Exception as e:
            raise RuntimeError(f"UTMOSv2 scoring failed: {e}.") from e

        return scores
