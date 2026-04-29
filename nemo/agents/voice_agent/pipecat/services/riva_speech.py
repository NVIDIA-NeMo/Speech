# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA Nemotron Speech ASR and TTS Services implementation.

This module provides integration with NVIDIA Nemotron Speech ASR and TTS Services, including:
- Text-to-Speech (TTS) with support for multiple voices and languages
- Automatic Speech Recognition (ASR) with streaming capabilities

The services can be configured to use either a local Nemotron Speech ASR and TTS Server or
NVIDIA's cloud-hosted models through NVCF.
"""

import asyncio
import concurrent.futures
import os
import re
import warnings
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

import riva.client
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    AggregatedTextFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import AggregationType, BaseTextAggregator
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt, traced_tts
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from nemo.agents.voice_agent.pipecat.frames.riva import (
    RivaFetchVoicesFrame,
    RivaInterimTranscriptionFrame,
    RivaTTSUpdateSettingsFrame,
    RivaVoicesFrame,
)

# Constants
DEFAULT_NVCF_SERVER = "grpc.nvcf.nvidia.com:443"


class NemotronTTSService(TTSService):
    """NVIDIA Nemotron Speech TTS service implementation.

    Provides speech synthesis using NVIDIA's Nemotron Speech TTS models with support for
    multiple voices, languages, and custom dictionaries.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = DEFAULT_NVCF_SERVER,
        voice_id: str = "Magpie-Multilingual.EN-US.Aria",
        sample_rate: int = 22050,
        function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969",
        language: Language | None = Language.EN_US,
        zero_shot_quality: int | None = 20,
        model: str = "magpie_tts_ensemble-Magpie-Multilingual",
        custom_dictionary: dict | None = None,
        encoding: AudioEncoding = AudioEncoding.LINEAR_PCM,
        zero_shot_audio_prompt_file: Path | None = None,
        audio_prompt_encoding: AudioEncoding = AudioEncoding.ENCODING_UNSPECIFIED,
        use_ssl: bool = False,
        tts_timeout: float | None = 20.0,
        text_aggregator: BaseTextAggregator | None = None,
        text_filters: Sequence[BaseTextFilter] | None = None,
        **kwargs,
    ):
        """Initializes the Nemotron Speech TTS service.

        Args:
            api_key (str | None, optional): API key for authentication. Defaults to None.
            server (str, optional): Server address for TTS service. Defaults to "grpc.nvcf.nvidia.com:443".
            voice_id (str, optional): Voice identifier. Defaults to "English-US.Female-1".
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 22050.
            function_id (str, optional): Function identifier for the service.
                Defaults to "0149dedb-2be8-4195-b9a0-e57e0e14f972".
            language (Language | None, optional): Language for synthesis. Defaults to Language.EN_US.
            zero_shot_quality (int | None, optional): Quality level for synthesis. Defaults to 20.
            model (str, optional): Model name for synthesis. Defaults to "fastpitch-hifigan-tts".
            custom_dictionary (dict | None, optional): Custom pronunciation dictionary. Defaults to None.
            encoding (AudioEncoding, optional): Audio encoding format. Defaults to AudioEncoding.LINEAR_PCM.
            zero_shot_audio_prompt_file (str | None, optional): Path to audio prompt file. Defaults to None.
            audio_prompt_encoding (AudioEncoding, optional): Encoding of audio prompt.
                Defaults to AudioEncoding.LINEAR_PCM.
            use_ssl (bool, optional): Whether to use SSL for connection. Defaults to False.
            text_aggregator (BaseTextAggregator | None, optional): Text aggregator for sentence detection.
                Defaults to None, which uses SimpleTextAggregator.
            text_filters (Sequence[BaseTextFilter] | None, optional): Filters applied after aggregation.
            tts_timeout (float | None, optional): Seconds to wait for audio from Nemotron Speech TTS
                before emitting an ErrorFrame. Set to None to disable the timeout.
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            Exception: If required modules are missing or connection fails.

        Usage:
            If server is not set then it defaults to "grpc.nvcf.nvidia.com:443" and use NVCF hosted models.
            Update function ID to use a different NVCF model. API key is required for NVCF hosted models.
            For using locally deployed Nemotron Speech ASR and TTS Server, set server to "localhost:50051" and
            follow the quick start guide to setup the server.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=False,
            push_stop_frames=True,
            text_aggregator=text_aggregator,
            text_filters=text_filters,
            **kwargs,
        )
        self._api_key = api_key
        self._function_id = function_id
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._language_code = language
        self._zero_shot_quality = zero_shot_quality
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._custom_dictionary = custom_dictionary
        self._encoding = encoding
        self._zero_shot_audio_prompt_file = zero_shot_audio_prompt_file
        self._backend_audio_prompt_file = zero_shot_audio_prompt_file
        self._audio_prompt_encoding = audio_prompt_encoding
        self._model = model
        self._cached_languages: dict[str, dict[str, list[str]]] | None = None
        self._lang_code_lookup: dict[str, str] = {}  # lowercase -> actual lang code (O(1) lookup)
        self._tts_timeout = tts_timeout

        metadata = [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ]

        if server == DEFAULT_NVCF_SERVER:
            use_ssl = True

        try:
            auth = riva.client.Auth(None, use_ssl, server, metadata)
            self._service = riva.client.SpeechSynthesisService(auth)
            # warm up the service
            _ = self._service.stub.GetRivaSynthesisConfig(riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest())
        except Exception as e:
            logger.error(
                "In order to use NVIDIA Nemotron Speech TTS and ASR Services, you will either need a locally "
                "deployed Nemotron Speech TTS model (Deploy TTS model using "
                "https://docs.nvidia.com/nim/riva/tts/latest/overview.html and set the server url to "
                "localhost:50051), or you can set the NVIDIA_API_KEY environment "
                "variable to connect with nvcf hosted models."
            )
            raise Exception(f"Missing module: {e}") from e

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: True as this service supports metric generation.
        """
        return True

    def _is_multilingual_tts(self) -> bool:
        """Check if MULTILINGUAL_TTS env variable is set to skip text cleaning."""
        return os.getenv("ENABLE_MULTILINGUAL", "").lower() in ("1", "true", "yes")

    def is_zeroshot_model(self) -> bool:
        """Check if the current model supports zero-shot voice cloning.

        Returns:
            bool: True if the model is a zero-shot model (contains 'ZeroShot' or 'zeroshot').
        """
        return "zeroshot" in self._model.lower() if self._model else False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle TTS control frames and delegate standard behavior.

        - RivaFetchVoicesFrame: Query Riva for available voices and emit a single
          RivaVoicesFrame containing all voice information (available voices,
          current selection, and custom audio prompt status).
        - RivaTTSUpdateSettingsFrame: Update voice settings (default or custom voice).
        - Any other frame: Delegate to the base TTSService implementation.
        """
        await super().process_frame(frame, direction)

        # Respond to RivaFetchVoicesFrame from the pipeline/UI
        if isinstance(frame, RivaFetchVoicesFrame):
            try:
                # Send consolidated voice information in a single frame
                zero_shot_prompt = self._zero_shot_audio_prompt_file.name if self._zero_shot_audio_prompt_file else ""
                await self.push_frame(
                    RivaVoicesFrame(
                        available_voices=self.list_available_voices(),
                        current_voice_id=getattr(self, "_voice_id", ""),
                        is_zeroshot_model=self.is_zeroshot_model(),
                        zero_shot_prompt=zero_shot_prompt,
                    ),
                    direction,
                )
            except Exception as e:
                logger.warning(f"{self} Unable to fetch voice information: {e}")
            return

        # Handle voice settings update
        if isinstance(frame, RivaTTSUpdateSettingsFrame):
            if frame.voice_type == "default":
                # Default voice: clear custom prompt and set voice_id
                self._zero_shot_audio_prompt_file = None
                if frame.identifier:
                    self.set_voice(frame.identifier)
                    logger.debug(f"Switched to default voice: {frame.identifier}")
            elif frame.voice_type == "custom":
                # Custom voice: use path from frame or fallback to backend
                prompt_id = frame.identifier

                if frame.custom_prompt_path and frame.custom_prompt_path.exists():
                    # Path provided directly in frame
                    prompt_path = frame.custom_prompt_path
                    self._zero_shot_audio_prompt_file = prompt_path
                    logger.debug(f"Switched to custom prompt: {prompt_id} -> {prompt_path}")
                elif prompt_id == "backend":
                    # Use initial backend-configured prompt
                    self._zero_shot_audio_prompt_file = self._backend_audio_prompt_file
                    logger.debug("Switched to backend prompt")
                else:
                    logger.warning(f"Custom prompt path not provided or invalid for: {prompt_id}")

    async def _push_tts_frames(
        self,
        src_frame: AggregatedTextFrame,
        includes_inter_frame_spaces: bool | None = False,
        append_tts_text_to_context: bool | None = True,
    ) -> None:
        """Capture pre-filter text for display when text_filters are applied.

        When text_filters are in use (e.g. RivaTextFilter for en-US), the text
        passed to run_tts is already filtered. We store the aggregated text
        before the base applies filters so run_tts can set display_text to the
        original text for transcript/UI. When text_filters are not used
        (emotion/multilingual paths), display_text remains the processed
        spoken text (clean_text) from run_tts.
        """
        agg_type = src_frame.aggregated_by
        text = src_frame.text
        if agg_type in self._skip_aggregator_types:
            await self.push_frame(src_frame)
            return
        text = text.lstrip("\n")
        if not text.strip():
            return
        if self._text_filters:
            self._display_text_before_filter = text
        else:
            self._display_text_before_filter = None
        await super()._push_tts_frames(src_frame, includes_inter_frame_spaces)

    def _strip_non_speech_content(self, text: str) -> str:
        """Strip MetaData and Translation fields from text - not meant to be spoken.

        Handles:
        - "Hello! MetaData: greeting" -> "Hello!"
        - "Gerne! Translation: Of course!" -> "Gerne!"
        """
        if not text:
            return text

        result = text

        # Strip MetaData field
        metadata_match = re.search(r"\s*MetaData\s*:?\s*.*$", result, re.IGNORECASE | re.DOTALL)
        if metadata_match:
            result = result[: metadata_match.start()].strip()
            logger.debug(f"Stripped MetaData, keeping: [{result}]")

        # Strip Translation field (LLM sometimes adds unwanted translations)
        translation_match = re.search(r"\s*Translation\s*:?\s*.*$", result, re.IGNORECASE | re.DOTALL)
        if translation_match:
            result = result[: translation_match.start()].strip()
            logger.debug(f"Stripped Translation, keeping: [{result}]")

        return result

    def _is_metadata_only(self, text: str) -> bool:
        """Check if text is only metadata content (not meant for TTS)."""
        lower = text.lower()
        return lower.startswith("metadata:") or lower.startswith("metadata ")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run text-to-speech synthesis.

        Handles LLM output formats:
        - Plain text: "Hello there!"
        - Emotion format: "Emotion: Happy Text: Hello!"
        - Language format: "Language: en-US Text: Hello! MetaData: greeting"
        - MetaData only: "MetaData: internal note" (skipped)
        """
        # Skip text with no speakable content
        if not any(c.isalnum() for c in text):
            logger.debug(f"Skipping TTS - no alphanumeric characters: [{text}]")
            return

        clean_text = text.strip()

        # Skip chunks that are only metadata
        if self._is_metadata_only(clean_text):
            logger.debug(f"Skipping MetaData chunk: [{clean_text}]")
            return

        # Process structured LLM output formats
        if clean_text.startswith("Emotion:"):
            emotion_tag, clean_text = self.parse_emotion_response(clean_text)
            await self.switch_emotion(emotion_tag)
        elif clean_text.startswith("Language:"):
            lang_code, clean_text = self.parse_language_response(clean_text)
            await self.switch_language(lang_code)
        else:
            # Plain text - strip any trailing non-speech content
            clean_text = self._strip_non_speech_content(clean_text)

        # Final check - ensure we have content to speak
        if not clean_text:
            logger.debug("Skipping TTS - no content after processing")
            return

        logger.debug(f"Generating TTS: [{clean_text}]")

        # Split text into <=200-character chunks at whitespace where possible
        def _split_text_into_chunks(s: str, max_len: int) -> list[str]:
            """Split into <= max_len chunks, preferring whitespace boundaries.

            Guarantees:
            - No empty chunks
            - No leading/trailing whitespace on chunks
            - Preserves words when possible (splits mid-word only if needed)
            """
            input_text = s.strip()
            if not input_text:
                return []
            if len(input_text) <= max_len:
                return [input_text]

            chunks: list[str] = []
            input_length = len(input_text)
            current_index = 0

            while current_index < input_length:
                window_end_index = min(current_index + max_len, input_length)
                if window_end_index < input_length:
                    # Work within a bounded window [current_index, window_end_index)
                    window_text = input_text[current_index:window_end_index]
                    last_whitespace_offset = next(
                        (k for k in range(len(window_text) - 1, -1, -1) if window_text[k].isspace()),
                        -1,
                    )
                    if last_whitespace_offset > 0:
                        chunk = window_text[:last_whitespace_offset].rstrip()
                        current_index += last_whitespace_offset
                        while current_index < input_length and input_text[current_index].isspace():
                            current_index += 1
                    else:
                        chunk = window_text
                        current_index = window_end_index
                else:
                    chunk = input_text[current_index:window_end_index]
                    current_index = window_end_index

                if chunk:
                    chunks.append(chunk)

            return chunks

        chunks = _split_text_into_chunks(clean_text, 200)

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        # Push text frame immediately after TTSStartedFrame.
        # TTSService base processor will push the tts text after sending generated tts audio downstream
        # Need to push the text before audio frame for better TTS transcription.
        # When text_filters were applied, display_text is the pre-filter text for transcript/UI;
        # otherwise (emotion/multilingual or no filter) use processed clean_text.
        display_text = getattr(self, "_display_text_before_filter", None) or clean_text
        self._display_text_before_filter = None  # clear after use so next run_tts doesn't reuse
        tts_text_frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        tts_text_frame.metadata["display_text"] = display_text
        yield tts_text_frame

        async def get_next_response(iterator):
            def _next():
                try:
                    return next(iterator)
                except StopIteration:
                    return None

            return await asyncio.get_event_loop().run_in_executor(None, _next)

        total_audio_length = 0
        ttfb_stopped = False
        # Synthesize audio for each 200-char chunk sequentially
        for chunk in chunks:
            try:
                responses = self._service.synthesize_online(
                    chunk.strip(),
                    self._voice_id,
                    self._language_code,
                    sample_rate_hz=self._sample_rate,
                    zero_shot_audio_prompt_file=self._zero_shot_audio_prompt_file,
                    audio_prompt_encoding=self._audio_prompt_encoding,
                    zero_shot_quality=self._zero_shot_quality,
                    custom_dictionary=self._custom_dictionary,
                    encoding=self._encoding,
                )

                response_iterator = iter(responses)

                while True:
                    try:
                        if self._tts_timeout:
                            resp = await asyncio.wait_for(
                                get_next_response(response_iterator),
                                timeout=self._tts_timeout,
                            )
                        else:
                            resp = await get_next_response(response_iterator)
                    except TimeoutError:
                        logger.error(f"{self} timeout waiting for TTS audio response")
                        yield ErrorFrame(error=f"{self} timeout waiting for TTS audio response")
                        break

                    if resp is None:
                        break

                    try:
                        total_audio_length += len(resp.audio)
                        if not ttfb_stopped:
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True
                        frame = TTSAudioRawFrame(
                            audio=resp.audio,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                        )
                        yield frame
                    except Exception as e:
                        logger.error(f"{self} Error processing TTS response: {e}")
                        break
            except Exception as e:
                logger.error(f"{self} Error invoking TTS: {e}")
                break

        if not ttfb_stopped:
            await self.stop_ttfb_metrics()

        # TODO: Remove this once Pipecat fixes transport output chunk cut-off issues
        # Add 400ms of silence at the end of each response
        silence_duration_ms = 400
        silence_bytes = int(self._sample_rate * 2 * silence_duration_ms / 1000)  # 2 bytes per sample (16-bit)
        silence_audio = bytes(silence_bytes)
        yield TTSAudioRawFrame(
            audio=silence_audio,
            sample_rate=self._sample_rate,
            num_channels=1,
        )

        await self.start_tts_usage_metrics(text)
        logger.debug(f"Total generated TTS audio length: {total_audio_length / (self._sample_rate * 2)} seconds")
        yield TTSStoppedFrame()

    def parse_emotion_response(self, text: str) -> tuple[str, str]:
        """Parse LLM output in `Emotion: <tag> Text: <response>` format.

        Handles formats:
        - "Emotion: Happy Text: Hello!"
        - "Emotion Happy Text ..." (legacy, optional colons)

        Strips MetaData and other non-speech content from the spoken text.

        Returns:
            tuple[str, str]: (emotion_tag, spoken_text) - emotion_tag may be empty
        """
        clean_text = text.strip()
        if not clean_text:
            return "", ""

        # Parse key-value format: Emotion[:]? <tag> Text[:]? <response>
        match = re.search(
            r"Emotion\s*:?\s*(.+?)(?=Text\s*:?)\s*Text\s*:?\s*(.+)$",
            clean_text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            # Attempt to salvage only the spoken text if available
            if text_only := re.search(r"Text\s*:?\s*(.+)$", clean_text, re.IGNORECASE | re.DOTALL):
                return "", self._strip_non_speech_content(text_only.group(1).strip())
            # If only emotion content exists, return empty spoken text
            if re.search(r"Emotion\s*:?\s*(.+)$", clean_text, re.IGNORECASE | re.DOTALL):
                return "", ""
            return "", self._strip_non_speech_content(clean_text)

        emotion_tag = match.group(1).strip()
        spoken_text = self._strip_non_speech_content(match.group(2).strip())
        if not emotion_tag:
            return "", spoken_text
        return emotion_tag, spoken_text

    async def switch_emotion(self, emotion_tag: str) -> None:
        """Switch TTS voice to the specified emotion variant.

        Resolves voice from current base voice + emotion tag (e.g. base.Happy).
        Emits a `RivaVoicesFrame` so RTVI-backed UIs can refresh the current
        voice when this processor is part of a UI-enabled pipeline.

        Args:
            emotion_tag: Emotion tag (e.g. "Happy", "Sad")
        """
        if not emotion_tag or self.is_zeroshot_model():
            if not emotion_tag:
                return
            logger.debug("Skipping emotion-based voice switch for zeroshot model")
            return

        try:
            current_voice = getattr(self, "_voice_id", "") or ""
            if not current_voice:
                return
            available_voices_map = self.list_available_voices()
            available_voices = {v for langs in available_voices_map.values() for v in langs.get("voices", [])}
            if not available_voices:
                logger.debug("No available voices returned; skipping emotion switch")
                return

            voice_parts = current_voice.split(".")
            if len(voice_parts) > 1 and ".".join(voice_parts[:-1]) in available_voices:
                base_voice = ".".join(voice_parts[:-1])
            else:
                base_voice = current_voice

            emotion_variant = f"{base_voice}.{emotion_tag}"
            if emotion_variant in available_voices:
                target_voice = emotion_variant
            else:
                logger.debug(f"Unable to find suitable voice for {base_voice} with tag [{emotion_tag}]")
                target_voice = base_voice

            if hasattr(self, "set_voice"):
                self.set_voice(target_voice)
            else:
                self._voice_id = target_voice

            logger.debug(f"Switched to voice: {target_voice}")
            await self.push_frame(
                RivaVoicesFrame(
                    available_voices=available_voices_map,
                    current_voice_id=target_voice,
                    is_zeroshot_model=self.is_zeroshot_model(),
                    zero_shot_prompt="",
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to switch voice based on emotion tag: {exc}")

    def parse_language_response(self, text: str) -> tuple[str, str]:
        """Parse LLM output in `Language: <code> Text: <response> MetaData: <info>` format.

        Handles formats:
        - "Language: en-US Text: ... MetaData: ..."
        - "Language en-US Text ..." (legacy)

        Strips MetaData and any translations added by the LLM.

        Returns:
            tuple[str, str]: (lang_code, spoken_text) - defaults to "en-US" if no valid lang code found
        """
        clean_text = text.strip()
        default_lang = "en-US"

        if not clean_text:
            return default_lang, ""

        # Parse key-value format: Language[:]? <code> Text[:]? <response>
        match = re.search(
            r"Language\s*:?\s*([a-z]{2}-[a-z]{2})\s*Text\s*:?\s*(.+)$",
            clean_text,
            re.IGNORECASE | re.DOTALL,
        )

        if not match:
            # Attempt to salvage only the spoken text if available
            if text_only := re.search(r"Text\s*:?\s*(.+)$", clean_text, re.IGNORECASE | re.DOTALL):
                return default_lang, text_only.group(1).strip()
            # If only language content exists, return empty spoken text
            if re.search(r"Language\s*:?\s*([a-z]{2}-[a-z]{2})", clean_text, re.IGNORECASE | re.DOTALL):
                return default_lang, ""
            return default_lang, clean_text

        lang_code = match.group(1).strip() or default_lang
        spoken_text = self._strip_non_speech_content(match.group(2).strip())

        return lang_code, spoken_text

    async def switch_language(self, lang_code: str) -> None:
        """Switch TTS voice to the specified language code.

        Args:
            lang_code: Language code (e.g., "en-US", "de-DE", "fr-FR")
        """
        if not lang_code or lang_code == self._language_code:
            return

        available_voices = self.list_available_voices()
        if not available_voices:
            logger.debug("No available voices returned; skipping language switch")
            return

        matched_lang = self._lang_code_lookup.get(lang_code.lower())

        if matched_lang and available_voices[matched_lang].get("voices"):
            new_voice = available_voices[matched_lang]["voices"][0]
            self._language_code = matched_lang
            self.set_voice(new_voice)
            logger.info(f"Switched TTS to language: {matched_lang}, voice: {new_voice}")
            update_frame = RivaTTSUpdateSettingsFrame(
                voice_type="default", identifier=new_voice, language_code=matched_lang
            )
            await self.push_frame(update_frame)
            await self.push_frame(update_frame, direction=FrameDirection.UPSTREAM)
        else:
            logger.warning(f"Language '{lang_code}' not supported. Available: {list(available_voices.keys())}")

    def list_available_voices(self) -> dict[str, dict[str, list[str]]]:
        """Return voices grouped by language for multilingual models."""
        if self._cached_languages:
            return self._cached_languages

        try:
            resp = self._service.stub.GetRivaSynthesisConfig(
                riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
            )

            subvoices = []
            result: dict[str, dict[str, list[str]]] = {}

            for cfg in resp.model_config:
                params = cfg.parameters
                voice_name = params.get("voice_name")
                language_codes = params.get("language_code")

                if not voice_name or not language_codes:
                    continue

                lang_map = {lc.strip().upper(): lc.strip() for lc in language_codes.split(",") if lc and lc.strip()}
                subvoices = params.get("subvoices", "")
                for sc in subvoices.split(","):
                    sc = sc.strip()
                    if not sc:
                        continue
                    voice_id = sc.split(":")[0].strip()
                    voice_lang_code = voice_id.split(".")[0].strip().upper()
                    if voice_lang_code in lang_map:
                        name = f"{voice_name}.{voice_id}"
                        result.setdefault(lang_map[voice_lang_code], {"voices": []})["voices"].append(name)

            self._cached_languages = result
            # Build O(1) lookup: lowercase lang code -> actual lang code
            self._lang_code_lookup = {lang.lower(): lang for lang in result}
            logger.info(f"list_available_voices returning {len(result)} languages: {list(result.keys())}")
            return result

        except Exception as e:
            logger.error(f"{self} Failed to list available voices: {e}")
            raise


class NemotronASRService(STTService):
    """NVIDIA Nemotron Speech ASR service.

    Provides streaming speech recognition using Nemotron Speech ASR models with support for:
        - Real-time transcription
        - Interim results
        - Interruption handling
        - Voice activity detection
        - Language model customization
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = DEFAULT_NVCF_SERVER,
        function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        language: Language | None = Language.EN_US,
        model: str = "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
        profanity_filter: bool = False,
        automatic_punctuation: bool = False,
        no_verbatim_transcripts: bool = True,
        boosted_lm_words: dict | None = None,
        boosted_lm_score: float = 4.0,
        start_history: int = -1,
        start_threshold: float = -1.0,
        stop_history: int = 500,
        stop_threshold: float = -1.0,
        stop_history_eou: int = 240,
        stop_threshold_eou: float = -1.0,
        custom_configuration: str = "enable_vad_endpointing:true,neural_vad.onset:0.65,apply_partial_itn:true",
        sample_rate: int = 16000,
        audio_channel_count: int = 1,
        max_alternatives: int = 1,
        interim_results: bool = True,
        generate_interruptions: bool = False,  # Only set to True if transport VAD is disabled
        idle_timeout: int = 30,  # Timeout for idle Nemotron Speech ASR request
        use_ssl: bool = False,
        **kwargs,
    ):
        """Initializes the Nemotron Speech ASR service.

        Args:
            api_key: NVIDIA API key for cloud access.
            server: Riva server address.
            function_id: NVCF function identifier.
            language: Language for recognition.
            model: ASR model name.
            profanity_filter: Enable profanity filtering.
            automatic_punctuation: Enable automatic punctuation.
            no_verbatim_transcripts: Disable verbatim transcripts.
            boosted_lm_words: Words to boost in language model.
            boosted_lm_score: Score for boosted words.
            start_history: VAD start history frames.
            start_threshold: VAD start threshold.
            stop_history: VAD stop history frames.
            stop_threshold: VAD stop threshold.
            stop_history_eou: End-of-utterance history frames.
            stop_threshold_eou: End-of-utterance threshold.
            custom_configuration: Additional configuration string.
            sample_rate: Audio sample rate in Hz.
            audio_channel_count: Number of audio channels.
            max_alternatives: Maximum number of alternatives.
            interim_results: Enable interim results.
            generate_interruptions: Enable interruption events.
            idle_timeout: Timeout for idle ASR request in seconds.
            use_ssl: Enable SSL connection.
            **kwargs: Additional arguments for STTService.

        Usage:
            If server is not set then it defaults to "grpc.nvcf.nvidia.com:443" and use NVCF hosted models.
            Update function ID to use a different NVCF model. API key is required for NVCF hosted models.
            For using locally deployed Riva Speech Server, set server to "localhost:50051" and
            follow the quick start guide to setup the server.
        """
        super().__init__(**kwargs)
        self._profanity_filter = profanity_filter
        self._automatic_punctuation = automatic_punctuation
        self._no_verbatim_transcripts = no_verbatim_transcripts
        self._language_code = language
        self._boosted_lm_words = boosted_lm_words
        self._boosted_lm_score = boosted_lm_score
        self._start_history = start_history
        self._start_threshold = start_threshold
        self._stop_history = stop_history
        self._stop_threshold = stop_threshold
        self._stop_history_eou = stop_history_eou
        self._stop_threshold_eou = stop_threshold_eou
        self._custom_configuration = custom_configuration
        self._sample_rate: int = sample_rate
        self._model = model
        self._audio_channel_count = audio_channel_count
        self._max_alternatives = max_alternatives
        self._interim_results = interim_results
        self._idle_timeout = idle_timeout
        self.last_transcript_frame = None
        self.set_model_name(model)

        metadata = [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ]

        if server == DEFAULT_NVCF_SERVER:
            use_ssl = True

        try:
            auth = riva.client.Auth(None, use_ssl, server, metadata)
            self._asr_service = riva.client.ASRService(auth)
        except Exception as e:
            logger.error(
                "In order to use nvidia Nemotron Speech TTS and ASR Services, you will either need a locally "
                "deployed Nemotron Speech ASR model (Deploy ASR model using "
                "https://docs.nvidia.com/nim/riva/asr/latest/overview.html and set the server url to "
                "localhost:50051), or you can set the NVIDIA_API_KEY environment "
                "variable to connect with nvcf hosted models."
            )
            raise Exception(f"Missing module: {e}") from e

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self._language_code,
                model=self._model,
                max_alternatives=self._max_alternatives,
                profanity_filter=self._profanity_filter,
                enable_automatic_punctuation=self._automatic_punctuation,
                verbatim_transcripts=not self._no_verbatim_transcripts,
                sample_rate_hertz=self._sample_rate,
                audio_channel_count=self._audio_channel_count,
            ),
            interim_results=self._interim_results,
        )
        riva.client.add_word_boosting_to_config(config, self._boosted_lm_words, self._boosted_lm_score)
        riva.client.add_endpoint_parameters_to_config(
            config,
            self._start_history,
            self._start_threshold,
            self._stop_history,
            self._stop_history_eou,
            self._stop_threshold,
            self._stop_threshold_eou,
        )
        riva.client.add_custom_configuration_to_config(config, self._custom_configuration)
        self._config = config

        self._queue = asyncio.Queue()
        self._generate_interruptions = generate_interruptions
        if self._generate_interruptions:
            self._vad_state = VADState.QUIET

        # Initialize the thread task and response task
        self._thread_task = None
        self._response_task = None
        # Initialize ASR compute latency tracking
        self._audio_duration_counter = 0.0  # Tracks cumulative audio duration sent to Riva (in seconds)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: False as this service does not support metric generation.
        """
        return False

    async def start(self, frame: StartFrame):
        """Start the ASR service.

        Args:
            frame: The StartFrame that triggered the start.
        """
        await super().start(frame)
        self._response_task = self.create_task(self._response_task_handler())
        self._response_queue = asyncio.Queue()

    async def stop(self, frame: EndFrame):
        """Stop the ASR service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the ASR service and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        if self._thread_task is not None and not self._thread_task.done():
            await self.cancel_task(self._thread_task)
        if self._response_task is not None and not self._response_task.done():
            await self.cancel_task(self._response_task)

    def _response_handler(self):
        try:
            logger.debug("Sending new ASR streaming request...")
            responses = self._asr_service.streaming_response_generator(
                audio_chunks=self,
                streaming_config=self._config,
            )
            for response in responses:
                if not response.results:
                    continue
                asyncio.run_coroutine_threadsafe(self._response_queue.put(response), self.get_event_loop())
        except Exception as e:
            logger.error(f"Error in ASR stream: {e}")
            raise
        logger.debug("ASR streaming request terminated.")

    async def _thread_task_handler(self):
        try:
            # Reset audio duration counter for new ASR session
            self._audio_duration_counter = 0.0
            self._thread_running = True
            await asyncio.to_thread(self._response_handler)
        except asyncio.CancelledError:
            self._thread_running = False
            raise

    async def _handle_interruptions(self, frame: Frame):
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band.
            if isinstance(frame, UserStartedSpeakingFrame):
                logger.debug("User started speaking")
                await self.push_frame(frame)
                await self.push_frame(UserStartedSpeakingFrame(), direction=FrameDirection.UPSTREAM)

                # Make sure we notify about interruptions quickly out-of-band.
                await self.push_interruption_task_frame_and_wait()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("User stopped speaking")
                await self.push_frame(frame)
                await self.push_frame(UserStoppedSpeakingFrame(), direction=FrameDirection.UPSTREAM)

    async def _handle_response(self, response):
        """Process ASR response and generate appropriate transcription frames.

        Handles three types of transcription results:
        1. Final results (is_final=True): Complete, confirmed transcriptions
        2. Stable interim results (stability=1.0): High-confidence partial results
        3. Partial results (stability<1.0): Lower-confidence, in-progress transcriptions

        Also manages voice activity detection (VAD) state and interruption handling
        when enabled. Each type of result generates appropriate transcription frames
        with different stability values.
        """
        partial_transcript = ""
        for result in response.results:
            if result and not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if transcript and len(transcript) > 0:
                await self.stop_ttfb_metrics()
                if result.is_final:
                    await self.stop_processing_metrics()
                    if self._generate_interruptions:
                        self._vad_state = VADState.QUIET
                        await self._handle_interruptions(UserStoppedSpeakingFrame())
                    # Calculate ASR compute latency
                    if result.audio_processed:
                        compute_latency = self._audio_duration_counter - result.audio_processed
                        logger.debug(f"{self.name} ASR compute latency: {compute_latency}")
                    logger.debug(f"Final user transcript: [{transcript}]")
                    await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601(), None))
                    await self._handle_transcription(transcript, True, self._language_code)
                    self.last_transcript_frame = None
                    break
                elif abs(result.stability - 1.0) < 1e-9:
                    if self._generate_interruptions and self._vad_state != VADState.SPEAKING:
                        self._vad_state = VADState.SPEAKING
                        await self._handle_interruptions(UserStartedSpeakingFrame())
                    if (
                        self.last_transcript_frame is None
                        or abs(self.last_transcript_frame.stability - 1.0) >= 1e-9
                        or (self.last_transcript_frame.text.rstrip() != transcript.rstrip())
                    ):
                        logger.debug(f"Interim user transcript: [{transcript}]")
                        frame = RivaInterimTranscriptionFrame(
                            transcript, "", time_now_iso8601(), None, stability=result.stability
                        )
                        await self.push_frame(frame)
                        await self._handle_transcription(transcript, False, self._language_code)
                        self.last_transcript_frame = frame
                    break
                else:
                    if self._generate_interruptions and self._vad_state != VADState.SPEAKING:
                        self._vad_state = VADState.SPEAKING
                        await self._handle_interruptions(UserStartedSpeakingFrame())
                    partial_transcript += transcript

        if len(partial_transcript) > 0 and (
            self.last_transcript_frame is None
            or (abs(self.last_transcript_frame.stability - 1.0) < 1e-9)
            or (self.last_transcript_frame.text.rstrip() != partial_transcript.rstrip())
        ):
            logger.debug(f"Partial user transcript: [{partial_transcript}]")
            frame = RivaInterimTranscriptionFrame(partial_transcript, "", time_now_iso8601(), None, stability=0.1)
            await self.push_frame(frame)
            self.last_transcript_frame = frame

    async def _response_task_handler(self):
        while True:
            try:
                response = await self._response_queue.get()
                await self._handle_response(response)
            except asyncio.CancelledError:
                break

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language | None = None):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text recognition.

        Args:
            audio: The audio data to process.

        Yields:
            Frame: A sequence of frames containing the recognition results.
        """
        if self._thread_task is None or self._thread_task.done():
            self._thread_task = self.create_task(self._thread_task_handler())
        await self._queue.put(audio)
        yield None

    def __next__(self) -> bytes:
        """Get the next audio chunk for processing.

        Returns:
            bytes: The next audio chunk.

        Raises:
            StopIteration: When no more audio chunks are available.
        """
        if not self._thread_running:
            raise StopIteration
        try:
            future = asyncio.run_coroutine_threadsafe(self._queue.get(), self.get_event_loop())
            result = future.result(timeout=self._idle_timeout)
            # Increment audio duration counter based on audio chunk size
            # Assuming LINEAR_PCM encoding: bytes_per_sample = 2, channels = self._audio_channel_count
            bytes_per_sample = 2  # 16-bit PCM
            total_samples = len(result) // (bytes_per_sample * self._audio_channel_count)
            duration_seconds = total_samples / self._sample_rate
            self._audio_duration_counter += duration_seconds
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.info(f"ASR service is idle for {self._idle_timeout} seconds, terminating active ASR request...")
            self._thread_task = None
            raise StopIteration from None
        except Exception as e:
            future.cancel()
            raise e
        return result

    def __iter__(self):
        """Get iterator for audio chunks.

        Returns:
            NemotronASRService: Self reference for iteration.
        """
        return self


# Deprecated aliases for backward compatibility


class RivaASRService(NemotronASRService):
    """Deprecated alias for NemotronASRService.

    .. deprecated:: 0.4.0
        Use :class:`NemotronASRService` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the deprecated RivaASRService alias.

        Args:
            *args: Positional arguments passed to NemotronASRService.
            **kwargs: Keyword arguments passed to NemotronASRService.
        """
        warnings.warn(
            "RivaASRService is deprecated and will be removed in a future version. "
            "Please use NemotronASRService instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class RivaTTSService(NemotronTTSService):
    """Deprecated alias for NemotronTTSService.

    .. deprecated:: 0.4.0
        Use :class:`NemotronTTSService` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the deprecated RivaTTSService alias.

        Args:
            *args: Positional arguments passed to NemotronTTSService.
            **kwargs: Keyword arguments passed to NemotronTTSService.
        """
        warnings.warn(
            "RivaTTSService is deprecated and will be removed in a future version. "
            "Please use NemotronTTSService instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
