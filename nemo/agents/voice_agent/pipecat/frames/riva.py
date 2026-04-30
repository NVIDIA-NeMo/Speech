# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Frames for Nemotron Speech ASR and TTS Services.

This module provides frame definitions for NVIDIA Nemotron Speech ASR and TTS Services,
specifically focused on interim transcription handling.

Classes:
    RivaInterimTranscriptionFrame: Frame for interim transcription results with stability metrics
    RivaFetchVoicesFrame: Frame to request TTS service to provide voice information
    RivaVoicesFrame: Frame carrying comprehensive voice information from Nemotron Speech TTS
    RivaTTSUpdateSettingsFrame: Frame to update Nemotron Speech TTS voice settings
"""

from dataclasses import dataclass
from pathlib import Path

from pipecat.frames.frames import ControlFrame, DataFrame, InterimTranscriptionFrame


@dataclass
class RivaInterimTranscriptionFrame(InterimTranscriptionFrame):
    """An interim transcription frame with stability metrics from Nemotron Speech ASR.

    Extends the base InterimTranscriptionFrame to include stability
    scoring for speculative speech processing. These frames are generated during
    active speech and help determine when to trigger early response generation.

    Also see:
    - InterimTranscriptionFrame : Base class for interim transcriptions

    Args:
        stability (float): Confidence score for the transcription, ranging 0.0-1.0.
            - 0.0: Highly unstable, likely to change
            - 1.0: Maximum stability, no expected changes
            Only transcripts with stability=1.0 are processed for speculative
            speech handling. Defaults to 0.1.
        user_id (str): Identifier of the speaking participant.
        text (str): The interim transcription text.
        language (str): Language code of the transcription.
        timestamp (float): Timestamp of when the transcription was generated.

    Typical usage example:
        >>> frame = RivaInterimTranscriptionFrame(
        ...     text="Hello world",
        ...     stability=0.95,
        ...     user_id="user_1",
        ...     language="en-US",
        ...     timestamp=1234567890.0
        ... )
        >>> print(frame)  # Output will be:
        RivaInterimTranscriptionFrame(
            user: user_1,
            text: [Hello world],
            stability: 0.95,
            language: en-US,
            timestamp: 1234567890.0
        )
    """

    stability: float = 0.1

    def __str__(self):
        """Return a string representation of the frame.

        Returns:
            str: A formatted string containing all frame attributes.
        """
        return (
            f"{self.name}(user: {self.user_id}, text: [{self.text}], "
            f"stability: {self.stability}, language: {self.language}, timestamp: {self.timestamp})"
        )


@dataclass
class RivaFetchVoicesFrame(ControlFrame):
    """Control frame to request TTS service to provide voice information.

    Triggers the TTS service to return available voices, current voice selection,
    and custom audio prompt status in a single RivaVoicesFrame response.
    """


@dataclass
class RivaVoicesFrame(DataFrame):
    """Data frame carrying comprehensive voice information from Nemotron Speech TTS.

    Consolidates available voices, current selection, and custom audio prompt status
    into a single frame to reduce communication overhead and simplify UI updates.

    Attributes:
        available_voices: Dictionary of available voices grouped by language.
            Format: { "en-US": { "voices": ["Voice.Subvoice", ...] }, ... }
        current_voice_id: The currently active voice identifier (e.g., "English-US.Female-1")
        is_zeroshot_model: Whether the active model supports zero-shot voice cloning
        zero_shot_prompt: Name/path of the active zero-shot audio prompt file. Empty string
            if no custom prompt is currently active. The UI can check if this is non-empty
            to determine if a custom zero-shot prompt is in use.
    """

    available_voices: dict[str, dict[str, list[str]]]
    current_voice_id: str
    is_zeroshot_model: bool
    zero_shot_prompt: str = ""


@dataclass
class RivaTTSUpdateSettingsFrame(ControlFrame):
    """Control frame to update Nemotron Speech TTS voice settings.

    Handles both default voice selection and custom zero-shot voice selection.

    Attributes:
        voice_type: Type of voice - "default" for standard voices, "custom" for zero-shot voices
        identifier: For default voices, this is the voice_id (e.g., "English-US.Female-1").
            For custom voices, this is the prompt_id (e.g., "backend" or user-uploaded prompt ID).
        language_code: Language code for the voice (e.g., "en-US", "de-DE"). Optional.
        custom_prompt_path: Optional path to custom voice prompt file (for custom voice type only).
    """

    voice_type: str  # "default" or "custom"
    identifier: str
    language_code: str = ""
    custom_prompt_path: Path | None = None
