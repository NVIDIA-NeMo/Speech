"""Riva-specific text cleaning filter."""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines into single spaces; keep edge intent."""
    return re.sub(r"\s+", " ", text)


class RivaTextFilter(BaseTextFilter):
    """Cleans text for TTS by removing markdown, bullets, and excess spacing."""

    async def filter(self, text: str) -> str:
        """Clean and normalize text prior to TTS synthesis."""
        text = re.sub(r"[*_`~\[\]\(\)\{\}<>]", "", text)
        text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
        text = re.sub(r"(?m)^\s*[•\-]\s+", "", text)
        text = re.sub(r"([\.!\?])(?=[A-Za-z0-9])", r"\1 ", text)
        text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\-']", " ", text)
        text = _normalize_whitespace(text)
        text = re.sub(r"\s+([,\.!\?])", r"\1", text)
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s*'\s*", "'", text)
        return text

    async def handle_interruption(self):
        """No-op interruption handler for compatibility.

        Filter is stateless, so nothing to reset on interruption.
        """
        return None

    async def reset_interruption(self):
        """No-op reset handler for compatibility.

        Filter keeps no internal buffers; nothing to restore.
        """
        return None
