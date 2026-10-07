# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Text-delimited source buffering for simultaneous translation."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


_SPACE_RE = re.compile(r"\s+")
_TERMINAL_RE = re.compile(r"[.!?。！？]+")
_INITIALISM_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")
_CLOSING_DELIMITERS = frozenset("\"'”’»)]}】」』")
_NON_TERMINAL_ABBREVIATIONS = {
    "dr.",
    "e.g.",
    "etc.",
    "fig.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "prof.",
    "st.",
    "vs.",
}


@dataclass(frozen=True)
class MTSourceDecision:
    """The stable source unit and suffix available at one streaming update."""

    source: str
    is_final: bool
    boundary_reason: str = ""
    retained_suffix: str = ""


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text or "").strip()


def _is_non_terminal_period(text: str, period_end: int) -> bool:
    if period_end < len(text) and period_end >= 2:
        if text[period_end - 2].isdigit() and text[period_end].isdigit():
            return True

    token = text[:period_end].rsplit(" ", 1)[-1].lower()
    return (
        token in _NON_TERMINAL_ABBREVIATIONS
        or _INITIALISM_RE.fullmatch(token) is not None
        or (len(token) == 2 and token[0].isalpha() and token.endswith("."))
    )


def last_text_boundary(text: str) -> int:
    """Return the end offset of the last reliable terminal punctuation span."""

    boundary = -1
    for match in _TERMINAL_RE.finditer(text):
        # Do not create punctuation-only units from residue punctuation at the
        # beginning of a new acoustic segment.
        if not any(character.isalnum() for character in text[max(0, boundary) : match.start()]):
            continue
        if match.group(0) == "." and _is_non_terminal_period(text, match.end()):
            continue
        boundary = match.end()
        while boundary < len(text) and text[boundary] in _CLOSING_DELIMITERS:
            boundary += 1
    return boundary


class TextMTSourceBuffer:
    """Accumulate stable ASR text independently of acoustic endpointing.

    Acoustic EoU resets only the cumulative-ASR comparison view. It does not
    finalize the active MT source. Stable terminal punctuation, safety limits,
    or an explicit stream-end flush finalize an MT source unit.
    """

    def __init__(self, max_source_units: int = 256, max_duration_ms: int = 30_000):
        self.max_source_units = max(1, int(max_source_units))
        self.max_duration_ms = max(1, int(max_duration_ms))
        self.active_source = ""
        self.previous_asr_view = ""
        self.active_duration_ms = 0

    @staticmethod
    def _stable_delta(previous: str, current: str) -> tuple[str, bool, int]:
        """Return unseen text, exact-continuation status, and tail rollback."""

        if not previous:
            return current, False, 0
        if current.startswith(previous):
            return current[len(previous) :], True, 0
        if previous.startswith(current):
            return "", True, 0

        # RNNT may revise earlier punctuation or spacing while preserving the
        # recent tail. Continue after that shared tail rather than appending the
        # complete cumulative view again.
        tail_matches = [
            block
            for block in difflib.SequenceMatcher(None, previous, current, autojunk=False).get_matching_blocks()
            if block.size > 0 and block.a + block.size == len(previous)
        ]
        if tail_matches:
            block = max(tail_matches, key=lambda item: item.size)
            if block.size >= min(8, len(previous)):
                return current[block.b + block.size :], True, 0

        common = 0
        for left, right in zip(previous, current):
            if left != right:
                break
            common += 1
        if common >= max(1, int(0.8 * len(previous))):
            return current[common:], True, len(previous) - common

        # The view probably belongs to a new acoustic segment. Preserve it;
        # source and duration limits prevent unbounded growth.
        return current, False, 0

    @staticmethod
    def _append(base: str, delta: str, *, exact_continuation: bool) -> str:
        if not delta:
            return base
        if not base:
            return delta.strip()
        if exact_continuation or delta[0].isspace() or delta[0] in ".,!?;:":
            return f"{base}{delta}"
        return f"{base} {delta}"

    def _split_at_unit_limit(self) -> tuple[str, str] | None:
        units = list(re.finditer(r"\S+", self.active_source))
        if len(units) <= self.max_source_units:
            return None
        cut = units[self.max_source_units - 1].end()
        return self.active_source[:cut].strip(), self.active_source[cut:].strip()

    def update(self, stable_asr_view: str, *, acoustic_eou: bool, elapsed_ms: int) -> MTSourceDecision:
        """Consume one cumulative stable-ASR view."""

        current = _normalize(stable_asr_view)
        delta, exact_continuation, rollback = self._stable_delta(self.previous_asr_view, current)
        if rollback and rollback <= len(self.active_source):
            self.active_source = self.active_source[:-rollback]
        self.active_source = self._append(self.active_source, delta, exact_continuation=exact_continuation)
        is_temporary_shortening = bool(
            self.previous_asr_view and self.previous_asr_view.startswith(current) and self.previous_asr_view != current
        )
        if acoustic_eou:
            self.previous_asr_view = ""
        elif not is_temporary_shortening:
            # A shorter cumulative view was ignored above. Keep the longer
            # comparison anchor so that a later re-extension is not appended twice.
            self.previous_asr_view = current
        self.active_duration_ms += max(0, int(elapsed_ms))

        boundary = last_text_boundary(self.active_source)
        if boundary >= 0:
            source = self.active_source[:boundary].strip()
            suffix = self.active_source[boundary:].strip()
            self.active_source = suffix
            self.active_duration_ms = 0
            return MTSourceDecision(source, True, "punctuation", suffix)

        limited = self._split_at_unit_limit()
        if limited is not None:
            source, suffix = limited
            self.active_source = suffix
            self.active_duration_ms = 0
            return MTSourceDecision(source, True, "max_source_units", suffix)

        if self.active_source and self.active_duration_ms >= self.max_duration_ms:
            source = self.active_source.strip()
            self.active_source = ""
            self.active_duration_ms = 0
            return MTSourceDecision(source, True, "max_duration")

        return MTSourceDecision(self.active_source.strip(), False)

    def defer_boundary(self, source: str, retained_suffix: str) -> None:
        """Restore a punctuation split whose target-side handoff was empty."""

        source = _normalize(source)
        if not source:
            raise ValueError("A deferred boundary requires a non-empty source")
        self.active_source = self._append(source, _normalize(retained_suffix), exact_continuation=False)

    def flush(self) -> MTSourceDecision:
        """Finalize source retained at the end of a stream."""

        source = self.active_source.strip()
        self.active_source = ""
        self.previous_asr_view = ""
        self.active_duration_ms = 0
        return MTSourceDecision(source, bool(source), "stream_end" if source else "")
