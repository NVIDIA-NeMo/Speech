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
"""Standalone target-text tokenizer for EasyMagpie pronunciation control."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EasyMagpieTextTokenizer:
    """Interleave ordinary text ids with offset IPA-BPE ids."""

    def __init__(
        self,
        base_tokenizer: Any,
        *,
        text_vocab_size: int,
        phoneme_tokenizer: Any = None,
        text_phoneme_token_offset: int | None = None,
        text_phoneme_vocab_size: int = 0,
        bop_marker: str = "<bop>",
        eop_marker: str = "<eop>",
    ) -> None:
        self.base_tokenizer = base_tokenizer
        self.text_vocab_size = int(text_vocab_size)
        self.phoneme_tokenizer = phoneme_tokenizer
        self.text_phoneme_token_offset = text_phoneme_token_offset
        self.text_phoneme_vocab_size = int(text_phoneme_vocab_size)
        self.bop_marker = bop_marker
        self.eop_marker = eop_marker
        self.enabled = phoneme_tokenizer is not None

        if not self.enabled:
            return
        if not bop_marker or not eop_marker or bop_marker == eop_marker:
            raise ValueError("Pronunciation-control markers must be non-empty and distinct.")
        if text_phoneme_token_offset is None:
            raise ValueError("Pronunciation-control tokenizer requires text_phoneme_token_offset.")
        if text_phoneme_token_offset + self.text_phoneme_vocab_size != self.text_vocab_size:
            raise ValueError(
                "Pronunciation-control IPA ids must occupy the trailing text-vocabulary range: "
                f"offset={text_phoneme_token_offset}, size={self.text_phoneme_vocab_size}, "
                f"text_vocab_size={self.text_vocab_size}."
            )
        raw_vocab_size = len(phoneme_tokenizer.get_vocab())
        if raw_vocab_size + 3 != self.text_phoneme_vocab_size:
            raise ValueError(
                "Pronunciation-control IPA tokenizer size does not match converted metadata: "
                f"raw tokenizer size={raw_vocab_size}, reserved special tokens=3, "
                f"text range size={self.text_phoneme_vocab_size}."
            )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "EasyMagpieTextTokenizer":
        """Load the base tokenizer and optional IPA tokenizer from a converted model."""
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer

        model_path = Path(model_path)
        config = json.loads((model_path / "config.json").read_text())
        base_tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        text_vocab_size = int(config.get("text_vocab_size", config.get("vocab_size", 0)))
        if not bool(config.get("enable_phoneme_text_input", False)):
            return cls(base_tokenizer, text_vocab_size=text_vocab_size)

        required_fields = (
            "text_phoneme_token_offset",
            "text_phoneme_vocab_size",
            "text_phoneme_tokenizer_file",
        )
        missing_fields = [field for field in required_fields if config.get(field) is None]
        if missing_fields:
            raise ValueError(
                "Pronunciation-control model config is missing required fields: " + ", ".join(missing_fields)
            )
        tokenizer_file = config["text_phoneme_tokenizer_file"]
        tokenizer_path = model_path / tokenizer_file
        if not tokenizer_path.is_file():
            raise ValueError(f"Pronunciation-control IPA tokenizer not found: {tokenizer_path}")

        return cls(
            base_tokenizer,
            text_vocab_size=text_vocab_size,
            phoneme_tokenizer=Tokenizer.from_file(str(tokenizer_path)),
            text_phoneme_token_offset=int(config["text_phoneme_token_offset"]),
            text_phoneme_vocab_size=int(config["text_phoneme_vocab_size"]),
            bop_marker=str(config.get("phoneme_text_bop_marker", "<bop>")),
            eop_marker=str(config.get("phoneme_text_eop_marker", "<eop>")),
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode target text, interpreting marked IPA spans when enabled."""
        if add_special_tokens:
            raise ValueError("EasyMagpie target text must be encoded with add_special_tokens=False.")
        encoder = self.incremental_encoder()
        return encoder.push(text, final=True)

    def encode_context(self, text: str) -> list[int]:
        """Encode context text with the base tokenizer only."""
        return list(self.base_tokenizer.encode(text))

    def incremental_encoder(self) -> "EasyMagpieIncrementalTextEncoder":
        return EasyMagpieIncrementalTextEncoder(self)

    def _encode_text(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.base_tokenizer.encode(text, add_special_tokens=False))

    def _encode_phoneme(self, text: str) -> list[int]:
        if not text:
            return []
        offset = int(self.text_phoneme_token_offset)
        return [offset + token_id for token_id in self.phoneme_tokenizer.encode(text).ids]


class EasyMagpieIncrementalTextEncoder:
    """Stateful parser for marker and IPA spans split across text chunks."""

    def __init__(self, tokenizer: EasyMagpieTextTokenizer) -> None:
        self.tokenizer = tokenizer
        self._buffer = ""
        self._in_phoneme_span = False

    @property
    def clean(self) -> bool:
        """Whether raw token ids may be safely interleaved at this boundary."""
        return not self._buffer and not self._in_phoneme_span

    def push(self, text: str, *, final: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Incremental text chunks must be strings.")
        if not self.tokenizer.enabled:
            return self.tokenizer._encode_text(text)

        self._buffer += text
        token_ids: list[int] = []
        while self._buffer:
            if self._in_phoneme_span:
                eop_idx = self._buffer.find(self.tokenizer.eop_marker)
                if eop_idx < 0:
                    if final:
                        raise ValueError(
                            f"Found `{self.tokenizer.bop_marker}` without a matching "
                            f"`{self.tokenizer.eop_marker}`."
                        )
                    break
                ipa_text = self._buffer[:eop_idx].strip()
                token_ids.extend(self.tokenizer._encode_phoneme(ipa_text))
                self._buffer = self._buffer[eop_idx + len(self.tokenizer.eop_marker) :]
                self._in_phoneme_span = False
                continue

            bop_idx = self._buffer.find(self.tokenizer.bop_marker)
            eop_idx = self._buffer.find(self.tokenizer.eop_marker)
            if eop_idx >= 0 and (bop_idx < 0 or eop_idx < bop_idx):
                raise ValueError(
                    f"Found `{self.tokenizer.eop_marker}` without a matching `{self.tokenizer.bop_marker}`."
                )
            if bop_idx >= 0:
                token_ids.extend(self.tokenizer._encode_text(self._buffer[:bop_idx]))
                self._buffer = self._buffer[bop_idx + len(self.tokenizer.bop_marker) :]
                self._in_phoneme_span = True
                continue

            held_suffix = 0 if final else self._marker_prefix_suffix_length(self._buffer)
            emit_end = len(self._buffer) - held_suffix
            token_ids.extend(self.tokenizer._encode_text(self._buffer[:emit_end]))
            self._buffer = self._buffer[emit_end:]
            break

        if final and self._in_phoneme_span:
            raise ValueError(f"Found `{self.tokenizer.bop_marker}` without a matching `{self.tokenizer.eop_marker}`.")
        return token_ids

    def finish(self) -> list[int]:
        """Flush literal marker prefixes and reject incomplete IPA spans."""
        return self.push("", final=True)

    def _marker_prefix_suffix_length(self, text: str) -> int:
        max_length = 0
        for marker in (self.tokenizer.bop_marker, self.tokenizer.eop_marker):
            for length in range(1, min(len(text), len(marker) - 1) + 1):
                if text.endswith(marker[:length]):
                    max_length = max(max_length, length)
        return max_length
