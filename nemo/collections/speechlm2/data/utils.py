# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import inflect
import re
import warnings


_inflect = inflect.engine()

_COMMA_RE    = re.compile(r"([0-9][0-9,]+[0-9])")
_DECIMAL_RE  = re.compile(r"\b([0-9]+)\.([0-9]+)\b")
_DOLLARS_RE  = re.compile(r"\$([0-9,]+(?:\.[0-9]+)?)")
_ORDINAL_RE  = re.compile(r"\b([0-9]+)(st|nd|rd|th)\b", re.IGNORECASE)
_NUMBER_RE   = re.compile(r"\b[0-9]+\b")

# Roman numerals: only real standalone uppercase numerals
_ROMAN_RE = re.compile(r"(?<![A-Z])[IVXLCDM]{2,}(?![A-Z])")

_ROMAN = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}


def get_pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad
    if pad_id is not None:
        return pad_id
    pad_id = tokenizer.unk_id
    if pad_id is not None:
        return pad_id
    warnings.warn(
        "The text tokenizer has no <pad> or <unk> tokens available, using ID 0 for padding (this may lead to silent bugs)."
    )
    return 0


# MCQ system prompt constants
MCQ_SYSTEM_PROMPT_NOTHINK = "Answer the following multiple choice question."
MCQ_SYSTEM_PROMPT_THINK = "Answer the following multiple choice question with an explanation for the answer."


def is_mcq_cut_train(cut, check_think: bool = False) -> bool:
    """Check if a cut is from MCQ training data based on shard_origin."""
    shard_origin = getattr(cut, 'shard_origin', None)
    if shard_origin is None:
        return False
    shard_origin = str(shard_origin)
    return (
        "MCQ_training" in shard_origin
        and "singleQA" not in shard_origin
        and (not check_think or "_think" in shard_origin)
    )


def is_mcq_cut_val(cut) -> bool:
    """Check if a cut is from MCQ validation data based on shard_origin."""
    shard_origin = getattr(cut, 'shard_origin', None)
    if shard_origin is None:
        return False
    return any(pattern in str(shard_origin) for pattern in ("openbookqa", "mmsu", "bbh"))


def is_asr_cut(cut) -> bool:
    """Check if a cut is from ASR data based on task attribute or formatter."""
    return (
        getattr(cut, 'task', None) == 'asr'
        or getattr(cut, 'formatter', None) == 'nemo_tarred_to_duplex'
    )


# --- Number normalization ---

def _roman_to_int(s):
    total = 0
    prev = 0
    for c in reversed(s):
        val = _ROMAN[c]
        total += -val if val < prev else val
        prev = val
    return total

def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal(m):
    return f"{_expand_number(int(m.group(1)))} point {_expand_digits(m.group(2))}"


def _expand_digits(s):
    return " ".join(_inflect.number_to_words(int(c)) for c in s)

def _expand_dollars(m):
    raw = m.group(1).replace(",", "")
    parts = raw.split(".")

    dollars = int(parts[0])
    cents = int(parts[1]) if len(parts) > 1 else 0

    out = []
    if dollars:
        out.append(f"{_expand_number(dollars)} {'dollar' if dollars == 1 else 'dollars'}")
    if cents:
        out.append(f"{_expand_number(cents)} {'cent' if cents == 1 else 'cents'}")
    return " ".join(out) if out else "zero dollars"

def _expand_ordinal(m):
    n = int(m.group(1))
    return _inflect.ordinal(_inflect.number_to_words(n))

def _expand_roman(m):
    return _inflect.number_to_words(_roman_to_int(m.group()))

def _expand_number(num):
    return _inflect.number_to_words(num, andword="")

def normalize_numbers(text):
    try:
        text = re.sub(_COMMA_RE, _remove_commas, text)
        text = re.sub(_ROMAN_RE, _expand_roman, text)
        text = re.sub(_DOLLARS_RE, _expand_dollars, text)
        text = re.sub(_DECIMAL_RE, _expand_decimal, text)
        text = re.sub(_ORDINAL_RE, _expand_ordinal, text)
        text = re.sub(_NUMBER_RE, lambda m: _expand_number(int(m.group())), text)
        return text

    except Exception:
        return text   # fallback: return input unchanged
