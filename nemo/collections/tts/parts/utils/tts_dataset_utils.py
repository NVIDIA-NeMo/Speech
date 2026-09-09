# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import functools
import importlib
import logging
import os
import random
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import torch
from einops import rearrange
from scipy import ndimage
from torch.special import gammaln
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCBPEModelWithPrompt
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import HybridRNNTCTCPromptTranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.common.parts.utils import mask_sequence_tensor

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    Normalizer = None
    PYNINI_AVAILABLE = False


def get_abs_rel_paths(input_path: Path, base_path: Path) -> Tuple[Path, Path]:
    """
    Get the absolute and relative paths of input file path.

    Args:
        input_path: An absolute or relative path.
        base_path: base directory the input is relative to.

    Returns:
        The absolute and relative paths of the file.
    """
    if os.path.isabs(input_path):
        abs_path = input_path
        rel_path = input_path.relative_to(base_path)
    else:
        rel_path = input_path
        abs_path = base_path / rel_path

    return abs_path, rel_path


def get_audio_filepaths(manifest_entry: Dict[str, Any], audio_dir: Path) -> Tuple[Path, Path]:
    """
    Get the absolute and relative paths of audio from a manifest entry.

    Args:
        manifest_entry: Manifest entry dictionary.
        audio_dir: base directory where audio is stored.

    Returns:
        The absolute and relative paths of the audio.
    """
    audio_filepath = Path(manifest_entry["audio_filepath"])
    audio_filepath_abs, audio_filepath_rel = get_abs_rel_paths(input_path=audio_filepath, base_path=audio_dir)
    return audio_filepath_abs, audio_filepath_rel


def normalize_volume(audio: np.array, volume_level: float = 0.95) -> np.array:
    """Apply peak normalization to the input audio."""
    if not (0.0 <= volume_level <= 1.0):
        raise ValueError(f"Volume must be in range [0.0, 1.0], received {volume_level}")

    if audio.size == 0:
        return audio

    max_sample = np.max(np.abs(audio))
    if max_sample == 0:
        return audio

    return volume_level * (audio / np.max(np.abs(audio)))


def _validate_probability(name: str, value: float):
    """Validate that a probability lies in the inclusive range ``[0.0, 1.0]``.

    Args:
        name: Parameter name to include in the error message.
        value: Probability value to validate.

    Raises:
        ValueError: If ``value`` is outside the valid probability range.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be in range [0.0, 1.0], received {value}")


def _sample_probability_range(name: str, min_value: float, max_value: float) -> float:
    """Sample a probability uniformly from an inclusive configured range.

    Args:
        name: Base parameter name to include in validation errors.
        min_value: Lower bound of the probability range.
        max_value: Upper bound of the probability range.

    Returns:
        A uniformly sampled probability between ``min_value`` and ``max_value``.

    Raises:
        ValueError: If either bound is outside ``[0.0, 1.0]`` or the lower
            bound is greater than the upper bound.
    """
    _validate_probability(f"{name}_min", min_value)
    _validate_probability(f"{name}_max", max_value)
    if min_value > max_value:
        raise ValueError(f"`{name}_min` must be <= `{name}_max`, received {min_value} > {max_value}")
    return random.uniform(min_value, max_value)


def has_phoneme_text_spans(text: str, bop_marker: str = "<bop>", eop_marker: str = "<eop>") -> bool:
    """Check whether text contains either an opening or closing phoneme-span marker.

    Checking for either marker also identifies malformed or incomplete spans so
    callers can avoid transforming the marked input before it is validated.

    Args:
        text: Text to inspect.
        bop_marker: Marker that opens an inline phoneme span.
        eop_marker: Marker that closes an inline phoneme span.

    Returns:
        ``True`` if either marker occurs in ``text``; otherwise ``False``.
    """
    return bop_marker in text or eop_marker in text


def _split_text_and_phoneme_spans(
    text: str,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> List[Tuple[str, str]]:
    """Split mixed text into ordered text and phoneme segments.

    Args:
        text: Input containing regular text and optionally marked phoneme spans.
        bop_marker: Marker that opens an inline phoneme span.
        eop_marker: Marker that closes an inline phoneme span.

    Returns:
        Non-empty ``("text", segment)`` and ``("phoneme", segment)`` tuples
        in their original order. Phoneme-span markers are omitted.

    Raises:
        ValueError: If an opening or closing marker has no matching counterpart.
    """
    segments = []
    cursor = 0

    while cursor < len(text):
        bop_idx = text.find(bop_marker, cursor)
        if bop_idx == -1:
            if eop_marker in text[cursor:]:
                raise ValueError(f"Found `{eop_marker}` without a matching `{bop_marker}` in text: {text}")
            segments.append(("text", text[cursor:]))
            break

        eop_before_bop_idx = text.find(eop_marker, cursor, bop_idx)
        if eop_before_bop_idx != -1:
            raise ValueError(f"Found `{eop_marker}` without a matching `{bop_marker}` in text: {text}")

        if bop_idx > cursor:
            segments.append(("text", text[cursor:bop_idx]))

        span_start = bop_idx + len(bop_marker)
        eop_idx = text.find(eop_marker, span_start)
        if eop_idx == -1:
            raise ValueError(f"Found `{bop_marker}` without a matching `{eop_marker}` in text: {text}")

        segments.append(("phoneme", text[span_start:eop_idx].strip()))
        cursor = eop_idx + len(eop_marker)

    return [(kind, segment) for kind, segment in segments if segment]


def partially_phonemize_text(
    text: str,
    ipa_alignment: Optional[List],
    partial_phoneme_portion: float,
    full_ipa_text: Optional[str] = None,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> str:
    """Replace a sampled portion of aligned words with marked IPA spans.

    The requested portion determines how many aligned words are selected.
    Adjacent selected words are merged into one span. If every aligned word is
    selected, the supplied full IPA transcription is emitted as a single span.
    Invalid or unavailable alignment data leaves the original text unchanged.

    Args:
        text: Original text whose character offsets are referenced by the alignment.
        ipa_alignment: Word alignments in the form
            ``(text_start, text_end, word, ipa_text)``.
        partial_phoneme_portion: Fraction of aligned words to replace, in
            ``[0.0, 1.0]``.
        full_ipa_text: IPA transcription of the complete input text.
        bop_marker: Marker to place before each generated IPA span.
        eop_marker: Marker to place after each generated IPA span.

    Returns:
        Text with sampled words replaced by marked IPA spans, or the original
        text when replacement cannot be performed.

    Raises:
        ValueError: If ``partial_phoneme_portion`` is outside ``[0.0, 1.0]``.
    """
    _validate_probability("partial_phoneme_portion", partial_phoneme_portion)
    if partial_phoneme_portion == 0.0 or not text or not ipa_alignment or not full_ipa_text:
        return text

    num_selected = min(len(ipa_alignment), max(1, round(partial_phoneme_portion * len(ipa_alignment))))
    if num_selected == len(ipa_alignment):
        return f"{bop_marker}{full_ipa_text}{eop_marker}"

    ipa_ranges = []
    ipa_cursor = 0
    for _, _, _, ipa_text in ipa_alignment:
        ipa_start = full_ipa_text.find(ipa_text.strip(), ipa_cursor)
        if ipa_start < 0:
            return text
        ipa_end = ipa_start + len(ipa_text.strip())
        ipa_ranges.append((ipa_start, ipa_end))
        ipa_cursor = ipa_end

    selected_indices = sorted(random.sample(range(len(ipa_alignment)), num_selected))
    selected_runs = []
    for index in selected_indices:
        if selected_runs and index == selected_runs[-1][1] + 1:
            selected_runs[-1] = (selected_runs[-1][0], index)
        else:
            selected_runs.append((index, index))

    output_parts = []
    cursor = 0
    for first_index, last_index in selected_runs:
        start = ipa_alignment[first_index][0]
        end = ipa_alignment[last_index][1]
        ipa_start = ipa_ranges[first_index][0]
        ipa_end = ipa_ranges[last_index][1]
        output_parts.append(text[cursor:start])
        output_parts.append(f"{bop_marker}{full_ipa_text[ipa_start:ipa_end]}{eop_marker}")
        cursor = end
    output_parts.append(text[cursor:])

    return ''.join(output_parts)


def tokenize_text_with_phoneme_spans(
    text_tokenizer,
    text_str: str,
    tokenizer_name: str,
    enable_phoneme_text_input: bool = False,
    phoneme_tokenizer=None,
    text_phoneme_token_offset: Optional[int] = None,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> List[int]:
    """Tokenize regular text with optional inline IPA spans marked by ``<bop>...<eop>``.

    Span markers are syntax only and are not emitted as token IDs. IPA span IDs are encoded with the phoneme tokenizer
    and shifted by ``text_phoneme_token_offset`` so they live in the text-channel vocabulary.

    Args:
        text_tokenizer: Text tokenizer that supports named tokenizer selection.
        text_str: Text to encode, optionally containing marked IPA spans.
        tokenizer_name: Name of the text tokenizer used for regular text segments.
        enable_phoneme_text_input: Whether to parse and separately encode marked
            phoneme spans. If disabled, the complete input is encoded as text.
        phoneme_tokenizer: Tokenizer used to encode IPA span contents. Required
            when ``enable_phoneme_text_input`` is enabled.
        text_phoneme_token_offset: Offset added to every phoneme token ID so it
            occupies the text-channel vocabulary. Required when phoneme text
            input is enabled.
        bop_marker: Marker that opens an inline phoneme span.
        eop_marker: Marker that closes an inline phoneme span.

    Returns:
        Token IDs containing interleaved text IDs and offset phoneme IDs.

    Raises:
        ValueError: If phoneme text input is enabled without its tokenizer or
            token offset, or if span markers are unmatched.
    """
    if not enable_phoneme_text_input:
        return text_tokenizer.encode(text=text_str, tokenizer_name=tokenizer_name)

    if phoneme_tokenizer is None:
        raise ValueError("`phoneme_tokenizer` is required when `enable_phoneme_text_input=True`.")
    if text_phoneme_token_offset is None:
        raise ValueError("`text_phoneme_token_offset` is required when `enable_phoneme_text_input=True`.")

    token_ids = []
    for segment_type, segment in _split_text_and_phoneme_spans(text_str, bop_marker=bop_marker, eop_marker=eop_marker):
        if segment_type == "text":
            token_ids.extend(text_tokenizer.encode(text=segment, tokenizer_name=tokenizer_name))
        else:
            token_ids.extend(text_phoneme_token_offset + token_id for token_id in phoneme_tokenizer.encode(segment))

    return token_ids


class BetaBinomialInterpolator:
    """
    This module calculates alignment prior matrices (based on beta-binomial distribution) using cached popular sizes and image interpolation.
    The implementation is taken from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/data_function.py
    """

    def __init__(self, round_mel_len_to=50, round_text_len_to=10, cache_size=500, scaling_factor: float = 1.0):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        cached_func = lambda x, y: beta_binomial_prior_distribution(x, y, scaling_factor=scaling_factor)
        self.bank = functools.lru_cache(maxsize=cache_size)(cached_func)

    @staticmethod
    def round(val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = BetaBinomialInterpolator.round(w, to=self.round_mel_len_to)
        bh = BetaBinomialInterpolator.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def general_padding(item, item_len, max_len, pad_value=0):
    if item_len < max_len:
        item = torch.nn.functional.pad(item, (0, max_len - item_len), value=pad_value)
    return item


def stack_tensors(tensors: List[torch.Tensor], max_lens: List[int], pad_value: float = 0.0) -> torch.Tensor:
    """
    Create batch by stacking input tensor list along the time axes.

    Args:
        tensors: List of tensors to pad and stack
        max_lens: List of lengths to pad each axis to, starting with the last axis
        pad_value: Value for padding

    Returns:
        Padded and stacked tensor.
    """
    padded_tensors = []
    for tensor in tensors:
        padding = []
        for i, max_len in enumerate(max_lens, 1):
            padding += [0, max_len - tensor.shape[-i]]

        padded_tensor = torch.nn.functional.pad(tensor, pad=padding, value=pad_value)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.stack(padded_tensors)
    return stacked_tensor


def logbeta(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logcombinations(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def logbetabinom(n, a, b, x):
    return logcombinations(n, x) + logbeta(x + a, n - x + b) - logbeta(a, b)


def beta_binomial_prior_distribution(phoneme_count: int, mel_count: int, scaling_factor: float = 1.0) -> np.array:
    x = rearrange(torch.arange(0, phoneme_count), "b -> 1 b")
    y = rearrange(torch.arange(1, mel_count + 1), "b -> b 1")
    a = scaling_factor * y
    b = scaling_factor * (mel_count + 1 - y)
    n = torch.FloatTensor([phoneme_count - 1])

    return logbetabinom(n, a, b, x).exp().numpy()


def get_base_dir(paths):
    def is_relative_to(path1, path2):
        try:
            path1.relative_to(path2)
            return True
        except ValueError:
            return False

    def common_path(path1, path2):
        while path1 is not None:
            if is_relative_to(path2, path1):
                return path1
            path1 = path1.parent if path1 != path1.parent else None
        return None

    base_dir = None
    for p in paths:
        audio_dir = Path(p).parent
        if base_dir is None:
            base_dir = audio_dir
            continue
        base_dir = common_path(base_dir, audio_dir)

    return base_dir


def filter_dataset_by_duration(entries: List[Dict[str, Any]], min_duration: float, max_duration: float):
    """
    Filter out manifest entries based on duration.

    Args:
        entries: List of manifest entry dictionaries.
        min_duration: Minimum duration below which entries are removed.
        max_duration: Maximum duration above which entries are removed.

    Returns:
        filtered_entries: List of manifest entries after filtering.
        total_hours: Total duration of original dataset, in hours
        filtered_hours: Total duration of dataset after filtering, in hours
    """
    filtered_entries = []
    total_duration = 0.0
    filtered_duration = 0.0
    for entry in entries:
        duration = entry["duration"]
        total_duration += duration
        if (min_duration and duration < min_duration) or (max_duration and duration > max_duration):
            continue

        filtered_duration += duration
        filtered_entries.append(entry)

    total_hours = total_duration / 3600.0
    filtered_hours = filtered_duration / 3600.0

    return filtered_entries, total_hours, filtered_hours


def get_weighted_sampler(
    sample_weights: List[float], batch_size: int, world_size: int, num_steps: int
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create pytorch sampler for doing weighted random sampling.

    Args:
        sample_weights: List of sampling weights for all elements in the dataset.
        batch_size: Batch size to sample.
        world_size: Number of devices being used.
        num_steps: Number of steps to be considered an epoch.

    Returns:
        Pytorch sampler
    """
    weights = torch.tensor(sample_weights, dtype=torch.float64)
    num_samples = batch_size * world_size * num_steps
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_samples)
    return sampler


def _read_audio(
    audio_filepath: Path, sample_rate: int, offset: float, duration: float, n_retries: int = 5
) -> AudioSegment:
    # File seeking sometimes fails when reading flac files with libsndfile < 1.0.30.
    # Read audio as int32 to minimize issues, and retry read on a different segment in case of failure.
    # https://github.com/bastibe/python-soundfile/issues/274
    for _ in range(n_retries):
        try:
            return AudioSegment.from_file(
                audio_filepath, target_sr=sample_rate, offset=offset, duration=duration, int_values=True
            )
        except Exception:
            traceback.print_exc()

    raise ValueError(f"Failed to read audio {audio_filepath}")


def _segment_audio(
    audio_filepath: Path,
    sample_rate: int,
    offset: float,
    n_samples: int,
    max_offset: Optional[float] = None,
    n_retries: int = 5,
) -> AudioSegment:
    for _ in range(n_retries):
        try:
            if max_offset:
                offset = random.uniform(offset, max_offset)
            return AudioSegment.segment_from_file(
                audio_filepath, target_sr=sample_rate, n_segments=n_samples, offset=offset, dtype="int32"
            )
        except Exception:
            traceback.print_exc()

    raise ValueError(f"Failed to segment audio {audio_filepath}")


def load_audio(
    manifest_entry: Dict[str, Any],
    audio_dir: Path,
    sample_rate: int,
    max_duration: Optional[float] = None,
    volume_norm: bool = False,
) -> Tuple[np.ndarray, Path, Path]:
    """
    Load audio file from a manifest entry.

    Args:
        manifest_entry: Manifest entry dictionary.
        audio_dir: base directory where audio is stored.
        sample_rate: Sample rate to load audio as.
        max_duration: Optional float, maximum amount of audio to read, in seconds.
        volume_norm: Whether to apply volume normalization to the loaded audio.

    Returns:
        Audio array, and absolute and relative paths to audio file.
    """
    audio_filepath_abs, audio_filepath_rel = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
    offset = manifest_entry.get("offset", 0.0)
    duration = manifest_entry.get("duration", 0.0)

    if max_duration is not None:
        duration = min(duration, max_duration)

    audio_segment = _read_audio(
        audio_filepath=audio_filepath_abs, sample_rate=sample_rate, offset=offset, duration=duration
    )
    audio = audio_segment.samples

    if volume_norm:
        audio = normalize_volume(audio)

    return audio, audio_filepath_abs, audio_filepath_rel


def sample_audio(
    manifest_entry: Dict[str, Any],
    audio_dir: Path,
    sample_rate: int,
    n_samples: int,
    volume_norm: bool = False,
) -> Tuple[np.ndarray, Path, Path]:
    """
    Randomly sample an audio segment from a manifest entry.

    Args:
        manifest_entry: Manifest entry dictionary.
        audio_dir: base directory where audio is stored.
        sample_rate: Sample rate to load audio as.
        n_samples: Size of audio segment to sample.
        volume_norm: Whether to apply volume normalization to the sampled audio.

    Returns:
        Audio array, and absolute and relative paths to audio file.
    """
    audio_filepath_abs, audio_filepath_rel = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
    offset = manifest_entry.get("offset", None)
    duration = manifest_entry.get("duration", 0.0)

    if offset is not None:
        audio_dur = librosa.get_duration(filename=audio_filepath_abs)
        max_end_sec = min(offset + duration, audio_dur - 0.1)
        max_offset = max(offset, max_end_sec - (n_samples / sample_rate))
    else:
        max_offset = None

    audio_segment = _segment_audio(
        audio_filepath=audio_filepath_abs,
        sample_rate=sample_rate,
        offset=offset,
        max_offset=max_offset,
        n_samples=n_samples,
    )
    audio = audio_segment.samples

    if volume_norm:
        audio = normalize_volume(audio)

    return audio, audio_filepath_abs, audio_filepath_rel


# Titles that should NEVER cause sentence splits (always followed by names)
_TITLE_ABBREVIATIONS = {
    'mr',
    'mrs',
    'ms',
    'dr',
    'prof',
    'sr',
    'jr',
    'rev',
    'gov',
    'gen',
    'col',
    'lt',
    'sgt',
    'capt',
}

# =============================================================================
# Sentence separator definitions
# =============================================================================

# Default sentence endings (used for all languages)
_DEFAULT_SENTENCE_ENDINGS = ['.', '?', '!']

# Language-specific sentence endings (includes default endings)
# Languages in this dict (ja, zh, hi) have special punctuation that splits
# regardless of following whitespace (since these languages don't use spaces between sentences)
_SENTENCE_ENDINGS = {
    "ja": ['。', '？', '！', '…', '.', '?', '!'],  # Japanese + Western
    "zh": ['。', '？', '！', '…', '.', '?', '!'],  # Chinese + Western
    "hi": ['।', '॥', '.', '?', '!'],  # Hindi Danda + Western
}


def split_by_sentence(
    paragraph: str,
    language: str = "en",
) -> List[str]:
    """
    Split a paragraph into sentences based on sentence-ending punctuation.

    Sentence separators are chosen from the given language (e.g. ".", "?", "!"
    for English; "。", "？", "！" plus Western punctuation for Japanese/Chinese).
    Handles edge cases like abbreviations (e.g., "Dr.", "Mr.", "a.m.") by
    requiring a space after the separator before splitting for Western punctuation;
    for languages like ja/zh/hi, native punctuation splits without requiring a space.
    Sentence-ending punctuation is preserved with each sentence.

    Args:
        paragraph: The input text paragraph to split into sentences.
        language: Language code (e.g. "en", "ja", "zh", "hi"). Determines which
            sentence-ending characters are used. Defaults to "en".

    Returns:
        List of sentence strings with punctuation preserved.

    Examples:
        >>> split_by_sentence("Hello world. How are you?")
        ["Hello world.", "How are you?"]

        >>> split_by_sentence("Dr. Smith is here. Good morning!")
        ["Dr. Smith is here.", "Good morning!"]

        >>> split_by_sentence("こんにちは。元気ですか？", language="ja")
        ["こんにちは。", "元気ですか？"]
    """
    # Get sentence separators for this language
    sentence_separators = _get_sentence_separators_for_language(language)

    # For special languages (ja, zh, hi), their native punctuation splits without whitespace
    # Special endings are those unique to the language (not in default)
    if language in _SENTENCE_ENDINGS:
        special_endings = set(sentence_separators) - set(_DEFAULT_SENTENCE_ENDINGS)
    else:
        special_endings = set()

    if not paragraph or not paragraph.strip():
        return []

    # Normalize text: replace hyphens with spaces, remove asterisks
    paragraph = paragraph.replace('-', ' ')
    paragraph = paragraph.replace('*', '')

    sentences = []
    last_sep_idx = -1

    for i, char in enumerate(paragraph):
        if char not in sentence_separators:
            continue
        # Check if current char is a separator and next char is a space
        # This avoids splitting abbreviations like "Dr." or "a.m."
        next_char = paragraph[i + 1] if i + 1 < len(paragraph) else ""
        # Determine if we should split at this position
        # Special language punctuation: split regardless of following character (no spaces in these languages)
        # Western punctuation: require whitespace or end-of-string
        if char in special_endings:
            should_split = True
        else:
            should_split = next_char == "" or next_char.isspace()

        if not should_split:
            continue

        # Check if this is a title abbreviation - never split on titles (Dr., Mr., etc.)
        if char == '.':
            # Get the word before the period
            word_start = last_sep_idx + 1 if last_sep_idx >= 0 else 0
            word_before = (
                paragraph[word_start:i].strip().split()[-1].lower() if paragraph[word_start:i].strip() else ""
            )

            # Never split on title abbreviations (they're always followed by names)
            if word_before in _TITLE_ABBREVIATIONS:
                continue

        # Extract the sentence (from after last separator to current separator inclusive)
        start_idx = last_sep_idx + 1 if last_sep_idx >= 0 else 0
        sentences.append(paragraph[start_idx : i + 1].strip())

        # Update last_sep_idx: if next char is whitespace, point to it (so +1 skips it)
        # If no whitespace (CJK), point to separator (so +1 gives us the next char)
        if next_char.isspace():
            last_sep_idx = i + 1  # Point to the whitespace, will be skipped
        else:
            last_sep_idx = i  # Point to separator, next sentence starts at i+1

    # Add remaining text as the last sentence
    if last_sep_idx < len(paragraph) - 1:
        start_idx = last_sep_idx + 1 if last_sep_idx >= 0 else 0
        remaining = paragraph[start_idx:].strip()
        if remaining:
            sentences.append(remaining)

    # Remove empty sentences and capitalize first letter
    sentences = [sent for sent in sentences if len(sent) > 0]
    sentences = [sent if sent[0].isupper() else sent[0].upper() + sent[1:] for sent in sentences if sent]

    return sentences


def _get_sentence_separators_for_language(language: str) -> List[str]:
    """
    Get language-specific sentence separators.

    For special languages (ja, zh, hi), returns their full punctuation set.
    For all other languages, returns the default sentence endings.

    Args:
        language: Language code (e.g., "en", "ja", "hi").

    Returns:
        List of sentence separator characters for the given language.
        Defaults to ['.', '?', '!'] for unlisted languages.
    """
    return _SENTENCE_ENDINGS.get(language, _DEFAULT_SENTENCE_ENDINGS)


def chunk_and_tokenize_text_by_sentence(
    text: str,
    tokenizer_name: str,
    text_tokenizer: Any,
    eos_token_id: int,
    language: str = "en",
    enable_phoneme_text_input: bool = False,
    phoneme_tokenizer=None,
    text_phoneme_token_offset: Optional[int] = None,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Tokenize text split by sentences, adding EOS token after each sentence.

    Args:
        text: Input text to tokenize.
        tokenizer_name: Name of the tokenizer to use (e.g., "english_phoneme").
        text_tokenizer: The tokenizer instance.
        eos_token_id: End-of-sequence token ID to append.
        language: Language code for selecting appropriate sentence separators.
            Supported: "en", "ja", "hi", "zh", "es", "fr", "it", "de", "vi".
            Defaults to "en".
        enable_phoneme_text_input: Whether to parse and separately encode marked
            phoneme spans.
        phoneme_tokenizer: Tokenizer used for marked phoneme spans. Required
            when ``enable_phoneme_text_input`` is enabled.
        text_phoneme_token_offset: Offset added to phoneme token IDs in the
            text-channel vocabulary.
        bop_marker: Marker that opens an inline phoneme span.
        eop_marker: Marker that closes an inline phoneme span.

    Returns:
        Tuple of:
            - chunked_tokens: List of token tensors, one per sentence.
            - chunked_tokens_len: List of token lengths.
            - chunked_text: List of sentence strings.
    """
    split_sentences = split_by_sentence(text, language=language)

    chunked_tokens = []
    chunked_tokens_len = []
    chunked_text = []

    for sentence in split_sentences:
        chunked_text.append(sentence)
        tokens = tokenize_text_with_phoneme_spans(
            text_tokenizer=text_tokenizer,
            text_str=sentence,
            tokenizer_name=tokenizer_name,
            enable_phoneme_text_input=enable_phoneme_text_input,
            phoneme_tokenizer=phoneme_tokenizer,
            text_phoneme_token_offset=text_phoneme_token_offset,
            bop_marker=bop_marker,
            eop_marker=eop_marker,
        )
        tokens = tokens + [eos_token_id]
        tokens = torch.tensor(tokens, dtype=torch.int32)
        tokens_len = tokens.shape[0]
        chunked_tokens.append(tokens)
        chunked_tokens_len.append(tokens_len)

    return chunked_tokens, chunked_tokens_len, chunked_text


@dataclass
class LanguageThresholds:
    """Language-specific word/character thresholds for determining when to split text.

    Text exceeding the threshold for its language will be split into sentences.
    Text below the threshold will be processed as a single chunk.

    The thresholds approximate ~20 seconds of audio per language.

    Attributes:
        thresholds: Dict mapping language code to word count threshold.
            For character-based languages (like Chinese, Japanese), this is character count;
            see get_word_count() for which languages use character vs word count.
    """

    thresholds: Dict[str, int] = field(
        default_factory=lambda: {
            "en": 45,  # English: ~20 seconds of audio
            "es": 73,  # Spanish
            "fr": 69,  # French
            "de": 50,  # German
            "it": 53,  # Italian
            "vi": 50,  # Vietnamese
            "zh": 100,  # Chinese (character count)
            "hi": 50,  # Hindi
            "ja": 80,  # Japanese (character count)
        }
    )

    def get_word_count(self, text: str, language: str) -> int:
        """Get word/character count for text based on language.

        Args:
            text: Input text to count.
            language: Language code (e.g., "en", "zh").

        Returns:
            Word count for most languages, character count for character-based languages.
        """
        if not text or not text.strip():
            return 0

        if language == "zh":
            # Chinese: count characters (no word boundaries)
            return len([c for c in text if not c.isspace()])

        if language == "ja":
            try:
                import pyopenjtalk

                # run_frontend returns list of word dictionaries (NJD format)
                njd = pyopenjtalk.run_frontend(text)
                # Filter out None/invalid entries and count words
                word_count = sum(1 for word in njd if isinstance(word, dict) and word.get('string', '').strip())
                return word_count if word_count > 0 else len([c for c in text if not c.isspace()])
            except ImportError:
                # Fallback: use character count for Japanese if pyopenjtalk not available
                return len([c for c in text if not c.isspace()])

        # Default: whitespace splitting for English, Hindi, and other languages
        return len(text.split())

    def exceeds_threshold(self, text: str, language: str) -> bool:
        """Check if text exceeds the threshold for the given language.

        Args:
            text: Input text to check.
            language: Language code.

        Returns:
            True if text should be split into sentences, False for single chunk.
        """
        threshold = self.thresholds.get(language, self.thresholds.get("en", 45))
        count = self.get_word_count(text, language)
        return count >= threshold


# Default language thresholds instance
DEFAULT_LANGUAGE_THRESHOLDS = LanguageThresholds()


# Centralized mapping from language codes to tokenizer name candidates
# Used by both do_tts() and ChunkedTTSInferenceDataset
LANGUAGE_TOKENIZER_MAP: Dict[str, List[str]] = {
    "en": ["english_phoneme", "english"],
    "de": ["german_phoneme", "german"],
    "es": ["spanish_phoneme", "spanish"],
    "fr": ["french_chartokenizer", "french"],
    "it": ["italian_phoneme", "italian"],
    "vi": ["vietnamese_phoneme", "vietnamese"],
    "zh": ["mandarin_phoneme", "mandarin", "chinese"],
    "hi": ["hindi_chartokenizer", "hindi"],
    "ja": ["japanese_phoneme", "japanese"],
}


def get_tokenizer_for_language(
    language: str,
    available_tokenizers: List[str],
    default_tokenizer: str = "english_phoneme",
    language_tokenizer_map: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
) -> str:
    """Get the appropriate tokenizer name for a language.

    Searches language_tokenizer_map for candidate tokenizers and returns
    the first one available. Falls back to default if no match found.

    Args:
        language: Language code (e.g., "en", "de", "zh").
        available_tokenizers: List of tokenizer names available in the model.
        default_tokenizer: Fallback tokenizer if no match found.
        language_tokenizer_map: Mapping of languages to a tokenizer or ordered tokenizer candidates.
            Defaults to ``LANGUAGE_TOKENIZER_MAP``.

    Returns:
        Tokenizer name to use.
    """
    if language_tokenizer_map is None:
        language_tokenizer_map = LANGUAGE_TOKENIZER_MAP
    if language in language_tokenizer_map:
        candidates = language_tokenizer_map[language]
        if isinstance(candidates, str):
            candidates = [candidates]
        for candidate in candidates:
            if candidate in available_tokenizers:
                return candidate

    # Fallback to default if available, else first available
    if default_tokenizer in available_tokenizers:
        return default_tokenizer
    return available_tokenizers[0] if available_tokenizers else default_tokenizer


def chunk_text_for_inference(
    text: str,
    language: str,
    tokenizer_name: str,
    text_tokenizer: Any,
    eos_token_id: int,
    language_thresholds: Optional[LanguageThresholds] = None,
    enable_phoneme_text_input: bool = False,
    phoneme_tokenizer=None,
    text_phoneme_token_offset: Optional[int] = None,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    Unified text chunking for inference: returns single chunk if below threshold,
    multiple sentence chunks if above threshold.

    This function unifies the standard and chunked inference paths by automatically
    determining whether to split text based on language-specific thresholds.

    Args:
        text: Input text to tokenize and potentially split.
        language: Language code (e.g., "en", "de", "zh").
        tokenizer_name: Name of the tokenizer to use (e.g., "english_phoneme").
        text_tokenizer: The tokenizer instance.
        eos_token_id: End-of-sequence token ID to append.
        language_thresholds: Optional custom thresholds. Uses defaults if None.
        enable_phoneme_text_input: Whether to parse and separately encode marked
            phoneme spans.
        phoneme_tokenizer: Tokenizer used for marked phoneme spans. Required
            when ``enable_phoneme_text_input`` is enabled.
        text_phoneme_token_offset: Offset added to phoneme token IDs in the
            text-channel vocabulary.
        bop_marker: Marker that opens an inline phoneme span.
        eop_marker: Marker that closes an inline phoneme span. Text containing
            either span marker is kept as one chunk to preserve span boundaries.

    Returns:
        Tuple of:
            - chunked_tokens: List of token tensors. Single element for short text,
              multiple elements (one per sentence) for long text.
            - chunked_tokens_len: List of token lengths.
            - chunked_text: List of text strings (original or split sentences).

    Examples:
        >>> # Short text - returns single chunk
        >>> tokens, lens, texts = chunk_text_for_inference(
        ...     "Hello world.", "en", "english_phoneme", tokenizer, eos_id
        ... )
        >>> len(tokens)
        1

        >>> # Long text - returns multiple chunks (sentences)
        >>> long_text = "First sentence. " * 50  # ~50 sentences
        >>> tokens, lens, texts = chunk_text_for_inference(
        ...     long_text, "en", "english_phoneme", tokenizer, eos_id
        ... )
        >>> len(tokens) > 1
        True
    """
    if language_thresholds is None:
        language_thresholds = DEFAULT_LANGUAGE_THRESHOLDS

    # Check if text exceeds threshold for this language
    has_explicit_phoneme_spans = has_phoneme_text_spans(text, bop_marker=bop_marker, eop_marker=eop_marker)
    should_split = language_thresholds.exceeds_threshold(text, language) and not has_explicit_phoneme_spans

    if should_split:
        # Long text: split by sentences
        return chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name=tokenizer_name,
            text_tokenizer=text_tokenizer,
            eos_token_id=eos_token_id,
            language=language,
            enable_phoneme_text_input=enable_phoneme_text_input,
            phoneme_tokenizer=phoneme_tokenizer,
            text_phoneme_token_offset=text_phoneme_token_offset,
            bop_marker=bop_marker,
            eop_marker=eop_marker,
        )
    else:
        # Short text: return as single chunk
        tokens = tokenize_text_with_phoneme_spans(
            text_tokenizer=text_tokenizer,
            text_str=text,
            tokenizer_name=tokenizer_name,
            enable_phoneme_text_input=enable_phoneme_text_input,
            phoneme_tokenizer=phoneme_tokenizer,
            text_phoneme_token_offset=text_phoneme_token_offset,
            bop_marker=bop_marker,
            eop_marker=eop_marker,
        )
        tokens = tokens + [eos_token_id]
        tokens_tensor = torch.tensor(tokens, dtype=torch.int32)
        tokens_len = tokens_tensor.shape[0]

        return [tokens_tensor], [tokens_len], [text]


def resample_batch(audio, audio_len, input_sample_rate, output_sample_rate):
    audio = resample(waveform=audio, orig_freq=input_sample_rate, new_freq=output_sample_rate)
    audio_len_scaled = audio_len.long() * output_sample_rate
    new_audio_len = audio_len_scaled / input_sample_rate
    # To avoid rounding issues at lower precisions, do not call torch.ceil when the length is divisible by the sample rate
    audio_len = torch.where(audio_len_scaled % input_sample_rate == 0, new_audio_len, torch.ceil(new_audio_len))
    audio_len = audio_len.int()
    audio = mask_sequence_tensor(audio, audio_len)
    return audio, audio_len


def normalize_text_by_pattern(text: str, pattern: str, replacement: str) -> str:
    """Normalize input text using a regular expression.

    This function will search for and replace any string matching the input 'pattern' surrounded by any punctuation.

    Example:
        >>> text = "Mr holmes told mr. watson to be careful."
        >>> normalized_text = normalize_text_by_pattern(text=text, pattern="[mM]r\.?", replacement="mister")
        >>> normalized_text
        "mister holmes told mister watson to be careful."

    Args:
        text: Text to normalize
        pattern: Text pattern to find and replace
        replacement: Text to substitute for the input pattern

    Returns:
        The normalized text string

    """
    # Finds all occurrences of the pattern, capturing any punctuation before and after it (including start and end of sentence).
    regex = re.compile(f"(^|[^A-Za-zÀ-ÖØ-öø-ÿ]){pattern}($|[^A-Za-zÀ-ÖØ-öø-ÿ])")
    match = regex.findall(string=text)

    output = text
    for surrounding_punct in match:
        repl = f"{surrounding_punct[0]}{replacement}{surrounding_punct[1]}"
        output = regex.sub(string=output, repl=repl, count=1)

    return output


class TextProcessor(ABC):
    """Interface for preprocessing text for TTS training, inference, and evaluation"""

    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """
        Preprocess text for model training and inference.
            Usually this involves language-specific rules for converting written text to spoken form.

        Args:
            text: Raw text string

        Returns:
            Normalized text string
        """
        pass

    @abstractmethod
    def process_text_for_wer(self, text: str) -> str:
        """
        Preprocess text for calculating word error rate and character error rate.
            This should include conversion of text to lower case, removal of punctuation, normalization,
            and possible post-processing steps for ASR model output.

        Args:
            text: Raw text string

        Returns:
            Processed text string
        """
        pass


class DefaultTextProcessor(TextProcessor):
    """Default text processing behavior, if language-specific processing is not yet implemented."""

    def normalize_text(self, text: str) -> str:
        return text

    def process_text_for_wer(self, text: str) -> str:
        text = text.lower()
        # Replace dash with a single space
        text = text.replace("-", " ")
        # Replace whitespace with a single space
        text = re.sub(pattern=r"\s\s+", string=text, repl=" ")
        # Remove all non-alphanumeric characters, making sure to keep accented and foreign characters
        text = "".join([c for c in text if c == " " or c.isalnum()])
        # Fix common ASR transcript artifacts
        text = text.replace("h t t p", "http")
        text = text.replace("w w w", "www")
        text = text.strip()
        return text


class NoSpaceTextProcessor(TextProcessor):
    """WER text processor for languages where ASR/tokenization spaces should be ignored."""

    def __init__(self):
        super().__init__()
        self.default_processor = DefaultTextProcessor()

    def normalize_text(self, text: str) -> str:
        return text

    def process_text_for_wer(self, text: str) -> str:
        text = self.default_processor.process_text_for_wer(text)
        text = text.replace(" ", "")
        return text


class JapaneseTextProcessor(NoSpaceTextProcessor):
    """Japanese WER text processor with Katakana reading conversion."""

    def __init__(self):
        super().__init__()
        try:
            self.pyopenjtalk = importlib.import_module("pyopenjtalk")
        except ImportError as e:
            raise ImportError(
                "JapaneseTextProcessor requires pyopenjtalk for Katakana CER computation. "
                "Install pyopenjtalk or do not request Japanese evaluation text processing."
            ) from e

    def text_to_katakana(self, text: str) -> str:
        """Convert Japanese text to its Katakana reading via pyopenjtalk.

        Used for an additional, reading-based Japanese CER metric that is robust to
        kanji/kana spelling variation between the reference and the ASR hypothesis.
        """
        if not text:
            return ""
        try:
            return self.pyopenjtalk.g2p(text, kana=True).strip()
        except Exception as e:  # noqa: BLE001
            logging.warning(f"pyopenjtalk failed for '{text[:40]}': {e}")
            return ""


class EnglishTextProcessor(TextProcessor):
    """English text processing, which catches some edge cases not covered by normal text normalization.

    English TN does not work on abbreviations when a period is missing. For example, "mr." will be normalized to "mister",
    but "mr" will not be. This class manually normalizes abbreviations commonly found in public datasets and ASR transcriptions.
    """

    def __init__(self, input_case: str = "cased"):
        super().__init__()
        self.default_processor = DefaultTextProcessor()

        if not PYNINI_AVAILABLE:
            logging.warning("`nemo_text_processing` is not installed, will skip default text normalization")
            self.normalizer = None
        else:
            self.normalizer = Normalizer(lang="en", input_case=input_case)

    def normalize_text(self, text: str) -> str:
        if self.normalizer is not None:
            text = self.normalizer.normalize(text)

        text = normalize_text_by_pattern(text=text, pattern="[mM]r\.?", replacement="mister")
        text = normalize_text_by_pattern(text=text, pattern="[mM]s\.?", replacement="miss")
        text = normalize_text_by_pattern(text=text, pattern="[mM]rs\.?", replacement="missus")
        text = normalize_text_by_pattern(text=text, pattern="[mM]me\.?", replacement="madame")
        text = normalize_text_by_pattern(text=text, pattern="[dD]r\.?", replacement="doctor")
        text = normalize_text_by_pattern(text=text, pattern="[eE]tc\.?", replacement="et cetera")
        return text

    def process_text_for_wer(self, text: str) -> str:
        text = self.normalize_text(text)
        text = self.default_processor.process_text_for_wer(text)
        return text


def get_text_processor(language: str) -> TextProcessor:
    language = (language or "").replace("_", "-").lower().split("-")[0]
    if language == "en":
        return EnglishTextProcessor()
    if language == "ja":
        return JapaneseTextProcessor()
    elif language == "zh":
        return NoSpaceTextProcessor()
    else:
        logging.info(f"Text processing not implemented for language {language}; using default processor")
        return DefaultTextProcessor()


class Transcriber(ABC):
    """Interface for transcribing TTS outputs with different ASR models"""

    @abstractmethod
    def transcribe(self, audio_paths: List[Path], batch_size: int, language: Optional[str]) -> List[str]:
        """
        Run batch transcription of a list of audio files.

        Args:
            audio_paths: list of paths to audio files to transcribe
            batch_size: batch size to use for inference
            language: optional language of input audio

        Returns:
            List with transcribed text for each audio path
        """
        pass


class NemoTranscriber(Transcriber):
    """Transcriber for NeMo ASR models"""

    def __init__(self, device, model_name="stt_en_conformer_transducer_large"):
        if model_name.endswith('.nemo'):
            model = ASRModel.restore_from(restore_path=model_name)
        else:
            model = ASRModel.from_pretrained(model_name=model_name)

        self.model = model.to(device).eval()

    def transcribe(self, audio_paths: List[Path], batch_size: int, language: Optional[str]) -> List[str]:
        override_config = TranscribeConfig(batch_size=batch_size, use_lhotse=False)
        transcribe_results = self.model.transcribe(audio_paths, override_config=override_config)
        transcriptions = [result.text for result in transcribe_results]
        return transcriptions


class NemoTranscriberWithPrompt(Transcriber):
    """Transcriber for NeMo ASR models that accept language as an optional prompt"""

    def __init__(self, device, model_name):
        if model_name.endswith('.nemo'):
            model = ASRModel.restore_from(restore_path=model_name)
        else:
            model = ASRModel.from_pretrained(model_name=model_name)

        self.model = model.to(device).eval()
        if not isinstance(self.model, EncDecHybridRNNTCTCBPEModelWithPrompt):
            raise ValueError(f"Model {model_name} does not support prompting")

        self.language_prompt_map = {
            "en": "en-US",
            "ar": "ar",
            "ko": "ko-KR",
            "hi": "hi-IN",
            "zh": "zh-CN",
            "it": "it-IT",
            "es": "es-ES",
            "de": "de-DE",
            "fr": "fr-FR",
            "ja": "ja-JP",
        }

    def transcribe(self, audio_paths: List[Path], batch_size: int, language: Optional[str]) -> List[str]:
        if language:
            prompt_lang = self.language_prompt_map[language]
            override_config = HybridRNNTCTCPromptTranscribeConfig(
                batch_size=batch_size, use_lhotse=False, target_lang=prompt_lang
            )
        else:
            override_config = HybridRNNTCTCPromptTranscribeConfig(batch_size=batch_size, use_lhotse=False)

        transcribe_results = self.model.transcribe(audio_paths, override_config=override_config)
        transcriptions = [result.text for result in transcribe_results]
        return transcriptions


class WhisperTranscriber(Transcriber):
    """Transcriber for Whisper ASR models"""

    def __init__(self, device, model_name="openai/whisper-large-v3"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        self.input_sample_rate = 16000

    def transcribe(self, audio_paths: List[Path], language: str, batch_size: int) -> List[str]:
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
        else:
            forced_decoder_ids = None

        all_transcriptions = []
        for start in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[start : start + batch_size]
            speech_arrays = [librosa.load(p, sr=self.input_sample_rate)[0] for p in batch_paths]
            inputs = self.processor(
                speech_arrays, sampling_rate=self.input_sample_rate, return_tensors="pt", padding=True
            ).input_features
            inputs = inputs.half().to(self.model.device)
            with torch.inference_mode():
                predicted_ids = self.model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
            transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            all_transcriptions.extend(transcriptions)
        return all_transcriptions
