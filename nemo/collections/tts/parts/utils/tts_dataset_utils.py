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
import logging
import os
import random
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from einops import rearrange
from scipy import ndimage
from torch.special import gammaln

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
        tokens = text_tokenizer.encode(text=sentence, tokenizer_name=tokenizer_name)
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
) -> str:
    """Get the appropriate tokenizer name for a language.

    Searches LANGUAGE_TOKENIZER_MAP for candidate tokenizers and returns
    the first one available. Falls back to default if no match found.

    Args:
        language: Language code (e.g., "en", "de", "zh").
        available_tokenizers: List of tokenizer names available in the model.
        default_tokenizer: Fallback tokenizer if no match found.

    Returns:
        Tokenizer name to use.
    """
    if language in LANGUAGE_TOKENIZER_MAP:
        for candidate in LANGUAGE_TOKENIZER_MAP[language]:
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
    should_split = language_thresholds.exceeds_threshold(text, language)

    if should_split:
        # Long text: split by sentences
        return chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name=tokenizer_name,
            text_tokenizer=text_tokenizer,
            eos_token_id=eos_token_id,
            language=language,
        )
    else:
        # Short text: return as single chunk
        tokens = text_tokenizer.encode(text=text, tokenizer_name=tokenizer_name)
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
    if language == "en":
        return EnglishTextProcessor()
    else:
        logging.info(f"Text processing not implemented for language {language}; using default processor")
        return DefaultTextProcessor()
