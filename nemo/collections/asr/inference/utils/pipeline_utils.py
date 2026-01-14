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


import re
from functools import partial, wraps
from typing import Iterable

import torch
from omegaconf import DictConfig, open_dict
from torch import Tensor

from nemo.collections.asr.inference.utils.constants import BIG_EPSILON, SENTENCEPIECE_UNDERSCORE, SMALL_EPSILON
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    get_confidence_aggregation_bank,
    get_confidence_measure_bank,
)
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def check_existance_of_required_attributes(obj: object, required_args: list[str]) -> None:
    """
    Check if the required attributes exist in the object
    Args:
        obj: (object) Object to check the attributes of
        required_args: (list[str]) List of required attributes
    """
    not_found_args = []
    for arg in required_args:
        if not hasattr(obj, arg):
            not_found_args.append(arg)
    if not_found_args:
        raise ValueError(f"Required attributes not found: {not_found_args}")


def normalize_features(features: Tensor, feature_lens: Tensor = None) -> Tensor:
    """Normalize the features.
    Args:
        features: (Tensor) features. Shape is torch.Size([B, C, T]).
        feature_lens: (Tensor) feature lengths. Shape is torch.Size([B]).
    Returns:
        (Tensor) normalized features. Shape is torch.Size([B, C, T]).
    """
    return normalize_batch(features, feature_lens, "per_feature")[0]


def ids_to_text_without_stripping(tokens: list[int], tokenizer: TokenizerSpec, sep: str = ' ') -> str:
    """
    Convert a list of token IDs to text without stripping.
    Args:
        tokens: (list[int]) List of token IDs.
        tokenizer: (TokenizerSpec) Tokenizer.
        sep: (str) Separator between words. Default is ' '.
    Returns:
        (str) Text.
    """
    pieces = tokenizer.ids_to_tokens(tokens)
    text = "".join(
        [(p.replace(SENTENCEPIECE_UNDERSCORE, sep) if p.startswith(SENTENCEPIECE_UNDERSCORE) else p) for p in pieces]
    )
    return text


def memoize_normalization_mode():
    """
    Decorator to memoize the normalization mode.
    In the first call, the normalization mode is detected and cached.
    In the subsequent calls, the cached normalization mode is used.
    """

    def decorator(func):
        mode = None  # Cache the detected format

        @wraps(func)
        def wrapper(log_probs: torch.Tensor) -> torch.Tensor:
            nonlocal mode

            if mode is None:
                ONE = torch.tensor(1.0, dtype=log_probs.dtype)
                if torch.allclose(log_probs[0][0].sum(), ONE, atol=BIG_EPSILON):
                    # assume that softmax is already applied
                    mode = 'prob'
                else:
                    if not torch.allclose(log_probs[0][0].exp().sum(), ONE, atol=BIG_EPSILON):
                        # It's neither prob nor log-softmax, need to apply log_softmax
                        mode = "logits"
                    else:
                        # It's already in log-softmax form
                        mode = "log_softmax"

            # Fast-path execution
            if mode == "prob":
                return torch.log(log_probs + SMALL_EPSILON)
            elif mode == 'logits':
                return torch.log_softmax(log_probs, dim=-1)
            else:
                return log_probs

        return wrapper

    return decorator


@memoize_normalization_mode()
def normalize_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """
    log_probs: (B, T, vocab_size) log probabilities
    Returns:
        (Tensor) normalized log probabilities. Shape is torch.Size([B, T, vocab_size]).
    """
    # Ensure log_probs are normalized
    return log_probs


def drop_trailing_features(features: Tensor, expected_feature_buffer_len: int) -> Tensor:
    """Drop the trailing features if the number of features is greater than the expected feature buffer length.
    Args:
        features: (Tensor) features. Shape is torch.Size([B, C, T1]).
        expected_feature_buffer_len: (int) Expected feature buffer length.
    Returns:
        (Tensor) features. Shape is torch.Size([B, C, T2]).
    """
    if features.shape[2] > expected_feature_buffer_len:
        features = features[:, :, :expected_feature_buffer_len]
    return features


def make_preprocessor_deterministic(asr_model_cfg: DictConfig, disable_normalization: bool = True) -> DictConfig:
    """
    Make the preprocessor deterministic by disabling normalization, dither and padding
    Args:
        asr_model_cfg: (DictConfig) ASR model configuration.
        disable_normalization: (bool) Whether to disable normalization. Default is True.
    Returns:
        (DictConfig) ASR model configuration with deterministic preprocessor.
    """
    # Enable config overwriting
    with open_dict(asr_model_cfg):
        # Normalization will be done per buffer in frame_bufferer
        # Do not normalize whatever the model's preprocessor setting is
        asr_model_cfg.preprocessor.dither = 0.0
        asr_model_cfg.preprocessor.pad_to = 0

        if disable_normalization:
            asr_model_cfg.preprocessor.normalize = "None"

    return asr_model_cfg


def get_confidence_utils(confidence_cfg: DictConfig) -> tuple:
    """
    Get the confidence function and the confidence aggregator
    Args:
        confidence_cfg: (DictConfig) Confidence configuration.
    Returns:
        (tuple) Confidence function and the confidence aggregator.
    """
    if confidence_cfg.method_cfg.name == "max_prob":
        conf_type = "max_prob"
        conf_alpha = 1.0
    else:
        conf_type = f"entropy_{confidence_cfg.method_cfg.entropy_type}_{confidence_cfg.method_cfg.entropy_norm}"
        conf_alpha = confidence_cfg.method_cfg.alpha

    conf_func = get_confidence_measure_bank()[conf_type]
    conf_func = partial(conf_func, t=conf_alpha)
    confidence_aggregator = get_confidence_aggregation_bank()[confidence_cfg.aggregation]
    return conf_func, confidence_aggregator


def get_leading_punctuation_regex_pattern(puncts: set[str]) -> str:
    """
    Get the regex pattern for the punctuation marks.
    Args:
        puncts (set[str]): Set of punctuation marks.
    Returns:
        (str) Regex pattern for the punctuation marks.
    """
    if not puncts:
        return ""
    escaped_puncts = '|'.join(re.escape(punct) for punct in puncts)
    return r'\s+(' + escaped_puncts + ')'


def get_repeated_punctuation_regex_pattern(puncts: set[str]) -> str:
    """
    Get the regex pattern for the repeated punctuation marks.
    Args:
        puncts (set[str]): Set of punctuation marks.
    Returns:
        (str) Regex pattern for the repeated punctuation marks.
    """
    if not puncts:
        return ""
    escaped_puncts = ''.join(re.escape(p) for p in puncts)
    return r'([' + escaped_puncts + r']){2,}'


def update_punctuation_and_language_tokens_timestamps(
    tokens: Tensor, timestamp: Tensor, tokens_to_move: set[int], underscore_id: int
) -> Tensor:
    """
    RNNT models predict punctuations and language tokens at the end of the sequence.
    Due to this, it appears as if there's a silence between the last word and the punctuation.
    This function moves the tokens close to preceding word in the list.
    Args:
        tokens: (Tensor) Tokens tensor.
        timestamp: (Tensor) Timestamps tensor.
        tokens_to_move: (set[int]) Set of tokens to move.
        underscore_id: (int) ID of the underscore token.
    Returns:
        (Tensor) Updated timestamps tensor.
    """

    n_tokens = tokens.shape[0]
    if n_tokens != timestamp.shape[0]:
        raise ValueError("Tokens and timestamps must have the same length")

    tokens_to_move_with_underscore = tokens_to_move.union({underscore_id})
    # If all tokens need moving, don't change timestamps (no content words to attach to)
    only_special_tokens = all(token.item() in tokens_to_move_with_underscore for token in tokens)
    if only_special_tokens:
        return timestamp

    groups = []
    i = 0
    while i < n_tokens:
        if tokens[i].item() in tokens_to_move_with_underscore:
            start_idx = i
            end_idx = i
            j = i + 1
            while j < n_tokens and (tokens[j].item() in tokens_to_move_with_underscore):
                if tokens[j].item() != underscore_id:
                    end_idx = j
                j += 1
            if j > start_idx and end_idx >= start_idx:
                left_timestamp = int(timestamp[start_idx - 1]) if start_idx > 0 else 0
                if start_idx == end_idx:
                    if tokens[start_idx].item() in tokens_to_move:
                        groups.append((start_idx, end_idx + 1, left_timestamp))
                else:
                    groups.append((start_idx, end_idx + 1, left_timestamp))
            i = j
        else:
            i += 1

    updated_timestamps = timestamp.clone()
    for start_idx, end_idx, left_timestamp in groups:
        for k in range(start_idx, end_idx):
            # Give all tokens_to_move the same timestamp as the preceding word
            updated_timestamps[k] = left_timestamp

    return updated_timestamps


def adjust_vad_segments(vad_segments: Tensor, left_padding_size: float) -> Tensor | None:
    """
    Adjust VAD segments for stateful mode by subtracting left_padding and applying clipping rules.
    Args:
        vad_segments: (Tensor) VAD segments tensor with shape [num_segments, 2] (start_time, end_time)
        left_padding_size: (float) Amount of left padding in seconds to subtract from segments
    Returns:
        (Tensor | None) Adjusted VAD segments tensor or None if no valid segments are left.
    """
    if vad_segments is None or len(vad_segments) == 0:
        return vad_segments

    # Vectorized operations on the entire tensor
    adjusted_segments = vad_segments - left_padding_size

    # Filter out segments that end before or at 0
    valid_mask = adjusted_segments[:, 1] > 0

    if not valid_mask.any():
        return None

    adjusted_segments = adjusted_segments[valid_mask]

    # Clip start times to 0
    adjusted_segments[:, 0] = torch.clamp(adjusted_segments[:, 0], min=0.0)

    return adjusted_segments


def seconds_to_frames(seconds: float | int | Iterable[float | int], model_stride_in_secs: float) -> int | list[int]:
    """
    Convert seconds to frames.
    Args:
        seconds: (float | int | Iterable[float | int]) Time in seconds
        model_stride_in_secs: (float) Stride of the model in seconds
    Returns:
        (int | list[int]) Number of frames
    """
    if isinstance(seconds, (float, int)):
        return int(seconds / model_stride_in_secs)

    if isinstance(seconds, Iterable):
        return [int(s / model_stride_in_secs) for s in seconds]

    raise ValueError(f"Invalid type for seconds: {type(seconds)}")
