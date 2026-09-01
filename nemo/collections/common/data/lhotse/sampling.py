# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
# pylint: disable=C0116
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from lhotse.cut import Cut, MonoCut
from lhotse.dataset import SamplingConstraint, TokenConstraint
from lhotse.dataset.sampling.dynamic_bucketing import FixedBucketBatchSizeConstraint
from lhotse.utils import ifnone

from nemo.collections.common.data.lhotse.audio_token_estimator import AudioTokenEstimator
from nemo.collections.common.data.lhotse.text_adapters import (
    Formattable,
    NeMoMultimodalConversation,
    measure_formattable_length,
)


@dataclass
class MultimodalSamplingConstraint(SamplingConstraint):
    """
    Sampling strategy that customizes Lhotse samplers to measure sequence lengths as token counts.
    It provides a unified interface for audio and text examples - audio duration is converted to
    an equivalent token count.
    """

    # How many seconds of audio is a text token worth; balances audio to text ratio in a mini-batch.
    # Generally set this to frame_shift * total_subsampling_factor of your audio encoder.
    token_equivalent_duration: float | None = None

    # Optional sample-exact description of the audio preprocessor, encoder
    # subsampling, and SALM encoder chunking policy. This supersedes the
    # duration-based approximation above when provided.
    audio_token_estimator: AudioTokenEstimator | None = None

    # Defines maximum batch size (may be lower than that if batch_length is also specified).
    batch_size: int | None = None

    # Defines the total number of tokens in a mini-batch.
    # Setting this enables dynamic batch sizes.
    # We will use ``token_equivalent_duration`` to convert audio examples to token sizes.
    batch_tokens: int | None = None

    # When specified, this value is inversely proportional to the penalty we assign
    # to longer examples when measuring their length/duration;
    # i.e. large quadratic factor is a small penalty, small quadratic factor is a large penalty.
    # Tweaking this helps equalize the GPU memory usage for dynamic batch sizes when using bucketing.
    quadratic_factor: float | None = None

    # Packed encoder batches allocate and compute by the sum of example lengths
    # rather than batch_size * longest_length. Opt in to padding-free accounting
    # so heterogeneous non-bucketing batches can fill batch_tokens.
    use_packed_sequence_sampling: bool = False

    # When False (default), we only consider the input part of the example to determine its length,
    # e.g. for a Cut that means its audio duration converted to tokens, for text that means len(context_ids), etc.
    # When True, we consider the sum of input and output lengths together (useful mostly for decoder-only models).
    measure_total_length: bool = False

    _internal = None

    def __post_init__(self):
        if self.use_packed_sequence_sampling:
            self._internal = PackedTokenConstraint(
                batch_tokens=self.batch_tokens,
                max_examples=self.batch_size,
                quadratic_length=self.quadratic_factor,
            )
        else:
            self._internal = TokenConstraint(
                max_tokens=self.batch_tokens,
                max_examples=self.batch_size,
                quadratic_length=self.quadratic_factor,
            )

    def add(self, example: Any) -> None:
        num_tokens = self.measure_length(example)
        example.num_tokens = num_tokens
        self._internal.add(example)

    def exceeded(self) -> bool:
        return self._internal.exceeded()

    def close_to_exceeding(self) -> bool:
        return self._internal.close_to_exceeding()

    def would_exceed(self, example: Any) -> bool:
        """Return whether adding ``example`` would exceed a packed batch limit."""
        if not self.use_packed_sequence_sampling:
            raise RuntimeError("would_exceed() is only valid for packed sequence sampling")
        num_tokens = self.measure_length(example)
        if self._internal.max_examples is not None and self._internal.num_examples + 1 > self._internal.max_examples:
            return True
        return (
            self._internal.batch_tokens is not None
            and self._internal.current + self._internal.budget_length(num_tokens) > self._internal.batch_tokens
        )

    def reached_limit(self) -> bool:
        """Return whether the current packed batch exactly reached a limit."""
        if not self.use_packed_sequence_sampling:
            raise RuntimeError("reached_limit() is only valid for packed sequence sampling")
        if self._internal.max_examples is not None and self._internal.num_examples >= self._internal.max_examples:
            return True
        return self._internal.batch_tokens is not None and self._internal.current >= self._internal.batch_tokens

    def measure_packing_length(self, example: Any) -> int:
        """Measure one example against the packed batch's effective token budget."""
        if not self.use_packed_sequence_sampling:
            raise RuntimeError("measure_packing_length() is only valid for packed sequence sampling")
        return self._internal.budget_length(self.measure_length(example))

    def reset(self) -> None:
        self._internal.reset()

    def measure_length(self, example: Any) -> float:
        if isinstance(example, Cut):
            audio_len_in_tokens = self._measure_audio(example)
            if self.measure_total_length:
                # Total length of a Cut (audio+text example) is counted as the sum of:
                # * num_tokens in each supervision segment ("utterance") in the Cut
                # * num_frames of audio (frame=token) given a token-equivalent-duration (basically a frame shift)
                text_tokens = 0
                for s in example.supervisions:
                    if s.has_custom("tokens"):
                        text_tokens += len(s.tokens)
                return audio_len_in_tokens + text_tokens
            else:
                return audio_len_in_tokens
        elif isinstance(example, Formattable):
            try:
                if (
                    self.use_packed_sequence_sampling
                    and isinstance(example, NeMoMultimodalConversation)
                    and example.has_audio_turns
                    and self.audio_token_estimator is None
                ):
                    raise ValueError(
                        "Exact packed sequence sampling with audio requires audio_token_estimator metadata that "
                        "matches the model's preprocessor, subsampling, and encoder chunking configuration. "
                        "token_equivalent_duration is only an approximation and cannot enforce a hard model-token cap."
                    )
                mode = "total" if self.measure_total_length else "input"
                return measure_formattable_length(
                    example,
                    mode,
                    audio_token_estimator=self.audio_token_estimator,
                )
            except (AttributeError, AssertionError) as e:
                raise RuntimeError(
                    "Couldn't determine the length of a text example; "
                    "have you provided both prompt_format and tokenizer when instantiating the dataloader?"
                ) from e
        raise RuntimeError(f"Unsupported example type: {type(example)}")

    def _measure_audio(self, example: Cut) -> int:
        if self.audio_token_estimator is not None:
            return self.audio_token_estimator.estimate_cut(example)
        if self.use_packed_sequence_sampling:
            raise ValueError(
                "Exact packed sequence sampling with audio requires audio_token_estimator metadata that matches "
                "the model's preprocessor, subsampling, and encoder chunking configuration. "
                "token_equivalent_duration is only an approximation and cannot enforce a hard model-token cap."
            )
        return math.ceil(example.duration / self.token_equivalent_duration)


@dataclass
class PackedTokenConstraint(SamplingConstraint):
    """Token constraint for batches that remain packed through the model.

    Generic TokenConstraint budgets padded work as num_examples times the
    longest example. For THD/packed execution, useful work and activation
    storage instead scale with the sum of per-example lengths. ``batch_tokens``
    remains a hard raw-token cap, while the optional example-count and
    quadratic-compute limits preserve the public ``batch_size`` and
    ``quadratic_factor`` configuration semantics. Candidate-aware batching
    enforces all configured limits; the current mean remains only as an
    end-of-stream fullness heuristic for drop-last behavior.
    """

    batch_tokens: int | None = None
    max_examples: int | None = None
    current: int = 0
    num_examples: int = 0
    quadratic_length: float | None = None

    def __post_init__(self) -> None:
        if self.batch_tokens is not None and self.batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive or null (got {self.batch_tokens})")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError(f"batch_size must be positive or null (got {self.max_examples})")
        if self.quadratic_length is not None and self.quadratic_length <= 0:
            raise ValueError(f"quadratic_factor must be positive or null (got {self.quadratic_length})")

    def budget_length(self, size: float) -> int:
        """Return the conservative integral cost used by exact subset packing."""
        if self.quadratic_length is None:
            return math.ceil(size)
        return math.ceil(size + size**2 / self.quadratic_length)

    def add(self, example: Any) -> None:
        self.current += self.budget_length(self.measure_length(example))
        self.num_examples += 1

    def exceeded(self) -> bool:
        if self.max_examples is not None and self.num_examples > self.max_examples:
            return True
        return self.batch_tokens is not None and self.current > self.batch_tokens

    def close_to_exceeding(self) -> bool:
        if self.max_examples is not None and self.num_examples >= self.max_examples:
            return True
        if self.batch_tokens is None:
            return False
        if self.num_examples == 0:
            return False
        mean_length = self.current / self.num_examples
        return self.current + mean_length > self.batch_tokens

    def reset(self) -> None:
        self.current = 0
        self.num_examples = 0

    def measure_length(self, example: Any) -> float:
        return example.num_tokens


@dataclass
class FixedBucketBatchSizeConstraint2D(FixedBucketBatchSizeConstraint):
    """
    Sampling strategy that customizes Lhotse samplers to support 2D bucket selection (it also supports 1D).
    It is intended only for audio examples (i.e., Lhotse Cut objects).

    When ``strict_2d`` is set, we only consider sub-buckets for a single bucket that is the best match.
    When set to ``False``, we'll promote an example to buckets with larger 1st dim if they can accommodate the 2nd dim.

    When ``max_ratio`` is set, it discards the examples that exceed a specific output-to-input length ratio.
    ``max_ratio`` must be a list with the same length as the number of buckets.
    ``max_ratio`` is only applied when ``strict_2d`` is set to ``True``.
    """

    strict_2d: bool = True
    max_ratio: list[float] | None = None

    def __post_init__(self):
        if isinstance(self.max_seq_len_buckets[0], Sequence):
            self.max_seq_len_buckets = np.asarray(self.max_seq_len_buckets)
        if self.max_ratio is not None:
            assert isinstance(self.max_ratio, Sequence), f"self.max_ratio must be a list, but we got: {self.max_ratio}"
            assert len(self.max_ratio) == len(
                self.max_seq_len_buckets
            ), f"{len(self.max_ratio)=} != {len(self.max_seq_len_buckets)=}"

    @property
    def bucketing_2d_enabled(self) -> bool:
        return isinstance(self.max_seq_len_buckets, np.ndarray)

    def measure_length(self, example: Cut) -> tuple[float, float] | float:
        if self.bucketing_2d_enabled:
            return example.duration, _measure_tokens(example)
        else:
            return example.duration

    def select_bucket(self, buckets: Any, example: Any = None, example_len: Any = None) -> int:
        if example_len is None:
            example_len = self.measure_length(example)
        return find_smallest_bucket(
            self.max_seq_len_buckets,
            example_len,
            strict=self.strict_2d,
            max_ratio=self.max_ratio,
        )


def find_smallest_bucket(
    buckets: np.ndarray,
    example_lens: float | Sequence[float],
    strict: bool = True,
    max_ratio: Sequence[float] | None = None,
) -> int | None:
    """
    Find the smallest bucket that fits a given example.
    Each bucket and ``example_lens`` are floats (1-D bucketing)
    or tuples of (dim0, dim1, dim2, ...) (N-D bucketing, typically 2-D).
    Assumes the buckets have been sorted ascendingly.
    Returns a tuple of (smallest_bin, bin_idx), or (None, None) if no bucket fits the example.
    """
    # 1D bucketing - binary search.
    if isinstance(example_lens, (float, int)):  # 1-D
        idx = bisect_left(buckets, example_lens)
        if idx == len(buckets):
            return None
        return idx

    # 2D bucketing 'strict' mode: only consider sub-buckets for the specific bucket that matches this example.
    # E.g. for buckets = [(10, 5), (10, 10), (20, 12), (20, 18)]
    #      and example_lens = (8, 11)
    #      we will return None because we only consider the first two buckets based on dim0 (=8).
    if strict:
        # Find the first 2D bucket that accepts this example
        dim0_begin = bisect_left(buckets[:, 0], example_lens[0])
        if dim0_begin == buckets.shape[0]:
            return None
        # Find the last 2D bucket that accepts this example
        dim0_end = dim0_begin
        while dim0_end < buckets.shape[0] and buckets[dim0_end, 0] == buckets[dim0_begin, 0]:
            dim0_end += 1
        # Find the smallest 2D bucket in this range that accepts this example
        dim1_begin = bisect_left(buckets[dim0_begin:dim0_end, 1], example_lens[1])
        if dim1_begin == dim0_end - dim0_begin:
            return None
        fit_idx = dim0_begin + dim1_begin
        # Apply max_ratio (token-per-second/token-per-token) filtering if requested
        if max_ratio is not None and example_lens[1] / example_lens[0] > max_ratio[fit_idx]:
            return None
        return fit_idx

    # 2D bucketing 'lenient' mode - linear search (as 2nd dim may not be growing monotonically).
    # E.g. for buckets = [(10, 5), (10, 10), (20, 12), (20, 18)]
    #      and example_lens = (8, 11)
    #      we will return bucket_idx=2 because (20, 12) fits (8, 11) at the cost of more padding.
    does_fit = np.all(np.asarray(example_lens) <= buckets, axis=1)
    min_fit_idx = np.argmax(does_fit)
    if min_fit_idx or does_fit[min_fit_idx]:
        return min_fit_idx.item()
    else:
        return None


@dataclass
class MultimodalFixedBucketBatchSizeConstraint2D(FixedBucketBatchSizeConstraint2D):
    """
    Sampling strategy that customizes Lhotse samplers to support both multimodal sampling and 2D bucket selection.
    It combines the capabilities of :class:`FixedBucketBatchSizeConstraint2D` and :class:`MultimodalSamplingConstraint`
    """

    # How many seconds of audio is a text token worth; balances audio to text ratio in a mini-batch.
    # Generally set this to frame_shift * total_subsampling_factor of your audio encoder.
    token_equivalent_duration: float | None = None

    # Optional sample-exact audio length estimator. Bucketing remains backward
    # compatible with token_equivalent_duration when this is unset.
    audio_token_estimator: AudioTokenEstimator | None = None

    # When False (default), we only consider the input part of the example to determine its length,
    # e.g. for a Cut that means its audio duration converted to tokens, for text that means len(context_ids), etc.
    # When True, we consider the sum of input and output lengths together (useful mostly for decoder-only models).
    measure_total_length: bool = False

    def measure_length(self, example: Any) -> float | tuple[float, float]:
        if isinstance(example, Cut):
            # Total length of a Cut (audio+text example) is counted as the sum of:
            # * num_tokens in each supervision segment ("utterance") in the Cut
            # * num_frames of audio (frame=token) given a token-equivalent-duration (basically a frame shift)
            audio_len_in_tokens = (
                self.audio_token_estimator.estimate_cut(example)
                if self.audio_token_estimator is not None
                else math.ceil(example.duration / self.token_equivalent_duration)
            )
            text_tokens = _measure_tokens(example)

            if self.bucketing_2d_enabled:
                return audio_len_in_tokens, text_tokens

            else:
                if self.measure_total_length:
                    return audio_len_in_tokens + text_tokens
                else:
                    return audio_len_in_tokens

        elif isinstance(example, Formattable):
            if self.bucketing_2d_enabled:
                return (
                    measure_formattable_length(
                        example,
                        "input",
                        audio_token_estimator=self.audio_token_estimator,
                    ),
                    measure_formattable_length(
                        example,
                        "output",
                        audio_token_estimator=self.audio_token_estimator,
                    ),
                )
            else:
                return measure_formattable_length(
                    example,
                    "total" if self.measure_total_length else "input",
                    audio_token_estimator=self.audio_token_estimator,
                )

        raise RuntimeError(f"Unsupported example type: {type(example)}")


class DurationFilter:
    """
    Callable, returns ``True`` if a cut's duration is in range [d_min, d_max] and ``False`` otherwise.
    Acts as a pass-through for objects of other type than Cut.
    """

    def __init__(self, d_min: float | None, d_max: float | None) -> None:
        self.d_min = ifnone(d_min, -1)
        self.d_max = ifnone(d_max, float("inf"))

    def __call__(self, example) -> bool:
        if isinstance(example, Cut):
            return self.d_min <= example.duration <= self.d_max
        elif isinstance(example, NeMoMultimodalConversation):
            if example.is_text_only:
                return True  # does not apply to text
            tot_dur = sum(c.duration for c in example.list_cuts())
            return self.d_min <= tot_dur <= self.d_max
        else:
            return True  # does not apply to text etc.


class ValidationStatusFilter:
    """
    Callable, returns ``True`` if a cut's validation status is equal to keep and ``False`` otherwise.
    Acts as a pass-through for objects of other type than Cut.
    """

    def __init__(self, keep: str = "pass") -> None:
        self.keep = keep

    def __call__(self, example) -> bool:
        if (
            isinstance(example, MonoCut)
            and example.has_custom("validation_status")
            and example.validation_status != self.keep
        ):
            return False
        else:
            return True


class CERFilter:
    """
    Callable, returns ``True`` if a cut's CER is less than max_cer and ``False`` otherwise.
    Acts as a pass-through for objects of other type than Cut.
    """

    def __init__(self, max_cer: float | None) -> None:
        self.max_cer = ifnone(max_cer, float("inf"))

    def __call__(self, example) -> bool:
        if (
            isinstance(example, MonoCut)
            and len(example.supervisions) > 0
            and example.supervisions[0].has_custom("cer")
        ):
            return example.supervisions[0].cer <= self.max_cer
        else:
            return True


class SpeakerFilter:
    """
    Callable, returns ``False`` if any supervision in a cut belongs to an excluded speaker.
    Checks configured supervision attributes/custom fields.
    Acts as a pass-through for objects of other type than Cut.
    """

    def __init__(
        self,
        excluded_speaker_ids: Sequence[str] | None = None,
        speaker_fields: Sequence[str] = None,
    ) -> None:
        self.excluded_speaker_ids = set(ifnone(excluded_speaker_ids, ()))
        self.enabled = len(self.excluded_speaker_ids) > 0
        if self.enabled and speaker_fields is None:
            raise ValueError(
                "SpeakerFilter requires speaker_fields when excluded_speaker_ids is set. "
                "Example: speaker_filter_fields=[speaker_id]"
            )
        self.speaker_fields = tuple(ifnone(speaker_fields, ()))

    def __call__(self, example) -> bool:
        if not self.enabled or not isinstance(example, Cut):
            return True

        excluded_speaker_ids = self.excluded_speaker_ids

        for supervision in example.supervisions:
            for field in self.speaker_fields:
                if supervision.has_custom(field):
                    speaker_id = getattr(supervision, field)
                else:
                    speaker_id = getattr(supervision, field, None)

                # Support the TTS speaker ID format:
                # | Language:en Dataset:<dataset_name> Speaker:<speaker_id> |
                if isinstance(speaker_id, str) and "Speaker:" in speaker_id:
                    speaker_id = speaker_id.rsplit("Speaker:", maxsplit=1)[-1].split("|", maxsplit=1)[0].strip()

                if speaker_id in excluded_speaker_ids:
                    return False
        return True


class ContextSpeakerSimilarityFilter:
    """
    Callable, returns ``True`` if a cut's context speaker similarity is greater than min_context_speaker_similarity and ``False`` otherwise.
    Acts as a pass-through for objects of other type than Cut.
    """

    def __init__(self, min_context_speaker_similarity: float | None) -> None:
        self.min_context_speaker_similarity = ifnone(min_context_speaker_similarity, -1)

    def __call__(self, example) -> bool:
        if (
            isinstance(example, MonoCut)
            and len(example.supervisions) > 0
            and example.supervisions[0].has_custom("context_speaker_similarity")
        ):
            return example.supervisions[0].context_speaker_similarity >= self.min_context_speaker_similarity
        else:
            return True


class TokenCountFilter:
    """
    Callable, returns ``True`` if an example's number of tokens is in range [t_min, t_max] and ``False`` otherwise.

    It is only applicable to data types that derive from class ``Formattable`` and lhotse ``Cut`` objects.
    Acts as a passthrough for Cuts.
    Raises exception if a non-Formattable and non-Cut data are provided.

    The ``measure_total_length`` option allows to select whether we should filter on context_ids length (=False)
    or input_ids length (=True).
    The difference is that for decoder-only models, we collapse input and output into a single sequence,
    so we should measure the example length using input_ids (measure_total_length=True).
    However, for models which have separate inputs and outputs such as encoder-decoder models,
    we want to measure the input lengths only here (measure_total_length=False),
    and enable ``TokenPerTokenFilter`` for additional filtering on the output sequence length.
    """

    def __init__(
        self,
        t_min: float | None,
        t_max: float | None,
        measure_total_length: bool,
        audio_token_estimator: AudioTokenEstimator | None = None,
    ) -> None:
        self.t_min = ifnone(t_min, -1)
        self.t_max = ifnone(t_max, float("inf"))
        self.measure_total_length = measure_total_length
        self.audio_token_estimator = audio_token_estimator
        self.enabled = self.t_min > 0 or self.t_max < float("inf")

    def __call__(self, example) -> bool:
        if not self.enabled or isinstance(example, Cut):
            return True  # does not apply to Cuts
        assert isinstance(example, Formattable), (
            f"TokenCountFilter can only be applied to data examples that derive Formattable class. "
            f"Formattable objects define properties input_length, output_length, and total_length that "
            f"allow us to select the right sequence length for filtering. We got: {example}"
        )
        try:
            length = measure_formattable_length(
                example,
                "total" if self.measure_total_length else "input",
                audio_token_estimator=self.audio_token_estimator,
            )
        except (AttributeError, AssertionError) as e:
            raise RuntimeError(
                f"Cannot measure token count for example: {example} "
                f"-- did you forget to apply prompt formatting? If instantiating Lhotse dataloader, "
                f"make sure you provided 'prompt_format' option and passed the tokenizer."
            ) from e
        return self.t_min <= length <= self.t_max


class TokenPerSecondFilter:
    """
    Callable, returns ``True`` if a cut's num_tokens (sum of len(tokens) for each supervision)
    is in range [tps_min, tps_max] and ``False`` otherwise.
    Acts as a pass-through for objects of other type than Cut.
    """

    def __init__(self, tps_min: float | None, tps_max: float | None) -> None:
        self.tps_min = ifnone(tps_min, -1)
        if isinstance(tps_max, Sequence):
            tps_max = float("inf")  # filtering handled in bucketing filter
        self.tps_max = ifnone(tps_max, float("inf"))
        assert tps_min <= tps_max, f"{tps_min=} {tps_max=}"
        self.enabled = tps_min > 0 or tps_max < float("inf")

    def __call__(self, example) -> bool:
        if not isinstance(example, Cut) or not self.enabled:
            return True  # pass-through for non-audio examples.
        tps = _measure_tps(example)
        return self.tps_min <= tps <= self.tps_max


class TokenPerTokenFilter:
    """
    Callable, returns ``True`` if a cut's num_tokens (sum of len(tokens) for each supervision)
    is in range [tps_min, tps_max] and ``False`` otherwise.
    Acts as a pass-through for audio examples (Cuts).
    """

    def __init__(self, tpt_min: float | None, tpt_max: float | None) -> None:
        self.tpt_min = ifnone(tpt_min, -1)
        if isinstance(tpt_max, Sequence):
            tpt_max = float("inf")  # filtering handled in bucketing filter
        self.tpt_max = ifnone(tpt_max, float("inf"))
        assert tpt_min <= tpt_max, f"{tpt_min=} {tpt_max=}"
        self.enabled = tpt_min > 0 or tpt_max < float("inf")

    def __call__(self, example) -> bool:
        if isinstance(example, Cut) or not self.enabled:
            return True  # pass-through for non-text examples.
        tpt = example.answer_ids.shape[0] / example.context_ids.shape[0]
        return self.tpt_min <= tpt <= self.tpt_max


class BucketingFilter:
    """
    Filters out examples that did not fit into any of the buckets.
    Intended mainly for 2D bucketing. This filter is only active when
    the constraint passed to it is of type ``FixedBucketBatchSizeConstraint2D``,
    and is otherwise disabled.
    """

    def __init__(self, sampling_constraint: SamplingConstraint) -> None:
        self.constraint = sampling_constraint
        self.enabled = isinstance(self.constraint, FixedBucketBatchSizeConstraint2D)

    def __call__(self, example) -> bool:
        if not self.enabled:
            return True
        return self.constraint.select_bucket(self.constraint.max_seq_len_buckets, example) is not None


def _measure_tokens(cut: Cut) -> int:
    if hasattr(cut, "input_ids"):
        return len(cut.input_ids)  # tokenized with prompt formatter
    supervisions_with_tokens = [s for s in cut.supervisions if hasattr(s, "tokens")]
    assert len(supervisions_with_tokens) > 0, (
        "Cannot measure the number of tokens with untokenized supervisions. "
        "Did you forget to provide the tokenizer argument to get_lhotse_dataloader_from_config() method?"
    )
    return sum(len(s.tokens) for s in supervisions_with_tokens)


def _measure_tps(cut: Cut) -> float:
    num_tokens = _measure_tokens(cut)
    return num_tokens / cut.duration
