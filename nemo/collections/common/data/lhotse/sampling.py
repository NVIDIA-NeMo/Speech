# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import bisect
import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

from lhotse.cut import Cut
from lhotse.dataset import SamplingConstraint, TokenConstraint
from lhotse.dataset.sampling.dynamic_bucketing import FixedBucketBatchSizeConstraint
from lhotse.utils import ifnone

from nemo.collections.common.data.lhotse.text_adapters import Formattable


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

    # When False (default), we only consider the input part of the example to determine its length,
    # e.g. for a Cut that means its audio duration converted to tokens, for text that means len(context_ids), etc.
    # When True, we consider the sum of input and output lengths together (useful mostly for decoder-only models).
    measure_total_length: bool = False

    _internal = None

    def __post_init__(self):
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

    def reset(self) -> None:
        self._internal.reset()

    def measure_length(self, example: Any) -> float:
        if isinstance(example, Cut):
            audio_len_in_tokens = math.ceil(example.duration / self.token_equivalent_duration)
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
                return example.total_length if self.measure_total_length else example.input_length
            except (AttributeError, AssertionError) as e:
                raise RuntimeError(
                    "Couldn't determine the length of a text example; "
                    "have you provided both prompt_format and tokenizer when instantiating the dataloader?"
                ) from e
        raise RuntimeError(f"Unsupported example type: {type(example)}")


@dataclass
class FixedBucketBatchSizeConstraint2D(FixedBucketBatchSizeConstraint):
    """
    Sampling strategy that customizes Lhotse samplers to support 2D bucket selection (it also supports 1D).
    It is intended only for audio examples (i.e., Lhotse Cut objects).
    """

    @property
    def bucketing_2d_enabled(self) -> bool:
        return isinstance(self.max_seq_len_buckets[0], Sequence) and len(self.max_seq_len_buckets[0]) == 2

    def measure_length(self, example: Cut) -> tuple[float, float] | float:
        if self.bucketing_2d_enabled:
            return example.duration, _measure_tokens(example)
        else:
            return example.duration

    def select_bucket(self, buckets: Any, example: Any = None, example_len: Any = None) -> int:
        if not self.bucketing_2d_enabled:
            return super().select_bucket(buckets=buckets, example=example, example_len=example_len)
        if example_len is None:
            example_len = self.measure_length(example)
        bucket_idx = bisect.bisect_left(buckets, example_len)
        # For 2D bucketing we have to refine the initially found bucket_idx, as bisect
        # looks primarily at the first index of a tuple (i.e. duration).
        # For example, with buckets [(1, 1), (1, 2), (2, 2), (2, 4)] and example (1.5, 3)
        # bisect would allocate it to bucket_idx=2 instead of bucket_idx=3.
        # To refine, we'll try to push the example to as many buckets to the right as possible,
        # as long as they have the same dim0 length (e.g. audio duration) and the example's dim1
        # is smaller than the bin's dim1 (e.g., output token sequence length).
        bin_dim0, bin_dim1 = self.max_seq_len_buckets[bucket_idx]
        num_buckets = len(self.max_seq_len_buckets)
        while (
            (next_idx := bucket_idx + 1) < num_buckets  # There is a next bucket
            and (bin := self.max_seq_len_buckets[next_idx])[0] == bin_dim0  # The next bucket has the same 1st dim.
            # The example's 2nd dim is between that of the current and the next bucket; or,
            # the next bucket's 2nd dim is still smaller than example.
            and (bin_dim1 < example_len[1] <= bin[1] or bin[1] < example_len[1])
        ):
            bucket_idx = next_idx
            bin_dim0, bin_dim1 = self.max_seq_len_buckets[bucket_idx]

        if example_len[0] > bin_dim0 or example_len[1] > bin_dim1:
            logging.warning(
                f"Data sample exceeds 2D bucket specification: lengths={example_len} bucket=({bin_dim0}, {bin_dim1}) "
                f"(there is no larger bucket that would fit this example). "
                f"We will keep it but expect OutOfMemoryError to happen during the training. "
                f"You can fix this by stricter filtering with max_duration, max_tokens, max_tps, max_tpt; "
                f"or re-estimating your bucket bins to match the actual data length distribution. "
                f"Details: {example=}"
            )

        return bucket_idx


@dataclass
class MultimodalFixedBucketBatchSizeConstraint2D(FixedBucketBatchSizeConstraint2D):
    """
    Sampling strategy that customizes Lhotse samplers to support both multimodal sampling and 2D bucket selection.
    It combines the capabilities of :class:`FixedBucketBatchSizeConstraint2D` and :class:`MultimodalSamplingConstraint`
    """

    # How many seconds of audio is a text token worth; balances audio to text ratio in a mini-batch.
    # Generally set this to frame_shift * total_subsampling_factor of your audio encoder.
    token_equivalent_duration: float | None = None

    # When False (default), we only consider the input part of the example to determine its length,
    # e.g. for a Cut that means its audio duration converted to tokens, for text that means len(context_ids), etc.
    # When True, we consider the sum of input and output lengths together (useful mostly for decoder-only models).
    measure_total_length: bool = False

    def measure_length(self, example: Any) -> float | tuple[float, float]:
        if isinstance(example, Cut):
            # Total length of a Cut (audio+text example) is counted as the sum of:
            # * num_tokens in each supervision segment ("utterance") in the Cut
            # * num_frames of audio (frame=token) given a token-equivalent-duration (basically a frame shift)
            audio_len_in_tokens = math.ceil(example.duration / self.token_equivalent_duration)
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
                return example.input_length, example.output_length
            else:
                return example.total_length if self.measure_total_length else example.input_length

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
        else:
            return True  # does not apply to text etc.


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

    def __init__(self, t_min: float | None, t_max: float | None, measure_total_length: bool) -> None:
        self.t_min = ifnone(t_min, -1)
        self.t_max = ifnone(t_max, float("inf"))
        self.measure_total_length = measure_total_length
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
            length = example.total_length if self.measure_total_length else example.input_length
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
        self.tpt_max = ifnone(tpt_max, float("inf"))
        assert tpt_min <= tpt_max, f"{tpt_min=} {tpt_max=}"
        self.enabled = tpt_min > 0 or tpt_max < float("inf")

    def __call__(self, example) -> bool:
        if isinstance(example, Cut) or not self.enabled:
            return True  # pass-through for non-text examples.
        tpt = example.answer_ids.shape[0] / example.context_ids.shape[0]
        return self.tpt_min <= tpt <= self.tpt_max


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
