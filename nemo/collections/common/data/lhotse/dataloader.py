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

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.cut.text import TextExample, TextPairExample
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    DynamicCutSampler,
    IterableDatasetWrapper,
    make_worker_init_fn,
)
from lhotse.dataset.sampling.base import SamplingConstraint, TimeConstraint, TokenConstraint
from lhotse.lazy import LazyFlattener
from lhotse.utils import fastcopy
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.utils import logging


@dataclass
class LhotseDataLoadingConfig:
    """
    Structured config used for OmegaConf schema validation.
    It's also a single source of truth for reading default option values.
    The options not supported anymore but present, e.g., in old configs,
    will be emitted in a DeprecationWarning and ignored.
    """

    # 1. Data inputs.
    #   a. "Classic" NeMo input path fields.
    input_cfg: Any = None  # TODO(pzelasko): typing
    manifest_filepath: Any = None  # str | list[list[str | float]] | None = None
    tarred_audio_filepaths: Any = None  # str | list[list[str]] | None = None
    #   b. Lhotse CutSet manifest / Lhotse Shar tar dir paths.
    cuts_path: str | None = None
    shar_path: Any = None  # str | list[str | tuple[str, float | int]] | None = None

    # 2. Batch size.
    #   a. Existing NeMo options.
    batch_size: int | None = None
    #   b. Lhotse dynamic batch sizes.
    batch_duration: float | None = None
    quadratic_duration: float | None = None
    #   c. Lhotse bucketing.
    use_bucketing: bool = False
    num_buckets: int = 30
    num_cuts_for_bins_estimate: int = 10000
    bucket_duration_bins: list[float] | None = None
    bucket_buffer_size: int = 10000
    #   d. Other Lhotse sampling options.
    shuffle_buffer_size: int | None = 10000
    drop_last: bool = False
    shard_seed: int | str = "trng"
    max_open_streams: int | None = None

    # 2.1 Multimodal sampling override options
    use_multimodal_sampling: bool = False
    token_equivalent_duration: float | None = None
    batch_tokens: int | None = None
    quadratic_factor: float | None = None

    # 3. Supported existing NeMo options.
    shuffle: bool = False
    sample_rate: int = 16000
    min_duration: float | None = -1
    max_duration: float | None = float("inf")
    seed: int | str = "randomized"  # int | "randomized" | "trng"; the latter two are lazily resolved by Lhotse in dloading worker processes
    num_workers: int = 0
    pin_memory: bool = False

    # 4. Optional Lhotse data augmentation.
    #   a. On-the-fly noise/audio mixing.
    noise_path: str | None = None
    noise_snr: tuple[float, float] = (10.0, 20.0)
    noise_mix_prob: float = 0.5
    #   b. On-the-fly 3-way speed perturbation.
    perturb_speed: bool = False
    #   c. Cut concatenation (glue together multiple utterances into a single one)
    concatenate_samples: bool = False
    concatenate_gap_seconds: float = 0.1
    concatenate_duration_factor: float = 1.0
    concatenate_merge_supervisions: bool = True
    db_norm: Optional[float] = -25.0  # from CodeSwitchingDataset
    #   d. On-the-fly cut truncation or window slicing
    #       I) truncate: select one chunk of a fixed duration for each cut
    truncate_duration: Optional[float] = None  # set this to enable
    truncate_offset_type: str = "random"  # "random" | "start" (fixed) | "end" (fixed, counted back)
    #       II) cut_into_windows: convert each cut to smaller cut using a sliding window (define hop for overlapping windows)
    cut_into_windows_duration: Optional[float] = None  # set this to enable
    cut_into_windows_hop: Optional[float] = None
    #       III) common options
    keep_excessive_supervisions: bool = True  # when a cut is truncated in the middle of a supervision, should we keep them.

    # 5. Other Lhotse options.
    text_field: str = "text"  # key to read the transcript from
    lang_field: str = "lang"  # key to read the language tag from
    # Enables iteration of NeMo non-tarred manifests that don't have a "sampling_rate" key without performing any I/O.
    # Note that this will not allow actual dataloading; it's only for manifest iteration as Lhotse objects.
    missing_sampling_rate_ok: bool = False


def get_lhotse_dataloader_from_config(
    config: DictConfig,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    tokenizer: TokenizerSpec | TokenizerWrapper = None,
) -> torch.utils.data.DataLoader:
    """
    Set up a Lhotse training dataloder.

    Expects a typical NeMo dataset configuration format, with additional fields: "use_lhotse=True".
    Some fields in the original NeMo configuration may be ignored.

    The ``dataset`` parameter should be an instance of a Lhotse-compatible PyTorch Dataset class.
    It only needs to define the following method ``__getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]``.
    This dataset is not expected to hold a reference to any actual data; it may be interpreted as a function
    mapping a Lhotse CutSet into a mini-batch of tensors.

    For an example, see: :class:`nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`,
    which is constructed from just a tokenizer and essentially loads and collates audio and tokenizes the transcript.

    The ``tokenizer`` is used when text-only datasets are included in dataloading.
    In these cases we will tokenize ``TextExample``s before sampling mini-batches so that
    we can account for their number of tokens.
    Note: this behaviour might eventually be extended to audio datasets too.

    Note that ``tokenizer`` can be any tokenizer type (e.g. both SentencePiece and Aggregate tokenizers work).
    """
    logging.info("We will be using a Lhotse DataLoader.")

    config = make_structured_with_schema_warnings(config)

    # 1. Load a manifest as a Lhotse CutSet.
    cuts, is_tarred = read_cutset_from_config(config)

    # Resample as a safeguard; it's a no-op when SR is already OK
    cuts = cuts.resample(config.sample_rate)

    # Expands cuts if multiple translations are provided.
    cuts = CutSet(LazyFlattener(cuts.map(_flatten_alt_text, apply_fn=None)))

    if config.use_multimodal_sampling:
        assert (
            tokenizer is not None
        ), "You must pass a tokenizer to `get_lhotse_dataloader_from_config` in order to read text-only datasets (enabled via use_multimodal_dataloading)"
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)
        # Note this code can also pre-tokenize the text in cuts, but for now we disable it with apply_fn.
        cuts = cuts.map(partial(tokenize, tokenizer=tokenizer), apply_fn=is_text)

    # 2. Optional augmentations.
    # 2.a. Noise mixing.
    if config.noise_path is not None:
        noise = CutSet.from_file(config.noise_path)
        cuts = cuts.mix(
            cuts=noise, snr=config.noise_snr, mix_prob=config.noise_mix_prob, seed="trng", random_mix_offset=True
        )

    # 2.b. On-the-fly speed perturbation.
    #    mux here ensures it's uniformly distributed throughout sampling,
    #    and applying it here (before sampler/dataset) ensures optimal
    #    bucket allocation.
    if config.perturb_speed:
        cuts = CutSet.mux(cuts, cuts.perturb_speed(0.9), cuts.perturb_speed(1.1),)

    # 2.d: truncation/slicing
    if config.truncate_duration is not None:
        cuts = cuts.truncate(
            max_duration=config.truncate_duration,
            offset_type=config.truncate_offset_type,
            keep_excessive_supervisions=config.keep_excessive_supervisions,
        )
    if config.cut_into_windows_duration is not None:
        cuts = cuts.cut_into_windows(
            duration=config.cut_into_windows_duration,
            hop=config.cut_into_windows_hop,
            keep_excessive_supervisions=config.keep_excessive_supervisions,
        )

    # Duration filtering, same as native NeMo dataloaders.
    # We can filter after the augmentations because they are applied only when calling load_audio().
    cuts = cuts.filter(DurationFilter(config.min_duration, config.max_duration))

    if config.use_multimodal_sampling:
        constraint = MultimodalSamplingConstraint(
            token_equivalent_duration=config.token_equivalent_duration,
            batch_size=config.batch_size,
            batch_tokens=config.batch_tokens,
            quadratic_factor=config.quadratic_factor,
        )
    else:
        constraint = TimeConstraint(
            max_cuts=config.batch_size,
            max_duration=config.batch_duration,
            quadratic_duration=config.quadratic_duration,
        )

    # 3. The sampler.
    if config.use_bucketing:
        # Bucketing. Some differences from NeMo's native bucketing:
        #    - we can tweak the number of buckets and bucket duration bins using the configuration
        #    - batch size is dynamic and configurable via a single param: max_duration (config: batch_duration)
        #    - quadratic_duration introduces a penalty to balance batch sizes for quadratic time complexity models
        logging.info(
            f"Creating a Lhotse DynamicBucketingSampler "
            f"(max_batch_duration={config.batch_duration} max_batch_size={config.batch_size})"
        )
        sampler = DynamicBucketingSampler(
            cuts,
            constraint=constraint,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
            num_buckets=config.num_buckets,
            duration_bins=config.bucket_duration_bins,
            num_cuts_for_bins_estimate=config.num_cuts_for_bins_estimate,
            buffer_size=config.bucket_buffer_size,
            rank=0 if is_tarred else global_rank,
            world_size=1 if is_tarred else world_size,
        )
    else:
        # Non-bucketing sampler, similar to original NeMo dataloading without bucketing,
        # but we also use batch_duration instead of batch_size here.
        # Recommended for dev/test.
        logging.info(
            f"Creating a Lhotse DynamicCutSampler (bucketing is disabled, "
            f"(max_batch_duration={config.batch_duration} max_batch_size={config.batch_size})"
        )
        sampler = DynamicCutSampler(
            cuts,
            constraint=constraint,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
            rank=0 if is_tarred else global_rank,
            world_size=1 if is_tarred else world_size,
        )

    if config.concatenate_samples:
        # Cut concatenation will produce longer samples out of shorter samples
        # by gluing them together from the shortest to longest not to exceed a duration
        # of longest_cut * duration_factor (greedy knapsack algorithm for minimizing padding).
        # Useful e.g. for simulated code-switching in multilingual setups.
        # We follow concatenation by ``merge_supervisions`` which creates a single supervision
        # object with texts joined by a whitespace so that "regular" dataset classes don't
        # have to add a special support for multi-supervision cuts.
        sampler = sampler.map(
            CutConcatenate(gap=config.concatenate_gap_seconds, duration_factor=config.concatenate_duration_factor,)
        )
        if config.db_norm is not None:
            sampler = sampler.map(partial(_normalize_loudness, db_norm=config.db_norm))
        if config.concatenate_merge_supervisions:
            sampler = sampler.map(_merge_supervisions)

    # 4. Creating dataloader.
    if is_tarred:
        # Wrapper here is necessary when using NeMo tarred data or Lhotse Shar data,
        # because then I/O happens upon sampler iteration. Normally, the sampler resides
        # in the training loop process, but when we use iterable dataset, we can move it to
        # the dataloading worker process.
        # We use lhotse's own worker_init_fn which leverages information such as rank, world_size,
        # worker_id, etc. to set a different random seed for each (node, worker) combination.
        # This together with infinite datasets removes the need to split data across nodes/workers.
        dloader_kwargs = dict(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler),
            worker_init_fn=make_worker_init_fn(rank=global_rank, world_size=world_size),
            persistent_workers=config.num_workers > 0,  # helps Lhotse Shar maintain shuffling state
        )
    else:
        # For non-tarred data, the sampler resides in the training loop process and
        # reads only light-weight JSON objects; it samples mini-batches and passes
        # the meta-data to Dataset, which performs the actual I/O inside its __getitem__ method.
        dloader_kwargs = dict(dataset=dataset, sampler=sampler)
    dloader = torch.utils.data.DataLoader(
        **dloader_kwargs, batch_size=None, num_workers=config.num_workers, pin_memory=config.pin_memory,
    )

    return dloader


def make_structured_with_schema_warnings(config: DictConfig) -> DictConfig:
    """
    Checks the schema and fills missing default option values.
    Warns the user if any of the fields are not supported by the current schema
    but does not raise exceptions.
    """
    default = OmegaConf.structured(LhotseDataLoadingConfig)

    # Remove unsupported keys and warn about them.
    supported_keys = set(OmegaConf.to_container(default).keys())
    received_keys = set(OmegaConf.to_container(config).keys())
    unsupported_keys = received_keys - supported_keys
    if unsupported_keys:
        warnings.warn(
            f"The following configuration keys are no longer supported " f"and ignored: {','.join(unsupported_keys)}",
            category=DeprecationWarning,
        )
    config = OmegaConf.masked_copy(config, list(supported_keys))

    return OmegaConf.merge(default, config)


@dataclass
class MultimodalSamplingConstraint(SamplingConstraint):
    # how many seconds of audio is a text token worth; balances audio to text ratio in a mini-batch
    token_equivalent_duration: float

    # defines maximum batch size (may be lower than that if batch_length is also specified)
    batch_size: int | None = None

    # defines the total number of tokens in a mini-batch
    # setting this enables dynamic batch sizes
    # we will use ``token_equivalent_duration`` to convert audio examples to token sizes
    batch_tokens: int | None = None

    # when specified, this value is inversely proportional to the penalty we assign
    # to longer examples when measuring their length/duration;
    # i.e. large quadratic factor is a small penalty, small quadratic factor is a large penalty
    # tweaking this helps equalize the GPU memory usage for dynamic batch sizes when using bucketing
    quadratic_factor: float | None = None

    _internal = None

    def __post_init__(self):
        self._internal = TokenConstraint(
            max_tokens=self.batch_tokens, max_examples=self.batch_size, quadratic_length=self.quadratic_factor,
        )

    def add(self, example: Any) -> None:
        if isinstance(example, Cut):
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
            return example.duration / self.token_equivalent_duration
        if isinstance(example, (TextExample, TextPairExample)):
            return example.num_tokens
        raise RuntimeError(f"Unsupported example type: {type(example)}")


def is_text(example) -> bool:
    return isinstance(example, (TextExample, TextPairExample))


Example = TypeVar("Example", bound=Union[Cut, TextExample, TextPairExample])


def tokenize(example: Example, tokenizer) -> Example:
    if isinstance(example, Cut):
        for s in example.supervisions:
            s.tokens = np.asarray(tokenizer(s.text, s.language))
    elif isinstance(example, TextExample):
        example.tokens = np.asarray(tokenizer(example.text, example.language))
    elif isinstance(example, TextPairExample):
        example.source.tokens = np.asarray(tokenizer(example.source.text, example.source.language))
        example.target.tokens = np.asarray(tokenizer(example.source.text, example.target.language))
    else:
        raise RuntimeError(f"Unsupported type of example: {type(example)}")
    return example


# The helper callables below exist to avoid passing lambdas into lhotse CutSet map/filter methods.
# Lambdas are not serializable across processes by pickle.
# Note: lhotse offers LHOTSE_DILL_ENABLED=1 and ``lhotse.lazy.set_dill_enabled(True)``
# to support pickling lambdas if its ever truly necessary.


class DurationFilter:
    """Callable, returns ``True`` if a cut's duration is in range [d_min, d_max] and ``False`` otherwise."""

    def __init__(self, d_min: float, d_max: float) -> None:
        self.d_min = d_min
        self.d_max = d_max

    def __call__(self, example) -> bool:
        if isinstance(example, Cut):
            return self.d_min <= example.duration <= self.d_max
        else:
            return True  # does not apply to text etc.


def _normalize_loudness(cuts: CutSet, db_norm: float) -> CutSet:
    return cuts.normalize_loudness(target=db_norm, mix_first=False)


def _merge_supervisions(cuts: CutSet) -> CutSet:
    return cuts.merge_supervisions()


def _flatten_alt_text(cut) -> list:
    ans = [cut]
    if not isinstance(cut, Cut) or cut.custom is None or cut.custom.get("alt_text") is None:
        return ans
    cut = cut.move_to_memory(audio_format="wav")  # performs I/O once and holds audio in memory from now on
    # Popping to ease eyesight on debug.
    paired_text = cut.custom.pop("alt_text")
    for data in paired_text.values():
        # Copy to avoid lazy dataloading issues
        data = data.copy()
        text_instance = cut.map_supervisions(lambda s: fastcopy(s, text=data["text"], language=data["lang"]))
        text_instance.custom = {"text": data.pop("text"), "lang": data.pop("lang"), **data}
        ans.append(text_instance)
    return ans
