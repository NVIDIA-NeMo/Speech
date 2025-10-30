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
import os
import random
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from lhotse import CutSet, RecordingSet
from lhotse.cut import Cut
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    DynamicCutSampler,
    IterableDatasetWrapper,
    ReverbWithImpulseResponse,
    RoundRobinSampler,
    ZipSampler,
    make_worker_init_fn,
)
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import CutSampler, SamplingConstraint, TimeConstraint
from lhotse.lazy import LazyFlattener
from lhotse.utils import fastcopy, fix_random_seed
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.data.lhotse.cutset import (
    IncompleteConfigError,
    guess_parse_cutset,
    read_cutset_from_config,
)
from nemo.collections.common.data.lhotse.sampling import (
    BucketingFilter,
    CERFilter,
    ContextSpeakerSimilarityFilter,
    DurationFilter,
    FixedBucketBatchSizeConstraint2D,
    MultimodalFixedBucketBatchSizeConstraint2D,
    MultimodalSamplingConstraint,
    TokenCountFilter,
    TokenPerSecondFilter,
    TokenPerTokenFilter,
    ValidationStatusFilter,
)
from nemo.collections.common.data.prompt_fn import apply_prompt_format_fn
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
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
    #  Enable this to support dataloading from JSON manifests that reference subsets of audio tar files.
    skip_missing_manifest_entries: bool = False
    tarred_random_access: bool = False  # deprecated, replaced by: skip_missing_manifest_entries
    # 2. Batch size.
    #   a. Existing NeMo options.
    batch_size: int | None = None
    #   b. Lhotse dynamic batch sizes.
    batch_duration: float | None = None
    quadratic_duration: float | None = None
    #   c. Lhotse bucketing.
    use_bucketing: bool = False
    bucket_batch_size: list[int] | None = None
    num_buckets: int = 30
    num_cuts_for_bins_estimate: int = 10000
    bucket_duration_bins: Any = None  # list[float] | list[list[float]] | None = None
    bucket_buffer_size: int = 10000
    concurrent_bucketing: bool = True  # fetches data in a background thread
    bucketing_2d_strict_mode: bool = True  # reduces padding by discarding significant outliers
    #   d. Other Lhotse sampling options.
    shuffle_buffer_size: int | None = 10000
    drop_last: bool = False
    shard_seed: int | str = "trng"
    max_open_streams: int | None = None
    cuda_expandable_segments: bool = True
    # e. Multi-config related options.
    #    Setting multi_config=True will scan the config for keys with DictConfig values,
    #    create a separate sampler for each, and fuse the samplers according to sampler_fusion.
    multi_config: bool = False
    sampler_fusion: str = "round_robin"  # round_robin | randomized_round_robin | zip
    sampler_weights: dict[str, float] | None = None  # only applicable to randomized_round_robin

    # 2.1 Multimodal sampling override options
    pretokenize: bool = True  # should we apply tokenizer before data sampling
    prompt_format: str | None = None  # when provided, we'll apply the prompt in addition to the tokenizer
    use_multimodal_sampling: bool = False
    audio_locator_tag: str | None = None  # global audio placeholder token, propagates to datasets in input_cfg
    token_equivalent_duration: float | None = None
    batch_tokens: int | None = None
    quadratic_factor: float | None = None

    # 2.2 Filters on sequence lengths.
    #   * Speech input
    min_duration: float | None = -1
    max_duration: float | None = float("inf")
    min_tps: int = -1  # allowed tokens per second (audio-only)
    max_tps: Any = float("inf")  # float | list[float]
    #   * Text input
    min_tokens: int | None = None
    max_tokens: int | None = None
    # When true, combine context+answer lengths into a total length; otherwise report context length.
    # For 2D bucketing it's always false, as we report a tuple of (context_len, answer_len).
    measure_total_length: bool = True
    min_tpt: int = -1  # allowed tokens per token (text-only)
    max_tpt: Any = float("inf")  # float | list[float]

    # 2.3 Filters on CER and/or cosine speaker similarity of the context audio serving for TTS use cases.
    max_cer: float | None = float("inf")
    min_context_speaker_similarity: float | None = -1

    # 2.4 Filters on validation status. If the validation status is not "pass", the cut will be filtered out.
    keep: str = "pass"

    # 3. Supported existing NeMo options.
    shuffle: bool = False
    sample_rate: int = 16000
    seed: int | str = 0
    num_workers: int = 0
    pin_memory: bool = False
    channel_selector: int | str | None = None

    # 4. Optional Lhotse data augmentation.
    #   a. On-the-fly noise/audio mixing.
    noise_path: Any | None = (
        None  # str | dict where dict can have any of keys:
        # manifest_filepath, tarred_audio_filepaths, cuts_path, shar_path
    )
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
    #       II) cut_into_windows: convert each cut to smaller cut using a sliding window
    #           (define hop for overlapping windows)
    cut_into_windows_duration: Optional[float] = None  # set this to enable
    cut_into_windows_hop: Optional[float] = None
    #       III) common options
    keep_excessive_supervisions: bool = (
        True  # when a cut is truncated in the middle of a supervision, should we keep them.
    )
    #   e. RIR augmentation (synthetic RIR if rir_path is None)
    #   at the moment supports only Lhotse recording manifests:
    #   e.g. https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/rir_noise.py
    rir_enabled: bool = False
    rir_path: str | None = None  # str, must point to a lhotse RecordingSet manifest
    rir_prob: float = 0.5
    #   f. Padding to a minimum duration. Examples shorter than this will be padded, others are unaffected.
    pad_min_duration: Optional[float] = None
    pad_direction: str = "right"  # "right" | "left" | "both" | "random"

    # 5. Other Lhotse options.
    text_field: str = "text"  # key to read the transcript from
    lang_field: str = "lang"  # key to read the language tag from
    # Enables iteration of NeMo non-tarred manifests that don't have a "sampling_rate" key without performing any I/O.
    # Note that this will not allow actual dataloading; it's only for manifest iteration as Lhotse objects.
    metadata_only: bool = False
    # Forces the resulting CutSet to be finite, so that the iteration will end after a full single epoch.
    # Do not turn this on unless you're sure that you know what you're doing.
    # In most cases (such as regular multi-GPU training) it will result in a deadlock due to
    # a different number of steps on different DDP ranks.
    force_finite: bool = False
    # The following two options may be used to override auto-detection of appropriate PyTorch dataset flavor
    #   for your data types. PyTorch DataLoader uses two objects to yield data: dataset and sampler.
    # *Map-dataset flavor.* There is one sampler per GPU that lives in the training loop process;
    #   it selects the examples to be prepared by map-dataset class.
    #   Each batch selection determined by the sampler is then passed by the dataloader
    #   to one of its worker processes to be processed by the dataset class.
    # *Iterable-dataset flavor.* Each dataloading worker has its own sampler replica instead;
    #   the sampler must have the logic for either data deduplication or unique order shuffling to avoid
    #   duplicated data across workers and GPUs. Lhotse relies on unique order shuffling.
    # The default settings are:
    # * use iterable dataset for tarred audio data.
    # * use iterable dataset for any text data.
    # * use map dataset for non-tarred audio data (we might change this in the future)
    force_map_dataset: bool = False
    force_iterable_dataset: bool = False
    # Force the dataloader to slice each data source.
    # This may improve sampling randomness for large-scale runs with many dataset sources and large shards
    # at the cost of some IO redundancy.
    # The slicing is achieved with a randomly-selected offset K used to skip the first K examples,
    # and reading them consecutively for ``slice_length`` iterations.
    # The first K examples will actually be read and then discarded, incurring the IO cost, due to
    # our support of object stores and gzipped files that generally don't have indexes of byte offsets per line.
    slice_length: Optional[int] = None


def determine_use_iterable_dataset(use_iterable_dataset: bool, config: DictConfig) -> bool:
    """Determine whether to use iterable dataset for a given configuration."""
    assert not (
        config.force_map_dataset and config.force_iterable_dataset
    ), "Conflicting options: force_map_dataset=True and force_iterable_dataset=True"
    use_iterable_dataset = (use_iterable_dataset or config.force_iterable_dataset) and not config.force_map_dataset
    return use_iterable_dataset


def get_lhotse_dataloader_from_config(
    config: Union[dict, DictConfig],
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    """
    Set up a Lhotse training dataloader.

    Expects a typical NeMo dataset configuration format, with additional fields: "use_lhotse=True".
    Some fields in the original NeMo configuration may be ignored.

    The ``dataset`` parameter should be an instance of a Lhotse-compatible PyTorch Dataset class.
    It only needs to define the following method ``__getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]``.
    This dataset is not expected to hold a reference to any actual data; it may be interpreted as a function
    mapping a Lhotse CutSet into a mini-batch of tensors.

    For an example, see: :class:`nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`,
    which is constructed from just a tokenizer and essentially loads and collates audio and tokenizes the transcript.

    The ``tokenizer`` is used both for audio and text datasets for on-the-fly tokenization.
    This allows us to stratify the bucketing by the count of input/output tokens (depending on modality).
    If "prompt_format" is additionally provided in the config, we will also apply a prompt formatter.
    Note that ``tokenizer`` can be any tokenizer type (e.g. both SentencePiece and Aggregate tokenizers work).
    """
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)

    # Providing default value because we haven't filled the config defaults yet.
    maybe_set_cuda_expandable_segments(enabled=config.get("cuda_expandable_segments", True))

    if config.get("multi_config", False):
        return get_lhotse_dataloader_from_multi_config(
            top_level_config=config,
            global_rank=global_rank,
            world_size=world_size,
            dataset=dataset,
            tokenizer=tokenizer,
        )
    else:
        return get_lhotse_dataloader_from_single_config(
            config=config, global_rank=global_rank, world_size=world_size, dataset=dataset, tokenizer=tokenizer
        )


def get_lhotse_dataloader_from_single_config(
    config: DictConfig,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    """
    Set up a Lhotse training dataloader.

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

    # First, resolve the random seed in case a string value was provided.
    config.seed = resolve_seed(config.seed)
    fix_random_seed(config.seed)

    sampler, use_iterable_dataset = get_lhotse_sampler_from_config(
        config=config, global_rank=global_rank, world_size=world_size, tokenizer=tokenizer
    )

    # 4. Creating dataloader.
    if use_iterable_dataset:
        # Wrapper here is necessary when using NeMo tarred data or Lhotse Shar data,
        # because then I/O happens upon sampler iteration. Normally, the sampler resides
        # in the training loop process, but when we use iterable dataset, we can move it to
        # the dataloading worker process.
        # We use lhotse's own worker_init_fn which leverages information such as rank, world_size,
        # worker_id, etc. to set a different random seed for each (node, worker) combination.
        # This together with infinite datasets removes the need to split data across nodes/workers.
        dloader_kwargs = dict(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler),
            worker_init_fn=make_worker_init_fn(rank=global_rank, world_size=world_size, seed=config.seed),
            persistent_workers=config.num_workers > 0,  # helps Lhotse Shar maintain shuffling state
        )
    else:
        # For non-tarred data, the sampler resides in the training loop process and
        # reads only light-weight JSON objects; it samples mini-batches and passes
        # the meta-data to Dataset, which performs the actual I/O inside its __getitem__ method.
        dloader_kwargs = dict(dataset=dataset, sampler=sampler)
    dloader = torch.utils.data.DataLoader(
        **dloader_kwargs,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dloader


def get_lhotse_dataloader_from_multi_config(
    top_level_config: DictConfig,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    """
    Set up a Lhotse training dataloder.

    It works similarly to :func:`get_lhotse_dataloader_from_config`, except that
    you can provide multiple configs to set up different sampling, batching, and
    augmentation settings for every dataset and decide how to merge them.

    The expected format is that the ``configs`` is a dict of group name -> actual config.

    The first config is treated as a "main" config that determines the RNG, CUDA allocator,
    and sampler fusion settings.
    """

    def gather_shared_opts():
        """
        In multi-config setting, the top-level config defines several attributes that overwrite
        the ones present in sub-configs.
        """
        assert all(k in top_level_config for k in ["seed", "shard_seed", "shuffle"]), (
            "In a multi-config setting (multi_config=True), the top-level namespace (typically train_ds)"
            "must define at least 'seed', 'shard_seed', and 'shuffle' keys that will be "
            "shared by all sub-configs."
        )
        overwriting_opts = [
            "seed",
            "shard_seed",
            "num_workers",
            "pin_memory",
            "shuffle",
            "sampler_fusion",
            "sampler_weights",
            "multi_config",
            "metadata_only",
            "force_finite",
        ]
        defaults = OmegaConf.structured(LhotseDataLoadingConfig)
        top_level_config["seed"] = resolve_seed(top_level_config["seed"])
        return OmegaConf.create({k: top_level_config.get(k, defaults[k]) for k in overwriting_opts})

    shared_opts = gather_shared_opts()
    fix_random_seed(shared_opts.seed)

    configs = {
        name: c
        for name, c in top_level_config.items()
        if isinstance(c, DictConfig) and name not in ("sampler_weights",)  # exclude dict opts
    }

    source_samplers, source_use_iterable_dataset = {}, []
    for name, config in configs.items():
        try:
            expanded_config = make_structured_with_schema_warnings(config)
            for k, v in shared_opts.items():
                expanded_config[k] = v
            s, t = get_lhotse_sampler_from_config(
                config=expanded_config, global_rank=global_rank, world_size=world_size, tokenizer=tokenizer
            )
        except IncompleteConfigError as e:
            raise IncompleteConfigError(
                "Cannot create a sampler for one of the sub-configs in a multi_config setup."
                f"The problematic config is under key={name} and has the following contents: {config}"
            ) from e
        source_samplers[name] = s
        source_use_iterable_dataset.append(t)

    assert all(st == source_use_iterable_dataset[0] for st in source_use_iterable_dataset[1:]), (
        "When using multiple input_cfg sources ensure they are all tarred or non-tarred (can't mix). "
        "You can provide force_iterable_dataset=True to each namespace to fix."
    )
    use_iterable_dataset = all(source_use_iterable_dataset)
    if shared_opts.sampler_fusion == "zip":
        sampler = ZipSampler(*source_samplers.values())
    elif shared_opts.sampler_fusion == "round_robin":
        sampler = RoundRobinSampler(*source_samplers.values())
    elif shared_opts.sampler_fusion == "randomized_round_robin":
        _samplers, _weights = [], []
        for key in source_samplers.keys():
            _samplers.append(source_samplers[key])
            if shared_opts.sampler_weights is not None:
                _weights.append(shared_opts.sampler_weights[key])
        sampler = RoundRobinSampler(
            *_samplers,
            randomize=_weights if len(_weights) > 0 else True,
            seed=shared_opts.seed,
        )
    else:
        raise RuntimeError(f"Unsupported sampler fusion strategy: {shared_opts.sampler_fusion}")

    # 4. Creating dataloader.
    if use_iterable_dataset:
        # Wrapper here is necessary when using NeMo tarred data or Lhotse Shar data,
        # because then I/O happens upon sampler iteration. Normally, the sampler resides
        # in the training loop process, but when we use iterable dataset, we can move it to
        # the dataloading worker process.
        # We use lhotse's own worker_init_fn which leverages information such as rank, world_size,
        # worker_id, etc. to set a different random seed for each (node, worker) combination.
        # This together with infinite datasets removes the need to split data across nodes/workers.
        dloader_kwargs = dict(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler),
            worker_init_fn=make_worker_init_fn(rank=global_rank, world_size=world_size, seed=shared_opts.seed),
            persistent_workers=shared_opts.num_workers > 0,  # helps Lhotse Shar maintain shuffling state
        )
    else:
        # For non-tarred data, the sampler resides in the training loop process and
        # reads only light-weight JSON objects; it samples mini-batches and passes
        # the meta-data to Dataset, which performs the actual I/O inside its __getitem__ method.
        dloader_kwargs = dict(dataset=dataset, sampler=sampler)
    dloader = torch.utils.data.DataLoader(
        **dloader_kwargs,
        batch_size=None,
        num_workers=shared_opts.num_workers,
        pin_memory=shared_opts.pin_memory,
    )

    return dloader


def get_lhotse_sampler_from_config(config, global_rank, world_size, tokenizer=None) -> tuple[CutSampler, bool]:
    """Create a CutSampler from a dataloader config."""
    # 1. Load a manifest as a Lhotse CutSet.
    cuts, use_iterable_dataset = read_cutset_from_config(config)
    use_iterable_dataset = determine_use_iterable_dataset(use_iterable_dataset, config)

    # Apply channel selector
    if config.channel_selector is not None:
        logging.info('Using channel selector %s.', config.channel_selector)
        cuts = cuts.map(partial(_select_channel, channel_selector=config.channel_selector))

    # Resample as a safeguard; it's a no-op when SR is already OK
    cuts = cuts.map(partial(resample, sampling_rate=config.sample_rate), apply_fn=None)

    # Expands cuts if multiple translations are provided.
    cuts = CutSet(LazyFlattener(cuts.map(_flatten_alt_text, apply_fn=None)))

    if config.use_multimodal_sampling:
        assert tokenizer is not None, (
            "You must pass a tokenizer to `get_lhotse_dataloader_from_config` in order to"
            "read text-only datasets (enabled via use_multimodal_dataloading)"
        )

    if tokenizer is not None and config.pretokenize:
        if not use_iterable_dataset:
            logging.warning(
                "You are using a non-tarred dataset and requested tokenization during data sampling "
                "(pretokenize=True). This will cause the tokenization to happen in the main (GPU) process,"
                "possibly impacting the training speed if your tokenizer is very large."
                "If the impact is noticable, set pretokenize=False in dataloader config."
                "(note: that will disable token-per-second filtering and 2D bucketing features)"
            )

        if config.prompt_format is not None:
            cuts = cuts.map(
                partial(tokenize_with_prompt, tokenizer=tokenizer, prompt_format=config.prompt_format), apply_fn=None
            )
        else:
            if not isinstance(tokenizer, TokenizerWrapper):
                tokenizer = TokenizerWrapper(tokenizer)
            cuts = cuts.map(partial(tokenize, tokenizer=tokenizer), apply_fn=None)

    # 2. Optional augmentations.
    # 2.a. Noise mixing.
    if config.noise_path is not None:
        noise = guess_parse_cutset(config.noise_path)
        cuts = cuts.mix(
            cuts=noise,
            snr=tuple(config.noise_snr),
            mix_prob=config.noise_mix_prob,
            seed=config.shard_seed,
            random_mix_offset=True,
        )

    # 2.b. On-the-fly speed perturbation.
    #    mux here ensures it's uniformly distributed throughout sampling,
    #    and applying it here (before sampler/dataset) ensures optimal
    #    bucket allocation.
    if config.perturb_speed:
        cuts = CutSet.mux(
            cuts,
            cuts.perturb_speed(0.9),
            cuts.perturb_speed(1.1),
        )

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

    if config.pad_min_duration is not None:
        cuts = cuts.pad(duration=config.pad_min_duration, direction=config.pad_direction, preserve_id=True)

    # Duration filtering, same as native NeMo dataloaders.
    # We can filter after the augmentations because they are applied only when calling load_audio().
    cuts = cuts.filter(DurationFilter(config.min_duration, config.max_duration))
    cuts = cuts.filter(
        TokenCountFilter(config.min_tokens, config.max_tokens, measure_total_length=config.measure_total_length)
    )

    # validation status filtering
    cuts = cuts.filter(ValidationStatusFilter(config.keep))
    # CER filtering, same as native NeMo dataloaders.
    cuts = cuts.filter(CERFilter(config.max_cer))
    # Context speaker similarity filtering, same as native NeMo dataloaders.
    cuts = cuts.filter(ContextSpeakerSimilarityFilter(config.min_context_speaker_similarity))

    if tokenizer is not None and config.pretokenize:
        cuts = cuts.filter(TokenPerSecondFilter(config.min_tps, config.max_tps))
        cuts = cuts.filter(TokenPerTokenFilter(config.min_tpt, config.max_tpt))

    # Select the strategy customizing Lhotse sampler behaviour.
    # Provides support for dynamic batch sizes, multimodal dataloading, 2D bucketing, etc.
    bucket_duration_bins = determine_bucket_duration_bins(config)
    cuts, constraint = determine_sampling_constraint(cuts, bucket_duration_bins, config)

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
        # Determine the bucket duration bins
        sampler = DynamicBucketingSampler(
            cuts,
            constraint=constraint,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.shard_seed,
            num_buckets=config.num_buckets,
            duration_bins=determine_bucket_duration_bins(config),
            num_cuts_for_bins_estimate=config.num_cuts_for_bins_estimate,
            buffer_size=config.bucket_buffer_size,
            concurrent=config.concurrent_bucketing,
            rank=0 if use_iterable_dataset else global_rank,
            world_size=1 if use_iterable_dataset else world_size,
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
            seed=config.shard_seed,
            rank=0 if use_iterable_dataset else global_rank,
            world_size=1 if use_iterable_dataset else world_size,
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
            CutConcatenate(
                gap=config.concatenate_gap_seconds,
                duration_factor=config.concatenate_duration_factor,
            )
        )
        if config.db_norm is not None:
            sampler = sampler.map(partial(_normalize_loudness, db_norm=config.db_norm))
        if config.concatenate_merge_supervisions:
            sampler = sampler.map(_merge_supervisions)

    if config.rir_enabled:
        sampler = sampler.map(
            ReverbWithImpulseResponse(
                rir_recordings=RecordingSet.from_file(config.rir_path) if config.rir_path is not None else None,
                p=config.rir_prob,
                randgen=random.Random(config.seed),
            )
        )

    return sampler, use_iterable_dataset


def determine_sampling_constraint(cuts: CutSet, bucket_duration_bins, config) -> tuple[CutSet, SamplingConstraint]:
    """
    Select an appropriate sampling strategy (constraint) for Lhotse samplers based on the configuration.
    Sampling constraint affects the batch size (static/dynamic) and bucketing behaviour (1D/2D).
    It is the appropriate customization point to introduce support of other modalities,
    as it defines a method for example sequence length measurement (audio duration, text tokens, etc.).

    Some constraints apply extra filter on ``cuts`` which is why we accept and return the ``CutSet``.

    Lhotse's default is :class:`TimeConstraint` for regular audio data, other available options are
    multimodal constraints (joint text + audio) and their 2D bucketing extensions.
    """
    if config.use_multimodal_sampling:
        if config.bucket_batch_size is not None:
            assert (
                bucket_duration_bins is not None
            ), "Cannot use bucket_batch_size option if bucket_duration_bins are not provided."
            constraint = MultimodalFixedBucketBatchSizeConstraint2D(
                max_seq_len_buckets=bucket_duration_bins,
                batch_sizes=config.bucket_batch_size,
                token_equivalent_duration=config.token_equivalent_duration,
                strict_2d=config.bucketing_2d_strict_mode,
                max_ratio=config.max_tpt if isinstance(config.max_tpt, Sequence) else None,
                measure_total_length=config.measure_total_length,
            )
            cuts = cuts.filter(BucketingFilter(constraint))
        else:
            constraint = MultimodalSamplingConstraint(
                token_equivalent_duration=config.token_equivalent_duration,
                batch_size=config.batch_size,
                batch_tokens=config.batch_tokens,
                quadratic_factor=config.quadratic_factor,
                measure_total_length=config.measure_total_length,
            )
    else:
        if config.bucket_batch_size is not None:
            assert (
                bucket_duration_bins is not None
            ), "Cannot use bucket_batch_size option if bucket_duration_bins are not provided."
            constraint = FixedBucketBatchSizeConstraint2D(
                max_seq_len_buckets=bucket_duration_bins,
                batch_sizes=config.bucket_batch_size,
                strict_2d=config.bucketing_2d_strict_mode,
                max_ratio=config.max_tps if isinstance(config.max_tps, Sequence) else None,
            )
            cuts = cuts.filter(BucketingFilter(constraint))
        else:
            constraint = TimeConstraint(
                max_cuts=config.batch_size,
                max_duration=config.batch_duration,
                quadratic_duration=config.quadratic_duration,
            )
    return cuts, constraint


def determine_bucket_duration_bins(config):
    """
    Returns appropriate bucket bins based on configuration.
    If user provided them explicitly, we just pass them along;
    otherwise, we try to create provisional bins when min/max duration is available.
    We might return None if it's impossible to determine the bins without computing data statistics,
    in which case it will be automatically done at the start of training (but may take a few minutes).
    """
    if config.bucket_duration_bins is not None:
        # Bucket duration bins are provided: just use them.
        ans = OmegaConf.to_container(config.bucket_duration_bins)
        if isinstance(ans[0], Sequence):
            # 2D bucketing. Ensure we're using tuples for correct behaviour of '<' operator
            # between the bucket bin tuples and the output of measure_length.
            ans = [tuple(item) for item in ans]
        return ans
    # Bucket duration bins are not set.
    if config.use_multimodal_sampling:
        # For multimodal sampling it's currently impossible to define a linspace over durations
        # because the buckets are counted in the number of tokens.
        # The bins will be auto-estimated by lhotse at the cost of a slight lag in the training start.
        return None
    elif config.max_duration is not None and config.max_duration < float("inf"):
        # If max duration is provided, we can use that to compute uniformly distant bucket bins.
        # This is not optimal but should be close enough for users who didn't want to estimate these up-front.
        begin = config.min_duration if config.min_duration is not None and config.min_duration > 0 else 0.0
        end = config.max_duration
        return np.linspace(begin, end, config.num_buckets + 1)[1:-1].tolist()
    else:
        # If we don't know max_duration, we can't guess a reasonable estimate of the upper bound of
        # durations.
        # The bins will be auto-estimated by lhotse at the cost of a slight lag in the training start.
        return None


def make_structured_with_schema_warnings(config: Union[DictConfig, dict]) -> DictConfig:
    """
    Checks the schema and fills missing default option values.
    Warns the user if any of the fields are not supported by the current schema
    but does not raise exceptions.
    """
    default = OmegaConf.structured(LhotseDataLoadingConfig)
    if not isinstance(config, DictConfig):
        config = DictConfig(config)

    # Remove unsupported keys and warn about them.
    supported_keys = set(OmegaConf.to_container(default).keys())
    received_keys = set(OmegaConf.to_container(config).keys())
    unsupported_keys = received_keys - supported_keys
    unsupported_keys.discard("use_lhotse")
    if unsupported_keys:
        logging.warning(
            f"The following configuration keys are ignored by Lhotse dataloader: {','.join(unsupported_keys)}",
        )
    config = OmegaConf.masked_copy(config, list(supported_keys))

    config = OmegaConf.merge(default, config)

    if config.get("tarred_random_access", False):
        logging.warning(
            "Option 'tarred_random_access' is deprecated and replaced with 'skip_missing_manifest_entries'.",
        )
        config.skip_missing_manifest_entries = True
    if config.skip_missing_manifest_entries:
        logging.warning(
            "Note: skip_missing_manifest_entries is set to True. "
            "If any of your manifests and tar files are mismatched, the entire "
            "tar file will be skipped without warning. It's your responsibility "
            "to ensure data integrity with this setting."
        )

    return config


def tokenize(example, tokenizer):
    """Return the text in the example according to the provided tokenizer."""
    if isinstance(example, Cut):
        for s in example.supervisions:
            if s.text is not None:
                s.tokens = np.asarray(tokenizer(s.text, s.language))
    elif hasattr(example, "tokenize") and callable(example.tokenize):
        example = example.tokenize(tokenizer)
    else:
        raise RuntimeError(f"Unsupported type of example: {type(example)}")
    return example


def tokenize_with_prompt(example, tokenizer, prompt_format: str | PromptFormatter):
    """Tokenize the example with the provided tokenizer and prompt format."""
    if isinstance(prompt_format, str):
        prompt_format = PromptFormatter.resolve(prompt_format)(tokenizer)
    encoded = apply_prompt_format_fn(example, prompt_format)
    for key, value in encoded.items():
        setattr(example, key, value)
    return example


# The helper callables below exist to avoid passing lambdas into lhotse CutSet map/filter methods.
# Lambdas are not serializable across processes by pickle.
# Note: lhotse offers LHOTSE_DILL_ENABLED=1 and ``lhotse.lazy.set_dill_enabled(True)``
# to support pickling lambdas if its ever truly necessary.


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


def maybe_set_cuda_expandable_segments(enabled: bool):
    """
    Configures PyTorch memory allocator to expand existing allocated segments
    instead of re-allocating them when tensor shape grows.
    This can help speed up the training when sequence length and/or batch size change often,
    and makes GPU more robust towards OOM.

    See here for more details:
    pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    """
    if enabled and torch.cuda.is_available():
        if (
            (value := os.environ.get("PYTORCH_CUDA_ALLOC_CONF")) is not None
            and len(value) > 0
            and "expandable_segments:True" not in value
        ):
            warnings.warn(
                "You have set PYTORCH_CUDA_ALLOC_CONF without expandable_segments:True option. "
                "We're setting that option anyway. To disable it, set cuda_expandable_segments=False "
                "in NeMo dataloader configuration."
            )

        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except RuntimeError:
            logging.info(
                "Failed to set expandable_segments:True for PyTorch CUDA allocator. "
                "You may get training speed improvements if you enable this "
            )


def resample(example, sampling_rate):
    from nemo.collections.common.data.lhotse.text_adapters import NeMoMultimodalConversation

    if isinstance(example, Cut):
        return example.resample(sampling_rate)
    elif isinstance(example, NeMoMultimodalConversation):
        for turn in example.turns:
            if hasattr(turn, "cut"):
                turn.cut = turn.cut.resample(sampling_rate)
        return example
    else:
        return example


def _select_channel(cut, channel_selector: int | str) -> list:
    if isinstance(channel_selector, int):
        channel_idx = channel_selector
    elif isinstance(channel_selector, str):
        if channel_selector in cut.custom:
            channel_idx = cut.custom[channel_selector]
        else:
            raise ValueError(f"Channel selector {channel_selector} not found in cut.custom")

    if channel_idx >= cut.num_channels:
        raise ValueError(
            f"Channel index {channel_idx} is larger than the actual number of channels {cut.num_channels}"
        )

    if cut.num_channels == 1:
        # one channel available and channel_idx==0
        return cut
    else:
        # with_channels only defined on MultiCut
        return cut.with_channels(channel_idx)
