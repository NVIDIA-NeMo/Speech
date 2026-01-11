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
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union

import hydra
from lhotse import CutSet, RecordingSet
from lhotse.cut import Cut
from lhotse.dataset import CutConcatenate, ReverbWithImpulseResponse
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.base import CutSampler
from omegaconf import DictConfig, ListConfig

from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset


def augment_examples(cuts: CutSet, config: DictConfig) -> CutSet:
    """
    Apply augmentation to individual examples in a CutSet.

    When ``config.augment_examples`` is set, it must be a list of dicts such as follows:

    .. code-block:: yaml

        augment_examples:
          - _target_:
    """
    if not config.augment_examples:
        return augment_examples_legacy(cuts, config)
    assert isinstance(config.augment_examples, (list, ListConfig))
    for item in config.augment_examples:
        if callable(item):
            aug = item
        else:
            assert isinstance(item, (dict, DictConfig))
            aug = hydra.utils.instantiate(item)
        cuts = aug(cuts)
    return cuts


def augment_batch(sampler: CutSampler, config: DictConfig) -> CutSampler:
    if config.augment_batch:
        assert isinstance(config.augment_batch, (list, ListConfig))
        for item in config.augment_batch:
            if callable(item):
                aug = item
            else:
                assert isinstance(item, (dict, DictConfig))
                aug = hydra.utils.instantiate(item)
            sampler = sampler.map(aug)
        return sampler
    return augment_batch_legacy(sampler, config)


def augment_examples_legacy(cuts: CutSet, config: DictConfig) -> CutSet:
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

    return cuts


def augment_batch_legacy(sampler: CutSampler, config: DictConfig) -> CutSampler:
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
    return sampler


# Augmentation objects that can be instantiated from the YAML config.


@dataclass
class Mix:
    """
    Noise/music/speech mixing augmentation that reads examples from ``noise_path``.
    The value for ``noise_path`` can be a path to NeMo manifest, Lhotse manifest, or a YAML ``input_cfg``.
    """

    noise_path: str
    noise_snr: tuple[float, float] = (10, 20)
    prob: float = (0.5,)
    seed: int = "trng"
    random_mix_offset: bool = True

    def __call__(self, cuts: CutSet) -> CutSet:
        noise_cuts = guess_parse_cutset(self.noise_path)
        return cuts.mix(
            cuts=noise_cuts,
            snr=self.noise_snr,
            mix_prob=self.prob,
            seed=self.seed,
            random_mix_offset=self.random_mix_offset,
        )


@dataclass
class PerturbSpeed:
    """
    Applies speed perturbation on examples.
    Following Kaldi-style augmentation, it replicates the data for each factor specified.
    """

    factors: list[float] = field(default_factory=lambda: [0.9, 1.1])

    def __call__(self, cuts: CutSet) -> CutSet:
        perturbed = [cuts.perturb_speed(factor) for factor in self.factors]
        # mux ensures uniform blending of perturbed and non-perturbed variants throughout time
        return CutSet.mux(cuts, *perturbed)


@dataclass
class Truncate:
    """
    Truncates each example to a fixed duration.
    It may be smaller than ``duration`` if there was not enough signal.
    ``keep_excessive_supervisions`` determines whether supervisions cut in the middle
    are kept (True, default) or discarded (False).
    """

    duration: float
    offset_type: str = "random"  # start | end | random
    keep_excessive_supervisions: bool = True

    def __call__(self, cuts: CutSet) -> CutSet:
        return cuts.truncate(
            max_duration=self.duration,
            offset_type=self.offset_type,
            keep_excessive_supervisions=self.keep_excessive_supervisions,
        )


@dataclass
class CutIntoWindows:
    """
    Cuts each example into one or more examples of duration at most ``duration``,
    with hop size determined by ``hop`` (equal to ``duration`` by default).
    The last window may be smaller than ``duration`` if there was not enough signal.
    ``keep_excessive_supervisions`` determines whether supervisions cut in the middle
    are kept (True, default) or discarded (False).
    """

    duration: float
    hop: float | None = None
    keep_excessive_supervisions: bool = True

    def __call__(self, cuts: CutSet) -> CutSet:
        return cuts.cut_into_windows(
            duration=self.duration,
            hop=self.hop,
            keep_excessive_supervisions=self.keep_excessive_supervisions,
        )


@dataclass
class MinPadding:
    """
    If needed, pads each example to have at least ``duration``.
    No-op for examples longer than ``duration``.
    """

    duration: float
    pad_direction: str = "right"  # right | left | both | random

    def __call__(self, cuts: CutSet) -> CutSet:
        return cuts.pad(duration=self.duration, direction=self.pad_direction, preserve_id=True)


@dataclass
class ReverbRIR:
    """
    Apply RIR reverberation to individual examples (CutSet input) or batches (CutSampler input).
    Two input variants are provided for convenience only as they are not functionally different.

    See Lhotse documentation for details:
    https://lhotse.readthedocs.io/en/latest/api.html#lhotse.cut.MonoCut.reverb_rir
    """

    rir_path: str | None = None
    prob: float = 0.5
    normalize_output: bool = True
    early_only: bool = False
    rir_channels: list = field(default_factory=lambda: [0])
    seed: int | str = "trng"

    def __call__(self, data: Union[CutSet, CutSampler]) -> Union[CutSet, CutSampler]:
        rir_recordings = RecordingSet.from_file(self.rir_path) if self.rir_path is not None else None
        rng = random.Random(resolve_seed(self.seed))
        if isinstance(data, CutSampler):  # RIR augmentation used in Batch context
            return data.map(
                ReverbWithImpulseResponse(
                    rir_recordings=rir_recordings,
                    p=self.prob,
                    normalize_output=self.normalize_output,
                    early_only=self.early_only,
                    rir_channels=self.rir_channels,
                    randgen=rng,
                )
            )
        elif isinstance(data, CutSet):
            return data.map(
                partial(
                    _reverb_one,
                    rir_recordings=rir_recordings,
                    prob=self.prob,
                    normalize_output=self.normalize_output,
                    early_only=self.early_only,
                    rir_channels=self.rir_channels,
                    rng=rng,
                )
            )
        else:
            raise RuntimeError(f"Unsupported input type for ReverbRIR: {type(data)}")


@dataclass
class ConcatenateExamples:
    """
    Cut concatenation will produce longer samples out of shorter samples
    by gluing them together from the shortest to longest not to exceed a duration
    of longest_cut * duration_factor (greedy knapsack algorithm for minimizing padding).
    Useful e.g. for simulated code-switching in multilingual setups.
    We follow concatenation by ``merge_supervisions`` which creates a single supervision
    object with texts joined by a whitespace so that "regular" dataset classes don't
    have to add a special support for multi-supervision cuts.
    """

    gap_seconds: float = 0.1
    duration_factor: float = 1.0
    db_norm: float = -25.0
    merge_supervisions: bool = True

    def __call__(self, sampler: CutSampler) -> CutSampler:
        sampler = sampler.map(
            CutConcatenate(
                gap=self.gap_seconds,
                duration_factor=self.duration_factor,
            )
        )
        if self.db_norm is not None:
            sampler = sampler.map(partial(_normalize_loudness, db_norm=self.db_norm))
        if self.merge_supervisions:
            sampler = sampler.map(_merge_supervisions)
        return sampler


# The helper callables below exist to avoid passing lambdas into lhotse CutSet map/filter methods.
# Lambdas are not serializable across processes by pickle.
# Note: lhotse offers LHOTSE_DILL_ENABLED=1 and ``lhotse.lazy.set_dill_enabled(True)``
# to support pickling lambdas if its ever truly necessary.


def _normalize_loudness(cuts: CutSet, db_norm: float) -> CutSet:
    return cuts.normalize_loudness(target=db_norm, mix_first=False)


def _merge_supervisions(cuts: CutSet) -> CutSet:
    return cuts.merge_supervisions()


def _reverb_one(
    cut: Cut,
    rir_recordings: Optional[RecordingSet],
    prob: float,
    normalize_output: bool,
    early_only: bool,
    rir_channels: list,
    rng: random.Random,
) -> Cut:
    if rng.random() > prob:
        return cut
    rir = rng.choice(rir_recordings) if rir_recordings is not None else None
    return cut.reverb_rir(
        rir_recording=rir,
        normalize_output=normalize_output,
        early_only=early_only,
        affix_id=True,
        rir_channels=rir_channels,
    )
