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

import numpy as np
import pytest
import torch.utils.data
from lhotse import CutSet, Seconds, SupervisionSegment
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.collation import collate_vectors
from lhotse.testing.dummies import dummy_cut
from lhotse.testing.random import deterministic_rng
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import (
    BucketingFilter,
    DurationTokenBinsFilter,
    FixedBucketBatchSizeConstraint2D,
    get_lhotse_dataloader_from_config,
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
    tokenize_with_prompt,
)
from nemo.collections.common.data.lhotse.text_adapters import TextExample
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


def make_cut(id_: int = 0, duration: Seconds = 1.0, num_tokens: int = 10):
    supervision = SupervisionSegment(f"blah-{id_}", f"blah-{id_}", 0.0, duration, text="a" * num_tokens)
    supervision.tokens = np.zeros((num_tokens,), dtype=np.int32).tolist()
    return dummy_cut(id_, duration=duration, supervisions=[supervision])


@pytest.fixture
def cuts():
    return CutSet(
        [make_cut(i, duration=2.0, num_tokens=4) for i in range(20)]
        + [make_cut(i, duration=2.0, num_tokens=8) for i in range(20)]
        + [make_cut(i, duration=2.0, num_tokens=12) for i in range(20)]
        + [make_cut(i, duration=8.0, num_tokens=8) for i in range(20)]
        + [make_cut(i, duration=8.0, num_tokens=12) for i in range(20)]
        + [make_cut(i, duration=8.0, num_tokens=16) for i in range(20)]
        + [make_cut(i, duration=14.0, num_tokens=12) for i in range(20)]
        + [make_cut(i, duration=14.0, num_tokens=16) for i in range(20)]
        + [make_cut(i, duration=14.0, num_tokens=20) for i in range(20)]
    )


def test_2d_bucketing_expected_bucket_allocation(cuts):

    duration_bins = [
        (5.0, 5),
        (5.0, 11),
        (5.0, 15),
        (7.0, 10),
        (7.0, 13),
        (7.0, 20),
        (8.0, 15),
        (8.0, 17),
        (8.0, 25),
        (15.0, 20),
        (15.0, 29),
        (15.0, 30),
    ]
    batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    sampler = DynamicBucketingSampler(
        cuts.repeat(),
        shuffle=True,
        duration_bins=duration_bins,
        constraint=FixedBucketBatchSizeConstraint2D(
            max_seq_len_buckets=duration_bins,
            batch_sizes=batch_sizes,
            strict_2d=False,
        ),
        buffer_size=1000,
        seed=0,
    )

    for batch_idx, batch in enumerate(sampler):
        # Run for 100 batches and check invariants on each.
        if batch_idx == 100:
            break
        # Note: batch_sizes are indexes into duration_bins when subtracting 1.
        # This way we can determine which bucket the data came from in this test.
        bin_index = len(batch) - 1
        max_duration, max_num_tokens = duration_bins[bin_index]
        for cut in batch:
            # First, check that the sampled examples are indeed below the max duration/num_tokens for its bucket.
            assert cut.duration <= max_duration
            assert len(cut.supervisions[0].tokens) <= max_num_tokens
            # Then, find the previous compatible bucket for each of training example's dimensions,
            # and verify that it was not possible to assign the example to that smaller bucket.
            # We should skip this for bucket_idx==0 (no previous buckets available).
            # Note: max will be an empty sequence in some cases, e.g. when it's the first bucket
            # with a given max_duration, it has the smallest max_num_tokens, leaving previous candidates list
            # for max_num_tokens empty.
            if bin_index > 0:
                # Check for smaller duration buckets (1st dim)
                candidates = [dur for dur, tok in duration_bins[:bin_index] if dur < max_duration]
                if candidates:  # Otherwise, it's already using the smallest bucket.
                    prev_max_duration = max(candidates)
                    assert cut.duration > prev_max_duration
                # Check for smaller token buckets (2nd dim)
                candidates = [
                    tok for dur, tok in duration_bins[:bin_index] if dur == max_duration and tok < max_num_tokens
                ]
                if candidates:  # Otherwise, it's already using the smallest bucket
                    prev_max_num_tokens = max(candidates)
                    assert len(cut.supervisions[0].tokens) > prev_max_num_tokens


@pytest.mark.parametrize(
    ["duration", "num_tokens", "should_keep", "bucket_idx"],
    [
        # Buckets for duration range [0.0-5.0]:
        # * Sweep num_tokens
        (2.0, 0, True, 0),
        (2.0, 5, True, 0),
        (2.0, 10, True, 0),
        (2.0, 11, True, 1),
        (2.0, 20, True, 1),
        (2.0, 21, True, 3),
        (2.0, 30, True, 3),
        (2.0, 31, False, None),
        # * Check the upper bound duration 5.0
        (5.0, 0, True, 0),
        (5.0, 5, True, 0),
        (5.0, 10, True, 0),
        (5.0, 11, True, 1),
        (5.0, 20, True, 1),
        (5.0, 21, True, 3),
        (5.0, 30, True, 3),
        (5.0, 31, False, None),
        # Buckets for duration range [5.0, 10.0]
        # * Sweep num_tokens
        (8.0, 0, True, 2),
        (8.0, 15, True, 2),
        (8.0, 16, True, 3),
        (8.0, 30, True, 3),
        (8.0, 31, False, None),
        # * Check the upper bound duration 10.0
        (10.0, 0, True, 2),
        (10.0, 15, True, 2),
        (10.0, 16, True, 3),
        (10.0, 30, True, 3),
        (10.0, 31, False, None),
        # Durations above max duration
        (20.0, 0, False, None),
        (20.0, 1000, False, None),
    ],
)
def test_2d_bucketing_filter_lenient(duration, num_tokens, should_keep, bucket_idx):
    buckets = [(5.0, 10), (5.0, 20), (10.0, 15), (10.0, 30)]
    batch_sizes = [4, 3, 2, 1]
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=False)
    filter_2d = BucketingFilter(constraint)

    cut = make_cut(duration=duration, num_tokens=num_tokens)
    assert filter_2d(cut) == should_keep
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == bucket_idx


@pytest.mark.parametrize(
    ["duration", "num_tokens", "should_keep", "bucket_idx"],
    [
        # Buckets for duration range [0.0-5.0]:
        # * Sweep num_tokens
        (2.0, 0, True, 0),
        (2.0, 5, True, 0),
        (2.0, 10, True, 0),
        (2.0, 11, True, 1),
        (2.0, 20, True, 1),
        (2.0, 21, False, None),  # <-- strict
        (2.0, 30, False, None),  # <-- strict
        (2.0, 31, False, None),
        # * Check the upper bound duration 5.0
        (5.0, 0, True, 0),
        (5.0, 5, True, 0),
        (5.0, 10, True, 0),
        (5.0, 11, True, 1),
        (5.0, 20, True, 1),
        (5.0, 21, False, None),  # <-- strict
        (5.0, 30, False, None),  # <-- strict
        (5.0, 31, False, None),
        # Buckets for duration range [5.0, 10.0]
        # * Sweep num_tokens
        (8.0, 0, True, 2),
        (8.0, 15, True, 2),
        (8.0, 16, True, 3),
        (8.0, 30, True, 3),
        (8.0, 31, False, None),
        # * Check the upper bound duration 10.0
        (10.0, 0, True, 2),
        (10.0, 15, True, 2),
        (10.0, 16, True, 3),
        (10.0, 30, True, 3),
        (10.0, 31, False, None),
        # Durations above max duration
        (20.0, 0, False, None),
        (20.0, 1000, False, None),
    ],
)
def test_2d_bucketing_filter_strict(duration, num_tokens, should_keep, bucket_idx):
    buckets = [(5.0, 10), (5.0, 20), (10.0, 15), (10.0, 30)]
    batch_sizes = [4, 3, 2, 1]
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=True)
    filter_2d = BucketingFilter(constraint)

    cut = make_cut(duration=duration, num_tokens=num_tokens)
    assert filter_2d(cut) == should_keep
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == bucket_idx


def test_2d_bucketing_filter_strict_max_ratio():
    buckets = [(5.0, 10), (5.0, 20), (10.0, 15), (10.0, 30)]
    max_ratio = [4.0, 4.0, 3.0, 3.0]
    batch_sizes = [4, 3, 2, 1]

    # Without max_ratio it works because both dims fit bucket at idx 1
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=True)
    filter_2d = BucketingFilter(constraint)
    cut = make_cut(duration=2.0, num_tokens=20)
    assert filter_2d(cut) == True
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == 1

    # With max_ratio it's filtered out because 20 / 2.0 = 10.0 but max_ratio is 4.0
    constraint = FixedBucketBatchSizeConstraint2D(buckets, batch_sizes, strict_2d=True, max_ratio=max_ratio)
    filter_2d = BucketingFilter(constraint)
    cut = make_cut(duration=2.0, num_tokens=20)
    assert filter_2d(cut) == False
    assert constraint.select_bucket(constraint.max_seq_len_buckets, cut) == None


class _Identity(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def test_2d_bucketing_strict_mode_flag_works(deterministic_rng, tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet([make_cut(0, duration=1.0, num_tokens=10), make_cut(0, duration=1.0, num_tokens=100)]).to_file(cuts_path)

    # Strict mode enabled
    dloader = get_lhotse_dataloader_from_config(
        {
            "cuts_path": cuts_path,
            "use_bucketing": True,
            "bucket_duration_bins": [(5.0, 10), (5.0, 20), (10.0, 150), (10.0, 300)],
            "bucket_batch_size": [1, 1, 1, 1],
            "bucketing_2d_strict_mode": True,
        },
        global_rank=0,
        world_size=1,
        dataset=_Identity(),
    )
    batches = [b for b in dloader]
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert len(batches[0][0].supervisions[0].tokens) == 10

    # Strict mode disabled
    dloader = get_lhotse_dataloader_from_config(
        {
            "cuts_path": cuts_path,
            "use_bucketing": True,
            "bucket_duration_bins": [(5.0, 10), (5.0, 20), (10.0, 150), (10.0, 300)],
            "bucket_batch_size": [1, 1, 1, 1],
            "bucketing_2d_strict_mode": False,
        },
        global_rank=0,
        world_size=1,
        dataset=_Identity(),
    )
    batches = [b for b in dloader]
    assert len(batches) == 2
    assert len(batches[0]) == 1
    assert len(batches[0][0].supervisions[0].tokens) == 100
    assert len(batches[1][0].supervisions[0].tokens) == 10


# Three duration groups with two sub-buckets each; the token bound of the last group is below the previous one.
FILTER_BINS = [(5.0, 10), (5.0, 20), (10.0, 15), (10.0, 30), (15.0, 12), (15.0, 25)]
FILTER_GRID = [
    (duration, num_tokens)
    for duration in (0.5, 5.0, 5.5, 8.0, 10.0, 12.0, 15.0, 15.5, 20.0)
    for num_tokens in (0, 10, 11, 20, 21, 25, 26, 30, 31)
]


def _fits_filter_bins(duration: float, num_tokens: int) -> bool:
    """The rule of strict 2D bucketing for FILTER_BINS, written out per duration group."""
    for max_duration, max_tokens in ((5.0, 20), (10.0, 30), (15.0, 25)):
        if duration <= max_duration:
            return num_tokens <= max_tokens
    return False


def _cut_id(index: int) -> str:
    return make_cut(index).id


FILTER_GRID_IDS = sorted(_cut_id(i) for i in range(len(FILTER_GRID)))
FILTER_GRID_KEPT_IDS = sorted(_cut_id(i) for i, lens in enumerate(FILTER_GRID) if _fits_filter_bins(*lens))


def _filter_grid_cuts() -> CutSet:
    return CutSet([make_cut(i, duration=lens[0], num_tokens=lens[1]) for i, lens in enumerate(FILTER_GRID)])


def _sampled_ids(config: dict, tokenizer=None) -> list[str]:
    sampler, _ = get_lhotse_sampler_from_config(
        make_structured_with_schema_warnings(OmegaConf.create(config)),
        global_rank=0,
        world_size=1,
        tokenizer=tokenizer,
    )
    return sorted(cut.id for batch in sampler for cut in batch)


def _bucketing_2d_config(cuts_path, bins) -> dict:
    """Strict 2D bucketing on the bins, the reference for the selection that filter_duration_bins makes."""
    return {
        "cuts_path": str(cuts_path),
        "force_finite": True,
        "bucket_duration_bins": bins,
        "bucket_batch_size": [1] * len(bins),
        "concurrent_bucketing": False,
    }


def _packed_sampling_config(cuts_path) -> dict:
    """Packed sequence sampling with budgets on the audio seconds and the output tokens of a batch."""
    return {
        "cuts_path": str(cuts_path),
        "force_finite": True,
        "use_multimodal_sampling": True,
        "use_packed_sequence_sampling": True,
        "measure_total_length": False,
        "batch_tokens": 64,
        "batch_output_tokens": 256,
        "audio_token_estimator": {
            "preprocessor": {"n_fft": 16000, "hop_length": 16000, "stft_pad_amount": 8000},
            "subsampling": [],
        },
    }


def test_duration_token_bins_filter_keeps_what_strict_2d_bucketing_keeps():
    cuts = _filter_grid_cuts()
    bucketing_filter = BucketingFilter(
        FixedBucketBatchSizeConstraint2D(FILTER_BINS, [1] * len(FILTER_BINS), strict_2d=True)
    )

    kept = sorted(cut.id for cut in cuts.filter(DurationTokenBinsFilter(FILTER_BINS)))

    assert kept == FILTER_GRID_KEPT_IDS
    assert kept == sorted(cut.id for cut in cuts.filter(bucketing_filter))
    # The second sub-bucket of the first group, and the lower token bound of the last group.
    assert _cut_id(FILTER_GRID.index((5.0, 20))) in kept
    assert _cut_id(FILTER_GRID.index((5.0, 21))) not in kept
    assert _cut_id(FILTER_GRID.index((10.0, 30))) in kept
    assert _cut_id(FILTER_GRID.index((12.0, 26))) not in kept
    assert _cut_id(FILTER_GRID.index((15.5, 0))) not in kept


@pytest.mark.parametrize(
    "bins",
    [
        [],
        5.0,
        "5.0",
        [5.0, 10.0],
        [(5.0, 10, 1)],
        [(10.0, 15), (5.0, 20)],
        [(5.0, 20), (5.0, 10)],
        [(5.0, float("nan"))],
        [(float("inf"), 10)],
        [(0.0, 10)],
        [(-5.0, 10)],
        [(5.0, -1)],
        [("5.0", "10")],
        [(5.0, None)],
        [(True, 10)],
    ],
)
def test_duration_token_bins_filter_rejects_invalid_bins(bins):
    with pytest.raises(ValueError, match="filter_duration_bins must be a non-empty, ascendingly sorted list"):
        DurationTokenBinsFilter(bins)


def test_duration_token_bins_filter_rejects_max_tps_of_other_length():
    with pytest.raises(ValueError, match="got 2 values for 6 bins"):
        DurationTokenBinsFilter(FILTER_BINS, max_tps=[2.0, 2.0])


def test_duration_token_bins_filter_counts_input_ids_of_prompted_cuts():
    cut = make_cut(duration=2.0, num_tokens=5)
    cut.input_ids = np.zeros((12,), dtype=np.int32)
    bucketing_filter = BucketingFilter(FixedBucketBatchSizeConstraint2D([(5.0, 8)], [1], strict_2d=True))

    # The 12 tokens of prompt and answer count, not the 5 tokens of the supervision.
    assert DurationTokenBinsFilter([(5.0, 8)])(cut) is False
    assert bucketing_filter(cut) is False
    assert DurationTokenBinsFilter([(5.0, 12)])(cut) is True


def test_duration_token_bins_filter_passes_text_examples():
    assert DurationTokenBinsFilter(FILTER_BINS)(TextExample("a text longer than every bin " * 20)) is True


def test_duration_token_bins_filter_raises_for_untokenized_cut():
    cut = dummy_cut(0, duration=1.0, supervisions=[SupervisionSegment("", "", 0, 1.0, text="dummy")])
    with pytest.raises((AssertionError, ValueError), match="untokenized supervisions"):
        DurationTokenBinsFilter(FILTER_BINS)(cut)


def test_filter_duration_bins_applies_a_max_tps_list_as_strict_2d_bucketing(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _filter_grid_cuts().to_file(cuts_path)
    max_tps = [2.0, 2.0, 3.0, 3.0, 1.5, 1.5]
    config = {"cuts_path": str(cuts_path), "force_finite": True, "batch_size": 4, "max_tps": max_tps}

    kept = _sampled_ids({**config, "filter_duration_bins": FILTER_BINS})

    assert kept == _sampled_ids({**_bucketing_2d_config(cuts_path, FILTER_BINS), "max_tps": max_tps})
    assert 0 < len(kept) < len(FILTER_GRID_KEPT_IDS)
    # 8 seconds: 20 tokens are 2.5 per second, within the 3.0 of the bin (10.0, 30); 25 tokens are above it.
    assert _cut_id(FILTER_GRID.index((8.0, 20))) in kept
    assert _cut_id(FILTER_GRID.index((8.0, 25))) not in kept


def test_filter_duration_bins_rejects_a_scalar_in_the_config(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _filter_grid_cuts().to_file(cuts_path)
    with pytest.raises(ValueError, match="filter_duration_bins must be"):
        _sampled_ids({"cuts_path": str(cuts_path), "batch_size": 4, "filter_duration_bins": 5.0})


@pytest.mark.parametrize(
    "sampling",
    [
        {"batch_size": 4},
        {"batch_duration": 40.0},
        {"bucket_duration_bins": [5.0, 10.0], "batch_duration": 40.0, "concurrent_bucketing": False},
    ],
    ids=["batch_size", "batch_duration", "bucketing_1d"],
)
def test_filter_duration_bins_selects_the_cuts_of_strict_2d_bucketing(tmp_path, sampling):
    cuts_path = tmp_path / "cuts.jsonl"
    _filter_grid_cuts().to_file(cuts_path)
    config = {"cuts_path": str(cuts_path), "force_finite": True, **sampling}

    kept = _sampled_ids({**config, "filter_duration_bins": FILTER_BINS})

    assert kept == FILTER_GRID_KEPT_IDS
    assert kept == _sampled_ids(_bucketing_2d_config(cuts_path, FILTER_BINS))
    # Without the option nothing is filtered.
    assert _sampled_ids(config) == FILTER_GRID_IDS


def test_filter_duration_bins_with_packed_sequence_sampling(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _filter_grid_cuts().to_file(cuts_path)
    config = {**_packed_sampling_config(cuts_path), "pretokenize": False}

    kept = _sampled_ids({**config, "filter_duration_bins": FILTER_BINS}, tokenizer=object())

    assert kept == FILTER_GRID_KEPT_IDS
    assert kept == _sampled_ids(_bucketing_2d_config(cuts_path, FILTER_BINS))
    assert _sampled_ids(config, tokenizer=object()) == FILTER_GRID_IDS


@pytest.fixture(scope="module")
def filter_bins_tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("filter_bins_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text("a short answer\na much longer answer with many more words in it\n" * 4)
    create_spt_model(
        str(text_path),
        vocab_size=64,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        remove_extra_whitespaces=True,
        bos=True,
        eos=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


@pytest.mark.parametrize("prompt_format", [None, "plain"])
def test_filter_duration_bins_counts_the_tokens_of_the_dataloader_tokenizer(
    tmp_path, filter_bins_tokenizer, prompt_format
):
    short, long = "a short answer", "a much longer answer with many more words in it " * 3
    cuts = []
    for index, (duration, text) in enumerate([(2.0, short), (2.0, long), (8.0, short), (8.0, long), (12.0, short)]):
        supervision = SupervisionSegment(f"sup-{index}", f"rec-{index}", 0.0, duration, text=text)
        cuts.append(dummy_cut(index, duration=duration, supervisions=[supervision]))
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet(cuts).to_file(cuts_path)
    if prompt_format is None:
        num_tokens = [len(filter_bins_tokenizer.text_to_ids(text)) for text in (short, long)]
    else:
        num_tokens = [
            len(tokenize_with_prompt(cut, filter_bins_tokenizer, prompt_format).input_ids) for cut in cuts[:2]
        ]
    # Up to 5 seconds only the short text fits, up to 10 seconds both texts, and nothing longer.
    bins = [(5.0, num_tokens[0]), (10.0, num_tokens[1])]
    tokenization = {"pretokenize": True, "prompt_format": prompt_format}

    kept = _sampled_ids(
        {**_packed_sampling_config(cuts_path), **tokenization, "filter_duration_bins": bins}, filter_bins_tokenizer
    )

    assert num_tokens[0] < num_tokens[1]
    assert kept == [cuts[0].id, cuts[2].id, cuts[3].id]
    assert kept == _sampled_ids({**_bucketing_2d_config(cuts_path, bins), **tokenization}, filter_bins_tokenizer)


class _CollatedTokensDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        tokens = [torch.as_tensor(cut.supervisions[0].tokens) for cut in cuts]
        return [cut.id for cut in cuts], torch.tensor([len(t) for t in tokens]), collate_vectors(tokens, 0)


@pytest.mark.parametrize("enabled", [False, True])
def test_filter_duration_bins_reaches_collated_dataloader(tmp_path, enabled):
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet([make_cut(i, duration=1.0, num_tokens=num_tokens) for i, num_tokens in enumerate([2, 3, 6])]).to_file(
        cuts_path
    )
    config = {"cuts_path": str(cuts_path), "force_finite": True, "batch_size": 3, "num_workers": 0, "shuffle": False}
    if enabled:
        config["filter_duration_bins"] = [(2.0, 3)]

    dataloader = get_lhotse_dataloader_from_config(
        config, global_rank=0, world_size=1, dataset=_CollatedTokensDataset()
    )
    batches = list(dataloader)

    expected = [2, 3] if enabled else [2, 3, 6]
    assert len(batches) == 1
    ids, lengths, tokens = batches[0]
    assert ids == [_cut_id(i) for i in range(len(expected))]
    assert lengths.tolist() == expected
    assert tokens.shape == (len(expected), max(expected))
