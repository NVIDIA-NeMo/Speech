# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from copy import deepcopy
from types import SimpleNamespace

import pytest
from lhotse import CutSet, SupervisionSegment
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler
from lhotse.dataset.iterable_dataset import IdentityDataset, IterableDatasetWrapper
from lhotse.indexing import create_jsonl_index
from lhotse.lazy import LazyIndexedManifestIterator
from lhotse.testing.dummies import dummy_cut
from lhotse.utils import fastcopy
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.audio_token_estimator import AudioTokenEstimator
from nemo.collections.common.data.lhotse.dataloader import (
    _auto_detect_bucketing_and_validate_batch_size,
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
    tokenize_with_prompt,
)
from nemo.collections.common.data.lhotse.packed_sequence_sampler import (
    PackedSequenceDynamicBucketingSampler,
    PackedSequenceDynamicCutSampler,
    _fit_output_cap,
    _select_best_fit_indices,
)
from nemo.collections.common.data.lhotse.sampling import MultimodalSamplingConstraint, PackedTokenConstraint
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


def _make_cuts(durations=(7.0, 4.0, 6.0, 3.0)):
    return CutSet.from_cuts(dummy_cut(index, duration=duration) for index, duration in enumerate(durations))


def _make_cuts_with_tokens(durations, token_counts):
    """Cuts whose single supervision carries ``token_counts[i]`` tokens in its ``tokens`` custom field."""
    cuts = []
    for index, (duration, count) in enumerate(zip(durations, token_counts)):
        cut = dummy_cut(index, duration=duration)
        supervision = SupervisionSegment(
            id=f"sup-{index}",
            recording_id=cut.recording_id,
            start=0.0,
            duration=duration,
            custom={"tokens": list(range(count))},
        )
        cuts.append(fastcopy(cut, supervisions=[supervision]))
    return CutSet.from_cuts(cuts)


def _make_sampler(
    *cuts,
    batch_tokens=10,
    batch_size=None,
    quadratic_factor=None,
    packing_buffer_size=4,
    batch_output_tokens=None,
):
    return PackedSequenceDynamicCutSampler(
        *cuts,
        constraint=MultimodalSamplingConstraint(
            token_equivalent_duration=1.0,
            audio_token_estimator=AudioTokenEstimator.from_config(
                {
                    "preprocessor": {
                        "n_fft": 16000,
                        "hop_length": 16000,
                        "stft_pad_amount": 8000,
                    },
                    "subsampling": [],
                },
                sample_rate=16000,
            ),
            batch_size=batch_size,
            batch_tokens=batch_tokens,
            batch_output_tokens=batch_output_tokens,
            quadratic_factor=quadratic_factor,
            measure_total_length=False,
            use_packed_sequence_sampling=True,
        ),
        # PackedSequenceDynamicCutSampler consumes this value but deliberately
        # passes shuffle=False to its DynamicCutSampler parent.
        shuffle=True,
        packing_buffer_size=packing_buffer_size,
        seed=0,
    )


def _make_bucketed_sampler(
    *cuts,
    batch_tokens=10,
    batch_size=None,
    quadratic_factor=None,
    packing_buffer_size=4,
    buffer_size=32,
    shuffle=False,
    drop_last=False,
    seed=0,
    duration_bins=(100,),
    bucket_batch_size=None,
    concurrent=False,
    batch_output_tokens=None,
):
    return PackedSequenceDynamicBucketingSampler(
        *cuts,
        constraint=MultimodalSamplingConstraint(
            token_equivalent_duration=1.0,
            audio_token_estimator=AudioTokenEstimator.from_config(
                {
                    "preprocessor": {
                        "n_fft": 16000,
                        "hop_length": 16000,
                        "stft_pad_amount": 8000,
                    },
                    "subsampling": [],
                },
                sample_rate=16000,
            ),
            batch_size=batch_size,
            batch_tokens=batch_tokens,
            batch_output_tokens=batch_output_tokens,
            quadratic_factor=quadratic_factor,
            measure_total_length=False,
            use_packed_sequence_sampling=True,
            bucket_duration_bins=list(duration_bins) if bucket_batch_size is not None else None,
            bucket_batch_size=bucket_batch_size,
        ),
        duration_bins=list(duration_bins),
        buffer_size=buffer_size,
        packing_buffer_size=packing_buffer_size,
        concurrent=concurrent,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )


def _drain(iterator):
    batches = []
    while True:
        try:
            batches.append(next(iterator))
        except StopIteration:
            return batches


def _batch_ids(batches):
    return [[cut.id for cut in batch] for batch in batches]


@pytest.mark.parametrize(
    ("packed", "sampler_type"),
    [(False, DynamicCutSampler), (True, PackedSequenceDynamicCutSampler)],
)
def test_packed_sequence_config_flag_selects_exact_sampler(tmp_path, packed, sampler_type):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "force_finite": True,
                "shuffle": True,
                "shuffle_buffer_size": 4,
                "packing_buffer_size": 7,
                "num_workers": 0,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 10,
                "token_equivalent_duration": 1.0,
                "audio_token_estimator": {
                    "preprocessor": {
                        "n_fft": 16000,
                        "hop_length": 16000,
                        "stft_pad_amount": 8000,
                    },
                    "subsampling": [],
                },
                "use_packed_sequence_sampling": packed,
                "measure_total_length": False,
                "seed": 0,
                "shard_seed": 0,
            }
        )
    )

    sampler, _ = get_lhotse_sampler_from_config(
        config,
        global_rank=0,
        world_size=1,
        tokenizer=object(),
    )

    assert type(sampler) is sampler_type
    if packed:
        assert sampler.packing_buffer_size == 7
    else:
        assert sampler.shuffle_buffer_size == 4
    assert sampler.shuffle is not packed


def test_packing_buffer_legacy_alias_accepts_structured_input():
    legacy_config = OmegaConf.create({"shuffle_buffer_size": 7})
    OmegaConf.set_struct(legacy_config, True)

    config = make_structured_with_schema_warnings(legacy_config)

    assert "packing_buffer_size" not in legacy_config
    assert config.shuffle_buffer_size == 7
    assert config.packing_buffer_size == 7


def test_packed_sequence_bucketing_config_selects_exact_bucketed_sampler(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    raw_config = OmegaConf.create(
        {
            "cuts_path": str(cuts_path),
            "force_finite": True,
            "shuffle": False,
            "num_workers": 0,
            "use_bucketing": True,
            "num_buckets": 2,
            "bucket_duration_bins": [100],
            "bucket_buffer_size": 17,
            "packing_buffer_size": 3,
            "concurrent_bucketing": False,
            "use_multimodal_sampling": True,
            "pretokenize": False,
            "batch_tokens": 10,
            "token_equivalent_duration": 1.0,
            "audio_token_estimator": {
                "preprocessor": {
                    "n_fft": 16000,
                    "hop_length": 16000,
                    "stft_pad_amount": 8000,
                },
                "subsampling": [],
            },
            "use_packed_sequence_sampling": True,
            "measure_total_length": False,
            "seed": 0,
            "shard_seed": 0,
        }
    )
    config = make_structured_with_schema_warnings(deepcopy(raw_config))

    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=object())

    assert type(sampler) is PackedSequenceDynamicBucketingSampler
    assert sampler.buffer_size == 17
    assert sampler.packing_buffer_size == 3

    legacy_raw_config = deepcopy(raw_config)
    del legacy_raw_config["packing_buffer_size"]
    legacy_raw_config.shuffle_buffer_size = 5
    legacy_config = make_structured_with_schema_warnings(legacy_raw_config)
    legacy_sampler, _ = get_lhotse_sampler_from_config(
        legacy_config,
        global_rank=0,
        world_size=1,
        tokenizer=object(),
    )
    assert legacy_sampler.packing_buffer_size == 5


def test_best_fit_subset_is_exact_and_prefers_earlier_candidates():
    assert _select_best_fit_indices([6, 6, 3], capacity=6) == [0]
    assert _select_best_fit_indices([5, 3, 2], capacity=5) == [0]
    assert _select_best_fit_indices([4, 3, 2], capacity=5) == [1, 2]
    assert _select_best_fit_indices([3, 2, 2], capacity=4, max_items=1) == [0]


def test_packed_fixed_bucket_config_selects_packed_sampler_and_preserves_caps(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "force_finite": True,
                "shuffle": False,
                "num_workers": 0,
                "use_bucketing": True,
                "bucket_duration_bins": [5, 10],
                "bucket_batch_size": [2, 1],
                "concurrent_bucketing": False,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 10,
                "token_equivalent_duration": 1.0,
                "use_packed_sequence_sampling": True,
                "seed": 0,
                "shard_seed": 0,
            }
        )
    )

    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=object())

    assert type(sampler) is PackedSequenceDynamicBucketingSampler
    assert sampler.constraint.max_examples_for_length(4) == 2
    assert sampler.constraint.max_examples_for_length(7) == 1
    assert sampler.constraint._internal.batch_tokens == 10
    assert sampler.constraint.bucket_duration_bins == [5, 10]
    assert sampler.constraint.bucket_batch_size == [2, 1]


def test_packed_fixed_bucket_config_requires_exact_token_budget(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "bucket_duration_bins": [5, 10],
                "bucket_batch_size": [2, 1],
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "token_equivalent_duration": 1.0,
                "use_packed_sequence_sampling": True,
            }
        )
    )

    with pytest.raises(ValueError, match="Packed per-bucket caps require batch_tokens"):
        get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=object())


def test_packed_two_dimensional_fixed_buckets_preserve_legacy_sampler(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "bucket_duration_bins": [[5, 10], [10, 20]],
                "bucket_batch_size": [2, 1],
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 10,
                "token_equivalent_duration": 1.0,
                "use_packed_sequence_sampling": True,
            }
        )
    )

    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=object())

    assert type(sampler) is DynamicBucketingSampler


def test_packed_fixed_buckets_filter_examples_above_highest_boundary(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((4.0, 11.0)).to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "force_finite": True,
                "shuffle": False,
                "num_workers": 0,
                "use_bucketing": True,
                "bucket_duration_bins": [5, 10],
                "bucket_batch_size": [2, 1],
                "concurrent_bucketing": False,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 12,
                "token_equivalent_duration": 1.0,
                "audio_token_estimator": {
                    "preprocessor": {
                        "n_fft": 16000,
                        "hop_length": 16000,
                        "stft_pad_amount": 8000,
                    },
                    "subsampling": [],
                },
                "use_packed_sequence_sampling": True,
                "seed": 0,
                "shard_seed": 0,
            }
        )
    )
    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=object())
    batches = list(sampler)

    assert _batch_ids(batches) == [["dummy-mono-cut-0000"]]


def test_packed_fixed_bucket_caps_intersect_global_batch_size(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((3.0, 2.0)).to_file(cuts_path)

    batches = list(
        _make_bucketed_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            batch_tokens=10,
            batch_size=1,
            duration_bins=(5,),
            bucket_batch_size=(2,),
        )
    )

    assert [len(batch) for batch in batches] == [1, 1]


def test_bucketed_packed_sampler_enforces_per_bucket_example_caps(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0)).to_file(cuts_path)

    batches = list(
        _make_bucketed_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            batch_tokens=10,
            packing_buffer_size=4,
            buffer_size=4,
            duration_bins=(5, 10),
            bucket_batch_size=(2, 1),
        )
    )

    assert sorted(cut.id for batch in batches for cut in batch) == [
        "dummy-mono-cut-0000",
        "dummy-mono-cut-0001",
        "dummy-mono-cut-0002",
        "dummy-mono-cut-0003",
    ]
    for batch in batches:
        bucket_index = 0 if max(cut.num_tokens for cut in batch) <= 5 else 1
        assert len(batch) <= (2, 1)[bucket_index]
    assert any(len(batch) == 2 for batch in batches)


def test_bucketed_packed_sampler_best_fits_beyond_prefix_and_enforces_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)

    batches = list(_make_bucketed_sampler(CutSet.from_jsonl_lazy(cuts_path), buffer_size=4))

    # A prefix batcher stops at the first 7-token example; best-fit packing
    # reaches past the non-fitting 4/6-token candidates for the 3-token item.
    assert _batch_ids(batches) == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0003"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
    ]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [10, 10]


def test_bucketed_packed_sampler_anchor_emits_every_example_without_starvation(
    tmp_path,
):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((6.0, 6.0, 4.0, 4.0, 4.0, 4.0)).to_file(cuts_path)

    batches = list(
        _make_bucketed_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            packing_buffer_size=4,
            buffer_size=6,
        )
    )

    assert _batch_ids(batches)[:2] == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0002"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0003"],
    ]
    assert sorted(cut_id for batch in _batch_ids(batches) for cut_id in batch) == [
        f"dummy-mono-cut-{index:04d}" for index in range(6)
    ]
    assert all(sum(cut.num_tokens for cut in batch) <= 10 for batch in batches)


def test_bucketed_packed_sampler_never_mixes_selected_buckets(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((2.0, 6.0, 2.0, 6.0, 2.0, 6.0, 2.0, 6.0)).to_file(cuts_path)
    short_ids = {f"dummy-mono-cut-{index:04d}" for index in (0, 2, 4, 6)}
    long_ids = {f"dummy-mono-cut-{index:04d}" for index in (1, 3, 5, 7)}

    batches = list(
        _make_bucketed_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            buffer_size=8,
            duration_bins=(4,),
        )
    )

    assert sorted(cut.id for batch in batches for cut in batch) == sorted(short_ids | long_ids)
    for batch in batches:
        batch_ids = {cut.id for cut in batch}
        assert batch_ids <= short_ids or batch_ids <= long_ids


def test_bucketed_packed_sampler_tuple_inputs(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _make_cuts().to_file(first_path)
    _make_cuts((1.0, 1.0, 1.0, 1.0)).to_file(second_path)

    batches = list(
        _make_bucketed_sampler(
            CutSet.from_jsonl_lazy(first_path),
            CutSet.from_jsonl_lazy(second_path),
            buffer_size=4,
        )
    )

    assert all(isinstance(batch, tuple) and len(batch) == 2 for batch in batches)
    assert [_batch_ids([left]) for left, _ in batches] == [
        [["dummy-mono-cut-0000", "dummy-mono-cut-0003"]],
        [["dummy-mono-cut-0001", "dummy-mono-cut-0002"]],
    ]
    assert [[cut.id for cut in right] for _, right in batches] == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0003"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
    ]


def test_bucketed_packed_sampler_shuffle_is_deterministic_for_seed(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((6.0, 5.0, 4.0, 3.0, 2.0, 4.0, 3.0, 2.0)).to_file(cuts_path)

    def sample():
        return _batch_ids(
            list(
                _make_bucketed_sampler(
                    CutSet.from_jsonl_lazy(cuts_path),
                    packing_buffer_size=4,
                    buffer_size=8,
                    shuffle=True,
                    seed=31,
                )
            )
        )

    assert sample() == sample()


@pytest.mark.parametrize(
    ("drop_last", "expected_ids"),
    [
        (False, [0, 1, 2, 3]),
        (True, [0, 1]),
    ],
)
def test_bucketed_packed_sampler_finite_drop_last(tmp_path, drop_last, expected_ids):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((5.0, 5.0, 2.0, 2.0)).to_file(cuts_path)

    batches = list(
        _make_bucketed_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            buffer_size=4,
            drop_last=drop_last,
        )
    )

    assert [cut.id for batch in batches for cut in batch] == [f"dummy-mono-cut-{index:04d}" for index in expected_ids]


def test_bucketed_packed_sampler_indexed_resume_preserves_arbitrary_queue_removals(
    tmp_path,
):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    create_jsonl_index(cuts_path)

    def make_source():
        return CutSet(LazyIndexedManifestIterator(cuts_path))

    sampler = _make_bucketed_sampler(make_source(), buffer_size=4)
    iterator = iter(sampler)
    assert _batch_ids([next(iterator)]) == [["dummy-mono-cut-0000", "dummy-mono-cut-0003"]]
    state = sampler.state_dict()
    assert state["packing_buffer_size"] == 4
    assert state["bucketer_state"]["bucket_tokens"][0] == [[1], [2]]
    expected_remaining = _batch_ids(_drain(iterator))

    resumed = _make_bucketed_sampler(make_source(), buffer_size=4, packing_buffer_size=1)
    resumed.load_state_dict(deepcopy(state))
    actual_remaining = _batch_ids(_drain(iter(resumed)))

    assert resumed.packing_buffer_size == 4
    assert actual_remaining == expected_remaining == [["dummy-mono-cut-0001", "dummy-mono-cut-0002"]]


def test_bucketed_packed_sampler_indexed_concurrent_resume_restarts_and_cleans_producer(
    tmp_path,
):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((5.0,) * 20).to_file(cuts_path)
    create_jsonl_index(cuts_path)

    def make_source():
        return CutSet(LazyIndexedManifestIterator(cuts_path))

    sampler = _make_bucketed_sampler(
        make_source(),
        buffer_size=8,
        packing_buffer_size=4,
        concurrent=True,
    )
    iterator = iter(sampler)
    next(iterator)

    original_bucketer = sampler._bucketer
    assert original_bucketer._producer_thread is not None
    assert original_bucketer._producer_thread.is_alive()
    state = sampler.state_dict()
    assert original_bucketer._producer_thread is not None
    assert original_bucketer._producer_thread.is_alive() or original_bucketer._source_exhausted
    expected_remaining = _batch_ids(_drain(iterator))

    resumed = _make_bucketed_sampler(
        make_source(),
        buffer_size=8,
        packing_buffer_size=4,
        concurrent=True,
    )
    resumed.load_state_dict(deepcopy(state))
    resumed_iterator = iter(resumed)
    first_resumed_batch = next(resumed_iterator)

    restored_bucketer = resumed._bucketer
    assert restored_bucketer._producer_thread is not None
    assert restored_bucketer._producer_thread.is_alive() or restored_bucketer._source_exhausted
    actual_remaining = _batch_ids([first_resumed_batch, *_drain(resumed_iterator)])

    assert actual_remaining == expected_remaining
    assert restored_bucketer._producer_thread is None


def test_packed_sequence_sampler_backfills_and_enforces_exact_token_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path)))

    assert _batch_ids(batches) == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0003"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
    ]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [10, 10]
    assert all(sum(cut.num_tokens for cut in batch) <= 10 for batch in batches)


def test_packed_sequence_sampler_enforces_output_token_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts_with_tokens((7.0, 4.0, 6.0, 3.0), (8, 2, 2, 2)).to_file(cuts_path)

    # the best fit on the audio budget pairs the anchor with the last cut, which the output cap forbids
    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_output_tokens=9))
    assert _batch_ids(batches) == [
        ["dummy-mono-cut-0000"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
        ["dummy-mono-cut-0003"],
    ]
    assert all(sum(cut.num_output_tokens for cut in batch) <= 9 for batch in batches)
    assert all(sum(cut.num_tokens for cut in batch) <= 10 for batch in batches)

    without_cap = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path)))
    assert _batch_ids(without_cap)[0] == ["dummy-mono-cut-0000", "dummy-mono-cut-0003"]


def test_packed_sequence_sampler_backfills_within_output_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts_with_tokens((5.0, 5.0, 3.0, 2.0), (6, 6, 1, 1)).to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_output_tokens=8))

    # the exact fit [0, 1] has 12 output tokens; cut 1 is dropped and the two small cuts fill both budgets exactly
    assert _batch_ids(batches) == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0002", "dummy-mono-cut-0003"],
        ["dummy-mono-cut-0001"],
    ]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [10, 5]
    assert [sum(cut.num_output_tokens for cut in batch) for batch in batches] == [8, 6]


def test_bucketed_packed_sampler_enforces_output_token_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts_with_tokens((7.0, 4.0, 6.0, 3.0), (8, 2, 2, 2)).to_file(cuts_path)

    batches = _drain(iter(_make_bucketed_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_output_tokens=9)))

    assert _batch_ids(batches) == [
        ["dummy-mono-cut-0000"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
        ["dummy-mono-cut-0003"],
    ]


def test_packed_sequence_sampler_rejects_single_example_over_output_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts_with_tokens((4.0, 4.0), (12, 2)).to_file(cuts_path)

    with pytest.raises(ValueError, match="output tokens.*exceed batch_output_tokens=10"):
        list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_output_tokens=10))


def test_fit_output_cap_drops_largest_then_backfills():
    # the anchor at index 0 leaves 10 budget and 8 output tokens for the tail
    budgets, outputs = [4, 5, 5, 3, 2], [2, 6, 6, 1, 1]
    caps = dict(batch_tokens=14, output_cap=10)
    # the later of two equally large outputs is dropped, then the small candidates fill both budgets
    assert _fit_output_cap([0, 1], budgets, outputs, max_items=None, **caps) == [0, 2, 3]
    assert _fit_output_cap([0, 1], budgets, outputs, max_items=2, **caps) == [0, 2]
    assert _fit_output_cap([0], [4, 5, 3], [2, 4, 4], batch_tokens=14, output_cap=22, max_items=None) == [0, 1]
    with pytest.raises(ValueError, match="exceed batch_output_tokens=5"):
        _fit_output_cap([], [4], [6], batch_tokens=14, output_cap=5, max_items=None)


def test_packed_token_constraint_tracks_output_tokens():
    constraint = PackedTokenConstraint(batch_tokens=100, batch_output_tokens=10)
    example = SimpleNamespace(num_tokens=5, num_output_tokens=6)
    constraint.add(example)
    assert not constraint.exceeded() and constraint.close_to_exceeding()
    constraint.add(example)
    assert constraint.exceeded()
    constraint.reset()
    assert constraint.current_output == 0 and not constraint.exceeded()
    with pytest.raises(ValueError, match="batch_output_tokens"):
        PackedTokenConstraint(batch_tokens=100, batch_output_tokens=0)
    with pytest.raises(ValueError, match="requires use_packed_sequence_sampling"):
        MultimodalSamplingConstraint(token_equivalent_duration=1.0, batch_tokens=10, batch_output_tokens=5)


def test_packed_sequence_config_accepts_output_token_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "force_finite": True,
                "shuffle": False,
                "num_workers": 0,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 10,
                "batch_output_tokens": 20,
                "token_equivalent_duration": 1.0,
                "audio_token_estimator": {
                    "preprocessor": {"n_fft": 16000, "hop_length": 16000, "stft_pad_amount": 8000},
                    "subsampling": [],
                },
                "use_packed_sequence_sampling": True,
                "measure_total_length": False,
                "seed": 0,
                "shard_seed": 0,
            }
        )
    )

    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=object())

    assert sampler.constraint.batch_output_tokens == 20
    assert sampler.constraint._internal.batch_output_tokens == 20


def _make_cuts_with_text(durations, texts):
    """Cuts whose single supervision carries ``texts[i]``, for tokenization with a prompt format."""
    cuts = []
    for index, (duration, text) in enumerate(zip(durations, texts)):
        cut = dummy_cut(index, duration=duration)
        supervision = SupervisionSegment(
            id=f"sup-{index}", recording_id=cut.recording_id, start=0.0, duration=duration, text=text
        )
        cuts.append(fastcopy(cut, supervisions=[supervision]))
    return CutSet.from_cuts(cuts)


@pytest.fixture(scope="module")
def plain_tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("packed_sampler_tokenizer")
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


def _prompted_packed_config(cuts_path, *, batch_output_tokens, bucketing):
    options = {
        "cuts_path": str(cuts_path),
        "force_finite": True,
        "shuffle": False,
        "num_workers": 0,
        "use_multimodal_sampling": True,
        "pretokenize": True,
        "prompt_format": "plain",
        "batch_tokens": 10,
        "batch_output_tokens": batch_output_tokens,
        "token_equivalent_duration": 1.0,
        "audio_token_estimator": {
            "preprocessor": {"n_fft": 16000, "hop_length": 16000, "stft_pad_amount": 8000},
            "subsampling": [],
        },
        "use_packed_sequence_sampling": True,
        "measure_total_length": False,
        "seed": 0,
        "shard_seed": 0,
    }
    if bucketing:
        options.update(
            {"use_bucketing": True, "num_buckets": 2, "bucket_duration_bins": [100], "concurrent_bucketing": False}
        )
    return make_structured_with_schema_warnings(OmegaConf.create(options))


@pytest.mark.parametrize("bucketing", [False, True])
def test_output_token_cap_counts_prompted_answer_tokens(tmp_path, plain_tokenizer, bucketing):
    durations = (7.0, 4.0, 6.0, 3.0)
    long_answer = "a much longer answer with many more words in it " * 3
    texts = (long_answer, "a short answer", "a short answer", long_answer)
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts_with_text(durations, texts).to_file(cuts_path)
    # the answer lengths the prompt formatter produces: the cuts the audio budget would pair (0 and 3) exceed the cap
    answer_lengths = [
        len(tokenize_with_prompt(cut, plain_tokenizer, "plain").answer_ids)
        for cut in _make_cuts_with_text(durations, texts)
    ]
    cap = answer_lengths[0] + answer_lengths[3] - 1
    assert answer_lengths[1] + answer_lengths[2] <= cap

    config = _prompted_packed_config(cuts_path, batch_output_tokens=cap, bucketing=bucketing)
    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=plain_tokenizer)
    batches = _drain(iter(sampler))

    assert sorted(cut.id for batch in batches for cut in batch) == sorted(f"dummy-mono-cut-{i:04d}" for i in range(4))
    assert all(sum(len(cut.answer_ids) for cut in batch) <= cap for batch in batches)
    assert all(sum(cut.num_tokens for cut in batch) <= 10 for batch in batches)
    assert not any({"dummy-mono-cut-0000", "dummy-mono-cut-0003"} <= set(cut.id for cut in batch) for batch in batches)


def test_output_token_cap_rejects_untokenized_cuts(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    with pytest.raises(ValueError, match="batch_output_tokens needs tokenized examples"):
        list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_output_tokens=9))


def test_packed_sequence_sampler_squeezes_around_large_next_candidate(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((9.0, 12.0, 4.0, 3.0)).to_file(cuts_path)

    batches = list(
        _make_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            batch_tokens=16,
            packing_buffer_size=4,
        )
    )

    assert _batch_ids(batches) == [
        [
            "dummy-mono-cut-0000",
            "dummy-mono-cut-0002",
            "dummy-mono-cut-0003",
        ],
        ["dummy-mono-cut-0001"],
    ]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [16, 12]


def test_packed_sequence_sampler_anchor_prevents_starvation(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((6.0, 6.0, 4.0, 4.0, 4.0, 4.0)).to_file(cuts_path)

    batches = list(
        _make_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            batch_tokens=10,
            packing_buffer_size=4,
        )
    )

    # Candidate 1 cannot fit beside candidate 0, but becomes the mandatory
    # anchor in the very next batch rather than being perpetually bypassed.
    assert _batch_ids(batches)[:2] == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0002"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0003"],
    ]
    assert sorted(cut_id for batch in _batch_ids(batches) for cut_id in batch) == [
        f"dummy-mono-cut-{index:04d}" for index in range(6)
    ]


def test_packed_sequence_sampler_preserves_batch_size_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((2.0, 2.0, 2.0, 2.0)).to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2]


def test_packed_sequence_sampler_preserves_quadratic_factor(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((5.0, 5.0)).to_file(cuts_path)

    batches = list(
        _make_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            quadratic_factor=10,
            packing_buffer_size=2,
        )
    )

    assert [len(batch) for batch in batches] == [1, 1]


def test_packed_sequence_validation_accepts_raw_config_without_optional_bucket_keys():
    config = OmegaConf.create(
        {
            "use_bucketing": False,
            "use_multimodal_sampling": True,
            "use_packed_sequence_sampling": True,
            "batch_tokens": 65536,
        }
    )

    _auto_detect_bucketing_and_validate_batch_size(config)

    assert config.use_bucketing is False


def test_packed_sequence_config_accepts_batch_size_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "force_finite": True,
                "num_workers": 0,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_size": 2,
                "batch_tokens": 10,
                "token_equivalent_duration": 1.0,
                "audio_token_estimator": {
                    "preprocessor": {
                        "n_fft": 16000,
                        "hop_length": 16000,
                        "stft_pad_amount": 8000,
                    },
                    "subsampling": [],
                },
                "use_packed_sequence_sampling": True,
            }
        )
    )

    sampler, _ = get_lhotse_sampler_from_config(
        config,
        global_rank=0,
        world_size=1,
        tokenizer=object(),
    )

    assert all(len(batch) <= 2 for batch in sampler)


@pytest.mark.parametrize("indexed", [False, True])
def test_packed_sequence_sampler_resume_preserves_packing_buffer(tmp_path, indexed):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    if indexed:
        create_jsonl_index(cuts_path)

        def make_source():
            return CutSet(LazyIndexedManifestIterator(cuts_path))

    else:

        def make_source():
            return CutSet.from_jsonl_lazy(cuts_path)

    sampler = _make_sampler(make_source())
    iterator = iter(sampler)
    assert _batch_ids([next(iterator)]) == [["dummy-mono-cut-0000", "dummy-mono-cut-0003"]]
    state = sampler.state_dict()
    assert state["packing_buffer_size"] == 4
    if indexed:
        assert state["packing_buffer_tokens"] == [(1,), (2,)]
    else:
        assert state["packing_buffer_tokens"] is None
    expected_remaining = _batch_ids(_drain(iterator))

    resumed = _make_sampler(make_source(), packing_buffer_size=1)
    resumed.load_state_dict(deepcopy(state))
    actual_remaining = _batch_ids(_drain(iter(resumed)))

    assert resumed.packing_buffer_size == 4
    assert actual_remaining == expected_remaining == [["dummy-mono-cut-0001", "dummy-mono-cut-0002"]]


def test_packed_sequence_sampler_indexed_state_contains_only_origin_tokens(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    create_jsonl_index(cuts_path)
    sampler = _make_sampler(CutSet(LazyIndexedManifestIterator(cuts_path)))
    iterator = iter(sampler)
    next(iterator)

    live_buffer = list(sampler._batcher.reuse_cuts_buffer)
    state = sampler.state_dict()

    assert state["packing_buffer_tokens"] == [(1,), (2,)]
    assert all(
        isinstance(token, int) for candidate_tokens in state["packing_buffer_tokens"] for token in candidate_tokens
    )
    live_buffer[0][0].custom["mutated_after_snapshot"] = True
    assert state["packing_buffer_tokens"] == [(1,), (2,)]


def test_packed_sequence_sampler_indexed_tuple_source_resume(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _make_cuts().to_file(first_path)
    _make_cuts((1.0, 1.0, 1.0, 1.0)).to_file(second_path)
    create_jsonl_index(first_path)
    create_jsonl_index(second_path)

    def make_sources():
        return (
            CutSet(LazyIndexedManifestIterator(first_path)),
            CutSet(LazyIndexedManifestIterator(second_path)),
        )

    sampler = _make_sampler(*make_sources())
    iterator = iter(sampler)
    first_batch = next(iterator)
    assert isinstance(first_batch, tuple)
    assert _batch_ids([first_batch[0]]) == [["dummy-mono-cut-0000", "dummy-mono-cut-0003"]]
    state = sampler.state_dict()
    assert state["packing_buffer_tokens"] == [(1, 1), (2, 2)]
    expected = [(_batch_ids([left]), _batch_ids([right])) for left, right in _drain(iterator)]

    resumed = _make_sampler(*make_sources())
    resumed.load_state_dict(deepcopy(state))
    actual = [(_batch_ids([left]), _batch_ids([right])) for left, right in _drain(iter(resumed))]

    assert actual == expected


def test_packed_sequence_sampler_multiworker_stateful_resume(tmp_path):
    stateful_dataloader = pytest.importorskip("torchdata.stateful_dataloader")
    StatefulDataLoader = stateful_dataloader.StatefulDataLoader
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet.from_cuts(dummy_cut(index, duration=[7.0, 4.0, 6.0, 3.0][index % 4]) for index in range(160)).to_file(
        cuts_path
    )
    create_jsonl_index(cuts_path)

    def make_loader():
        source = CutSet(LazyIndexedManifestIterator(cuts_path))
        wrapper = IterableDatasetWrapper(
            IdentityDataset(),
            _make_sampler(source, packing_buffer_size=16),
        )
        return StatefulDataLoader(
            wrapper,
            batch_size=None,
            num_workers=2,
            persistent_workers=True,
            snapshot_every_n_steps=1,
            timeout=10,
        )

    def consume(loader):
        return _batch_ids(list(loader))

    full = make_loader()
    expected = consume(full)
    full._iterator._shutdown_workers()

    partial = make_loader()
    iterator = iter(partial)
    prefix = _batch_ids([next(iterator) for _ in range(24)])
    state = partial.state_dict()
    partial._iterator._shutdown_workers()

    resumed = make_loader()
    resumed.load_state_dict(deepcopy(state))
    suffix = consume(resumed)
    resumed._iterator._shutdown_workers()

    assert prefix + suffix == expected


def test_packed_sequence_sampler_rejects_single_example_over_budget():
    cuts = CutSet.from_cuts([dummy_cut(0, duration=11.0)])
    with pytest.warns(UserWarning, match="eagerly read CutSet"):
        sampler = _make_sampler(cuts)
    iterator = iter(sampler)
    with pytest.raises(ValueError, match="individual example.*exceeds batch_tokens=10"):
        next(iterator)


def test_packed_sequence_sampler_rejects_invalid_buffer_size():
    with pytest.raises(ValueError, match="positive packing-buffer size"):
        _make_sampler(_make_cuts(), packing_buffer_size=0)
