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

import pytest
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler
from lhotse.dataset.iterable_dataset import IdentityDataset, IterableDatasetWrapper
from lhotse.indexing import create_jsonl_index
from lhotse.lazy import LazyIndexedManifestIterator
from lhotse.testing.dummies import dummy_cut
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.audio_token_estimator import AudioTokenEstimator
from nemo.collections.common.data.lhotse.dataloader import (
    _auto_detect_bucketing_and_validate_batch_size,
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
)
from nemo.collections.common.data.lhotse.packed_sequence_sampler import (
    PackedSequenceDynamicBucketingSampler,
    PackedSequenceDynamicCutSampler,
    _select_best_fit_indices,
)
from nemo.collections.common.data.lhotse.sampling import MultimodalSamplingConstraint


def _make_cuts(durations=(7.0, 4.0, 6.0, 3.0)):
    return CutSet.from_cuts(dummy_cut(index, duration=duration) for index, duration in enumerate(durations))


def _make_sampler(
    *cuts,
    batch_tokens=10,
    batch_size=None,
    quadratic_factor=None,
    packing_buffer_size=4,
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
    concurrent=False,
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
            quadratic_factor=quadratic_factor,
            measure_total_length=False,
            use_packed_sequence_sampling=True,
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


def test_packed_fixed_bucket_config_preserves_regular_sampler(tmp_path):
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
                "token_equivalent_duration": 1.0,
                "use_packed_sequence_sampling": True,
                "seed": 0,
                "shard_seed": 0,
            }
        )
    )

    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1, tokenizer=object())

    assert type(sampler) is DynamicBucketingSampler


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
