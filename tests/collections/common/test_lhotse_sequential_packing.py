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

import math
import random
from copy import deepcopy

import pytest
from lhotse import CutSet, SupervisionSegment
from lhotse.dataset.iterable_dataset import IdentityDataset
from lhotse.indexing import create_jsonl_index
from lhotse.lazy import LazyIndexedManifestIterator
from lhotse.testing.dummies import dummy_cut
from lhotse.utils import fastcopy
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.audio_token_estimator import AudioTokenEstimator
from nemo.collections.common.data.lhotse.dataloader import (
    get_lhotse_dataloader_from_config,
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
)
from nemo.collections.common.data.lhotse.packed_sequence_sampler import PackedSequenceDynamicCutSampler
from nemo.collections.common.data.lhotse.sampling import MultimodalSamplingConstraint
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

SAMPLE_RATE = 16000
# one token per second of audio
SECONDS_ESTIMATOR = {
    "preprocessor": {"n_fft": SAMPLE_RATE, "hop_length": SAMPLE_RATE, "stft_pad_amount": SAMPLE_RATE // 2},
    "subsampling": [],
}
# one token per audio sample
SAMPLES_ESTIMATOR = {"preprocessor": {"n_fft": 2, "hop_length": 1, "stft_pad_amount": 1}, "subsampling": []}


def _make_cuts(durations, token_counts=None):
    """Cuts in the given order; with ``token_counts``, each carries that many supervision tokens."""
    cuts = []
    for index, duration in enumerate(durations):
        cut = dummy_cut(index, duration=duration)
        if token_counts is not None:
            supervision = SupervisionSegment(
                id=f"sup-{index}",
                recording_id=cut.recording_id,
                start=0.0,
                duration=duration,
                custom={"tokens": list(range(token_counts[index]))},
            )
            cut = fastcopy(cut, supervisions=[supervision])
        cuts.append(cut)
    return CutSet.from_cuts(cuts)


def _make_sampler(
    *cuts,
    packing_policy="sequential",
    batch_tokens=10,
    batch_output_tokens=None,
    batch_size=None,
    quadratic_factor=None,
    packing_buffer_size=4,
    drop_last=False,
    estimator=SECONDS_ESTIMATOR,
):
    return PackedSequenceDynamicCutSampler(
        *cuts,
        constraint=MultimodalSamplingConstraint(
            audio_token_estimator=AudioTokenEstimator.from_config(estimator, sample_rate=SAMPLE_RATE),
            batch_size=batch_size,
            batch_tokens=batch_tokens,
            batch_output_tokens=batch_output_tokens,
            quadratic_factor=quadratic_factor,
            measure_total_length=False,
            use_packed_sequence_sampling=True,
        ),
        packing_buffer_size=packing_buffer_size,
        packing_policy=packing_policy,
        drop_last=drop_last,
        seed=0,
    )


def _in_order_batches(
    lengths, output_lengths, *, batch_tokens, batch_output_tokens, batch_size, quadratic_factor=None
):
    """Reference rule: fill in reading order, close the batch before the first example over a limit."""
    batches = []
    current = []
    total = total_output = 0
    for index, (length, output_length) in enumerate(zip(lengths, output_lengths)):
        if quadratic_factor is not None:
            length = math.ceil(length + length**2 / quadratic_factor)
        over_limit = (
            (batch_size is not None and len(current) >= batch_size)
            or total + length > batch_tokens
            or (batch_output_tokens is not None and total_output + output_length > batch_output_tokens)
        )
        if current and over_limit:
            batches.append(current)
            current = []
            total = total_output = 0
        current.append(index)
        total += length
        total_output += output_length
    if current:
        batches.append(current)
    return batches


def _drain(iterator):
    """Read a running sampler to its end; ``list()`` would restart it."""
    batches = []
    while True:
        try:
            batches.append(next(iterator))
        except StopIteration:
            return batches


def _batch_ids(batches):
    return [[cut.id for cut in batch] for batch in batches]


def _ids(index_batches):
    return [[f"dummy-mono-cut-{index:04d}" for index in batch] for batch in index_batches]


def _config(cuts_path, **overrides):
    return make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "force_finite": True,
                "num_workers": 0,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 10,
                "audio_token_estimator": SECONDS_ESTIMATOR,
                "use_packed_sequence_sampling": True,
                "measure_total_length": False,
                "seed": 0,
                "shard_seed": 0,
                **overrides,
            }
        )
    )


@pytest.mark.unit
def test_sequential_policy_closes_batch_before_first_example_over_budget(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0)).to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path)))

    assert _batch_ids(batches) == _ids([[0], [1, 2], [3]])
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [7, 10, 3]


@pytest.mark.unit
def test_best_fit_policy_stays_the_default(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0)).to_file(cuts_path)

    sampler = PackedSequenceDynamicCutSampler(
        CutSet.from_jsonl_lazy(cuts_path),
        constraint=MultimodalSamplingConstraint(
            audio_token_estimator=AudioTokenEstimator.from_config(SECONDS_ESTIMATOR, sample_rate=SAMPLE_RATE),
            batch_tokens=10,
            use_packed_sequence_sampling=True,
        ),
        packing_buffer_size=4,
    )

    assert sampler.packing_policy == "best_fit"
    assert _batch_ids(list(sampler)) == _ids([[0, 3], [1, 2]])
    explicit = _make_sampler(CutSet.from_jsonl_lazy(cuts_path), packing_policy="best_fit")
    assert _batch_ids(list(explicit)) == _ids([[0, 3], [1, 2]])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("batch_tokens", "batch_output_tokens", "batch_size", "quadratic_factor"),
    [
        (40, None, None, None),
        (40, 25, None, None),
        (40, None, 3, None),
        (40, 25, 3, None),
        (1000, 25, None, None),
        (1000, None, 5, None),
        (80, 25, 5, 30.0),
    ],
)
@pytest.mark.parametrize("packing_buffer_size", [1, 4, 64])
def test_sequential_policy_matches_in_order_reference(
    tmp_path, batch_tokens, batch_output_tokens, batch_size, quadratic_factor, packing_buffer_size
):
    rng = random.Random(7)
    durations = [float(rng.randint(1, 23)) for _ in range(61)]
    token_counts = [rng.randint(1, 13) for _ in durations]
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts(durations, token_counts).to_file(cuts_path)
    limits = dict(
        batch_tokens=batch_tokens,
        batch_output_tokens=batch_output_tokens,
        batch_size=batch_size,
        quadratic_factor=quadratic_factor,
    )

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), packing_buffer_size=packing_buffer_size, **limits))

    expected = _in_order_batches([int(duration) for duration in durations], token_counts, **limits)
    assert _batch_ids(batches) == _ids(expected)
    # every example once, in reading order
    assert [cut.id for batch in batches for cut in batch] == [f"dummy-mono-cut-{i:04d}" for i in range(61)]


@pytest.mark.unit
def test_sequential_policy_at_sample_budgets(tmp_path):
    # budgets in audio samples and target tokens, as a packed encoder-decoder training sets them
    rng = random.Random(11)
    durations = [rng.randint(1600, 640000) / SAMPLE_RATE for _ in range(400)]
    token_counts = [max(1, int(duration * rng.uniform(2.0, 8.0))) for duration in durations]
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts(durations, token_counts).to_file(cuts_path)
    limits = dict(batch_tokens=20_000_000, batch_output_tokens=5_000, batch_size=45)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), estimator=SAMPLES_ESTIMATOR, **limits))

    samples = [cut.num_samples for cut in CutSet.from_jsonl_lazy(cuts_path)]
    expected = _in_order_batches(samples, token_counts, **limits)
    assert len(expected) > 3
    assert _batch_ids(batches) == _ids(expected)
    # the example cap closes some batches, a token budget the others
    assert {len(batch) == 45 for batch in batches[:-1]} == {True, False}
    assert all(sum(cut.num_tokens for cut in batch) <= 20_000_000 for batch in batches)
    assert all(sum(cut.num_output_tokens for cut in batch) <= 5_000 for batch in batches)


@pytest.mark.unit
def test_sequential_policy_fills_a_budget_exactly(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((6.0, 4.0, 5.0, 5.0, 1.0), (3, 3, 2, 4, 1)).to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_output_tokens=6))

    # an example that reaches a limit exactly still fits
    assert _batch_ids(batches) == _ids([[0, 1], [2, 3], [4]])
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [10, 10, 1]
    assert [sum(cut.num_output_tokens for cut in batch) for batch in batches] == [6, 6, 1]


@pytest.mark.unit
def test_sequential_policy_tuple_inputs(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0)).to_file(first_path)
    _make_cuts((1.0, 1.0, 1.0, 1.0)).to_file(second_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(first_path), CutSet.from_jsonl_lazy(second_path)))

    # the first cut set is the measured one
    assert [[cut.id for cut in first] for first, _ in batches] == _ids([[0], [1, 2], [3]])
    assert [[cut.id for cut in second] for _, second in batches] == _ids([[0], [1, 2], [3]])


@pytest.mark.unit
@pytest.mark.parametrize(("drop_last", "num_batches"), [(False, 3), (True, 2)])
def test_sequential_policy_drop_last(tmp_path, drop_last, num_batches):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((5.0, 5.0, 4.0, 4.0, 5.0, 1.0)).to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), drop_last=drop_last))

    assert _batch_ids(batches) == _ids([[0, 1], [2, 3], [4, 5]][:num_batches])


@pytest.mark.unit
def test_sequential_policy_rejects_single_example_over_budget(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((4.0, 11.0, 4.0)).to_file(cuts_path)
    iterator = iter(_make_sampler(CutSet.from_jsonl_lazy(cuts_path)))

    assert _batch_ids([next(iterator)]) == _ids([[0]])
    with pytest.raises(ValueError, match="individual example.*exceeds batch_tokens=10"):
        next(iterator)


@pytest.mark.unit
def test_sequential_policy_rejects_single_example_over_output_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((4.0, 4.0), (2, 12)).to_file(cuts_path)
    iterator = iter(_make_sampler(CutSet.from_jsonl_lazy(cuts_path), batch_output_tokens=10))

    assert _batch_ids([next(iterator)]) == _ids([[0]])
    with pytest.raises(ValueError, match="output tokens.*exceed batch_output_tokens=10"):
        next(iterator)


@pytest.mark.unit
@pytest.mark.parametrize("indexed", [False, True])
def test_sequential_policy_resume_keeps_the_deferred_example(tmp_path, indexed):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0, 2.0)).to_file(cuts_path)
    if indexed:
        create_jsonl_index(cuts_path)

        def make_source():
            return CutSet(LazyIndexedManifestIterator(cuts_path))

    else:

        def make_source():
            return CutSet.from_jsonl_lazy(cuts_path)

    sampler = _make_sampler(make_source())
    iterator = iter(sampler)
    assert _batch_ids([next(iterator)]) == _ids([[0]])
    state = sampler.state_dict()
    # the example that closed the first batch is the only buffered one
    assert state["packing_buffer_tokens"] == ([(1,)] if indexed else None)
    remaining = _batch_ids(_drain(iterator))

    resumed = _make_sampler(make_source())
    resumed.load_state_dict(deepcopy(state))

    assert _batch_ids(_drain(iter(resumed))) == remaining == _ids([[1, 2], [3, 4]])


@pytest.mark.unit
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(("saved_policy", "first_batch"), [("sequential", [0]), ("best_fit", [0, 3])])
def test_restored_sampler_continues_with_the_saved_policy(tmp_path, indexed, saved_policy, first_batch):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0, 2.0, 5.0, 1.0, 8.0, 2.0, 2.0, 9.0, 1.0)).to_file(cuts_path)
    if indexed:
        create_jsonl_index(cuts_path)

        def make_source():
            return CutSet(LazyIndexedManifestIterator(cuts_path))

    else:

        def make_source():
            return CutSet.from_jsonl_lazy(cuts_path)

    sampler = _make_sampler(make_source(), packing_policy=saved_policy)
    iterator = iter(sampler)
    assert _batch_ids([next(iterator)]) == _ids([first_batch])
    state = sampler.state_dict()
    assert state["packing_policy"] == saved_policy
    remaining = _batch_ids(_drain(iterator))

    other_policy = "best_fit" if saved_policy == "sequential" else "sequential"
    resumed = _make_sampler(make_source(), packing_policy=other_policy)
    resumed.load_state_dict(deepcopy(state))

    assert resumed.packing_policy == saved_policy
    assert _batch_ids(_drain(iter(resumed))) == remaining


@pytest.mark.unit
def test_state_without_policy_restores_best_fit(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0)).to_file(cuts_path)
    sampler = _make_sampler(CutSet.from_jsonl_lazy(cuts_path), packing_policy="best_fit")
    iterator = iter(sampler)
    next(iterator)
    state = sampler.state_dict()
    del state["packing_policy"]

    resumed = _make_sampler(CutSet.from_jsonl_lazy(cuts_path))
    resumed.load_state_dict(state)

    assert resumed.packing_policy == "best_fit"
    assert _batch_ids(_drain(iter(resumed))) == _ids([[1, 2]])


@pytest.mark.unit
def test_sequential_policy_through_dataloader_with_tokenized_output_cap(tmp_path):
    text_path = tmp_path / "text.txt"
    text_path.write_text("a short answer\na much longer answer with many more words in it\n" * 4)
    create_spt_model(
        str(text_path),
        vocab_size=64,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmp_path),
        remove_extra_whitespaces=True,
        bos=True,
        eos=True,
    )
    tokenizer = SentencePieceTokenizer(str(tmp_path / "tokenizer.model"))
    long_answer = "a much longer answer with many more words in it " * 3
    texts = ("a short answer", long_answer, "a short answer", "a short answer", long_answer)
    cuts = []
    for index, text in enumerate(texts):
        cut = dummy_cut(index, duration=2.0)
        supervision = SupervisionSegment(
            id=f"sup-{index}", recording_id=cut.recording_id, start=0.0, duration=2.0, text=text
        )
        cuts.append(fastcopy(cut, supervisions=[supervision]))
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet.from_cuts(cuts).to_file(cuts_path)

    def make_dataloader(batch_output_tokens):
        return get_lhotse_dataloader_from_config(
            _config(cuts_path, packing_policy="sequential", pretokenize=True, batch_output_tokens=batch_output_tokens),
            global_rank=0,
            world_size=1,
            dataset=IdentityDataset(),
            tokenizer=tokenizer,
        )

    # the audio budget admits all five cuts; the token counts are the ones the dataloader's tokenization produces
    (everything,) = list(make_dataloader(10_000))
    output_lengths = [cut.num_output_tokens for cut in everything]
    cap = output_lengths[0] + output_lengths[1]

    batches = list(make_dataloader(cap))

    expected = _in_order_batches([2] * 5, output_lengths, batch_tokens=10, batch_output_tokens=cap, batch_size=None)
    assert len(expected) > 1
    assert _batch_ids(batches) == _ids(expected)


@pytest.mark.unit
def test_sequential_policy_from_config(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0)).to_file(cuts_path)

    sampler, _ = get_lhotse_sampler_from_config(
        _config(cuts_path, packing_policy="sequential"), global_rank=0, world_size=1, tokenizer=object()
    )

    assert type(sampler) is PackedSequenceDynamicCutSampler
    assert sampler.packing_policy == "sequential"
    assert _batch_ids(list(sampler)) == _ids([[0], [1, 2], [3]])

    default, _ = get_lhotse_sampler_from_config(_config(cuts_path), global_rank=0, world_size=1, tokenizer=object())
    assert default.packing_policy == "best_fit"
    assert _batch_ids(list(default)) == _ids([[0, 3], [1, 2]])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"packing_policy": "in_order"}, "packing_policy must be one of"),
        (
            {"packing_policy": "sequential", "use_bucketing": True, "bucket_duration_bins": [5.0]},
            "packing_policy='sequential' is available only for packed sequence sampling without bucketing",
        ),
        (
            {"packing_policy": "sequential", "use_packed_sequence_sampling": False},
            "packing_policy='sequential' is available only for packed sequence sampling without bucketing",
        ),
    ],
)
def test_packing_policy_rejects_unsupported_configurations(tmp_path, overrides, message):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((7.0, 4.0, 6.0, 3.0)).to_file(cuts_path)

    with pytest.raises(ValueError, match=message):
        get_lhotse_sampler_from_config(
            _config(cuts_path, **overrides), global_rank=0, world_size=1, tokenizer=object()
        )
