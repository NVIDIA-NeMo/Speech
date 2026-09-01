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

import random
from collections.abc import Sequence
from itertools import islice
from typing import Any

import torch
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, DynamicCutSampler
from lhotse.dataset.dataloading import resolve_seed
from lhotse.dataset.sampling.dynamic import DurationBatcher, Filter
from lhotse.dataset.sampling.dynamic_bucketing import BucketSelectionState, DynamicBucketer
from lhotse.lazy import get_graph_origin, resolve_iterator_source


def _select_best_fit_indices(lengths: Sequence[int], capacity: int, max_items: int | None = None) -> list[int]:
    """Select an exact best-fit subset, preferring earlier items on ties."""
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative (got {capacity})")
    if any(length < 0 for length in lengths):
        raise ValueError("lengths must be non-negative")
    if not lengths or capacity == 0 or max_items == 0:
        return []

    mask = (1 << (capacity + 1)) - 1

    if max_items is None or max_items >= len(lengths):
        # Prefix reachability makes reconstruction deterministic: when the
        # target was already reachable without the current (later) item, skip
        # it. Consequently equal-quality solutions favor earlier candidates.
        prefixes = [1]
        for length in lengths:
            reachable = prefixes[-1]
            prefixes.append(reachable | ((reachable << length) & mask))

        target = prefixes[-1].bit_length() - 1
        selected = []
        for index in range(len(lengths) - 1, -1, -1):
            if (prefixes[index] >> target) & 1:
                continue
            selected.append(index)
            target -= lengths[index]
        selected.reverse()
        return selected

    max_items = min(max_items, len(lengths))
    if max_items < 0:
        raise ValueError(f"max_items must be non-negative (got {max_items})")

    # Count-aware reachability is only needed when batch_size is configured.
    layers = [1] + [0] * max_items
    prefixes = [tuple(layers)]
    for item_index, length in enumerate(lengths):
        for count in range(min(max_items, item_index + 1), 0, -1):
            layers[count] |= (layers[count - 1] << length) & mask
        prefixes.append(tuple(layers))

    reachable = 0
    for layer in layers:
        reachable |= layer
    target = reachable.bit_length() - 1
    # Prefer fewer examples when token utilization ties, then earlier examples.
    count = next(count for count, layer in enumerate(layers) if (layer >> target) & 1)
    selected = []
    for index in range(len(lengths) - 1, -1, -1):
        if (prefixes[index][count] >> target) & 1:
            continue
        selected.append(index)
        target -= lengths[index]
        count -= 1
    selected.reverse()
    return selected


class ExactTokenBatcher(DurationBatcher):
    """Bounded-lookahead best-fit batching for padding-free sequences."""

    def __init__(self, *args, packing_buffer_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        if packing_buffer_size <= 0:
            raise ValueError(
                "packing_buffer_size must be a positive packing-buffer size " f"(got {packing_buffer_size})"
            )
        self.packing_buffer_size = packing_buffer_size
        self._source_exhausted = False

    @staticmethod
    def _detuplify(examples):
        if isinstance(examples[0], tuple):
            if len(examples[0]) == 1:
                return CutSet.from_cuts(example[0] for example in examples)
            tuple_of_example_lists = list(zip(*examples))
            return tuple(CutSet.from_cuts(items) for items in tuple_of_example_lists)
        return CutSet.from_cuts(examples)

    @staticmethod
    def _measured_example(example_or_tuple):
        return example_or_tuple[0] if isinstance(example_or_tuple, tuple) else example_or_tuple

    def _fill_packing_buffer(self) -> None:
        while len(self.reuse_cuts_buffer) < self.packing_buffer_size and not self._source_exhausted:
            try:
                self.reuse_cuts_buffer.append(next(self.cuts_iter))
            except StopIteration:
                self._source_exhausted = True

    def _measure_integer_length(self, example_or_tuple) -> int:
        length = self.constraint.measure_length(self._measured_example(example_or_tuple))
        integer_length = int(length)
        if integer_length != length:
            raise ValueError("Packed sequence sampling requires integer token lengths, " f"but measured {length!r}.")
        if integer_length <= 0:
            raise ValueError("Packed sequence sampling requires positive token lengths, " f"but measured {length!r}.")
        return integer_length

    def _measure_budget_length(self, example_or_tuple) -> int:
        measured = self._measured_example(example_or_tuple)
        measure_packing_length = getattr(self.constraint, "measure_packing_length", None)
        return (
            measure_packing_length(measured)
            if callable(measure_packing_length)
            else self._measure_integer_length(example_or_tuple)
        )

    def _limits(self) -> tuple[int, int | None]:
        batch_tokens = getattr(self.constraint, "batch_tokens", None)
        max_examples = getattr(self.constraint, "batch_size", None)
        if batch_tokens is None:
            internal = getattr(self.constraint, "_internal", None)
            batch_tokens = getattr(internal, "batch_tokens", None)
            max_examples = getattr(internal, "max_examples", max_examples)
        if batch_tokens is None:
            raise ValueError("Packed sequence sampling requires batch_tokens to define the exact token cap.")
        batch_tokens = int(batch_tokens)
        if batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive (got {batch_tokens})")
        if max_examples is not None:
            max_examples = int(max_examples)
            if max_examples <= 0:
                raise ValueError(f"batch_size must be positive or null (got {max_examples})")
        return batch_tokens, max_examples

    def _discard(self, examples) -> None:
        try:
            self.diagnostics.discard(examples)
        except AttributeError:
            self.diagnostics.discard(examples[0])

    def _collect_batch(self):
        self._fill_packing_buffer()
        if not self.reuse_cuts_buffer:
            raise StopIteration()

        pool = list(self.reuse_cuts_buffer)
        raw_lengths = [self._measure_integer_length(example) for example in pool]
        budget_lengths = [self._measure_budget_length(example) for example in pool]
        batch_tokens, max_examples = self._limits()

        anchor_length = raw_lengths[0]
        anchor_budget = budget_lengths[0]
        if anchor_length > batch_tokens:
            raise ValueError(
                f"An individual example ({anchor_length} tokens) exceeds "
                f"batch_tokens={batch_tokens}. Set max_tokens less than or equal "
                "to batch_tokens so it is filtered before batching."
            )
        if anchor_budget > batch_tokens:
            raise ValueError(
                f"An individual example's effective packed budget ({anchor_budget} tokens) exceeds "
                f"batch_tokens={batch_tokens}; increase batch_tokens or quadratic_factor."
            )

        remaining_items = None if max_examples is None else max_examples - 1
        tail_indices = _select_best_fit_indices(
            budget_lengths[1:],
            batch_tokens - anchor_budget,
            max_items=remaining_items,
        )
        selected_indices = {0, *(index + 1 for index in tail_indices)}
        examples = [example for index, example in enumerate(pool) if index in selected_indices]
        deferred = [example for index, example in enumerate(pool) if index not in selected_indices]
        self.reuse_cuts_buffer.clear()
        self.reuse_cuts_buffer.extend(deferred)

        self.constraint.reset()
        for example in examples:
            self.constraint.add(self._measured_example(example))
        if self.constraint.exceeded():
            raise AssertionError("Best-fit packed batch exceeded its configured constraint.")

        is_final_batch = self._source_exhausted and not self.reuse_cuts_buffer
        if is_final_batch and self.drop_last and not self.constraint.close_to_exceeding():
            self._discard(examples)
            raise StopIteration()

        return self._detuplify(examples)


class PackedSequenceDynamicBucketer(DynamicBucketer):
    """Dynamic bucketer that best-fit packs a bounded pool in one bucket.

    ``buffer_size`` remains the global occupancy across all buckets, while
    ``packing_buffer_size`` bounds the subset-sum lookahead inside the bucket
    selected for the next batch. The oldest item is always the anchor, which
    guarantees that best-fit backfilling cannot starve a difficult example.

    This class intentionally extends Lhotse's bucketer rather than copying its
    queue/checkpoint implementation. Its iteration loop mirrors the small
    commit/refill portion of :class:`DynamicBucketer` because upstream does not
    expose a hook for replacing ``DurationBatcher``.
    """

    def __init__(self, *args, packing_buffer_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        if packing_buffer_size <= 0:
            raise ValueError(f"packing_buffer_size must be positive (got {packing_buffer_size})")
        self.packing_buffer_size = packing_buffer_size

    @staticmethod
    def _measured_example(example_or_tuple):
        return example_or_tuple[0] if isinstance(example_or_tuple, tuple) else example_or_tuple

    @staticmethod
    def _detuplify(examples):
        if isinstance(examples[0], tuple):
            if len(examples[0]) == 1:
                return CutSet.from_cuts(example[0] for example in examples)
            tuple_of_example_lists = list(zip(*examples))
            return tuple(CutSet.from_cuts(items) for items in tuple_of_example_lists)
        return CutSet.from_cuts(examples)

    def _measure_integer_length(self, example_or_tuple) -> int:
        length = self.constraint.measure_length(self._measured_example(example_or_tuple))
        integer_length = int(length)
        if integer_length != length:
            raise ValueError("Packed sequence sampling requires integer token lengths, " f"but measured {length!r}.")
        if integer_length <= 0:
            raise ValueError("Packed sequence sampling requires positive token lengths, " f"but measured {length!r}.")
        return integer_length

    def _measure_budget_length(self, example_or_tuple) -> int:
        measured = self._measured_example(example_or_tuple)
        measure_packing_length = getattr(self.constraint, "measure_packing_length", None)
        return (
            measure_packing_length(measured)
            if callable(measure_packing_length)
            else self._measure_integer_length(example_or_tuple)
        )

    def _limits(self) -> tuple[int, int | None]:
        batch_tokens = getattr(self.constraint, "batch_tokens", None)
        max_examples = getattr(self.constraint, "batch_size", None)
        if batch_tokens is None:
            internal = getattr(self.constraint, "_internal", None)
            batch_tokens = getattr(internal, "batch_tokens", None)
            max_examples = getattr(internal, "max_examples", max_examples)
        if batch_tokens is None:
            raise ValueError("Packed sequence sampling requires batch_tokens to define the exact token cap.")
        batch_tokens = int(batch_tokens)
        if batch_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive (got {batch_tokens})")
        if max_examples is not None:
            max_examples = int(max_examples)
            if max_examples <= 0:
                raise ValueError(f"batch_size must be positive or null (got {max_examples})")
        return batch_tokens, max_examples

    def _plan_batch(self, bucket, *, randomize_tail: bool) -> tuple[list[int], list[Any]]:
        with bucket.mutex:
            bucket_size = len(bucket.queue)
            if bucket_size == 0:
                raise StopIteration()
            if randomize_tail:
                # random.sample(range(...), k) is O(k) in the bounded
                # lookahead size instead of shuffling the whole bucket.
                tail_indices = self.rng.sample(
                    range(1, bucket_size),
                    k=min(self.packing_buffer_size - 1, bucket_size - 1),
                )
                candidate_indices = [0, *tail_indices]
                candidates = [bucket.queue[index] for index in candidate_indices]
            else:
                candidates = list(islice(bucket.queue, self.packing_buffer_size))
                candidate_indices = list(range(len(candidates)))

        # Preserve a FIFO anchor for starvation resistance. In shuffle mode,
        # draw the remaining lookahead candidates from the whole bucket; the
        # deterministic prefix is retained as a fullness fallback below.
        raw_lengths = [self._measure_integer_length(example) for example in candidates]
        budget_lengths = [self._measure_budget_length(example) for example in candidates]
        batch_tokens, max_examples = self._limits()
        if raw_lengths[0] > batch_tokens:
            raise ValueError(
                f"An individual example ({raw_lengths[0]} tokens) exceeds "
                f"batch_tokens={batch_tokens}. Set max_tokens less than or equal "
                "to batch_tokens so it is filtered before batching."
            )
        if budget_lengths[0] > batch_tokens:
            raise ValueError(
                f"An individual example's effective packed budget ({budget_lengths[0]} tokens) exceeds "
                f"batch_tokens={batch_tokens}; increase batch_tokens or quadratic_factor."
            )

        remaining_items = None if max_examples is None else max_examples - 1
        tail_selection = _select_best_fit_indices(
            budget_lengths[1:],
            batch_tokens - budget_lengths[0],
            max_items=remaining_items,
        )
        selected_positions = [0, *(index + 1 for index in tail_selection)]
        selected_indices = [candidate_indices[position] for position in selected_positions]
        examples = [candidates[position] for position in selected_positions]
        return selected_indices, examples

    def _batch_is_full(self, examples) -> bool:
        constraint = self.constraint.copy()
        constraint.reset()
        for example in examples:
            constraint.add(self._measured_example(example))
        if constraint.exceeded():
            raise AssertionError("Best-fit packed batch exceeded its configured constraint.")
        return constraint.close_to_exceeding()

    def _is_ready(self, bucket) -> bool:
        if bucket.qsize() == 0:
            return False
        _, examples = self._plan_batch(bucket, randomize_tail=False)
        return self._batch_is_full(examples)

    def _collect_packed_batch(self, bucket) -> tuple[list[int], Any]:
        selected_indices, examples = self._plan_batch(bucket, randomize_tail=self.shuffle)
        if self.shuffle and not self._batch_is_full(examples):
            # Bucket selection established readiness from its deterministic
            # prefix. A random bounded pool may be unusually sparse; fall back
            # to that prefix instead of emitting an avoidably partial batch.
            fallback_indices, fallback_examples = self._plan_batch(bucket, randomize_tail=False)
            if self._batch_is_full(fallback_examples):
                selected_indices, examples = fallback_indices, fallback_examples
        return selected_indices, self._detuplify(examples)

    def __iter__(self):
        self.cuts_iter = iter(self.cuts)
        if self._saved_state is not None:
            state = self._restore_from_saved_state()
            self._selection_state = state
            if self.concurrent:
                # Indexed restore reconstructs the buffered queues and resumes
                # the source after them. Restart the producer so restored
                # buffers continue to replenish as they are consumed.
                self._source_exhausted = False
                self._start_data_producer_thread()
                self._maybe_wait_for_producer()
        else:
            if self.concurrent:
                self._source_exhausted = False
                self._start_data_producer_thread()
                self._maybe_wait_for_producer()
            else:
                self._collect_cuts_in_buckets(self.buffer_size)
            state = BucketSelectionState(
                bucket_rng=self.bucket_rng,
                num_buckets=len(self.buckets),
                world_size=self.world_size,
            )
            self._selection_state = state

        try:
            while True:
                sampling_bucket = self._select_bucket(self._selection_state)
                selected_indices, batch = self._collect_packed_batch(sampling_bucket)
                # Commit arbitrary queue removals before yielding so state_dict
                # always describes the next batch, including indexed O(1) resume.
                with sampling_bucket.mutex:
                    for index in sorted(selected_indices, reverse=True):
                        del sampling_bucket.queue[index]
                batch_size = len(selected_indices)
                stop_after_yield = False
                if self.concurrent:
                    try:
                        self._maybe_wait_for_producer()
                    except StopIteration:
                        stop_after_yield = True
                else:
                    try:
                        self._collect_cuts_in_buckets(batch_size)
                    except StopIteration:
                        stop_after_yield = True
                yield batch
                if stop_after_yield:
                    break
        except StopIteration:
            pass
        finally:
            if self.concurrent and self._producer_thread is not None:
                if self._producer_thread.is_alive():
                    self._source_exhausted = True
                    self._producer_thread.join()
                self._producer_thread = None
            self.cuts_iter = None


class PackedSequenceDynamicBucketingSampler(DynamicBucketingSampler):
    """DynamicBucketingSampler with bounded best-fit packed batches.

    The subclass relies on the parent sampler's iterator initialization and
    checkpoint fields because Lhotse currently hardcodes its bucketer class.
    Bucket queues, RNG, synchronized bucket selection, filters, tuple inputs,
    and indexed graph tokens are all still owned by upstream Lhotse.
    """

    def __init__(self, *args, packing_buffer_size: int = 128, **kwargs):
        if packing_buffer_size <= 0:
            raise ValueError(f"packing_buffer_size must be positive (got {packing_buffer_size})")
        super().__init__(*args, **kwargs)
        self.packing_buffer_size = packing_buffer_size

    def state_dict(self) -> dict[str, Any]:
        bucketer = getattr(self, "_bucketer", None)
        producer_was_live = bool(
            bucketer is not None
            and bucketer.concurrent
            and bucketer._producer_thread is not None
            and bucketer._producer_thread.is_alive()
            and not bucketer._source_exhausted
        )
        if producer_was_live:
            # Freeze both the source iterator and bucket queues before the
            # parent snapshots cuts_state followed by bucketer_state.
            bucketer._source_exhausted = True
            bucketer._producer_thread.join()
            bucketer._producer_thread = None
            bucketer._source_exhausted = False
        try:
            state = super().state_dict()
            state["packing_buffer_size"] = self.packing_buffer_size
            return state
        finally:
            if producer_was_live:
                bucketer._start_data_producer_thread()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        packing_buffer_size = state_dict.pop("packing_buffer_size", self.packing_buffer_size)
        if packing_buffer_size <= 0:
            raise ValueError(f"Restored packing_buffer_size must be positive (got {packing_buffer_size})")
        self.packing_buffer_size = packing_buffer_size
        super().load_state_dict(state_dict)

    def __iter__(self) -> "PackedSequenceDynamicBucketingSampler":
        if getattr(self, "_needs_fast_forward", False):
            self._needs_fast_forward = False
            self._fast_forward()
            return self
        if self._just_restored_state:
            return self

        seed = resolve_seed(self.seed)
        self.rng = random.Random(seed + self.epoch)
        if self.sync_buckets:
            bucket_rng_seed = 1234
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                bucket_rng_seed += worker_info.id
            bucket_rng = random.Random(bucket_rng_seed)
        else:
            bucket_rng = None
        if getattr(self, "_skip_diagnostics_reset_once", False):
            self._skip_diagnostics_reset_once = False
        else:
            self.diagnostics.reset_current_epoch()

        restore_sources = [resolve_iterator_source(source) for source in self.cuts]
        source_iterators = [iter(source) for source in restore_sources]
        filtered_examples = Filter(
            iterator=zip(*source_iterators),
            predicate=lambda examples: all(self._filter_fn(example) for example in examples),
            diagnostics=self.diagnostics,
        )
        self._bucketer = PackedSequenceDynamicBucketer(
            filtered_examples,
            duration_bins=self.duration_bins,
            world_size=self.world_size,
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            constraint=self.constraint,
            drop_last=self.drop_last,
            buffer_size=self.buffer_size,
            quadratic_duration=self.quadratic_duration,
            shuffle=self.shuffle,
            rng=self.rng,
            bucket_rng=bucket_rng,
            concurrent=self.concurrent,
            diagnostics=self.diagnostics,
            restore_sources=restore_sources,
            packing_buffer_size=self.packing_buffer_size,
        )
        self.cuts_iter = iter(self._bucketer)
        return self


class PackedSequenceDynamicCutSampler(DynamicCutSampler):
    """Dynamic sampler with one bounded best-fit pool and an exact token cap.

    ``packing_buffer_size`` controls the post-filter best-fit pool. The parent
    reservoir shuffler is intentionally disabled because indexed data sources
    already provide an O(1)-memory Feistel permutation. Consequently,
    ``shuffle_buffer_size`` retains its parent-class meaning but has no effect
    on packing lookahead in this sampler.
    """

    def __init__(
        self,
        *args,
        shuffle: bool = False,
        packing_buffer_size: int = 128,
        shuffle_buffer_size: int = 20000,
        **kwargs,
    ):
        if packing_buffer_size is None or packing_buffer_size <= 0:
            raise ValueError(
                "packing_buffer_size must be a positive packing-buffer size " f"(got {packing_buffer_size})"
            )
        # Consume the public `shuffle` argument for config compatibility, but
        # do not allocate DynamicCutSampler's second, reservoir-style buffer.
        del shuffle
        super().__init__(
            *args,
            shuffle=False,
            shuffle_buffer_size=shuffle_buffer_size,
            **kwargs,
        )
        self.packing_buffer_size = packing_buffer_size
        self._batcher = None
        self._restored_packing_buffer_tokens = []
        self._restored_legacy_examples = []
        self._inject_restored_packing_buffer = False

    def _uses_indexed_restore(self) -> bool:
        return bool(self.cuts) and all(getattr(source, "has_constant_time_access", False) for source in self.cuts)

    @staticmethod
    def _capture_packing_buffer_tokens(buffer) -> list[tuple[Any, ...]]:
        saved = []
        for example_or_tuple in buffer:
            examples = example_or_tuple if isinstance(example_or_tuple, tuple) else (example_or_tuple,)
            tokens = tuple(get_graph_origin(example) for example in examples)
            if any(token is None for token in tokens):
                raise RuntimeError(
                    "PackedSequenceDynamicCutSampler could not checkpoint its packing buffer: "
                    "an indexed candidate is missing graph-origin metadata."
                )
            saved.append(tokens)
        return saved

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["packing_buffer_size"] = self.packing_buffer_size
        if self._uses_indexed_restore():
            if self._batcher is not None:
                state["packing_buffer_tokens"] = self._capture_packing_buffer_tokens(self._batcher.reuse_cuts_buffer)
            else:
                state["packing_buffer_tokens"] = list(self._restored_packing_buffer_tokens)
        else:
            # Replay restoration deterministically rebuilds the post-filter pool.
            state["packing_buffer_tokens"] = None
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        packing_buffer_size = state_dict.pop("packing_buffer_size", self.packing_buffer_size)
        if packing_buffer_size is None or packing_buffer_size <= 0:
            raise ValueError(
                "Restored packing_buffer_size must be a positive packing-buffer size "
                f"(got {packing_buffer_size})"
            )
        self.packing_buffer_size = packing_buffer_size
        tokens = state_dict.pop("packing_buffer_tokens", None)
        self._restored_packing_buffer_tokens = [] if tokens is None else list(tokens)
        # Read checkpoints created by the previous one-candidate implementation.
        self._restored_legacy_examples = list(state_dict.pop("deferred_examples", []))
        self._batcher = None
        super().load_state_dict(state_dict)

    def allow_iter_to_reset_state(self):
        super().allow_iter_to_reset_state()
        self._restored_packing_buffer_tokens = []
        self._restored_legacy_examples = []

    def _fast_forward(self):
        # Indexed restoration resumes each source after all buffered candidates;
        # reconstruct them by immutable graph token. O(N) replay rebuilds the pool.
        self._inject_restored_packing_buffer = self._uses_indexed_restore()
        try:
            super()._fast_forward()
        finally:
            self._inject_restored_packing_buffer = False
        self._restored_packing_buffer_tokens = []
        self._restored_legacy_examples = []

    def _restore_packing_buffer(self) -> list[tuple[Any, ...]]:
        active_sources = self._active_cuts or []
        restored = []
        for tokens in self._restored_packing_buffer_tokens:
            if len(tokens) != len(active_sources):
                raise RuntimeError(
                    "Packed sampler checkpoint source count does not match the active source graph: "
                    f"{len(tokens)} != {len(active_sources)}."
                )
            restored.append(
                tuple(resolve_iterator_source(source)[token] for source, token in zip(active_sources, tokens))
            )
        restored.extend(self._restored_legacy_examples)
        return restored

    def _initialize_epoch_iterator(self, *, rebuild_sources: bool) -> None:
        if rebuild_sources or self._active_cuts is None:
            self._active_cuts = self._make_epoch_sources()
        source_iterators = [iter(resolve_iterator_source(source)) for source in self._active_cuts]
        filtered_examples = Filter(
            iterator=zip(*source_iterators),
            predicate=lambda examples: all(self._filter_fn(example) for example in examples),
            diagnostics=self.diagnostics,
        )
        self._batcher = ExactTokenBatcher(
            filtered_examples,
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            constraint=self.constraint,
            drop_last=self.drop_last,
            quadratic_duration=self.quadratic_duration,
            diagnostics=self.diagnostics,
            packing_buffer_size=self.packing_buffer_size,
        )
        if self._inject_restored_packing_buffer:
            self._batcher.reuse_cuts_buffer.extend(self._restore_packing_buffer())
        self.cuts_iter = iter(self._batcher)
