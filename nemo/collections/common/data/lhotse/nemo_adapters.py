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

"""Lhotse adapters for NeMo datasets including Parquet support."""
import bisect
import json
import os
import random
import re
import tarfile
from collections.abc import Mapping, Sequence
from contextlib import closing
from io import BytesIO
from pathlib import Path
from typing import Generator, Iterable, List, Literal, Union

try:
    import pyarrow.parquet as pq

    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False
import soundfile
from cytoolz import groupby
from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment
from lhotse.audio.backend import LibsndfileBackend
from lhotse.cut import Cut
from lhotse.dataset.dataloading import resolve_seed
from lhotse.lazy import LazyIteratorChain, LazyJsonlIterator
from lhotse.serialization import open_best
from lhotse.utils import compute_num_samples, ifnone

from nemo.collections.common.data.lhotse._compat import (
    GraphOriginDict,
    IteratorNode,
    LazyIndexedManifestIterator,
    PartitionedIndexedIterator,
    attach_graph_origin,
    normalize_graph_token,
)
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging
from nemo.utils.data_utils import is_datastore_path

# NeMo tarred manifests support per-recording offsets via "-subN" audio_filepath
# suffixes. We use this pattern in both indexed and streaming code paths to
# recover the actual tar member name (offsets share a single member).
_OFFSET_PATTERN = re.compile(r'^(?P<stem>.+)(?P<sub>-sub\d+)(?P<ext>\.\w+)?$')
ShardKey = Union[int, tuple[int, int]]


_MALFORMED_INDEXED_MANIFEST_WARNING_KEYS: set[tuple[str, str]] = set()


def _warn_malformed_indexed_manifest_record(ex: BaseException, idx: int, path: str | Path) -> None:
    key = (str(path), type(ex).__name__)
    if key in _MALFORMED_INDEXED_MANIFEST_WARNING_KEYS:
        return
    _MALFORMED_INDEXED_MANIFEST_WARNING_KEYS.add(key)
    logging.warning(
        "Skipping malformed indexed NeMo manifest records; "
        f"first occurrence path={path!r} idx={idx} error={type(ex).__name__}: {ex}. "
        "Further records with the same path/error type are suppressed in this worker."
    )


class LazyNeMoIterator(IteratorNode):
    """
    ``LazyNeMoIterator`` reads a NeMo (non-tarred) JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text" (overridable with ``text_field`` argument)

    Specially supported keys are:
    - [recommended] "sampling_rate" allows us to provide a valid Lhotse
     ``Recording`` object without checking the audio file
    - "offset" for partial recording reads
    - "lang" is mapped to Lhotse superivsion's language (overridable with ``lang_field`` argument)

    Every other key found in the manifest will be attached to Lhotse Cut and accessible via ``cut.custom[key]``.

    .. caution:: We will perform some I/O (as much as required by soundfile.info) to discover the sampling rate
        of the audio file. If this is not acceptable, convert the manifest to Lhotse format which contains
        sampling rate info. For pure metadata iteration purposes we also provide a ``metadata_only`` flag that
        will create only partially valid Lhotse objects (with metadata related to sampling rate / num samples missing).

    Example::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator("nemo_manifests/train.json"))

    We allow attaching custom metadata to cuts from files other than the manifest via ``extra_fields`` argument.
    In the example below, we'll iterate file "questions.txt" together with the manifest and attach each line
    under ``cut.question`` using the field type ``text_iter``::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator(
        ...     "nemo_manifests/train.json",
        ...     extra_fields=[{"type": "text_iter", "name": "question", "path": "questions.txt"}],
        ... ))

    We also support random sampling of lines with field type ``text_sample``::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator(
        ...     "nemo_manifests/train.json",
        ...     extra_fields=[{"type": "text_sample", "name": "question", "path": "questions.txt"}],
        ... ))

    Indexed mode (``indexed=True``)
    -------------------------------

    When the underlying manifest is uncompressed JSONL, set ``indexed=True`` to enable
    O(1) random access and exact graph-token checkpointing through
    :class:`lhotse.indexing.IndexedJsonlReader`. In indexed mode this iterator becomes
    an indexed ``IteratorNode`` that can be combined with ``StatefulDataLoader`` for
    bit-exact mid-epoch resume.

    Indexed mode requires:

    * the manifest path(s) to use ``.jsonl`` extension and be uncompressed;
    * ``extra_fields`` to be unset (lookup-based fields are positional and cannot be
      reproduced after a Feistel-permuted random access).

    Sharded indexed inputs are composed via :class:`lhotse.lazy.LazyIteratorChain`,
    which picks a Feistel cross-shard permutation for true item-level shuffling.
    """

    def __init__(
        self,
        path: str | Path | list[str],
        text_field: str = "text",
        lang_field: str = "lang",
        metadata_only: bool = False,
        shuffle_shards: bool = False,
        shard_seed: int | Literal["randomized", "trng"] = "trng",
        extra_fields: list[dict[str, str]] | None = None,
        indexed: bool = False,
        indexes_root: str | Path | None = None,
        skip_missing_manifest_entries: bool = False,
    ) -> None:
        self.path = path
        self.shuffle_shards = shuffle_shards
        self.shard_seed = shard_seed
        self.text_field = text_field
        self.lang_field = lang_field
        self.metadata_only = metadata_only
        self.extra_fields = extra_fields
        self.indexed = indexed
        self.indexes_root = indexes_root
        self.skip_missing_manifest_entries = skip_missing_manifest_entries
        validate_extra_fields(self.extra_fields)
        paths = expand_sharded_filepaths(path)

        if indexed:
            if extra_fields:
                raise ValueError(
                    "LazyNeMoIterator(indexed=True) does not support 'extra_fields' because "
                    "their values are positional/streaming and cannot be reconstructed under "
                    "graph-token random access."
                )
            from lhotse.indexing import index_file_path

            seed = resolve_seed(shard_seed) if shard_seed not in (None, "trng", "randomized") else 0
            indexed_sources = [
                LazyIndexedManifestIterator(
                    p,
                    index_path=index_file_path(p, indexes_root),
                    decode=GraphOriginDict,
                    skip_decode_errors=skip_missing_manifest_entries,
                    decode_error_callback=_warn_malformed_indexed_manifest_record,
                )
                for p in paths
            ]
            if len(indexed_sources) == 1:
                self.source = indexed_sources[0]
            else:
                self.source = LazyIteratorChain(*indexed_sources, shuffle_iters=shuffle_shards, seed=seed)
        else:
            if len(paths) == 1:
                self.source = LazyJsonlIterator(paths[0])
            else:
                self.source = LazyIteratorChain(
                    *(LazyJsonlIterator(p) for p in paths),
                    shuffle_iters=self.shuffle_shards,
                    seed=self.shard_seed,
                )

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def __iter__(self) -> Generator[Cut, None, None]:
        seed = resolve_seed(self.shard_seed)
        # Propagate the random seed
        extra_fields = [ExtraField.from_dict({"seed": seed, **field_cfg}) for field_cfg in self.extra_fields or ()]
        for data in self.source:
            graph_token = getattr(data, "_graph_origin", None) if self.indexed else None
            # filter out entries with valid "_skipme" values.
            if data.get("_skipme", False):
                continue
            cut = self._build_cut_from_dict(data)
            for extra_field in extra_fields:
                extra_field.attach_to(cut)
            if graph_token is not None:
                attach_graph_origin(cut, graph_token)
            yield cut

    def __getitem__(self, token):
        token = normalize_graph_token(token)
        if self.extra_fields:
            raise NotImplementedError(
                "LazyNeMoIterator does not support __getitem__ when extra_fields are configured."
            )
        data = self.source[token]
        cut = self._build_cut_from_dict(data)
        return attach_graph_origin(cut, token) if self.indexed else cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)

    def state_dict(self) -> dict:
        if not self.indexed:
            return {}
        return {"source": self.source.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        if "source" in sd:
            self.source.load_state_dict(sd["source"])

    def _build_cut_from_dict(self, data: dict) -> Cut:
        # Note: ``data`` may be reused across calls in indexed mode (the reader returns
        # a fresh dict each time, but we still avoid mutating the inner object).
        data = dict(data)
        audio_path = get_full_path(str(data.pop("audio_filepath")), str(self.path), force_cache=False)
        duration = data.pop("duration")
        offset = data.pop("offset", None)
        sampling_rate = data.pop("sampling_rate", None)
        if sampling_rate is None:
            sampling_rate = data.pop("sample_rate", None)
        cut = self._create_cut(
            audio_path=audio_path,
            offset=offset,
            duration=duration,
            sampling_rate=sampling_rate,
        )
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.recording_id,
                start=0,
                duration=cut.duration,
                channel=cut.channel,
                text=data.get(self.text_field),
                language=data.get(self.lang_field),
            )
        )
        cut.custom = data
        return cut

    def _create_cut(
        self,
        audio_path: str,
        offset: float,
        duration: float,
        sampling_rate: int | None = None,
    ) -> Cut:
        if not self.metadata_only:
            recording = self._create_recording(audio_path, duration, sampling_rate)
            cut = recording.to_cut()
            if offset is not None:
                cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
                cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
        else:
            # Only metadata requested.
            # We'll provide accurate metadata for Cut but inaccurate metadata for Recording to avoid
            # incurring IO penalty (note that Lhotse manifests contain more information than
            # NeMo manifests, so for actual dataloading we have to fill it using the audio file).
            sr = ifnone(sampling_rate, 16000)  # fake sampling rate
            offset = ifnone(offset, 0.0)
            cut = MonoCut(
                id=audio_path,
                start=offset,
                duration=duration,
                channel=0,
                supervisions=[],
                recording=Recording(
                    id=audio_path,
                    sources=[AudioSource(type="dummy", channels=[0], source="")],
                    sampling_rate=sr,
                    duration=offset + duration,
                    num_samples=compute_num_samples(offset + duration, sr),
                ),
            )
        return cut

    def _create_recording(
        self,
        audio_path: str,
        duration: float,
        sampling_rate: int | None = None,
    ) -> Recording:
        if sampling_rate is not None:
            # TODO(pzelasko): It will only work with single-channel audio in the current shape.

            source_type = "url" if is_datastore_path(audio_path) else "file"
            return Recording(
                id=audio_path,
                sources=[AudioSource(type=source_type, channels=[0], source=audio_path)],
                sampling_rate=sampling_rate,
                num_samples=compute_num_samples(duration, sampling_rate),
                duration=duration,
                channel_ids=[0],
            )
        else:
            return Recording.from_file(audio_path)


class LazyNeMoTarredIterator(IteratorNode):
    r"""
    ``LazyNeMoTarredIterator`` reads a NeMo tarred JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text" (overridable with text_field argument)
    - "shard_id"

    Specially supported keys are:
    - "lang" is mapped to Lhotse superivsion's language (overridable with ``lang_field`` argument)

    Every other key found in the manifest will be attached to Lhotse Cut and accessible via ``cut.custom[key]``.

    Args ``manifest_path`` and ``tar_paths`` can be either a path/string to a single file, or a string in NeMo format
    that indicates multiple paths (e.g. "[[data/bucket0/tarred_audio_paths.json],[data/bucket1/...]]").
    We discover shard ids from sharded tar and json files by parsing the input specifier/path and
    searching for the following pattern: ``(manifest|audio)[^/]*_(\d+)[^/]*\.(json|tar)``.
    It allows filenames such as ``manifest_0.json``, ``manifest_0_normalized.json``, ``manifest_normalized_0.json``,
    ``manifest_0.jsonl.gz``, etc. (anologusly the same applies to tar files).

    We also support generalized input specifiers that imitate webdataset's pipes (also very similar to Kaldi's pipes).
    These are arbitrary shell commands to be lazily executed which yield manifest or tar audio contents.
    For example, ``tar_paths`` can be set to ``pipe:ais get ais://my-bucket/audio_{0..127}.tar -``
    to indicate that we want to read tarred audio data from shards on an AIStore bucket.
    This can be used for other cloud storage APIs such as S3, GCS, etc.
    The same mechanism applies to ``manifest_path``.

    If your data has been filtered so that the JSON manifests refer to just a subset of recordings,
    set ``skip_missing_manifest_entries` to ``True``.
    This will still read the tar files sequentially (very fast) and discard the audio files that
    are not present in the corresponding manifest.

    The ``shard_seed`` argument is used to seed the RNG shuffling the shards.
    By default, it's ``trng`` which samples a seed number from OS-provided TRNG (see Python ``secrets`` module).
    Seed is resolved lazily so that every dataloading worker may sample a different one.
    Override with an integer value for deterministic behaviour and consult Lhotse documentation for details:
    https://lhotse.readthedocs.io/en/latest/datasets.html#handling-random-seeds

    Set ``slice_length`` to enable random slicing mode: for each shard, we'll randomly select an offset K
    and skip the first K examples (but will actually read them first). Then, we'll yield only ``slice_length``
    examples. This setting can improve the sampling randomness when there are many datasets with many shards
    but only a limited run time.

    Example of CutSet with inter-shard shuffling enabled::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path=["nemo_manifests/sharded_manifests/manifest_0.json", ...],
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     shuffle_shards=True,
        ... ))

    We allow attaching custom metadata to cuts from files other than the manifest via ``extra_fields`` argument.
    In the example below, we'll iterate file "questions.txt" together with the manifest and attach each line
    under ``cut.question`` using the field type ``text_iter``::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path=["nemo_manifests/sharded_manifests/manifest_0.json", ...],
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     extra_fields=[{"type": "text_iter", "name": "question", "path": "questions.txt"}],
        ... ))

    We also support random sampling of lines with field type ``text_sample``::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path=["nemo_manifests/sharded_manifests/manifest_0.json", ...],
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     extra_fields=[{"type": "text_sample", "name": "question", "path": "questions.txt"}],
        ... ))
    """

    def __init__(
        self,
        manifest_path: str | Path | list[str],
        tar_paths: str | list,
        shuffle_shards: bool = False,
        shard_seed: int | Literal["trng", "randomized"] = "trng",
        text_field: str = "text",
        lang_field: str = "lang",
        skip_missing_manifest_entries: bool = False,
        extra_fields: list[dict[str, str]] | None = None,
        slice_length: int = None,
        indexed: bool = False,
        indexes_root: str | Path | None = None,
    ) -> None:
        self.skip_missing_manifest_entries = skip_missing_manifest_entries
        self._malformed_manifest_warning_keys: set[tuple[str, ShardKey]] = set()
        self.indexed = indexed
        self.indexes_root = indexes_root
        self.shard_id_to_manifest: dict[ShardKey, Iterable[dict]]
        self._shard_key_to_manifest_path: dict[ShardKey, str] = {}
        self.paths = expand_sharded_filepaths(manifest_path)
        if len(self.paths) == 1:
            if not indexed:
                logging.warning(
                    f"You are using Lhotse dataloading for tarred audio with a non-sharded manifest. "
                    f"This will incur significant memory overhead. To prevent this, please shard file "
                    f"'{self.paths[0]}' using 'scripts/speech_recognition/convert_to_tarred_audio_dataset.py' "
                    f"WITHOUT '--no_shard_manifest'"
                )
            self.source = LazyJsonlIterator(self.paths[0])
            if indexed:
                # In indexed mode we will not consume self.source for grouping — the per-shard
                # IndexedJsonlReaders below take over, keyed by the position-derived shard_id 0.
                self.shard_id_to_manifest = {0: self.source}
            else:
                self.shard_id_to_manifest = groupby("shard_id", self.source)
        else:
            json_pattern = re.compile(r"manifest[^/]*_(\d+)[^/]*\.json")
            shard_keys, _ = _extract_unique_shard_keys(self.paths, json_pattern, path_kind="manifest")
            self._shard_key_to_manifest_path = {key: path for key, path in zip(shard_keys, self.paths)}
            self.shard_id_to_manifest = {
                key: LazyJsonlIterator(path) for key, path in self._shard_key_to_manifest_path.items()
            }
            self.source = LazyIteratorChain(*self.shard_id_to_manifest.values())

        self.tar_paths = expand_sharded_filepaths(tar_paths)
        tar_pattern = re.compile(r"audio[^/]*_(\d+)[^/]*\.tar")
        shard_keys, _ = _extract_unique_shard_keys(self.tar_paths, tar_pattern, path_kind="tar")
        self.shard_id_to_tar_path: dict[ShardKey, str] = {key: path for key, path in zip(shard_keys, self.tar_paths)}

        self.shuffle_shards = shuffle_shards
        self.shard_seed = shard_seed
        self.text_field = text_field
        self.lang_field = lang_field
        self.extra_fields = extra_fields
        self.slice_length = slice_length
        self.epoch = 0
        self._validate()
        self.use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"

        if indexed:
            self._init_indexed()

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def _init_indexed(self) -> None:
        """Build per-shard IndexedJsonlReaders + audio-tar index for indexed/random access."""
        from lhotse.indexing import IndexedJsonlReader, index_file_path

        from nemo.collections.common.data.lhotse.indexed_adapters import IndexedTarMemberReader

        if self.extra_fields:
            raise ValueError(
                "LazyNeMoTarredIterator(indexed=True) does not support 'extra_fields' "
                "because their values are positional and cannot be reproduced under "
                "graph-token random access."
            )
        if self.slice_length is not None:
            raise ValueError("LazyNeMoTarredIterator(indexed=True) does not support 'slice_length'.")

        # Order shards by stable shard key so global indices are reproducible.
        # Multi-bucket NeMo specs may expand to paths such as
        # bucket_1/audio_0.tar and bucket_2/audio_0.tar; the occurrence suffix in
        # ShardKey prevents those duplicate numeric shard ids from overwriting.
        self._sorted_shard_ids: list[ShardKey] = sorted(self.shard_id_to_tar_path.keys())
        self._cuts_readers: dict[ShardKey, IndexedJsonlReader] = {}
        # In USE_AIS_GET_BATCH mode we never open the tar files locally — audio is
        # fetched lazily via URL/file AudioSource by AudioSamples (typically batched).
        self._tar_readers: dict[ShardKey, IndexedTarMemberReader] = {}

        # Map shard key → manifest path (single or multi-file).
        if len(self.paths) == 1:
            shard_id_to_manifest_path = {sid: self.paths[0] for sid in self._sorted_shard_ids}
        else:
            shard_id_to_manifest_path = self._shard_key_to_manifest_path

        cum = 0
        cum_lens = [0]
        for sid in self._sorted_shard_ids:
            jsonl_path = shard_id_to_manifest_path[sid]
            tar_path = self.shard_id_to_tar_path[sid]
            self._cuts_readers[sid] = IndexedJsonlReader(
                jsonl_path, index_path=index_file_path(jsonl_path, self.indexes_root)
            )
            if not self.use_ais_get_batch:
                self._tar_readers[sid] = IndexedTarMemberReader(
                    tar_path, idx_path=index_file_path(tar_path, self.indexes_root)
                )
            cum += len(self._cuts_readers[sid])
            cum_lens.append(cum)
        self._cum_lens = cum_lens
        self._total_len = cum
        self._iter_state = PartitionedIndexedIterator()

    def to_shards(self) -> List["LazyNeMoTarredIterator"]:
        """Convert this iterator to a list of separate iterators for each shard.

        Forwards every constructor knob (notably ``indexed``/``indexes_root``,
        ``extra_fields``, ``slice_length``, ``skip_missing_manifest_entries``)
        so per-shard sub-iterators behave identically to the parent. Dropping
        these silently re-enters streaming mode, which a downstream caller
        like ``mux(..., max_open_streams=N)`` won't notice until the bucketer
        fails to checkpoint.
        """
        if len(self.paths) == 1:
            # Cannot do that if the JSON manifest is a single file for all shards;
            # just return self.
            return [self]
        else:
            return [
                LazyNeMoTarredIterator(
                    manifest_path=path,
                    tar_paths=tarpath,
                    shuffle_shards=False,
                    shard_seed=self.shard_seed,
                    text_field=self.text_field,
                    lang_field=self.lang_field,
                    skip_missing_manifest_entries=self.skip_missing_manifest_entries,
                    extra_fields=self.extra_fields,
                    slice_length=self.slice_length,
                    indexed=self.indexed,
                    indexes_root=self.indexes_root,
                )
                for path, tarpath in zip(self.paths, self.shard_id_to_tar_path.values())
            ]

    def _validate(self) -> None:
        if self.indexed:
            # Indexed mode pairs tar and manifest paths by stable shard key in
            # ``_init_indexed``. The streaming-time shard_id consistency check below
            # would otherwise reject single-file inputs when the jsonl groups by a
            # different shard_id field.
            validate_extra_fields(self.extra_fields)
            return
        shard_ids_tars = set(self.shard_id_to_tar_path)
        shard_ids_manifest = set(self.shard_id_to_manifest)
        assert shard_ids_tars == shard_ids_manifest, (
            f"Mismatch between shard IDs. Details:\n"
            f"* JSON manifest(s) {self.paths}\n"
            f"* Tar files: {self.tar_paths}\n"
            f"* JSON manifest(s) indicate(s) IDs: {sorted(shard_ids_manifest)}\n"
            f"* Tar path(s) indicate(s) IDs: {sorted(shard_ids_tars)}\n"
        )
        validate_extra_fields(self.extra_fields)

    def _get_seed(self) -> int:
        return resolve_seed(self.shard_seed) + self.epoch

    @property
    def shard_ids(self) -> List[ShardKey]:
        return sorted(self.shard_id_to_manifest.keys())

    def _iter_batch_for_ais_get_batch(
        self, tar_path, shard_manifest, manifest_path, rng, extra_fields
    ) -> Generator[Cut, None, None]:
        """
        Iterator for batch reading mode (AIS get batch).
        Yields cuts with URL-based recordings without opening tar files.
        """
        # Calculate slice offset for random skipping
        total_entries = sum(len(entries) for entries in shard_manifest.values())
        slice_offset = (
            rng.randint(0, total_entries - self.slice_length)
            if self.slice_length is not None and self.slice_length < total_entries
            else -1
        )
        cntr = 0
        entries_processed = 0

        for audio_filename, manifest_entries in shard_manifest.items():
            for data in manifest_entries:
                # Skip entries if we haven't reached the slice offset yet
                if entries_processed < slice_offset:
                    entries_processed += 1
                    continue
                # Stop if we've reached the slice length limit
                elif cntr == self.slice_length:
                    break

                # filter out entries with valid "_skipme" values.
                if data.get("_skipme", False):
                    entries_processed += 1
                    continue

                # Construct URL: tar_path/audio_filename
                audio_url = f"{tar_path.rstrip('/')}/{audio_filename.lstrip('/')}"

                # Get metadata from manifest
                duration = data.get("duration")
                if duration is None:
                    logging.warning(f"Skipping '{audio_filename}' - missing duration in manifest")
                    entries_processed += 1
                    continue

                offset = data.get("offset", 0.0)
                sampling_rate = data.get("sampling_rate", 16000)  # default to 16kHz if not specified

                # Create URL-based recording
                recording = Recording(
                    id=audio_filename,
                    sources=[AudioSource(type="url", channels=[0], source=audio_url)],
                    sampling_rate=sampling_rate,
                    num_samples=compute_num_samples(duration, sampling_rate),
                    duration=duration,
                )

                # Create cut from recording (audio will be loaded lazily from URL when needed)
                cut = recording.to_cut()
                if offset > 0:
                    cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
                    cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"

                # Add supervision (transcript metadata)
                cut.supervisions.append(
                    SupervisionSegment(
                        id=cut.id,
                        recording_id=cut.recording_id,
                        start=0,
                        duration=cut.duration,
                        text=data.get(self.text_field),
                        language=data.get(self.lang_field),
                    )
                )

                # Attach custom fields and metadata
                cut.custom = _to_custom_attr_dict(data)
                cut.manifest_origin = manifest_path
                cut.tar_origin = tar_path
                for extra_field in extra_fields:
                    extra_field.attach_to(cut)

                cntr += 1
                entries_processed += 1
                yield cut

            # Break outer loop if we've reached the slice length limit
            if cntr == self.slice_length:
                break

    def _iter_sequential(
        self, tar_path, shard_manifest, manifest_path, rng
    ) -> Generator[tuple[dict, bytes], None, None]:
        slice_offset = (
            rng.randint(0, len(shard_manifest) - self.slice_length)
            if self.slice_length is not None and self.slice_length < len(shard_manifest)
            else -1
        )
        cntr = 0
        with tarfile.open(fileobj=open_best(tar_path, mode="rb"), mode="r|*") as tar:
            for idx, tar_info in enumerate(tar):
                if idx < slice_offset:
                    continue
                elif cntr == self.slice_length:
                    break
                try:
                    data = shard_manifest[tar_info.name]
                    raw_audio = tar.extractfile(tar_info).read()
                    yield data, raw_audio, tar_info
                    cntr += 1
                except KeyError as e:
                    if self.skip_missing_manifest_entries:
                        continue
                    else:
                        raise RuntimeError(
                            f"Mismatched entry between JSON manifest ('{manifest_path}') and tar file ('{tar_path}'). "
                            f"Cannot locate JSON entry for tar file '{tar_info.name}'"
                        ) from e

    # ---------------------------------------------------------------------- indexed
    def _resolve_global_idx(self, idx: int) -> tuple[ShardKey, int]:
        if idx < 0:
            idx += self._total_len
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"index {idx} out of range for LazyNeMoTarredIterator with {self._total_len} cuts")
        shard_pos = bisect.bisect_right(self._cum_lens, idx) - 1
        sid = self._sorted_shard_ids[shard_pos]
        return sid, idx - self._cum_lens[shard_pos]

    def _audio_member_name_from_entry(self, entry: dict) -> str:
        af = entry["audio_filepath"]
        m = _OFFSET_PATTERN.match(af)
        if m is None:
            return af
        return m.group("stem") + ifnone(m.group("ext"), "")

    def _attach_supervision_and_metadata(self, cut: Cut, data: dict, manifest_path: str, tar_path: str) -> Cut:
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.recording_id,
                start=0,
                duration=cut.duration,
                text=data.get(self.text_field),
                language=data.get(self.lang_field),
            )
        )
        cut.custom = _to_custom_attr_dict(data)
        cut.manifest_origin = manifest_path
        cut.tar_origin = tar_path
        return cut

    def _build_indexed_cut(self, data: dict, audio_bytes: bytes, manifest_path: str, tar_path: str) -> Cut | None:
        """Decode a single (manifest_entry, audio_bytes) pair into a Cut, mirroring the streaming path."""
        if data.get("_skipme", False):
            return None
        try:
            meta = soundfile.info(BytesIO(audio_bytes))
        except Exception:
            logging.warning(
                f"Skipped corrupted audio member referenced by '{data.get('audio_filepath')}' in {tar_path=}."
            )
            return None
        recording = Recording(
            id=str(data["audio_filepath"]),
            sources=[AudioSource(type="memory", channels=list(range(meta.channels)), source=audio_bytes)],
            sampling_rate=int(meta.samplerate),
            num_samples=meta.frames,
            duration=meta.duration,
        )
        cut = make_cut_with_subset_inmemory_recording(
            recording, offset=data.get("offset", 0.0), duration=data.get("duration")
        )
        return self._attach_supervision_and_metadata(cut, data, manifest_path, tar_path)

    def _build_indexed_url_cut(self, data: dict, manifest_path: str, tar_path: str) -> Cut | None:
        """
        AIS GetBatch counterpart of ``_build_indexed_cut``: produces a Cut backed
        by a URL/file AudioSource (no audio bytes loaded), so that
        ``AudioSamples(use_batch_loader=True)`` can fetch the entire minibatch in
        a single AIS GetBatch request. Mirrors ``_iter_batch_for_ais_get_batch``.
        """
        if data.get("_skipme", False):
            return None
        duration = data.get("duration")
        if duration is None:
            logging.warning(f"Skipping '{data.get('audio_filepath')}' - missing duration in manifest")
            return None
        audio_filename = self._audio_member_name_from_entry(data)
        audio_url = f"{tar_path.rstrip('/')}/{audio_filename.lstrip('/')}"
        # ``open_best`` handles ais://, http(s)://, and local paths uniformly;
        # the AIS GetBatch loader still keys off the URL scheme.
        source_type = "url" if "://" in tar_path else "file"
        offset = data.get("offset", 0.0)
        sampling_rate = data.get("sampling_rate", 16000)
        recording = Recording(
            id=audio_filename,
            sources=[AudioSource(type=source_type, channels=[0], source=audio_url)],
            sampling_rate=sampling_rate,
            num_samples=compute_num_samples(duration, sampling_rate),
            duration=duration,
        )
        cut = recording.to_cut()
        if offset > 0:
            cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
            cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
        return self._attach_supervision_and_metadata(cut, data, manifest_path, tar_path)

    def _decode_cut_at(self, idx: int) -> Cut | None:
        """Build the Cut for a global index in indexed mode (AIS or local).

        Returns ``None`` if the manifest entry/audio member is missing or
        malformed and ``skip_missing_manifest_entries`` is set, or if the
        entry has ``_skipme=True`` / undecodable audio.
        """
        sid, local_idx = self._resolve_global_idx(idx)
        cuts_reader = self._cuts_readers[sid]
        manifest_path = cuts_reader.path
        try:
            data = cuts_reader[local_idx]
        except (json.JSONDecodeError, UnicodeDecodeError):
            if self.skip_missing_manifest_entries:
                warning_key = (str(manifest_path), sid)
                if warning_key not in self._malformed_manifest_warning_keys:
                    self._malformed_manifest_warning_keys.add(warning_key)
                    logging.warning(
                        "Skipping malformed manifest entries in indexed Lhotse dataloader: "
                        f"{manifest_path=} {sid=} first_local_idx={local_idx} first_global_idx={idx}. "
                        "Further malformed entries for this manifest/shard will be skipped without additional "
                        "warnings."
                    )
                return None
            raise
        tar_path = self.shard_id_to_tar_path[sid]
        if self.use_ais_get_batch:
            return self._build_indexed_url_cut(data, manifest_path, tar_path)
        member_name = self._audio_member_name_from_entry(data)
        try:
            audio_bytes = self._tar_readers[sid].get(member_name)
        except KeyError:
            if self.skip_missing_manifest_entries:
                return None
            raise
        return self._build_indexed_cut(data, audio_bytes, manifest_path, tar_path)

    def __getitem__(self, token):
        if not self.indexed:
            raise NotImplementedError(
                "LazyNeMoTarredIterator only supports __getitem__ when constructed with indexed=True."
            )
        idx = int(normalize_graph_token(token))
        cut = self._decode_cut_at(idx)
        if cut is None:
            raise IndexError(f"Cut at global index {idx} is not decodable; cannot satisfy random-access __getitem__.")
        return attach_graph_origin(cut, idx)

    def __len__(self) -> int:
        if self.indexed:
            return self._total_len
        return len(self.source)

    def state_dict(self) -> dict:
        if not self.indexed:
            return {}
        return {**self._iter_state.state_dict(), "epoch": self.epoch}

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        self._iter_state.load_state_dict(sd)
        self.epoch = sd.get("epoch", 0)

    def _iter_indexed(self) -> Generator[Cut, None, None]:
        for global_idx in self._iter_state.iterate(self._total_len):
            cut = self._decode_cut_at(global_idx)
            if cut is None:
                continue
            attach_graph_origin(cut, global_idx)
            yield cut
        self.epoch += 1

    # ---------------------------------------------------------------- streaming
    def __iter__(self) -> Generator[Cut, None, None]:
        if self.indexed:
            yield from self._iter_indexed()
            return

        shard_ids = self.shard_ids

        seed = self._get_seed()
        rng = random.Random(seed)
        if self.shuffle_shards:
            rng.shuffle(shard_ids)

        # Propagate the random seed
        extra_fields = [ExtraField.from_dict({"seed": seed, **field_cfg}) for field_cfg in self.extra_fields or ()]

        # NeMo tarred manifests can have multiple JSONL entries pointing at the
        # same audio member with -subN audio_filepath suffixes (per-offset cuts).
        for sid in shard_ids:
            manifest_path = self._shard_key_to_manifest_path[sid] if len(self.paths) > 1 else self.paths[0]

            def basename(d: dict) -> str:
                return (
                    m.group("stem") + ifnone(m.group("ext"), "")
                    if (m := _OFFSET_PATTERN.match(k := d["audio_filepath"])) is not None
                    else k
                )

            shard_manifest: dict[str, list[dict]] = groupby(basename, self.shard_id_to_manifest[sid])
            tar_path = self.shard_id_to_tar_path[sid]

            if self.use_ais_get_batch:
                # Use batch reading mode - URL-based recordings without opening tar files
                yield from self._iter_batch_for_ais_get_batch(
                    tar_path, shard_manifest, manifest_path, rng, extra_fields
                )
                continue
            try:
                for data, raw_audio, tar_info in self._iter_sequential(tar_path, shard_manifest, manifest_path, rng):
                    try:
                        meta = soundfile.info(BytesIO(raw_audio))
                    except Exception:
                        logging.warning(f"Skipped corrupted file '{tar_info.path}' in {tar_path=}.")
                        continue
                    recording = Recording(
                        id=tar_info.path,
                        sources=[AudioSource(type="memory", channels=list(range(meta.channels)), source=raw_audio)],
                        sampling_rate=int(meta.samplerate),
                        num_samples=meta.frames,
                        duration=meta.duration,
                    )
                    cuts_for_recording = []
                    for data in sorted(shard_manifest[tar_info.name], key=lambda d: d["audio_filepath"]):
                        # filter out entries with valid "_skipme" values.
                        if data.get("_skipme", False):
                            continue
                        # Cut the recording into corresponding segment and discard audio data outside the segment.
                        cut = make_cut_with_subset_inmemory_recording(
                            recording, offset=data.get("offset", 0.0), duration=data.get("duration")
                        )
                        cut.supervisions.append(
                            SupervisionSegment(
                                id=cut.id,
                                recording_id=cut.recording_id,
                                start=0,
                                duration=cut.duration,
                                text=data.get(self.text_field),
                                language=data.get(self.lang_field),
                            )
                        )
                        cut.custom = _to_custom_attr_dict(data)
                        cut.manifest_origin = manifest_path
                        cut.tar_origin = tar_path
                        for extra_field in extra_fields:
                            extra_field.attach_to(cut)
                        cuts_for_recording.append(cut)
                    del recording  # free the memory - helps with very large audio files
                    del raw_audio
                    yield from cuts_for_recording
            except tarfile.ReadError:
                logging.warning(
                    f"Skipping tar file due to read errors (unstable storage or bad file?): {tar_path=}",
                )

        self.epoch += 1

    def __add__(self, other):
        return LazyIteratorChain(self, other)


def make_cut_with_subset_inmemory_recording(
    recording: Recording, offset: float = 0.0, duration: float | None = None
) -> Cut:
    """
    This method is built specifically to optimize CPU memory usage during dataloading
    when reading tarfiles containing very long recordings (1h+).
    Normally each cut would hold a reference to the long in-memory recording and load
    the necessary subset of audio (there wouldn't be a separate copy of the long recording for each cut).
    This is fairly efficient already, but we don't actually need to hold the unused full recording in memory.
    Instead, we re-create each cut so that it only holds a reference to the subset of recording necessary.
    This allows us to discard unused data which would otherwise be held in memory as part of sampling buffering.
    """

    # Fast path: no offset and (almost) matching duration (within 200ms; leeway for different audio codec behavior).
    cut = recording.to_cut()
    if offset == 0.0 and duration is None or abs(duration - recording.duration) < 0.2:
        return cut

    # Otherwise, apply the memory optimization.
    try:
        cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
    except Exception as e:
        raise RuntimeError(
            f"Lhotse cut.truncate failed with offset={offset}, duration={duration}, recording={recording}: {e}"
        ) from e

    audiobytes = BytesIO()
    LibsndfileBackend().save_audio(audiobytes, cut.load_audio(), sampling_rate=cut.sampling_rate, format="wav")
    audiobytes.seek(0)
    new_recording = Recording(
        id=recording.id,
        sampling_rate=recording.sampling_rate,
        num_samples=cut.num_samples,
        duration=cut.duration,
        sources=[
            AudioSource(
                type="memory",
                channels=recording.channel_ids,
                source=audiobytes.getvalue(),
            )
        ],
    )
    return new_recording.to_cut()


class ExtraField:
    TYPE = None
    SUPPORTED_TYPES = {}

    def attach_to(self, cut):
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in ExtraField.SUPPORTED_TYPES:
            ExtraField.SUPPORTED_TYPES[cls.TYPE] = cls
        super().__init_subclass__(**kwargs)

    @staticmethod
    def from_dict(data: dict) -> "ExtraField":
        assert data["type"] in ExtraField.SUPPORTED_TYPES, f"Unknown transform type: {data['type']}"
        return ExtraField.SUPPORTED_TYPES[data["type"]](**{k: v for k, v in data.items() if k != 'type'})

    @classmethod
    def is_supported(cls, field_type: str) -> bool:
        return field_type in cls.SUPPORTED_TYPES

    @classmethod
    def supported_types(cls) -> list[str]:
        return list(cls.SUPPORTED_TYPES)


class TextIteratorExtraField(ExtraField):
    TYPE = "text_iter"

    def __init__(self, name: str, path: str, seed=None):
        self.name = name
        self.path = path
        self.iterator = None

    def _maybe_init(self):
        if self.iterator is None:
            self.iterator = iter(map(str.strip, open_best(self.path)))

    def attach_to(self, cut):
        self._maybe_init()
        try:
            attached_value = next(self.iterator)
        except StopIteration:
            raise RuntimeError(f"Not enough lines in file {self.path} to attach to cuts under field {self.name}.")
        setattr(cut, self.name, attached_value)
        return cut


class TextSampleExtraField(ExtraField):
    TYPE = "text_sample"

    def __init__(self, name: str, path: str, seed: int | str):
        self.name = name
        self.path = path
        self.seed = seed
        self.population = None
        self.rng = None

    def _maybe_init(self):
        if self.population is None:
            self.population = list(map(str.strip, open_best(self.path)))
            self.rng = random.Random(resolve_seed(self.seed))

    def attach_to(self, cut):
        self._maybe_init()
        attached_value = self.rng.choice(self.population)
        setattr(cut, self.name, attached_value)
        return cut


def validate_extra_fields(extra_fields):
    if extra_fields is None:
        return
    assert isinstance(
        extra_fields, Sequence
    ), f"The argument provided to 'extra_fields' must be a list of dicts. We received {extra_fields=}"
    for field in extra_fields:
        assert isinstance(
            field, Mapping
        ), f"Each item in 'extra_fields' must be a dict. We received {field=} in {extra_fields=}"
        field_type = field.get("type")
        assert ExtraField.is_supported(field_type), (
            f"Each item in 'extra_fields' must contain a 'type' field with one of "
            f"the supported values ({ExtraField.supported_types()}). "
            f"We got {field_type=} in {extra_fields=}"
        )
        assert "name" in field, (
            f"Each item in 'extra_fields' must contain a 'name' field so that the field is available under cut.<name>."
            f"We found {field=} in {extra_fields=}"
        )


def expand_sharded_filepaths(paths: str | Path | list[str]) -> list[str]:
    # local import to avoid circular imports
    from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths as _expand_sharded_filepaths

    if isinstance(paths, Path):
        paths = str(paths)

    return _expand_sharded_filepaths(paths, shard_strategy="replicate", world_size=1, global_rank=0)


def _to_custom_attr_dict(d: dict, _excluded_fields: set[str] = {"duration", "audio_filepath"}) -> dict:
    return {k: v for k, v in d.items() if k not in _excluded_fields}


class LazyParquetIterator(IteratorNode):
    """
    LazyParquetIterator reads a Parquet file (local or remote) and yields Lhotse Cut objects.
    It streams data using PyArrow's iter_batches to avoid loading the full file into memory.

    Args:
        path (str | Path): Path to the .parquet file.
        audio_field (str): Name of the column containing audio bytes (default: "audio").
        text_field (str): Name of the column containing transcript (default: "text").
        duration_field (str): Name of the column containing duration (default: "duration").
        lang_field (str): Name of the column containing language (default: "lang").
        sampling_rate (int): Fallback sampling rate if not found in metadata (default: 16000).
        indexed (bool): When True, enable O(1) random access via row-group lookup
            and graph-token checkpointing. Requires the parquet file to expose
            row-group statistics (the default for files written by pyarrow/pandas).

    Indexed mode reads one row group at a time on demand and caches the most
    recently used row group, so unshuffled or locality-friendly access patterns
    avoid repeated decompression.
    """

    def __init__(
        self,
        path: str | Path,
        audio_field: str = "audio",
        text_field: str = "text",
        duration_field: str = "duration",
        lang_field: str = "lang",
        sampling_rate: int = 16000,
        indexed: bool = False,
    ) -> None:
        # SAFETY CHECK: Ensure pyarrow is actually installed
        if not HAVE_PYARROW:
            raise ImportError(
                "PyArrow is required to read Parquet manifests. Please install it using: pip install pyarrow"
            )

        self.path = str(path)
        self.audio_field = audio_field
        self.text_field = text_field
        self.duration_field = duration_field
        self.lang_field = lang_field
        self.sampling_rate = sampling_rate
        self.indexed = indexed
        self._row_group_offsets: list[int] | None = None
        self._num_row_groups: int | None = None
        self._total_rows: int | None = None
        self._cached_row_group_idx: int | None = None
        self._cached_row_group: list[dict] | None = None
        self._iter_state = PartitionedIndexedIterator()
        if indexed:
            self._ensure_row_group_offsets()

    @property
    def is_checkpointable(self) -> bool:
        return self.indexed

    @property
    def is_indexed(self) -> bool:
        return self.indexed

    @property
    def has_constant_time_access(self) -> bool:
        return self.indexed

    def _ensure_row_group_offsets(self) -> None:
        if self._row_group_offsets is not None:
            return
        try:
            with closing(pq.ParquetFile(self.path)) as parquet_file:
                offsets = [0]
                for i in range(parquet_file.num_row_groups):
                    offsets.append(offsets[-1] + parquet_file.metadata.row_group(i).num_rows)
                self._row_group_offsets = offsets
                self._num_row_groups = parquet_file.num_row_groups
                self._total_rows = offsets[-1]
        except Exception as e:
            raise RuntimeError(f"Failed to open Parquet file: {self.path}") from e

    def _load_row_group(self, rg_idx: int) -> list[dict]:
        if self._cached_row_group_idx == rg_idx and self._cached_row_group is not None:
            return self._cached_row_group
        with closing(pq.ParquetFile(self.path)) as parquet_file:
            df = parquet_file.read_row_group(rg_idx).to_pandas()
        rows = df.to_dict("records")
        self._cached_row_group_idx = rg_idx
        self._cached_row_group = rows
        return rows

    def _resolve_row_group(self, idx: int) -> tuple[int, int]:
        # Find row group containing global ``idx`` via simple linear/bisect lookup.
        offsets = self._row_group_offsets
        # Linear scan is fine because num_row_groups is typically small.
        for rg_idx in range(self._num_row_groups):
            if idx < offsets[rg_idx + 1]:
                return rg_idx, idx - offsets[rg_idx]
        raise IndexError(f"index {idx} out of range for parquet file with {self._total_rows} rows")

    def _build_cut_from_row(self, row: dict, fallback_idx: int) -> Cut | None:
        audio_data = row.get(self.audio_field)
        if isinstance(audio_data, dict) and 'bytes' in audio_data:
            audio_bytes = audio_data['bytes']
        elif isinstance(audio_data, bytes):
            audio_bytes = audio_data
        else:
            logging.warning(f"Skipping row {fallback_idx}: Audio column '{self.audio_field}' format unrecognized.")
            return None

        text = row.get(self.text_field, "")
        language = row.get(self.lang_field, None)
        row_id = str(row.get('id', f"{Path(self.path).stem}_{fallback_idx}"))
        try:
            recording = Recording.from_bytes(data=audio_bytes, recording_id=row_id)
        except (RuntimeError, ValueError, TypeError) as e:
            logging.warning(f"Skipping row {row_id}: Failed to decode audio bytes. {e}")
            return None
        cut = recording.to_cut()
        cut.supervisions.append(
            SupervisionSegment(
                id=row_id,
                recording_id=row_id,
                start=0.0,
                duration=cut.duration,
                channel=0,
                text=text,
                language=language,
            )
        )
        cut.custom = {k: v for k, v in row.items() if k != self.audio_field}
        return cut

    def __getitem__(self, token):
        self._ensure_row_group_offsets()
        idx = int(normalize_graph_token(token))
        if idx < 0:
            idx += self._total_rows
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"index {token} out of range for parquet file with {self._total_rows} rows")
        rg_idx, local_idx = self._resolve_row_group(idx)
        rows = self._load_row_group(rg_idx)
        cut = self._build_cut_from_row(rows[local_idx], fallback_idx=idx)
        if cut is None:
            raise IndexError(f"Row {idx} in {self.path} is not decodable; cannot satisfy random-access __getitem__.")
        return attach_graph_origin(cut, idx)

    def __len__(self) -> int:
        self._ensure_row_group_offsets()
        return self._total_rows

    def state_dict(self) -> dict:
        if not self.indexed:
            return {}
        return self._iter_state.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        if not self.indexed:
            return
        self._iter_state.load_state_dict(sd)

    def __iter__(self) -> Generator[Cut, None, None]:
        if self.indexed:
            yield from self._iter_indexed()
        else:
            yield from self._iter_streaming()

    def _iter_indexed(self) -> Generator[Cut, None, None]:
        for global_idx in self._iter_state.iterate(self._total_rows):
            rg_idx, local_idx = self._resolve_row_group(global_idx)
            rows = self._load_row_group(rg_idx)
            cut = self._build_cut_from_row(rows[local_idx], fallback_idx=global_idx)
            if cut is None:
                continue
            attach_graph_origin(cut, global_idx)
            yield cut

    def _iter_streaming(self) -> Generator[Cut, None, None]:
        # Open Parquet file in streaming mode inside __iter__
        # This ensures each DataLoader worker gets its own file handle.
        try:
            parquet_file = pq.ParquetFile(self.path)
        except Exception as e:
            raise RuntimeError(f"Failed to open Parquet file: {self.path}") from e

        # Stream batches to keep memory usage low
        for batch in parquet_file.iter_batches():
            df = batch.to_pandas()

            for idx, row in df.iterrows():
                cut = self._build_cut_from_row(row, fallback_idx=idx)
                if cut is None:
                    continue
                yield cut


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_unique_shard_keys(
    paths: list[str], pattern: re.Pattern, *, path_kind: str
) -> tuple[list[ShardKey], list[int]]:
    """Extract shard ids while preserving duplicate ids from expanded paths.

    NeMo tarred dataset specs may contain multiple independent path dimensions,
    e.g. ``bucket_OP_1..8_CL_/audio__OP_0..127_CL_.tar``. After expansion,
    every bucket contains numeric tar shard ids ``0..127``. Keying readers only
    by that numeric id silently overwrites all but the last bucket, shrinking the
    effective dataset and causing extreme oversampling of the remaining shards.

    When numeric ids are unique, keep the historical ``int`` keys. When a
    numeric id repeats, key each occurrence as ``(shard_id, occurrence)`` so
    manifest and tar paths remain paired one-to-one across all expanded files.
    The raw ids are returned for callers that need the original parsed values.
    """
    raw_ids = []
    for path in paths:
        match = pattern.search(path)
        assert match is not None, (
            f"Cannot determine shard_id from {path_kind} input specifier: "
            f"we searched with regex '{pattern.pattern}' in input '{path}'"
        )
        raw_ids.append(int(match.group(1)))
    if len(set(raw_ids)) == len(raw_ids):
        return raw_ids, raw_ids
    occurrences: dict[int, int] = {}
    keys: list[ShardKey] = []
    for shard_id in raw_ids:
        occurrence = occurrences.get(shard_id, 0)
        occurrences[shard_id] = occurrence + 1
        keys.append((shard_id, occurrence))
    return keys, raw_ids
