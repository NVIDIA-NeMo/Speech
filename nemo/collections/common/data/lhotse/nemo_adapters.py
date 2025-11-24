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
import re
import tarfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Generator, Iterable, List, Literal

import soundfile
from cytoolz import groupby
from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment
from lhotse.audio.backend import LibsndfileBackend
from lhotse.cut import Cut
from lhotse.dataset.dataloading import resolve_seed
from lhotse.lazy import LazyIteratorChain, LazyJsonlIterator
from lhotse.serialization import open_best
from lhotse.utils import compute_num_samples, ifnone

from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging
from nemo.utils.data_utils import is_datastore_path


class LazyNeMoIterator:
    """
    ``LazyNeMoIterator`` reads a NeMo (non-tarred) JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text" (overridable with ``text_field`` argument)

    Specially supported keys are:
    - [recommended] "sampling_rate" allows us to provide a valid Lhotse ``Recording`` object without checking the audio file
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
    ) -> None:
        self.path = path
        self.shuffle_shards = shuffle_shards
        self.shard_seed = shard_seed
        paths = expand_sharded_filepaths(path)

        if len(paths) == 1:
            self.source = LazyJsonlIterator(paths[0])
        else:
            self.source = LazyIteratorChain(
                *(LazyJsonlIterator(p) for p in paths), shuffle_iters=self.shuffle_shards, seed=self.shard_seed
            )
        self.text_field = text_field
        self.lang_field = lang_field
        self.metadata_only = metadata_only
        self.extra_fields = extra_fields
        validate_extra_fields(self.extra_fields)

    def __iter__(self) -> Generator[Cut, None, None]:
        seed = resolve_seed(self.shard_seed)
        # Propagate the random seed
        extra_fields = [ExtraField.from_dict({"seed": seed, **field_cfg}) for field_cfg in self.extra_fields or ()]
        for data in self.source:
            # filter out entries with valid "_skipme" values.
            if data.get("_skipme", False):
                continue
            audio_path = get_full_path(str(data.pop("audio_filepath")), str(self.path), force_cache=False)
            duration = data.pop("duration")
            offset = data.pop("offset", None)
            cut = self._create_cut(
                audio_path=audio_path, offset=offset, duration=duration, sampling_rate=data.pop("sampling_rate", None)
            )
            # Note that start=0 and not start=offset because supervision's start if relative to the
            # start of the cut; and cut.start is already set to offset
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
            for extra_field in extra_fields:
                extra_field.attach_to(cut)
            yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)

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


class LazyNeMoTarredIterator:
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
    ) -> None:
        self.skip_missing_manifest_entries = skip_missing_manifest_entries
        self.shard_id_to_manifest: dict[int, Iterable[dict]]
        self.paths = expand_sharded_filepaths(manifest_path)
        if len(self.paths) == 1:
            logging.warning(
                f"""You are using Lhotse dataloading for tarred audio with a non-sharded manifest.
                            This will incur significant memory overhead and slow-down training. To prevent this error message
                            please shard file '{self.paths[0]}' using 'scripts/speech_recognition/convert_to_tarred_audio_dataset.py'
                            WITHOUT '--no_shard_manifest'"""
            )
            self.source = LazyJsonlIterator(self.paths[0])
            self.shard_id_to_manifest = groupby("shard_id", self.source)
        else:
            json_pattern = re.compile(r"manifest[^/]*_(\d+)[^/]*\.json")
            shard_ids = []
            for p in self.paths:
                m = json_pattern.search(p)
                assert m is not None, (
                    f"Cannot determine shard_id from manifest input specified: "
                    f"we searched with regex '{json_pattern.pattern}' in input '{p}'"
                )
                shard_ids.append(int(m.group(1)))
            self.shard_id_to_manifest = {sid: LazyJsonlIterator(p) for sid, p in zip(shard_ids, self.paths)}
            self.source = LazyIteratorChain(*self.shard_id_to_manifest.values())

        self.tar_paths = expand_sharded_filepaths(tar_paths)
        tar_pattern = re.compile(r"audio[^/]*_(\d+)[^/]*\.tar")
        shard_ids = []
        for p in self.tar_paths:
            m = tar_pattern.search(p)
            assert m is not None, (
                f"Cannot determine shard_id from tar input specifier: "
                f"we searched with regex '{tar_pattern.pattern}' in input '{p}'"
            )
            shard_ids.append(int(m.group(1)))
        self.shard_id_to_tar_path = dict(zip(shard_ids, self.tar_paths))

        self.shuffle_shards = shuffle_shards
        self.shard_seed = shard_seed
        self.text_field = text_field
        self.lang_field = lang_field
        self.extra_fields = extra_fields
        self.slice_length = slice_length
        self.epoch = 0
        self._validate()
        self.use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"

    def to_shards(self) -> List["LazyNeMoTarredIterator"]:
        """Convert this iterator to a list of separate iterators for each shard."""
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
                )
                for path, tarpath in zip(self.paths, self.shard_id_to_tar_path.values())
            ]

    def _validate(self) -> None:
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
    def shard_ids(self) -> List[int]:
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

    def __iter__(self) -> Generator[Cut, None, None]:
        shard_ids = self.shard_ids

        seed = self._get_seed()
        rng = random.Random(seed)
        if self.shuffle_shards:
            rng.shuffle(shard_ids)

        # Propagate the random seed
        extra_fields = [ExtraField.from_dict({"seed": seed, **field_cfg}) for field_cfg in self.extra_fields or ()]

        # Handle NeMo tarred manifests with offsets.
        # They have multiple JSONL entries where audio paths end with '-sub1', '-sub2', etc. for each offset.
        offset_pattern = re.compile(r'^(?P<stem>.+)(?P<sub>-sub\d+)(?P<ext>\.\w+)?$')

        for sid in shard_ids:
            manifest_path = self.paths[sid] if len(self.paths) > 1 else self.paths[0]

            def basename(d: dict) -> str:
                return (
                    m.group("stem") + ifnone(m.group("ext"), "")
                    if (m := offset_pattern.match(k := d["audio_filepath"])) is not None
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

    def __len__(self) -> int:
        return len(self.source)

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
