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

import logging
import random
import re
import tarfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Generator, Iterable, List, Literal

import lhotse.serialization
import soundfile
from cytoolz import groupby
from lhotse import AudioSource, Recording, SupervisionSegment
from lhotse.cut import Cut
from lhotse.dataset.dataloading import resolve_seed
from lhotse.lazy import LazyIteratorChain, LazyJsonlIterator
from lhotse.serialization import open_best
from lhotse.utils import compute_num_samples

from nemo.collections.common.parts.preprocessing.manifest import get_full_path


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
            audio_path = get_full_path(str(data.pop("audio_filepath")), str(self.path))
            duration = data.pop("duration")
            offset = data.pop("offset", None)
            recording = self._create_recording(audio_path, duration, data.pop("sampling_rate", None))
            cut = recording.to_cut()
            if offset is not None:
                cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
                cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
            # Note that start=0 and not start=offset because supervision's start if relative to the
            # start of the cut; and cut.start is already set to offset
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
            cut.custom = data
            for extra_field in extra_fields:
                extra_field.attach_to(cut)
            yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)

    def _create_recording(
        self,
        audio_path: str,
        duration: float,
        sampling_rate: int | None = None,
    ) -> Recording:
        if sampling_rate is not None:
            # TODO(pzelasko): It will only work with single-channel audio in the current shape.
            return Recording(
                id=audio_path,
                sources=[AudioSource(type="file", channels=[0], source=audio_path)],
                sampling_rate=sampling_rate,
                num_samples=compute_num_samples(duration, sampling_rate),
                duration=duration,
                channel_ids=[0],
            )
        elif self.metadata_only:
            return Recording(
                id=audio_path,
                sources=[AudioSource(type="file", channels=[0], source=audio_path)],
                sampling_rate=-1,
                num_samples=-1,
                duration=duration,
                channel_ids=[0],
            )
        else:
            return Recording.from_file(audio_path)


class LazyNeMoTarredIterator:
    """
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

    The ``shard_seed`` argument is used to seed the RNG shuffling the shards.
    By default, it's ``trng`` which samples a seed number from OS-provided TRNG (see Python ``secrets`` module).
    Seed is resolved lazily so that every dataloading worker may sample a different one.
    Override with an integer value for deterministic behaviour and consult Lhotse documentation for details:
    https://lhotse.readthedocs.io/en/latest/datasets.html#handling-random-seeds

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
        extra_fields: list[dict[str, str]] | None = None,
    ) -> None:
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
        self._validate()

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

    @property
    def shard_ids(self) -> List[int]:
        return sorted(self.shard_id_to_manifest.keys())

    def __iter__(self) -> Generator[Cut, None, None]:
        shard_ids = self.shard_ids

        seed = resolve_seed(self.shard_seed)
        if self.shuffle_shards:
            random.Random(seed).shuffle(shard_ids)

        # Propagate the random seed
        extra_fields = [ExtraField.from_dict({"seed": seed, **field_cfg}) for field_cfg in self.extra_fields or ()]

        for sid in shard_ids:
            manifest_path = self.paths[sid] if len(self.paths) > 1 else self.paths[0]
            shard_manifest = {data["audio_filepath"]: data for data in self.shard_id_to_manifest[sid]}
            tar_path = self.shard_id_to_tar_path[sid]
            with tarfile.open(fileobj=open_best(tar_path, mode="rb"), mode="r|*") as tar:
                for tar_info in tar:
                    assert tar_info.name in shard_manifest, (
                        f"Mismatched entry between JSON manifest ('{manifest_path}') and tar file ('{tar_path}'). "
                        f"Cannot locate JSON entry for tar file '{tar_info.name}'"
                    )
                    data = shard_manifest[tar_info.name]
                    raw_audio = tar.extractfile(tar_info).read()
                    # Note: Lhotse has a Recording.from_bytes() utility that we won't use here because
                    #       the profiling indicated significant overhead in torchaudio ffmpeg integration
                    #       that parses full audio instead of just reading the header for WAV files.
                    # recording = lhotse.Recording.from_bytes(raw_audio, recording_id=tar_info.path)
                    meta = soundfile.info(BytesIO(raw_audio))
                    recording = Recording(
                        id=tar_info.path,
                        sources=[AudioSource(type="memory", channels=list(range(meta.channels)), source=raw_audio)],
                        sampling_rate=int(meta.samplerate),
                        num_samples=meta.frames,
                        duration=meta.duration,
                    )
                    cut = recording.to_cut()
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
                    for extra_field in extra_fields:
                        extra_field.attach_to(cut)
                    yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)


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
