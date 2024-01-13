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

import random
import re
import secrets
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

import soundfile
from cytoolz import groupby
from lhotse import AudioSource, Recording, SupervisionSegment
from lhotse.lazy import ImitatesDict, LazyIteratorChain, LazyJsonlIterator
from lhotse.serialization import open_best
from lhotse.utils import compute_num_samples


class LazyNeMoIterator(ImitatesDict):
    """
    ``LazyNeMoIterator`` reads a NeMo (non-tarred) JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text" (overridable with text_field argument)

    Every other key found in the manifest will be attached to Lhotse Cut and accessible via ``cut.custom[key]``.

    .. caution:: If sampling rate is not specified, we will perform some I/O (as much as required by soundfile.info)
        to discover the sampling rate of the audio file. Otherwise, if sampling rate is provided, we assume every
        audio file has the same sampling rate.

    Example::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator("nemo_manifests/train.json", sampling_rate=16000))
    """

    def __init__(
        self, path: str | Path, sampling_rate: int | None = None, text_field: str = "text", lang_field: str = "lang"
    ) -> None:
        self.source = LazyJsonlIterator(path)
        self.sampling_rate = sampling_rate
        self.text_field = text_field
        self.lang_field = lang_field

    @property
    def path(self) -> str | Path:
        return self.source.path

    def __iter__(self):
        for data in self.source:
            audio_path = data.pop("audio_filepath")
            duration = data.pop("duration")
            if self.sampling_rate is None:
                recording = Recording.from_file(audio_path)
            else:
                recording = Recording(
                    id=Path(audio_path).name,
                    sources=[AudioSource(type="file", channels=[0], source=audio_path)],
                    sampling_rate=self.sampling_rate,
                    duration=duration,
                    num_samples=compute_num_samples(duration, self.sampling_rate),
                )
            cut = recording.to_cut()
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.recording_id,
                    start=0,
                    duration=cut.duration,
                    text=data[self.text_field],
                    language=data.get(self.lang_field),
                )
            )
            cut.custom = data
            yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)


class LazyNeMoTarredIterator(ImitatesDict):
    """
    ``LazyNeMoTarredIterator`` reads a NeMo tarred JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text" (overridable with text_field argument)
    - "shard_id"

    Every other key found in the manifest will be attached to Lhotse Cut and accessible via ``cut.custom[key]``.

    Args ``manifest_path`` and ``tar_paths`` can be either a path/string to a single file, or a string in NeMo format
    that indicates multiple paths (e.g. "[[data/bucket0/tarred_audio_paths.json],[data/bucket1/...]]").

    Example of CutSet with inter-shard shuffling enabled::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path="nemo_manifests/train.json",
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     shuffle_shards=True,
        ... ))
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tar_paths: str | list,
        shuffle_shards: bool = False,
        text_field: str = "text",
        lang_field: str = "lang",
    ) -> None:
        def strip_pipe(p):
            if isinstance(p, str):
                if p.startswith("pipe:"):
                    p = p[5:]
                return Path(p)
            return p

        self.shard_id_to_manifest: dict[int, Iterable[dict]]
        self.paths = expand_sharded_filepaths(manifest_path)
        if len(self.paths) == 1:
            self.source = LazyJsonlIterator(self.paths[0])
            self.shard_id_to_manifest = groupby("shard_id", self.source)
        else:
            pattern = re.compile(r".+_(\d+)\.jsonl?(?:.gz)?")
            shard_ids = []
            for p in self.paths:
                m = pattern.match(p)
                assert m is not None, f"Cannot determine shard_id from manifest path: {p}"
                shard_ids.append(int(m.group(1)))
            self.shard_id_to_manifest = {sid: LazyJsonlIterator(p) for sid, p in zip(shard_ids, self.paths)}
            self.source = LazyIteratorChain(*self.shard_id_to_manifest.values())

        tar_paths = expand_sharded_filepaths(tar_paths)
        self.shard_id_to_tar_path: dict[int, str] = {int(strip_pipe(p).stem.split("_")[1]): p for p in tar_paths}
        self.shuffle_shards = shuffle_shards
        self.text_field = text_field
        self.lang_field = lang_field
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
                    text_field=self.text_field,
                    lang_field=self.lang_field,
                )
                for path, tarpath in zip(self.paths, self.shard_id_to_tar_path.values())
            ]

    def _validate(self) -> None:
        shard_ids_tars = set(self.shard_id_to_tar_path)
        shard_ids_manifest = set(self.shard_id_to_manifest)
        assert shard_ids_tars == shard_ids_manifest, (
            f"Mismatch between shard IDs discovered from tar files ({len(shard_ids_tars)=}) and "
            f"JSON manifest ({len(shard_ids_manifest)=}): {shard_ids_tars - shard_ids_manifest=}"
        )

    @property
    def shard_ids(self) -> List[int]:
        return sorted(self.shard_id_to_manifest.keys())

    def __iter__(self):
        shard_ids = self.shard_ids

        if self.shuffle_shards:
            # Use TRNG for 100% randomness
            random.Random(secrets.randbelow(2 ** 32)).shuffle(shard_ids)

        for sid in shard_ids:
            shard_manifest = self.shard_id_to_manifest[sid]
            tar_path = self.shard_id_to_tar_path[sid]
            with tarfile.open(fileobj=open_best(tar_path, mode="rb"), mode="r|*") as tar:
                for data, tar_info in zip(shard_manifest, tar):
                    print(f"{data=}, {tar_info=}")
                    assert (
                        data["audio_filepath"] == tar_info.name
                    ), f"Mismatched JSON manifest and tar file. {data['audio_filepath']=} != {tar_info.name=}"
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
                            text=data[self.text_field],
                            language=data.get(self.lang_field),
                        )
                    )
                    cut.custom = _to_custom_attr_dict(data)
                    yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)


def expand_sharded_filepaths(path: str | Path) -> list[str]:
    # local import to avoid circular imports
    from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths as _expand_sharded_filepaths

    return _expand_sharded_filepaths(str(path), shard_strategy="replicate", world_size=1, global_rank=0)


def _to_custom_attr_dict(d: dict, _excluded_fields: set[str] = {"duration", "audio_filepath"}) -> dict:
    return {k: v for k, v in d.items() if k not in _excluded_fields}
