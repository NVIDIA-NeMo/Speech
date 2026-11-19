# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import gzip
import json
import os
import random
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from nemo.collections.asr.parts.preprocessing import perturb
from nemo.collections.asr.parts.preprocessing.perturb import (
    ImpulsePerturbation,
    NoisePerturbation,
    NoisePerturbationWithNormalization,
)
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000
NUM_FILES = 8


@pytest.fixture(params=[True, False], ids=["with_idx", "without_idx"])
def tarred_audio(tmp_path, request):
    """Eight distinct 1.2 s audio files in two NeMo-style tars, with or without .idx files.

    NoisePerturbationWithNormalization requires noise of at least 1 s.
    """
    manifest_path = tmp_path / "manifest.json"
    tar_paths = [tmp_path / f"audio_{i}.tar" for i in range(2)]
    tars = [tarfile.open(path, "w") for path in tar_paths]
    with manifest_path.open("w") as manifest:
        for i in range(NUM_FILES):
            t = np.arange(int(1.2 * SAMPLE_RATE)) / SAMPLE_RATE
            samples = 0.1 * np.sin(2 * np.pi * (200 + 150 * i) * t) * np.exp(-t * (i + 1))
            wav_path = tmp_path / f"audio_{i}.wav"
            sf.write(wav_path, samples.astype(np.float32), SAMPLE_RATE)
            tars[i % 2].add(wav_path, arcname=wav_path.name)
            manifest.write(json.dumps({"audio_filepath": wav_path.name, "duration": 1.2, "text": ""}) + "\n")
    for tar in tars:
        tar.close()
    if request.param:
        for path in tar_paths:
            create_tar_index(str(path), str(path) + ".idx")
    return str(manifest_path), [str(path) for path in tar_paths]


def _make(kind, manifest_path, tar_paths):
    if kind == "noise":
        return NoisePerturbation(manifest_path, min_snr_db=10, max_snr_db=10, audio_tar_filepaths=tar_paths)
    if kind == "impulse":
        return ImpulsePerturbation(manifest_path, audio_tar_filepaths=tar_paths, shuffle_n=4)
    return NoisePerturbationWithNormalization(
        manifest_path, min_snr_db=10, max_snr_db=10, audio_tar_filepaths=[tar_paths], shuffle_n=4
    )


def _apply(perturbation, times):
    outputs = []
    for _ in range(times):
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        segment = AudioSegment(samples=(0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sample_rate=SAMPLE_RATE)
        perturbation.perturb(segment)
        outputs.append(segment.samples.copy())
    return outputs


@pytest.mark.parametrize("kind", ["noise", "impulse", "noise_with_normalization"])
def test_tarred_audio_choice_is_restored_from_injected_rng(tarred_audio, kind):
    manifest_path, tar_paths = tarred_audio
    perturbation = _make(kind, manifest_path, tar_paths)
    rng = random.Random(0)
    perturbation.rng = rng
    _apply(perturbation, 2)
    state = rng.getstate()
    expected = _apply(perturbation, 4)

    # A resumed run builds a fresh perturbation (with a fresh tar iterator) and restores the generator state.
    resumed = _make(kind, manifest_path, tar_paths)
    restored_rng = random.Random()
    restored_rng.setstate(state)
    resumed.rng = restored_rng
    actual = _apply(resumed, 4)

    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    # Different noise/RIR files are picked over the calls.
    assert len({out.tobytes() for out in expected}) > 1


@pytest.mark.parametrize("kind", ["noise", "impulse", "noise_with_normalization"])
def test_tarred_audio_without_injected_rng_keeps_sequential_reading(tarred_audio, kind, monkeypatch):
    manifest_path, tar_paths = tarred_audio
    perturbation = _make(kind, manifest_path, tar_paths)

    def fail(*args, **kwargs):
        raise AssertionError("random access must only be used with an injected generator")

    monkeypatch.setattr(perturb.TarredAudioRandomAccess, "load", fail)

    assert len(_apply(perturbation, 3)) == 3


def _write_tar(tar_path, members):
    """Write a tar with ``(arcname, samples)`` members, in order (names may repeat)."""
    with tarfile.open(tar_path, "w") as tar:
        for i, (arcname, samples) in enumerate(members):
            wav_path = tar_path.parent / f"{tar_path.stem}_{i}.wav"
            sf.write(wav_path, samples.astype(np.float32), SAMPLE_RATE)
            tar.add(wav_path, arcname=arcname)


def _tone(frequency, seconds=1.2):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return 0.1 * np.sin(2 * np.pi * frequency * t)


def _manifest(path, entries):
    path.write_text("".join(json.dumps({"text": "", **entry}) + "\n" for entry in entries))
    return str(path)


def _entry(audio_file, offset=None, duration=1.2):
    return SimpleNamespace(audio_file=audio_file, offset=offset, duration=duration)


def test_noise_sources_keep_their_own_file_names(tmp_path):
    # Two sources each contain a different "noise.wav"; each entry must be read from its own source's tar.
    source_dirs = [tmp_path / "a", tmp_path / "b"]
    manifests, tars = [], []
    for source_dir, frequency in zip(source_dirs, [300, 900]):
        source_dir.mkdir()
        _write_tar(source_dir / "noise.tar", [("noise.wav", _tone(frequency))])
        manifests.append(_manifest(source_dir / "manifest.json", [{"audio_filepath": "noise.wav", "duration": 1.2}]))
        tars.append(str(source_dir / "noise.tar"))

    def make():
        return NoisePerturbationWithNormalization(manifests, audio_tar_filepaths=tars, shuffle_n=1)

    perturbation = make()
    rng = random.Random(0)
    perturbation.rng = rng
    state = rng.getstate()
    noises = [perturbation.read_one_audiosegment(SAMPLE_RATE).samples for _ in range(16)]

    expected = {frequency: _tone(frequency).astype(np.float32) for frequency in [300, 900]}
    matched = [
        next(f for f, ref in expected.items() if np.allclose(noise, ref, atol=1e-4)) for noise in noises
    ]  # raises StopIteration if a noise matches neither source
    assert set(matched) == {300, 900}

    resumed = make()
    restored_rng = random.Random()
    restored_rng.setstate(state)
    resumed.rng = restored_rng
    for noise in noises:
        np.testing.assert_array_equal(resumed.read_one_audiosegment(SAMPLE_RATE).samples, noise)


def test_duplicate_tar_member_names_are_rejected(tmp_path):
    tar_path = tmp_path / "noise.tar"
    _write_tar(tar_path, [("noise.wav", _tone(300)), ("other.wav", _tone(500)), ("noise.wav", _tone(700))])

    with pytest.raises(ValueError, match="noise.wav"):
        perturb.TarredAudioRandomAccess(str(tar_path)).load(_entry("noise.wav"), SAMPLE_RATE)


def test_manifest_offsets_into_one_member_load_their_slices(tmp_path):
    tar_path = tmp_path / "noise.tar"
    samples = np.concatenate([_tone(300, 1.0), _tone(900, 1.0)])
    _write_tar(tar_path, [("noise.wav", samples)])
    access = perturb.TarredAudioRandomAccess(str(tar_path))

    first = access.load(_entry("noise.wav", offset=0.0, duration=1.0), SAMPLE_RATE).samples
    second = access.load(_entry("noise.wav", offset=1.0, duration=1.0), SAMPLE_RATE).samples

    np.testing.assert_allclose(first, _tone(300, 1.0), atol=1e-4)
    np.testing.assert_allclose(second, _tone(900, 1.0), atol=1e-4)


def test_remote_tar_uses_its_existing_index(tarred_audio, monkeypatch):
    manifest_path, tar_paths = tarred_audio
    for path in tar_paths:
        if not os.path.exists(path + ".idx"):
            create_tar_index(path, path + ".idx")
    # Treat the local tar as remote: its index must exist next to it and is never created.
    monkeypatch.setattr(perturb, "_is_remote_tar", lambda path: True)

    access = perturb.TarredAudioRandomAccess(tar_paths)
    loaded = access.load(_entry("audio_3.wav"), SAMPLE_RATE).samples

    assert loaded.shape == (int(1.2 * SAMPLE_RATE),)


def test_remote_tar_without_index_raises_actionable_error(tmp_path, monkeypatch):
    tar_path = tmp_path / "noise.tar"
    _write_tar(tar_path, [("noise.wav", _tone(300))])
    # Treat the local tar as remote: its index must exist next to it and is never created.
    monkeypatch.setattr(perturb, "_is_remote_tar", lambda path: True)

    with pytest.raises(FileNotFoundError, match="build_indexes.py"):
        perturb.TarredAudioRandomAccess(str(tar_path)).load(_entry("noise.wav"), SAMPLE_RATE)


def test_random_access_does_not_keep_tar_files_open(tmp_path):
    # Readers are closed after each use, so reading from many shards does not accumulate open files
    # (otherwise a corpus with more shards than the open-file limit fails).
    num_shards = 32
    for i in range(num_shards):
        _write_tar(tmp_path / f"noise_{i}.tar", [(f"noise_{i}.wav", _tone(200 + i, 0.1))])
    access = perturb.TarredAudioRandomAccess([str(tmp_path / f"noise_{i}.tar") for i in range(num_shards)])
    fd_dir = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    num_open_before = len(os.listdir(fd_dir))

    rng = random.Random(0)
    for _ in range(2 * num_shards):
        i = rng.randrange(num_shards)
        access.load(_entry(f"noise_{i}.wav", duration=0.1), SAMPLE_RATE)

    assert len(os.listdir(fd_dir)) <= num_open_before + 2


@pytest.mark.parametrize("with_index", [False, True], ids=["without_idx", "with_idx"])
def test_adjacent_duplicate_tar_members_are_rejected(tmp_path, with_index):
    # The index groups consecutive members with the same stem into one entry; both members must still be checked.
    tar_path = tmp_path / "noise.tar"
    _write_tar(tar_path, [("noise.wav", _tone(300)), ("noise.wav", _tone(900))])
    if with_index:
        create_tar_index(str(tar_path), str(tar_path) + ".idx")

    with pytest.raises(ValueError, match="noise.wav"):
        perturb.TarredAudioRandomAccess(str(tar_path)).load(_entry("noise.wav"), SAMPLE_RATE)


def test_compressed_tar_with_injected_rng_raises(tmp_path):
    tar_path = tmp_path / "noise.tar"
    _write_tar(tar_path, [("noise.wav", _tone(300))])
    gz_path = tmp_path / "noise.tar.gz"
    with open(tar_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    manifest = _manifest(tmp_path / "manifest.json", [{"audio_filepath": "noise.wav", "duration": 1.2}])
    perturbation = NoisePerturbation(manifest, min_snr_db=10, max_snr_db=10, audio_tar_filepaths=str(gz_path))
    perturbation.rng = random.Random(0)

    with pytest.raises(ValueError, match="compressed archive"):
        _apply(perturbation, 1)


@pytest.mark.parametrize("strategy", ["scatter", "replicate"])
def test_noise_with_normalization_respects_the_shard_partition(tmp_path, strategy, monkeypatch):
    # Two shards with distinct tones; under 'scatter' rank r must read only shard r.
    tones = {0: 300, 1: 900}
    tar_paths = []
    for shard, frequency in tones.items():
        tar_path = tmp_path / f"noise_{shard}.tar"
        _write_tar(tar_path, [(f"noise_{shard}.wav", _tone(frequency))])
        tar_paths.append(str(tar_path))
    manifest = _manifest(
        tmp_path / "manifest.json", [{"audio_filepath": f"noise_{shard}.wav", "duration": 1.2} for shard in tones]
    )
    opened = []
    original = perturb._open_indexed_tar
    monkeypatch.setattr(
        perturb, "_open_indexed_tar", lambda cls, path: opened.append(os.path.basename(path)) or original(cls, path)
    )

    def make(rank):
        return NoisePerturbationWithNormalization(
            [manifest],
            audio_tar_filepaths=[tar_paths],
            shuffle_n=1,
            global_rank=rank,
            world_size=2,
            shard_strategy=strategy,
        )

    for rank in [0, 1]:
        opened.clear()
        perturbation = make(rank)
        rng = random.Random(0)
        perturbation.rng = rng
        state = rng.getstate()
        noises = [perturbation.read_one_audiosegment(SAMPLE_RATE).samples for _ in range(12)]
        frequencies = {
            next(f for f in tones.values() if np.allclose(noise, _tone(f).astype(np.float32), atol=1e-4))
            for noise in noises
        }
        if strategy == "scatter":
            assert frequencies == {tones[rank]}
            assert set(opened) == {f"noise_{rank}.tar"}
        else:
            assert frequencies == set(tones.values())

        resumed = make(rank)
        restored = random.Random()
        restored.setstate(state)
        resumed.rng = restored
        for noise in noises:
            np.testing.assert_array_equal(resumed.read_one_audiosegment(SAMPLE_RATE).samples, noise)
