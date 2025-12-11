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

"""Tests for GetBatch logic in LazyNeMoTarredIterator with AIS enabled."""

import tarfile
from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.serialization import load_jsonl, save_to_jsonl
from lhotse.testing.dummies import DummyManifest

from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator


@pytest.fixture
def nemo_tarred_manifest_path_for_slicing(tmp_path_factory):
    """Create a tarred NeMo manifest with 20 utterances (2 shards of 10 each) for slice testing."""
    tmpdir = tmp_path_factory.mktemp("nemo_tarred_slice_data")

    # Create dummy audio files
    cuts = DummyManifest(CutSet, begin_id=0, end_id=20, with_data=True).save_audios(tmpdir, progress_bar=False)

    # Create two tar files (shard 0 and shard 1)
    for shard_id in [0, 1]:
        tar_path = tmpdir / f"audio_{shard_id}.tar"
        manifest_path = tmpdir / f"manifest_{shard_id}.json"

        start_idx = shard_id * 10
        end_idx = start_idx + 10
        shard_cuts = list(cuts)[start_idx:end_idx]

        # Create tar file
        with tarfile.open(tar_path, "w") as tar:
            for idx, c in enumerate(shard_cuts):
                audio_file = c.recording.sources[0].source
                tar.add(audio_file, arcname=f"audio_{start_idx + idx}.wav")

        # Create manifest
        manifest = []
        for idx, c in enumerate(shard_cuts):
            manifest.append(
                {
                    "audio_filepath": f"audio_{start_idx + idx}.wav",
                    "text": f"utterance {start_idx + idx}",
                    "duration": c.duration,
                    "lang": "en",
                    "shard_id": shard_id,
                }
            )

        save_to_jsonl(manifest, manifest_path)

    # Return paths using NeMo's shard notation
    manifest_paths = f"{tmpdir}/manifest__OP_0..1_CL_.json"
    tar_paths = f"{tmpdir}/audio__OP_0..1_CL_.tar"

    return str(manifest_paths), str(tar_paths)


@pytest.mark.unit
def test_batch_reading_with_slice_offset(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that slice_length and slice_offset work correctly for batch reading mode."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    manifest_path, tar_path = nemo_tarred_manifest_path_for_slicing

    # Test with slice_length=5 (should get 5 entries per shard = 10 total)
    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=5,
        shard_seed=42,
    )

    cuts = list(iterator)

    # Should have exactly 5 cuts per shard = 10 cuts total
    assert len(cuts) == 10

    # Verify cuts have valid metadata
    for cut in cuts:
        assert cut.has_recording
        assert cut.supervisions[0].text.startswith("utterance")
        assert cut.duration == 1.0


@pytest.mark.unit
def test_batch_reading_slice_offset_randomness(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that slice_offset varies across epochs for batch reading."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    manifest_path, tar_path = nemo_tarred_manifest_path_for_slicing

    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=5,
        shard_seed=42,
    )

    # Collect IDs from multiple epochs
    epoch_ids_list = []
    for _ in range(10):
        epoch_ids_list.append(tuple([cut.id for cut in iterator]))

    # Verify at least some epochs differ (randomness is working)
    unique_epochs = len(set(epoch_ids_list))
    assert (
        unique_epochs > 1
    ), f"Expected multiple unique epochs, got {unique_epochs}. All epochs identical - randomness not working."


@pytest.mark.unit
def test_batch_reading_without_slice_length(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that batch reading without slice_length returns all entries."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    manifest_path, tar_path = nemo_tarred_manifest_path_for_slicing

    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=None,  # No slicing
        shard_seed=42,
    )

    cuts = list(iterator)

    # Should have all 20 cuts (10 per shard)
    assert len(cuts) == 20


@pytest.mark.unit
def test_batch_reading_with_skipme(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that batch reading correctly skips entries with _skipme=True."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    tmpdir = Path(nemo_tarred_manifest_path_for_slicing[0]).parent

    # Modify manifest to add _skipme to some entries
    for shard_id in [0, 1]:
        manifest_path = tmpdir / f"manifest_{shard_id}.json"

        items = list(load_jsonl(manifest_path))
        # Mark every other item as skipme
        for idx, item in enumerate(items):
            if idx % 2 == 0:
                item['_skipme'] = True

        save_to_jsonl(items, manifest_path)

    manifest_paths = f"{tmpdir}/manifest__OP_0..1_CL_.json"
    tar_paths = f"{tmpdir}/audio__OP_0..1_CL_.tar"

    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_paths,
        tar_paths=tar_paths,
        shuffle_shards=False,
        slice_length=None,
        shard_seed=42,
    )

    cuts = list(iterator)

    # Should have 10 cuts (half were skipped)
    assert len(cuts) == 10

    # Verify none of the cuts have _skipme in their custom fields
    for cut in cuts:
        assert not cut.custom.get('_skipme', False)


@pytest.mark.unit
def test_batch_reading_slice_length_larger_than_manifest(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that slice_length larger than manifest size returns all entries."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    manifest_path, tar_path = nemo_tarred_manifest_path_for_slicing

    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=100,  # Much larger than 10 entries per shard
        shard_seed=42,
    )

    cuts = list(iterator)

    # Should still return all 20 cuts
    assert len(cuts) == 20


@pytest.mark.unit
def test_batch_reading_slice_offset_respects_entries_processed(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that slice_offset correctly counts all entries including skipped ones."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    tmpdir = Path(nemo_tarred_manifest_path_for_slicing[0]).parent

    # Modify manifest to add _skipme to first 3 entries of each shard
    for shard_id in [0, 1]:
        manifest_path = tmpdir / f"manifest_{shard_id}.json"

        items = list(load_jsonl(manifest_path))
        for idx in range(3):
            items[idx]['_skipme'] = True

        save_to_jsonl(items, manifest_path)

    manifest_paths = f"{tmpdir}/manifest__OP_0..1_CL_.json"
    tar_paths = f"{tmpdir}/audio__OP_0..1_CL_.tar"

    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_paths,
        tar_paths=tar_paths,
        shuffle_shards=False,
        slice_length=5,
        shard_seed=42,
    )

    cuts = list(iterator)

    # Should have at most 10 cuts (5 per shard), but could be less if slice_offset skips valid entries
    assert len(cuts) <= 10

    # All returned cuts should not have _skipme
    for cut in cuts:
        assert not cut.custom.get('_skipme', False)


@pytest.mark.unit
def test_batch_reading_creates_url_sources(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that batch mode creates URL-based recording sources."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    manifest_path, tar_path = nemo_tarred_manifest_path_for_slicing

    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=5,
        shard_seed=42,
    )

    cuts = list(iterator)

    # Verify recording sources are URLs
    for cut in cuts:
        source = cut.recording.sources[0]
        assert source.type == "url"
        assert ".tar/" in source.source
        assert source.source.endswith(".wav")


@pytest.mark.unit
def test_batch_reading_url_format(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that URL format is tar_path/audio_filename."""
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    manifest_path, tar_path = nemo_tarred_manifest_path_for_slicing

    iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=None,
        shard_seed=42,
    )

    cuts = list(iterator)
    tmpdir = Path(manifest_path.split("__OP_")[0]).parent

    # Verify URL structure
    for cut in cuts:
        url = cut.recording.sources[0].source
        assert str(tmpdir) in url
        assert "audio_" in url

        # Verify exactly one .tar in URL
        parts = url.split("/")
        tar_parts = [p for p in parts if p.endswith(".tar")]
        assert len(tar_parts) == 1


@pytest.mark.unit
def test_batch_vs_sequential_mode(nemo_tarred_manifest_path_for_slicing, monkeypatch):
    """Test that batch and sequential modes produce different source types."""
    manifest_path, tar_path = nemo_tarred_manifest_path_for_slicing

    # Batch mode
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    batch_iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=None,
        shard_seed=42,
    )
    batch_cuts = list(batch_iterator)

    # Sequential mode
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")
    seq_iterator = LazyNeMoTarredIterator(
        manifest_path=manifest_path,
        tar_paths=tar_path,
        shuffle_shards=False,
        slice_length=None,
        shard_seed=42,
    )
    seq_cuts = list(seq_iterator)

    # Batch mode should use URL sources
    assert batch_cuts[0].recording.sources[0].type == "url"

    # Sequential mode should use different approach
    assert (
        seq_cuts[0].recording.sources[0].type != "url"
        or seq_cuts[0].recording.sources[0].source != batch_cuts[0].recording.sources[0].source
    )
