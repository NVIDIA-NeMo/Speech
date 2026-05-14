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

"""
Tests for AIStore GetBatch wiring in the multimodal conversation adapters, the
collate helper, and SALMDataset. Actual HTTP retrieval against AIStore is not
exercised — we validate cut metadata and loader-call semantics only.
"""

import tarfile
from pathlib import Path

import lhotse
import pytest
from lhotse import Recording
from lhotse.dataset import AudioSamples
from lhotse.testing.dummies import dummy_recording

from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    NeMoMultimodalConversation,
    NeMoMultimodalConversationJsonlAdapter,
    NeMoMultimodalConversationShareGPTJsonlAdapter,
    NeMoMultimodalConversationTarWriter,
    TextTurn,
    collate_conversation_audio_fault_tolerant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def jsonl_manifest_path(tmp_path_factory):
    """A plain (non-tarred) JSONL manifest with two audio turns in a single conversation."""
    tmp_path = tmp_path_factory.mktemp("multi_convo_ais_src")
    manifest_path = tmp_path / "manifest.json"
    data = [
        {
            "id": "convo_a",
            "conversations": [
                {"value": "hello", "from": "User", "type": "text"},
                {"value": "a.wav", "from": "User", "type": "audio", "duration": 1.0},
                {"value": "b.wav", "from": "Assistant", "type": "audio", "duration": 2.0, "offset": 0.25},
            ],
        }
    ]
    lhotse.serialization.save_to_jsonl(data, manifest_path)
    dummy_recording(0, 1.0, with_data=True).to_cut().save_audio(tmp_path / "a.wav")
    dummy_recording(1, 3.0, with_data=True).to_cut().save_audio(tmp_path / "b.wav")
    return manifest_path


@pytest.fixture(scope="session")
def tarred_jsonl_manifest(jsonl_manifest_path, tmp_path_factory):
    """Sharded tarred JSONL built via NeMoMultimodalConversationTarWriter (2 shards of 5)."""
    (conversation,) = list(NeMoMultimodalConversationJsonlAdapter(jsonl_manifest_path, "[audio]"))
    tar_dir = tmp_path_factory.mktemp("multi_convo_ais_tar")
    with NeMoMultimodalConversationTarWriter(tar_dir, shard_size=5) as writer:
        for i in range(10):
            conversation.id = f'convo-{i}'
            writer.write(conversation)
    return str(tar_dir / "manifest_{0..1}.jsonl"), str(tar_dir / "audio_{0..1}.tar")


@pytest.fixture(scope="session")
def tarred_sharegpt_manifest(tmp_path_factory):
    """Build a ShareGPT manifest + matching tar file so the adapter can iterate both modes."""
    tmp_path = tmp_path_factory.mktemp("sharegpt_ais")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # 4 rows, one audio each.
    audio_files = []
    manifest = []
    for i in range(4):
        audio_name = f"sound_{i}.wav"
        dummy_recording(i, 1.0, with_data=True).to_cut().save_audio(audio_dir / audio_name)
        audio_files.append((audio_name, audio_dir / audio_name))
        manifest.append(
            {
                "id": f"sg_{i}",
                "sound": audio_name,
                "conversations": [
                    {"from": "human", "value": f"Describe this <sound>", "duration": 1.0},
                    {"from": "gpt", "value": "It sounds like a test."},
                ],
            }
        )
    manifest_path = tmp_path / "manifest_0.jsonl"
    lhotse.serialization.save_to_jsonl(manifest, manifest_path)
    tar_path = tmp_path / "audio_0.tar"
    with tarfile.open(tar_path, "w") as tar:
        for arcname, src in audio_files:
            tar.add(src, arcname=arcname)
    return str(manifest_path), str(tar_path)


@pytest.fixture(scope="session")
def skipme_manifest(tmp_path_factory):
    """JSONL + tar where half the rows carry custom._skipme=True."""
    tmp_path = tmp_path_factory.mktemp("multi_convo_skipme_src")
    manifest_path = tmp_path / "manifest.json"
    data = [
        {
            "id": "x",
            "conversations": [{"value": "x.wav", "from": "User", "type": "audio", "duration": 1.0}],
        }
    ]
    lhotse.serialization.save_to_jsonl(data, manifest_path)
    dummy_recording(0, 1.0, with_data=True).to_cut().save_audio(tmp_path / "x.wav")

    (conv,) = list(NeMoMultimodalConversationJsonlAdapter(manifest_path, "[audio]"))
    tar_dir = tmp_path_factory.mktemp("multi_convo_skipme_tar")
    with NeMoMultimodalConversationTarWriter(tar_dir, shard_size=4) as writer:
        for i in range(4):
            conv.id = f"convo-{i}"
            conv.custom = {"_skipme": i % 2 == 1}
            writer.write(conv)
    return str(tar_dir / "manifest_0.jsonl"), str(tar_dir / "audio_0.tar")


# ---------------------------------------------------------------------------
# NeMoMultimodalConversationJsonlAdapter — GetBatch mode
# ---------------------------------------------------------------------------


def _iter(adapter):
    return list(adapter)


def _audio_turns(conversation):
    return [t for t in conversation.turns if isinstance(t, AudioTurn)]


@pytest.mark.unit
def test_jsonl_batch_creates_url_sources(tarred_jsonl_manifest, monkeypatch):
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    manifest, tar = tarred_jsonl_manifest
    adapter = NeMoMultimodalConversationJsonlAdapter(
        manifest_filepath=manifest,
        tarred_audio_filepaths=tar,
        audio_locator_tag="[audio]",
    )
    conversations = _iter(adapter)
    assert len(conversations) == 10
    tar_shard_paths = [p for p in Path(tar).parent.glob("audio_*.tar")]
    tar_shard_names = {p.name for p in tar_shard_paths}
    for conv in conversations:
        for turn in _audio_turns(conv):
            rec = turn.cut.recording
            assert len(rec.sources) == 1
            src = rec.sources[0]
            assert src.type == "url"
            # URL is {tar_shard}/{audio_filename}.
            parent, _, filename = src.source.rpartition("/")
            assert Path(parent).name in tar_shard_names, src.source
            assert filename.endswith(".flac"), src.source


@pytest.mark.unit
def test_jsonl_batch_offset_affects_id(tarred_jsonl_manifest, monkeypatch):
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    manifest, tar = tarred_jsonl_manifest
    conversations = _iter(
        NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=manifest,
            tarred_audio_filepaths=tar,
            audio_locator_tag="[audio]",
        )
    )
    for conv in conversations:
        audio_turns = _audio_turns(conv)
        # First audio turn in the fixture has offset=0 -> plain stem id.
        assert "_" not in audio_turns[0].cut.id.split(".")[0]
        # Second audio turn has offset=0.25, duration=2.0 -> stem_{offset:.3f}_{duration:.3f}.
        assert audio_turns[1].cut.id.endswith("_0.250_2.000")


@pytest.mark.unit
def test_jsonl_batch_skipme(skipme_manifest, monkeypatch):
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    manifest, tar = skipme_manifest
    conversations = _iter(
        NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=manifest,
            tarred_audio_filepaths=tar,
            audio_locator_tag="[audio]",
        )
    )
    # 4 rows total, half marked _skipme -> 2 yielded.
    assert len(conversations) == 2
    assert all(c.id in {"convo-0", "convo-2"} for c in conversations)


@pytest.mark.unit
def test_jsonl_batch_system_prompt_and_context(tarred_jsonl_manifest, monkeypatch):
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    manifest, tar = tarred_jsonl_manifest
    conv = _iter(
        NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=manifest,
            tarred_audio_filepaths=tar,
            audio_locator_tag="[audio]",
            system_prompt="SYS",
            context="please answer:",
        )
    )[0]
    assert isinstance(conv.turns[0], TextTurn)
    assert conv.turns[0].role == "system"
    assert conv.turns[0].value == "SYS"
    # Original first turn is a user TextTurn "hello" (not an AudioTurn), so the context
    # prefix does not kick in — verify system prompt is present and original turns follow.
    assert conv.turns[1].role == "user"
    assert isinstance(conv.turns[1], TextTurn)
    assert conv.turns[1].value == "hello"


@pytest.mark.unit
def test_jsonl_batch_vs_tar_parity(tarred_jsonl_manifest, monkeypatch):
    manifest, tar = tarred_jsonl_manifest
    monkeypatch.delenv("USE_AIS_GET_BATCH", raising=False)
    tar_mode = _iter(
        NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=manifest,
            tarred_audio_filepaths=tar,
            audio_locator_tag="[audio]",
        )
    )
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    ais_mode = _iter(
        NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=manifest,
            tarred_audio_filepaths=tar,
            audio_locator_tag="[audio]",
        )
    )
    assert [c.id for c in tar_mode] == [c.id for c in ais_mode]
    assert [len(c.turns) for c in tar_mode] == [len(c.turns) for c in ais_mode]
    # Cut ids match too (both modes route through _make_cut_id).
    for a, b in zip(tar_mode, ais_mode):
        assert [t.cut.id for t in _audio_turns(a)] == [t.cut.id for t in _audio_turns(b)]


# ---------------------------------------------------------------------------
# NeMoMultimodalConversationShareGPTJsonlAdapter — GetBatch mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sharegpt_batch_creates_url_sources(tarred_sharegpt_manifest, monkeypatch):
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    manifest, tar = tarred_sharegpt_manifest
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=manifest,
        tarred_audio_filepaths=tar,
        audio_locator_tag="[audio]",
    )
    conversations = list(adapter)
    assert len(conversations) == 4
    for conv in conversations:
        for turn in _audio_turns(conv):
            src = turn.cut.recording.sources[0]
            assert src.type == "url"
            assert src.source.startswith(str(tar))
            assert src.source.endswith(".wav")


@pytest.mark.unit
def test_sharegpt_batch_slice_length(tarred_sharegpt_manifest, monkeypatch):
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    manifest, tar = tarred_sharegpt_manifest
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=manifest,
        tarred_audio_filepaths=tar,
        audio_locator_tag="[audio]",
        slice_length=2,
        shard_seed=0,
    )
    conversations = list(adapter)
    assert len(conversations) == 2


# ---------------------------------------------------------------------------
# collate_conversation_audio_fault_tolerant
# ---------------------------------------------------------------------------


def _build_conversation(cut_id: str, audio_path: Path) -> NeMoMultimodalConversation:
    cut = Recording.from_file(audio_path).to_cut().with_id(cut_id)
    turn = AudioTurn(cut=cut, role="user", audio_locator_tag="[audio]")
    return NeMoMultimodalConversation(id=f"conv_{cut_id}", turns=[turn])


@pytest.fixture
def local_convs(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"c{i}.wav"
        dummy_recording(i, 1.0, with_data=True).to_cut().save_audio(p)
        paths.append(p)
    return [_build_conversation(f"c{i}", p) for i, p in enumerate(paths)], paths


@pytest.mark.unit
def test_collate_all_succeed(local_convs):
    convs, _ = local_convs
    loader = AudioSamples(fault_tolerant=True)
    audios, audio_lens, ok = collate_conversation_audio_fault_tolerant(convs, loader)
    assert len(ok) == 3
    assert audios.ndim == 2
    assert audios.shape[0] == 3
    assert audio_lens.shape == (3,)


@pytest.mark.unit
def test_collate_fault_tolerance_drops_conversation(local_convs):
    convs, paths = local_convs
    # Sabotage conv[1] by pointing its cut at a missing file.
    broken_cut = convs[1].turns[0].cut
    broken_cut.recording.sources[0].source = str(paths[1]) + ".missing"
    loader = AudioSamples(fault_tolerant=True)
    audios, audio_lens, ok = collate_conversation_audio_fault_tolerant(convs, loader)
    kept_ids = [c.id for c in ok]
    assert "conv_c0" in kept_ids
    assert "conv_c2" in kept_ids
    assert "conv_c1" not in kept_ids
    assert audios.shape[0] == 2


@pytest.mark.unit
def test_collate_empty_audio_conversations():
    text_only = NeMoMultimodalConversation(
        id="text_only",
        turns=[TextTurn(role="user", value="hi"), TextTurn(role="assistant", value="hello")],
    )
    loader = AudioSamples(fault_tolerant=True)
    audios, audio_lens, ok = collate_conversation_audio_fault_tolerant([text_only], loader)
    assert audios.numel() == 0
    assert audio_lens.numel() == 0
    assert list(ok)[0] is text_only


# ---------------------------------------------------------------------------
# SALMDataset wiring
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub exposing what ``get_pad_id`` needs."""

    pad = 0
    unk_id = 0


@pytest.mark.unit
def test_salm_dataset_batch_loader_enabled(monkeypatch):
    pytest.importorskip("aistore")  # AISBatchLoader requires the aistore client.
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    from nemo.collections.speechlm2.data.salm_dataset import SALMDataset

    ds = SALMDataset(tokenizer=_FakeTokenizer())
    assert isinstance(ds.load_audio, AudioSamples)
    assert ds.load_audio.use_batch_loader is True


@pytest.mark.unit
def test_salm_dataset_batch_loader_disabled(monkeypatch):
    monkeypatch.delenv("USE_AIS_GET_BATCH", raising=False)
    from nemo.collections.speechlm2.data.salm_dataset import SALMDataset

    ds = SALMDataset(tokenizer=_FakeTokenizer())
    assert isinstance(ds.load_audio, AudioSamples)
    assert ds.load_audio.use_batch_loader is False
