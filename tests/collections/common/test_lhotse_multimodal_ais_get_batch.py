# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Tests for AIStore GetBatch wiring in the multimodal conversation adapters, the
collate helper, and SALMDataset. Actual HTTP retrieval against AIStore is not
exercised — we validate cut metadata and loader-call semantics only.
"""

import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

import lhotse
import pytest
import torch
from lhotse import Recording
from lhotse.dataset import AudioSamples
from lhotse.testing.dummies import dummy_recording

from nemo.collections.common.data.lhotse import text_adapters as text_adapters_module
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    NeMoMultimodalConversation,
    NeMoMultimodalConversationJsonlAdapter,
    NeMoMultimodalConversationShareGPTJsonlAdapter,
    NeMoMultimodalConversationTarWriter,
    TextTurn,
    collate_conversation_audio_fault_tolerant,
    collate_conversation_audio_packed_fault_tolerant,
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
            skip_missing_manifest_entries=False,
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


@pytest.mark.unit
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_jsonl_indexed_missing_tar_member_obeys_only_audio_policy(
    tarred_jsonl_manifest, monkeypatch, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest, tar = tarred_jsonl_manifest

    def fail_get(self, name):
        raise KeyError(f"missing {name}")

    monkeypatch.setattr(
        "nemo.collections.common.data.lhotse.indexed_adapters.IndexedTarMemberReader.get",
        fail_get,
    )
    adapter = NeMoMultimodalConversationJsonlAdapter(
        manifest_filepath=manifest,
        tarred_audio_filepaths=tar,
        audio_locator_tag="[audio]",
        indexed=True,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if fault_tolerant_audio_loading:
        assert list(adapter) == []
    else:
        with pytest.raises(RuntimeError, match="Failed to load multimodal audio member"):
            next(iter(adapter))


@pytest.mark.unit
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_jsonl_streaming_missing_tar_member_obeys_only_audio_policy(
    tarred_jsonl_manifest, monkeypatch, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest, tar = tarred_jsonl_manifest
    monkeypatch.delenv("USE_AIS_GET_BATCH", raising=False)
    monkeypatch.setattr(
        "nemo.collections.common.data.lhotse.text_adapters.TarIterator",
        lambda path: iter(()),
    )
    adapter = NeMoMultimodalConversationJsonlAdapter(
        manifest_filepath=manifest,
        tarred_audio_filepaths=tar,
        audio_locator_tag="[audio]",
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if fault_tolerant_audio_loading:
        assert list(adapter) == []
    else:
        with pytest.raises(RuntimeError, match="Failed to load multimodal tar shard"):
            next(iter(adapter))


# ---------------------------------------------------------------------------
# NeMoMultimodalConversationShareGPTJsonlAdapter — GetBatch mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_sequential_pairing_skips_only_tar_members_absent_from_jsonl(
    skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    expected_recording = object()
    tar = iter([(object(), Path("extra.wav")), (expected_recording, Path("expected.wav"))])

    if skip_missing_manifest_entries:
        recording, path = text_adapters_module._next_matching_paired_audio(
            tar,
            "expected.wav",
            manifest_path="manifest.jsonl",
            tar_path="audio.tar",
            skip_missing_manifest_entries=True,
        )
        assert recording is expected_recording
        assert path == Path("expected.wav")
    else:
        with pytest.raises(ValueError, match="no corresponding JSONL entry"):
            text_adapters_module._next_matching_paired_audio(
                tar,
                "expected.wav",
                manifest_path="manifest.jsonl",
                tar_path="audio.tar",
                skip_missing_manifest_entries=False,
            )


@pytest.mark.unit
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_trailing_tar_member_policy_is_independent_of_audio_policy(
    skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    kwargs = {
        "manifest_path": "manifest.jsonl",
        "tar_path": "audio.tar",
        "skip_missing_manifest_entries": skip_missing_manifest_entries,
        "fault_tolerant_audio_loading": fault_tolerant_audio_loading,
    }
    tar = iter([(object(), Path("trailing.wav"))])
    if skip_missing_manifest_entries:
        assert text_adapters_module._validate_no_trailing_paired_audio(tar, **kwargs) is None
    else:
        with pytest.raises(ValueError, match="no corresponding JSONL entry"):
            text_adapters_module._validate_no_trailing_paired_audio(tar, **kwargs)


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


@pytest.mark.unit
def test_collate_packed_audio_avoids_padded_loader_and_preserves_samples(tmp_path):
    conversations = []
    expected_rows = []
    for idx, duration in enumerate((0.25, 0.5, 0.125)):
        path = tmp_path / f"packed-{idx}.wav"
        dummy_recording(idx, duration, with_data=True).to_cut().save_audio(path)
        conversation = _build_conversation(f"packed-{idx}", path)
        conversations.append(conversation)
        samples = torch.from_numpy(conversation.list_cuts()[0].load_audio()).mean(dim=0)
        expected_rows.append(samples)

    extra_path = tmp_path / "packed-extra.wav"
    dummy_recording(10, 0.375, with_data=True).to_cut().save_audio(extra_path)
    extra_cut = Recording.from_file(extra_path).to_cut().with_id("packed-extra")
    conversations[1].turns.append(AudioTurn(cut=extra_cut, role="user", audio_locator_tag="[audio]"))
    expected_rows.insert(2, torch.from_numpy(extra_cut.load_audio()).mean(dim=0))

    class NoPaddedCollationAudioSamples(AudioSamples):
        def __call__(self, cuts, recording_field=None):
            raise AssertionError("packed collation must not call AudioSamples.__call__")

    packed, cu_seqlens, audio_lens, ok = collate_conversation_audio_packed_fault_tolerant(
        conversations,
        NoPaddedCollationAudioSamples(fault_tolerant=True, mono_downmix=True),
    )

    expected_lens = torch.tensor([row.numel() for row in expected_rows], dtype=torch.long)
    assert [conversation.id for conversation in ok] == [conversation.id for conversation in conversations]
    assert torch.equal(audio_lens, expected_lens)
    assert torch.equal(cu_seqlens, torch.tensor([0, *expected_lens.cumsum(0).tolist()], dtype=torch.long))
    assert torch.equal(packed, torch.cat(expected_rows))
    assert packed.numel() == int(audio_lens.sum())
    assert packed.numel() < len(expected_rows) * int(audio_lens.max())


@pytest.mark.unit
def test_collate_packed_audio_preserves_text_only_batch():
    text_only = NeMoMultimodalConversation(
        id="text_only",
        turns=[TextTurn(role="user", value="hi"), TextTurn(role="assistant", value="hello")],
    )

    packed, cu_seqlens, audio_lens, ok = collate_conversation_audio_packed_fault_tolerant(
        [text_only], AudioSamples(fault_tolerant=True, mono_downmix=True)
    )

    assert packed.dtype == torch.float32
    assert packed.numel() == 0
    assert torch.equal(cu_seqlens, torch.tensor([0], dtype=torch.long))
    assert audio_lens.numel() == 0
    assert list(ok)[0] is text_only


@pytest.mark.unit
def test_collate_packed_audio_keeps_text_only_rows_when_all_audio_rows_fail(local_convs):
    conversations, paths = local_convs
    for conversation, path in zip(conversations, paths):
        conversation.turns[0].cut.recording.sources[0].source = str(path) + ".missing"
    text_only = NeMoMultimodalConversation(
        id="text_only",
        turns=[TextTurn(role="user", value="hi"), TextTurn(role="assistant", value="hello")],
    )

    packed, cu_seqlens, audio_lens, ok = collate_conversation_audio_packed_fault_tolerant(
        [text_only, *conversations], AudioSamples(fault_tolerant=True, mono_downmix=True)
    )

    assert packed.numel() == 0
    assert torch.equal(cu_seqlens, torch.tensor([0], dtype=torch.long))
    assert audio_lens.numel() == 0
    assert list(ok) == [text_only]


@pytest.mark.unit
def test_collate_packed_audio_preserves_fault_tolerance_and_ais_batch_call(local_convs):
    conversations, paths = local_convs
    broken_cut = conversations[1].turns[0].cut
    broken_cut.recording.sources[0].source = str(paths[1]) + ".missing"
    loader = AudioSamples(fault_tolerant=True, mono_downmix=True)
    loader.use_batch_loader = True
    loader.ais_batch_loader = Mock(side_effect=lambda cuts: cuts)

    packed, cu_seqlens, audio_lens, ok = collate_conversation_audio_packed_fault_tolerant(conversations, loader)

    loader.ais_batch_loader.assert_called_once()
    assert [conversation.id for conversation in ok] == ["conv_c0", "conv_c2"]
    assert packed.numel() == int(audio_lens.sum())
    assert torch.equal(cu_seqlens[1:] - cu_seqlens[:-1], audio_lens)


# ---------------------------------------------------------------------------
# SALMDataset wiring
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub exposing what ``get_pad_id`` needs."""

    pad = 0
    unk_id = 0


@pytest.mark.unit
def test_salm_dataset_batch_loader_enabled(monkeypatch):
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    from nemo.collections.speechlm2.data.salm_dataset import SALMDataset

    with patch("nemo.collections.speechlm2.data.salm_dataset.AudioSamples") as audio_samples:
        ds = SALMDataset(tokenizer=_FakeTokenizer())

    audio_samples.assert_called_once_with(
        fault_tolerant=True, use_batch_loader=True, ais_force_individual=False, mono_downmix=True
    )
    assert ds.load_audio is audio_samples.return_value


@pytest.mark.unit
def test_salm_dataset_batch_loader_disabled(monkeypatch):
    monkeypatch.delenv("USE_AIS_GET_BATCH", raising=False)
    from nemo.collections.speechlm2.data.salm_dataset import SALMDataset

    with patch("nemo.collections.speechlm2.data.salm_dataset.AudioSamples") as audio_samples:
        ds = SALMDataset(tokenizer=_FakeTokenizer())

    audio_samples.assert_called_once_with(
        fault_tolerant=True, use_batch_loader=False, ais_force_individual=False, mono_downmix=True
    )
    assert ds.load_audio is audio_samples.return_value
