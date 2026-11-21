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
from io import BytesIO
from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.cut import Cut, PaddingCut
from lhotse.serialization import SequentialJsonlWriter, save_to_jsonl
from lhotse.shar.writers import TarWriter
from lhotse.testing.dummies import DummyManifest, dummy_cut, dummy_supervision
from lhotse.utils import fastcopy
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.sampling import CustomFieldFilter
from nemo.collections.common.data.lhotse.text_adapters import NeMoMultimodalConversation, TextExample, TextTurn

# The "verified" value of each manifest row, in row order; ``_ABSENT`` rows do not have the key.
_ABSENT = object()
_VERIFIED_VALUES = [True, False, _ABSENT, True, False, _ABSENT, 1, 0, False, True]


def _keeps_verified_or_unmarked(cut: Cut) -> bool:
    """The predicate that ``custom_field_filter: {verified: true}`` stands for."""
    return bool(cut.verified) if cut.has_custom("verified") else True


class _Identity:
    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


@pytest.fixture(scope="module")
def manifest_rows(tmp_path_factory) -> list[dict]:
    """Ten one-second utterances as NeMo manifest rows, with ``verified`` true, false or absent."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=len(_VERIFIED_VALUES), with_data=True)
    cuts = cuts.save_audios(tmp_path_factory.mktemp("custom_field_filter_audio"))
    rows = []
    for idx, (cut, verified) in enumerate(zip(cuts, _VERIFIED_VALUES)):
        row = {
            "audio_filepath": cut.recording.sources[0].source,
            "text": f"utt-{idx}",
            "duration": cut.duration,
            "id": f"row-{idx}",
        }
        if verified is not _ABSENT:
            row["verified"] = verified
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def manifest_config(manifest_rows: list[dict], tmp_path_factory) -> dict:
    path = tmp_path_factory.mktemp("custom_field_filter_manifest") / "manifest.json"
    save_to_jsonl(manifest_rows, path)
    return {"manifest_filepath": str(path)}


@pytest.fixture(scope="module")
def tarred_manifest_config(manifest_rows: list[dict], tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("custom_field_filter_tarred")
    with (
        TarWriter(f"{root}/audios_%01d.tar", shard_size=5) as tar_writer,
        SequentialJsonlWriter(root / "manifest.jsonl") as manifest_writer,
    ):
        for idx, row in enumerate(manifest_rows):
            name = Path(row["audio_filepath"]).name
            with open(row["audio_filepath"], "rb") as f:
                tar_writer.write(name, BytesIO(f.read()))
            manifest_writer.write({**row, "audio_filepath": name, "shard_id": idx // 5})
    return {
        "manifest_filepath": str(manifest_writer.path),
        "tarred_audio_filepaths": f"{root}/audios__OP_0..1_CL_.tar",
        "force_finite": True,
    }


def _read_all_cuts(source_config: dict, **options) -> list[Cut]:
    config = {**source_config, "sample_rate": 16000, "batch_size": 3, "seed": 0, "shard_seed": 0, **options}
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
    return [cut for batch in dl for cut in batch]


class TestCustomFieldFilter:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "custom, field_values, expected",
        [
            ({"verified": True}, {"verified": True}, True),
            ({"verified": False}, {"verified": True}, False),
            ({"verified": 1}, {"verified": True}, True),
            ({"verified": 0}, {"verified": True}, False),
            ({"verified": None}, {"verified": True}, False),
            ({}, {"verified": True}, True),
            (None, {"verified": True}, True),
            ({"verified": False}, {"verified": False}, True),
            ({"source": "read"}, {"source": "read"}, True),
            ({"source": "spontaneous"}, {"source": "read"}, False),
            ({"snr_bin": 3}, {"snr_bin": 3}, True),
            ({"snr_bin": 2}, {"snr_bin": 3}, False),
            ({"verified": True, "source": "read"}, {"verified": True, "source": "read"}, True),
            ({"verified": True, "source": "spontaneous"}, {"verified": True, "source": "read"}, False),
            ({"verified": False, "source": "read"}, {"verified": True, "source": "read"}, False),
            ({"source": "read"}, {"verified": True, "source": "read"}, True),
            ({"id": "row-3"}, {"id": "row-3"}, True),
            ({"id": "row-4"}, {"id": "row-3"}, False),
            ({"channel": 1}, {"channel": 1}, True),
            ({"channel": 1}, {"channel": 0}, False),
            ({"verified": False}, OmegaConf.create({"verified": False}), True),
            ({"verified": False}, OmegaConf.create({"verified": True}), False),
            ({"verified": False}, None, True),
            ({"verified": False}, {}, True),
        ],
    )
    def test_keeps_matching_and_unmarked_cuts(self, custom, field_values, expected):
        cut = dummy_cut(0)
        cut.custom = custom
        assert CustomFieldFilter(field_values)(cut) is expected

    @pytest.mark.unit
    def test_looks_at_cut_fields_not_supervision_fields(self):
        supervision = dummy_supervision(0)
        supervision.custom = {"verified": False}
        cut = dummy_cut(0, supervisions=[supervision])
        cut.custom = None
        assert CustomFieldFilter({"verified": True})(cut) is True

    @pytest.mark.unit
    @pytest.mark.parametrize("custom, expected", [({"verified": True}, True), ({"verified": False}, False)])
    def test_reads_the_fields_of_a_padded_cut(self, custom, expected):
        cut = dummy_cut(0)
        cut.custom = custom
        assert CustomFieldFilter({"verified": True})(cut.pad(duration=cut.duration + 1.0)) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "first, second, expected", [(True, False, True), (False, True, False), (None, False, False)]
    )
    def test_mixed_cut_takes_the_first_track_that_has_the_field(self, first, second, expected):
        cuts = [dummy_cut(0), dummy_cut(1)]
        for cut, verified in zip(cuts, (first, second)):
            cut.custom = None if verified is None else {"verified": verified}
        assert CustomFieldFilter({"verified": True})(cuts[0].mix(cuts[1])) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize("second, expected", [(True, True), (None, True), (False, False)])
    def test_mixed_cut_skips_a_muted_track(self, second, expected):
        cuts = [dummy_cut(0), dummy_cut(1)]
        cuts[0].custom = {"verified": False}
        cuts[1].custom = None if second is None else {"verified": second}
        mixed = cuts[0].mix(cuts[1])
        mixed = fastcopy(mixed, tracks=[fastcopy(mixed.tracks[0], mute=True), mixed.tracks[1]])
        assert mixed.has_custom("verified") is (second is not None)
        assert CustomFieldFilter({"verified": True})(mixed) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize("custom, expected", [({"verified": True}, True), ({"verified": False}, False)])
    def test_filters_text_and_conversation_examples_by_their_custom_fields(self, custom, expected):
        custom_field_filter = CustomFieldFilter({"verified": True})
        assert custom_field_filter(TextExample("some text", custom=custom)) is expected
        conversation = NeMoMultimodalConversation(id="conversation", turns=[TextTurn("hi", "user")], custom=custom)
        assert custom_field_filter(conversation) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "example",
        [TextExample("some text"), PaddingCut(id="padding", duration=1.0, sampling_rate=16000, feat_value=0.0)],
    )
    def test_passes_examples_without_custom_fields(self, example):
        assert CustomFieldFilter({"verified": True})(example) is True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field_values",
        [
            {"verified": [True, False]},
            {"verified": None},
            {"verified": {"nested": True}},
            {"verified": float("nan")},
            {"": True},
            {3: True},
            "verified",
            ["verified"],
        ],
    )
    def test_rejects_invalid_field_values(self, field_values):
        with pytest.raises(ValueError, match="custom_field_filter"):
            CustomFieldFilter(field_values)


class TestCustomFieldFilterDataloader:
    @pytest.mark.unit
    @pytest.mark.parametrize("source", ["manifest_config", "tarred_manifest_config"])
    def test_selects_the_cuts_of_the_predicate(self, source, request):
        source_config = request.getfixturevalue(source)
        all_cuts = _read_all_cuts(source_config)
        expected = sorted(cut.supervisions[0].text for cut in all_cuts if _keeps_verified_or_unmarked(cut))

        cuts = _read_all_cuts(source_config, custom_field_filter={"verified": True})

        assert sorted(cut.supervisions[0].text for cut in cuts) == expected
        assert expected == ["utt-0", "utt-2", "utt-3", "utt-5", "utt-6", "utt-9"]

    @pytest.mark.unit
    @pytest.mark.parametrize("source", ["manifest_config", "tarred_manifest_config"])
    @pytest.mark.parametrize("options", [{}, {"custom_field_filter": None}, {"custom_field_filter": {}}])
    def test_keeps_all_cuts_by_default(self, source, options, request):
        cuts = _read_all_cuts(request.getfixturevalue(source), **options)
        assert sorted(cut.supervisions[0].text for cut in cuts) == sorted(f"utt-{idx}" for idx in range(10))

    @pytest.mark.unit
    def test_other_value_selects_the_complement_of_marked_cuts(self, manifest_config):
        cuts = _read_all_cuts(manifest_config, custom_field_filter={"verified": False})
        texts = sorted(cut.supervisions[0].text for cut in cuts)
        assert texts == ["utt-1", "utt-2", "utt-4", "utt-5", "utt-7", "utt-8"]

    @pytest.mark.unit
    def test_field_named_like_a_cut_attribute(self, manifest_config):
        cuts = _read_all_cuts(manifest_config, custom_field_filter={"id": "row-3"})
        assert [cut.supervisions[0].text for cut in cuts] == ["utt-3"]

    @pytest.mark.unit
    def test_multi_config_takes_the_option_per_sub_config(self, manifest_config):
        sub_config = {**manifest_config, "sample_rate": 16000, "batch_size": 3}
        config = {
            "multi_config": True,
            "seed": 0,
            "shard_seed": 0,
            "shuffle": False,
            "filtered": {**sub_config, "custom_field_filter": {"verified": True}},
            "unfiltered": sub_config,
        }
        dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
        texts = sorted(cut.supervisions[0].text for batch in dl for cut in batch)
        expected = [f"utt-{idx}" for idx in range(10)] + ["utt-0", "utt-2", "utt-3", "utt-5", "utt-6", "utt-9"]
        assert texts == sorted(expected)

        for top_level_value in ({"verified": True}, {}):
            config = {**config, "custom_field_filter": top_level_value}
            with pytest.raises(ValueError, match="custom_field_filter.*sub-config"):
                get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())

    @pytest.mark.unit
    def test_input_cfg_tags_are_custom_fields(self, manifest_config):
        input_cfg = [
            {"type": "nemo", **manifest_config, "tags": {"origin": "first"}},
            {"type": "nemo", **manifest_config, "tags": {"origin": "second"}},
        ]
        source_config = {"input_cfg": input_cfg, "force_finite": True}
        cuts = _read_all_cuts(source_config, custom_field_filter={"verified": True})
        assert len(cuts) == 12

        cuts = _read_all_cuts(source_config, custom_field_filter={"verified": True, "origin": "second"})
        texts = sorted(cut.supervisions[0].text for cut in cuts)
        assert texts == ["utt-0", "utt-2", "utt-3", "utt-5", "utt-6", "utt-9"]
        assert all(cut.origin == "second" for cut in cuts)

    @pytest.mark.unit
    def test_rejected_cuts_are_not_tokenized(self, manifest_config):
        tokenized = []

        def tokenizer(text: str) -> list[int]:
            tokenized.append(text)
            return [1]

        config = {**manifest_config, "sample_rate": 16000, "batch_size": 3, "custom_field_filter": {"verified": True}}
        dl = get_lhotse_dataloader_from_config(
            config=config, global_rank=0, world_size=1, dataset=_Identity(), tokenizer=tokenizer
        )
        assert len([cut for batch in dl for cut in batch]) == 6
        assert sorted(tokenized) == ["utt-0", "utt-2", "utt-3", "utt-5", "utt-6", "utt-9"]

    @pytest.mark.unit
    def test_invalid_value_raises_at_construction(self, manifest_config):
        config = {**manifest_config, "sample_rate": 16000, "batch_size": 3}
        config["custom_field_filter"] = {"verified": [True, False]}
        with pytest.raises(ValueError, match="custom_field_filter.*'verified'"):
            get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
