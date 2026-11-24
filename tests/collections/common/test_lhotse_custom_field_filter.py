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
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
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
_SNR_VALUES = [0.0, 2.0, 4.0, 6.0, _ABSENT, 10.0, 12, None, 16.0, 18.0]
_QUALITY_VALUES = ["high", "low", "medium", "high", _ABSENT, "low", "very_high", "medium", "low", "high"]


def _keeps_verified_or_unmarked(cut: Cut) -> bool:
    """The predicate that ``custom_field_filter: {verified: true}`` stands for."""
    return bool(cut.verified) if cut.has_custom("verified") else True


class _Identity:
    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


@pytest.fixture(scope="module")
def manifest_rows(tmp_path_factory) -> list[dict]:
    """Ten one-second utterances as NeMo manifest rows; ``verified``, ``snr`` and ``quality`` are absent in some."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=len(_VERIFIED_VALUES), with_data=True)
    cuts = cuts.save_audios(tmp_path_factory.mktemp("custom_field_filter_audio"))
    rows = []
    for idx, cut in enumerate(cuts):
        row = {
            "audio_filepath": cut.recording.sources[0].source,
            "text": f"utt-{idx}",
            "duration": cut.duration,
            "id": f"row-{idx}",
        }
        for name, values in (("verified", _VERIFIED_VALUES), ("snr", _SNR_VALUES), ("quality", _QUALITY_VALUES)):
            if values[idx] is not _ABSENT:
                row[name] = values[idx]
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
        "custom, conditions, expected",
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
    def test_keeps_matching_and_unmarked_cuts(self, custom, conditions, expected):
        cut = dummy_cut(0)
        cut.custom = custom
        assert CustomFieldFilter(conditions)(cut) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value, condition, expected",
        [
            (True, {"eq": True}, True),
            (False, {"eq": True}, False),
            ("read", {"neq": "read"}, False),
            ("spontaneous", {"neq": "read"}, True),
            (None, {"neq": "read"}, True),
            (5.0, {"gt": 5.0}, False),
            (5.5, {"gt": 5.0}, True),
            (5.0, {"ge": 5.0}, True),
            (4.5, {"ge": 5.0}, False),
            (5.0, {"lt": 5.0}, False),
            (4.5, {"lt": 5.0}, True),
            (5.0, {"le": 5.0}, True),
            (5.5, {"le": 5.0}, False),
            (7, {"gt": 5.0, "le": 10.5}, True),
            (10.5, {"gt": 5.0, "le": 10.5}, True),
            (5.0, {"gt": 5.0, "le": 10.5}, False),
            (11, {"gt": 5.0, "le": 10.5}, False),
            (7, {"gt": 5, "neq": 7}, False),
            ("high", {"in": ["very_high", "high", "medium"]}, True),
            ("low", {"in": ["very_high", "high", "medium"]}, False),
            (3, {"in": [1, 3]}, True),
            (2, {"in": [1, 3]}, False),
            (None, {"in": ["high"]}, False),
            (["high"], {"in": ["high"]}, False),
            (None, {"gt": 5.0}, False),
            ("7", {"gt": 5.0}, False),
            ([7], {"le": 5.0}, False),
            (float("nan"), {"gt": 5.0}, False),
            (float("nan"), {"le": 5.0}, False),
            (True, {"gt": 0.5}, True),
            (np.float32(7), {"gt": 5.0, "le": 10.5}, True),
            (np.int64(7), {"in": [1, 7]}, True),
            (np.array([7, 8]), {"gt": 5.0}, False),
            (np.array([7, 8]), {"eq": 7}, False),
            (np.array([7, 8]), {"in": [7]}, False),
            (np.array([7, 8]), {"neq": 7}, False),
            (np.array([7]), {"eq": 7}, False),
            (np.bool_(True), {"eq": True}, True),
            (np.bool_(True), {"gt": 0.5}, True),
            (["read"], {"neq": "read"}, False),
            ({"read": 1}, {"neq": "read"}, False),
            (7 + 0j, {"eq": 7}, False),
            (7, OmegaConf.create({"gt": 5.0, "le": 10.5}), True),
            ("low", OmegaConf.create({"in": ["high", "medium"]}), False),
            (7, {"gt": 5.0, "keep_if_missing": False}, True),
            (3, {"gt": 5.0, "keep_if_missing": False}, False),
            (None, {"keep_if_missing": False}, True),
        ],
    )
    def test_operators(self, value, condition, expected):
        cut = dummy_cut(0)
        cut.custom = {"field": value}
        assert CustomFieldFilter({"field": condition})(cut) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "condition, expected",
        [
            (True, True),
            ({"eq": True}, True),
            ({"gt": 5.0}, True),
            ({"in": ["high"]}, True),
            ({"eq": True, "keep_if_missing": True}, True),
            ({"eq": True, "keep_if_missing": False}, False),
            ({"gt": 5.0, "le": 10.5, "keep_if_missing": False}, False),
            ({"keep_if_missing": False}, False),
            (OmegaConf.create({"in": ["high"], "keep_if_missing": False}), False),
        ],
    )
    @pytest.mark.parametrize("custom", [None, {}, {"other": 7}])
    def test_keep_if_missing_decides_for_an_example_without_the_field(self, custom, condition, expected):
        custom_field_filter = CustomFieldFilter({"field": condition})
        cut = dummy_cut(0)
        cut.custom = custom
        assert custom_field_filter(cut) is expected
        assert custom_field_filter(TextExample("some text", custom=custom)) is expected
        other = dummy_cut(1)
        other.custom = None
        assert custom_field_filter(cut.mix(other)) is expected

    @pytest.mark.unit
    def test_conditions_of_all_fields_must_hold(self):
        custom_field_filter = CustomFieldFilter(
            {
                "verified": True,
                "snr": {"gt": 5.0, "le": 10.5},
                "quality": {"in": ["high", "medium"], "keep_if_missing": False},
            }
        )
        cut = dummy_cut(0)
        for custom, expected in [
            ({"verified": True, "snr": 7.0, "quality": "high"}, True),
            ({"snr": 7.0, "quality": "medium"}, True),
            ({"quality": "medium"}, True),
            ({"verified": False, "snr": 7.0, "quality": "high"}, False),
            ({"verified": True, "snr": 11.0, "quality": "high"}, False),
            ({"verified": True, "snr": 7.0, "quality": "low"}, False),
            ({"verified": True, "snr": 7.0}, False),
        ]:
            cut.custom = custom
            assert custom_field_filter(cut) is expected, custom

    @pytest.mark.unit
    def test_mixed_cut_with_all_tracks_muted_has_no_field(self):
        cuts = [dummy_cut(0), dummy_cut(1)]
        for cut in cuts:
            cut.custom = {"snr": 7.0}
        mixed = cuts[0].mix(cuts[1])
        mixed = fastcopy(mixed, tracks=[fastcopy(track, mute=True) for track in mixed.tracks])
        assert mixed.has_custom("snr") is False
        assert CustomFieldFilter({"snr": {"gt": 5.0}})(mixed) is True
        assert CustomFieldFilter({"snr": {"gt": 5.0, "keep_if_missing": False}})(mixed) is False

    @pytest.mark.unit
    def test_filter_can_be_pickled(self):
        conditions = {"verified": True, "snr": {"gt": 5.0, "keep_if_missing": False}, "quality": {"in": ["high"]}}
        custom_field_filter = pickle.loads(pickle.dumps(CustomFieldFilter(OmegaConf.create(conditions))))
        cut = dummy_cut(0)
        cut.custom = {"snr": 7.0, "quality": "high"}
        assert custom_field_filter(cut) is True
        cut.custom = {"snr": 7.0, "quality": "low"}
        assert custom_field_filter(cut) is False

    @pytest.mark.unit
    def test_keep_if_missing_does_not_apply_to_examples_that_cannot_have_custom_fields(self):
        padding = PaddingCut(id="padding", duration=1.0, sampling_rate=16000, feat_value=0.0)
        assert CustomFieldFilter({"verified": {"eq": True, "keep_if_missing": False}})(padding) is True

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
        "conditions",
        [
            {"verified": [True, False]},
            {"verified": None},
            {"verified": {"nested": True}},
            {"verified": float("nan")},
            {"verified": {}},
            {"verified": {"eq": [True]}},
            {"verified": {"eq": None}},
            {"verified": {"neq": float("nan")}},
            {"verified": {"equals": True}},
            {"verified": {"gt": "5"}},
            {"verified": {"ge": True}},
            {"verified": {"lt": None}},
            {"verified": {"le": float("nan")}},
            {"verified": {"in": "high"}},
            {"verified": {"in": []}},
            {"verified": {"in": [["high"]]}},
            {"verified": {"in": [None]}},
            {"verified": {"in": 3}},
            {"verified": {"eq": True, "keep_if_missing": "no"}},
            {"verified": {"keep_if_missing": 0}},
            {"keep_if_missing": False, "verified": True},
            {"keep_if_missing": True},
            OmegaConf.create({"verified": {"in": []}}),
            OmegaConf.create({"verified": {"between": [1, 2]}}),
            {"": True},
            {3: True},
            "verified",
            ["verified"],
        ],
    )
    def test_rejects_invalid_conditions(self, conditions):
        with pytest.raises(ValueError, match="custom_field_filter"):
            CustomFieldFilter(conditions)


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
    @pytest.mark.parametrize("source", ["manifest_config", "tarred_manifest_config"])
    @pytest.mark.parametrize(
        "snr_keep_if_missing, expected", [(True, ["utt-3", "utt-4", "utt-6"]), (False, ["utt-3", "utt-6"])]
    )
    def test_operators_select_the_cuts_of_the_predicate(self, source, snr_keep_if_missing, expected, request):
        def predicate(cut: Cut) -> bool:
            if cut.has_custom("snr"):
                if cut.snr is None or not 5.0 < cut.snr <= 12.0:
                    return False
            elif not snr_keep_if_missing:
                return False
            return not cut.has_custom("quality") or cut.quality in ("very_high", "high", "medium")

        source_config = request.getfixturevalue(source)
        all_cuts = _read_all_cuts(source_config)
        assert sorted(cut.supervisions[0].text for cut in all_cuts if predicate(cut)) == expected

        custom_field_filter = {
            "snr": {"gt": 5.0, "le": 12.0, "keep_if_missing": snr_keep_if_missing},
            "quality": {"in": ["very_high", "high", "medium"]},
        }
        cuts = _read_all_cuts(source_config, custom_field_filter=custom_field_filter)

        assert sorted(cut.supervisions[0].text for cut in cuts) == expected

    @pytest.mark.unit
    def test_conditions_from_yaml_text(self, manifest_config):
        yaml_text = (
            "custom_field_filter:\n"
            "  verified: true\n"
            "  snr: {ge: 6, lt: 18.0, keep_if_missing: false}\n"
            "  quality: {in: [very_high, high]}\n"
        )
        cuts = _read_all_cuts(manifest_config, **OmegaConf.create(yaml_text))
        assert sorted(cut.supervisions[0].text for cut in cuts) == ["utt-3", "utt-6"]

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
        for condition in ([True, False], {"equals": True}, {"gt": "5"}):
            config["custom_field_filter"] = {"verified": condition}
            with pytest.raises(ValueError, match="custom_field_filter.*'verified'"):
                get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
