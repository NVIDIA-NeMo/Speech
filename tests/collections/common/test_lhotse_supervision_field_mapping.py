# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import pytest
from lhotse import CutSet, MonoCut, SupervisionSegment
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.cutset import SupervisionFieldMapping, read_cutset_from_config
from nemo.collections.common.data.lhotse.sampling import ContextSpeakerSimilarityFilter


def make_cut(identifier="sample"):
    return MonoCut(
        id=identifier,
        start=0,
        duration=1,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id=f"{identifier}-{i}",
                recording_id=identifier,
                start=0,
                duration=1,
                custom={
                    "ipa": "old",
                    "alternative_ipa": "new",
                    "ipa_alignment": [],
                    "context_speaker_similarity": 0.1,
                },
            )
            for i in range(2)
        ],
    )


def test_mapping_all_supervisions_without_mutating_source():
    cut = make_cut()
    original = deepcopy(cut)
    mapped = SupervisionFieldMapping(
        {"ipa": "alternative_ipa", "ipa_alignment": None, "context_speaker_similarity": None}
    )(cut)
    for supervision in mapped.supervisions:
        assert supervision.ipa == "new"
        assert supervision.alternative_ipa == "new"
        assert not supervision.has_custom("ipa_alignment")
        assert not supervision.has_custom("context_speaker_similarity")
    assert cut == original
    assert not ContextSpeakerSimilarityFilter(0.6)(cut)
    assert ContextSpeakerSimilarityFilter(0.6)(mapped)


def test_mapping_uses_original_values_for_swaps_and_deletions():
    mapped = SupervisionFieldMapping({"ipa": "alternative_ipa", "alternative_ipa": "ipa"})(make_cut())
    assert mapped.supervisions[0].ipa == "new"
    assert mapped.supervisions[0].alternative_ipa == "old"
    mapped = SupervisionFieldMapping({"alternative_ipa": None, "ipa": "alternative_ipa"})(make_cut())
    assert mapped.supervisions[0].ipa == "new"


def test_missing_source_and_deleting_missing_field_are_allowed():
    mapped = SupervisionFieldMapping({"ipa": "absent"})(make_cut())
    assert mapped.supervisions[0].ipa == "old"
    cut = make_cut()
    cut.supervisions[0].custom = None
    assert SupervisionFieldMapping({"ipa": None})(cut).supervisions[0].custom == {}


def test_ignore_missing_source_preserves_destination_and_applies_other_mappings():
    cut = make_cut()
    del cut.supervisions[0].custom["alternative_ipa"]
    mapped = SupervisionFieldMapping({"ipa": "alternative_ipa", "ipa_alignment": None})(cut)
    assert mapped.supervisions[0].ipa == "old"
    assert mapped.supervisions[1].ipa == "new"
    assert all(not supervision.has_custom("ipa_alignment") for supervision in mapped.supervisions)
    assert cut.supervisions[0].has_custom("ipa_alignment")
    cut.supervisions[0].custom = None
    mapped = SupervisionFieldMapping({"ipa": "alternative_ipa"})(cut)
    assert not mapped.supervisions[0].has_custom("ipa")


def test_present_null_source_preserves_destination():
    cut = make_cut()
    cut.supervisions[0].custom["alternative_ipa"] = None
    mapped = SupervisionFieldMapping({"ipa": "alternative_ipa"})(cut)
    assert mapped.supervisions[0].has_custom("ipa")
    assert mapped.supervisions[0].ipa == "old"


@pytest.mark.parametrize("source_value", [None, "", " \t\n", "written 100", 0, False])
@pytest.mark.parametrize("existing_destination", [False, True])
def test_copied_source_preserves_destination_when_empty(source_value, existing_destination):
    cut = make_cut()
    for supervision in cut.supervisions:
        supervision.text = "spoken one hundred"
        supervision.custom["itn_text_raw"] = source_value
        if existing_destination:
            supervision.custom["normalized_text"] = "existing transcript"
    original = deepcopy(cut)
    mapped = SupervisionFieldMapping({"normalized_text": "itn_text_raw", "ipa_alignment": None})(cut)
    empty = source_value is None or (isinstance(source_value, str) and not source_value.strip())
    for supervision in mapped.supervisions:
        assert supervision.text == "spoken one hundred"
        assert not supervision.has_custom("ipa_alignment")
        if empty:
            assert supervision.has_custom("normalized_text") == existing_destination
            if existing_destination:
                assert supervision.normalized_text == "existing transcript"
        else:
            assert supervision.normalized_text == source_value
            assert type(supervision.normalized_text) is type(source_value)
    assert cut == original


def test_missing_source_uses_original_values():
    cut = make_cut()
    mapped = SupervisionFieldMapping(
        {
            "alternative_ipa": None,
            "ipa": "alternative_ipa",
            "normalized_text": "absent",
        }
    )(cut)
    assert all(supervision.ipa == "new" for supervision in mapped.supervisions)
    assert all(not supervision.has_custom("alternative_ipa") for supervision in mapped.supervisions)
    assert all(not supervision.has_custom("normalized_text") for supervision in mapped.supervisions)


@pytest.mark.parametrize(
    "mapping", [[], {"ipa": 1}, {"ipa": ""}, {"text": None}, {"ipa": "text"}, {"ipa": {"source": "alternative_ipa"}}]
)
def test_invalid_mapping(mapping):
    with pytest.raises(ValueError, match="supervision_field_mapping"):
        SupervisionFieldMapping(mapping)


@pytest.mark.parametrize("ignore_missing", [False, True])
def test_per_dataset_yaml_mapping(tmp_path, ignore_missing):
    paths = [tmp_path / f"{i}.jsonl.gz" for i in range(2)]
    for i, path in enumerate(paths):
        cut = make_cut(str(i))
        if ignore_missing and i == 0:
            del cut.supervisions[0].custom["alternative_ipa"]
        CutSet.from_cuts([cut]).to_file(path)
    config_path = tmp_path / "datasets.yaml"
    OmegaConf.save(
        OmegaConf.create(
            [
                {
                    "type": "lhotse",
                    "cuts_path": str(paths[0]),
                    "supervision_field_mapping": {"ipa": "alternative_ipa", "context_speaker_similarity": None},
                },
                {"type": "lhotse", "cuts_path": str(paths[1])},
            ]
        ),
        config_path,
    )
    cuts, _ = read_cutset_from_config({"input_cfg": str(config_path), "force_finite": True})
    by_id = {cut.id: cut for cut in cuts}
    assert by_id["0"].supervisions[0].ipa == ("old" if ignore_missing else "new")
    assert by_id["0"].supervisions[1].ipa == "new"
    assert not by_id["0"].supervisions[0].has_custom("context_speaker_similarity")
    assert by_id["1"].supervisions[0].ipa == "old"
    assert by_id["1"].supervisions[0].has_custom("context_speaker_similarity")
