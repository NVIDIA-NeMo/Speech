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


def test_missing_source_is_error_but_deleting_missing_field_is_allowed():
    with pytest.raises(ValueError, match="missing.*absent"):
        SupervisionFieldMapping({"ipa": "absent"})(make_cut())
    cut = make_cut()
    cut.supervisions[0].custom = None
    assert SupervisionFieldMapping({"ipa": None})(cut).supervisions[0].custom == {}


@pytest.mark.parametrize("mapping", [[], {"ipa": 1}, {"ipa": ""}, {"text": None}, {"ipa": "text"}])
def test_invalid_mapping(mapping):
    with pytest.raises(ValueError, match="supervision_field_mapping"):
        SupervisionFieldMapping(mapping)


def test_per_dataset_yaml_mapping(tmp_path):
    paths = [tmp_path / f"{i}.jsonl.gz" for i in range(2)]
    for i, path in enumerate(paths):
        CutSet.from_cuts([make_cut(str(i))]).to_file(path)
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
    assert by_id["0"].supervisions[0].ipa == "new"
    assert not by_id["0"].supervisions[0].has_custom("context_speaker_similarity")
    assert by_id["1"].supervisions[0].ipa == "old"
    assert by_id["1"].supervisions[0].has_custom("context_speaker_similarity")
