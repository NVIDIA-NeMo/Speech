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
from copy import deepcopy

import pytest
import torch
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.sampling import SupervisionCustomFilter


REQUIRED = {
    "valid_tn_raw_transcript": True,
    "valid_ipa": True,
    "early_cut_off": False,
    "valid_context_audio_text": True,
}


class IdentityDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        return cuts


def make_cut(identifier, metadata):
    return MonoCut(
        id=identifier,
        start=0,
        duration=1,
        channel=0,
        recording=Recording(
            id=identifier,
            sources=[AudioSource(type="file", channels=[0], source="unused.wav")],
            sampling_rate=16000,
            num_samples=16000,
            duration=1,
        ),
        supervisions=[
            SupervisionSegment(
                id=identifier,
                recording_id=identifier,
                start=0,
                duration=1,
                channel=0,
                text="A sample transcript.",
                custom=metadata,
            )
        ],
    )


@pytest.mark.parametrize("field", list(REQUIRED))
def test_supervision_filters_check_present_fields_and_skip_missing(field):
    predicate = SupervisionCustomFilter(REQUIRED)
    assert predicate(make_cut("good", REQUIRED.copy()))
    metadata = REQUIRED.copy()
    metadata[field] = not metadata[field]
    assert not predicate(make_cut("wrong", metadata))
    del metadata[field]
    assert predicate(make_cut("missing", metadata))
    assert predicate(make_cut("unannotated", {}))
    assert predicate(make_cut("no-custom", None))


@pytest.mark.parametrize("value", [None, "false", 0])
def test_supervision_filters_reject_wrong_boolean_types(value):
    assert not SupervisionCustomFilter(REQUIRED)(make_cut("wrong-type", dict(REQUIRED, early_cut_off=value)))


@pytest.mark.parametrize("first_metadata", [None, {}, REQUIRED])
def test_supervision_filters_check_only_first_supervision(first_metadata):
    cut = make_cut("first", deepcopy(first_metadata))
    predicate = SupervisionCustomFilter(REQUIRED)
    cut.supervisions.append(make_cut("second", {"early_cut_off": True}).supervisions[0])
    assert predicate(cut)
    cut.supervisions.reverse()
    assert not predicate(cut)


def test_supervision_filters_pass_other_examples():
    predicate = SupervisionCustomFilter(REQUIRED)
    cut = make_cut("no-supervisions", None)
    cut.supervisions = []
    assert predicate(cut)
    assert predicate("text example")


@pytest.mark.parametrize("value", [True, False, 3, 0.5, "pass", None])
def test_supervision_filters_scalar_values(value):
    predicate = SupervisionCustomFilter({"quality": value})
    assert predicate(make_cut("matching", {"quality": value}))
    assert predicate(make_cut("missing", {}))
    assert not predicate(make_cut("wrong", {"quality": "wrong"}))


@pytest.mark.parametrize("filters", [None, {}])
def test_supervision_filters_disabled(filters):
    predicate = SupervisionCustomFilter(filters)
    assert predicate(make_cut("unannotated", None))
    assert predicate(make_cut("mismatch", {"early_cut_off": True}))
    assert predicate("text example")


@pytest.mark.parametrize("filters", [["valid_ipa"], {"valid_ipa": [True]}, {"valid_ipa": {"eq": True}}, {1: True}])
def test_supervision_filters_reject_invalid_config(filters):
    with pytest.raises(ValueError, match="supervision_filters"):
        SupervisionCustomFilter(filters)


@pytest.mark.parametrize("source_type", ["lhotse", "lhotse_shar"])
@pytest.mark.parametrize("filters", [None, {}, REQUIRED])
@pytest.mark.parametrize("multi_config", [False, True])
def test_supervision_filters_in_global_pipeline(tmp_path, source_type, filters, multi_config):
    annotated = [
        make_cut("good", REQUIRED.copy()),
        make_cut("cutoff", dict(REQUIRED, early_cut_off=True)),
        make_cut("high-cer", dict(REQUIRED, cer=0.5)),
        make_cut("low-similarity", dict(REQUIRED, context_speaker_similarity=0.1)),
    ]
    baseline = [
        make_cut("unannotated", None),
        make_cut("partial", {"valid_ipa": True}),
        make_cut("partial-mismatch", {"valid_ipa": False}),
    ]
    sources = []
    for name, cuts in [("annotated", annotated), ("baseline", baseline)]:
        path = tmp_path / name
        path.mkdir()
        manifest = CutSet.from_cuts(cuts)
        if source_type == "lhotse_shar":
            manifest.to_shar(path, fields={})
            sources.append({"type": source_type, "shar_path": str(path)})
        else:
            path = path / "cuts.jsonl.gz"
            manifest.to_file(path)
            sources.append({"type": source_type, "cuts_path": str(path)})
    config = {
        "input_cfg": sources,
        "batch_size": 2,
        "use_bucketing": False,
        "drop_last": False,
        "force_finite": True,
        "seed": 0,
        "shard_seed": 0,
        "shuffle": False,
        "max_cer": 0.03,
        "min_context_speaker_similarity": 0.6,
    }
    if filters is not None:
        config["supervision_filters"] = deepcopy(filters)
    if multi_config:
        config = {
            "multi_config": True,
            "seed": 0,
            "shard_seed": 0,
            "shuffle": False,
            "force_finite": True,
            "tts_data": config,
        }
    loader = get_lhotse_dataloader_from_config(
        config=OmegaConf.create(config), global_rank=0, world_size=1, dataset=IdentityDataset()
    )
    ids = sorted(cut.id for batch in loader for cut in batch)
    expected = ["good", "partial", "unannotated"]
    if not filters:
        expected += ["cutoff", "partial-mismatch"]
    assert ids == sorted(expected)
