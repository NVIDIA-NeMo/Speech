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
from lhotse import CutSet, MonoCut, SupervisionSegment
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.cutset import SupervisionCustomFilter, parse_group


REQUIRED = {
    "valid_tn_raw_transcript": True,
    "valid_ipa": True,
    "early_cut_off": False,
    "valid_context_audio_text": True,
}


def make_cut(identifier, metadata):
    return MonoCut(
        id=identifier,
        start=0,
        duration=1,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id=identifier,
                recording_id=identifier,
                start=0,
                duration=1,
                channel=0,
                custom=metadata,
            )
        ],
    )


@pytest.mark.parametrize("source_type", ["lhotse", "lhotse_shar"])
def test_supervision_filters_are_dataset_scoped(tmp_path, source_type):
    good = make_cut("good", REQUIRED.copy())
    cuts = [good]
    for field, value in REQUIRED.items():
        metadata = REQUIRED.copy()
        metadata[field] = not value
        cuts.append(make_cut("wrong-" + field, metadata))
        metadata = REQUIRED.copy()
        del metadata[field]
        cuts.append(make_cut("missing-" + field, metadata))
    for value in [None, "false", 0]:
        cuts.append(make_cut("type-" + str(value), dict(REQUIRED, early_cut_off=value)))
    multiple = deepcopy(good)
    multiple.id = "multiple"
    multiple.supervisions.append(make_cut("bad", {}).supervisions[0])
    cuts.append(multiple)
    empty = deepcopy(good)
    empty.id = "empty"
    empty.supervisions = []
    cuts.append(empty)
    manifest = CutSet.from_cuts(cuts)
    if source_type == "lhotse_shar":
        manifest.to_shar(tmp_path, fields={})
        source = {"type": source_type, "shar_path": str(tmp_path), "force_finite": True}
    else:
        path = tmp_path / "cuts.jsonl.gz"
        manifest.to_file(path)
        source = {"type": source_type, "cuts_path": str(path)}
    config = OmegaConf.create(
        {
            "type": "group",
            "input_cfg": [
                dict(source, supervision_filters=REQUIRED),
                dict(source, tags={"unfiltered": True}),
            ],
        }
    )
    result, _ = parse_group(
        config,
        propagate_attrs={
            "reweight_temperature": None,
            "max_open_streams": None,
            "shard_seed": 0,
            "force_finite": True,
            "metadata_only": False,
        },
    )
    result = list(result)
    assert [c.id for c in result if not getattr(c, "unfiltered", False)] == ["good"]
    assert sum(getattr(c, "unfiltered", False) for c in result) == len(cuts)


def test_supervision_filters_on_group(tmp_path):
    path = tmp_path / "cuts.jsonl.gz"
    CutSet.from_cuts([make_cut("good", REQUIRED), make_cut("bad", {})]).to_file(path)
    config = OmegaConf.create(
        {
            "type": "group",
            "supervision_filters": REQUIRED,
            "input_cfg": [{"type": "lhotse", "cuts_path": str(path)}],
        }
    )
    cuts, _ = parse_group(
        config,
        propagate_attrs={
            "reweight_temperature": None,
            "max_open_streams": None,
            "shard_seed": 0,
            "force_finite": True,
            "metadata_only": False,
        },
    )
    assert [c.id for c in cuts] == ["good"]


@pytest.mark.parametrize("filters", [["valid_ipa"], {"valid_ipa": [True]}, {"valid_ipa": {"eq": True}}])
def test_supervision_filters_reject_invalid_config(tmp_path, filters):
    path = tmp_path / "cuts.jsonl.gz"
    CutSet.from_cuts([make_cut("good", REQUIRED)]).to_file(path)
    with pytest.raises(ValueError, match="supervision_filters"):
        parse_group(
            OmegaConf.create(
                {
                    "type": "lhotse",
                    "cuts_path": str(path),
                    "supervision_filters": filters,
                }
            ),
            propagate_attrs={
                "reweight_temperature": None,
                "max_open_streams": None,
                "shard_seed": 0,
                "force_finite": True,
                "metadata_only": False,
            },
        )


@pytest.mark.parametrize("value", [True, False, 3, 0.5, "pass", None])
def test_supervision_filters_scalar_values(value):
    predicate = SupervisionCustomFilter({"quality": value})
    assert predicate(make_cut("matching", {"quality": value}))
    assert not predicate(make_cut("missing", {}))
    assert not predicate(make_cut("no-custom", None))
    assert not predicate("text example")


def test_empty_supervision_filters_disable_filtering():
    predicate = SupervisionCustomFilter({})
    assert predicate(make_cut("unannotated", None))
    assert predicate("text example")
