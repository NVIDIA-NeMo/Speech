# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    find_first_nonzero,
    get_hidden_length_from_sample_length,
    read_rttm_supervisions_lenient,
)


def _write_rttm(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "SPEAKER rec1 1 1.25 2.50 <NA> <NA> speaker_A <NA>",
        "SPEAKER rec1 1 1.25 2.50 <NA> <NA> speaker_A <NA> <NA>",
    ],
)
def test_read_rttm_supervisions_lenient_accepts_9_and_10_column_lines(tmp_path, line):
    rttm_path = _write_rttm(tmp_path / "valid.rttm", [line])

    supervisions = read_rttm_supervisions_lenient(rttm_path)

    assert len(supervisions) == 1
    segment = supervisions[0]
    assert segment.id == "rec1-000000"
    assert segment.recording_id == "rec1"
    assert segment.channel == 1
    assert segment.start == pytest.approx(1.25)
    assert segment.duration == pytest.approx(2.50)
    assert segment.speaker == "speaker_A"


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "SPEAKER rec1 1 0.0 1.0 <NA> <NA>",
        "SPEAKER rec1 1 0.0 1.0",
        "too short",
    ],
)
def test_read_rttm_supervisions_lenient_rejects_short_lines(tmp_path, line):
    rttm_path = _write_rttm(tmp_path / "invalid.rttm", [line])

    with pytest.raises(ValueError, match="Invalid RTTM line"):
        read_rttm_supervisions_lenient(rttm_path)


@pytest.mark.unit
def test_read_rttm_supervisions_lenient_skips_blank_and_zero_duration_lines(tmp_path):
    rttm_path = _write_rttm(
        tmp_path / "skip.rttm",
        [
            "",
            "SPEAKER rec1 1 0.00 0.00 <NA> <NA> speaker_A <NA>",
            "SPEAKER rec1 1 3.00 1.25 <NA> <NA> speaker_B <NA>",
        ],
    )

    supervisions = read_rttm_supervisions_lenient(rttm_path)

    assert len(supervisions) == 1
    segment = supervisions[0]
    assert segment.id == "rec1-000002"
    assert segment.start == pytest.approx(3.00)
    assert segment.duration == pytest.approx(1.25)
    assert segment.speaker == "speaker_B"


@pytest.mark.unit
def test_read_rttm_supervisions_lenient_accepts_multiple_files(tmp_path):
    rttm_path_a = _write_rttm(tmp_path / "a.rttm", ["SPEAKER rec_a 1 0.00 1.00 <NA> <NA> speaker_A <NA>"])
    rttm_path_b = _write_rttm(tmp_path / "b.rttm", ["SPEAKER rec_b 2 1.00 2.00 <NA> <NA> speaker_B <NA>"])

    supervisions = read_rttm_supervisions_lenient([rttm_path_a, rttm_path_b])

    assert len(supervisions) == 2
    assert [segment.recording_id for segment in supervisions] == ["rec_a", "rec_b"]
    assert [segment.channel for segment in supervisions] == [1, 2]
    assert [segment.speaker for segment in supervisions] == ["speaker_A", "speaker_B"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("num_samples", "expected_hidden_length"),
    [
        (0, 0),
        (1, 1),
        (1280, 1),
        (1281, 2),
    ],
)
def test_get_hidden_length_rounds_up_to_encoder_frames(num_samples, expected_hidden_length):
    assert get_hidden_length_from_sample_length(num_samples) == expected_hidden_length


@pytest.mark.unit
def test_find_first_nonzero_returns_first_threshold_crossing_or_cap():
    mat = torch.tensor(
        [
            [0.0, 0.2, 0.6],
            [0.0, 0.0, 0.0],
            [0.7, 0.8, 0.0],
        ]
    )

    result = find_first_nonzero(mat, max_cap_val=99, thres=0.5)

    assert torch.equal(result, torch.tensor([2, 99, 0]))
