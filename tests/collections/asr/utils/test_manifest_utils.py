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

import json

import pytest

from nemo.collections.asr.parts.utils.manifest_utils import get_subsegment_dict
from nemo.collections.asr.parts.utils.speaker_utils import get_subsegments_scriptable


@pytest.mark.unit
def test_get_subsegment_dict_keeps_all_subsegments(tmp_path):
    """
    Regression test: get_subsegment_dict must append one ('ts', 'json_dic') entry
    per subsegment, not just the last one.

    Previously the two ``.append(...)`` calls were indented outside the
    ``for subsegment in subsegments`` loop, so ``start, dur`` only held the last
    subsegment and every earlier subsegment was silently dropped.
    """
    window, shift, deci = 1.5, 0.75, 3
    offset, duration = 0.0, 5.0

    # Sanity: these parameters yield multiple subsegments (6), so the bug is observable.
    expected_subsegments = get_subsegments_scriptable(offset=offset, window=window, shift=shift, duration=duration)
    assert len(expected_subsegments) > 1

    manifest_path = tmp_path / "subsegments_manifest.json"
    entry = {
        "audio_filepath": "/tmp/example.wav",
        "offset": offset,
        "duration": duration,
        "uniq_id": "example",
    }
    manifest_path.write_text(json.dumps(entry) + "\n")

    subsegment_dict = get_subsegment_dict(str(manifest_path), window=window, shift=shift, deci=deci)

    assert set(subsegment_dict.keys()) == {"example"}
    ts = subsegment_dict["example"]["ts"]
    json_dic = subsegment_dict["example"]["json_dic"]

    # One entry per subsegment (would be 1 with the bug).
    assert len(ts) == len(expected_subsegments)
    assert len(json_dic) == len(expected_subsegments)

    expected_ts = [[round(start, deci), round(start + dur, deci)] for start, dur in expected_subsegments]
    assert ts == expected_ts
