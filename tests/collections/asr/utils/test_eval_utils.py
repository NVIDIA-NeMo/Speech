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

import json

import pytest

from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer


def _write_manifest(path, samples):
    with open(path, 'w', encoding='utf-8') as fp:
        for sample in samples:
            fp.write(json.dumps(sample) + '\n')


class TestCalWriteWer:
    @pytest.mark.unit
    def test_computes_wer(self, tmp_path):
        manifest = tmp_path / "pred.json"
        _write_manifest(manifest, [{"text": "hello world", "pred_text": "hello word"}])

        output_manifest, total_res, eval_metric = cal_write_wer(pred_manifest=str(manifest))

        assert output_manifest == str(manifest)
        assert eval_metric == "wer"
        assert total_res["wer"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_falls_back_to_text_when_gt_attr_missing(self, tmp_path):
        manifest = tmp_path / "pred.json"
        _write_manifest(manifest, [{"text": "hello world", "pred_text": "hello word"}])

        output_manifest, total_res, _ = cal_write_wer(pred_manifest=str(manifest), gt_text_attr_name="answer")

        assert output_manifest == str(manifest)
        assert total_res["samples"] == 1
        assert total_res["wer"] == pytest.approx(0.5)

    @pytest.mark.unit
    def test_returns_none_without_ground_truth(self, tmp_path):
        manifest = tmp_path / "pred.json"
        _write_manifest(manifest, [{"pred_text": "hello word"}])

        assert cal_write_wer(pred_manifest=str(manifest), gt_text_attr_name="answer") == (None, None, "wer")
