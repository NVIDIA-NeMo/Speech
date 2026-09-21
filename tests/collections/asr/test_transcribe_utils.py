# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.transcribe_utils import write_transcription


def test_write_transcription_preserves_unicode_for_audio_dir(tmp_path):
    output = tmp_path / "transcriptions.json"
    cfg = OmegaConf.create(
        {
            "append_pred": False,
            "output_filename": str(output),
            "audio_dir": str(tmp_path),
        }
    )

    write_transcription(
        transcriptions=["Grüße François"],
        cfg=cfg,
        model_name="test-model",
        filepaths=["sample.wav"],
    )

    raw = output.read_text(encoding="utf-8")
    assert "Grüße François" in raw
    assert "\\u" not in raw
    assert json.loads(raw)["pred_text"] == "Grüße François"


def test_write_transcription_preserves_unicode_for_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"audio_filepath": "sample.wav", "text": "référence"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "transcriptions.json"
    cfg = OmegaConf.create(
        {
            "append_pred": False,
            "output_filename": str(output),
            "audio_dir": None,
            "dataset_manifest": str(manifest),
        }
    )

    write_transcription(
        transcriptions=["Grüße François"],
        cfg=cfg,
        model_name="test-model",
    )

    raw = output.read_text(encoding="utf-8")
    assert "Grüße François" in raw
    assert "référence" in raw
    assert "\\u" not in raw
    record = json.loads(raw)
    assert record["pred_text"] == "Grüße François"
    assert record["text"] == "référence"
