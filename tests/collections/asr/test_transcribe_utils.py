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

from nemo.collections.asr.parts.utils.transcribe_utils import (
    read_and_maybe_sort_manifest,
    restore_transcription_order,
    write_transcription,
)


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


def _write_manifest_with_blank_line(tmp_path):
    rows = [
        {"audio_filepath": "a.wav", "duration": 1.0, "text": "a"},
        {"audio_filepath": "b.wav", "duration": 3.0, "text": "b"},
        {"audio_filepath": "c.wav", "duration": 2.0, "text": "c"},
    ]
    lines = [json.dumps(rows[0]), "", json.dumps(rows[1]), json.dumps(rows[2])]
    manifest = tmp_path / "manifest.json"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_restore_transcription_order_skips_blank_lines(tmp_path):
    manifest = _write_manifest_with_blank_line(tmp_path)
    # The model transcribes the entries sorted by duration, longest first.
    items = read_and_maybe_sort_manifest(str(manifest), try_sort=True)
    assert [item["audio_filepath"] for item in items] == ["b.wav", "c.wav", "a.wav"]

    transcriptions = ["pred " + item["text"] for item in items]
    assert restore_transcription_order(str(manifest), transcriptions) == ["pred a", "pred b", "pred c"]


def test_write_transcription_skips_blank_lines_in_manifest(tmp_path):
    manifest = _write_manifest_with_blank_line(tmp_path)
    output = tmp_path / "transcriptions.json"
    cfg = OmegaConf.create(
        {
            "append_pred": False,
            "output_filename": str(output),
            "audio_dir": None,
            "dataset_manifest": str(manifest),
        }
    )

    write_transcription(transcriptions=["pred a", "pred b", "pred c"], cfg=cfg, model_name="test-model")

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(record["audio_filepath"], record["pred_text"]) for record in records] == [
        ("a.wav", "pred a"),
        ("b.wav", "pred b"),
        ("c.wav", "pred c"),
    ]
