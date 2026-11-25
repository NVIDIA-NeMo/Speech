# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the ASCEND data CLI against a tiny local Hub fixture."""

import hashlib
import io
import json
import runpy
import sys
from pathlib import Path

import huggingface_hub
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[3] / ".claude/skills/rl-finetune-speech-llm/scripts/prepare_ascend.py"


@pytest.mark.unit
def test_ascend_subset_and_split_contract(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    wave = io.BytesIO()
    sf.write(wave, np.zeros((160, 2)), 16000, format="WAV", subtype="PCM_16")
    files = {}
    for split, count in [("train", 8), ("validation", 3), ("test", 4)]:
        rows = [
            {
                "id": str(i),
                "audio": {"bytes": wave.getvalue()},
                "transcription": f"你好 number {i}",
                "language": "mixed",
                "original_speaker_id": i,
            }
            for i in range(count)
        ]
        filename = f"main/{split}-00000.parquet"
        local = hub / (split + ".parquet")
        pq.write_table(pa.Table.from_pylist(rows), local)
        files[filename] = str(local)

    class LocalHub:
        def list_repo_files(self, repo, *, repo_type, revision):
            assert (repo, repo_type, revision) == ("CAiRE/ASCEND", "dataset", "fixture-revision")
            return list(files)

    def download(repo, filename, *, repo_type, revision):
        assert (repo, repo_type, revision) == ("CAiRE/ASCEND", "dataset", "fixture-revision")
        return files[filename]

    monkeypatch.setattr(huggingface_hub, "HfApi", LocalHub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output", str(output), "--train-samples", "3", "--revision", "fixture-revision"]
    )
    runpy.run_path(str(SCRIPT), run_name="__main__")
    rows = pq.read_table(output / "train.parquet").to_pylist()
    selected = sorted(range(8), key=lambda i: hashlib.sha256(f"42:{i}".encode()).hexdigest())[:3]
    assert [r["id"] for r in rows] == [f"ascend_train_{i}" for i in selected]
    for row in rows:
        assert row["messages"] == [{"role": "user", "content": "Transcribe the audio into text."}]
        assert row["reward_metric"] == "mer"
        assert row["text"] == row["expected_answer"]
        assert row["text"] not in row["messages"][0]["content"]
        audio = sf.info(row["audio_filepath"])
        assert (audio.samplerate, audio.channels, audio.frames) == (16000, 1, 160)
    receipt = json.loads((output / "DATA_DONE.json").read_text())
    assert {key: value["rows"] for key, value in receipt["splits"].items()} == {"train": 3, "validation": 3, "test": 4}
    for split, item in receipt["splits"].items():
        assert item["sha256"] == hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()
        assert all(identity.startswith(f"ascend_{split}_") for identity in item["ids"])
    assert (output / "pilot32.jsonl").read_bytes() == (output / "train.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="preserve prior data"):
        runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.unit
@pytest.mark.parametrize("count", ["0", "1001"])
def test_ascend_size_bound(tmp_path, monkeypatch, count):
    output = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output), "--train-samples", count])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert error.value.code == 2
    assert not output.exists()
