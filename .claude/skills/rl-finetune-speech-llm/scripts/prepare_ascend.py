# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a reproducible 500-utterance ASCEND GSPO subset on cluster storage."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--revision", default="737e9800ae31be9932ba8464c80366559bd28424")
parser.add_argument("--train-samples", type=int, default=500)
args = parser.parse_args()
if not 1 <= args.train_samples <= 1000:
    parser.error("--train-samples must be between 1 and 1000")
root = args.output.resolve()
if root.exists() and any(root.iterdir()):
    raise FileExistsError(f"Use a new output directory to preserve prior data: {root}")
root.mkdir(parents=True, exist_ok=True)
repo = "CAiRE/ASCEND"
revision = args.revision
files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
receipt = {
    "dataset": repo,
    "revision": revision,
    "selection": "smallest sha256(42:id), train only",
    "splits": {},
}
for split in ("train", "validation", "test"):
    records = []
    for filename in sorted(f for f in files if f.startswith("main/" + split + "-") and f.endswith(".parquet")):
        local = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        records.extend(pq.read_table(local).to_pylist())
    if split == "train":
        records.sort(key=lambda row: hashlib.sha256(("42:" + str(row["id"])).encode()).hexdigest())
        records = records[: args.train_samples]
        assert len(records) == args.train_samples
    rows = []
    audio_dir = root / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        uid = f"ascend_{split}_{r['id']}"
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            import math

            import scipy.signal

            gcd = math.gcd(sr, 16000)
            audio = scipy.signal.resample_poly(audio, 16000 // gcd, sr // gcd)
        path = audio_dir / (uid + ".wav")
        sf.write(path, audio, 16000, subtype="PCM_16")
        prompt = "Transcribe the audio into text."
        rows.append(
            {
                "id": uid,
                "audio_filepath": str(path),
                "duration": len(audio) / 16000,
                "text": r["transcription"],
                "expected_answer": r["transcription"],
                "prompt": prompt,
                "context": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "language": r["language"],
                "speaker": str(r["original_speaker_id"]),
                "task": "ordinary_asr",
                "reward_metric": "mer",
                "responses_create_params": {"max_output_tokens": None},
            }
        )
    assert len({r["id"] for r in rows}) == len(rows)
    manifest = root / (split + ".jsonl")
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if split == "train":
        pq.write_table(pa.Table.from_pylist(rows), root / "train.parquet")
        (root / "pilot32.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:32]))
    receipt["splits"][split] = {
        "rows": len(rows),
        "hours": sum(r["duration"] for r in rows) / 3600,
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "ids": [r["id"] for r in rows],
    }
    print(split, len(rows), flush=True)
(root / "DATA_DONE.json").write_text(json.dumps(receipt, indent=2) + "\n")
