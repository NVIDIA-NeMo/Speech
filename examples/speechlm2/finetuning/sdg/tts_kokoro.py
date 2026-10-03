# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kokoro-82M synthesis of a text jsonl into 16 kHz FLAC + NeMo manifest.

Requires `pip install kokoro "misaki[en]"` (plus the spaCy `en_core_web_sm` model) in an environment with torch.
The TTS input is the row's `spoken` field when present (how the text is said aloud); the label stays `text`.
Voices: English G2P ('a' American or 'b' British) with voices from several Kokoro languages, for accent variety.
"""
import argparse
import json
import os
import random

import numpy as np
import soundfile as sf

VOICES_A = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "af_nova",
    "af_river",
    "af_kore",
    "af_aoede",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_fenrir",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",  # Hindi voices speaking English
    "ef_dora",
    "em_alex",
    "if_sara",
    "im_nicola",
    "pf_dora",
    "pm_alex",
    "ff_siwis",
]
VOICES_B = ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--voices", default=None, help="comma list; default: all (american+british pipelines)")
    p.add_argument("--exclude-voices", default="", help="comma list of voices to hold out (e.g. for a dev set)")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    import librosa
    import torch
    from kokoro import KModel, KPipeline

    si, sn = map(int, a.shard.split("/"))
    rows = [json.loads(line) for line in open(a.texts)][si::sn]
    rows = rows[: a.limit] if a.limit else rows
    excl = set(filter(None, a.exclude_voices.split(",")))
    voices = [v for v in (a.voices.split(",") if a.voices else VOICES_A + VOICES_B) if v not in excl]
    rng = random.Random(a.seed + si)
    a.out_dir = os.path.abspath(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda").eval()
    pipes = {k: KPipeline(lang_code=k, repo_id="hexgrad/Kokoro-82M", model=model) for k in ("a", "b")}
    out = []
    with torch.inference_mode():
        for i, r in enumerate(rows):
            v = voices[rng.randrange(len(voices))]
            pipe = pipes["b" if v[0] == "b" else "a"]
            speed = rng.uniform(0.9, 1.15)
            uid = f"{os.path.basename(a.out_dir)}_{si}_{i:06d}"
            path = os.path.join(a.out_dir, uid + ".flac")
            if os.path.exists(path):  # resume: audio from an interrupted run (same seed -> same voice)
                try:
                    dur = sf.info(path).duration
                except RuntimeError:
                    dur = 0.0
                if dur > 0:
                    out.append(
                        {**r, "id": uid, "audio_filepath": path, "duration": round(dur, 3), "voice": f"kokoro:{v}"}
                    )
                    continue
            chunks = [
                res.audio.cpu().numpy()
                for res in pipe(r.get("spoken") or r["text"], voice=v, speed=speed)
                if res.audio is not None
            ]
            if not chunks:
                continue
            x = librosa.resample(np.concatenate(chunks), orig_sr=24000, target_sr=16000)
            sf.write(path, x, 16000)
            out.append(
                {**r, "id": uid, "audio_filepath": path, "duration": round(len(x) / 16000, 3), "voice": f"kokoro:{v}"}
            )
            if i % 500 == 0:
                print(f"[kokoro] {i}/{len(rows)}", flush=True)
    with open(a.manifest, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[kokoro] wrote {a.manifest}: {len(out)} rows, {sum(r['duration'] for r in out) / 3600:.2f} h")


if __name__ == "__main__":
    main()
