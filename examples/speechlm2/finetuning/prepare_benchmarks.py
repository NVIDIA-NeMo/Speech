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
"""Download a public audio benchmark from the Hugging Face hub and write NeMo manifests for SALM fine-tuning.

    python prepare_benchmarks.py <benchmark> [options]      # see tutorials/speechlm2/finetuning/

Benchmarks: fleurs, covost2, heysquad, slurp, cremad, speechcommands, esc50, vocalsound, gtzan
(+ gtzan_crops), nsynth, commonlanguage, ascend, edacc, speechocean, asvspoof, sep28k, and
librispeech (test-clean, the general-English guard), and ami_ihm (extra
conversational audio for EdAcc).

Audio is decoded from the parquet bytes with soundfile (which reads the true sample rate from
the file header; some mirrors declare the wrong rate) and written as 16 kHz mono FLAC under
$SALM_FT_WORK/data/<benchmark>/. Manifests go to $SALM_FT_WORK/manifests/<benchmark>/ and are
NeMo JSON (``*.json``, never ``*.jsonl``), one utterance per line with

    audio_filepath, duration, text      -- target / reference
    context                             -- the task prompt; read by ``lhotse_as_conversation`` at
                                           training time and by vllm_task_eval.py at evaluation time
    + benchmark-specific metadata

Each ``cmd_*`` documents the protocol decisions (splits, label repairs) it makes.
"""
import argparse
import collections
import csv
import io
import json
import os
import random
import re
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

SR = 16000
# Everything (audio, manifests) is written under $SALM_FT_WORK (default: ./salm_ft_work).
WORK = Path(os.environ.get("SALM_FT_WORK", "salm_ft_work")).absolute()


# ---------------------------------------------------------------- audio helpers


def _resample(x, sr):
    if sr == SR:
        return x
    import soxr

    return soxr.resample(x, sr, SR)


def _write_flac(args):
    """(bytes | path, out_path) -> duration. Runs in a worker process."""
    src, out = args
    if not os.path.exists(out):
        data, sr = sf.read(
            io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, dtype="float32", always_2d=True
        )
        x = _resample(data.mean(axis=1), sr)
        sf.write(out, x, SR, format="FLAC")
        return len(x) / SR
    return sf.info(out).duration


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects cgroup/taskset CPU limits
    except AttributeError:
        return os.cpu_count() or 1


WORKERS = min(int(os.environ.get("SALM_FT_PREP_WORKERS", 48)), available_cpus())


def write_audio(jobs, workers=None):
    with ProcessPoolExecutor(max(1, min(workers or WORKERS, available_cpus()))) as ex:
        return list(ex.map(_write_flac, jobs, chunksize=16))


def write_manifest(rows, path):
    path = Path(path)
    assert path.suffix == ".json", "NeMo manifests must end in .json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in rows) / 3600
    print(f"wrote {path}: {len(rows)} rows, {hours:.2f} h")


def parquet_rows(repo, prefix, columns=None, max_files=None):
    files = sorted(
        f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet")
    )
    if max_files:
        files = files[:max_files]
    for fn in files:
        local = hf_hub_download(repo, fn, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=columns).to_pylist():
                yield r


# ---------------------------------------------------------------- CoVoST 2 (En -> X speech translation)

LANG_NAMES = {
    "de": "German",
    "ja": "Japanese",
    "zh-CN": "Chinese",
    "tr": "Turkish",
    "ar": "Arabic",
    "fa": "Persian",
    "et": "Estonian",
    "mn": "Mongolian",
    "cy": "Welsh",
    "sl": "Slovenian",
    "lv": "Latvian",
    "ta": "Tamil",
    "id": "Indonesian",
    "ca": "Catalan",
    "sv-SE": "Swedish",
}


def st_prompt(tgt):
    return f"Translate the English speech into {LANG_NAMES[tgt]}."


def cmd_covost2(a):
    repo = "fixie-ai/covost2"
    out_dir = WORK / "data" / "covost2"
    for split, max_files, limit in (
        ("test", None, a.limit_eval),
        ("validation", None, a.limit_eval),
        ("train", a.train_shards, a.limit_train),
    ):
        if limit == 0:
            continue
        adir = out_dir / split
        adir.mkdir(parents=True, exist_ok=True)
        # Audio is identical across en_X configs; read it once from the first target.
        meta, jobs = [], []
        for r in parquet_rows(repo, f"en_{a.tgt[0]}/{split}-", max_files=max_files):
            p = str(adir / f"{r['id']}.flac")
            jobs.append((r["audio"]["bytes"], p))
            meta.append({"id": r["id"], "audio_filepath": p, "source": r["sentence"], "client_id": r["client_id"]})
            if limit and len(meta) >= limit:
                break
        durs = write_audio(jobs)
        for m, d in zip(meta, durs):
            m["duration"] = round(d, 3)
        for tgt in a.tgt:
            trans = {}
            for r in parquet_rows(repo, f"en_{tgt}/{split}-", columns=["id", "translation"], max_files=max_files):
                trans[r["id"]] = r["translation"]
            rows = [
                {**m, "text": trans[m["id"]], "context": st_prompt(tgt), "tgt_lang": tgt}
                for m in meta
                if m["id"] in trans
            ]
            write_manifest(rows, WORK / "manifests" / "covost2" / f"en_{tgt}_{split}{a.suffix}.json")


# ---------------------------------------------------------------- FLEURS (ASR in a new language)


def asr_prompt():
    return "Transcribe the audio."


def cmd_fleurs(a):
    repo = "google/fleurs"
    for lang in a.langs:
        for split in a.splits:
            adir = WORK / "data" / "fleurs" / lang / split
            adir.mkdir(parents=True, exist_ok=True)
            tsv = hf_hub_download(repo, f"data/{lang}/{split}.tsv", repo_type="dataset")
            tgz = hf_hub_download(repo, f"data/{lang}/audio/{split}.tar.gz", repo_type="dataset")
            meta = {}
            with open(tsv) as f:
                for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    # id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
                    meta[row[1]] = {"fleurs_id": row[0], "raw": row[2], "norm": row[3], "gender": row[6]}
            jobs, rows = [], []
            with tarfile.open(tgz) as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    name = os.path.basename(m.name)
                    if name not in meta:
                        continue
                    p = str(adir / name.replace(".wav", ".flac"))
                    jobs.append((tf.extractfile(m).read(), p))
                    rows.append(
                        {
                            "id": f"{lang}_{name[:-4]}",
                            "audio_filepath": p,
                            "text": meta[name]["raw"],
                            "text_norm": meta[name]["norm"],
                            "fleurs_id": meta[name]["fleurs_id"],
                            "gender": meta[name]["gender"],
                            "lang": lang,
                            "context": asr_prompt(),
                        }
                    )
                    if a.limit and len(rows) >= a.limit:
                        break
            for r, d in zip(rows, write_audio(jobs)):
                r["duration"] = round(d, 3)
            write_manifest(rows, WORK / "manifests" / "fleurs" / f"{lang}_{split}{a.suffix}.json")


# ---------------------------------------------------------------- HeySQuAD (spoken question answering)

SQA_INSTRUCTION = (
    "Answer the spoken question using the passage below. Reply with the shortest span copied "
    "verbatim from the passage. If the passage does not contain the answer, reply "
    "\"unanswerable\".\n\nPassage: "
)


def cmd_heysquad(a):
    repo = "yijingwu/HeySQuAD_human"
    for split, max_files, out_name in (("validation", None, "validation"), ("train", a.train_shards, "train_pool")):
        adir = WORK / "data" / "heysquad" / split
        adir.mkdir(parents=True, exist_ok=True)
        jobs, rows = [], []
        for i, r in enumerate(parquet_rows(repo, f"data/{split}-", max_files=max_files)):
            uid = f"{r['id']}_{i}"
            p = str(adir / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            answers = sorted({x["text"] for x in (r["answers"] or [])}) or [""]
            target = r["answers"][0]["text"] if r["answers"] else "unanswerable"
            rows.append(
                {
                    "id": uid,
                    "squad_id": r["id"],
                    "audio_filepath": p,
                    "text": target,
                    "answers": answers,
                    "question": r["question"],
                    "asr_transcript": r["transcription"],
                    "passage": r["context"],
                    "is_impossible": bool(r["is_impossible"]),
                    "context": SQA_INSTRUCTION + r["context"],
                }
            )
        for r, d in zip(rows, write_audio(jobs)):
            r["duration"] = round(d, 3)
        write_manifest(rows, WORK / "manifests" / "heysquad" / f"{out_name}.json")


# ---------------------------------------------------------------- SLURP (spoken intent classification)


def slu_prompt(labels):
    return (
        "Classify the intent of the spoken request. Answer with exactly one label from this list: "
        + ", ".join(labels)
        + "."
    )


def cmd_slurp(a):
    repo = "marcel-gohsen/slurp"
    # Do not use the `intent` field as the label. For ~1.6k utterances it holds only the
    # action (`query` for weather/calendar/news/... queries, bare `set`, `remove`, ...),
    # both in the HF mirrors and in SLURP's own jsonl, giving 91 inconsistent labels.
    # SLURP's official evaluation scores intent as f"{scenario}_{action}", so rebuild it
    # from the official annotations, joined on slurp_id
    # (github.com/pswietojanski/slurp, dataset/slurp/<split>.jsonl).
    official = {}
    odir = WORK / "data" / "slurp" / "official"
    odir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "devel", "test"):
        if not (odir / f"{split}.jsonl").exists():
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/{split}.jsonl",
                odir / f"{split}.jsonl",
            )
        with open(odir / f"{split}.jsonl") as f:
            for line in f:
                d = json.loads(line)
                official[d["slurp_id"]] = (f"{d['scenario']}_{d['action']}", {x["file"] for x in d["recordings"]})
    splits = {}
    for split in ("train", "devel", "test"):
        adir = WORK / "data" / "slurp" / split
        adir.mkdir(parents=True, exist_ok=True)
        jobs, rows = [], []
        for i, r in enumerate(parquet_rows(repo, f"data/{split}-")):
            uid = f"slurp_{split}_{i:06d}"
            p = str(adir / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            intent, recs = official[int(r["id"])]
            rec = os.path.basename((r["audio"] or {}).get("path") or "")
            assert rec in recs, (r["id"], rec)
            rows.append(
                {
                    "id": uid,
                    "slurp_id": int(r["id"]),
                    "audio_filepath": p,
                    "text": intent,
                    "mirror_intent": r["intent"],
                    "transcript": r["transcript"],
                    "scenario": intent.split("_")[0],
                    "recording": rec,
                }
            )
        for r, d in zip(rows, write_audio(jobs)):
            r["duration"] = round(d, 3)
        splits[split] = rows
    labels = sorted({r["text"] for r in splits["train"]})
    (WORK / "manifests" / "slurp").mkdir(parents=True, exist_ok=True)
    (WORK / "manifests" / "slurp" / "labels.txt").write_text("\n".join(labels) + "\n")
    prompt = slu_prompt(labels)
    for split, rows in splits.items():
        for r in rows:
            r["context"] = prompt
        write_manifest(rows, WORK / "manifests" / "slurp" / f"{split}.json")


# ---------------------------------------------------------------- IEMOCAP (speech emotion recognition)

SER_LABELS = ["neutral", "happy", "angry", "sad"]
SER_PROMPT = "What emotion does the speaker express? Answer with exactly one word: neutral, happy, angry, or sad."


def cmd_iemocap(a):
    repo = "AbstractTTS/IEMOCAP"
    adir = WORK / "data" / "iemocap"
    adir.mkdir(parents=True, exist_ok=True)
    jobs, rows = [], []
    for r in parquet_rows(repo, "data/train-"):
        emo = r["major_emotion"]
        # Standard 4-class protocol: excited is merged into happy; everything else dropped.
        emo = {"excited": "happy"}.get(emo, emo)
        if emo not in SER_LABELS:
            continue
        name = r["file"].replace(".wav", "")
        p = str(adir / f"{name}.flac")
        jobs.append((r["audio"]["bytes"], p))
        # Soft annotator agreement for the (merged) label: IEMOCAP's canonical 4-class set
        # keeps only utterances with a majority vote, which this mirror does not mark.
        agree = r["happy"] + r["excited"] if emo == "happy" else r[emo]
        rows.append(
            {
                "id": name,
                "audio_filepath": p,
                "text": emo,
                "agree": round(agree, 3),
                "session": int(name[3:5]),
                "dialog": name.rsplit("_", 1)[0],
                "gender": r["gender"],
                "transcript": r["transcription"].strip(),
                "context": SER_PROMPT,
            }
        )
    for r, d in zip(rows, write_audio(jobs)):
        r["duration"] = round(d, 3)
    (WORK / "manifests" / "iemocap").mkdir(parents=True, exist_ok=True)
    (WORK / "manifests" / "iemocap" / "labels.txt").write_text("\n".join(SER_LABELS) + "\n")
    # Speaker-independent split: Session 5 is test (the common single-fold protocol);
    # half of Session 4's dialogs are dev; the rest trains.
    s4 = sorted({r["dialog"] for r in rows if r["session"] == 4})
    rng = random.Random(0)
    dev_dialogs = set(rng.sample(s4, len(s4) // 2))
    # Keep utterances whose label has majority soft agreement (> 0.5). IEMOCAP's canonical
    # 4-class set only includes majority-vote utterances; this mirror's `major_emotion` is an
    # argmax that also covers no-majority ones. With the filter Session 5 has 1,312
    # utterances vs 1,241 in the canonical set; without it, 1,507.
    rows = [r for r in rows if r["agree"] > 0.5]
    test = [r for r in rows if r["session"] == 5]
    dev = [r for r in rows if r["dialog"] in dev_dialogs]
    train = [r for r in rows if r["session"] < 5 and r["dialog"] not in dev_dialogs]
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        write_manifest(part, WORK / "manifests" / "iemocap" / f"{name}.json")


# ---------------------------------------------------------------- CREMA-D (acted speech emotion, 6 classes)

CREMAD_LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
CREMAD_PROMPT = (
    "What emotion does the speaker express? Answer with exactly one word: "
    "anger, disgust, fear, happy, neutral, or sad."
)


def cmd_cremad(a):
    """CREMA-D: 91 actors each read the same 12 sentences in six emotions, so the words carry
    no emotional information and the label must come from the acoustics. The mirror's
    train/validation/test split is random over clips (all 91 actors appear in all three
    splits); we re-split by actor so that test measures unseen speakers."""
    repo = "confit/cremad-parquet"
    adir = WORK / "data" / "cremad"
    adir.mkdir(parents=True, exist_ok=True)
    jobs, rows = [], []
    for split in ("train", "validation", "test"):
        for r in parquet_rows(repo, f"data/{split}-"):
            name = os.path.basename(r["file"]).replace(".wav", "")
            actor, sentence, _, level = name.split("_")
            p = str(adir / f"{name}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append(
                {
                    "id": name,
                    "audio_filepath": p,
                    "text": r["emotion"],
                    "actor": actor,
                    "sentence": sentence,
                    "intensity": level,
                    "context": CREMAD_PROMPT,
                }
            )
    for r, d in zip(rows, write_audio(jobs)):
        r["duration"] = round(d, 3)
    actors = sorted({r["actor"] for r in rows})
    rng = random.Random(0)
    rng.shuffle(actors)
    test_a, dev_a = set(actors[:14]), set(actors[14:23])
    out = WORK / "manifests" / "cremad"
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.txt").write_text("\n".join(CREMAD_LABELS) + "\n")
    (out / "split_actors.json").write_text(json.dumps({"test": sorted(test_a), "dev": sorted(dev_a)}, indent=1))
    write_manifest([r for r in rows if r["actor"] in test_a], out / "test.json")
    write_manifest([r for r in rows if r["actor"] in dev_a], out / "dev.json")
    write_manifest([r for r in rows if r["actor"] not in test_a | dev_a], out / "train.json")


def out_dir(name):
    d = WORK / "manifests" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def adir(name, split):
    d = WORK / "data" / name / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def cls_prompt(question, labels):
    return f"{question} Answer with exactly one label from this list: " + ", ".join(labels) + "."


def finish(rows, jobs):
    for r, d in zip(rows, write_audio(jobs)):
        r["duration"] = round(d, 3)
    return rows


def split_by_group(rows, key, frac, seed=0):
    groups = sorted({r[key] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    held = set(groups[: max(1, round(len(groups) * frac))])
    return [r for r in rows if r[key] not in held], [r for r in rows if r[key] in held]


def save_labels(name, labels):
    (out_dir(name) / "labels.txt").write_text("\n".join(labels) + "\n")


# ---------------------------------------------------------------- keyword spotting


def cmd_speechcommands(a):
    repo, name = "pollen-robotics/speech-commands-v0.02", "speechcommands"
    parts = {}
    rng = random.Random(0)
    for split in ("train", "validation", "test"):
        rows, jobs = [], []
        # The shards are SORTED BY LABEL: a prefix of the train shards holds only 10 of the 35
        # keywords (this trained a model that answers "one" to everything). Read every shard
        # and draw a uniform random subset instead.
        for r in parquet_rows(repo, f"data/{split}/"):
            if r["label"].startswith("_"):  # 6 long _background_noise_ files in train
                continue
            if split == "train" and rng.random() > a.train_frac:
                continue
            uid = r["file"].replace("/", "__").replace(".wav", "")
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append({"id": uid, "audio_filepath": p, "text": r["label"], "speaker": r["speaker_id"]})
        parts[split] = finish(rows, jobs)
    missing = {r["text"] for r in parts["test"]} - {r["text"] for r in parts["train"]}
    assert not missing, f"test labels absent from train: {missing}"
    print("train label counts:", collections.Counter(r["text"] for r in parts["train"]).most_common())
    labels = sorted({r["text"] for r in parts["test"]})
    prompt = cls_prompt("Which keyword is spoken?", labels)
    for rows in parts.values():
        for r in rows:
            r["context"] = prompt
    save_labels(name, labels)
    random.Random(0).shuffle(parts["validation"])
    write_manifest(parts["train"], out_dir(name) / "train.json")
    write_manifest(parts["validation"][:2000], out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")


# ---------------------------------------------------------------- environmental sound


def cmd_esc50(a):
    repo, name = "ashraq/esc50", "esc50"
    rows, jobs = [], []
    for r in parquet_rows(repo, "data/train-"):
        uid = r["filename"].replace(".wav", "")
        p = str(adir(name, "all") / f"{uid}.flac")
        jobs.append((r["audio"]["bytes"], p))
        rows.append(
            {
                "id": uid,
                "audio_filepath": p,
                "text": r["category"].replace("_", " "),
                "fold": r["fold"],
                "src_file": r["src_file"],
            }
        )
    finish(rows, jobs)
    labels = sorted({r["text"] for r in rows})
    prompt = cls_prompt("Which sound event is heard in this recording?", labels)
    for r in rows:
        r["context"] = prompt
    save_labels(name, labels)
    # Official folds keep clips of one source recording together. Folds 1-3 train,
    # fold 4 dev, fold 5 test (a single fold of the standard 5-fold protocol).
    write_manifest([r for r in rows if r["fold"] <= 3], out_dir(name) / "train.json")
    write_manifest([r for r in rows if r["fold"] == 4], out_dir(name) / "dev.json")
    write_manifest([r for r in rows if r["fold"] == 5], out_dir(name) / "test.json")


# ---------------------------------------------------------------- non-verbal vocalisations


def cmd_vocalsound(a):
    repo, name = "lmms-lab-audio/vocalsound", "vocalsound"
    parts = {}
    for split in ("val", "test"):
        rows, jobs = [], []
        for i, r in enumerate(parquet_rows(repo, f"data/{split}-")):
            uid = f"{split}_{i:05d}_{r['spk_id']}"
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append({"id": uid, "audio_filepath": p, "text": r["answer"].lower(), "speaker": r["spk_id"]})
        parts[split] = finish(rows, jobs)
    labels = sorted({r["text"] for r in parts["test"]})
    prompt = cls_prompt("Which non-speech vocal sound does the person make?", labels)
    for rows in parts.values():
        for r in rows:
            r["context"] = prompt
    save_labels(name, labels)
    # No train split is published as parquet: the official validation set (300 speakers,
    # speaker-disjoint from test) is the training pool; 15% of its speakers are held out as dev.
    train, dev = split_by_group(parts["val"], "speaker", 0.15)
    write_manifest(train, out_dir(name) / "train.json")
    write_manifest(dev, out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")


# ---------------------------------------------------------------- music genre


def cmd_gtzan(a):
    repo, name = "confit/gtzan-parquet", "gtzan"
    parts = {}
    for split in ("train", "validation", "test"):
        rows, jobs = [], []
        for i, r in enumerate(parquet_rows(repo, f"data/{split}-")):
            uid = f"{split}_{i:04d}_{r['genre']}"
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append({"id": uid, "audio_filepath": p, "text": r["genre"]})
        parts[split] = finish(rows, jobs)
    labels = sorted({r["text"] for r in parts["train"]})
    prompt = cls_prompt("What is the genre of this music?", labels)
    for rows in parts.values():
        for r in rows:
            r["context"] = prompt
    save_labels(name, labels)
    write_manifest(parts["train"], out_dir(name) / "train.json")
    write_manifest(parts["validation"], out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")


# ---------------------------------------------------------------- language identification

CL_NAMES = [
    'Arabic',
    'Basque',
    'Breton',
    'Catalan',
    'Chinese_China',
    'Chinese_Hongkong',
    'Chinese_Taiwan',
    'Chuvash',
    'Czech',
    'Dhivehi',
    'Dutch',
    'English',
    'Esperanto',
    'Estonian',
    'French',
    'Frisian',
    'Georgian',
    'German',
    'Greek',
    'Hakha_Chin',
    'Indonesian',
    'Interlingua',
    'Italian',
    'Japanese',
    'Kabyle',
    'Kinyarwanda',
    'Kyrgyz',
    'Latvian',
    'Maltese',
    'Mangolian',
    'Persian',
    'Polish',
    'Portuguese',
    'Romanian',
    'Romansh_Sursilvan',
    'Russian',
    'Sakha',
    'Slovenian',
    'Spanish',
    'Swedish',
    'Tamil',
    'Tatar',
    'Turkish',
    'Ukranian',
    'Welsh',
]
# Readable, correctly spelled label strings (the dataset's `Mangolian`/`Ukranian` typos and
# underscores would otherwise penalise a model for spelling the language correctly).
CL_FIX = {
    "Chinese_China": "Chinese Mainland",
    "Chinese_Hongkong": "Chinese Hong Kong",
    "Chinese_Taiwan": "Chinese Taiwan",
    "Hakha_Chin": "Hakha Chin",
    "Mangolian": "Mongolian",
    "Romansh_Sursilvan": "Romansh Sursilvan",
    "Ukranian": "Ukrainian",
}


def cmd_commonlanguage(a):
    repo, name = "regisss/common_language", "commonlanguage"
    labels = sorted(CL_FIX.get(n, n) for n in CL_NAMES)
    prompt = cls_prompt("Which language is spoken?", labels)
    parts = {}
    for split in ("train", "validation", "test"):
        rows, jobs = [], []
        for i, r in enumerate(parquet_rows(repo, f"data/{split}-")):
            uid = f"{split}_{i:05d}"
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            lang = CL_NAMES[r["language"]]
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "text": CL_FIX.get(lang, lang),
                    "speaker": r["client_id"],
                    "sentence": r["sentence"],
                    "context": prompt,
                }
            )
        parts[split] = finish(rows, jobs)
    save_labels(name, labels)
    random.Random(0).shuffle(parts["validation"])
    write_manifest(parts["train"], out_dir(name) / "train.json")
    write_manifest(parts["validation"][:2500], out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")


# ---------------------------------------------------------------- code-switching ASR


def cmd_ascend(a):
    repo, name = "CAiRE/ASCEND", "ascend"
    for split in ("train", "validation", "test"):
        rows, jobs = [], []
        for r in parquet_rows(repo, f"main/{split}-"):
            uid = f"ascend_{split}_{r['id']}"
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "text": r["transcription"],
                    "language": r["language"],
                    "speaker": r["original_speaker_id"],
                    "context": "Transcribe the audio.",
                }
            )
        finish(rows, jobs)
        write_manifest(rows, out_dir(name) / f"{ {'validation': 'dev'}.get(split, split)}.json")


# ---------------------------------------------------------------- accented conversational English ASR

_TAG = re.compile(r"<[^>]*>")


def cmd_edacc(a):
    repo, name = "edinburghcstr/edacc", "edacc"
    parts = {}
    for split in ("validation", "test"):
        rows, jobs = [], []
        for i, r in enumerate(parquet_rows(repo, f"data/{split}-")):
            uid = f"edacc_{split}_{i:05d}"
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            clean = re.sub(r"\s+", " ", _TAG.sub(" ", r["text"])).strip()
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "text": clean,
                    "raw_text": r["text"],
                    "speaker": r["speaker"],
                    "conversation": r["speaker"].rsplit("-", 1)[0],
                    "accent": r["accent"],
                    "l1": r["l1"],
                    "context": "Provide a verbatim transcript of the audio.",
                }
            )
        # Tag-only segments have no words to score. IGNORE_TIME_SEGMENT_IN_SCORING marks the read-aloud passages;
        # by the NIST convention (also used by ESB) they are excluded from training and scoring. Kept, they are
        # scored as one-word references against ~80-word hypotheses, and a fine-tune that learns to emit the
        # marker literally gets a large, fake WER reduction.
        parts[split] = [
            x for x in finish(rows, jobs) if x["text"] and "IGNORE_TIME_SEGMENT_IN_SCORING" not in x["text"]
        ]
    # No train split: the validation set (speaker-disjoint from test) is the training pool,
    # with 15% of its conversations held out as dev.
    train, dev = split_by_group(parts["validation"], "conversation", 0.15)
    # Training targets: lower-cased (the references are all upper case; see SKILL.md on style),
    # and only segments the encoder sees in one window with a sane length.
    # v1 used a 0.5 s floor, which removed the backchannels ("mm hmm", "yeah") that dominate
    # EdAcc's test set includes very short segments; keep everything from 0.05 s.
    train = [dict(r, text=r["text"].lower()) for r in train if a.min_dur <= r["duration"] <= 30]
    random.Random(0).shuffle(train)
    if a.train_out:
        write_manifest(train, out_dir(name) / a.train_out)
        return
    write_manifest(train, out_dir(name) / "train.json")
    write_manifest(dev, out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")


# ---------------------------------------------------------------- pronunciation assessment

SO_PROMPT = (
    "Rate this non-native English speaker's pronunciation of the sentence \"{sentence}\" on four 0-10 scales: "
    "accuracy, fluency, prosody and total. Reply exactly in the form: "
    "accuracy X, fluency X, prosodic X, total X"
)


def cmd_speechocean(a):
    repo, name = "mispeech/speechocean762", "speechocean"
    parts = {}
    for split in ("train", "test"):
        rows, jobs = [], []
        for i, r in enumerate(parquet_rows(repo, f"data/{split}-")):
            uid = f"so_{split}_{i:05d}"
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            sc = {k: int(r[k]) for k in ("accuracy", "fluency", "prosodic", "total")}
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "sentence": r["text"],
                    "speaker": r["speaker"],
                    "scores": sc,
                    "text": ", ".join(f"{k} {v}" for k, v in sc.items()),
                    "context": SO_PROMPT.format(sentence=r["text"].capitalize()),
                }
            )
        parts[split] = finish(rows, jobs)
    train, dev = split_by_group(parts["train"], "speaker", 0.12)
    write_manifest(train, out_dir(name) / "train.json")
    write_manifest(dev, out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")


# ---------------------------------------------------------------- anti-spoofing

AS_PROMPT = (
    "Is this speech recording genuine human speech or a synthetic/converted spoof? "
    "Answer with one word: bonafide or spoof."
)


def cmd_asvspoof(a):
    repo, name = "Bisher/ASVspoof_2019_LA", "asvspoof"
    parts = {}
    for split in ("train", "validation", "test"):
        rows, jobs = [], []
        for r in parquet_rows(repo, f"data/{split}-"):
            uid = r["audio_file_name"].replace(".flac", "")
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "text": ["bonafide", "spoof"][r["key"]],
                    "speaker": r["speaker_id"],
                    "system_id": r["system_id"],
                    "context": AS_PROMPT,
                }
            )
        parts[split] = finish(rows, jobs)
    save_labels(name, ["bonafide", "spoof"])
    dev = parts["validation"][:]
    random.Random(0).shuffle(dev)  # the file is sorted by class
    write_manifest(parts["train"], out_dir(name) / "train.json")
    write_manifest(dev[:3000], out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")


# ---------------------------------------------------------------- stuttering

SEP_EVENTS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]
SEP_PROMPT = (
    "Which stuttering events occur in this clip? Choose any of: prolongation, block, sound repetition, "
    "word repetition, interjection. List all that occur, separated by commas, or answer none."
)
SEP_NAMES = {
    "Prolongation": "prolongation",
    "Block": "block",
    "SoundRep": "sound repetition",
    "WordRep": "word repetition",
    "Interjection": "interjection",
}


def cmd_sep28k(a):
    repo, name = "psusac/sep28k", "sep28k"
    parts = {}
    for split in ("train", "dev", "test"):
        rows, jobs = [], []
        for i, r in enumerate(parquet_rows(repo, f"SEP28k-E/{split}-")):
            uid = f"sep_{split}_{i:05d}"
            # Standard binarisation: an event is present if >= 2 of 3 annotators marked it.
            ev = [SEP_NAMES[e] for e in SEP_EVENTS if r[e] >= 2]
            if r["PoorAudioQuality"] >= 2 or r["Music"] >= 2 or r["NoSpeech"] >= 2 or r["Unsure"] >= 2:
                continue
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "text": ", ".join(ev) if ev else "none",
                    "events": ev,
                    "speaker": r["speaker"],
                    "context": SEP_PROMPT,
                }
            )
        parts[split] = finish(rows, jobs)
    # SEP28k-E train holds only 3 speakers (podcast hosts); its dev split (141 speakers) is the
    # diverse pool. Train = E-train + 85% of E-dev speakers; dev = the other 15%; test = E-test.
    tr_extra, dev = split_by_group(parts["dev"], "speaker", 0.15)
    write_manifest(parts["train"] + tr_extra, out_dir(name) / "train.json")
    write_manifest(dev, out_dir(name) / "dev.json")
    write_manifest(parts["test"], out_dir(name) / "test.json")
    print(
        "event counts (test):",
        collections.Counter(e for r in parts["test"] for e in r["events"]),
        "none:",
        sum(not r["events"] for r in parts["test"]),
    )


# ---------------------------------------------------------------- GTZAN v2: 10 s crops (training only)


def cmd_gtzan_crops(a):
    """Add three 10 s crops (offsets 0/10/20 s) of every training clip to the full clips: 4x the
    examples for a 443-clip training set. Pure manifest offsets; dev/test stay full 30 s clips."""
    d = out_dir("gtzan")
    with open(d / "train_ft.json") as f:
        rows = [json.loads(line) for line in f]
    out = []
    for r in rows:
        out.append(r)
        for k, off in enumerate((0.0, 10.0, 20.0)):
            if r["duration"] >= off + 9.5:
                out.append(dict(r, id=f"{r['id']}_c{k}", offset=off, duration=min(10.0, r["duration"] - off)))
    write_manifest(out, d / "train_crops_ft.json")


# ---------------------------------------------------------------- NSynth instrument family

NS_LABELS = [
    "bass",
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "organ",
    "reed",
    "string",
    "synth lead",
    "vocal",
]


def cmd_nsynth(a):
    """confit/nsynth-parquet `instrument` config. Train: random row groups from 30 random shards
    (~6k notes; shards may be ordered, so no contiguous prefix). Dev: 1,500 of validation. Test: official 4,096."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, list_repo_files

    repo, name = "confit/nsynth-parquet", "nsynth"
    fs = HfFileSystem()
    files = sorted(
        f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("instrument/") and f.endswith(".parquet")
    )
    rng = random.Random(0)
    prompt = cls_prompt("Which instrument family plays this note?", NS_LABELS)

    def take(split, nfiles, rgs_per_file, limit=None):
        fl = [f for f in files if f.startswith(f"instrument/{split}-")]
        if nfiles:
            fl = rng.sample(fl, min(nfiles, len(fl)))
        rows, jobs = [], []
        for f in fl:
            pf = pq.ParquetFile(fs.open(f"datasets/{repo}/{f}"))
            rgs = (
                range(pf.num_row_groups)
                if not rgs_per_file
                else rng.sample(range(pf.num_row_groups), min(rgs_per_file, pf.num_row_groups))
            )
            for g in rgs:
                for i, r in enumerate(pf.read_row_group(g).to_pylist()):
                    uid = f"ns_{split}_{f.split('-')[1]}_{g}_{i}"
                    p = str(adir(name, split) / f"{uid}.flac")
                    jobs.append((r["audio"]["bytes"], p))
                    rows.append(
                        {"id": uid, "audio_filepath": p, "text": r["instrument"].replace("_", " "), "context": prompt}
                    )
        return finish(rows, jobs)

    train = take("train", 30, 2)
    val = take("validation", None, None)
    test = take("test", None, None)
    rng.shuffle(val)
    save_labels(name, NS_LABELS)
    print("train classes:", collections.Counter(r["text"] for r in train))
    write_manifest(train, out_dir(name) / "train.json")
    write_manifest(val[:1500], out_dir(name) / "dev.json")
    write_manifest(test, out_dir(name) / "test.json")


# ---------------------------------------------------------------- Eka Care medical ASR (test-only benchmark)


def cmd_eka(a):
    """Eka Care Medical ASR Evaluation (en): 3,619 recordings of Indian-accented medical speech (drug names, findings,
    dictations, conversations). Test-only: there is no training split, so it must never be used for training or
    checkpoint selection. `medical_entities` (entity strings with types and offsets) is kept for keyword WER."""
    repo, name = "ekacare/eka-medical-asr-evaluation-dataset", "eka"
    rows, jobs = [], []
    for i, r in enumerate(parquet_rows(repo, "en/test-")):
        uid = f"eka_{i:05d}"
        p = str(adir(name, "test") / f"{uid}.flac")
        audio = r["audio"]  # this dataset stores raw bytes, not an {"bytes": ...} struct
        jobs.append((audio["bytes"] if isinstance(audio, dict) else audio, p))
        rows.append(
            {
                "id": uid,
                "audio_filepath": p,
                "text": r["text"].strip(),
                "medical_entities": r["medical_entities"],
                "recording_context": r["recording_context"],
                "type_concept": r["type_concept"],
                "speaker": r["speaker"],
                "session_id": r["session_id"],
                "context": "Provide a verbatim transcript of the audio.",
            }
        )
    write_manifest(finish(rows, jobs), out_dir(name) / "test.json")


# ---------------------------------------------------------------- real speech for medical adaptation


def cmd_cv_indian(a):
    """Common Voice English clips with an Indian / South Asian accent tag (CC0): real accented read speech. Samples
    `--hours` uniformly across all shards (never a shard prefix)."""
    repo, name = "ishands/commonvoice-indian_accent", "cv_indian"
    rng = random.Random(0)
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    rows, jobs, total = [], [], 0.0
    budget = a.hours * 3600 / len(files)
    for fn in files:
        t = pq.read_table(hf_hub_download(repo, fn, repo_type="dataset")).to_pylist()
        rng.shuffle(t)
        got = 0.0
        for r in t:
            dur = int(r["duration_ms"]) / 1000
            if got >= budget or not r["sentence"].strip() or int(r["down_votes"]) > 0:
                continue
            uid = f"cvin_{len(rows):06d}"
            p = str(adir(name, "train") / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "text": r["sentence"].strip(),
                    "speaker": r["client_id"][:16],
                    "context": "Provide a verbatim transcript of the audio.",
                }
            )
            got += dur
        total += got
    rows = finish(rows, jobs)
    rng.shuffle(rows)
    write_manifest(rows, out_dir(name) / "train.json")


def cmd_multimed_en(a):
    """MultiMed English (MIT): real medical speech (lectures, interviews, podcasts). Train and eval splits."""
    repo, name = "leduckhai/MultiMed", "multimed_en"
    for split in ("train", "eval"):
        rows, jobs = [], []
        for i, r in enumerate(parquet_rows(repo, f"English/{split}-")):
            if not (r.get("text") or "").strip():
                continue
            uid = f"mmen_{split}_{i:06d}"
            p = str(adir(name, split) / f"{uid}.flac")
            jobs.append((r["audio"]["bytes"], p))
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": p,
                    "text": r["text"].strip(),
                    "context": "Provide a verbatim transcript of the audio.",
                }
            )
        rows = [x for x in finish(rows, jobs) if 0.3 <= x["duration"] <= 30]
        random.Random(0).shuffle(rows)
        write_manifest(rows, out_dir(name) / f"{split}.json")


# ---------------------------------------------------------------- LibriSpeech test-clean (general-domain guard)


def cmd_librispeech(a):
    """OpenSLR LibriSpeech test-clean: 2,620 utterances of read English. Every fine-tuned model is decoded on
    it with the standard transcription prompt, to show whether ordinary English ASR survived the fine-tune."""
    root = WORK / "data" / "librispeech"
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "test-clean.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve("https://www.openslr.org/resources/12/test-clean.tar.gz", tgz)
    if not (root / "LibriSpeech" / "test-clean").is_dir():
        with tarfile.open(tgz) as tf:
            tf.extractall(root, filter="data")
    rows = []
    for trans in sorted((root / "LibriSpeech" / "test-clean").rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            flac = trans.parent / f"{uid}.flac"
            rows.append(
                {
                    "id": uid,
                    "audio_filepath": str(flac),
                    "duration": round(sf.info(str(flac)).duration, 3),
                    "text": text.lower(),
                    "context": "Provide a verbatim transcript of the audio.",
                }
            )
    write_manifest(rows, out_dir("librispeech") / "test_clean.json")


# ---------------------------------------------------------------- AMI IHM (extra conversational data for EdAcc)


def cmd_ami_ihm(a):
    """20 h of AMI close-talk meeting speech (many non-native speakers), styled like the EdAcc training targets
    (lower case, no dotted letter spellings) and shuffled. Written as manifests/edacc/ami_ihm_train_ft.json for
    blending into EdAcc training (tutorial 13)."""
    repo = "edinburghcstr/ami"
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("ihm/") and "/train-" in f)[
        :12
    ]
    root = WORK / "data" / "ami_ihm"
    rows, total = [], 0.0
    for fn in files:
        for r in pq.read_table(hf_hub_download(repo, fn, repo_type="dataset")).to_pylist():
            text = (r.get("text") or "").strip()
            if r.get("audio") is None or not text:
                continue
            data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
            data = data.mean(axis=1)
            dur = len(data) / sr
            if not 0.3 <= dur <= 30.0:
                continue
            seg = r.get("segment_id") or f"ami_{len(rows)}"
            (root / seg[:3]).mkdir(parents=True, exist_ok=True)
            p = root / seg[:3] / f"{seg}.flac"
            if not p.exists():
                sf.write(str(p), _resample(data, sr), SR, format="FLAC")
            t = re.sub(r"\s+", " ", text.lower().replace(".", "")).strip()
            if t:
                rows.append(
                    {
                        "id": seg,
                        "audio_filepath": str(p),
                        "duration": round(dur, 3),
                        "text": t,
                        "meeting_id": r.get("meeting_id"),
                        "speaker_id": r.get("speaker_id"),
                        "context": "Provide a verbatim transcript of the audio.",
                    }
                )
            total += dur
            if total >= 20 * 3600:
                break
        if total >= 20 * 3600:
            break
    random.Random(0).shuffle(rows)
    write_manifest(rows, out_dir("edacc") / "ami_ihm_train_ft.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=sorted(n[4:] for n in globals() if n.startswith("cmd_")))
    p.add_argument("--tgt", nargs="+", default=["de"], help="covost2: target languages")
    p.add_argument("--langs", nargs="+", default=None, help="fleurs: languages, e.g. sw_ke")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="fleurs: splits")
    p.add_argument("--limit", type=int, default=None, help="fleurs: max utterances per split")
    p.add_argument("--limit-train", type=int, default=None, help="covost2: max train utterances")
    p.add_argument("--limit-eval", type=int, default=None, help="covost2: max dev/test utterances")
    p.add_argument("--suffix", default="", help="suffix for manifest names")
    p.add_argument("--train-shards", type=int, default=None, help="covost2/heysquad: number of train shards")
    p.add_argument("--train-frac", type=float, default=0.36, help="speechcommands: random share of train")
    p.add_argument("--hours", type=float, default=30.0, help="cv_indian: hours to sample")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="audio-writing processes (default: $SALM_FT_PREP_WORKERS or 48, capped by available CPUs)",
    )
    p.add_argument("--min-dur", type=float, default=0.5, help="edacc: shortest training segment (s)")
    p.add_argument("--train-out", default=None, help="edacc: write only the training manifest, under this name")
    a = p.parse_args()
    if a.workers is not None:
        if a.workers <= 0:
            p.error("--workers must be positive")
        global WORKERS
        WORKERS = min(a.workers, available_cpus())
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
