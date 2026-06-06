#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Multi-GPU EasyMagpieTTS / NemotronTTS multiturn inference evaluation.

Key behavior:
  - Uses torchrun env vars RANK, LOCAL_RANK, WORLD_SIZE for sharding/GPU assignment.
  - Does NOT initialize torch.distributed. This avoids NeMo ASR doing distributed
    collectives during metric computation.
  - Generation runs first for all assigned samples.
  - ASR and SECS are loaded only after generation is done and the TTS/codec model
    has been deleted from GPU memory.
  - ASR and SECS are loaded sequentially: ASR first, then released; SECS second.
  - For --profile_multiturn_inference, metrics are computed turn-by-turn.
    Final filewise outputs are grouped back to one row per original sample, with
    lists for asr_hyp/reference_text/cer_turns/wer_turns/secs_turns.
  - Uses DistributedSampler with explicit rank/world_size. A few repeated samples
    may appear when len(dataset) is not divisible by world_size. Filewise final
    metrics deduplicate sampler-padding repeats by (run_id, dataset_index,
    turn_id), then group turns into one row per sample with metric lists, while
    preserving --num_eval_runs repetitions.
  - --sort_by_text_token_count sorts samples by total text-token count before
    sharding to improve GPU load balance.
  - Saves audio in out_dir/audios/.
  - Saves metrics in out_dir/.

Recommended single-node torchrun:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --standalone --nproc_per_node=8 easy_magpietts_inference_multiturn_multigpu_postgen_metrics.py ...

Recommended single-node srun wrapper:
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --container-image=... \
    bash -lc 'torchrun --standalone --nproc_per_node=8 easy_magpietts_inference_multiturn_multigpu_postgen_metrics.py ...'
"""

import argparse
import csv
import json
import os
import socket
import time
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Iterable, List, Tuple

import librosa
import soundfile as sf
import torch
from omegaconf import open_dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from nemo.collections.tts.parts.utils.tts_dataset_utils import normalize_volume
from nemo.utils import logging
from whisper_normalizer.english import EnglishTextNormalizer


torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


# -----------------------------
# Rank / file helpers
# -----------------------------


def get_rank_info() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    distributed = world_size > 1
    return distributed, rank, local_rank, world_size


def get_visible_device_index(local_rank: int) -> int:
    if not torch.cuda.is_available():
        return -1
    ndev = torch.cuda.device_count()
    if ndev <= 0:
        return -1
    return local_rank % ndev


def setup_distributed():
    """
    Do not initialize torch.distributed.

    We only need RANK/LOCAL_RANK/WORLD_SIZE for rank assignment and dataset
    sharding. Initializing a process group can cause NeMo ASR to run distributed
    collectives during transcribe(), which may hang when ranks have different
    audio lengths or workloads.
    """
    distributed, rank, local_rank, world_size = get_rank_info()
    device_index = get_visible_device_index(local_rank)

    if torch.cuda.is_available() and device_index >= 0:
        torch.cuda.set_device(device_index)

    return distributed, rank, local_rank, world_size, device_index


def cleanup_distributed():
    return


def all_rank_print(rank: int, msg: str):
    print(f"[rank={rank}] {msg}", flush=True)


def rank0_print(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def get_audio_out_dir(args) -> str:
    return os.path.join(args.out_dir, "audios")


def get_generated_turn_audio_dir(args) -> str:
    return os.path.join(get_audio_out_dir(args), "metric_turns")


def get_context_metric_audio_dir(args) -> str:
    return os.path.join(get_audio_out_dir(args), "metric_context")


def write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_text_atomic(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def wait_for_files(paths: List[str], timeout_sec: float = 7200.0, poll_sec: float = 5.0):
    start = time.time()
    while True:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            return
        if time.time() - start > timeout_sec:
            raise TimeoutError("Timed out waiting for files:\n" + "\n".join(missing))
        time.sleep(poll_sec)


def wait_for_rank_metric_files(args, world_size: int):
    paths = [os.path.join(args.out_dir, f"metrics_rank{r:04d}.json") for r in range(world_size)]
    wait_for_files(paths)


def wait_for_rank_filewise_metric_files(args, world_size: int):
    paths = [os.path.join(args.out_dir, f"filewise_metrics_rank{r:04d}.jsonl") for r in range(world_size)]
    wait_for_files(paths)


def scalarize_metric_value(v: Any):
    if torch.is_tensor(v):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return v.detach().cpu().tolist()
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return float(v.item())
    except Exception:
        pass
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def metric_dict_to_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): scalarize_metric_value(v) for k, v in d.items()}


def safe_metric_scalar(metric_dict: Dict[str, Any], preferred_keys: List[str]):
    for key in preferred_keys:
        if key in metric_dict:
            value = metric_dict[key]
            if torch.is_tensor(value):
                return float(value.detach().cpu().item())
            return float(value)
    return None


def get_first_metric(metrics: Dict[str, Any], names: List[str], default=None):
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


def format_final_metric_text(final_metrics: Dict[str, Any]) -> str:
    intelligibility = final_metrics.get("intelligibility", {})
    secs = final_metrics.get("secs", {})

    cer = get_first_metric(intelligibility, ["cer", "cer_dataset"])
    wer = get_first_metric(intelligibility, ["wer", "wer_dataset"])
    secs_value = get_first_metric(secs, ["secs", "secs_dataset"])

    def fmt(x):
        if x is None:
            return "nan"
        try:
            return f"{float(x):.10f}"
        except Exception:
            return str(x)

    return f"Average CER: {fmt(cer)}\nAverage WER: {fmt(wer)}\nSECS: {fmt(secs_value)}\n"


def format_filewise_final_metric_text(filewise_summary: Dict[str, Any]) -> str:
    def fmt(x):
        if x is None:
            return "nan"
        try:
            return f"{float(x):.10f}"
        except Exception:
            return str(x)

    return (
        f"Average CER: {fmt(filewise_summary.get('cer'))}\n"
        f"Average WER: {fmt(filewise_summary.get('wer'))}\n"
        f"SECS: {fmt(filewise_summary.get('secs'))}\n"
    )


def write_filewise_csv(path: str, rows: List[Dict[str, Any]]):
    """Write sample-level filewise metrics.

    Several fields are lists (turn_ids, reference_text, asr_hyp, cer_turns,
    etc.), so they are JSON-encoded inside CSV cells.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"

    fieldnames = [
        "run_id",
        "dataset_index",
        "rank",
        "num_turns",
        "cer",
        "wer",
        "secs",
        "turn_ids",
        "cer_turns",
        "wer_turns",
        "secs_turns",
        "pred_audio_seconds_turns",
        "target_audio_path",
        "context_audio_path",
        "pred_audio_paths",
        "reference_text",
        "asr_hyp",
    ]

    def csv_value(v):
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return v

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, None)) for k in fieldnames})

    os.replace(tmp_path, path)

# -----------------------------
# Dataset helpers
# -----------------------------


def _combined_audio_name(first_audio_filepath: str, paths: List[str]) -> str:
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in paths if p]
    ext = os.path.splitext(paths[-1])[1] if paths and paths[-1] else ""
    combined_name = "_".join(base_names) + ext
    dir_name = os.path.dirname(first_audio_filepath)
    return os.path.join(dir_name, combined_name) if dir_name else combined_name


class EvalJSONLDataset(Dataset):
    def __init__(self, file_path: str, num_turns: int = 1):
        self.samples = []
        raw_samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    sample["__dataset_index__"] = len(raw_samples)
                    raw_samples.append(sample)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_idx}: {e}")

        if num_turns <= 1:
            self.samples = raw_samples
            return

        single_turn_by_speaker = {}
        for sample in raw_samples:
            if isinstance(sample["text"], list):
                self.samples.append(sample)
            else:
                speaker = sample.get("speaker", "unknown")
                single_turn_by_speaker.setdefault(speaker, []).append(sample)

        synthetic_index = len(raw_samples)
        for _, speaker_samples in single_turn_by_speaker.items():
            buffer_texts, buffer_paths = [], []
            first_sample_meta = None

            for sample in speaker_samples:
                if not buffer_texts:
                    first_sample_meta = dict(sample)

                buffer_texts.append(sample["text"])
                buffer_paths.append(sample.get("audio_filepath", ""))

                if len(buffer_texts) == num_turns:
                    first_sample_meta["text"] = buffer_texts
                    first_sample_meta["audio_filepath"] = _combined_audio_name(
                        first_sample_meta.get("audio_filepath", ""),
                        buffer_paths,
                    )
                    first_sample_meta["__dataset_index__"] = synthetic_index
                    synthetic_index += 1
                    self.samples.append(first_sample_meta)
                    buffer_texts, buffer_paths, first_sample_meta = [], [], None

            if buffer_texts and first_sample_meta is not None:
                first_sample_meta["text"] = buffer_texts
                first_sample_meta["audio_filepath"] = _combined_audio_name(
                    first_sample_meta.get("audio_filepath", ""),
                    buffer_paths,
                )
                first_sample_meta["__dataset_index__"] = synthetic_index
                synthetic_index += 1
                self.samples.append(first_sample_meta)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _sample_text_segments_for_count(sample: Dict[str, Any], max_eval_turns=None) -> List[str]:
    text_data = sample.get("text", "")
    if isinstance(text_data, list):
        segments = text_data
        if max_eval_turns is not None:
            segments = segments[: int(max_eval_turns)]
        return [str(x) for x in segments]
    return [str(text_data)]


def estimate_text_token_count(sample: Dict[str, Any], model, max_eval_turns=None) -> int:
    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
    total = 0
    for segment in _sample_text_segments_for_count(sample, max_eval_turns=max_eval_turns):
        total += len(model.tokenizer.encode(segment, tokenizer_name=main_tokenizer_name)) + 1
    return int(total)


class SortedByTextTokenCountDataset(Dataset):
    def __init__(self, dataset: Dataset, model, max_eval_turns=None, descending: bool = True):
        self.dataset = dataset
        scored = []
        for i in range(len(dataset)):
            sample = dict(dataset[i])
            token_count = estimate_text_token_count(sample, model=model, max_eval_turns=max_eval_turns)
            sample["__text_token_count__"] = int(token_count)
            scored.append((token_count, i, sample))

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=bool(descending))
        self.indices = [i for _, i, _ in scored]
        self.token_counts = {i: int(tok) for tok, i, _ in scored}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        original_idx = self.indices[local_idx]
        sample = dict(self.dataset[original_idx])
        sample["__text_token_count__"] = self.token_counts[original_idx]
        return sample


# -----------------------------
# Audio / collate helpers
# -----------------------------


def _resolve_audio_path(path, root_path):
    if path is None:
        return None
    if root_path is not None and not os.path.isabs(path):
        return os.path.join(root_path, path)
    return path


def _load_audio(path, sample_rate, normalize=True, use_librosa=False):
    if path is None or not os.path.exists(path):
        return torch.zeros(1, dtype=torch.float32)

    if use_librosa:
        wav, sr = librosa.load(path, sr=sample_rate, mono=True)
        if normalize:
            wav = normalize_volume(wav)
        return torch.as_tensor(wav, dtype=torch.float32)

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if normalize:
        wav = normalize_volume(wav)

    wav = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0)
    return resample(wav, sr, sample_rate).squeeze(0)


def collate_and_tokenize_custom(
    batch,
    model,
    extra_duration_thrshould=1.3,
    sample_rate=22050,
    root_path=None,
    emulate_duplex_inference=False,
    add_interruption_token=False,
    pad_factor_text_speech=10,
    force_interruption=False,
    normalize_audio_volume=True,
    use_librosa=False,
    profile_multiturn_inference=False,
    max_eval_turns=None,
):
    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]

    if max_eval_turns is not None:
        max_eval_turns = int(max_eval_turns)
        if max_eval_turns <= 0:
            raise ValueError("--max_eval_turns must be > 0 when provided.")

        truncated_batch = []
        for s in batch:
            s = dict(s)
            if isinstance(s["text"], list):
                s["text"] = s["text"][:max_eval_turns]
                if isinstance(s.get("user_audio_file_path"), list):
                    s["user_audio_file_path"] = s["user_audio_file_path"][:max_eval_turns]
            truncated_batch.append(s)
        batch = truncated_batch

    is_profile = profile_multiturn_inference
    is_duplex = emulate_duplex_inference and not is_profile

    out_dict = {
        "duplex_multiturn": is_duplex,
        "regular_multiturn": (not is_duplex) and (not is_profile),
        "profile_multiturn": is_profile,
        "dataset_indices": [int(s.get("__dataset_index__", -1)) for s in batch],
        "text_token_counts": [int(s.get("__text_token_count__", -1)) for s in batch],
    }

    tokenized_list = []
    batched_turns = []
    batched_turn_lens = []
    valid_turn_masks = []

    if is_duplex:
        for s in batch:
            text_data = s["text"]

            if isinstance(text_data, list):
                full_ids = []
                for segment in text_data:
                    seg_ids = model.tokenizer.encode(segment, tokenizer_name=main_tokenizer_name) + [model.eos_id]
                    pad_ids = [model.pad_id] * (len(seg_ids) * pad_factor_text_speech)

                    if force_interruption:
                        fname = s["audio_filepath"]
                        no_ext = fname.split(".")[0]
                        sample_id = int(no_ext.split("_")[-1])
                        case = sample_id % 3

                        if case == 0:
                            if len(seg_ids) >= 2:
                                seg_ids[-2] = model.interruption_token_id
                                seg_ids[-1] = model.pad_id
                            else:
                                pad_ids[0] = model.interruption_token_id
                        elif case == 1:
                            eos_idx = min(6, len(pad_ids) - 1)
                            pad_ids[eos_idx] = model.interruption_token_id
                        else:
                            pad_ids[0] = model.interruption_token_id

                    elif add_interruption_token:
                        eos_idx = int(len(pad_ids) * 0.7)
                        pad_ids[eos_idx] = model.interruption_token_id

                    full_ids.extend(seg_ids)
                    full_ids.extend(pad_ids)

                tokenized_list.append(torch.as_tensor(full_ids, dtype=torch.long))
            else:
                tokenized_list.append(
                    torch.as_tensor(
                        model.tokenizer.encode(text_data, tokenizer_name=main_tokenizer_name) + [model.eos_id],
                        dtype=torch.long,
                    )
                )

        prefix = torch.full((25,), model.pad_id, dtype=torch.long)
        tokenized_list = [torch.cat([prefix, x]) for x in tokenized_list]
        out_dict["input_lengths"] = torch.tensor([len(x) for x in tokenized_list], dtype=torch.long)
        out_dict["input_ids"] = pad_sequence(tokenized_list, batch_first=True, padding_value=model.pad_id)

    else:
        max_turns = 1
        for s in batch:
            if isinstance(s["text"], list):
                max_turns = max(max_turns, len(s["text"]))

        for t in range(max_turns):
            turn_t_tokens, turn_t_lens, turn_t_valid = [], [], []

            for s in batch:
                text_data = s["text"]

                if isinstance(text_data, list):
                    if t < len(text_data):
                        seg_ids = model.tokenizer.encode(text_data[t], tokenizer_name=main_tokenizer_name) + [
                            model.eos_id
                        ]
                        turn_t_tokens.append(torch.as_tensor(seg_ids, dtype=torch.long))
                        turn_t_lens.append(len(seg_ids))
                        turn_t_valid.append(True)
                    else:
                        turn_t_tokens.append(torch.as_tensor([model.pad_id], dtype=torch.long))
                        turn_t_lens.append(1)
                        turn_t_valid.append(False)
                else:
                    if t == 0:
                        seg_ids = model.tokenizer.encode(text_data, tokenizer_name=main_tokenizer_name) + [
                            model.eos_id
                        ]
                        turn_t_tokens.append(torch.as_tensor(seg_ids, dtype=torch.long))
                        turn_t_lens.append(len(seg_ids))
                        turn_t_valid.append(True)
                    else:
                        turn_t_tokens.append(torch.as_tensor([model.pad_id], dtype=torch.long))
                        turn_t_lens.append(1)
                        turn_t_valid.append(False)

            batched_turns.append(pad_sequence(turn_t_tokens, batch_first=True, padding_value=model.pad_id))
            batched_turn_lens.append(torch.tensor(turn_t_lens, dtype=torch.long))
            valid_turn_masks.append(torch.tensor(turn_t_valid, dtype=torch.bool))

        out_dict["batched_turns"] = batched_turns
        out_dict["batched_turn_lens"] = batched_turn_lens
        out_dict["valid_turn_masks"] = valid_turn_masks

    audio_list, audio_lengths, target_num_frames = [], [], []
    context_audio_paths = []
    max_turns_for_user_audio = len(batched_turns) if not is_duplex else 0

    if is_profile and max_turns_for_user_audio > 0:
        user_audio_by_turn = [[] for _ in range(max_turns_for_user_audio)]
        user_audio_lens_by_turn = [[] for _ in range(max_turns_for_user_audio)]
    else:
        user_audio_by_turn, user_audio_lens_by_turn = [], []

    for i, s in enumerate(batch):
        audio_path = _resolve_audio_path(s.get("context_audio_filepath"), root_path)
        context_audio_paths.append(audio_path)
        wav = _load_audio(audio_path, sample_rate, normalize=normalize_audio_volume, use_librosa=use_librosa)
        audio_list.append(wav)
        audio_lengths.append(len(wav))

        if is_profile and max_turns_for_user_audio > 0:
            user_audio_paths = s.get("user_audio_file_path", None)

            for t in range(max_turns_for_user_audio):
                has_valid_text_turn = (isinstance(s["text"], list) and t < len(s["text"])) or (
                    not isinstance(s["text"], list) and t == 0
                )

                if (
                    isinstance(user_audio_paths, list)
                    and t < len(user_audio_paths)
                    and user_audio_paths[t]
                    and has_valid_text_turn
                ):
                    ua_path = _resolve_audio_path(user_audio_paths[t], root_path)
                    ua_wav = _load_audio(
                        ua_path,
                        sample_rate=sample_rate,
                        normalize=normalize_audio_volume,
                        use_librosa=use_librosa,
                    )
                else:
                    ua_wav = torch.zeros(int(2 * sample_rate), dtype=torch.float32)

                user_audio_by_turn[t].append(ua_wav)
                user_audio_lens_by_turn[t].append(len(ua_wav))

        tdur_audio_path = _resolve_audio_path(s["audio_filepath"], root_path)

        if tdur_audio_path and os.path.exists(tdur_audio_path):
            wav_dur = _load_audio(
                tdur_audio_path,
                sample_rate,
                normalize=normalize_audio_volume,
                use_librosa=use_librosa,
            )
            tdur = wav_dur.shape[0] // model.input_samples_per_frame
            target_num_frames.append(tdur * extra_duration_thrshould)
        else:
            if is_duplex:
                current_text_len = len(tokenized_list[i])
                target_num_frames.append(current_text_len if isinstance(s["text"], list) else current_text_len * 5)
            else:
                target_num_frames.append(sum([l[i].item() for l in batched_turn_lens]) * 5)

    max_audio_len = max(audio_lengths)
    B = len(audio_lengths)
    padded_audio = torch.zeros((B, max_audio_len), dtype=torch.float32)

    for i, wav in enumerate(audio_list):
        padded_audio[i, : len(wav)] = wav

    if is_profile and max_turns_for_user_audio > 0:
        padded_user_audio_turns, padded_user_audio_turns_lens = [], []

        for t in range(max_turns_for_user_audio):
            turn_lens = user_audio_lens_by_turn[t]
            max_turn_audio_len = max(turn_lens)
            padded_turn_audio = torch.zeros((B, max_turn_audio_len), dtype=torch.float32)

            for i, wav in enumerate(user_audio_by_turn[t]):
                padded_turn_audio[i, : len(wav)] = wav

            padded_user_audio_turns.append(padded_turn_audio)
            padded_user_audio_turns_lens.append(torch.tensor(turn_lens, dtype=torch.long))

        out_dict["user_audio_turns"] = padded_user_audio_turns
        out_dict["user_audio_turns_lens"] = padded_user_audio_turns_lens

    raw_turn_texts = []
    for s in batch:
        if isinstance(s["text"], list):
            raw_turn_texts.append([str(x) for x in s["text"]])
        else:
            raw_turn_texts.append([str(s["text"])])

    out_dict["context_audio"] = padded_audio
    out_dict["context_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)
    out_dict["context_audio_paths"] = context_audio_paths
    out_dict["target_audio_paths"] = [s["audio_filepath"] for s in batch]
    out_dict["target_num_frames"] = target_num_frames
    out_dict["raw_turn_texts"] = raw_turn_texts
    out_dict["raw_text"] = [" ".join(x) for x in raw_turn_texts]

    return out_dict


# -----------------------------
# Model / generation
# -----------------------------


def attach_dtype_counter(model):
    handles = []
    stats = {}
    examples = {}

    def is_leaf(module):
        return len(list(module.children())) == 0

    def get_dtype(x):
        if torch.is_tensor(x):
            return str(x.dtype)
        if isinstance(x, (list, tuple)):
            for t in x:
                if torch.is_tensor(t):
                    return str(t.dtype)
        return "other"

    def get_module_group(name):
        return name.split(".")[0] if "." in name else name

    def hook_fn(name):
        def fn(module, inputs, outputs):
            dtype = get_dtype(outputs)
            if dtype not in ["torch.float16", "torch.bfloat16", "torch.float32"]:
                dtype = "other"
            group = get_module_group(name)
            if group not in stats:
                stats[group] = {
                    "torch.float16": 0,
                    "torch.bfloat16": 0,
                    "torch.float32": 0,
                    "other": 0,
                }
                examples[group] = {
                    "torch.float16": [],
                    "torch.bfloat16": [],
                    "torch.float32": [],
                    "other": [],
                }
            stats[group][dtype] += 1
            if len(examples[group][dtype]) < 3:
                examples[group][dtype].append(module.__class__.__name__)

        return fn

    for name, module in model.named_modules():
        if is_leaf(module):
            handles.append(module.register_forward_hook(hook_fn(name)))
    return handles, stats, examples


def report_dtype_stats(handles, stats, examples, rank=0):
    for h in handles:
        h.remove()
    logging.info(f"[rank={rank}] === DTYPE USAGE PER MODULE ===")
    for group, group_stats in stats.items():
        total = sum(group_stats.values())
        if total == 0:
            continue
        logging.info(f"[rank={rank}] --- {group} ---")
        for dtype, count in group_stats.items():
            if count > 0:
                logging.info(f"[rank={rank}] {dtype}: {count} ({100 * count / total:.2f}%)")
    logging.info(f"[rank={rank}] === DTYPE EXAMPLES ===")
    for group, group_examples in examples.items():
        for dtype, mods in group_examples.items():
            if mods:
                logging.info(f"[rank={rank}] {group} {dtype}: {mods}")


def build_model_and_codec(args, target_device, target_dtype):
    model_cfg = EasyMagpieTTSInferenceModel.restore_from(args.checkpoint_path, return_config=True)

    with open_dict(model_cfg):
        model_cfg.target = "nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel"
        model_cfg.codecmodel_path = args.codec_model_path
        model_cfg.train_ds = None
        model_cfg.validation_ds = None
        model_cfg.run_val_inference = False
        model_cfg.use_utmos = False
        model_cfg.use_meta_init_for_decoder = True

        if args.phoneme_tokenizer_path and getattr(model_cfg, "phoneme_tokenizer", None) is not None:
            model_cfg.phoneme_tokenizer.tokenizer_path = args.phoneme_tokenizer_path

    model = EasyMagpieTTSInferenceModel.restore_from(
        args.checkpoint_path,
        override_config_path=model_cfg,
        map_location=torch.device("cpu"),
    )
    model.use_kv_cache_for_inference = True
    model.to(dtype=target_dtype)
    model.eval().to(target_device)

    model.input_samples_per_frame = int(model.codec_model_samples_per_frame * model.frame_stacking_factor)
    model.target_samples_per_frame = model.input_samples_per_frame / (model.sample_rate / model.output_sample_rate)

    codec_model = AudioCodecModel.restore_from(args.codec_model_path, strict=False, map_location=torch.device("cpu"))
    if hasattr(codec_model, "discriminator"):
        del codec_model.discriminator
    codec_model.freeze()
    codec_model = codec_model.to(target_device).eval()

    codec_converter = None
    if getattr(model, "_codec_converter", None) is not None:
        vq_new = deepcopy(model._codec_converter.vector_quantizer_new).to(target_device).eval()
        codec_converter = VectorQuantizerIndexConverter(
            vector_quantizer_original=codec_model.vector_quantizer,
            vector_quantizer_new=vq_new,
        ).to(target_device).eval()

    model._codec_helper = CodecHelper(codec_model=codec_model, codec_converter=codec_converter)
    model._generate_codec_silence_buffer()

    return model


def prepare_inputs_for_device(inputs, model, args, target_dtype, speaker_wav=None):
    B = inputs["context_audio"].size(0)
    device = model.device

    inputs["context_audio"] = inputs["context_audio"].to(device, dtype=target_dtype)
    inputs["context_audio_lengths"] = inputs["context_audio_lengths"].to(device)

    if args.user_custom_speaker_reference and speaker_wav is not None:
        inputs["context_audio"] = speaker_wav.repeat(B, 1).detach()
        inputs["context_audio_lengths"] = torch.full((B,), speaker_wav.size(-1), dtype=torch.long, device=device)

    if "user_audio_turns" in inputs:
        inputs["user_audio_turns"] = [x.to(device, dtype=target_dtype) for x in inputs["user_audio_turns"]]
        inputs["user_audio_turns_lens"] = [x.to(device) for x in inputs["user_audio_turns_lens"]]

    return inputs


def run_generation(model, inputs, args, codec_sil_codes):
    B = inputs["context_audio"].size(0)
    device = model.device
    profile_turn_frame_ranges = []
    profile_decode_start_frame = 0

    with torch.inference_mode():
        wav = inputs["context_audio"]
        wav_len = inputs["context_audio_lengths"]
        codes, codes_lens = model._codec_helper.audio_to_codes(wav, wav_len)

        use_lang = bool(getattr(model, "add_language_to_context_text", False))
        ctx_text = f"[{args.language.upper()}]" if use_lang else "[NO TEXT CONTEXT]"
        ctx_text_ids = model.tokenizer.encode(ctx_text, tokenizer_name=model.text_conditioning_tokenizer_name)

        ctx_toks = torch.tensor([ctx_text_ids], dtype=torch.long, device=device).expand(B, -1)
        ctx_toks_lens = torch.tensor([len(ctx_text_ids)] * B, dtype=torch.long, device=device)

        state = model.streaming_init(
            context_audio_codes=codes,
            context_audio_codes_lens=codes_lens,
            context_text_tokens=ctx_toks,
            context_text_tokens_lens=ctx_toks_lens,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            use_local_transformer=True,
            temperature=args.temperature,
            topk=args.topk,
            phoneme_input_type="pred",
            phoneme_sampling_method="argmax",
            use_inference_mode=True,
        )

        if inputs["duplex_multiturn"]:
            text = inputs["input_ids"].to(device)
            text_lens = inputs["input_lengths"].to(device)

            in_initial_silence = torch.ones(B, dtype=torch.bool, device=device)
            in_post_speech_silence = torch.zeros(B, dtype=torch.bool, device=device)
            text_exhausted = state.text_tokens_seen >= text_lens

            while not text_exhausted.all():
                state.finished = state.finished & text_exhausted
                state.text_finished = state.text_finished & text_exhausted

                if hasattr(state, "phoneme_stream_ended"):
                    state.phoneme_stream_ended = state.phoneme_stream_ended & text_exhausted

                positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                current_tokens = text[torch.arange(B, device=device), positions]
                current_tokens = torch.where(
                    text_exhausted,
                    torch.full_like(current_tokens, model.eos_id),
                    current_tokens,
                )

                is_pad_or_eos = (current_tokens == model.pad_id) | (current_tokens == model.eos_id)
                in_initial_silence = in_initial_silence & is_pad_or_eos
                in_post_speech_silence = in_post_speech_silence & is_pad_or_eos

                state, audio_codes, _ = model.streaming_step(
                    state=state,
                    text_tokens=current_tokens,
                    use_inference_mode=True,
                )

                if audio_codes is not None and args.force_speech_sil_codes:
                    force_silence_mask = in_initial_silence | in_post_speech_silence
                    if force_silence_mask.any():
                        expanded_sil = codec_sil_codes.view(1, -1, 1).expand_as(audio_codes)
                        mask_3d = force_silence_mask.view(B, 1, 1)
                        state.all_predictions[-1] = torch.where(mask_3d, expanded_sil, audio_codes)

                in_post_speech_silence = in_post_speech_silence | state.finished
                text_exhausted = state.text_tokens_seen >= text_lens

        elif inputs["regular_multiturn"]:
            batched_turns = inputs["batched_turns"]
            batched_turn_lens = inputs["batched_turn_lens"]
            valid_turn_masks = inputs["valid_turn_masks"]
            turn_offsets = torch.zeros(B, dtype=torch.long, device=device)

            for t in range(len(batched_turns)):
                turn_text = batched_turns[t].to(device)
                turn_lens = batched_turn_lens[t].to(device)
                valid_mask = valid_turn_masks[t].to(device)

                state.finished = state.finished & (~valid_mask)
                state.text_finished = state.text_finished & (~valid_mask)

                if hasattr(state, "phoneme_stream_ended"):
                    state.phoneme_stream_ended = state.phoneme_stream_ended & (~valid_mask)

                if state.finished.all():
                    continue

                turn_offsets = torch.where(valid_mask, state.text_tokens_seen, turn_offsets)
                turn_steps = 0

                while not state.finished.all() and turn_steps < args.max_tts_steps:
                    turn_steps += 1
                    relative_positions = state.text_tokens_seen - turn_offsets
                    positions = relative_positions.clamp(min=0, max=turn_text.size(1) - 1)
                    current_tokens = turn_text[torch.arange(B, device=device), positions]

                    exhausted = relative_positions >= turn_lens
                    current_tokens = torch.where(
                        exhausted,
                        torch.full_like(current_tokens, model.eos_id),
                        current_tokens,
                    )

                    state, _, _ = model.streaming_step(
                        state=state,
                        text_tokens=current_tokens,
                        use_inference_mode=True,
                    )

        elif inputs["profile_multiturn"]:
            if B != 1:
                raise RuntimeError("--profile_multiturn_inference requires --batch_size=1 per process.")

            batched_turns = inputs["batched_turns"]
            batched_turn_lens = inputs["batched_turn_lens"]
            valid_turn_masks = inputs["valid_turn_masks"]

            for t in range(len(batched_turns)):
                turn_text = batched_turns[t].to(device)
                turn_lens = batched_turn_lens[t].to(device)
                valid_mask = valid_turn_masks[t].to(device)

                if not bool(valid_mask[0].item()):
                    continue

                state.finished.zero_()
                state.text_finished.zero_()
                state.audio_prediction_end_idx.fill_(-1)

                if hasattr(state, "turn_text_tokens_seen"):
                    state.turn_text_tokens_seen.zero_()
                if hasattr(state, "phoneme_steps"):
                    state.phoneme_steps.zero_()
                if hasattr(state, "phoneme_stream_ended"):
                    state.phoneme_stream_ended.zero_()
                if hasattr(state, "phoneme_eos_detected"):
                    state.phoneme_eos_detected.zero_()
                state.last_phoneme_tokens = None

                if not model.cfg.get("condition_on_user_speech", False):
                    if "user_audio_turns" in inputs:
                        profile_T = int(round(inputs["user_audio_turns"][t].size(-1) / model.input_samples_per_frame))
                        profile_seconds = profile_T * model.input_samples_per_frame / model.sample_rate
                    else:
                        profile_seconds = args.profile_pad_min_sec + torch.rand((), device=device).item() * (
                            args.profile_pad_max_sec - args.profile_pad_min_sec
                        )
                        profile_T = max(
                            1,
                            int(round(profile_seconds * model.sample_rate / model.input_samples_per_frame)),
                        )

                    profile_tokens = torch.full((1, profile_T), model.pad_id, dtype=torch.long, device=device)
                    user_audio_channel_embedding = None

                else:
                    if "user_audio_turns" in inputs:
                        user_audio = inputs["user_audio_turns"][t]
                        user_audio_lens = inputs["user_audio_turns_lens"][t]
                    else:
                        user_audio = inputs["context_audio"]
                        user_audio_lens = inputs["context_audio_lengths"]

                    user_audio_codes, user_audio_codes_lens = model._codec_helper.audio_to_codes(
                        user_audio,
                        user_audio_lens,
                    )

                    if model._codec_converter is not None:
                        user_audio_codes = model._codec_converter.convert_original_to_new(
                            audio_tokens=user_audio_codes,
                            audio_lens=user_audio_codes_lens,
                        ).long()

                    user_audio_codes, user_audio_codes_lens = model.stack_codes(
                        user_audio_codes,
                        user_audio_codes_lens,
                        model.audio_bos_id,
                        model.audio_eos_id,
                        model.frame_stacking_factor,
                        model.num_audio_codebooks,
                    )

                    user_audio_embedded = model.embed_audio_tokens(user_audio_codes)

                    boundary_trim = model.cfg.get("user_audio_boundary_trim", 4)
                    boundary_trim = 0 if boundary_trim is None else int(boundary_trim)

                    if boundary_trim == 0:
                        real_start = 0
                        real_end = int(user_audio_codes_lens[0].item())
                    else:
                        turn_len_with_special = int(user_audio_codes_lens[0].item())
                        real_start = 1
                        real_end = max(real_start, turn_len_with_special - 1)

                    user_audio_embedded = user_audio_embedded[:, real_start:real_end]
                    copy_len = user_audio_embedded.size(1)

                    if boundary_trim > 0:
                        trim = min(boundary_trim, copy_len // 2)
                        if trim > 0:
                            user_audio_embedded[:, :trim] = 0.0
                            user_audio_embedded[:, copy_len - trim :] = 0.0

                    bos_user_pad = torch.zeros(
                        user_audio_embedded.size(0),
                        1,
                        user_audio_embedded.size(2),
                        device=user_audio_embedded.device,
                        dtype=user_audio_embedded.dtype,
                    )
                    user_audio_embedded = torch.cat([bos_user_pad, user_audio_embedded], dim=1)

                    profile_T = user_audio_embedded.size(1)
                    profile_tokens = torch.full((B, profile_T), model.pad_id, dtype=torch.long, device=device)
                    user_audio_channel_embedding = user_audio_embedded
                    profile_seconds = profile_T * model.input_samples_per_frame / model.sample_rate

                delay_tokens = int(state.config.training_mode.streaming_speech_delay)
                delay_tokens = min(delay_tokens, int(turn_lens[0].item()), profile_T)

                warmup_tokens = turn_text[:, :delay_tokens]
                turn_text = turn_text[:, delay_tokens:]
                turn_lens = torch.clamp(turn_lens - delay_tokens, min=0)

                if user_audio_channel_embedding is not None and delay_tokens > 0:
                    warmup_user_audio = user_audio_channel_embedding[:, -delay_tokens:]
                    user_audio_channel_embedding = user_audio_channel_embedding[:, :-delay_tokens]
                    profile_tokens = profile_tokens[:, :-delay_tokens]
                else:
                    warmup_user_audio = None

                if profile_tokens.size(1) > 0:
                    state = model.streaming_prefill_profile(
                        state=state,
                        text_tokens=profile_tokens,
                        use_inference_mode=True,
                        user_audio_channel_embedding=user_audio_channel_embedding,
                    )

                for i in range(delay_tokens):
                    user_step_emb = warmup_user_audio[:, i] if warmup_user_audio is not None else None

                    state.finished.zero_()
                    state, _, _ = model.streaming_step(
                        state=state,
                        text_tokens=warmup_tokens[:, i],
                        user_audio_channel_embedding=user_step_emb,
                        prefill_like_step=not bool(model.cfg.get("agent_mask_include_transition_prefix", False)),
                        prefill_like_is_last_step=(i == delay_tokens - 1),
                        use_inference_mode=True,
                    )

                logging.info(f"[profile_multiturn] turn={t} prefilled {profile_T} steps ({profile_seconds:.2f}s)")

                turn_start_frame = sum(p.size(-1) for p in state.all_predictions)
                if t == 0:
                    state.audio_prediction_start_idx.fill_(turn_start_frame)
                    profile_decode_start_frame = turn_start_frame

                turn_offset = state.text_tokens_seen.clone()
                turn_steps = 0
                saw_audio = False
                turn_ended_with_audio_eos = False

                while turn_steps < args.max_tts_steps:
                    turn_steps += 1
                    state.finished.zero_()

                    relative_position = state.text_tokens_seen - turn_offset
                    text_exhausted = relative_position >= turn_lens

                    if turn_text.size(1) == 0:
                        current_tokens = torch.full((B,), model.eos_id, dtype=torch.long, device=device)
                    else:
                        position = relative_position.clamp(min=0, max=turn_text.size(1) - 1)
                        current_tokens = turn_text[torch.arange(B, device=device), position]
                        current_tokens = torch.where(
                            text_exhausted,
                            torch.full_like(current_tokens, model.eos_id),
                            current_tokens,
                        )

                    state, audio_codes, _ = model.streaming_step(
                        state=state,
                        text_tokens=current_tokens,
                        use_inference_mode=True,
                    )

                    if audio_codes is not None and not saw_audio:
                        saw_audio = True

                    if bool(text_exhausted[0].item()) and bool(state.finished[0].item()):
                        turn_ended_with_audio_eos = True
                        break

                state.audio_prediction_end_idx.fill_(-1)
                state.finished.zero_()

                turn_end_frame = sum(p.size(-1) for p in state.all_predictions)
                profile_turn_frame_ranges.append((t, turn_start_frame, turn_end_frame))

                logging.info(
                    f"[profile_multiturn] turn={t} steps={turn_steps} "
                    f"saw_audio={saw_audio} ended_with_audio_eos={turn_ended_with_audio_eos}"
                )

        bos_id = getattr(model, "audio_bos_id", -1)
        eos_id = getattr(model, "audio_eos_id", -1)
        speaking_id = getattr(model, "audio_user_speaking_id", -1)
        speaking_end_id = getattr(model, "audio_user_speaking_end_id", -1)
        sil_injection = codec_sil_codes.view(1, -1, 1)

        for step_idx in range(len(state.all_predictions)):
            pred = state.all_predictions[step_idx]
            mask = (pred == bos_id) | (pred == eos_id) | (pred == speaking_id) | (pred == speaking_end_id)
            frame_mask = mask.any(dim=1, keepdim=True)
            if frame_mask.any():
                state.all_predictions[step_idx] = torch.where(frame_mask, sil_injection.expand_as(pred), pred)

        if inputs["duplex_multiturn"] or inputs["profile_multiturn"]:
            state.audio_prediction_end_idx.fill_(-1)

        finalize_output = model.streaming_finalize(state, use_inference_mode=True)

    return finalize_output, profile_turn_frame_ranges, profile_decode_start_frame


def load_speaker_wav_if_needed(args, model, target_dtype):
    if args.user_custom_speaker_reference and args.inference_speaker_reference:
        return _load_audio(
            args.inference_speaker_reference,
            model.sample_rate,
            normalize=args.normalize_volume,
            use_librosa=args.use_librosa,
        ).unsqueeze(0).to(model.device, dtype=target_dtype)

    return None


# -----------------------------
# Save generation outputs and metric manifests
# -----------------------------


def write_audio_1d(path: str, wav: torch.Tensor, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav_np = wav.detach().cpu().float().numpy()
    sf.write(path, wav_np, samplerate=sr)


def build_metric_item(
    run_id: int,
    rank: int,
    dataset_index: int,
    turn_id: int,
    target_audio_path: str,
    reference_text: str,
    pred_audio_path: str,
    context_audio_path: str,
    pred_audio_samples: int,
    context_audio_samples: int,
    output_sample_rate: int,
    context_sample_rate: int,
):
    return {
        "run_id": int(run_id),
        "rank": int(rank),
        "dataset_index": int(dataset_index),
        "turn_id": int(turn_id),
        "target_audio_path": target_audio_path,
        "reference_text": reference_text,
        "pred_audio_path": pred_audio_path,
        "context_audio_path": context_audio_path,
        "pred_audio_samples": int(pred_audio_samples),
        "context_audio_samples": int(context_audio_samples),
        "pred_audio_seconds": float(pred_audio_samples / output_sample_rate),
        "context_audio_seconds": float(context_audio_samples / context_sample_rate),
        "output_sample_rate": int(output_sample_rate),
        "context_sample_rate": int(context_sample_rate),
    }


def save_generation_outputs_and_build_metric_items(
    model,
    inputs,
    finalize_output,
    profile_turn_frame_ranges,
    profile_decode_start_frame,
    args,
    rank: int,
    run_id: int,
):
    device = model.device
    B = inputs["context_audio"].size(0)

    with fp32_precision():
        audio_f32 = finalize_output.audio.float()
        audio_len = finalize_output.audio_len.int()

        expected_audio_lens = (
            torch.tensor(inputs["target_num_frames"], device=device) * model.target_samples_per_frame
        ).int()

        if inputs["duplex_multiturn"]:
            text_lens = inputs["input_lengths"].to(device)
            audio_len = (text_lens * model.target_samples_per_frame).int()
            audio_len = torch.min(audio_len, torch.tensor(audio_f32.size(1), device=device))
        elif inputs["profile_multiturn"]:
            audio_len = finalize_output.audio_len.int()
        else:
            audio_len = torch.min(audio_len, expected_audio_lens)

        audio_out_dir = get_audio_out_dir(args)
        metric_turn_dir = get_generated_turn_audio_dir(args)
        metric_context_dir = get_context_metric_audio_dir(args)
        os.makedirs(audio_out_dir, exist_ok=True)
        os.makedirs(metric_turn_dir, exist_ok=True)
        os.makedirs(metric_context_dir, exist_ok=True)

        audio_f32_cpu = audio_f32.detach().cpu()
        audio_len_cpu = audio_len.detach().cpu()
        metric_items = []

        for i in range(B):
            target_path = inputs["target_audio_paths"][i]
            base_name = os.path.basename(target_path)
            stem, ext = os.path.splitext(base_name)
            if not ext:
                ext = ".wav"

            dataset_idx = int(inputs.get("dataset_indices", [-1] * B)[i])
            safe_stem = (
                f"run{run_id:02d}_idx{dataset_idx:08d}_{stem}"
                if dataset_idx >= 0
                else f"run{run_id:02d}_rank{rank}_{stem}"
            )

            context_len = int(inputs["context_audio_lengths"][i].detach().cpu().item())
            context_wav = inputs["context_audio"][i, :context_len].detach().cpu().float()
            context_metric_path = os.path.join(metric_context_dir, f"{safe_stem}_context.wav")
            write_audio_1d(context_metric_path, context_wav, model.sample_rate)

            if inputs["profile_multiturn"]:
                full_len = int(audio_len_cpu[i].item())
                full_wav_t = audio_f32_cpu[i, :full_len].float()

                samples_per_prediction_frame = model.codec_model_samples_per_frame / (
                    model.sample_rate / model.output_sample_rate
                )

                aligned_agent = torch.zeros_like(full_wav_t)
                raw_turn_texts = inputs.get("raw_turn_texts", [[] for _ in range(B)])

                for turn_id, start_frame, end_frame in profile_turn_frame_ranges:
                    rel_start_frame = start_frame - profile_decode_start_frame
                    rel_end_frame = end_frame - profile_decode_start_frame

                    start_sample = int(round(rel_start_frame * samples_per_prediction_frame))
                    end_sample = int(round(rel_end_frame * samples_per_prediction_frame))

                    start_sample = max(0, min(start_sample, full_len))
                    end_sample = max(start_sample, min(end_sample, full_len))

                    aligned_agent[start_sample:end_sample] = full_wav_t[start_sample:end_sample]

                    turn_wav = aligned_agent[start_sample:end_sample].float()
                    turn_out_path = os.path.join(audio_out_dir, f"{safe_stem}_turn_{turn_id}{ext}")
                    write_audio_1d(turn_out_path, turn_wav, model.output_sample_rate)

                    metric_turn_path = os.path.join(metric_turn_dir, f"{safe_stem}_turn_{turn_id}.wav")
                    write_audio_1d(metric_turn_path, turn_wav, model.output_sample_rate)

                    if turn_id < len(raw_turn_texts[i]):
                        metric_items.append(
                            build_metric_item(
                                run_id=run_id,
                                rank=rank,
                                dataset_index=dataset_idx,
                                turn_id=turn_id,
                                target_audio_path=target_path,
                                reference_text=str(raw_turn_texts[i][turn_id]),
                                pred_audio_path=metric_turn_path,
                                context_audio_path=context_metric_path,
                                pred_audio_samples=int(turn_wav.numel()),
                                context_audio_samples=int(context_wav.numel()),
                                output_sample_rate=model.output_sample_rate,
                                context_sample_rate=model.sample_rate,
                            )
                        )

                full_out_path = os.path.join(audio_out_dir, f"{safe_stem}{ext}")
                write_audio_1d(full_out_path, aligned_agent, model.output_sample_rate)

                if "user_audio_turns" in inputs:
                    user_segments = []

                    first_user_len_in = int(inputs["user_audio_turns_lens"][0][i].item())
                    first_user_delay_out = int(round(first_user_len_in * model.output_sample_rate / model.sample_rate))

                    for turn_id, start_frame, _ in profile_turn_frame_ranges:
                        if turn_id >= len(inputs["user_audio_turns"]):
                            continue

                        turn_audio = inputs["user_audio_turns"][turn_id][i].detach().cpu().float()
                        turn_audio_len = int(inputs["user_audio_turns_lens"][turn_id][i].item())
                        turn_audio = turn_audio[:turn_audio_len]

                        turn_audio_out = resample(
                            turn_audio.unsqueeze(0),
                            model.sample_rate,
                            model.output_sample_rate,
                        ).squeeze(0)

                        if turn_id == 0:
                            user_start_sample = 0
                        else:
                            prev_turn_end_frame = profile_turn_frame_ranges[turn_id - 1][2]
                            rel_prev_end_frame = prev_turn_end_frame - profile_decode_start_frame
                            user_start_sample = first_user_delay_out + int(
                                round(rel_prev_end_frame * samples_per_prediction_frame)
                            )

                        user_segments.append((user_start_sample, turn_audio_out.detach().cpu().float()))

                    total_user_len = 0
                    for s, wav_seg in user_segments:
                        total_user_len = max(total_user_len, s + wav_seg.numel())

                    user_ch = torch.zeros(total_user_len)
                    for s, wav_seg in user_segments:
                        e = s + wav_seg.numel()
                        user_ch[s:e] += wav_seg

                    agent_ch = torch.cat([torch.zeros(first_user_delay_out, dtype=aligned_agent.dtype), aligned_agent])

                    stereo_len = max(user_ch.numel(), agent_ch.numel())
                    user_pad = torch.zeros(stereo_len)
                    agent_pad = torch.zeros(stereo_len)

                    user_pad[: user_ch.numel()] = user_ch
                    agent_pad[: agent_ch.numel()] = agent_ch

                    stereo = torch.stack([user_pad, agent_pad], dim=1).numpy()
                    aligned_path = os.path.join(audio_out_dir, f"{safe_stem}_user_agent_aligned{ext}")
                    sf.write(aligned_path, stereo, samplerate=model.output_sample_rate)

            else:
                full_len = int(audio_len_cpu[i].item())
                wav = audio_f32_cpu[i, :full_len].float()
                out_path = os.path.join(audio_out_dir, f"{safe_stem}{ext}")
                write_audio_1d(out_path, wav, model.output_sample_rate)

                metric_turn_path = os.path.join(metric_turn_dir, f"{safe_stem}_turn_0.wav")
                write_audio_1d(metric_turn_path, wav, model.output_sample_rate)

                metric_items.append(
                    build_metric_item(
                        run_id=run_id,
                        rank=rank,
                        dataset_index=dataset_idx,
                        turn_id=0,
                        target_audio_path=target_path,
                        reference_text=str(inputs["raw_text"][i]),
                        pred_audio_path=metric_turn_path,
                        context_audio_path=context_metric_path,
                        pred_audio_samples=int(wav.numel()),
                        context_audio_samples=int(context_wav.numel()),
                        output_sample_rate=model.output_sample_rate,
                        context_sample_rate=model.sample_rate,
                    )
                )

    return metric_items


# -----------------------------
# Metrics after generation
# -----------------------------


def torch_rms_norm(wav: torch.Tensor, db_level: float = -27.0) -> torch.Tensor:
    denom = torch.sum(wav**2)
    if denom <= 0:
        return wav
    r = 10 ** (db_level / 20)
    a = torch.sqrt((wav.size(-1) * (r**2)) / denom)
    return wav * a


def _load_audio_for_metric(path: str, sample_rate: int):
    wav = _load_audio(path, sample_rate=sample_rate, normalize=False, use_librosa=False)
    if wav.numel() == 0:
        wav = torch.zeros(1, dtype=torch.float32)
    return wav.float()


def _pad_audio_1d_list(wavs: List[torch.Tensor], device, dtype=torch.float32):
    if len(wavs) == 0:
        return torch.zeros((0, 1), device=device, dtype=dtype), torch.zeros((0,), device=device, dtype=torch.long)

    lens = torch.tensor([max(1, int(w.numel())) for w in wavs], device=device, dtype=torch.long)
    max_len = int(lens.max().item())
    out = torch.zeros((len(wavs), max_len), device=device, dtype=dtype)

    for i, w in enumerate(wavs):
        w = w.to(device=device, dtype=dtype).flatten()
        if w.numel() == 0:
            continue
        out[i, : w.numel()] = w

    return out, lens


def chunk_list(xs: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(xs), chunk_size):
        yield xs[start : start + chunk_size]


def _metric_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_metric_batch_audio(batch_items: List[Dict[str, Any]], args):
    pred_wavs = []
    context_wavs = []

    for item in batch_items:
        pred = _load_audio_for_metric(item["pred_audio_path"], sample_rate=int(item["output_sample_rate"]))
        context = _load_audio_for_metric(item["context_audio_path"], sample_rate=int(item["context_sample_rate"]))

        if args.max_metric_audio_sec is not None:
            max_pred_len = int(float(args.max_metric_audio_sec) * int(item["output_sample_rate"]))
            pred = pred[: max(1, max_pred_len)]

        pred_wavs.append(pred)
        context_wavs.append(context)

    device = _metric_device()
    pred_audio, pred_lens = _pad_audio_1d_list(pred_wavs, device=device)
    context_audio, context_lens = _pad_audio_1d_list(context_wavs, device=device)
    output_sample_rate = int(batch_items[0]["output_sample_rate"])
    context_sample_rate = int(batch_items[0]["context_sample_rate"])

    return pred_audio, pred_lens, context_audio, context_lens, output_sample_rate, context_sample_rate


def compute_metrics_after_generation(args, rank: int, world_size: int, metric_items: List[Dict[str, Any]]):
    """
    Load metric models only after generation is complete.

    Order:
      1. Load ASR, compute turn-level CER/WER and ASR hyps, then free ASR.
      2. Load SECS speaker encoder and compute turn-level SECS.
      3. Save rank-level aggregate metrics from the same turn-level rows.

    SECS is always computed turn-by-turn, like CER/WER. The grouped filewise
    output stores secs_turns and sample-level secs, and metrics_final.* receives
    the turn-level aggregate SECS.
    """
    metric_start = time.time()

    if len(metric_items) == 0:
        return {
            "rank": int(rank),
            "world_size": int(world_size),
            "num_processed": 0,
            "num_metric_items": 0,
            "metric_elapsed_sec": 0.0,
            "intelligibility": {},
            "secs": {},
        }, []

    normalizer = EnglishTextNormalizer()
    normalizer.ignore_patterns = r"$^"
    filewise_rows = []

    # ASR pass.
    all_rank_print(rank, f"loading ASR after generation: {args.asr_model_name}")
    with fp32_precision():
        intelligibility = Intelligibility(args.asr_model_name, reuse_asr_hyps=False).reset()

    for batch_items in chunk_list(metric_items, args.metric_batch_size):
        refs = [x["reference_text"] for x in batch_items]
        pred_audio, pred_lens, _, _, output_sr, _ = _load_metric_batch_audio(batch_items, args)

        with fp32_precision():
            pred_16k = resample(pred_audio, output_sr, 16000)
            pred_16k_lens = (pred_lens / output_sr * 16000).to(torch.long)
            pred_16k = torch_rms_norm(pred_16k)

            asr_hyps = intelligibility.update(
                name="dataset",
                refs=refs,
                pred_audio=pred_16k,
                pred_audio_lens=pred_16k_lens,
                asr_hyps=None,
            )

        for item, hyp in zip(batch_items, asr_hyps):
            ref_norm = normalizer(str(item["reference_text"])).strip()
            hyp_norm = normalizer(str(hyp)).strip()
            if ref_norm == "":
                cer = None
                wer = None
            else:
                cer = float(word_error_rate([hyp_norm], [ref_norm], use_cer=True))
                wer = float(word_error_rate([hyp_norm], [ref_norm], use_cer=False))

            row = dict(item)
            row["asr_hyp"] = hyp
            row["cer"] = cer
            row["wer"] = wer
            row["secs"] = None
            filewise_rows.append(row)

    with fp32_precision():
        cer_wer = metric_dict_to_jsonable(intelligibility.compute())
    del intelligibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SECS pass. This is intentionally turn-level, matching CER/WER.
    # We keep one aggregate SECS metric for metrics_final.* and also compute
    # one SECS value per filewise turn row so grouped outputs have secs_turns.
    all_rank_print(rank, f"loading speaker encoder after ASR is released: {args.secs_model_name}")
    with fp32_precision():
        secs_metric = SECS(args.secs_model_name).reset()

    # Aggregate turn-level SECS for metrics_final.json / metrics_final.txt.
    for batch_items in chunk_list(metric_items, args.metric_batch_size):
        pred_audio, pred_lens, context_audio, context_lens, output_sr, context_sr = _load_metric_batch_audio(
            batch_items, args
        )

        with fp32_precision():
            pred_16k = resample(pred_audio, output_sr, 16000)
            pred_16k_lens = (pred_lens / output_sr * 16000).to(torch.long)
            context_16k = resample(context_audio, context_sr, 16000)
            context_16k_lens = (context_lens / context_sr * 16000).to(torch.long)

            pred_16k = torch_rms_norm(pred_16k)
            context_16k = torch_rms_norm(context_16k)

            secs_metric.update(
                name="dataset",
                target_audio=context_16k,
                target_audio_lens=context_16k_lens,
                pred_audio=pred_16k,
                pred_audio_lens=pred_16k_lens,
            )

    with fp32_precision():
        secs_scores = metric_dict_to_jsonable(secs_metric.compute())
    del secs_metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Per-turn SECS for filewise/grouped outputs. This is always computed so
    # secs_turns and sample-level secs are never null in final filewise metrics.
    # It is slower than aggregate-only SECS, but it matches the turn-level
    # semantics requested for CER/WER/SECS.
    all_rank_print(rank, "computing per-turn SECS rows")
    for row in filewise_rows:
        pred_audio, pred_lens, context_audio, context_lens, output_sr, context_sr = _load_metric_batch_audio([row], args)

        with fp32_precision():
            one_secs = SECS(args.secs_model_name).reset()
            pred_16k = resample(pred_audio, output_sr, 16000)
            pred_16k_lens = (pred_lens / output_sr * 16000).to(torch.long)
            context_16k = resample(context_audio, context_sr, 16000)
            context_16k_lens = (context_lens / context_sr * 16000).to(torch.long)

            pred_16k = torch_rms_norm(pred_16k)
            context_16k = torch_rms_norm(context_16k)

            one_secs.update(
                name="dataset",
                target_audio=context_16k,
                target_audio_lens=context_16k_lens,
                pred_audio=pred_16k,
                pred_audio_lens=pred_16k_lens,
            )
            one_secs_metrics = metric_dict_to_jsonable(one_secs.compute())

        row["secs"] = safe_metric_scalar(one_secs_metrics, ["secs", "secs_dataset"])
        row["secs_metrics"] = one_secs_metrics
        del one_secs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_elapsed = time.time() - metric_start

    rank_metrics = {
        "rank": int(rank),
        "world_size": int(world_size),
        "num_processed": len({(x["run_id"], x["dataset_index"]) for x in metric_items}),
        "num_metric_items": int(len(metric_items)),
        "metric_elapsed_sec": float(metric_elapsed),
        "intelligibility": cer_wer,
        "secs": secs_scores,
    }

    return rank_metrics, filewise_rows


# -----------------------------
# Merge helpers
# -----------------------------


def compute_and_save_rank_metrics_file(args, rank_metrics: Dict[str, Any], rank: int):
    rank_path = os.path.join(args.out_dir, f"metrics_rank{rank:04d}.json")
    write_json(rank_path, rank_metrics)
    return rank_metrics


def merge_metrics_on_rank0(args, rank, world_size):
    if rank != 0:
        return None

    rank_metric_files = [os.path.join(args.out_dir, f"metrics_rank{r:04d}.json") for r in range(world_size)]

    rank_metrics = []
    for path in rank_metric_files:
        if not os.path.exists(path):
            logging.warning(f"Missing rank metric file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            rank_metrics.append(json.load(f))

    total_n = sum(int(m.get("num_metric_items", m.get("num_processed", 0))) for m in rank_metrics)

    def weighted_average(section: str):
        keys = set()
        for m in rank_metrics:
            keys.update(m.get(section, {}).keys())

        out = {}
        for k in sorted(keys):
            numerator = 0.0
            denominator = 0

            for m in rank_metrics:
                n = int(m.get("num_metric_items", m.get("num_processed", 0)))
                if n <= 0:
                    continue

                value = m.get(section, {}).get(k, None)
                if value is None or isinstance(value, str):
                    continue

                try:
                    value = float(value)
                except Exception:
                    continue

                numerator += value * n
                denominator += n

            if denominator > 0:
                out[k] = numerator / denominator

        return out

    final_metrics = {
        "world_size": int(world_size),
        "num_metric_items": int(total_n),
        "aggregation": "sum(rank_metric * rank_num_metric_items) / total_num_metric_items",
        "intelligibility": weighted_average("intelligibility"),
        "secs": weighted_average("secs"),
        "ranks": rank_metrics,
    }

    final_json_path = os.path.join(args.out_dir, "metrics_final.json")
    final_txt_path = os.path.join(args.out_dir, "metrics_final.txt")

    write_json(final_json_path, final_metrics)

    final_text = format_final_metric_text(final_metrics)
    write_text_atomic(final_txt_path, final_text)

    print("\n--- Final Evaluation Metrics ---", flush=True)
    print(final_text, flush=True)

    logging.info(f"Final metrics JSON saved to: {final_json_path}")
    logging.info(f"Final metrics TXT saved to: {final_txt_path}")
    logging.info(json.dumps(final_metrics, indent=2, sort_keys=True))

    return final_metrics


def merge_filewise_metrics_on_rank0(args, rank: int, world_size: int):
    """Merge per-turn rank metric rows into one row per original sample.

    Rank files still contain one row per turn because metrics are computed
    turn-by-turn. The final filewise outputs group those turn rows by
    (run_id, dataset_index), producing one JSONL/CSV row per original sample
    with list fields:
      reference_text, asr_hyp, cer_turns, wer_turns, secs_turns.

    DistributedSampler padding repeats are deduplicated by
    (run_id, dataset_index, turn_id), but repetitions from --num_eval_runs are
    preserved because run_id is part of the key.
    """
    if rank != 0 or not args.save_filewise_metrics:
        return []

    turn_rows = []

    for r in range(world_size):
        path = os.path.join(args.out_dir, f"filewise_metrics_rank{r:04d}.jsonl")
        if not os.path.exists(path):
            logging.warning(f"Missing filewise metrics file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turn_rows.append(json.loads(line))

    # Deduplicate DistributedSampler padding repeats, but preserve --num_eval_runs.
    deduped_turns = {}
    for row in turn_rows:
        run_id = int(row.get("run_id", 0))
        idx = int(row.get("dataset_index", -1))
        turn_id = int(row.get("turn_id", 0))
        key = (run_id, idx, turn_id)
        if key not in deduped_turns:
            deduped_turns[key] = row

    turn_rows = list(deduped_turns.values())

    # Group turn rows into one row per original file/sample.
    grouped = {}
    for row in turn_rows:
        run_id = int(row.get("run_id", 0))
        idx = int(row.get("dataset_index", -1))
        key = (run_id, idx)

        if key not in grouped:
            grouped[key] = {
                "run_id": run_id,
                "dataset_index": idx,
                "rank": int(row.get("rank", -1)),
                "target_audio_path": row.get("target_audio_path", ""),
                "context_audio_path": row.get("context_audio_path", ""),
                "turn_rows": [],
            }

        grouped[key]["turn_rows"].append(row)

    def avg(vals):
        vals = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
        return None if not vals else sum(vals) / len(vals)

    sample_rows = []
    for _, group in grouped.items():
        turns = sorted(group["turn_rows"], key=lambda x: int(x.get("turn_id", 0)))

        cer_turns = [r.get("cer") for r in turns]
        wer_turns = [r.get("wer") for r in turns]
        secs_turns = [r.get("secs") for r in turns]

        sample_row = {
            "run_id": group["run_id"],
            "dataset_index": group["dataset_index"],
            "rank": group["rank"],
            "num_turns": len(turns),
            "turn_ids": [int(r.get("turn_id", 0)) for r in turns],
            "target_audio_path": group["target_audio_path"],
            "context_audio_path": group["context_audio_path"],
            "pred_audio_paths": [r.get("pred_audio_path", "") for r in turns],
            "pred_audio_seconds_turns": [r.get("pred_audio_seconds") for r in turns],
            "reference_text": [r.get("reference_text", "") for r in turns],
            "asr_hyp": [r.get("asr_hyp", "") for r in turns],
            "cer_turns": cer_turns,
            "wer_turns": wer_turns,
            "secs_turns": secs_turns,
            "cer": avg(cer_turns),
            "wer": avg(wer_turns),
            "secs": avg(secs_turns),
        }

        sample_rows.append(sample_row)

    # Sort samples by average CER descending for failure analysis.
    sample_rows.sort(
        key=lambda x: (
            x.get("cer") is not None,
            float(x.get("cer")) if x.get("cer") is not None else -1.0,
        ),
        reverse=True,
    )

    jsonl_path = os.path.join(args.out_dir, "filewise_metrics_sorted_by_cer.jsonl")
    csv_path = os.path.join(args.out_dir, "filewise_metrics_sorted_by_cer.csv")

    write_jsonl(jsonl_path, sample_rows)
    write_filewise_csv(csv_path, sample_rows)

    logging.info(f"Saved sample-level filewise metrics JSONL to: {jsonl_path}")
    logging.info(f"Saved sample-level filewise metrics CSV to: {csv_path}")

    topk = min(int(args.filewise_metrics_topk_log), len(sample_rows))
    if topk > 0:
        logging.info(f"Top {topk} worst CER samples:")
        for row in sample_rows[:topk]:
            logging.info(
                "run_id=%s dataset_index=%s num_turns=%s cer=%s wer=%s secs=%s path=%s"
                % (
                    row.get("run_id"),
                    row.get("dataset_index"),
                    row.get("num_turns"),
                    row.get("cer"),
                    row.get("wer"),
                    row.get("secs"),
                    row.get("target_audio_path"),
                )
            )

    return sample_rows

def compute_aggregates_from_filewise_rows(rows: List[Dict[str, Any]]):
    """Aggregate over sample-level rows.

    Each row may internally contain multiple turn metrics in cer_turns/wer_turns,
    but the final filewise average is over original samples/files.
    """
    if len(rows) == 0:
        return {
            "cer": None,
            "wer": None,
            "secs": None,
            "num_samples": 0,
        }

    def avg_key(key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if len(vals) == 0:
            return None
        return sum(vals) / len(vals)

    return {
        "cer": avg_key("cer"),
        "wer": avg_key("wer"),
        "secs": avg_key("secs"),
        "num_samples": len(rows),
    }

def save_filewise_final_summary(args, filewise_rows: List[Dict[str, Any]]):
    filewise_summary = compute_aggregates_from_filewise_rows(filewise_rows)

    obj = {
        "aggregation": "mean_over_sample_metrics_each_sample_contains_turn_metric_lists",
        **filewise_summary,
    }

    path = os.path.join(args.out_dir, "metrics_final_filewise_average.json")
    write_json(path, obj)

    sample_metrics_final_path = os.path.join(args.out_dir, "metrics_final_sample_average.json")
    write_json(sample_metrics_final_path, obj)

    final_txt_path = os.path.join(args.out_dir, "metrics_final.txt")
    final_text = format_filewise_final_metric_text(filewise_summary)
    write_text_atomic(final_txt_path, final_text)

    print("\n--- Final Sample-Averaged Evaluation Metrics ---", flush=True)
    print(final_text, flush=True)

    logging.info(f"Filewise averaged final metrics saved to: {path}")
    logging.info(f"Sample averaged metrics_final JSON saved to: {sample_metrics_final_path}")
    logging.info(f"Final metrics TXT saved to: {final_txt_path}")

    return obj


# -----------------------------
# Args / main
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="EasyMagpieTTS Multi-GPU Inference Evaluation")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--codec_model_path", type=str, required=True)
    parser.add_argument("--datasets_json_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--phoneme_tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--inference_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--debug_dtype", action="store_true")
    parser.add_argument("--debug_gpu_assignment", action="store_true")
    parser.add_argument("--use_librosa", action="store_true")

    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_turns", type=int, default=1)
    parser.add_argument("--pad_factor_text_speech", type=int, default=10)

    parser.add_argument("--emulate_duplex_inference", action="store_true")
    parser.add_argument("--add_interruption_token", action="store_true")
    parser.add_argument("--force_interruption", action="store_true")
    parser.add_argument("--profile_multiturn_inference", action="store_true")
    parser.add_argument("--profile_pad_min_sec", type=float, default=2.0)
    parser.add_argument("--profile_pad_max_sec", type=float, default=2.0)
    parser.add_argument("--max_eval_turns", type=int, default=6)

    parser.add_argument("--user_custom_speaker_reference", action="store_true")
    parser.add_argument("--inference_speaker_reference", type=str, default=None)
    parser.add_argument("--language", type=str, default="en")

    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--max_tts_steps", type=int, default=2000)
    parser.add_argument("--force_speech_sil_codes", action="store_true")
    parser.add_argument("--normalize_volume", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)

    parser.add_argument(
        "--save_filewise_metrics",
        action="store_true",
        help="Save per-turn/file CER/WER metrics sorted by CER descending.",
    )
    parser.add_argument(
        "--compute_filewise_secs",
        action="store_true",
        help="Also compute per-turn/file SECS. Slower because it runs SECS per row.",
    )
    parser.add_argument(
        "--filewise_metrics_topk_log",
        type=int,
        default=20,
        help="Number of worst CER samples to print on rank 0.",
    )
    parser.add_argument(
        "--num_eval_runs",
        type=int,
        default=1,
        help="Repeat the full eval set N times. Repetitions are preserved in final filewise average.",
    )
    parser.add_argument(
        "--sort_by_text_token_count",
        action="store_true",
        help="Sort eval samples by total text token count before distributed sharding for better load balancing.",
    )
    parser.add_argument(
        "--metric_batch_size",
        type=int,
        default=8,
        help="Batch size used for post-generation ASR/SECS metric computation.",
    )
    parser.add_argument(
        "--max_metric_audio_sec",
        type=float,
        default=120.0,
        help="Clamp generated audio length used for ASR/SECS metrics to avoid metric OOM/hangs.",
    )
    parser.add_argument(
        "--asr_model_name",
        type=str,
        default="stt_en_fastconformer_transducer_large",
        help="Pretrained NeMo ASR model used for CER/WER.",
    )
    parser.add_argument(
        "--secs_model_name",
        type=str,
        default="titanet_large",
        help="Pretrained speaker encoder model used for SECS.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(get_audio_out_dir(args), exist_ok=True)
    os.makedirs(get_generated_turn_audio_dir(args), exist_ok=True)
    os.makedirs(get_context_metric_audio_dir(args), exist_ok=True)

    distributed, rank, local_rank, world_size, device_index = setup_distributed()

    if args.profile_multiturn_inference and args.batch_size != 1:
        raise RuntimeError(
            "--profile_multiturn_inference requires --batch_size=1 per process. "
            "Use multiple GPUs/processes for parallelism instead of increasing batch_size."
        )

    if args.profile_pad_max_sec < args.profile_pad_min_sec:
        raise RuntimeError("--profile_pad_max_sec must be >= --profile_pad_min_sec.")

    if args.num_eval_runs <= 0:
        raise RuntimeError("--num_eval_runs must be >= 1.")

    target_device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() and device_index >= 0 else "cpu")
    target_dtype = getattr(torch, args.inference_dtype)
    torch.set_default_dtype(target_dtype)

    hostname = socket.gethostname()
    cuda_name = torch.cuda.get_device_name(target_device) if torch.cuda.is_available() and device_index >= 0 else "cpu"

    all_rank_print(
        rank,
        f"host={hostname} local_rank={local_rank} world_size={world_size} "
        f"device={target_device} device_name={cuda_name}",
    )

    model = build_model_and_codec(args, target_device, target_dtype)
    codec_sil_codes = model.codec_sil_codes

    if args.debug_dtype:
        handles, stats, examples = attach_dtype_counter(model)
    else:
        handles = stats = examples = None

    full_eval_dataset = EvalJSONLDataset(args.datasets_json_path, num_turns=args.num_turns)
    # debug
    # full_eval_dataset.samples = full_eval_dataset.samples[:7]

    if args.sort_by_text_token_count:
        full_eval_dataset = SortedByTextTokenCountDataset(
            full_eval_dataset,
            model=model,
            max_eval_turns=args.max_eval_turns,
            descending=True,
        )

    collate_fn = partial(
        collate_and_tokenize_custom,
        model=model,
        extra_duration_thrshould=1.5,
        sample_rate=model.sample_rate,
        root_path=args.audio_dir,
        emulate_duplex_inference=args.emulate_duplex_inference,
        add_interruption_token=args.add_interruption_token,
        pad_factor_text_speech=args.pad_factor_text_speech,
        force_interruption=args.force_interruption,
        normalize_audio_volume=args.normalize_volume,
        use_librosa=args.use_librosa,
        profile_multiturn_inference=args.profile_multiturn_inference,
        max_eval_turns=args.max_eval_turns,
    )

    speaker_wav = load_speaker_wav_if_needed(args, model, target_dtype)

    generation_start = time.time()
    all_metric_items = []
    total_batches = 0
    total_generated_samples = 0

    for run_id in range(args.num_eval_runs):
        if distributed:
            sampler = DistributedSampler(
                full_eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            sampler.set_epoch(run_id)
        else:
            sampler = SequentialSampler(full_eval_dataset)

        if args.debug_gpu_assignment:
            try:
                assigned_indices = list(iter(sampler))
                assigned_dataset_indices = [
                    int(full_eval_dataset[i].get("__dataset_index__", -1)) for i in assigned_indices
                ]
                all_rank_print(
                    rank,
                    f"run_id={run_id} assigned {len(assigned_dataset_indices)} / {len(full_eval_dataset)} "
                    f"samples to gpu={local_rank}: dataset_indices={assigned_dataset_indices}",
                )
            except Exception as e:
                all_rank_print(rank, f"Could not print assigned indices: {repr(e)}")

        dataloader = DataLoader(
            dataset=full_eval_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        for batch_id, inputs in enumerate(dataloader):
            total_batches += 1
            batch_indices = inputs.get("dataset_indices", [])
            total_generated_samples += len(batch_indices)

            if args.debug_gpu_assignment:
                all_rank_print(
                    rank,
                    f"run_id={run_id} gpu={local_rank} batch_id={batch_id} "
                    f"dataset_indices={batch_indices} text_token_counts={inputs.get('text_token_counts', [])} "
                    f"target_paths={inputs.get('target_audio_paths', [])}",
                )

            inputs = prepare_inputs_for_device(inputs, model, args, target_dtype, speaker_wav=speaker_wav)

            finalize_output, profile_turn_frame_ranges, profile_decode_start_frame = run_generation(
                model=model,
                inputs=inputs,
                args=args,
                codec_sil_codes=codec_sil_codes,
            )

            metric_items = save_generation_outputs_and_build_metric_items(
                model=model,
                inputs=inputs,
                finalize_output=finalize_output,
                profile_turn_frame_ranges=profile_turn_frame_ranges,
                profile_decode_start_frame=profile_decode_start_frame,
                args=args,
                rank=rank,
                run_id=run_id,
            )
            all_metric_items.extend(metric_items)

            if args.debug_dtype and batch_id == 0 and run_id == 0:
                report_dtype_stats(handles, stats, examples, rank=rank)

    generation_elapsed = time.time() - generation_start

    # Save pre-metric manifest for debugging and restartability.
    metric_manifest_path = os.path.join(args.out_dir, f"metric_items_rank{rank:04d}.jsonl")
    write_jsonl(metric_manifest_path, all_metric_items)

    # Free TTS/codec model memory before loading ASR and speaker encoder metrics.
    del model
    if speaker_wav is not None:
        del speaker_wav
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_rank_print(
        rank,
        f"generation done: batches={total_batches} generated_samples_with_sampler_padding={total_generated_samples} "
        f"metric_items={len(all_metric_items)} elapsed_sec={generation_elapsed:.2f}. "
        "Loading ASR/SECS metrics now.",
    )

    rank_metrics, rank_filewise_rows = compute_metrics_after_generation(
        args=args,
        rank=rank,
        world_size=world_size,
        metric_items=all_metric_items,
    )
    rank_metrics["generation_elapsed_sec"] = float(generation_elapsed)
    rank_metrics["num_generated_samples_with_sampler_padding"] = int(total_generated_samples)

    rank_metrics = compute_and_save_rank_metrics_file(args, rank_metrics, rank)
    all_rank_print(rank, f"saved rank metrics: {json.dumps(rank_metrics, sort_keys=True)}")

    if args.save_filewise_metrics:
        rank_filewise_rows.sort(
            key=lambda x: (
                x.get("cer") is not None,
                float(x.get("cer")) if x.get("cer") is not None else -1.0,
            ),
            reverse=True,
        )

        rank_filewise_path = os.path.join(args.out_dir, f"filewise_metrics_rank{rank:04d}.jsonl")
        write_jsonl(rank_filewise_path, rank_filewise_rows)
        all_rank_print(rank, f"saved filewise metrics: {rank_filewise_path}")

    if rank == 0:
        wait_for_rank_metric_files(args, world_size)

    merge_metrics_on_rank0(args, rank, world_size)

    if args.save_filewise_metrics:
        if rank == 0:
            wait_for_rank_filewise_metric_files(args, world_size)

        filewise_rows = merge_filewise_metrics_on_rank0(args, rank, world_size)

        if rank == 0:
            save_filewise_final_summary(args, filewise_rows)

    cleanup_distributed()


if __name__ == "__main__":
    main()
