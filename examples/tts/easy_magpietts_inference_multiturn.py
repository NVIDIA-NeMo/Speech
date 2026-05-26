# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Evaluation script for custom EasyMagpieTTS models.
Features explicit Duplex (10x Padding) and Regular (Turn-by-turn) multi-turn modes.

Usage:
    python easy_magpietts_eval.py \
        --checkpoint_path=/path/to/magpie/model.ckpt \
        --codec_model_path=/path/to/codec/model.ckpt \
        --datasets_json_path=/path/to/evalset_config.jsonl \
        --out_dir=/path/to/out/audio \
        --batch_size=6 \
        --use_cfg \
        --use_librosa
"""

import argparse
import json
import os
from copy import deepcopy
from functools import partial

import librosa
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf, open_dict

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.utils import logging

# --- EasyMagpieTTS Imports ---
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

from nemo.collections.tts.parts.utils.tts_dataset_utils import normalize_volume

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def torch_rms_norm(wav, db_level=-27.0):
    r = 10 ** (db_level / 20)
    a = torch.sqrt((wav.size(-1) * (r**2)) / torch.sum(wav**2))
    return wav * a

def attach_dtype_counter(model):
    handles = []
    stats = {}
    examples = {}

    def is_leaf(module):
        return len(list(module.children())) == 0

    def get_dtype(x):
        if torch.is_tensor(x):
            return str(x.dtype)
        elif isinstance(x, (list, tuple)):
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
                    "torch.float16": 0, "torch.bfloat16": 0, "torch.float32": 0, "other": 0,
                }
                examples[group] = {
                    "torch.float16": [], "torch.bfloat16": [], "torch.float32": [], "other": [],
                }
            stats[group][dtype] += 1
            if len(examples[group][dtype]) < 3:
                examples[group][dtype].append(module.__class__.__name__)
        return fn

    for name, module in model.named_modules():
        if is_leaf(module):
            handles.append(module.register_forward_hook(hook_fn(name)))
    return handles, stats, examples


def report_dtype_stats(handles, stats, examples):
    for h in handles:
        h.remove()
    logging.info("\n=== DTYPE USAGE PER MODULE ===")
    for group, group_stats in stats.items():
        total = sum(group_stats.values())
        if total == 0: continue
        logging.info(f"\n--- {group} ---")
        for dtype, count in group_stats.items():
            if count > 0:
                logging.info(f"{dtype}: {count} ({100*count/total:.2f}%)")
    logging.info("\n=== EXAMPLES ===")
    for group, group_examples in examples.items():
        logging.info(f"\n--- {group} ---")
        for dtype, mods in group_examples.items():
            if mods:
                logging.info(f"{dtype}: {mods}")


class EvalJSONLDataset(Dataset):
    def __init__(self, file_path, num_turns=1):
        self.samples = []
        raw_samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    raw_samples.append(json.loads(line))
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
                if speaker not in single_turn_by_speaker:
                    single_turn_by_speaker[speaker] = []
                single_turn_by_speaker[speaker].append(sample)

        for speaker, speaker_samples in single_turn_by_speaker.items():
            buffer_texts, buffer_paths = [], []
            first_sample_meta = None

            for sample in speaker_samples:
                if not buffer_texts:
                    first_sample_meta = dict(sample)
                buffer_texts.append(sample["text"])
                buffer_paths.append(sample.get("audio_filepath", ""))

                if len(buffer_texts) == num_turns:
                    first_sample_meta["text"] = buffer_texts
                    base_names = [os.path.splitext(os.path.basename(p))[0] for p in buffer_paths if p]
                    ext = os.path.splitext(buffer_paths[-1])[1] if buffer_paths[-1] else ""
                    combined_name = "_".join(base_names) + ext
                    dir_name = os.path.dirname(first_sample_meta.get("audio_filepath", ""))
                    if dir_name:
                        first_sample_meta["audio_filepath"] = os.path.join(dir_name, combined_name)
                    else:
                        first_sample_meta["audio_filepath"] = combined_name

                    self.samples.append(first_sample_meta)
                    buffer_texts, buffer_paths, first_sample_meta = [], [], None

            if buffer_texts:
                first_sample_meta["text"] = buffer_texts
                base_names = [os.path.splitext(os.path.basename(p))[0] for p in buffer_paths if p]
                ext = os.path.splitext(buffer_paths[-1])[1] if buffer_paths[-1] else ""
                combined_name = "_".join(base_names) + ext
                dir_name = os.path.dirname(first_sample_meta.get("audio_filepath", ""))
                if dir_name:
                    first_sample_meta["audio_filepath"] = os.path.join(dir_name, combined_name)
                else:
                    first_sample_meta["audio_filepath"] = combined_name
                self.samples.append(first_sample_meta)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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

    # --- MULTI-TURN MODE DECISION ---
    is_profile = profile_multiturn_inference
    is_duplex = emulate_duplex_inference and not is_profile

    out_dict = {
        "duplex_multiturn": is_duplex,
        "regular_multiturn": (not is_duplex) and (not is_profile),
        "profile_multiturn": is_profile,
    }

    tokenized_list = []
    batched_turns = []
    batched_turn_lens = []
    valid_turn_masks = []
    
    if is_duplex:
        # -------------------------------------------------------------
        # DUPLEX MODE (Continuous sequence with 10x pad injection)
        # -------------------------------------------------------------
        for s in batch:
            text_data = s["text"]

            if isinstance(text_data, list):
                full_ids = []
                for segment in text_data:
                    seg_ids = model.tokenizer.encode(segment, tokenizer_name=main_tokenizer_name) + [model.eos_id]
                    seg_len = len(seg_ids)
                    pad_len = seg_len * pad_factor_text_speech
                    pad_ids = [model.pad_id] * pad_len

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
                            eos_idx = 0
                            pad_ids[eos_idx] = model.interruption_token_id
                    else:
                        if add_interruption_token:
                            eos_idx = int(len(pad_ids) * 0.7)
                            pad_ids[eos_idx] = model.interruption_token_id

                    full_ids.extend(seg_ids)
                    full_ids.extend(pad_ids)

                tokenized_list.append(torch.as_tensor(full_ids, dtype=torch.long))
            else:
                tokenized_list.append(
                    torch.as_tensor(model.tokenizer.encode(text_data, tokenizer_name=main_tokenizer_name) + [model.eos_id], dtype=torch.long)
                )

        pad_len = 25
        prefix = torch.full((pad_len,), model.pad_id, dtype=torch.long)
        for i in range(len(tokenized_list)):
            tokenized_list[i] = torch.cat([prefix, tokenized_list[i]])
        input_lengths = torch.tensor([len(x) for x in tokenized_list], dtype=torch.long)

        input_ids = pad_sequence(tokenized_list, batch_first=True, padding_value=model.pad_id)

        out_dict["input_ids"] = input_ids
        out_dict["input_lengths"] = input_lengths

    else:
        # -------------------------------------------------------------
        # REGULAR MODE (Turn-by-turn discrete packaging)
        # -------------------------------------------------------------
        max_turns = 1
        for s in batch:
            if isinstance(s["text"], list):
                max_turns = max(max_turns, len(s["text"]))
                
        for t in range(max_turns):
            turn_t_tokens = []
            turn_t_lens = []
            turn_t_valid = []
            
            for s in batch:
                text_data = s["text"]
                if isinstance(text_data, list):
                    if t < len(text_data):
                        seg_ids = model.tokenizer.encode(text_data[t], tokenizer_name=main_tokenizer_name) + [model.eos_id]
                        turn_t_tokens.append(torch.as_tensor(seg_ids, dtype=torch.long))
                        turn_t_lens.append(len(seg_ids))
                        turn_t_valid.append(True)
                    else:
                        # Dummy pad to keep shapes consistent for items with fewer turns
                        turn_t_tokens.append(torch.as_tensor([model.pad_id], dtype=torch.long))
                        turn_t_lens.append(1)
                        turn_t_valid.append(False)
                else:
                    if t == 0:
                        seg_ids = model.tokenizer.encode(text_data, tokenizer_name=main_tokenizer_name) + [model.eos_id]
                        turn_t_tokens.append(torch.as_tensor(seg_ids, dtype=torch.long))
                        turn_t_lens.append(len(seg_ids))
                        turn_t_valid.append(True)
                    else:
                        turn_t_tokens.append(torch.as_tensor([model.pad_id], dtype=torch.long))
                        turn_t_lens.append(1)
                        turn_t_valid.append(False)
                        
            padded_turn_t = pad_sequence(turn_t_tokens, batch_first=True, padding_value=model.pad_id)
            batched_turns.append(padded_turn_t)
            batched_turn_lens.append(torch.tensor(turn_t_lens, dtype=torch.long))
            valid_turn_masks.append(torch.tensor(turn_t_valid, dtype=torch.bool))
            
        out_dict["batched_turns"] = batched_turns
        out_dict["batched_turn_lens"] = batched_turn_lens
        out_dict["valid_turn_masks"] = valid_turn_masks

    # --- AUDIO LOADING ---
    audio_list = []
    audio_lengths = []
    target_num_frames = []

    max_turns_for_user_audio = len(batched_turns) if (not is_duplex) else 0

    if is_profile and max_turns_for_user_audio > 0:
        user_audio_by_turn = [[] for _ in range(max_turns_for_user_audio)]
        user_audio_lens_by_turn = [[] for _ in range(max_turns_for_user_audio)]
    else:
        user_audio_by_turn = []
        user_audio_lens_by_turn = []

    for i, s in enumerate(batch):
        audio_path = _resolve_audio_path(s.get("context_audio_filepath"), root_path)
        wav = _load_audio(
            audio_path,
            sample_rate,
            normalize=normalize_audio_volume,
            use_librosa=use_librosa,
        )

        audio_list.append(wav)
        audio_lengths.append(len(wav))

        # Optional per-turn user audio.
        # Expected JSONL field:
        #   "user_audio": ["turn0_user.wav", "turn1_user.wav", ...]
        if is_profile and max_turns_for_user_audio > 0:
            user_audio_paths = s.get("user_audio_file_path", None)

            for t in range(max_turns_for_user_audio):
                has_valid_text_turn = (
                    isinstance(s["text"], list) and t < len(s["text"])
                ) or (
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
                    print("User audio not founded, using silence two seconds audio")
                    ua_wav = torch.zeros(int(2 * sample_rate), dtype=torch.float32)

                user_audio_by_turn[t].append(ua_wav)
                user_audio_lens_by_turn[t].append(len(ua_wav))

        tdur_audio_path = _resolve_audio_path(s["audio_filepath"], root_path)

        if tdur_audio_path and os.path.exists(tdur_audio_path):
            wav_dur = _load_audio(tdur_audio_path, sample_rate, normalize=normalize_audio_volume, use_librosa=use_librosa)
            tdur = wav_dur.shape[0] // model.input_samples_per_frame
            target_num_frames.append(tdur * extra_duration_thrshould)
        else:
            # Fallback estimation
            if is_duplex:
                current_text_len = len(tokenized_list[i])
                if isinstance(s["text"], list):
                    target_num_frames.append(current_text_len)
                else:
                    target_num_frames.append(current_text_len * 5)
            else:
                target_num_frames.append(sum([l[i].item() for l in batched_turn_lens]) * 5)

    max_audio_len = max(audio_lengths)
    B = len(audio_lengths)
    padded_audio = torch.zeros((B, max_audio_len), dtype=torch.float32)

    for i, wav in enumerate(audio_list):
        padded_audio[i, : len(wav)] = wav


    if is_profile and max_turns_for_user_audio > 0:
        padded_user_audio_turns = []
        padded_user_audio_turns_lens = []

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

    out_dict["context_audio"] = padded_audio
    out_dict["context_audio_lengths"] = torch.tensor(audio_lengths, dtype=torch.long)
    out_dict["target_audio_paths"] = [s["audio_filepath"] for s in batch]
    out_dict["target_num_frames"] = target_num_frames

    out_dict["raw_text"] = [" ".join(s["text"]) if isinstance(s["text"], list) else s["text"] for s in batch]

    return out_dict

def _mix_user_turns_on_timeline(user_audio_turns, user_audio_turns_lens, sample_rate):
    total_len = int(sum(x.item() for x in user_audio_turns_lens))
    mixed = torch.zeros(total_len, dtype=torch.float32)

    offset = 0
    for wav, wav_len in zip(user_audio_turns, user_audio_turns_lens):
        wav_len = int(wav_len.item())
        mixed[offset : offset + wav_len] = wav[:wav_len]
        offset += wav_len

    return mixed

def main():
    parser = argparse.ArgumentParser(description="EasyMagpieTTS Inference Evaluation")
    
    # Required Paths
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the EasyMagpie model")
    parser.add_argument("--codec_model_path", type=str, required=True, help="Path to the audio codec")
    parser.add_argument("--datasets_json_path", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save audio outputs")
    
    # Optional Paths & General
    parser.add_argument("--phoneme_tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None, help="Root dir for audio paths in JSONL")
    parser.add_argument("--inference_dtype", type=str, default="float32")
    parser.add_argument("--debug_dtype", action="store_true")
    parser.add_argument("--use_librosa", action="store_true", help="Use librosa instead of soundfile+torch for audio load")
    
    # Dataloader & Batching
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_turns", type=int, default=1)
    parser.add_argument("--pad_factor_text_speech", type=int, default=10)
    
    # Text Processing Boolean Flags
    parser.add_argument("--emulate_duplex_inference", action="store_true")
    parser.add_argument("--add_interruption_token", action="store_true")
    parser.add_argument("--force_interruption", action="store_true")
    parser.add_argument("--profile_multiturn_inference", action="store_true")
    parser.add_argument("--profile_pad_min_sec", type=float, default=2.0)
    parser.add_argument("--profile_pad_max_sec", type=float, default=2.0)
    parser.add_argument(
        "--max_eval_turns",
            type=int,
            default=6,
            help="Maximum number of turns to evaluate per sample. None means use all turns.",
    )


    # Speaker & Prompt Configurations
    parser.add_argument("--user_custom_speaker_reference", action="store_true")
    parser.add_argument("--inference_speaker_reference", type=str, default=None)
    parser.add_argument("--language", type=str, default="en")
    
    # Generation Kwargs
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--max_tts_steps", type=int, default=2000)
    parser.add_argument("--force_speech_sil_codes", action="store_true")
    parser.add_argument("--normalize_volume", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)

    args = parser.parse_args()

    if args.profile_multiturn_inference and args.batch_size != 1:
        raise RuntimeError("--profile_multiturn_inference currently requires --batch_size=1.")

    if args.profile_pad_max_sec < args.profile_pad_min_sec:
        raise RuntimeError("--profile_pad_max_sec must be >= --profile_pad_min_sec.")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    target_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    target_dtype = getattr(torch, args.inference_dtype)
    torch.set_default_dtype(target_dtype)

    model_cfg = EasyMagpieTTSInferenceModel.restore_from(args.checkpoint_path, return_config=True)
    with open_dict(model_cfg):
        model_cfg.target = 'nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel'
        model_cfg.codecmodel_path = args.codec_model_path
        model_cfg.train_ds = None
        model_cfg.validation_ds = None
        model_cfg.run_val_inference = False
        model_cfg.use_utmos = False
        model_cfg.use_meta_init_for_decoder = True

        # Guarantees silence for pad tokens
        model_cfg.use_multiturn_dataset = True 
        
        if args.phoneme_tokenizer_path and getattr(model_cfg, "phoneme_tokenizer", None) is not None:
            model_cfg.phoneme_tokenizer.tokenizer_path = args.phoneme_tokenizer_path

    # Load to CPU first to prevent OOM
    model = EasyMagpieTTSInferenceModel.restore_from(
        args.checkpoint_path, override_config_path=model_cfg, map_location=torch.device("cpu")
    )
    model.use_kv_cache_for_inference = True
    model.to(dtype=target_dtype)
    model.eval().to(target_device)

    # --- DATALOADER COMPATIBILITY PATCHES ---
    model.input_samples_per_frame = int(model.codec_model_samples_per_frame * model.frame_stacking_factor)
    model.target_samples_per_frame = model.input_samples_per_frame / (model.sample_rate / model.output_sample_rate)

    # Load to CPU first to prevent OOM
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

    if not hasattr(model, "_codec_helper") or model._codec_helper is None:
        model._codec_helper = CodecHelper(codec_model=codec_model, codec_converter=codec_converter)

    model._generate_codec_silence_buffer()
    codec_sil_codes = model.codec_sil_codes

    if args.debug_dtype:
        handles, stats, examples = attach_dtype_counter(model)

    with fp32_precision():
        intelligibility = Intelligibility("stt_en_fastconformer_transducer_large", reuse_asr_hyps=False).reset()
        secs_metric = SECS("titanet_large").reset()

    eval_dataset = EvalJSONLDataset(args.datasets_json_path, num_turns=args.num_turns)
    # debug
    # eval_dataset.samples = eval_dataset.samples[:100]

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

    dataloader = DataLoader(
        dataset=eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False,
    )

    if args.user_custom_speaker_reference and args.inference_speaker_reference:
        speaker_wav = _load_audio(
            args.inference_speaker_reference,
            model.sample_rate,
            normalize=args.normalize_volume,
            use_librosa=args.use_librosa,
        ).unsqueeze(0).to(model.device, dtype=target_dtype)

    for batch_id, inputs in enumerate(dataloader):
        B = inputs["context_audio"].size(0)
        device = model.device

        inputs["context_audio"] = inputs["context_audio"].to(device, dtype=target_dtype)
        inputs["context_audio_lengths"] = inputs["context_audio_lengths"].to(device)

        if args.user_custom_speaker_reference and args.inference_speaker_reference:
            inputs["context_audio"] = speaker_wav.repeat(B, 1).detach()
            inputs["context_audio_lengths"] = torch.full((B,), speaker_wav.size(-1), dtype=torch.long, device=device)

        if "user_audio_turns" in inputs:
            inputs["user_audio_turns"] = [
                x.to(device, dtype=target_dtype) for x in inputs["user_audio_turns"]
            ]
            inputs["user_audio_turns_lens"] = [
                x.to(device) for x in inputs["user_audio_turns_lens"]
            ]

        profile_turn_frame_ranges = []
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
            # ---------------------------------------------------------
            # MODE 1: DUPLEX (Continuous Padding Token Stream)
            # ---------------------------------------------------------
            if inputs["duplex_multiturn"]:
                text = inputs["input_ids"].to(device)
                text_lens = inputs["input_lengths"].to(device)

                # Trackers for our two forced-silence zones
                in_initial_silence = torch.ones(B, dtype=torch.bool, device=device)
                in_post_speech_silence = torch.zeros(B, dtype=torch.bool, device=device)

                text_exhausted = state.text_tokens_seen >= text_lens
                while not text_exhausted.all():
                    # 1. WAKE UP OVERRIDE: Keep the text pipeline awake to read pads!
                    state.finished = state.finished & text_exhausted
                    state.text_finished = state.text_finished & text_exhausted
                    if hasattr(state, "phoneme_stream_ended"):
                        state.phoneme_stream_ended = state.phoneme_stream_ended & text_exhausted
                    
                    # 2. Safely index text using the model's internal pointer
                    positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                    current_tokens = text[torch.arange(B, device=device), positions]

                    current_tokens = torch.where(
                        text_exhausted, torch.full_like(current_tokens, model.eos_id), current_tokens
                    )

                    # 3. Update our trackers BEFORE the step
                    is_pad_or_eos = (current_tokens == model.pad_id) | (current_tokens == model.eos_id)
                    in_initial_silence = in_initial_silence & is_pad_or_eos
                    in_post_speech_silence = in_post_speech_silence & is_pad_or_eos

                    # 4. Step the model
                    state, audio_codes, _ = model.streaming_step(state=state, text_tokens=current_tokens, use_inference_mode=True)
                    
                    # 5. SILENCE FORCING INJECTION
                    if audio_codes is not None and args.force_speech_sil_codes:
                        force_silence_mask = in_initial_silence | in_post_speech_silence

                        if force_silence_mask.any():
                            # Expand silence codes [C] -> [1, C, 1] to match audio_codes [B, C, 1]
                            expanded_sil = codec_sil_codes.view(1, -1, 1).expand_as(audio_codes)
                            # Expand mask [B] -> [B, 1, 1] for broadcasting
                            mask_3d = force_silence_mask.view(B, 1, 1)
                            # Overwrite the prediction with silence codes where the mask is True.
                            overwritten_codes = torch.where(mask_3d, expanded_sil, audio_codes)
                            # Inject back into the model's KV cache history
                            state.all_predictions[-1] = overwritten_codes

                    # 6. TRIGGER POST-SPEECH SILENCE FOR THE *NEXT* FRAME
                    in_post_speech_silence = in_post_speech_silence | state.finished

                    # Update exhaustion tracker for the next iteration
                    text_exhausted = state.text_tokens_seen >= text_lens

            # ---------------------------------------------------------
            # MODE 2: REGULAR (Turn-by-Turn Re-wakes)
            # ---------------------------------------------------------
            elif inputs["regular_multiturn"]:
                batched_turns = inputs["batched_turns"]
                batched_turn_lens = inputs["batched_turn_lens"]
                valid_turn_masks = inputs["valid_turn_masks"]
                
                max_turns = len(batched_turns)
                turn_offsets = torch.zeros(B, dtype=torch.long, device=device)
                
                for t in range(max_turns):
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
                        current_tokens = torch.where(exhausted, torch.full_like(current_tokens, model.eos_id), current_tokens)
                        
                        state, _, _ = model.streaming_step(state=state, text_tokens=current_tokens, use_inference_mode=True)

            # ---------------------------------------------------------
            # MODE 3: PROFILE MULTI-TURN
            # ---------------------------------------------------------
            elif inputs["profile_multiturn"]:
                if B != 1:
                    raise RuntimeError(
                        "--profile_multiturn_inference currently supports only batch_size=1. "
                        "Use --batch_size=1 for this mode."
                    )

                batched_turns = inputs["batched_turns"]
                batched_turn_lens = inputs["batched_turn_lens"]
                valid_turn_masks = inputs["valid_turn_masks"]

                max_turns = len(batched_turns)
                prev_turn_ended_with_audio_eos = True  # profile before turn 0
                for t in range(max_turns):
                    turn_ended_with_audio_eos = False
                    turn_text = batched_turns[t].to(device)
                    turn_lens = batched_turn_lens[t].to(device)
                    valid_mask = valid_turn_masks[t].to(device)

                    if not bool(valid_mask[0].item()):
                        continue

                    # Re-open stream for this turn.
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
                        # Prefill on the begining of each turn
                        if "user_audio_turns" in inputs:
                            profile_T = int(round(inputs["user_audio_turns"][t].size(-1) / model.input_samples_per_frame))
                            profile_seconds = profile_T * model.input_samples_per_frame / model.sample_rate
                        else:
                            profile_seconds = (
                                args.profile_pad_min_sec
                                + torch.rand((), device=device).item()
                                * (args.profile_pad_max_sec - args.profile_pad_min_sec)
                            )
                            profile_T = max(
                                1,
                                int(round(profile_seconds * model.sample_rate / model.input_samples_per_frame)),
                            )

                        profile_tokens = torch.full(
                            (1, profile_T),
                            model.pad_id,
                            dtype=torch.long,
                            device=device,
                        )
                        user_audio_channel_embedding = None
                    else:
                        user_audio_channel_embedding = None
                        if "user_audio_turns" in inputs:
                            user_audio = inputs["user_audio_turns"][t]
                            user_audio_lens = inputs["user_audio_turns_lens"][t]
                        else:
                            print("Warning!! USING CONTEXT AUDIO AS USER AUDIO FOR TESTING !!")
                            user_audio = inputs["context_audio"]
                            user_audio_lens = inputs["context_audio_lengths"]

                        user_audio_codes, user_audio_codes_lens = model._codec_helper.audio_to_codes(
                            user_audio,
                            user_audio_lens,
                        )

                        if model._codec_converter is not None:
                            user_audio_codes = model._codec_converter.convert_original_to_new(
                                audio_tokens=user_audio_codes, audio_lens=user_audio_codes_lens
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

                        # Remove BOS/EOS from the user-audio turn, same as training.
                        if boundary_trim == 0:
                            real_start = 0
                            real_end = int(user_audio_codes_lens[0].item())
                        else:
                            turn_len_with_special = int(user_audio_codes_lens[0].item())
                            real_start = 1
                            real_end = max(real_start, turn_len_with_special - 1)

                        user_audio_embedded = user_audio_embedded[:, real_start:real_end]

                        # Optional: trim boundaries exactly like training.
                        copy_len = user_audio_embedded.size(1)

                        if boundary_trim > 0:
                            trim = min(boundary_trim, copy_len // 2)

                            if trim > 0:
                                user_audio_embedded[:, :trim] = 0.0
                                user_audio_embedded[:, copy_len - trim:] = 0.0

                        # Add BOS-aligned zero frame, because audio input timeline has BOS at t=0.
                        bos_user_pad = torch.zeros(
                            user_audio_embedded.size(0),
                            1,
                            user_audio_embedded.size(2),
                            device=user_audio_embedded.device,
                            dtype=user_audio_embedded.dtype,
                        )
                        user_audio_embedded = torch.cat([bos_user_pad, user_audio_embedded], dim=1)

                        profile_T = user_audio_embedded.size(1)
                        profile_tokens = torch.full(
                            (B, profile_T),
                            model.pad_id,
                            dtype=torch.long,
                            device=device,
                        )

                        user_audio_channel_embedding = user_audio_embedded
                        profile_seconds = profile_T * model.input_samples_per_frame / model.sample_rate

                    # add text tokens needed for profilling
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
                            user_audio_channel_embedding=user_audio_channel_embedding
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

                    logging.info(
                        f"[profile_multiturn] turn={t} prefilled {profile_T} steps "
                        f"({profile_seconds:.2f}s)"
                    )

                    turn_start_frame = sum(p.size(-1) for p in state.all_predictions)
                    if t == 0:
                        state.audio_prediction_start_idx.fill_(turn_start_frame)
                        profile_decode_start_frame = turn_start_frame

                    turn_offset = state.text_tokens_seen.clone()
                    turn_steps = 0
                    saw_audio = False
                    first_audio_step_finished = False

                    turn_text_done = False

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
                            first_audio_step_finished = bool(state.finished[0].item())

                        if bool(text_exhausted[0].item()) and bool(state.finished[0].item()):
                            turn_ended_with_audio_eos = True
                            break

                    prev_turn_ended_with_audio_eos = turn_ended_with_audio_eos

                    # keep generated codes, but don't let this turn's EOS crop finalize output
                    state.audio_prediction_end_idx.fill_(-1)
                    state.finished.zero_()

                    logging.info(
                        f"[profile_multiturn] turn={t} steps={turn_steps} "
                        f"saw_audio={saw_audio} immediate_eos={prev_turn_ended_with_audio_eos}"
                    )
                    turn_end_frame = sum(p.size(-1) for p in state.all_predictions)
                    profile_turn_frame_ranges.append((t, turn_start_frame, turn_end_frame))

            # if state.audio_prediction_end_idx[0].item() >= 0:
            #     last_audio_prediction_end_idx.copy_(state.audio_prediction_end_idx)

            # Scrub Special Tokens (BOS/EOS) from Audio Codes ---
            # Because we force-decode the entire uncropped sequence, any BOS or EOS 
            # tokens left in the array will produce loud artifacts in the codec.
            bos_id = getattr(model, "audio_bos_id", -1)
            eos_id = getattr(model, "audio_eos_id", -1)
            speaking_id = getattr(model, "audio_user_speaking_id", -1)
            speaking_end_id = getattr(model, "audio_user_speaking_end_id", -1)

            sil_injection = codec_sil_codes.view(1, -1, 1)

            for step_idx in range(len(state.all_predictions)):
                pred = state.all_predictions[step_idx]
                # Check if any codebook in the frame has any special token
                mask = (pred == bos_id) | (pred == eos_id) | (pred == speaking_id) | (pred == speaking_end_id)
                frame_mask = mask.any(dim=1, keepdim=True) 

                if frame_mask.any():
                    state.all_predictions[step_idx] = torch.where(
                        frame_mask, 
                        sil_injection.expand_as(pred), 
                        pred
                    )

            if inputs["duplex_multiturn"]:
                # Erase the internal memory of Turn 1's EOS token so `streaming_finalize` 
                # decodes the entire physical sequence!
                state.audio_prediction_end_idx.fill_(-1)
            
            if inputs["profile_multiturn"]:
                state.audio_prediction_end_idx.fill_(-1)

            # Finalize decodes the collected Codec states globally regardless of which loop was run
            finalize_output = model.streaming_finalize(state, use_inference_mode=True)

        if args.debug_dtype and batch_id == 0:
            report_dtype_stats(handles, stats, examples)

        with fp32_precision():
            audio_f32 = finalize_output.audio.float()
            audio_len = finalize_output.audio_len.int()

            expected_audio_lens = (torch.tensor(inputs["target_num_frames"], device=device) * model.target_samples_per_frame).int()
            
            if inputs["duplex_multiturn"]:
                # Use exact math based on the output samples multiplier!
                audio_len = (text_lens * model.target_samples_per_frame).int()
                
                # Cap the expected length so it physically cannot exceed the actual generated tensor size
                audio_len = torch.min(audio_len, torch.tensor(audio_f32.size(1), device=device))
            elif inputs["profile_multiturn"]:
                audio_len = finalize_output.audio_len.int()
            else:
                audio_len = torch.min(audio_len, expected_audio_lens)

            metric_audio_pred = resample(audio_f32, model.output_sample_rate, 16000)
            metric_audio_pred_lens = (audio_len / model.output_sample_rate * 16000).to(torch.long)

            context_audio = resample(inputs["context_audio"].float(), model.sample_rate, 16000)
            context_audio_lens = (inputs["context_audio_lengths"] / model.sample_rate * 16000).to(torch.long)

            # normalize volume
            metric_audio_pred = torch_rms_norm(metric_audio_pred)
            context_audio = torch_rms_norm(context_audio)

            intelligibility.update(
                name="dataset",
                refs=inputs["raw_text"],
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
                asr_hyps=None,
            )

            secs_metric.update(
                name="dataset",
                target_audio=context_audio,
                target_audio_lens=context_audio_lens,
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
            )

            os.makedirs(args.out_dir, exist_ok=True)
            audio_f32 = audio_f32.detach().cpu()
            audio_len = audio_len.cpu()

            for i in range(B):
                target_path = inputs["target_audio_paths"][i]
                base_name = os.path.basename(target_path)
                stem, ext = os.path.splitext(base_name)
                if not ext:
                    ext = ".wav"

                if inputs["profile_multiturn"]:
                    full_len = int(audio_len[i].item())
                    full_wav_t = audio_f32[i, :full_len].detach().cpu().float()

                    samples_per_prediction_frame = (
                        model.codec_model_samples_per_frame
                        / (model.sample_rate / model.output_sample_rate)
                    )

                    # Build artifact-free aligned agent audio:
                    # start from true zeros and copy only generated turn regions.
                    aligned_agent = torch.zeros_like(full_wav_t)

                    print(profile_turn_frame_ranges)

                    for turn_id, start_frame, end_frame in profile_turn_frame_ranges:
                        rel_start_frame = start_frame - profile_decode_start_frame
                        rel_end_frame = end_frame - profile_decode_start_frame

                        start_sample = int(round(rel_start_frame * samples_per_prediction_frame))
                        end_sample = int(round(rel_end_frame * samples_per_prediction_frame))

                        start_sample = max(0, min(start_sample, full_len))
                        end_sample = max(start_sample, min(end_sample, full_len))

                        print(
                            "Turn:", turn_id,
                            "Start:", start_sample,
                            "End:", end_sample,
                            "Start S:", start_sample / model.output_sample_rate,
                            "End S:", end_sample / model.output_sample_rate,
                        )

                        # Copy only this turn into the aligned full output.
                        aligned_agent[start_sample:end_sample] = full_wav_t[start_sample:end_sample]

                        # Save individual turn from the same aligned region.
                        turn_wav = aligned_agent[start_sample:end_sample].numpy()
                        out_path = os.path.join(args.out_dir, f"{stem}_turn_{turn_id}{ext}")
                        sf.write(out_path, turn_wav, samplerate=model.output_sample_rate)
                        logging.info(f"Saved: {out_path}")

                    # Save full artifact-scrubbed agent audio.
                    wav = aligned_agent.numpy()
                    out_path = os.path.join(args.out_dir, base_name)
                    sf.write(out_path, wav, samplerate=model.output_sample_rate)
                    logging.info(f"Full aligned agent audio saved: {out_path}")

                    # ---------------------------------------------------------
                    # Save aligned stereo conversation:
                    # channel 0 = user conditioning audio
                    # channel 1 = generated agent audio, zeroed outside turns
                    # ---------------------------------------------------------
                    if "user_audio_turns" in inputs:
                        user_segments = []

                        first_user_len_in = int(inputs["user_audio_turns_lens"][0][i].item())
                        first_user_delay_out = int(
                            round(first_user_len_in * model.output_sample_rate / model.sample_rate)
                        )

                        for turn_id, start_frame, end_frame in profile_turn_frame_ranges:
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

                            user_segments.append((user_start_sample, turn_audio_out))

                        total_user_len = 0
                        for s, wav_seg in user_segments:
                            total_user_len = max(total_user_len, s + wav_seg.numel())

                        user_ch = torch.zeros(total_user_len)

                        for s, wav_seg in user_segments:
                            e = s + wav_seg.numel()
                            user_ch[s:e] += wav_seg

                        # Agent channel keeps the same previous offset, but uses aligned_agent.
                        agent_ch = torch.cat(
                            [
                                torch.zeros(first_user_delay_out, dtype=aligned_agent.dtype),
                                aligned_agent,
                            ]
                        )

                        stereo_len = max(user_ch.numel(), agent_ch.numel())

                        user_pad = torch.zeros(stereo_len)
                        agent_pad = torch.zeros(stereo_len)

                        user_pad[: user_ch.numel()] = user_ch
                        agent_pad[: agent_ch.numel()] = agent_ch

                        stereo = torch.stack([user_pad, agent_pad], dim=1).numpy()

                        aligned_path = os.path.join(
                            args.out_dir,
                            f"{stem}_user_agent_aligned{ext}",
                        )

                        sf.write(
                            aligned_path,
                            stereo,
                            samplerate=model.output_sample_rate,
                        )

                        logging.info(f"Aligned user/agent stereo audio saved: {aligned_path}")

                else:
                    wav = audio_f32[i, : audio_len[i]].numpy()
                    out_path = os.path.join(args.out_dir, base_name)
                    sf.write(out_path, wav, samplerate=model.output_sample_rate)
                    logging.info(f"Saved: {out_path}")

    with fp32_precision():
        logging.info("\n--- Evaluation Metrics ---")
        cer_wer = intelligibility.compute()
        for k, m in cer_wer.items():
            logging.info(f"Intelligibility - {k}: {m}")

        secs_scores = secs_metric.compute()
        for k, m in secs_scores.items():
            logging.info(f"SECS - {k}: {m}")


if __name__ == "__main__":
    main()
