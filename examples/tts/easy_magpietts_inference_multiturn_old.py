# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Evaluation script for custom EasyMagpieTTS models trained on 10x padded inputs.
Stripped of Hydra config requirements. Uses standard argparse.
Features an explicitly exposed, fully batched autoregressive loop for easy multi-turn modding.

Usage:
    python easy_magpietts_eval.py \
        --checkpoint_path=/path/to/magpie/model.ckpt \
        --codec_model_path=/path/to/codec/model.ckpt \
        --datasets_json_path=/path/to/evalset_config.jsonl \
        --out_dir=/path/to/out/audio \
        --batch_size=6 \
        --add_interruption_token \
        --add_beginning_pad_tokens \
        --use_cfg
"""

import argparse
import json
import os
import time
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

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


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


def collate_and_tokenize_custom(
    batch,
    model,
    extra_duration_thrshould=1.3,
    sample_rate=22050,
    root_path=None,
    add_beginning_pad_tokens=False,
    add_interruption_token=False,
    pad_factor_text_speech=10,
    force_interruption=False,
):
    tokenized_list = []
    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
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

    if add_beginning_pad_tokens:
        pad_len = 25
        prefix = torch.full((pad_len,), model.pad_id, dtype=torch.long)
        for i in range(len(tokenized_list)):
            tokenized_list[i] = torch.cat([prefix, tokenized_list[i]])

    # Capture the true sequence length before pad_sequence applies batch alignment padding
    input_lengths = torch.tensor([len(x) for x in tokenized_list], dtype=torch.long)
    input_ids = pad_sequence(tokenized_list, batch_first=True, padding_value=model.pad_id)

    audio_list = []
    audio_lengths = []
    target_num_frames = []

    for i, s in enumerate(batch):
        audio_path = s["context_audio_filepath"]
        if root_path is not None:
            audio_path = os.path.join(root_path, audio_path)

        if os.path.exists(audio_path):
            wav, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            wav = torch.as_tensor(wav, dtype=torch.float32)
        else:
            wav = torch.zeros(1, dtype=torch.float32)

        audio_list.append(wav)
        audio_lengths.append(len(wav))

        tdur_audio_path = s["audio_filepath"]
        if root_path is not None:
            tdur_audio_path = os.path.join(root_path, tdur_audio_path)

        if tdur_audio_path and os.path.exists(tdur_audio_path):
            wav_dur, sr_ = librosa.load(tdur_audio_path, sr=sample_rate, mono=True)
            tdur = wav_dur.shape[0] // model.target_samples_per_frame
            target_num_frames.append(tdur * extra_duration_thrshould)
        else:
            current_text_len = len(tokenized_list[i])
            if isinstance(s["text"], list):
                target_num_frames.append(current_text_len)
            else:
                target_num_frames.append(current_text_len * 5)

    max_audio_len = max(audio_lengths)
    B = len(audio_lengths)
    padded_audio = torch.zeros((B, max_audio_len), dtype=torch.float32)

    for i, wav in enumerate(audio_list):
        padded_audio[i, : len(wav)] = wav

    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
    B, L = input_ids.shape
    target_len = int(max(target_num_frames))
    target_len = max(target_len, L)

    padded_input_ids = torch.full((B, target_len), fill_value=model.pad_id, dtype=input_ids.dtype)
    padded_input_ids[:, :L] = input_ids

    collapsed_raw_text = [" ".join(s["text"]) if isinstance(s["text"], list) else s["text"] for s in batch]

    return {
        "input_ids": padded_input_ids,
        "input_lengths": input_lengths,
        "raw_text": collapsed_raw_text,
        "context_audio": padded_audio,
        "context_audio_lengths": audio_lengths,
        "target_audio_paths": [s["audio_filepath"] for s in batch],
        "target_num_frames": target_num_frames,
    }


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
    parser.add_argument("--inference_dtype", type=str, default="float16")
    parser.add_argument("--debug_dtype", action="store_true")
    
    # Dataloader & Batching
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_turns", type=int, default=1)
    parser.add_argument("--pad_factor_text_speech", type=int, default=10)
    
    # Text Processing Boolean Flags
    parser.add_argument("--add_beginning_pad_tokens", action="store_true")
    parser.add_argument("--add_interruption_token", action="store_true")
    parser.add_argument("--force_interruption", action="store_true")
    
    # Speaker & Prompt Configurations
    parser.add_argument("--user_custom_speaker_reference", action="store_true")
    parser.add_argument("--inference_speaker_reference", type=str, default=None)
    parser.add_argument("--language", type=str, default="en")
    
    # Generation Kwargs
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--max_tts_steps", type=int, default=1000)

    args = parser.parse_args()

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
        if args.phoneme_tokenizer_path and getattr(model_cfg, "phoneme_tokenizer", None) is not None:
            model_cfg.phoneme_tokenizer.tokenizer_path = args.phoneme_tokenizer_path

    model = EasyMagpieTTSInferenceModel.restore_from(
        args.checkpoint_path, override_config_path=model_cfg, map_location=target_device
    )
    model.use_kv_cache_for_inference = True
    model.eval().to(target_device)
    model.to(dtype=target_dtype)

    # --- DATALOADER COMPATIBILITY PATCHES ---
    model.target_samples_per_frame = getattr(model, "codec_model_samples_per_frame", 320)
    model.target_sample_rate = getattr(model, "sample_rate", 22050)
    model.pad_id = getattr(model.tokenizer, "pad_id", 0)
    model.text_eos_id = model.eos_id

    codec_model = AudioCodecModel.restore_from(args.codec_model_path, strict=False, map_location=target_device)
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

    if args.debug_dtype:
        handles, stats, examples = attach_dtype_counter(model)

    with fp32_precision():
        intelligibility = Intelligibility("stt_en_fastconformer_transducer_large", reuse_asr_hyps=False).reset()
        secs_metric = SECS("titanet_large").reset()

    eval_dataset = EvalJSONLDataset(args.datasets_json_path, num_turns=args.num_turns)
    
    collate_fn = partial(
        collate_and_tokenize_custom,
        model=model,
        extra_duration_thrshould=1.5,
        sample_rate=model.target_sample_rate,
        root_path=args.audio_dir,
        add_beginning_pad_tokens=args.add_beginning_pad_tokens,
        add_interruption_token=args.add_interruption_token,
        pad_factor_text_speech=args.pad_factor_text_speech,
        force_interruption=args.force_interruption,
    )

    dataloader = DataLoader(
        dataset=eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False,
    )

    if args.user_custom_speaker_reference and args.inference_speaker_reference:
        wav, sr = librosa.load(args.inference_speaker_reference, sr=model.target_sample_rate, mono=True)
        speaker_wav = torch.as_tensor(wav, dtype=target_dtype).unsqueeze(0).to(model.device)

    for batch_id, inputs in enumerate(dataloader):
        B = inputs["input_ids"].size(0)
        device = model.device
        
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["input_lengths"] = inputs["input_lengths"].to(device)
        inputs["context_audio"] = inputs["context_audio"].to(device, dtype=target_dtype)
        inputs["context_audio_lengths"] = inputs["context_audio_lengths"].to(device)
        
        if args.user_custom_speaker_reference and args.inference_speaker_reference:
            inputs["context_audio"] = speaker_wav.expand(B, *speaker_wav.shape[1:])
            inputs["context_audio_lengths"][:] = speaker_wav.size(-1)

        # 1. Prepare Context & Initialize
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
            # EXPOSED BATCHED GENERATION LOOP (Ready for multi-turn edits!)
            # ---------------------------------------------------------
            text = inputs["input_ids"]
            text_lens = inputs["input_lengths"]

            gen_step = 0
            while not state.finished.all() and len(state.all_predictions) < args.max_tts_steps:
                gen_step += 1
                
                # Fetch current token dynamically based on state.text_tokens_seen
                positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                current_tokens = text[torch.arange(B, device=device), positions]

                # Mask out sequences that have finished their true length
                text_exhausted = state.text_tokens_seen >= text_lens
                current_tokens = torch.where(
                    text_exhausted, torch.full_like(current_tokens, model.eos_id), current_tokens
                )

                # Feed tokens to the model step-by-step
                state, audio_codes, _ = model.streaming_step(
                    state=state,
                    text_tokens=current_tokens,
                    use_inference_mode=True,
                )

            # Bulk Decode using the exposed state
            finalize_output = model.streaming_finalize(state, use_inference_mode=True)
            # ---------------------------------------------------------

        if args.debug_dtype and batch_id == 0:
            report_dtype_stats(handles, stats, examples)

        with fp32_precision():
            # Grab output directly from streaming_finalize
            audio_f32 = finalize_output.audio.float()
            audio_len = finalize_output.audio_len.int()

            expected_audio_lens = (torch.tensor(inputs["target_num_frames"], device=device) * model.target_samples_per_frame).int()
            audio_len = torch.min(audio_len, expected_audio_lens) 

            metric_audio_pred = resample(audio_f32, getattr(model, "output_sample_rate", 24000), 16000)
            metric_audio_pred_lens = (audio_len / getattr(model, "output_sample_rate", 24000) * 16000).to(torch.long)

            intelligibility.update(
                name="dataset",
                refs=inputs["raw_text"],
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
                asr_hyps=None,
            )

            secs_metric.update(
                name="dataset",
                target_audio=resample(inputs["context_audio"].float(), model.target_sample_rate, 16000),
                target_audio_lens=(inputs["context_audio_lengths"] / model.target_sample_rate * 16000).to(torch.long),
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
            )

            os.makedirs(args.out_dir, exist_ok=True)
            audio_f32 = audio_f32.detach().cpu()
            audio_len = audio_len.cpu()

            for i in range(B):
                wav = audio_f32[i, : audio_len[i]].numpy()
                target_path = inputs["target_audio_paths"][i]
                base_name = os.path.basename(target_path)
                out_path = os.path.join(args.out_dir, base_name)
                sf.write(out_path, wav, samplerate=getattr(model, "output_sample_rate", 24000))
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
