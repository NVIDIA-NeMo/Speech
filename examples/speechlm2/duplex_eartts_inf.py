# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Evaluation script for Duplex EARTTS models.

Args:
    config-path (str): Path to the directory containing the YAML configuration file.
    config-name (str): Name of the YAML configuration file.
    checkpoint_path (str): Path to the Duplex EARTTS checkpoint file.

Usage:
    python duplex_eartts_inf.py \
        --config-path=conf/ \
        --config-name=duplex_eartts.yaml \
        ++checkpoint_path=duplex_eartts_results/duplex_eartts/model.ckpt \
        ++datasets_json_path=/path/to/evalset_config.json \
        ++out_dir=duplex_eartts_results/duplex_eartts/audio_samples/dummy_dataset
"""

import os
import soundfile as sf
from nemo.collections.audio.parts.utils.resampling import resample
import torch
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexEARTTSDataset

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

import json


def read_jsonl_batches(
    file_path,
    batch_size,
    drop_last=False,
    max_batches=None,   # <-- DEBUG OPTION
):
    """
    Reads a JSONL file and yields batches of size batch_size.

    Args:
        file_path (str): Path to the JSONL file
        batch_size (int): Number of samples per batch
        drop_last (bool): If True, drop the last incomplete batch
        max_batches (int or None): If set, only yield this many batches (debug mode)

    Yields:
        List[dict]: A batch of samples
    """
    batch = []
    num_batches = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_idx}: {e}")

            batch.append(sample)

            if len(batch) == batch_size:
                yield batch
                batch = []
                num_batches += 1

                # --- DEBUG STOP ---
                if max_batches is not None and num_batches >= max_batches:
                    return

    if batch and not drop_last:
        yield batch

import os
import torch
import librosa
from torch.nn.utils.rnn import pad_sequence

def collate_and_tokenize_custom(
    batch,
    model,
    extra_duration_thrshould=1.3,
    sample_rate=22050,
    root_path=None,
    drop_BOS=False
):
    tokenized_list = []
    
    # --- TEXT TOKENIZATION ---
    for s in batch:
        text_data = s["text"]
        
        # Check if text is a list (New Logic)
        if isinstance(text_data, list):
            # Start with BOS
            full_ids = []

            for segment in text_data:
                # Tokenize segment
                if drop_BOS:
                    seg_ids = []
                else:
                    seg_ids = [model.tokenizer.bos]
                seg_ids = seg_ids + model.tokenizer.text_to_ids(segment)
                seg_len = len(seg_ids)

                # Calculate pad length (4x the size of the text)
                pad_len = seg_len * 10

                # Construct: text + 4x pads
                # We extend the list with the tokens and then the pad tokens
                full_ids.extend(seg_ids)
                full_ids.extend([model.text_pad_id] * pad_len)

            # Convert to tensor
            tokenized_list.append(
                torch.as_tensor(full_ids, dtype=torch.long, device=model.device)
            )

        else:
            # Standard String Handling
            if drop_BOS:
                tokenized_list.append(
                    torch.as_tensor(
                        model.tokenizer.text_to_ids(text_data),
                        dtype=torch.long,
                        device=model.device
                    )
                )
            else:
                tokenized_list.append(
                    torch.as_tensor(
                        [model.tokenizer.bos] + model.tokenizer.text_to_ids(text_data),
                        dtype=torch.long,
                        device=model.device
                    )
                )

    # Pad the text sequences (batch-wise)
    input_ids = pad_sequence(
        tokenized_list,
        batch_first=True,
        padding_value=model.text_pad_id
    )

    # load the target audio if available
    audio_list = []
    audio_lengths = []
    target_num_frames = []

    for i, s in enumerate(batch):
        # 1. Load Context Audio (Conditioning)
        audio_path = s["context_audio_filepath"]
        if root_path is not None:
            audio_path = os.path.join(root_path, audio_path)

        # Safety check for context audio presence, though usually required
        if os.path.exists(audio_path):
            wav, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            wav = torch.as_tensor(wav, dtype=torch.float32)
        else:
            # Fallback if context missing (optional safety)
            wav = torch.zeros(1, dtype=torch.float32)
        
        audio_list.append(wav)
        audio_lengths.append(len(wav))

        # 2. Handle Target Audio / Duration
        tdur_audio_path = s["audio_filepath"]
        if root_path is not None:
            tdur_audio_path = os.path.join(root_path, tdur_audio_path)

        # Check availability
        if tdur_audio_path and os.path.exists(tdur_audio_path):
            wav_dur, sr_ = librosa.load(tdur_audio_path, sr=sample_rate, mono=True)
            tdur = wav_dur.shape[0] // model.target_samples_per_frame 
            target_num_frames.append(tdur * extra_duration_thrshould)
        else:
            # Audio not available: Derive size from text channel
            # We follow the 4x approach logic here to determine frames.
            # If text was a list, it already has physical pads (1 + 4 ratio). 
            # We map 1 token roughly to 1 frame (or whatever the model scale is).
            # Assuming 1 token ~ 1 frame in the model's alignment, we just take the input length.
            
            current_text_len = len(tokenized_list[i])
            
            if isinstance(s["text"], list):
                # The text tokens are already physically padded 4x. 
                # Target frames should match this structure exactly.
                target_num_frames.append(current_text_len)
            else:
                # If text was a string (no physical pads added), but audio is missing,
                # we simulate the 4x duration expansion (1 part text, 4 parts silence = 5x total).
                target_num_frames.append(current_text_len * 5)

    # --- BATCH PADDING (AUDIO) ---
    max_audio_len = max(audio_lengths)
    B = len(audio_lengths)

    padded_audio = torch.zeros(
        (B, max_audio_len),
        dtype=torch.float32
    )

    for i, wav in enumerate(audio_list):
        padded_audio[i, :len(wav)] = wav

    padded_audio = padded_audio.to(model.device)
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    # --- RESIZE INPUT_IDS TO MATCH TARGET DURATION ---
    # Expand text length to match expected output speech duration
    B, L = input_ids.shape
    target_len = int(max(target_num_frames))
    
    # Ensure target_len is at least as long as the input text 
    # (prevents truncation if calc was slightly off)
    target_len = max(target_len, L)

    padded_input_ids = torch.full(
        (B, target_len),
        fill_value=model.text_pad_id,
        dtype=input_ids.dtype,
        device=input_ids.device
    )
    
    # Copy the actual tokens (which might already contain list-based padding)
    padded_input_ids[:, :L] = input_ids

    # If text is a list ["Hi", "There"], join it into "Hi There"
    collapsed_raw_text = [
        " ".join(s["text"]) if isinstance(s["text"], list) else s["text"]
        for s in batch
    ]

    return {
        "input_ids": padded_input_ids,
        "raw_text": collapsed_raw_text,
        "context_audio": padded_audio,
        "context_audio_lengths": audio_lengths,
        "target_audio_paths": [s["audio_filepath"] for s in batch],
        "target_num_frames": target_num_frames,
    }

@hydra_runner(config_path="conf", config_name="duplex_eartts")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.get("checkpoint_path", None):
        model = DuplexEARTTS.load_from_checkpoint(
            cfg.checkpoint_path,
            cfg=OmegaConf.to_container(cfg, resolve=True),
        ).eval()
    else:
        raise ValueError("For evaluation, you must provide `cfg.checkpoint_path`.")
    
    target_dtype = getattr(torch, cfg.get("inference_dtype", "float32"))
    # Move and cast
    if target_dtype != torch.float32:
        model.to(dtype=target_dtype)

    if cfg.get("reinit_audio_prompt_frozen_projection", False):
        D = model.tts_model.hidden_size
        Q, _ = torch.linalg.qr(torch.randn(D, D, device=model.tts_model.audio_prompt_projection_W.device, dtype=model.tts_model.audio_prompt_projection_W.dtype))
        model.tts_model.audio_prompt_projection_W.copy_(Q)

    intelligibility = Intelligibility("stt_en_fastconformer_transducer_large", reuse_asr_hyps=False).reset()

    for batch_id, batch in enumerate(read_jsonl_batches(cfg.datasets_json_path, cfg.batch_size, max_batches=None)):
        inputs = collate_and_tokenize_custom(batch, model, extra_duration_thrshould=1.5, sample_rate=model.target_sample_rate, root_path=cfg.audio_dir, drop_BOS=cfg.get("drop_BOS", False))
        if cfg.get("user_custom_speaker_reference", None):
            wav, sr = librosa.load(cfg.model.inference_speaker_reference, sr=model.target_sample_rate, mono=True)
            wav = torch.as_tensor(wav, dtype=target_dtype).unsqueeze(0)
            inputs["context_audio"] = wav.expand(inputs["input_ids"].size(0), *wav.shape[1:])
            inputs["context_audio_lengths"][:] = wav.size(-1)
            inputs["context_audio"] = inputs["context_audio"].to(model.device)
            inputs["context_audio_lengths"] = inputs["context_audio_lengths"].to(model.device).long()

        if target_dtype == torch.float32:
            with torch.no_grad():
                model.set_init_inputs(
                    speaker_audio=inputs["context_audio"],
                    speaker_audio_lens=inputs["context_audio_lengths"],
                    system_prompt=cfg.get("inference_system_prompt", None)
                )
                init_inputs = model.get_init_inputs(B=inputs["input_ids"].size(0))

                audio, audio_len = model.offline_inference(
                    next_subword_ids=inputs["input_ids"],
                    formatter="custom",
                    init_inputs=init_inputs,
                )
        else:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=target_dtype):
                    model.set_init_inputs(
                        speaker_audio=inputs["context_audio"],
                        speaker_audio_lens=inputs["context_audio_lengths"],
                        system_prompt=cfg.get("inference_system_prompt", None)
                    )
                    init_inputs = model.get_init_inputs(B=inputs["input_ids"].size(0))

                    audio, audio_len = model.offline_inference(
                        next_subword_ids=inputs["input_ids"],
                        formatter="custom",
                        init_inputs=init_inputs,
                    )

    
        audio = audio.float()
        # wav_dur = int(inputs["target_num_frames"][i] * model.target_samples_per_frame)
        # reset audio len to the actual size removing extra long audio padding
        audio_len = (torch.tensor(inputs["target_num_frames"]) * model.target_samples_per_frame).int()
 
        # resample audio to the asr sampling rate
        metric_audio_pred = resample(audio, model.target_sample_rate, 16000)
        metric_audio_pred_lens = (audio_len / model.target_sample_rate * 16000).to(torch.long)

        intelligibility.update(
            name="dataset",
            refs=inputs["raw_text"],
            pred_audio=metric_audio_pred,
            pred_audio_lens=metric_audio_pred_lens,
            asr_hyps=None,
        )

        # save audio to cfg.out_dir
        os.makedirs(cfg.out_dir, exist_ok=True)

        audio = audio.detach().cpu().float()
        audio_len = audio_len.cpu()

        for i in range(audio.size(0)):
            wav = audio[i, : audio_len[i]].numpy()
            #wav = audio[i, : wav_dur].numpy() # use precomputed estimated duration to avoid longer audios
            # Use original target audio filename
            target_path = inputs["target_audio_paths"][i]
            base_name = os.path.basename(target_path)
            out_path = os.path.join(cfg.out_dir, base_name)

            sf.write(
                out_path,
                wav,
                samplerate=model.target_sample_rate,
            )

            print(f"Saved: {out_path}")

    cer_wer = intelligibility.compute()
    for k, m in cer_wer.items():
        print(k, m)

if __name__ == "__main__":
    inference()
