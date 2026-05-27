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
Evaluation script for Duplex EARTTS models following MagpieTTS evaluation recipe.

Args:
    config-path (str):
        Path to the directory containing the YAML configuration file.

    config-name (str):
        Name of the YAML configuration file.

    checkpoint_path (str):
        Path to the Duplex EARTTS checkpoint file.

    datasets_json_path (str):
        Path to a JSONL (JSON Lines) file describing the evaluation dataset.
        Each line must be a valid JSON object representing one sample.

        Supported formats:

        ----------------------------------------------------------------------
        1) SINGLE-TURN FORMAT
        ----------------------------------------------------------------------
        "text" is a string.

        Example:
        {"text": "Like really quickly and they go haha and then they run off.",
         "context_audio_filepath": "speaker_1.wav",
         "audio_filepath": "audio_1.wav"}

        {"text": "Sure. Okay.",
         "context_audio_filepath": "speaker_2.wav",
         "audio_filepath": "audio_2.wav"}

        ----------------------------------------------------------------------
        2) MULTI-TURN FORMAT
        ----------------------------------------------------------------------
        "text" is a list of utterances (List[str]).
        Each element represents one conversational turn. The model will
        tokenize and pad each segment sequentially.

        Example:
        {"text": ["Okay yeah.", "Yeah.", "Right.", "I get what you’re saying.", "That makes sense."],
         "context_audio_filepath": "speaker_1.wav",
         "audio_filepath": "dummy_blank_audio_mt_0001.wav"}

        {"text": ["Okay.", "Really?", "Yeah, okay.", "I didn’t know that.", "That’s interesting."],
         "context_audio_filepath": "speaker_2.wav",
         "audio_filepath": "audio_2.wav"}

        ----------------------------------------------------------------------
        FIELD DESCRIPTIONS
        ----------------------------------------------------------------------

        text:
            Either:
                - str (single-turn)
                - List[str] (multi-turn)

        context_audio_filepath:
            Path to the reference speaker audio used for conditioning.
            This can be overridden by setting:
                ++user_custom_speaker_reference=<path>

        audio_filepath:
            Output audio file name.
            This is used only as the base filename for saving generated audio
            inside `out_dir`. The file does NOT need to exist beforehand.

    out_dir (str):
        Directory where generated audio samples will be saved.

    inference_dtype (str, optional):
        Target dtype used during inference. This controls the precision
        of model weights and operations.

        Supported values:
            - "float32" (default)
            - "float16"
            - "bfloat16"

        Notes:
            - If set to a lower precision (e.g., float16), the model weights
              and/or execution dtype will be adjusted accordingly.
            - Internally mapped via `getattr(torch, inference_dtype)`.

    keep_codec_original_dtype (bool, optional):
        Controls whether the audio codec module keeps its original dtype
        when `inference_dtype` is not float32.

        If True (default):
            - Only the TTS backbone (`model.tts_model`) is cast to the target dtype.
            - The codec remains in its original precision (typically float32).
            - Useful to isolate precision effects and avoid degradation from
              codec quantization.

        If False:
            - The entire model (including codec) is cast to `inference_dtype`.
            - `model.audio_codec_run_dtype` is also set accordingly.

    debug_dtype (bool, optional):
        Enables runtime inspection of tensor dtypes flowing through the model.

        If True:
            - Forward hooks are attached to all leaf modules.
            - During the first batch, dtype usage statistics are collected
              and logged.
            - Outputs include:
                - Per-module-group dtype distribution
                - Example module names per dtype

Usage:
    # Example with fp32 inference
        python duplex_eartts_eval.py \
            --config-path=conf/ \
            --config-name=duplex_eartts.yaml \
            ++checkpoint_path=duplex_eartts_results/duplex_eartts/model.ckpt \
            ++datasets_json_path=/path/to/evalset_config.jsonl \
            ++out_dir=duplex_eartts_results/duplex_eartts/audio_samples/dummy_dataset

    # Example with fp16 inference and dtype debugging
        python duplex_eartts_eval.py \
            --config-path=conf/ \
            --config-name=duplex_eartts.yaml \
            ++checkpoint_path=duplex_eartts_results/duplex_eartts/model.ckpt \
            ++datasets_json_path=/path/to/evalset_config.jsonl \
            ++out_dir=uplex_eartts_results/duplex_eartts/audio_samples/dummy_dataset \
            ++inference_dtype=float16 \
            ++keep_codec_original_dtype=True \
            ++debug_dtype=True
"""

import json
import os
from functools import partial

import librosa
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from nemo.collections.audio.parts.utils.transforms import resample

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from contextlib import nullcontext

from omegaconf import OmegaConf

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.core.config import hydra_runner
from nemo.utils import logging

# Use .get() to avoid crashing when running a single GPU without torchrun
if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def attach_dtype_counter(model):
    """
    Attaches forward hooks to all leaf modules of a model to track the dtype
    of their outputs during inference.

    This utility is designed for debugging precision behavior, especially when
    using mixed precision or reduced precision (fp16 / bf16).

    Behavior:
        - Registers a forward hook on each leaf module (modules with no children).
        - For each forward pass, records the dtype of the module output.
        - Aggregates statistics grouped by top-level module name.
        - Stores a few example module class names per dtype.

    Returns:
        handles (List[RemovableHandle]):
            List of hook handles. These must be removed manually to avoid
            memory leaks or performance degradation.

        stats (Dict[str, Dict[str, int]]):
            Nested dictionary containing dtype counts per module group.
            Structure:
                stats[module_group][dtype] = count

            Example:
                {
                    "tts_model": {
                        "torch.float16": 120,
                        "torch.float32": 0,
                        "torch.bfloat16": 0,
                        "other": 2
                    }
                }

        examples (Dict[str, Dict[str, List[str]]]):
            Stores up to 3 example module class names per dtype per group.
            Useful for quickly identifying which layers are running in
            unexpected precision.

    Notes:
        - Only inspects outputs (not inputs or parameters).
        - Dtype is inferred from the first tensor found in the output.
        - Non-floating dtypes are categorized as "other".
        - Grouping is based on the top-level module name (prefix before first dot).

    Typical usage:
        handles, stats, examples = attach_dtype_counter(model)

        # Run inference ...

        for h in handles:
            h.remove()
    """
    handles = []

    # structure: stats[module_group][dtype] = count
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
        # top-level module (before first dot)
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

            # store a few examples per dtype per group
            if len(examples[group][dtype]) < 3:
                examples[group][dtype].append(module.__class__.__name__)

        return fn

    for name, module in model.named_modules():
        if is_leaf(module):
            handles.append(module.register_forward_hook(hook_fn(name)))

    return handles, stats, examples


def report_dtype_stats(handles, stats, examples):
    """
    Cleans up monitoring hooks and logs a detailed report of the tensor precisions
    (dtypes) observed during the model forward pass.

    This function should be called after at least one inference iteration has
    completed while hooks are attached. It removes the hooks to prevent
    performance overhead and prints a structured summary of which module groups
    executed in which dtypes.

    Args:
        handles (List[torch.utils.hooks.RemovableHandle]): The list of hooks
            returned by `attach_dtype_counter`.
        stats (Dict): Nested dictionary containing dtype counts per module group.
        examples (Dict): Dictionary containing example module names for each
            observed dtype.
    """
    for h in handles:
        h.remove()

    logging.info("\n=== DTYPE USAGE PER MODULE ===")

    for group, group_stats in stats.items():
        total = sum(group_stats.values())
        if total == 0:
            continue

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
    """
    Standard PyTorch Dataset for reading JSONL evaluation files.
    """

    def __init__(self, file_path):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_idx}: {e}")

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
    add_eos=False,
    pad_factor_text_speech=10,
    force_interruption=False,
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
                seg_ids = [model.tokenizer.bos]
                seg_ids = seg_ids + model.tokenizer.text_to_ids(segment)
                seg_len = len(seg_ids)

                # Calculate pad length (pad_factor_text_speechx the size of the text)
                pad_len = seg_len * pad_factor_text_speech

                # Construct: text + 4x pads
                # We extend the list with the tokens and then the pad tokens
                pad_ids = [model.text_pad_id] * pad_len

                if force_interruption:
                    fname = s["audio_filepath"]
                    no_ext = fname.split(".")[0]
                    sample_id = int(no_ext.split("_")[-1])

                    case = sample_id % 3  # 0,1,2 -> ~33% each

                    if case == 0:
                        # 33%: emulate interruption where text was not fully processed
                        # (no pad eos placement at all)
                        if len(seg_ids) >= 2:
                            seg_ids[-2] = model.text_eos_id
                            seg_ids[-1] = model.text_pad_id
                        else:
                            # fallback: if seg_ids is too short, emulate with pad EOS at 0
                            pad_ids[0] = model.text_eos_id
                    elif case == 1:
                        # 33%: put EOS at pad index 6 - so 0.5 seconds after the whole text was processed
                        eos_idx = min(6, len(pad_ids) - 1)
                        pad_ids[eos_idx] = model.text_eos_id
                    else:
                        # 33%: put EOS at pad index 0
                        eos_idx = 0
                        pad_ids[eos_idx] = model.text_eos_id
                else:
                    if (
                        add_eos
                    ):  # add eos in the end of the paddding sequence keep 70% for the speech and the rest for after EOS
                        eos_idx = int(len(pad_ids) * 0.7)
                        pad_ids[eos_idx] = model.text_eos_id

                full_ids.extend(seg_ids)
                full_ids.extend(pad_ids)

            tokenized_list.append(torch.as_tensor(full_ids, dtype=torch.long))

        else:
            # Standard String Handling
            tokenized_list.append(
                torch.as_tensor([model.tokenizer.bos] + model.tokenizer.text_to_ids(text_data), dtype=torch.long)
            )

    if add_beginning_pad_tokens:
        pad_len = 25
        prefix = torch.full((pad_len,), model.text_pad_id, dtype=torch.long)
        for i in range(len(tokenized_list)):
            tokenized_list[i] = torch.cat([prefix, tokenized_list[i]])

    # Pad the text sequences (batch-wise)
    input_ids = pad_sequence(tokenized_list, batch_first=True, padding_value=model.text_pad_id)

    # load the target audio if available
    audio_list = []
    audio_lengths = []
    target_num_frames = []

    for i, s in enumerate(batch):
        # Load Context Audio
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

        # Handle Target Audio / Duration
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
                # The text tokens are already physically padded 10x.
                # Target frames should match this structure exactly.
                target_num_frames.append(current_text_len)
            else:
                # If text was a string (no physical pads added), but audio is missing,
                # we simulate the 4x duration expansion (1 part text, 4 parts silence = 5x total).
                target_num_frames.append(current_text_len * 5)

    # audio padding
    max_audio_len = max(audio_lengths)
    B = len(audio_lengths)

    padded_audio = torch.zeros((B, max_audio_len), dtype=torch.float32)

    for i, wav in enumerate(audio_list):
        padded_audio[i, : len(wav)] = wav

    # Keep on CPU
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    # Expand text length to match expected output speech duration
    B, L = input_ids.shape
    target_len = int(max(target_num_frames))

    # Ensure target_len is at least as long as the input text
    # (prevents truncation if calc was slightly off)
    target_len = max(target_len, L)

    padded_input_ids = torch.full((B, target_len), fill_value=model.text_pad_id, dtype=input_ids.dtype)

    # Copy the actual tokens (which might already contain list-based padding)
    padded_input_ids[:, :L] = input_ids

    # If text is a list ["Hi", "There"], join it into "Hi There"
    collapsed_raw_text = [" ".join(s["text"]) if isinstance(s["text"], list) else s["text"] for s in batch]

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

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # Dynamically determine the correct GPU for this process
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        target_device = torch.device(f"cuda:{local_rank}")
    else:
        target_device = torch.device("cpu")

    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    target_dtype = getattr(torch, cfg.get("inference_dtype", "float32"))
    if target_dtype != torch.float32:
        torch.set_default_dtype(target_dtype)

    if cfg.get("checkpoint_path", None):
        model = DuplexEARTTS.load_from_checkpoint(
            cfg.checkpoint_path, cfg=OmegaConf.to_container(cfg, resolve=True), map_location=target_device
        ).eval()
    else:
        raise ValueError("For evaluation, you must provide `cfg.checkpoint_path`.")

    if target_dtype != torch.float32:
        if cfg.get("keep_codec_original_dtype", True):
            model.tts_model.to(dtype=target_dtype)
            model.ensures_codec_target_dtype()  # ensures that codec is in the right precision
        else:
            model.audio_codec_run_dtype = target_dtype
            model.to(dtype=target_dtype)

    if cfg.get("debug_dtype", False):
        handles, stats, examples = attach_dtype_counter(model)

    with fp32_precision():
        intelligibility = Intelligibility("stt_en_fastconformer_transducer_large", reuse_asr_hyps=False).reset()
        secs_metric = SECS("titanet_large").reset()

    # Initialize the Dataset
    eval_dataset = EvalJSONLDataset(cfg.datasets_json_path)

    # Use partial to bind the model and config parameters to the collate function
    collate_fn = partial(
        collate_and_tokenize_custom,
        model=model,
        extra_duration_thrshould=1.5,
        sample_rate=model.target_sample_rate,
        root_path=cfg.audio_dir,
        add_beginning_pad_tokens=cfg.get("add_beginning_pad_tokens", True),
        add_eos=cfg.get("add_eos", True),
        pad_factor_text_speech=cfg.get("pad_factor_text_speech", 10),
        force_interruption=cfg.get("force_interruption", False),
    )

    # Initialize the DataLoader
    dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    if cfg.get("user_custom_speaker_reference", None):
        wav, sr = librosa.load(cfg.model.inference_speaker_reference, sr=model.target_sample_rate, mono=True)
        speaker_wav = torch.as_tensor(wav, dtype=target_dtype).unsqueeze(0).to(model.device)

    # Iterate over the DataLoader
    for batch_id, inputs in enumerate(dataloader):

        # Move required tensors to the GPU immediately
        inputs["input_ids"] = inputs["input_ids"].to(model.device)
        inputs["context_audio"] = inputs["context_audio"].to(model.device)
        inputs["context_audio_lengths"] = inputs["context_audio_lengths"].to(model.device)

        if cfg.get("user_custom_speaker_reference", None):
            inputs["context_audio"] = speaker_wav.expand(inputs["input_ids"].size(0), *speaker_wav.shape[1:])
            inputs["context_audio_lengths"][:] = speaker_wav.size(-1)

        with torch.no_grad():
            model.set_init_inputs(
                speaker_audio=inputs["context_audio"],
                speaker_audio_lens=inputs["context_audio_lengths"],
                system_prompt=cfg.get("inference_system_prompt", None),
            )
            init_inputs = model.get_init_inputs(B=inputs["input_ids"].size(0))

            audio, audio_len = model.offline_inference(
                next_subword_ids=inputs["input_ids"],
                task="custom",
                init_inputs=init_inputs,
            )

        if cfg.get("debug_dtype", False) and batch_id == 0:
            report_dtype_stats(handles, stats, examples)

        with fp32_precision():
            audio = audio.float()

            # reset audio len to the actual size removing extra long audio padding
            audio_len = (
                torch.tensor(inputs["target_num_frames"], device=audio.device) * model.target_samples_per_frame
            ).int()

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

            secs_metric.update(
                name="dataset",
                target_audio=resample(inputs["context_audio"], model.target_sample_rate, 16000),
                target_audio_lens=(inputs["context_audio_lengths"] / model.target_sample_rate * 16000).to(torch.long),
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
            )

            # save audio to cfg.out_dir
            os.makedirs(cfg.out_dir, exist_ok=True)
            audio = audio.detach().cpu().float()
            audio_len = audio_len.cpu()

            for i in range(audio.size(0)):
                wav = audio[i, : audio_len[i]].numpy()
                # Use original target audio filename
                target_path = inputs["target_audio_paths"][i]
                base_name = os.path.basename(target_path)
                out_path = os.path.join(cfg.out_dir, base_name)

                sf.write(
                    out_path,
                    wav,
                    samplerate=model.target_sample_rate,
                )

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
    inference()
