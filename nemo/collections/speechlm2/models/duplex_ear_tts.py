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
import os
import random
import tempfile
import numpy as np
import time

import glob
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache
import math

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from transformers import AutoModelForCausalLM

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.core.classes.module import NeuralModule
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.intelligibility import Intelligibility
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.mimi_codec_modules import ReshapeTransformerEncoder
from nemo.collections.speechlm2.modules.ear_tts_commons import SCRIPT_PLACEHOLDER

from nemo.collections.speechlm2.modules.cfm import MatchaTTSCFM
from types import SimpleNamespace


from nemo.collections.speechlm2.modules.rvq_ear_tts_model import RVQEARTTSModel, RVQEARTTSConfig, build_vocabs, SubwordFlagEmbedding, RMSNorm
from nemo.collections.speechlm2.modules.rvq_ear_tts_vae import RVQVAEModel
from nemo.collections.speechlm2.data.duplex_ear_tts_dataset import normalize_text_fn

import torch
import torch.nn as nn
import copy

def maybe_to(x, dtype):
    if x is None:
        return None
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype)
    return x

from collections import Counter
from contextlib import contextmanager
import torch

@contextmanager
def ensures_16_precision(mixed_dtype):
    """
    Workaround for precision related issues when training with bf16-true PyTorch Lightning precision setting.
    In bf16-true, PTL changes PyTorch's default dtype, which may break implicit assumptions for some models.
    This context manager restores default float32 precision and runs the computation in float32 autocast context.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(mixed_dtype)
    try:
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=mixed_dtype):
            yield
    finally:
        torch.set_default_dtype(default_dtype)


def make_tts_model_mixed_precision_definite(model, inputs,
                                            mixed_dtype=torch.bfloat16,
                                            bf16_min=1e-2, bf16_max=1e2,
                                            safety_factor=1.0):
    safe_min = bf16_min * safety_factor
    safe_max = bf16_max * safety_factor

    # 1️⃣ Collect activation stats in FP32
    model_fp32 = copy.deepcopy(model).eval().to(torch.float32)
    stats = {}
    hooks = []

    def _activation_hook(name):
        def hook(_, __, out):
            if isinstance(out, tuple):
                out = out[0]
            if torch.is_tensor(out):
                stats[name] = {"min": float(out.detach().min()), "max": float(out.detach().max())}
        return hook

    for name, module in model_fp32.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            hooks.append(module.register_forward_hook(_activation_hook(name)))

    with torch.no_grad():
        _ = model_fp32(
            code=inputs["code"],
            audio_mask=maybe_to(inputs["audio_mask"], torch.float32),
            attention_mask=maybe_to(inputs["attention_mask"], torch.float32),
            position_ids=inputs["position_ids"],
            context_hidden_state=maybe_to(inputs["context_hidden_state"], torch.float32),
            subword_ids=inputs["subword_ids"],
            subword_mask=maybe_to(inputs["subword_mask"], torch.float32),
            non_prompt_mask=maybe_to(inputs["non_prompt_mask"], torch.float32)
        )

    for h in hooks:
        h.remove()

    # 2️⃣ Patch model for mixed precision with safe propagation
    model_patched = copy.deepcopy(model).eval()
    bf16_layers, fp32_layers = [], []

    all_modules = list(model_patched.named_modules())
    num_modules = len(all_modules)

    # flag to propagate FP32 to next safe layers
    propagate_fp32 = False

    for idx, (name, module) in enumerate(all_modules):
        if name not in stats:
            continue
        mn, mx = stats[name]["min"], stats[name]["max"]
        safe = (abs(mn) < safe_max and abs(mx) < safe_max
                and not (abs(mn) < safe_min and abs(mx) < safe_min))

        is_sensitive = False
        if isinstance(module, (nn.LayerNorm, nn.Embedding)):
            is_sensitive = True
        elif isinstance(module, nn.Linear):
            if not safe:
                is_sensitive = True

        # mark this layer
        if is_sensitive:
            if name not in fp32_layers:
                fp32_layers.append(name)
            propagate_fp32 = True  # propagate FP32 to next layers if safe
        else:
            if propagate_fp32:
                # next layer is safe but preceded by FP32-sensitive -> still FP32
                fp32_layers.append(name)
                propagate_fp32 = False  # stop propagation after one safe layer
            else:
                # layer itself is safe and no FP32 propagation -> use BF16/FP16
                if isinstance(module, nn.Linear):
                    bf16_layers.append(name)

    # 3️⃣ Wrap forwards to enforce precision
    def wrap_forward(module, is_fp32_sensitive):
        if hasattr(module, "_original_forward"):
            return
        module._original_forward = module.forward

        def new_forward(*args, **kwargs):
            if is_fp32_sensitive:
                with fp32_precision():
                    return module._original_forward(*args, **kwargs)
            else:
                new_args = tuple(a.to(mixed_dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)
                new_kwargs = {k: v.to(mixed_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                              for k, v in kwargs.items()}
                # with torch.cuda.amp.autocast(enabled=True, dtype=mixed_dtype):
                with ensures_16_precision(mixed_dtype):
                    return module._original_forward(*new_args, **new_kwargs)

        module.forward = new_forward

    for name, module in model_patched.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            wrap_forward(module, name in fp32_layers)

    # 4️⃣ Count actual running dtype
    running_dtypes = Counter()
    hook_handles = []

    def dtype_counter_hook(module, inputs, outputs):
        for x in inputs:
            if isinstance(x, torch.Tensor):
                running_dtypes[str(x.dtype)] += 1
        outputs_list = outputs if isinstance(outputs, (tuple, list)) else [outputs]
        for x in outputs_list:
            if isinstance(x, torch.Tensor):
                running_dtypes[str(x.dtype)] += 1

    for name, module in model_patched.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            hook_handles.append(module.register_forward_hook(dtype_counter_hook))

    with torch.no_grad():
        _ = model_patched(
            code=inputs["code"],
            audio_mask=maybe_to(inputs["audio_mask"], torch.float32),
            attention_mask=maybe_to(inputs["attention_mask"], torch.float32),
            position_ids=inputs["position_ids"],
            context_hidden_state=maybe_to(inputs["context_hidden_state"], torch.float32),
            subword_ids=inputs["subword_ids"],
            subword_mask=maybe_to(inputs["subword_mask"], torch.float32),
            non_prompt_mask=maybe_to(inputs["non_prompt_mask"], torch.float32)
        )

    for h in hook_handles:
        h.remove()

    num_bf16_fp16 = running_dtypes.get("torch.bfloat16", 0) + running_dtypes.get("torch.float16", 0)
    num_fp32 = running_dtypes.get("torch.float32", 0)

    summary = {
        "bf16_layers": bf16_layers,
        "fp32_layers": fp32_layers,
        "num_bf16_fp16": num_bf16_fp16,
        "num_fp32": num_fp32,
        "stats": stats,
        "safe_min": safe_min,
        "safe_max": safe_max,
        "safety_factor": safety_factor,
    }

    # print("Num. BF16/FP16 activations:", num_bf16_fp16)
    # print("Num. FP32 activations:", num_fp32)
    print("Num. BF16/FP16 candidate layers:", len(bf16_layers))
    print("Num. FP32 layers (sensitive + propagated):", len(fp32_layers))

    return model_patched, summary
    


def generate_multiturn_speaking_mask(input_ids: torch.Tensor, bos_token_id: int = 0, eos_token_id: int = 1):
    """
    Efficient, batched speaking mask generator that marks 1 between <bos> and <eos> pairs.
    If <eos> is missing after a <bos>, mask continues to end. Handles multiple turns.

    Args:
        input_ids (torch.Tensor): LongTensor of shape (B, T)
        bos_token_id (int): Token ID for <bos>
        eos_token_id (int): Token ID for <eos>

    Returns:
        torch.Tensor: FloatTensor of shape (B, T), with 1.0 for speaking, 0.0 for silence.

    Note BOS is considered as speaking (1) and EOS as non speaking 0
    """
    B, T = input_ids.shape
    device = input_ids.device
    bos_mask = (input_ids == bos_token_id).to(torch.int32).to(device)
    eos_mask = (input_ids == eos_token_id).to(torch.int32).to(device)
    bos_cumsum = torch.cumsum(bos_mask, dim=1)
    eos_cumsum = torch.cumsum(eos_mask, dim=1)
    speaking_mask = (bos_cumsum > eos_cumsum).to(torch.float32)
    return speaking_mask.long()


def replace_control_speech_codes(speech_codes: torch.Tensor, control_codes: torch.Tensor, silence_tokens: torch.Tensor = None) -> torch.Tensor:
    """
    Replaces control codes (speech BOS, EOS, etc) in `speech_codes` with the first frame which is
    assumed to consist of 'valid' codes representing silence.
    """
    if silence_tokens is not None:
        # Expand to [B, 1, 74]
        silence_tokens_expanded = silence_tokens.unsqueeze(0).unsqueeze(1).expand(speech_codes.shape[0], 1, -1)
        return torch.where(torch.isin(speech_codes, control_codes), silence_tokens_expanded, speech_codes)

    if torch.isin(speech_codes[:, :1], control_codes).any():
        return torch.where(torch.isin(speech_codes, control_codes), torch.zeros_like(speech_codes[:, :1]), speech_codes)
    else:
        return torch.where(torch.isin(speech_codes, control_codes), speech_codes[:, :1], speech_codes)


def get_mask_from_lengths(
    lengths: torch.Tensor = None,
    x: torch.Tensor = None,
    pad_to_factor: int = None
) -> torch.Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths
    Args:
        lengths: torch.tensor (torch.tensor): 1D tensor with lengths
        x: torch.tensor = tensor to be used on, last dimension is for mask
    Returns:
        mask (torch.tensor): num_sequences x max_length binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]

    if pad_to_factor is not None:
        with fp32_precision():
            max_len = torch.ceil(max_len / pad_to_factor) * pad_to_factor

    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask


from transformers import MimiModel, AutoFeatureExtractor
class MimiCodec(NeuralModule):
    def __init__(self, model_path_or_name="kyutai/mimi", num_codebooks=12):
        super().__init__()
        self.codec = MimiModel.from_pretrained(model_path_or_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path_or_name)
        self.num_codebooks = num_codebooks

    @property
    def device(self):
        return next(self.codec.parameters()).device

    @property
    def _codebook_size(self):
        return self.codec.config.codebook_size

    @property
    def _num_codebooks(self):
        return self.num_codebooks

    @property
    def samples_per_frame(self):
        return int(self.feature_extractor.sampling_rate // self.codec.config.frame_rate)

    def encode(self, audio, audio_len):
        audio = audio.squeeze(1)
        with fp32_precision():
            # make the audio divisible by frame rate and also by self.frame_stacking_factor with extra frames of 1 to avoid issues because we are removing a audio frame to shift target and input for TF
            audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.samples_per_frame, extra_frames=0)
            # explicitly encode then decode the audio inputs
            encoder_outputs = self.codec.encode(audio.unsqueeze(1).to(self.device), num_quantizers=self.num_codebooks)
            codes = encoder_outputs.audio_codes
            tokens_len = audio_len // self.samples_per_frame
            return codes.transpose(1, 2), tokens_len

    def decode(self, tokens, tokens_len):
        with fp32_precision():
            tokens = tokens.transpose(1, 2)
            # tokens: B, T', C'
            audio = self.codec.decode(tokens).audio_values.squeeze(1)
            audio_len = tokens_len * self.samples_per_frame
        return audio, audio_len

    def forward(self, audio, audio_len):
        tokens, tokens_len = self.encode(audio, audio_len)
        audio, audio_len = self.decode(tokens, tokens_len)
        return audio, audio_len


    def pad_audio_to_factor(self, audio, audio_len, samples_per_frame, extra_frames: int = 0):
        """
        Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `samples_per_frame * frame_stacking_factor`.

        Args:
            audio: input time-domain signal (B, T)
            audio_len: valid length for each example in the batch (B,)
            samples_per_frame: number of samples per frame

        Returns:
            padded_audio: Padded time-domain signal (B, T')
            padded_len: Adjusted valid lengths (B,)
        """
        with fp32_precision():
            padded_len = (samples_per_frame * torch.ceil(audio_len / samples_per_frame).int()) + (extra_frames * samples_per_frame)
        max_len = padded_len.max().int().item()
        num_padding = (max_len - audio.shape[1])
        padded_audio = F.pad(audio, (0, num_padding))   
        return padded_audio, padded_len



def setup_rvq_audio_codec(model):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = RVQVAEModel.from_pretrained(model.cfg.pretrained_ae_dir, strict=False).eval().to(model.device)
    for p in model.audio_codec.parameters():
        p.requires_grad = False

def setup_audio_codec(self):
    setup_rvq_audio_codec(self)
    assert callable(self.tts_model.set_rvq_embs)
    self.tts_model.set_rvq_embs(torch.stack([x.detach() for x in self.audio_codec.prvq.mus_list], 0))
    self.tts_model.rvq_embs = self.tts_model.rvq_embs.to(next(self.tts_model.parameters()).dtype)
    # compute target fps
    self.target_fps = self.target_sample_rate / self.audio_codec.config.wav_to_token_ratio
    self.target_samples_per_frame = self.audio_codec.config.wav_to_token_ratio

def subwords_to_chars(subword_ids: torch.Tensor,
                                 subword_id_to_char_ids: dict[int, tuple[int, ...]],
                                 bos_id: int,
                                 eos_id: int,
                                 pad_id: int):
    """
    Fully vectorized subword->char expansion across all BOS..EOS spans:
    - Handles multiple spans per batch
    - Preserves BOS/EOS
    - Truncates expansions to fit each span
    - Very fast on GPU
    """
    device = subword_ids.device
    B, T = subword_ids.shape

    # Build LUT
    max_subword_id = int(subword_ids.max().item())
    max_chars = max(len(v) for v in subword_id_to_char_ids.values()) if subword_id_to_char_ids else 0
    if max_chars == 0:
        return subword_ids.clone()

    char_expansion = torch.full((max_subword_id + 1, max_chars),
                                fill_value=pad_id, device=device, dtype=subword_ids.dtype)
    expansion_len = torch.zeros(max_subword_id + 1, dtype=torch.long, device=device)
    for k, v in subword_id_to_char_ids.items():
        if k <= max_subword_id:
            v_t = torch.tensor(v, device=device, dtype=subword_ids.dtype)
            char_expansion[k, :len(v_t)] = v_t
            expansion_len[k] = len(v_t)

    # Output initialized with PAD
    output = torch.full_like(subword_ids, fill_value=pad_id)
    special_mask = (subword_ids == bos_id) | (subword_ids == eos_id)
    output[special_mask] = subword_ids[special_mask]

    # Find next EOS for each position
    pos = torch.arange(T, device=device)
    bos_mask = (subword_ids == bos_id)
    eos_mask = (subword_ids == eos_id)
    eos_pos_tensor = torch.where(eos_mask, pos.unsqueeze(0).expand(B, T),
                                 torch.full((B, T), T, device=device))
    next_eos_idx = torch.flip(torch.cummin(torch.flip(eos_pos_tensor, [1]), dim=1).values, [1])

    # Collect all BOS coordinates
    bos_coords = torch.nonzero(bos_mask, as_tuple=False)
    if bos_coords.numel() == 0:
        return output

    batch_ids = bos_coords[:, 0]
    span_starts = bos_coords[:, 1] + 1
    span_ends = next_eos_idx[batch_ids, bos_coords[:, 1]]
    span_lens = (span_ends - span_starts).clamp(min=0)
    S = span_lens.numel()
    if S == 0:
        return output

    # Max span length
    max_span_len = int(span_lens.max().item())

    # Gather subwords for all spans [S, max_span_len]
    rel = torch.arange(max_span_len, device=device).unsqueeze(0).expand(S, -1)
    span_idx = span_starts.unsqueeze(1) + rel
    span_idx_clamped = span_idx.clamp(0, T-1)
    batch_idx_expand = batch_ids.unsqueeze(1).expand(-1, max_span_len)
    sub_span = subword_ids[batch_idx_expand, span_idx_clamped]

    # Mask positions beyond actual span length
    valid_pos_mask = rel < span_lens.unsqueeze(1)
    sub_span = torch.where(valid_pos_mask, sub_span, torch.full_like(sub_span, pad_id))

    # Expand subwords -> chars
    expanded = char_expansion[sub_span]  # [S, max_span_len, max_chars]
    S_len = max_span_len * max_chars
    expanded_flat = expanded.view(S, S_len)
    valid_char_mask = expanded_flat != pad_id
    valid_cumsum = torch.cumsum(valid_char_mask.long(), dim=1)
    span_lens_exp = span_lens.unsqueeze(1).expand(-1, S_len)
    keep_mask = valid_char_mask & (valid_cumsum <= span_lens_exp)

    # Compute flattened indices to scatter
    rank_flat = (valid_cumsum - 1).clamp(min=0).view(-1)
    values_flat = expanded_flat.view(-1)
    keep_flat = keep_mask.view(-1)
    kept_values = values_flat[keep_flat]

    target_positions = (span_starts.unsqueeze(1).repeat(1, S_len).view(-1))[keep_flat] + rank_flat[keep_flat]
    target_batches = batch_ids.unsqueeze(1).repeat(1, S_len).view(-1)[keep_flat]

    # Safety clamp
    within_T = target_positions < T
    kept_values = kept_values[within_T]
    target_positions = target_positions[within_T]
    target_batches = target_batches[within_T]

    # Scatter in one shot
    output[target_batches, target_positions] = kept_values

    return output


def subwords_to_chars_batched(subword_ids: torch.Tensor,
                              subword_id_to_char_ids: dict[int, tuple[int, ...]],
                              bos_id: int,
                              eos_id: int,
                              pad_id: int,
                              silence_id: int = 0):
    """
    Batched subword->char expansion per BOS..EOS span.
    - Multiple spans per batch
    - Fully vectorized (no Python loop over spans)
    - BOS/EOS exact
    - Silences between spans
    """
    B, T = subword_ids.shape
    device = subword_ids.device

    # Build LUT
    max_subword_id = int(subword_ids.max().item())
    max_chars = max(len(v) for v in subword_id_to_char_ids.values()) if subword_id_to_char_ids else 0
    if max_chars == 0:
        return subword_ids.clone()

    char_expansion = torch.full((max_subword_id + 1, max_chars),
                                fill_value=pad_id, device=device, dtype=subword_ids.dtype)
    expansion_len = torch.zeros(max_subword_id + 1, dtype=torch.long, device=device)
    for k, v in subword_id_to_char_ids.items():
        if k <= max_subword_id:
            v_t = torch.tensor(v, device=device, dtype=subword_ids.dtype)
            char_expansion[k, :len(v_t)] = v_t
            expansion_len[k] = len(v_t)

    # Output initialized with PAD
    output = torch.full_like(subword_ids, fill_value=pad_id)
    special_mask = (subword_ids == bos_id) | (subword_ids == eos_id)
    output[special_mask] = subword_ids[special_mask]

    # Masks
    bos_mask = (subword_ids == bos_id)
    eos_mask = (subword_ids == eos_id)

    # Compute next EOS per position
    pos = torch.arange(T, device=device)
    eos_pos_tensor = torch.where(eos_mask, pos.unsqueeze(0).expand(B, T),
                                 torch.full((B, T), T, device=device))
    next_eos_idx = torch.flip(torch.cummin(torch.flip(eos_pos_tensor, [1]), dim=1).values, [1])

    # Collect all spans
    bos_coords = torch.nonzero(bos_mask, as_tuple=False)
    if bos_coords.numel() == 0:
        return output

    batch_ids = bos_coords[:, 0]
    span_starts = bos_coords[:, 1] + 1
    span_ends = next_eos_idx[batch_ids, bos_coords[:, 1]]
    span_lens = (span_ends - span_starts).clamp(min=0)
    S = span_lens.numel()
    if S == 0:
        return output

    # Gather subwords for all spans
    max_span_len = int(span_lens.max().item())
    rel = torch.arange(max_span_len, device=device).unsqueeze(0).expand(S, -1)
    span_idx = span_starts.unsqueeze(1) + rel
    span_idx_clamped = span_idx.clamp(0, T - 1)
    batch_idx_expand = batch_ids.unsqueeze(1).expand(-1, max_span_len)
    sub_span = subword_ids[batch_idx_expand, span_idx_clamped]

    # Mask positions beyond actual span length
    valid_pos_mask = rel < span_lens.unsqueeze(1)
    sub_span = torch.where(valid_pos_mask, sub_span, torch.full_like(sub_span, pad_id))

    # Expand subwords -> chars
    expanded = char_expansion[sub_span]                 # [S, max_span_len, max_chars]
    S_len = max_span_len * max_chars
    expanded_flat = expanded.view(S, S_len)
    valid_char_mask = expanded_flat != pad_id
    valid_cumsum = torch.cumsum(valid_char_mask.long(), dim=1)
    span_lens_exp = span_lens.unsqueeze(1).expand(-1, S_len)
    keep_mask = valid_char_mask & (valid_cumsum <= span_lens_exp)

    # Compute target positions
    rank_flat = (valid_cumsum - 1).clamp(min=0).view(-1)
    values_flat = expanded_flat.view(-1)
    keep_flat = keep_mask.view(-1)
    kept_values = values_flat[keep_flat]

    target_positions = (span_starts.unsqueeze(1).repeat(1, S_len).view(-1))[keep_flat] + rank_flat[keep_flat]
    target_batches = batch_ids.unsqueeze(1).repeat(1, S_len).view(-1)[keep_flat]

    # Safety clamp
    within_T = target_positions < T
    kept_values = kept_values[within_T]
    target_positions = target_positions[within_T]
    target_batches = target_batches[within_T]

    # Scatter in one shot
    output[target_batches, target_positions] = kept_values

    return output


def build_char_expansion_lut(subword_id_to_char_ids: dict[int, tuple[int, ...]],
                             pad_id: int,
                             device: str = "cuda"):
    """
    Prebuild the LUT once for training.
    Returns:
        char_expansion: [max_subword_id+1, max_chars]
        expansion_len: number of chars per subword
    """
    if not subword_id_to_char_ids:
        return None, None

    max_subword_id = max(subword_id_to_char_ids.keys())
    max_chars = max(len(v) for v in subword_id_to_char_ids.values())
    char_expansion = torch.full((max_subword_id + 1, max_chars),
                                fill_value=pad_id, device=device, dtype=torch.long)
    expansion_len = torch.zeros(max_subword_id + 1, device=device, dtype=torch.long)

    for k, v in subword_id_to_char_ids.items():
        if k <= max_subword_id:
            v_t = torch.tensor(v, device=device, dtype=torch.long)
            char_expansion[k, :len(v_t)] = v_t
            expansion_len[k] = len(v_t)

    return char_expansion, expansion_len


def subwords_to_chars_batched_fast(subword_ids: torch.Tensor,
                                   char_expansion: torch.Tensor,
                                   expansion_len: torch.Tensor,
                                   bos_id: int,
                                   eos_id: int,
                                   pad_id: int):
    """
    Fast batched subword->char expansion using prebuilt LUT.
    Fully vectorized, multiple spans per batch, no autograd overhead.
    """
    with torch.no_grad():
        if char_expansion is None:
            return subword_ids.clone()

        B, T = subword_ids.shape
        device = subword_ids.device

        # Initialize output
        output = torch.full_like(subword_ids, pad_id)
        special_mask = (subword_ids == bos_id) | (subword_ids == eos_id)
        output[special_mask] = subword_ids[special_mask]

        # Masks
        bos_mask = (subword_ids == bos_id)
        eos_mask = (subword_ids == eos_id)

        # Next EOS per position
        pos = torch.arange(T, device=device)
        eos_pos_tensor = torch.where(eos_mask, pos.unsqueeze(0).expand(B, T),
                                    torch.full((B, T), T, device=device))
        next_eos_idx = torch.flip(torch.cummin(torch.flip(eos_pos_tensor, [1]), dim=1).values, [1])

        # Collect all spans
        bos_coords = torch.nonzero(bos_mask, as_tuple=False)
        if bos_coords.numel() == 0:
            return output

        batch_ids = bos_coords[:, 0]
        span_starts = bos_coords[:, 1] + 1
        span_ends = next_eos_idx[batch_ids, bos_coords[:, 1]]
        span_lens = (span_ends - span_starts).clamp(min=0)
        S = span_lens.numel()
        if S == 0:
            return output

        # Gather subwords
        max_span_len = int(span_lens.max().item())
        rel = torch.arange(max_span_len, device=device).unsqueeze(0).expand(S, -1)
        span_idx = span_starts.unsqueeze(1) + rel
        span_idx_clamped = span_idx.clamp(0, T-1)
        batch_idx_expand = batch_ids.unsqueeze(1).expand(-1, max_span_len)
        sub_span = subword_ids[batch_idx_expand, span_idx_clamped]

        valid_pos_mask = rel < span_lens.unsqueeze(1)
        sub_span = torch.where(valid_pos_mask, sub_span, torch.full_like(sub_span, pad_id))

        # Expand using prebuilt LUT
        expanded = char_expansion[sub_span]                # [S, max_span_len, max_chars]
        S_len = max_span_len * char_expansion.shape[1]
        expanded_flat = expanded.view(S, S_len)

        valid_char_mask = expanded_flat != pad_id
        valid_cumsum = torch.cumsum(valid_char_mask.long(), dim=1)
        span_lens_exp = span_lens.unsqueeze(1).expand(-1, S_len)
        keep_mask = valid_char_mask & (valid_cumsum <= span_lens_exp)

        rank_flat = (valid_cumsum - 1).clamp(min=0).view(-1)
        values_flat = expanded_flat.view(-1)
        keep_flat = keep_mask.view(-1)
        kept_values = values_flat[keep_flat]

        target_positions = (span_starts.unsqueeze(1).repeat(1, S_len).view(-1))[keep_flat] + rank_flat[keep_flat]
        target_batches = batch_ids.unsqueeze(1).repeat(1, S_len).view(-1)[keep_flat]

        within_T = target_positions < T
        kept_values = kept_values[within_T]
        target_positions = target_positions[within_T]
        target_batches = target_batches[within_T]

        output[target_batches, target_positions] = kept_values

    return output


class WordSepTokenizer(AutoTokenizer):
    """
    Tokenizer wrapper that inserts a special word-separator token before each token 
    that starts a new word. This is useful for Speech-LLM and TTS pipelines 
    that require explicit word boundaries in the token sequence.

    Supported models:
        - LLaMA-3.1-family
        - NVIDIA Nemotron Nano-9B-v2

    Attributes:
        word_sep_token (str): The special token used to mark word boundaries.
        word_boundary_prefix (str): The token prefix indicating a word boundary.
        word_sep_id (int): The token ID corresponding to `word_sep_token`.
    """

    def __init__(self, model_name: str, *args, **kwargs):
        """
        Initializes the WordSepTokenizer.

        Args:
            model_name (str): Name of the model to load. Determines the special 
                              word-separator token and word boundary prefix.
            *args: Additional positional arguments passed to the base `AutoTokenizer`.
            **kwargs: Additional keyword arguments passed to the base `AutoTokenizer`.

        Raises:
            ValueError: If `model_name` is not supported.
        """
        super().__init__(model_name, *args, **kwargs)

        model_name_lower = model_name.lower()
        if "llama-3.1" in model_name_lower:
            self.word_sep_token = "<|reserved_special_token_0|>"
            self.word_boundary_prefix = "Ġ"
        elif "qwen2.5" in model_name_lower:
            self.word_sep_token = "<|box_start|>"
            self.word_boundary_prefix = "Ġ"
        elif "nvidia-nemotron-nano-9b-v2" in model_name_lower:
            self.word_sep_token = "<SPECIAL_10>"
            self.word_boundary_prefix = "Ġ"
        else:
            raise ValueError(
                f"WordSepTokenizer does not support model '{model_name}'. "
                "Supported: LLaMA-3.1-family, NVIDIA Nemotron Nano-9B-v2."
            )

        self.word_sep_id = self.tokenizer.convert_tokens_to_ids(self.word_sep_token)

    def text_to_ids(self, text: str):
        """
        Converts input text into token IDs, inserting the word-separator ID 
        before tokens that start a new word.

        Args:
            text (str): Input string to tokenize.

        Returns:
            List[int]: Token IDs with word-separator IDs inserted.

        Notes:
            - If `text` is empty or tokenization returns no tokens, returns an empty list.
            - The first token separator (if any) is removed to avoid leading separators.
        """
        if not text:
            return []

        # ensures that first word has a space to avoid different tokens for the first word
        if text[0] != " ":
            text = " " + text

        # Original token IDs
        ids = super().text_to_ids(text)
        if not ids:
            return []

        # Convert IDs to tokens safely (must be CPU Python list, no separator IDs yet)
        tokens = self.tokenizer.convert_ids_to_tokens(list(ids))

        # Mask for tokens starting with word boundary
        mask = [t.startswith(self.word_boundary_prefix) for t in tokens]

        # Prepare result
        result = []
        for tid, m in zip(ids, mask):
            if m:
                result.append(self.word_sep_id)
            result.append(tid)

        # Remove leading separator if present
        if result and result[0] == self.word_sep_id:
            result = result[1:]

        return result

    def ids_to_text(self, ids):
        """
        Converts token IDs back to text, replacing word-separator tokens with spaces.

        Args:
            ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text with word separators converted to spaces.
        """
        text = super().ids_to_text(ids)
        return text.replace(self.word_sep_token, " ")


class NeMoGroupedCodec(NeuralModule):
    def __init__(self, codec, frame_stacking_factor=1):
        super().__init__()
        self.codec = codec
        self.frame_stacking_factor = frame_stacking_factor

    @property
    def device(self):
        return self.codec.device

    @property
    def _codebook_size(self):
        return self.codec.vector_quantizer.codebook_size_per_group

    @property
    def _num_codebooks(self):
        return self.codec.vector_quantizer.num_groups * self.frame_stacking_factor

    @property
    def samples_per_frame(self):
        return self.codec.samples_per_frame * self.frame_stacking_factor

    def encode(self, audio, audio_len):
        audio = audio.squeeze(1)
        with fp32_precision():
            # make the audio divisible by frame rate and also by self.frame_stacking_factor with extra frames of 1 to avoid issues because we are removing a audio frame to shift target and input for TF
            audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.samples_per_frame, extra_frames=0)
            # encodes audio using the codec
            tokens, tokens_len = self.codec.encode(audio=audio, audio_len=audio_len)  # B, C, T
            tokens = tokens.transpose(1, 2)  # → B, T, C
            B, T, C = tokens.shape
            assert T % self.frame_stacking_factor == 0
            grouped = tokens.reshape(B, T // self.frame_stacking_factor, C * self.frame_stacking_factor)
            tokens_len = tokens_len // self.frame_stacking_factor
            # grouped = grouped.transpose(1, 2)

            return grouped, tokens_len

    def decode(self, tokens, tokens_len):
        with fp32_precision():
            # tokens = tokens.transpose(1, 2)
            # tokens: B, T', C'
            B, T, Cg = tokens.shape
            assert Cg % self.frame_stacking_factor == 0
            C = Cg // self.frame_stacking_factor
            ungrouped = tokens.reshape(B, T * self.frame_stacking_factor, C)  # → [B, T, C]
            ungrouped = ungrouped.transpose(1, 2)      # → [B, C, T] for decode
            tokens_len = torch.ceil(tokens_len * self.frame_stacking_factor).to(tokens_len.dtype)
            audio, audio_len = self.codec.decode(tokens=ungrouped, tokens_len=tokens_len)
        return audio, audio_len

    def decode_audio(self, inputs: torch.Tensor, input_len: torch.Tensor):
        """Apply decoder on the input. Note that the input is a non-quantized encoder output or a dequantized representation.

        Args:
            inputs: encoded signal
            input_len: valid length for each example in the batch

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        with fp32_precision():
            if self.frame_stacking_factor > 1:
                inputs = inputs.transpose(1, 2)
                B, T, Cg = inputs.shape
                C = Cg // self.frame_stacking_factor
                inputs = inputs.reshape(B, T * self.frame_stacking_factor, C)  # → [B, T, C]
                input_len = torch.ceil(input_len * self.frame_stacking_factor).to(input_len.dtype)
                inputs = inputs.transpose(1, 2)

            audio, audio_len = self.codec.audio_decoder(inputs=inputs, input_len=input_len)
        return audio, audio_len

    def dequantize(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> torch.Tensor:
        """Convert the discrete tokens into a continuous encoded representation.

        Args:
            tokens: discrete tokens for each codebook for each time frame
            tokens_len: valid length of each example in the batch

        Returns:
            Continuous encoded representation of the discrete input representation.
        """
        with fp32_precision():
            # reshape to dequantize
            if self.frame_stacking_factor > 1:
                tokens = tokens.transpose(1, 2)
                # tokens: B, T', C'
                B, T, Cg = tokens.shape
                assert Cg % self.frame_stacking_factor == 0
                C = Cg // self.frame_stacking_factor
                tokens = tokens.reshape(B, T * self.frame_stacking_factor, C)  # → [B, T, C]
                tokens = tokens.transpose(1, 2)      # → [B, C, T] for decode
                tokens_len = torch.ceil(tokens_len * self.frame_stacking_factor).to(tokens_len.dtype)
            dequantized = self.codec.dequantize(tokens=tokens, tokens_len=tokens_len)
            # reshape back to the compress form if needed
            if self.frame_stacking_factor > 1:
                dequantized = dequantized.transpose(1, 2)  # → B, T, C
                B, T, C = dequantized.shape
                assert T % self.frame_stacking_factor == 0
                dequantized = dequantized.reshape(B, T // self.frame_stacking_factor, C * self.frame_stacking_factor)
                dequantized = dequantized.transpose(1, 2)  # → B, C, T

        return dequantized

    def forward(self, audio, audio_len):
        tokens, tokens_len = self.encode(audio, audio_len)
        audio, audio_len = self.decode(tokens, tokens_len)
        return audio, audio_len

    def pad_audio_to_factor(self, audio, audio_len, samples_per_frame, extra_frames: int = 0):
        """
        Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `samples_per_frame * frame_stacking_factor`.

        Args:
            audio: input time-domain signal (B, T)
            audio_len: valid length for each example in the batch (B,)
            samples_per_frame: number of samples per frame

        Returns:
            padded_audio: Padded time-domain signal (B, T')
            padded_len: Adjusted valid lengths (B,)
        """
        with fp32_precision():
            padded_len = (samples_per_frame * torch.ceil(audio_len / samples_per_frame).int()) + (extra_frames * samples_per_frame)
        max_len = padded_len.max().int().item()
        num_padding = (max_len - audio.shape[1])
        padded_audio = F.pad(audio, (0, num_padding))   
        return padded_audio, padded_len

import math

def compare_dicts(dict_a, dict_b):
    all_keys = set(dict_a.keys()).union(dict_b.keys())
    equal = True
    differing_keys = []

    for key in sorted(all_keys):
        a_val = dict_a.get(key, None)
        b_val = dict_b.get(key, None)

        # Skip if value is None in either dict
        if a_val is None or b_val is None:
            continue

        # Handle both being NaN (float)
        if (isinstance(a_val, float) and math.isnan(a_val)) and \
           (isinstance(b_val, float) and math.isnan(b_val)):
            continue

        # Handle both being tensors
        if isinstance(a_val, torch.Tensor) and isinstance(b_val, torch.Tensor):
            # Shape mismatch
            if a_val.shape != b_val.shape:
                print(f"❌ Shape mismatch at key '{key}': {a_val.shape} vs {b_val.shape}")
                equal = False
                differing_keys.append(key)
                continue

            # Compare tensors elementwise (treating NaNs as equal)
            diff_mask = ~(torch.isclose(a_val, b_val, equal_nan=True))
            if diff_mask.any():
                equal = False
                differing_keys.append(key)
                idx = torch.nonzero(diff_mask, as_tuple=False)
                print(f"❌ Tensor mismatch at key '{key}': {idx.shape[0]} differing positions, shape: ", a_val.shape, b_val.shape)
                # Print up to first 10 differences
                for i, pos in enumerate(idx[:10]):
                    pos_tuple = tuple(pos.tolist())
                    a_item = a_val[pos_tuple].item()
                    b_item = b_val[pos_tuple].item()
                    print(f"    Position {pos_tuple}: {a_item} vs {b_item}")
                if idx.shape[0] > 10:
                    print(f"    ... and {idx.shape[0] - 10} more differences")
            continue

        # Fallback: direct comparison
        if a_val != b_val:
            print(f"❌ Value mismatch at key '{key}': {a_val} vs {b_val}")
            equal = False
            differing_keys.append(key)

    if equal:
        print("✅ All comparable keys and values match!")
    else:
        print("⚠️ Some keys/values differ (see above).")

    return equal, differing_keys

import copy
def extract_first_tensor(x):
    """Recursively find the first tensor in nested structures."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = extract_first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = extract_first_tensor(v)
            if t is not None:
                return t
    return None

def compare_tts_model_fp32_bf16_old(tts_model, inputs, atol=1e-3, topk=15):
    model_fp32 = copy.deepcopy(tts_model).eval().to(torch.float32)
    model_bf16 = copy.deepcopy(tts_model).eval().to(torch.bfloat16)

    diffs = {}

    def make_hook(name, tag):
        def hook_fn(module, inp, out):
            tensor = extract_first_tensor(out)
            if tensor is not None:
                tensor = tensor.detach().float().cpu()
                if name not in diffs:
                    diffs[name] = {}
                diffs[name][tag] = tensor
        return hook_fn

    # Register hooks independently
    hooks_fp32 = []
    for name, module in model_fp32.named_modules():
        hooks_fp32.append(module.register_forward_hook(make_hook(name, "fp32")))

    hooks_bf16 = []
    for name, module in model_bf16.named_modules():
        hooks_bf16.append(module.register_forward_hook(make_hook(name, "bf16")))

    def maybe_to(x, dtype):
        if x is None:
            return None
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.to(dtype)
        return x

    with torch.no_grad():
        # BF16 forward
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _ = model_bf16(
                code=maybe_to(inputs["code"], torch.bfloat16),
                audio_mask=inputs["audio_mask"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                context_hidden_state=maybe_to(inputs["context_hidden_state"], torch.bfloat16),
                subword_ids=inputs["subword_ids"],
                subword_mask=inputs["subword_mask"],
                non_prompt_mask=inputs["non_prompt_mask"]
            )

        # FP32 forward
        _ = model_fp32(
            code=maybe_to(inputs["code"], torch.float32),
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=maybe_to(inputs["context_hidden_state"], torch.float32),
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"]
        )

    # Compute diffs for matching layers
    diff_list = []
    for name, val in diffs.items():
        if "fp32" in val and "bf16" in val:
            delta = (val["fp32"] - val["bf16"]).abs().mean().item()
            diff_list.append((name, delta))

    diff_list.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {topk} layers with largest FP32 vs BF16 diff:")
    if not diff_list:
        print("⚠️ No matching tensor outputs found. Try increasing atol or check nested outputs.")
    else:
        for name, delta in diff_list[:topk]:
            print(f"{name:<60} mean abs diff = {delta:.6f}")

    for h in hooks_fp32 + hooks_bf16:
        h.remove()

    return diff_list

def compare_tts_model_fp32_bf16_mixed(tts_model, inputs, topk=15):
    """
    Compare FP32 vs BF16-safe (with fp32_precision layers) outputs.
    tts_model can have patched FP32 layers; these will run in FP32.
    """
    import copy
    diffs = {}

    def extract_first_tensor(x):
        if isinstance(x, (tuple, list)):
            for y in x:
                if torch.is_tensor(y):
                    return y
            return None
        if torch.is_tensor(x):
            return x
        return None

    def make_hook(name, tag):
        def hook_fn(module, inp, out):
            tensor = extract_first_tensor(out)
            if tensor is not None:
                tensor = tensor.detach().float().cpu()
                if name not in diffs:
                    diffs[name] = {}
                diffs[name][tag] = tensor
        return hook_fn

    # FP32 reference model
    model_fp32 = copy.deepcopy(tts_model).eval().to(torch.float32)

    hooks_fp32 = [m.register_forward_hook(make_hook(n, "fp32")) for n, m in model_fp32.named_modules()]
    hooks_bf16 = [m.register_forward_hook(make_hook(n, "bf16")) for n, m in tts_model.named_modules()]

    def maybe_to(x, dtype):
        if x is None:
            return None
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.to(dtype)
        return x

    with torch.no_grad():
        # BF16-safe forward (patched FP32 layers run in FP32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = tts_model(
                code=maybe_to(inputs["code"], torch.bfloat16),
                audio_mask=inputs["audio_mask"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                context_hidden_state=maybe_to(inputs["context_hidden_state"], torch.bfloat16),
                subword_ids=inputs["subword_ids"],
                subword_mask=inputs["subword_mask"],
                non_prompt_mask=inputs["non_prompt_mask"]
            )

        # FP32 forward
        _ = model_fp32(
            code=maybe_to(inputs["code"], torch.float32),
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=maybe_to(inputs["context_hidden_state"], torch.float32),
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"]
        )

    # Compute diffs
    diff_list = []
    for name, val in diffs.items():
        if "fp32" in val and "bf16" in val:
            delta = (val["fp32"] - val["bf16"]).abs().mean().item()
            diff_list.append((name, delta))

    diff_list.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop {topk} layers with largest FP32 vs BF16 diff:")
    for name, delta in diff_list[:topk]:
        print(f"{name:<60} mean abs diff = {delta:.6f}")

    for h in hooks_fp32 + hooks_bf16:
        h.remove()

    return diff_list

def rescale_state_dict(
    state_dict,
    target_std=0.02,
    first_n_layers=None,
    layer_prefix="tts_model.backbone.layers."
):
    """
    Rescale trainable weights in a state_dict for BF16 stability.

    Args:
        state_dict: PyTorch state_dict
        target_std: desired target std for weights
        first_n_layers: if not None, rescale only the first N transformer blocks
        layer_prefix: prefix for layer names (default: "tts_model.backbone.layers.")
    Returns:
        new_state_dict
    """
    weight_tensors = []

    # Compute which prefixes to match if first_n_layers is set
    prefixes_to_match = []
    if first_n_layers is not None:
        prefixes_to_match = [f"{layer_prefix}{i}" for i in range(first_n_layers)]

    for name, param in state_dict.items():
        if not torch.is_tensor(param):
            continue

        if "rvq_embs" in name:
            continue

        # Skip biases & 1-dim params (norm weights/gates)
        if param.ndim <= 1:
            continue

        # Skip layers not in the first N
        if first_n_layers is not None and not any(name.startswith(pfx) for pfx in prefixes_to_match):
            continue

        weight_tensors.append(param.float())

    if not weight_tensors:
        if first_n_layers is not None:
            print(f"⚠️ No weights found for first {first_n_layers} layers with prefix '{layer_prefix}'.")
        else:
            print("⚠️ No weights found to rescale in state_dict.")
        return state_dict

    # Compute global std across selected weights (on CPU)
    cpu_weights = [p.detach().cpu() for p in weight_tensors]
    flat = torch.cat([p.flatten() for p in cpu_weights])
    current_std = float(torch.std(flat))
    scale = target_std / (current_std + 1e-8)

    print(
        f"📦 Rescaling state_dict "
        f"{'(first N layers)' if first_n_layers else '(all layers)'}: "
        f"current std = {current_std:.6f}, target = {target_std}, scale = {scale:.6f}"
    )

    # Apply scaling
    new_state_dict = {}
    for name, param in state_dict.items():
        if (
            torch.is_tensor(param)
            and param.ndim > 1
            and (first_n_layers is None or any(name.startswith(pfx) for pfx in prefixes_to_match))
        ):
            new_state_dict[name] = param * scale
        else:
            new_state_dict[name] = param

    print("✅ Done: weights rescaled.")
    return new_state_dict


class DuplexEARTTS(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexEARTTS as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.trainer_config = cfg.trainer
        self.data_cfg = cfg.data
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.normalize_text = cfg.data.get("normalize_text", False)
        self.model_16_precision_safe = None

        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # Load ForCausalLM
        if self.cfg.tts_config.context_hidden_size is not None:
            self.language_model = self._load_language_model(self.cfg)
            self.embed_tokens = self._load_embed_tokens(self.cfg)
            # delete llm because we use it only to get the  embbeding tokens
            del self.language_model
            if self.cfg.tts_config.get("use_subword_flag_emb", False):
                self.subword_flag_emb = SubwordFlagEmbedding(self.cfg.pretrained_lm_name, self.cfg.tts_config.context_hidden_size)

        # instanciate eartts model and codec
        self._load_tts_model(self.cfg)
        self._codebook_size = self.tts_model.config.codebook_size

        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps
        self.source_samples_per_frame = int(self.source_sample_rate//self.source_fps)

        # get codec silence tokens
        self.codec_silence_tokens = self.get_codec_silence_frame()

        # Load tokenizer
        if self.cfg.get("use_word_sep_tokenizer", False):
            self.tokenizer = WordSepTokenizer(self.cfg.pretrained_lm_name, use_fast=True, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_lm_name, use_fast=True, trust_remote_code=True) # Note that we are using fast tokenizer

        if 'Qwen2.5' in self.cfg.pretrained_lm_name:
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'

        elif 'Nemotron' in self.cfg.pretrained_lm_name:
            # ====== NEMOTRON-SPECIFIC HANDLING ======
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token = '<SPECIAL_12>'

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_pad_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False
        if self.cfg.get("pretrained_model", None):
            self.init_model_from_another_checkpoint(self.cfg.pretrained_model)

    def get_codec_silence_frame_last_one(self):
        audio = torch.zeros(1, 10*self.target_sample_rate).float().to(self.device)
        audio_len = torch.tensor([audio.size(-1)]).long()
        audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.target_samples_per_frame)

        with fp32_precision(), torch.no_grad():
            sil_codes, sil_codes_lens = self.audio_codec.encode(
                    audio.unsqueeze(1), audio_len
                )
            return sil_codes[0, -1]

    def get_codec_silence_frame(self):
        from collections import Counter

        # Generate long zero waveform (silence)
        audio = torch.zeros(1, 10 * self.target_sample_rate).float().to(self.device)
        audio_len = torch.tensor([audio.size(-1)]).long()
        audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.target_samples_per_frame)

        with fp32_precision(), torch.no_grad():
            sil_codes, _ = self.audio_codec.encode(audio.unsqueeze(1), audio_len)  # [1, T, C]
            sil_codes = sil_codes[0]  # [T, C]

        # Convert each frame (C tokens) into a tuple
        combos = [tuple(row.tolist()) for row in sil_codes]

        # Count frequencies
        counter = Counter(combos)

        # Pick the most common combination
        most_common_combo, freq = counter.most_common(1)[0]

        # Return as tensor [C]
        return torch.tensor(most_common_combo, device=self.device, dtype=torch.long)

    def _load_embed_tokens(self, cfg) -> nn.Embedding:
        """Load token embedding layer for RVQ-EAR-TTS."""
        if self.language_model:
            assert callable(self.language_model.get_input_embeddings)
            embed_tokens: nn.Embedding = self.language_model.get_input_embeddings()
        else:
            embed_tokens_state_dict = torch.load(
                cfg.pretrained_lm_embedding_path, map_location="cpu", weights_only=True
            )

            # Create token embedding layer
            vocab_size, hidden_size = embed_tokens_state_dict["weight"].size()
            embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
            embed_tokens.load_state_dict(embed_tokens_state_dict)
        return embed_tokens

    def _load_tts_model(self, cfg) -> nn.Module:
        """Load TTS model for RVQ-EAR-TTS."""
        if self.cfg.get("pretrained_tts_model", None):
            self.tts_model = RVQEARTTSModel.from_pretrained(cfg.pretrained_tts_model, RVQEARTTSConfig(**cfg.tts_config), strict=False)
        else:
            # start the model from scratch
            self.tts_model = RVQEARTTSModel(RVQEARTTSConfig(**cfg.tts_config))

        setup_audio_codec(self)

    def _load_language_model(self, cfg):
        """Load language model for RVQ-EAR-TTS."""
        if cfg.pretrained_lm_name:
            language_model = load_pretrained_hf(self.cfg.pretrained_lm_name, pretrained_weights=True, trust_remote_code=True).eval()
        else:
            language_model = None
        return language_model

    def setup_speaker_encoder(self):
        with fp32_precision():
            self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(model_name=self.speaker_encoder_model_name)

        # freeze the pretrained speaker encoder
        self.speaker_encoder.eval()
        self.speaker_encoder.freeze()

        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

    def init_model_from_another_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())

            if self.cfg.get("rescale_pretrained_weights", None):
                checkpoint_state = rescale_state_dict(checkpoint_state, first_n_layers=self.cfg.get("rescale_first_n_layers", None))

            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        if self.use_local_transformer and self.local_transformer_type == "nar": # add extra token for mask
            return self._codebook_size + 4
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size + 2

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_pad_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_pad_id", None):
            return self.cfg.get("custom_speech_pad_id")
        return self._codebook_size

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_zstts_task_id(self) -> int:
        return self.tokenizer.text_to_ids("<|box_start|>") # uses <|box_start|> special token as zstts task id token

    @property
    def text_cont_task_id(self) -> int:
        return self.tokenizer.text_to_ids("<|object_ref_start|>") # uses <|object_ref_start|> special token as cont task id token

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def pad_audio_to_factor(self, audio, audio_len, samples_per_frame, downsampling_factor: int = 1):
        """
        Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `samples_per_frame * downsampling_factor`.

        Args:
            audio: input time-domain signal (B, T)
            audio_len: valid length for each example in the batch (B,)
            samples_per_frame: number of samples per frame
            downsampling_factor: how much each frame is downsampled in later processing

        Returns:
            padded_audio: Padded time-domain signal (B, T')
            padded_len: Adjusted valid lengths (B,)
        """
        with fp32_precision():
            total_factor = samples_per_frame * downsampling_factor
            padded_len = total_factor * torch.ceil(audio_len / total_factor).int()
            max_len = padded_len.max().int().item()
            num_padding = max_len - audio.shape[1]
            padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len
    
    def prepare_inputs(self, batch: dict):
        """
        """
        """
        import hashlib
        import torch

        def hash_texts(text_list):
            hashes = []
            for t in text_list:
                norm = t.strip().lower()
                h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                hashes.append(h)
            return hashes

        # --- Safe batch filtering function ---
        def filter_batch_by_indices(batch, keep_indices):
            if not keep_indices:
                # No common samples: empty all fields
                new_batch = {}
                for k, v in batch.items():
                    if isinstance(v, list):
                        new_batch[k] = []
                    elif hasattr(v, "__getitem__") and not isinstance(v, str):
                        try:
                            new_batch[k] = v[0:0]  # empty tensor
                        except Exception:
                            new_batch[k] = v
                    else:
                        new_batch[k] = v
                return new_batch

            new_batch = {}
            for k, v in batch.items():
                try:
                    if isinstance(v, list):
                        new_batch[k] = [v[i] for i in keep_indices if i < len(v)]
                    elif hasattr(v, "__getitem__") and not isinstance(v, str):
                        slices = [i for i in keep_indices if i < v.shape[0]]
                        if slices:
                            new_batch[k] = v[slices]
                        else:
                            new_batch[k] = v[0:0]  # empty tensor
                    else:
                        new_batch[k] = v  # keep metadata as-is
                except Exception:
                    new_batch[k] = v  # fallback if indexing fails
            return new_batch

        # --- Compute sample IDs ---
        target_texts = batch["target_texts"]
        batch["sample_id"] = target_texts  # using text itself as unique ID
        print("Sample ids:", batch["sample_id"])

        if self.training:
            # --- Track sample IDs and store full batch ---
            if not hasattr(self, "train_sample_ids"):
                self.train_sample_ids = set(batch["sample_id"])
                self.train_batches_by_hash = dict()
            else:
                self.train_sample_ids.update(batch["sample_id"])

            # Save the full batch per sample
            for i, sid in enumerate(batch["sample_id"]):
                self.train_batches_by_hash[sid] = {
                    k: (v[i] if isinstance(v, list) else v[i:i+1])
                    for k, v in batch.items()
                }

        else:
            # --- Validation: keep only common samples ---
            if not hasattr(self, "eval_common_ids"):
                self.eval_common_ids = set()

            # Only consider validation samples that exist in training
            keep_indices = [i for i, sid in enumerate(batch["sample_id"])
                            if sid in self.train_batches_by_hash]

            # Safe filtering
            batch = filter_batch_by_indices(batch, keep_indices)

            if keep_indices:
                print(f"Keeping only {len(keep_indices)} common samples from validation!")
                # Update eval_common_ids
                self.eval_common_ids.update(batch["sample_id"])
                print(
                    f"total_common={len(self.eval_common_ids)}, "
                    f"train_total={len(self.train_sample_ids)}"
                )

                # --- Compare the first common sample ---
                first_sid = batch["sample_id"][0]
                train_sample = self.train_batches_by_hash[first_sid]
                val_sample = {k: (v[0] if isinstance(v, list) else v[0:1])
                            for k, v in batch.items()}

                # --- Slice tensors to minimal overlapping shape ---
                for k in val_sample.keys():
                    t_val = val_sample[k]
                    t_train = train_sample.get(k, t_val)
                    if isinstance(t_val, torch.Tensor) and isinstance(t_train, torch.Tensor):
                        min_shape = tuple(min(s1, s2) for s1, s2 in zip(t_val.shape, t_train.shape))
                        if all(s > 0 for s in min_shape):
                            slices = tuple(slice(0, s) for s in min_shape)
                            val_sample[k] = t_val[slices]
                            train_sample[k] = t_train[slices]

                print(f"Comparing first common sample (sid={first_sid})")
                compare_dicts(train_sample, val_sample)
                exit()
            else:
                print("No common samples found in this validation batch!")

        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["speaker_reference_audio"].size(0) == batch["target_audio"].size(0)

        target_audio = batch["target_audio"]
        target_audio_lens = batch["target_audio_lens"]
        input_text_tokens = batch["input_text_tokens"]
        audio_mask = batch["audio_mask"]
        desc_mask = batch["desc_mask"]
        non_prompt_mask = batch["non_prompt_mask"]
        aligned_attention_mask = batch["aligned_attention_mask"]
        aligned_position_ids = batch["aligned_position_ids"]

        # extract target audio codes
        with fp32_precision(), torch.no_grad():
            target_audio, target_audio_lens = self.pad_audio_to_factor(target_audio, target_audio_lens, self.target_samples_per_frame, 1)
            target_codes, target_codes_lens = self.audio_codec.encode(
                target_audio.unsqueeze(1), target_audio_lens
            )

        # ToDo: consider use the source audio
        """
        # resample source audio if needed
        if self.source_sample_rate != self.target_sample_rate:
            source_audio = resample(source_audio, self.source_sample_rate, self.target_sample_rate)
            with fp32_precision():
                source_audio_lens = (source_audio_lens * (self.target_sample_rate/self.source_sample_rate)).to(lengths.dtype)
        # ToDo: Add a transformer encoder to help the model to better extract contextual information, replace the code bellow with it
        # extract embedding for context audios
        with fp32_precision(), torch.no_grad():
            source_audio, source_audio_lens = self.pad_audio_to_factor(source_audio, source_audio_lens, self.target_samples_per_frame, 1)
            source_codes, source_codes_lens = self.audio_codec.encode(
                source_audio.unsqueeze(1), source_audio_lens
            )
            source_codes = source_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)
        """

        with fp32_precision():
            target_len = target_codes.shape[1]

            # Pad or truncate sequence variables
            def pad_or_truncate(x, pad_value=0):
                if x.dim() == 2:  # [B, T]
                    L = x.shape[1]
                    if L < target_len:
                        return F.pad(x, (0, target_len - L), value=pad_value)
                    else:
                        return x[:, :target_len]
                return x  # leave others for now

            input_text_tokens = pad_or_truncate(input_text_tokens, pad_value=self.text_pad_id)
            audio_mask = pad_or_truncate(audio_mask, pad_value=0)
            desc_mask = pad_or_truncate(desc_mask, pad_value=0)
            non_prompt_mask = pad_or_truncate(non_prompt_mask, pad_value=0)
            aligned_position_ids = pad_or_truncate(aligned_position_ids, pad_value=0)

            # Correct attention mask padding/truncation
            B, H, L1, L2 = aligned_attention_mask.shape
            new_len = target_len
            if L1 < new_len or L2 < new_len:
                pad_rows = new_len - L1
                pad_cols = new_len - L2
                aligned_attention_mask = F.pad(aligned_attention_mask, (0, pad_cols, 0, pad_rows))
            elif L1 > new_len or L2 > new_len:
                aligned_attention_mask = aligned_attention_mask[:, :, :new_len, :new_len]

        if self.cfg.get("disable_speech_pad", False):
            target_codes_aligned = target_codes
        else:
            # ToDo: desc_mask is one for the end of the sequence, this is what cause the artifact issue in the end, fix it.
            # set the pad token when there is desc as in https://gitlab-master.nvidia.com/jaehyeonk/easy-ar-tts/-/blame/simple-bq/scripts/train_tts_with_rvqvae.py#L69
            target_codes_aligned = torch.where(
                desc_mask.unsqueeze(-1),                    # (B, T, 1) for broadcasting
                torch.full_like(target_codes, self.speech_pad_id),  # fill with pad id
                target_codes
            )

        if self.cfg.get("ignore_audio_prompt_on_loss", False):
            # set audio_mask as non_prompt_mask to avoid the audio prompt in loss computation
            audio_mask = non_prompt_mask

        if self.cfg.get("add_pad_speech_token_in_last_prompt_frame", False) and not self.cfg.get("disable_speech_pad", False):
            # set special token in the last audio prompt (it will works as a BOS token)
            pos = non_prompt_mask.float().argmax(dim=1)  # shape: [B]
            row_idx = torch.arange(B, device=self.device)
            # set the extra self.speech_pad_id at first 1 position in non_prompt_mask
            target_codes_aligned[row_idx, pos] = self.speech_pad_id

        B, T = input_text_tokens.shape

        # shift text tokens
        subword_ids = F.pad(input_text_tokens[:, 1:], [0, 1])
        # note that we are using a text mask where we are ignoring the desc + audio prompt but we are keeping 1 until the audio ends to support duplex
        subword_mask = F.pad(non_prompt_mask[:, 1:], [0, 1])

        # ToDo: implement context from the llm
        # detach embedding as in eartts
        if self.cfg.tts_config.context_hidden_size is not None:
            context_hidden_state = self.embed_tokens(input_text_tokens).detach()
            if self.cfg.tts_config.get("use_subword_flag_emb", False):
                context_hidden_state = self.subword_flag_emb(context_hidden_state, input_text_tokens)
        else:
            context_hidden_state = None

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_text_tokens.shape[1] - 1) % tp_world_size) != 0:
                input_text_tokens = input_text_tokens[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                audio_mask = audio_mask[:, :-remainder]
                desc_mask = desc_mask[:, :-remainder]
                subword_ids = subword_ids[:, :-remainder]
                subword_mask = subword_mask[:, :-remainder]

        return {
            "code": target_codes_aligned,
            "audio_mask": audio_mask,
            "attention_mask": aligned_attention_mask,
            "position_ids": aligned_position_ids,
            "subword_ids": subword_ids,
            "subword_mask": subword_mask,
            "context_hidden_state": context_hidden_state,
            "output_lens": target_codes_lens,
            "non_prompt_mask": non_prompt_mask,
            "input_text_tokens": input_text_tokens
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.tts_model, ):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=inputs["context_hidden_state"],
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"]
        )
        loss_dict = {"lm_loss": tts_output.lm_loss, "c_loss": tts_output.c_loss, "k_loss": tts_output.k_loss}
        backbone_out = tts_output.hidden_states
        loss = sum(loss_dict.values())

        num_frames = inputs["output_lens"].sum()
        B, T = inputs["code"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
            **loss_dict,
        }

        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        setup_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32

    def on_train_epoch_end(self) -> None:
        # log model stats to debug gradient weights issues
        self.log_model_stats()

    def log_model_stats(self):
        total_w_sq = 0.0
        total_w_params = 0
        max_abs_w = 0.0
        sum_w = 0.0

        total_g_sq = 0.0
        total_g_params = 0

        for p in self.parameters():
            if not p.requires_grad:
                continue

            # ----- weights -----
            w = p.detach().cpu().float()  # ✅ safe offline copy
            total_w_sq += (w * w).sum().item()
            total_w_params += w.numel()
            max_abs_w = max(max_abs_w, w.abs().max().item())
            sum_w += w.sum().item()

            # ----- grads (optional, disabled for speed) -----
            if p.grad is not None:
                g = p.grad.detach().cpu().float()
                total_g_sq += (g * g).sum().item()
                total_g_params += g.numel()

        # L2 norms
        weight_l2 = (total_w_sq ** 0.5) if total_w_sq > 0 else 0.0
        grad_l2   = (total_g_sq ** 0.5) if total_g_sq > 0 else 0.0

        # RMS (global)
        weight_rms = ((total_w_sq / total_w_params) ** 0.5) if total_w_params > 0 else 0.0
        grad_rms   = ((total_g_sq / total_g_params) ** 0.5) if total_g_params > 0 else 0.0

        # Mean
        weight_mean = sum_w / total_w_params if total_w_params > 0 else 0.0

        # direct float logging avoids device sync penalty
        self.log("weights/L2",      weight_l2,   on_epoch=True, sync_dist=True)
        self.log("weights/RMS",     weight_rms,  on_epoch=True, sync_dist=True)
        self.log("weights/max_abs", max_abs_w,   on_epoch=True, sync_dist=True)
        self.log("weights/mean",    weight_mean, on_epoch=True, sync_dist=True)

        # ignore the grads stats for now
        # self.log("grads/L2",       grad_l2,    on_epoch=True, sync_dist=True)
        # self.log("grads/RMS",      grad_rms,   on_epoch=True, sync_dist=True)

    def on_validation_epoch_start(self) -> None:
        setup_audio_codec(self)
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.intelligibility = Intelligibility(self.cfg.scoring_asr, reuse_asr_hyps=True).reset()
        self.secs = SECS(self.cfg.get("scoring_se", "titanet_large")).reset()
        
    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        cer_wer = self.intelligibility.compute()
        for k, m in cer_wer.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        secs = self.secs.compute()
        for k, m in secs.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def get_teacher_force_inference_audio(self, batch, guidance_enabled=True):
        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=inputs["context_hidden_state"],
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"],
            generation_config=self._get_generation_config(guidance_enabled=guidance_enabled),
            teacher_forcing_inference=True,
            guidance_enabled=guidance_enabled
        )
        tf_audio_codes_pred = tts_output["codes"].squeeze(2)

        # decode audio
        tf_audio_codes_pred = replace_control_speech_codes(tf_audio_codes_pred, self._control_codes, self.codec_silence_tokens)
        with fp32_precision(), torch.no_grad():
            audio_pred, audio_len = self.audio_codec.decode(
                tf_audio_codes_pred, inputs["output_lens"]
            )

        return audio_pred.squeeze(1), audio_len

    def _get_generation_config(self, guidance_enabled: bool = False):
        """Get default generation config for EAR-TTS."""
        return {
            "num_iter": 8,
            "guidance_scale": self.cfg.get("inference_guidance_scale", 0.5) if guidance_enabled else None,
            "top_p_or_k": self.cfg.get("inference_top_p_or_k", 0.8),
            "noise_scale": self.cfg.get("inference_noise_scale", 0.8),
            "eos_threshold": -3.0,
        }

    def offline_inference_with_custom_sentences(self, test_sentences: torch.Tensor, inference_speaker_reference: torch.Tensor, speech_text_ratio: float = 3.5):
        B = len(test_sentences)
        # load and get speaker reference
        speaker_audio, sr = torchaudio.load(inference_speaker_reference)
        speaker_audio = resample(speaker_audio, sr, self.target_sample_rate)
        speaker_audio = speaker_audio.repeat(B, 1).to(self.device) 
        # lengths -> [B]
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long().repeat(B)

        # Tokenize sentences
        if self.normalize_text:
            tokenized = [
                torch.as_tensor([self.tokenizer.bos] + self.tokenizer.text_to_ids(normalize_text_fn(text)), dtype=torch.long, device=self.device)
                for text in test_sentences
            ]
        else:
            tokenized = [
                torch.as_tensor([self.tokenizer.bos] + self.tokenizer.text_to_ids(text), dtype=torch.long, device=self.device)
                for text in test_sentences
            ]

        # Get max length and target length
        max_len = max(len(t) for t in tokenized)
        # Pad each to double length
        target_len = int(speech_text_ratio * max_len) # make text longer to ensures that we have enough steps for speech gen
        next_subword_ids = torch.stack([
            torch.cat([
                torch.tensor([self.text_pad_id], dtype=torch.long, device=self.device),  # shift right adding one padding token
                t,
                torch.full((target_len - len(t) - 1,), self.text_pad_id, dtype=torch.long, device=self.device)  # remaining padding
            ])
            for t in tokenized
        ])

        audio, audio_len = self.offline_inference(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
            next_subword_ids=next_subword_ids,
            guidance_enabled=self.cfg.get("inference_guidance_enabled", True)
        )
        return audio, audio_len, speaker_audio, speaker_audio_lens

    def run_evaluation_one_batch(self, name, dataset_batch, use_dataloader_init=False):
        results = {}
        inputs = self.prepare_inputs(dataset_batch)

        # 
        # exit()
        # first evaluation, make the model bf16 safe
        if not self.model_16_precision_safe and self.cfg.get("ensures_16_safe", False) and str(self.trainer_config.precision) != str(32):
            self.tts_model, summary = make_tts_model_mixed_precision_definite(self.tts_model, inputs, safety_factor=1.0, mixed_dtype=torch.float16 if str(self.trainer_config.precision) == str(16) else torch.bfloat16)
            # self.tts_model, summary = make_tts_model_mixed_precision_safe(self.tts_model, inputs, safety_factor=1.0)
            self.model_16_precision_safe = True

            print("Current FP32 layers:", summary["fp32_layers"])
            # compare_tts_model_fp32_bf16_mixed(self.tts_model, inputs)
            # exit()

        results["audio_tf"], results["audio_tf_len"] = self.get_teacher_force_inference_audio(dataset_batch)
        if use_dataloader_init:
            # cut it on prompt
            init_inputs = {
                "code": inputs["code"],
                "audio_mask": inputs["audio_mask"],
                "non_prompt_mask": inputs["non_prompt_mask"],
                "context_hidden_state": inputs["context_hidden_state"],
                "subword_ids": inputs["subword_ids"],
                "subword_mask": inputs["subword_mask"]
            }
            # cut init_inputs to consider only the prompt
            for key in init_inputs:
                if init_inputs[key] is not None:
                    init_inputs[key] = torch.stack([
                        init_inputs[key][i, :l]
                        for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"])
                    ])

        # drop items without description to avoid issues
        """
        lens = dataset_batch["desc_plus_audio_prompt_lens"]  # list of lengths

        # Example condition: keep only those with the maximum length
        max_len = max(lens)
        keep_indices = [i for i, l in enumerate(lens) if l == max_len]

        # Convert indices to tensor for indexing torch tensors
        keep_indices  = torch.tensor(keep_indices, dtype=torch.long)

        # Now filter every key in dataset_batch
        for k, v in dataset_batch.items():
            if isinstance(v, torch.Tensor):
                dataset_batch[k] = v[keep_indices]
            elif isinstance(v, list):
                dataset_batch[k] = [v[i] for i in keep_indices]

        # Do the same for inputs
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v[keep_indices]
            elif isinstance(v, list):
                inputs[k] = [v[i] for i in keep_indices]
        """

        # remove the prompt from the input_text_tokens to emulate S2S connected inference
        next_subword_ids = torch.stack([
            inputs["subword_ids"][i, l:]  # slice each element
            for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"])
        ])

        if self.cfg.get("use_asr_speech_tokens", False) and self.cfg.get("only_semantic_to_speech", False):
            inp_asr_speech_tokens = torch.stack([
                inputs["target_asr_speech_tokens"][i, l:]  # slice each element
                for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"])
            ])
        else:
            inp_asr_speech_tokens = None

        results["audio"], results["audio_len"] = self.offline_inference(
            speaker_audio=dataset_batch["speaker_reference_audio"],
            speaker_audio_lens=dataset_batch["speaker_reference_audio_lens"],
            next_subword_ids=next_subword_ids,
            formatter=dataset_batch["formatter"][0],
            inp_asr_speech_tokens=inp_asr_speech_tokens,
            init_inputs=init_inputs if use_dataloader_init else None,
        )

        # remove prompt padding from the user audio as autoregressive inference does not return the prompt
        dataset_batch["source_audio"] = dataset_batch["source_audio"][:, -int(next_subword_ids.size(-1)*self.source_samples_per_frame):]

        # clean prompt from the audio
        results["audio_tf"] = results["audio_tf"][:, -int(next_subword_ids.size(-1)*self.target_samples_per_frame):]
        # remove prompt from target audio
        target_audio_no_prompt = dataset_batch["target_audio"][:, -int(next_subword_ids.size(-1)*self.target_samples_per_frame):]
        target_audio_no_prompt_lens = dataset_batch["target_audio_lens"] - (torch.tensor(dataset_batch["desc_plus_audio_prompt_lens"], dtype=torch.long, device=dataset_batch["target_audio_lens"].device) * self.target_samples_per_frame)
        # for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"]):
        #    results["audio_tf"][i, :l*self.target_samples_per_frame] = 0.0

        with fp32_precision():  # resample is fragile to bfloat16 default dtype
            metric_audio_pred = results["audio"]
            metric_audio_pred_lens = results["audio_len"]

            # resample audio to the asr sampling rate
            metric_audio_pred = resample(metric_audio_pred, self.target_sample_rate, 16000)
            metric_audio_pred_lens = (metric_audio_pred_lens / self.target_sample_rate * 16000).to(torch.long)
            # reshape target audio without prompt
            target_audio_no_prompt_16khz = resample(target_audio_no_prompt, self.target_sample_rate, 16000)
            target_audio_no_prompt_lens_16khz = (target_audio_no_prompt_lens / self.target_sample_rate * 16000).to(torch.long)
            if self.cfg.get("use_GT_transcriptions_for_metrics", True):
                # use target audio transcription for metrics
                target_asr_texts = self.asr_bleu.asr.transcribe(
                    [audio[:alen] for audio, alen in zip(target_audio_no_prompt_16khz, target_audio_no_prompt_lens_16khz)],
                    batch_size=target_audio_no_prompt_16khz.shape[0],
                    verbose=False,
                )
                metric_text = [asr_hyp.text for asr_hyp in target_asr_texts]
            else:
                metric_text = dataset_batch["target_texts"]

            asr_hyps = self.asr_bleu.update(
                name=name,
                refs=metric_text,
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
            )

            self.intelligibility.update(
                name=name,
                refs=metric_text,
                pred_audio=metric_audio_pred,
                pred_audio_lens=metric_audio_pred_lens,
                asr_hyps=asr_hyps,
            )
            
            # add ground truth intelligibility metrics
            self.intelligibility.update(
                name=name+"_gt",
                refs=dataset_batch["target_texts"],
                pred_audio=target_audio_no_prompt_16khz,
                pred_audio_lens=target_audio_no_prompt_lens_16khz,
                asr_hyps=metric_text if self.cfg.get("use_GT_transcriptions_for_metrics", True) else None, # reuse GT transcription
            )

            self.secs.update(
                name=name,
                target_audio=resample(dataset_batch["target_audio"], self.target_sample_rate, 16000),
                target_audio_lens=(dataset_batch["target_audio_lens"] / self.target_sample_rate * 16000).to(torch.long),
                pred_audio=resample(results["audio"], self.target_sample_rate, 16000),
                pred_audio_lens=(results["audio_len"] / self.target_sample_rate * 16000).to(torch.long),
            )

            eou_labels = generate_multiturn_speaking_mask(
                next_subword_ids, bos_token_id=self.text_bos_id, eos_token_id=self.text_eos_id
            )

            self.results_logger.update(
                name=name,
                refs=dataset_batch["target_texts"],
                hyps=metric_text,
                asr_hyps=asr_hyps,
                samples_id=dataset_batch['sample_id'],
                pred_audio=results["audio"].float(),
                pred_audio_tf=results["audio_tf"].float(),
                pre_audio_trimmed=None,
                reference_audio=dataset_batch["speaker_reference_audio"].float(),
                target_audio=target_audio_no_prompt.float(),
                pred_audio_sr=self.target_sample_rate,
                user_audio=dataset_batch["source_audio"].float(),
                user_audio_sr=self.source_sample_rate,
                eou_pred=eou_labels,
                fps=self.target_fps,
                results=results if self.cfg.get("dump_tokens_text", False) else None,
                tokenizer=self.tokenizer,
            )

    def validation_step(self, batch: dict, batch_idx: int):
        if self.cfg.get("test_sentences", None) and self.cfg.get("inference_speaker_reference", None):
            for name in self.cfg.test_sentences.keys():
                logging.info(f"Generating {name} custom sentences.")
                test_sentences = self.cfg.test_sentences[name]
                results = {}
                results["audio"], results["audio_len"], speaker_audio, speaker_audio_lens = self.offline_inference_with_custom_sentences(test_sentences, self.cfg.inference_speaker_reference)
                with fp32_precision():  # resample is fragile to bfloat16 default dtype
                    metric_audio_pred = results["audio"]
                    metric_audio_pred_lens = results["audio_len"]

                    # resample audio to the asr sampling rate
                    metric_audio_pred = resample(metric_audio_pred, self.target_sample_rate, 16000)
                    metric_audio_pred_lens = (metric_audio_pred_lens / self.target_sample_rate * 16000).to(torch.long)

                    asr_hyps = self.asr_bleu.update(
                        name=name,
                        refs=test_sentences,
                        pred_audio=metric_audio_pred,
                        pred_audio_lens=metric_audio_pred_lens,
                    )

                    self.intelligibility.update(
                        name=name,
                        refs=test_sentences,
                        pred_audio=metric_audio_pred,
                        pred_audio_lens=metric_audio_pred_lens,
                        asr_hyps=asr_hyps,
                    )

                    self.secs.update(
                        name=name,
                        target_audio=resample(speaker_audio, self.target_sample_rate, 16000),
                        target_audio_lens=(speaker_audio_lens / self.target_sample_rate * 16000).to(torch.long),
                        pred_audio=resample(results["audio"], self.target_sample_rate, 16000),
                        pred_audio_lens=(results["audio_len"] / self.target_sample_rate * 16000).to(torch.long),
                    )

                    self.results_logger.update(
                        name=name,
                        refs=test_sentences,
                        hyps=test_sentences,
                        asr_hyps=asr_hyps,
                        samples_id=[str(i) for i in range(len(test_sentences))],
                        pred_audio=results["audio"].float(),
                        pred_audio_tf=None,
                        pre_audio_trimmed=None,
                        reference_audio=speaker_audio.float(),
                        target_audio=None,
                        pred_audio_sr=self.target_sample_rate,
                        user_audio=None,
                        user_audio_sr=None,
                        eou_pred=None,
                        fps=self.target_fps,
                        results=None,
                        tokenizer=self.tokenizer,
                    )

        else:
            for name, dataset_batch in batch.items():
                if dataset_batch is None:
                    continue  # some dataset is exhausted
                # run inference for multiples references
                if self.cfg.get("inference_speaker_reference_path", None):
                    B = len(dataset_batch['sample_id'])
                    for inference_speaker_reference in  glob.glob(os.path.join(self.cfg.inference_speaker_reference_path, "**"), recursive=True):
                        if not os.path.isfile(inference_speaker_reference):
                            continue
                        print("Generating sample for speaker refernce:", inference_speaker_reference)
                        new_dataset_batch = copy.deepcopy(dataset_batch)
                        # Get only the file name
                        ref_name = os.path.basename(inference_speaker_reference)
                        # Append to each sample_id
                        new_dataset_batch['sample_id'] = [
                            f"{sid}_{ref_name}" for sid in dataset_batch['sample_id']
                        ]
                        speaker_audio, sr = torchaudio.load(inference_speaker_reference)
                        speaker_audio = resample(speaker_audio, sr, self.target_sample_rate)
                        speaker_audio = speaker_audio.repeat(B, 1).to(self.device) 
                        # lengths -> [B]
                        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long().repeat(B)
                        new_dataset_batch["speaker_reference_audio"] = speaker_audio
                        new_dataset_batch["speaker_reference_audio_lens"] = speaker_audio_lens
                        self.run_evaluation_one_batch(name, new_dataset_batch, use_dataloader_init=False)
                # run inference for a custom speaker reference
                elif self.cfg.get("inference_speaker_reference", None):
                    new_dataset_batch = copy.deepcopy(dataset_batch)
                    speaker_audio, sr = torchaudio.load(inference_speaker_reference)
                    speaker_audio = resample(speaker_audio, sr, self.target_sample_rate)
                    speaker_audio = speaker_audio.repeat(B, 1).to(self.device) 
                    # lengths -> [B]
                    speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long().repeat(B)
                    new_dataset_batch["speaker_reference_audio"] = speaker_audio
                    new_dataset_batch["speaker_reference_audio_lens"] = speaker_audio_lens
                    self.run_evaluation_one_batch(name, new_dataset_batch, use_dataloader_init=False)
                # run inference using dataloader speaker references
                else:
                    self.run_evaluation_one_batch(name, dataset_batch, use_dataloader_init=False)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def get_system_prompt(self, system_prompt=None, user_prompt=None):
        messages = []
        if system_prompt is None:
            system_prompt = (
                "You engage in conversation with the user. When delivering your response as speech, "
                "if the user provides a description such as emotions, scene details, "
                "or speaker style, you adjust your speaking style accordingly when delivering the response. "
                "However, this description should influence only the delivery of your response, not its content. "
                "Your response should remain independent of any stylistic instructions."
            )
        messages.append({"role": "system", "content": system_prompt})
        
        # ToDo: implement dataloading support for descriptions
        """for desc in example["descriptions"]:
            user_prompt = ""
            if random.random() > self.p_drop_description and desc:
                user_prompt += f"```\n{desc}\n```"
            if random.random() > self.p_drop_description:
                if user_prompt:
                    user_prompt += "\n\n"
                user_prompt += self.rng.choice(self.user_prompts)
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})
        """

        # given that descriptions are currently not supported, only added the user prompt
        if user_prompt is None:
            user_prompt = "Can you tell me something interesting?"
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})
        non_script_list = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).split(SCRIPT_PLACEHOLDER + self.tokenizer.eos_token)[:-1]

        input_ids = []
        for i, non_script in enumerate(non_script_list):
            desc_ids = self.tokenizer.text_to_ids(non_script)
            input_ids.extend(desc_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).view(1, -1)
        return input_ids

    def get_init_inputs(self, speaker_audio, speaker_audio_lens, system_prompt=None, user_prompt=None):
        # compute prompt audio size and slice it
        with fp32_precision():
            """ 
            # old pad that can add long silences in the end 
            prompt_audio_size = int(((self.data_cfg.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame) * self.target_samples_per_frame)
            B, T = speaker_audio.shape  # [batch, time]
            if T >= prompt_audio_size:
                # Just crop if longer
                prompt_audio = speaker_audio[:, :prompt_audio_size]
            else:
                # Repeat along time until we have enough, then crop
                repeat_factor = (prompt_audio_size + T - 1) // T # ceil division
                expanded = speaker_audio.repeat(1, repeat_factor)
                prompt_audio = expanded[:, :prompt_audio_size]
            """
            # compute the exact number of samples for the prompt duration
            prompt_audio_size = int(
                ((self.data_cfg.audio_prompt_duration * self.target_sample_rate)
                // self.target_samples_per_frame)
                * self.target_samples_per_frame
            )

            B, T = speaker_audio.shape
            device = speaker_audio.device
            dtype = speaker_audio.dtype

            # allocate result
            prompt_audio = torch.zeros(B, prompt_audio_size, device=device, dtype=dtype)

            # process each example independently
            for b in range(B):
                valid_len = min(speaker_audio_lens[b].item(), T)

                # handle empty
                if valid_len <= 0:
                    continue

                # valid (non-padded) segment
                valid_segment = speaker_audio[b, :valid_len]

                if valid_len >= prompt_audio_size:
                    # enough valid audio → crop from start (no silence)
                    prompt_audio[b] = valid_segment[:prompt_audio_size]
                else:
                    # too short → repeat and crop
                    repeat_factor = (prompt_audio_size + valid_len - 1) // valid_len  # ceil division
                    expanded = valid_segment.repeat(repeat_factor)
                    prompt_audio[b] = expanded[:prompt_audio_size]

        # add a silence in the end to smooth the transition between prompt and audio tokens
        prompt_audio[:, -int(self.target_samples_per_frame * 2):] = 0

        # get prompt audio size
        with fp32_precision():
            prompt_audio_text_pad_size = int(prompt_audio_size // self.target_samples_per_frame)
        
        # get description tokens
        desc_tokens_ids = self.get_system_prompt(system_prompt=system_prompt, user_prompt=user_prompt)

        # create a padding tensor
        prompt_audio_text_pad = torch.ones(prompt_audio_text_pad_size, device=self.device, dtype=desc_tokens_ids.dtype) * self.text_pad_id
        prompt_audio_text_pad[-1] = self.tokenizer.eos

        # Add eos to simulate the end of a turn as in EAR-TTS inference
        desc_tokens_ids = torch.cat([desc_tokens_ids.squeeze(), torch.tensor([self.tokenizer.eos], dtype=desc_tokens_ids.dtype, device=desc_tokens_ids.device)])
        # Add padding equivalent to the audio prompt size in number of tokens
        input_text_tokens = torch.cat([desc_tokens_ids.to(desc_tokens_ids.dtype), prompt_audio_text_pad.to(desc_tokens_ids.dtype)])

        # create pad audio for the description
        pad_size = desc_tokens_ids.size(-1) * self.target_samples_per_frame
        pad_audio = torch.zeros(pad_size, device=prompt_audio.device, dtype=prompt_audio.dtype).unsqueeze(0).repeat(prompt_audio.size(0), 1)

        # repeat to reaches the batch size
        input_text_tokens = input_text_tokens.unsqueeze(0).repeat(prompt_audio.size(0), 1)
        target_audio = torch.cat([pad_audio, prompt_audio], dim=1)

        # extract code codes
        target_audio_len = torch.tensor([target_audio.size(-1)] * target_audio.size(0), dtype=torch.long, device=self.device)
        with fp32_precision(), torch.no_grad():
            code, _ = self.audio_codec.encode(target_audio.unsqueeze(1), target_audio_len)

        # get context hidden
        if self.cfg.tts_config.context_hidden_size is not None:
            context_hidden_state = self.embed_tokens(input_text_tokens)
            if self.cfg.tts_config.get("use_subword_flag_emb", False):
                context_hidden_state = self.subword_flag_emb(context_hidden_state, input_text_tokens)
        else:
            context_hidden_state = None

        # create masks
        # non_prompt_mask is all zeros, because all processed is prompt
        non_prompt_mask = torch.zeros_like(input_text_tokens)
        non_prompt_mask[:, -2:] = 1 # set last valid prompt frame as 1 to allow the addition of BOS in the right place
        subword_mask = torch.zeros_like(input_text_tokens) # subword_mask is almost all zeros because on the warmup there is only the prompt
        subword_mask[:, -3:] = 1 # -3 because of the it start right after the first valid prompt token and it is shifted by 1
        # audio mask is all ones except for description
        audio_mask = torch.ones_like(input_text_tokens) 
        audio_mask[:, :desc_tokens_ids.size(-1)] = 0
        # desc mask is all zeros except the description
        desc_mask = torch.zeros_like(input_text_tokens)
        desc_mask[:, :desc_tokens_ids.size(-1)] = 1


        if not self.cfg.get("disable_speech_pad", False):
            # add special tokens on audio codes
            code = torch.where(
                desc_mask.unsqueeze(-1).bool(),                    # (B, T, 1) for broadcasting
                torch.full_like(code, self.speech_pad_id),  # fill with pad id
                code
            )

        # shift subword_ids
        # subword_ids = F.pad(input_text_tokens[:, 1:], [0, 1], value=current_subword_id)
        subword_ids = F.pad(input_text_tokens[:, 1:], [0, 1], value=0.0)

        if self.cfg.get("ignore_audio_prompt_on_loss", False):
            # set audio_mask as non_prompt_mask to avoid the audio prompt in loss computation
            audio_mask = non_prompt_mask

        if self.cfg.get("add_pad_speech_token_in_last_prompt_frame", False) and not self.cfg.get("disable_speech_pad", False):
            # set special token in the last audio prompt (it will works as a BOS token)
            pos = non_prompt_mask.float().argmax(dim=1)  # shape: [B]
            row_idx = torch.arange(B, device=self.device)
            # set the extra self.speech_pad_id at first 1 position in non_prompt_mask
            code[row_idx, pos] = self.speech_pad_id

        init_inputs = {
            "code": code[:, :-1],
            "audio_mask": audio_mask.bool()[:, :-1],
            "context_hidden_state": context_hidden_state[:, :-1] if context_hidden_state is not None else None,
            "subword_ids": subword_ids[:, :-1],
            "subword_mask": subword_mask.bool()[:, :-1],
            "non_prompt_mask": non_prompt_mask.bool()[:, :-1]
        }

        return init_inputs

    @torch.no_grad()
    def offline_inference(
        self,
        next_subword_ids: torch.Tensor,
        speaker_audio: torch.Tensor,
        speaker_audio_lens: torch.Tensor,
        formatter: str = "",
        system_prompt: str = None,
        user_prompt: str = None,
        guidance_enabled: bool = True,
        generation_config: dict = None,
        init_inputs: dict = None,
        inp_asr_speech_tokens: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        B = next_subword_ids.size(0)

        # init_inputs, code, past_key_values = self.init_model_for_ar_inference(speaker_audio=speaker_audio, speaker_audio_lens=speaker_audio_lens, system_prompt=system_prompt, user_prompt=user_prompt, guidance_enabled=guidance_enabled, generation_config=generation_config)

        # ToDo: verify why codes differ from dataloader init_inputs when using nanocodec
        if init_inputs is None:
            init_inputs = self.get_init_inputs(speaker_audio, speaker_audio_lens, system_prompt=system_prompt, user_prompt=user_prompt)
        # compare_dicts(init_inputs_fn, init_inputs)

        if self.cfg.get("use_asr_speech_tokens", False) and self.cfg.get("only_semantic_to_speech", False):
            # set mask to zero and subword ids to self.text_pad_id as in training
            init_inputs["subword_mask"] = torch.full_like(init_inputs["subword_mask"], 0.0)
            init_inputs["subword_ids"] = torch.full_like(init_inputs["subword_ids"], self.text_pad_id)
            next_subword_ids = torch.full_like(next_subword_ids, self.text_pad_id)

        if generation_config is None:
            generation_config = self._get_generation_config(guidance_enabled)
            logging.info(f"Doing inference using the following config: {generation_config} !")

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})

        # warmup the model and generate the very first audio token
        outputs = self.tts_model(**init_inputs)

        if self.cfg.get("inference_skip_first_code_prediction_on_init", True):
            # use the last token on init, because we are shifthing it in the model forward, so we dont really need to compute it
            code = init_inputs["code"][:, -1:]
        else:
            code, _, _ = self.tts_model.generate_step(outputs.hidden_states[:, -1:], **generation_config)

        past_key_values = outputs["past_key_values"]

        # get current asr speech token
        if self.cfg.get("use_asr_speech_tokens", False):
            if self.cfg.get("only_semantic_to_speech", False):
                cur_asr_speech_tokens = inp_asr_speech_tokens[:, 0].unsqueeze(-1)
            else:    
                if guidance_enabled and self.cfg.get("asr_speech_tokens_use_guidance", True):
                    hidden_states, uncond_hidden_states = outputs.hidden_states.chunk(2, dim=0)
                    logits = self.asr_speech_tokens_head(hidden_states + (generation_config["guidance_scale"] * (hidden_states - uncond_hidden_states)))
                else:
                    hidden_states, _ = outputs.hidden_states.chunk(2, dim=0)
                    logits = self.asr_speech_tokens_head(hidden_states)

                cur_asr_speech_tokens = logits.argmax(dim=-1)[:, -1].unsqueeze(-1)

        # use the text tokens to stop generation
        max_steps = next_subword_ids.size(-1)
        # create variable to store the audios
        gen_audio_codes = torch.zeros(B, max_steps, self.tts_model.config.num_quantizers, device=self.device, dtype=torch.long)

        # init subwork as all ones
        subword_mask = torch.ones(B, max_steps, device=self.device, dtype=torch.bool)
        # get first context subword_id, that is the last subword_ids from the warmup
        first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)

        # reset cache of cumulative_word_emb
        if self.cfg.tts_config.get("use_cumulative_word_emb", False):
            self.tts_model.embed_subword.cumulative_word_emb.reset(B)

        for i in range(max_steps-1):
            step_start = time.time()
            # current subword id is always seem
            current_subword_id = next_subword_ids[:, i].unsqueeze(-1)

            if self.cfg.tts_config.context_hidden_size is not None:
                # get context_hidden_state it is always one step behind current_subword_id
                # for the first step uses the last step from warmup
                if i == 0:
                    context_subword_id = first_context_subword_id
                else:
                    context_subword_id = next_subword_ids[:, i-1].unsqueeze(-1)

                context_hidden_state = self.embed_tokens(context_subword_id)
                if self.cfg.tts_config.get("use_subword_flag_emb", False):
                    context_hidden_state = self.subword_flag_emb(context_hidden_state, context_subword_id)
            else:
                context_hidden_state = None

            # create subword_mask
            current_subword_mask = subword_mask[:, i].unsqueeze(-1)

            # get subword_ids
            inputs = {
                "code": code,
                "context_hidden_state": context_hidden_state,
                "subword_ids": current_subword_id,
                "subword_mask": current_subword_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
                "guidance_enabled": guidance_enabled,
                "generation_config": generation_config,
                "ignore_eos_flag_stop": True
            }

            outputs = self.tts_model(**inputs)

            code = outputs["codes"]
            past_key_values = outputs["past_key_values"]
            # ToDo: check why it is -1
            gen_audio_codes[:, i-1] = code.squeeze(1)

            if self.cfg.get("use_asr_speech_tokens", False) and not self.cfg.get("only_semantic_to_speech", False):
                if guidance_enabled and self.cfg.get("asr_speech_tokens_use_guidance", True):
                    hidden_states, uncond_hidden_states = outputs.hidden_states.chunk(2, dim=0)
                    logits = self.asr_speech_tokens_head(hidden_states + (generation_config["guidance_scale"] * (hidden_states - uncond_hidden_states)))
                else:
                    hidden_states, _ = outputs.hidden_states.chunk(2, dim=0)
                    logits = self.asr_speech_tokens_head(hidden_states)

                cur_asr_speech_tokens = logits.argmax(dim=-1)[:, -1].unsqueeze(-1)

            # force silence as next token 
            if self.cfg.get('inference_force_speech_silence_on_eos', None):
                silence_codes = self.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                code = torch.where(
                    current_subword_id.unsqueeze(-1) == self.text_eos_id,
                    silence_codes,  # silence
                    code,  # keep original
                )

            step_time = time.time()-step_start
            logging.info(f"Autoregressive inference step: {i} of {max_steps} take around {step_time}s")


        gen_audio_codes_lens = torch.tensor([gen_audio_codes.shape[1]] * gen_audio_codes.shape[0]).to(self.device)
        # decode audio. Note that it is not necessary because the prompt is removed, so no special token should be on the output, but lets do it for safety
        gen_audio_codes = replace_control_speech_codes(gen_audio_codes, self._control_codes, self.codec_silence_tokens)
        with fp32_precision(), torch.no_grad():
            audio_pred, audio_len = self.audio_codec.decode(
                gen_audio_codes, gen_audio_codes_lens
            )

        return audio_pred.squeeze(1), audio_len


    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_text_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.tts_model.backbone
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.tts_model.mog_head, self.tts_model.embed_subword, self.tts_model.embed_context, self.tts_model.embed_code, self.tts_model.null_emb, self.tts_model.bos_emb, self.tts_model.lm_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)

            for idx in range(self.tts_model._num_codebooks):
                self.tts_model.audio_embeddings[idx] = fully_shard(self.tts_model.audio_embeddings[idx], **fsdp_config)
                
            if self.tts_model.use_local_transformer:
                self.tts_model.local_transformer = fully_shard(self.tts_model.local_transformer, **fsdp_config)
                self.tts_model.local_transformer_in_projection = fully_shard(self.tts_model.local_transformer_in_projection, **fsdp_config)
            else:
                self.embed_text_tokens = fully_shard(self.embed_text_tokens, **fsdp_config)
                # self.tts_model = fully_shard(self.tts_model, **fsdp_config)
                self.tts_model.mog_head = fully_shard(self.tts_model.mog_head, **fsdp_config)
                self.tts_model.embed_subword = fully_shard(self.tts_model.embed_subword, **fsdp_config)
                self.tts_model.embed_context = fully_shard(self.tts_model.embed_context, **fsdp_config)
                self.tts_model.embed_code = fully_shard(self.tts_model.embed_code, **fsdp_config)
                self.tts_model.null_emb = fully_shard(self.tts_model.null_emb, **fsdp_config)
                self.tts_model.bos_emb = fully_shard(self.tts_model.bos_emb, **fsdp_config)
                self.tts_model.lm_head = fully_shard(self.tts_model.lm_head, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
