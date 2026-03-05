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
import copy
import os
import time
from collections import Counter
from contextlib import contextmanager

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
from peft import PeftModel
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

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.modules.ear_tts_model import RVQEARTTSModel
from nemo.collections.speechlm2.modules.ear_tts_vae_codec import RVQVAEModel
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_checkpoint,
    load_pretrained_hf,
    set_model_dict_for_partial_init,
)
from nemo.utils import logging


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
        self.trainer_config = cfg.get("trainer", None)
        self.data_cfg = cfg.data
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.normalize_text = cfg.data.get("normalize_text", False)

        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # Load ForCausalLM
        if self.cfg.tts_config.context_hidden_size is not None:
            self.language_model = self._load_language_model(self.cfg)
            self.embed_tokens = self._load_embed_tokens(self.cfg)
            # delete llm because we use it only to get the  embbeding tokens
            del self.language_model

        # get codec run precision
        self.audio_codec_run_dtype = getattr(torch, self.cfg.get("audio_codec_run_dtype", "float32"), torch.float32)

        # Load tokenizer
        self.tokenizer = AutoTokenizer(
            self.cfg.pretrained_lm_name,
            use_fast=True,
            trust_remote_code=True,
            bos_token=self.cfg.get("bos_token", None),
            eos_token=self.cfg.get("eos_token", None),
            pad_token=self.cfg.get("pad_token", None),
        )  # Note that we are using fast tokenizer

        # Instantiate TTS model
        self.tts_model = RVQEARTTSModel(DictConfig(self.cfg.tts_config), tokenizer=self.tokenizer)
        # Load and initialize audio codec, and bind RVQ embeddings to the TTS model
        setup_audio_codec(self)

        self._codebook_size = self.tts_model.config.codebook_size

        # compute samples per frame
        self.source_samples_per_frame = int(self.source_sample_rate * cfg.data.frame_length)

        # get codec silence tokens
        codec_silence_tokens = self.get_codec_silence_frame()
        self.register_buffer("codec_silence_tokens", codec_silence_tokens)

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_pad_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False

    def get_codec_silence_frame_last_one(self):
        audio = torch.zeros(1, 10 * self.target_sample_rate).float().to(self.device)
        audio_len = torch.tensor([audio.size(-1)]).long()
        audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.target_samples_per_frame)

        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            sil_codes, sil_codes_lens = self.audio_codec.encode(audio.unsqueeze(1), audio_len)
            return sil_codes[0, -1]

    def get_codec_silence_frame(self):

        # Generate long zero waveform (silence)
        audio = torch.zeros(1, 10 * self.target_sample_rate).float().to(self.device)
        audio_len = torch.tensor([audio.size(-1)]).long()
        audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.target_samples_per_frame)

        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
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
            embed_tokens_state_dict = torch.load(cfg.pretrained_lm_embedding_path, map_location="cpu")

            # Create token embedding layer
            vocab_size, hidden_size = embed_tokens_state_dict["weight"].size()
            embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
            embed_tokens.load_state_dict(embed_tokens_state_dict)
        return embed_tokens

    def _load_language_model(self, cfg):
        """Load language model for RVQ-EAR-TTS."""
        if cfg.pretrained_lm_name:
            language_model = load_pretrained_hf(
                self.cfg.pretrained_lm_name, pretrained_weights=True, trust_remote_code=True
            ).eval()
        else:
            language_model = None
        return language_model

    def restore_from_pretrained_checkpoint(self, checkpoint_path):
        """
        Loads model weights a pretrained checkpoint file, supporting partial loading from safetensor and PyTorch formats.

        Args:
            checkpoint_path (str): Path to checkpoint file.

        Returns:
            None. The model is updated in-place.
        """
        if checkpoint_path is not None:
            checkpoint_state = load_checkpoint(checkpoint_path)
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())

            if self.cfg.get("rescale_pretrained_weights", None):
                checkpoint_state = rescale_state_dict(
                    checkpoint_state, first_n_layers=self.cfg.get("rescale_first_n_layers", None)
                )

            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        if self.use_local_transformer and self.local_transformer_type == "nar":  # add extra token for mask
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
        Prepare inputs, extracting audio tokens and padding if needed.
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["audio_prompt"].size(0) == batch["target_audio"].size(0)

        target_audio = batch["target_audio"]
        target_audio_lens = batch["target_audio_lens"]
        target_text_tokens = batch["target_text_tokens"]
        non_prompt_mask = batch["non_prompt_mask"]
        aligned_attention_mask = batch["aligned_attention_mask"]
        aligned_position_ids = batch["aligned_position_ids"]

        if self.training and (self.cfg.get("empty_turn_probability", 0.0) > 0):
            # Randomly decide whether this batch gets emptied
            if torch.rand(1).item() < self.cfg.empty_turn_probability:
                # Zero out audio
                target_audio = torch.zeros_like(target_audio)

                # Create mask for tokens we want to drop
                # Keep BOS and EOS, drop the rest.
                keep_mask = (target_text_tokens == self.text_bos_id) | (target_text_tokens == self.text_eos_id)
                full_dropout_mask = ~keep_mask  # True = positions to replace with PAD

                # Replace all non-BOS/EOS with PAD
                target_text_tokens = torch.where(
                    full_dropout_mask, torch.full_like(target_text_tokens, self.text_pad_id), target_text_tokens
                )

        # extract target audio codes
        target_audio, target_audio_lens = self.pad_audio_to_factor(
            target_audio, target_audio_lens, self.target_samples_per_frame, 1
        )
        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(target_audio.unsqueeze(1), target_audio_lens)

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

            target_text_tokens = pad_or_truncate(target_text_tokens, pad_value=self.text_pad_id)
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

        # set the pad token for the first BOS frame
        target_codes_aligned = target_codes.clone()
        target_codes_aligned[:, 0] = self.speech_pad_id

        # set special token in the last audio prompt (it will works as a BOS token)
        pos = non_prompt_mask.float().argmax(dim=1)  # shape: [B]
        row_idx = torch.arange(B, device=self.device)
        # set the extra self.speech_pad_id at first 1 position in non_prompt_mask
        target_codes_aligned[row_idx, pos] = self.speech_pad_id

        # EOS dropout to make the model more robust
        if self.training and self.cfg.get("text_eos_dropout_prob", 0.0) > 0:
            # Mask EOS positions
            eos_mask = target_text_tokens == self.text_eos_id

            # Random dropout only on EOS positions
            dropout_mask = (
                torch.rand(eos_mask.sum(), device=target_text_tokens.device) < self.cfg.text_eos_dropout_prob
            )

            # Scatter dropout decisions into [B, T]
            full_dropout_mask = torch.zeros_like(target_text_tokens, dtype=torch.bool)
            full_dropout_mask[eos_mask] = dropout_mask

            # Replace dropped EOS with PAD
            target_text_tokens = torch.where(
                full_dropout_mask, torch.full_like(target_text_tokens, self.text_pad_id), target_text_tokens
            )

        if self.training and self.cfg.get("text_eos_duplicate_prob", 0.0) > 0:
            p = self.cfg.text_eos_duplicate_prob

            # [B, T] mask of EOS positions
            eos_mask = target_text_tokens == self.text_eos_id

            # Flatten EOS positions: tensor of shape [N, 2] where each row = (batch_idx, time_idx)
            eos_positions = eos_mask.nonzero(as_tuple=False)  # [N, 2]

            if eos_positions.numel() > 0:
                N = eos_positions.shape[0]

                # One random decision per EOS occurrence
                duplicate_decision = torch.rand(N, device=target_text_tokens.device) < p  # [N]

                # Filter only EOS tokens that will be duplicated and are not at position t=0
                valid = (eos_positions[:, 1] > 0) & duplicate_decision  # [N]

                if valid.any():
                    # Select only valid EOS positions
                    valid_positions = eos_positions[valid]  # [M, 2]

                    # Indices for the token BEFORE the EOS (t-1)
                    b_idx = valid_positions[:, 0]
                    t_idx = valid_positions[:, 1] - 1

                    # Replace token before EOS with an EOS
                    target_text_tokens[b_idx, t_idx] = self.text_eos_id

        # BOS dropout to make the model more robust
        if self.training and self.cfg.get("text_bos_dropout_prob", 0.0) > 0:
            # Mask BOS positions
            bos_mask = target_text_tokens == self.text_bos_id

            # Random dropout only on BOS positions
            dropout_mask = (
                torch.rand(bos_mask.sum(), device=target_text_tokens.device) < self.cfg.text_bos_dropout_prob
            )

            # Scatter dropout decisions into [B, T]
            full_dropout_mask = torch.zeros_like(target_text_tokens, dtype=torch.bool)
            full_dropout_mask[bos_mask] = dropout_mask

            # Replace dropped BOS with PAD
            target_text_tokens = torch.where(
                full_dropout_mask,
                torch.full_like(target_text_tokens, self.text_pad_id),
                target_text_tokens,
            )

        # shift text tokens
        subword_ids = F.pad(target_text_tokens[:, 1:], [0, 1])
        # note that we are using a text mask where we are ignoring the desc + audio prompt but we are keeping 1 until the audio ends to support duplex
        subword_mask = F.pad(non_prompt_mask[:, 1:], [0, 1])

        # detach embedding as in eartts
        if self.cfg.tts_config.context_hidden_size is not None:
            context_hidden_state = self.embed_tokens(target_text_tokens).detach()
        else:
            context_hidden_state = None

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (target_text_tokens.shape[1] - 1) % tp_world_size) != 0:
                target_text_tokens = target_text_tokens[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                subword_ids = subword_ids[:, :-remainder]
                subword_mask = subword_mask[:, :-remainder]

        return {
            "code": target_codes_aligned,
            "audio_mask": non_prompt_mask,  # set audio_mask as non_prompt_mask to avoid the audio prompt in loss computation
            "attention_mask": aligned_attention_mask,
            "position_ids": aligned_position_ids,
            "subword_ids": subword_ids,
            "subword_mask": subword_mask,
            "context_hidden_state": context_hidden_state,
            "output_lens": target_codes_lens,
            "non_prompt_mask": non_prompt_mask,
            "target_text_tokens": target_text_tokens,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.tts_model,):
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
            non_prompt_mask=inputs["non_prompt_mask"],
        )
        loss_dict = {"lm_loss": tts_output.lm_loss, "c_loss": tts_output.c_loss, "k_loss": tts_output.k_loss}
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
        ensures_codec_target_dtype(
            self
        )  # potentially reloads the audio codec to make sure it's in target codec precision

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
            w = p.detach().cpu().float()  # safe offline copy
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
        weight_l2 = (total_w_sq**0.5) if total_w_sq > 0 else 0.0

        # RMS (global)
        weight_rms = ((total_w_sq / total_w_params) ** 0.5) if total_w_params > 0 else 0.0

        # Mean
        weight_mean = sum_w / total_w_params if total_w_params > 0 else 0.0

        # direct float logging avoids device sync penalty
        self.log("weights/L2", weight_l2, on_epoch=True, sync_dist=True)
        self.log("weights/RMS", weight_rms, on_epoch=True, sync_dist=True)
        self.log("weights/max_abs", max_abs_w, on_epoch=True, sync_dist=True)
        self.log("weights/mean", weight_mean, on_epoch=True, sync_dist=True)

    def on_validation_epoch_start(self) -> None:
        ensures_codec_target_dtype(
            self
        )  # potentially reloads the audio codec to make sure it's in target codec precision

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
            guidance_enabled=guidance_enabled,
        )
        tf_audio_codes_pred = tts_output["codes"].squeeze(2)

        # decode audio
        tf_audio_codes_pred = replace_control_speech_codes(
            tf_audio_codes_pred, self._control_codes, self.codec_silence_tokens
        )
        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            audio_pred, audio_len = self.audio_codec.decode(tf_audio_codes_pred, inputs["output_lens"])

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

    def run_evaluation_one_batch(self, name, dataset_batch, use_dataloader_init=False):
        """
        Runs evaluation and scoring for a single data batch, logging metrics and updating result buffers.

        Args:
            name (str): Name/id for the batch (for logging).
            dataset_batch (dict): Batch of data inputs, supports batched text/audio/etc.
            use_dataloader_init (bool, optional): If True, use dataloader initialization for prompts.

        Returns:
            None. Outputs are logged and stored in result buffers.
        """
        results = {}
        inputs = self.prepare_inputs(dataset_batch)

        results["audio_tf"], results["audio_tf_len"] = self.get_teacher_force_inference_audio(dataset_batch)
        if use_dataloader_init:
            # cut it on prompt
            init_inputs = {
                "code": inputs["code"],
                "audio_mask": inputs["audio_mask"],
                "non_prompt_mask": inputs["non_prompt_mask"],
                "context_hidden_state": inputs["context_hidden_state"],
                "subword_ids": inputs["subword_ids"],
                "subword_mask": inputs["subword_mask"],
            }
            # cut init_inputs to consider only the prompt
            for key in init_inputs:
                if init_inputs[key] is not None:
                    init_inputs[key] = torch.stack(
                        [init_inputs[key][i, :plen] for i, plen in enumerate(dataset_batch["prompt_lens"])]
                    )
        else:
            # set init inputs and get it
            self.set_init_inputs(
                speaker_audio=dataset_batch["audio_prompt"],
                speaker_audio_lens=dataset_batch["audio_prompt_lens"],
            )
            init_inputs = self.get_init_inputs(B=inputs["subword_ids"].size(0))

        # remove the prompt from the target_text_tokens to emulate S2S connected inference
        next_subword_ids = torch.stack(
            [
                inputs["subword_ids"][i, plen:]  # slice each element
                for i, plen in enumerate(dataset_batch["prompt_lens"])
            ]
        )

        results["audio"], results["audio_len"] = self.offline_inference(
            next_subword_ids=next_subword_ids,
            task=dataset_batch["task"][0],
            init_inputs=init_inputs,
        )

        # remove prompt padding from the user audio as autoregressive inference does not return the prompt
        dataset_batch["source_audio"] = dataset_batch["source_audio"][
            :, -int(next_subword_ids.size(-1) * self.source_samples_per_frame) :
        ]

        # clean prompt from the audio
        results["audio_tf"] = results["audio_tf"][:, -int(next_subword_ids.size(-1) * self.target_samples_per_frame) :]
        # remove prompt from target audio
        target_audio_no_prompt = dataset_batch["target_audio"][
            :, -int(next_subword_ids.size(-1) * self.target_samples_per_frame) :
        ]
        target_audio_no_prompt_lens = dataset_batch["target_audio_lens"] - (
            torch.tensor(
                dataset_batch["prompt_lens"],
                dtype=torch.long,
                device=dataset_batch["target_audio_lens"].device,
            )
            * self.target_samples_per_frame
        )

        with fp32_precision():  # resample is fragile to bfloat16 default dtype
            metric_audio_pred = results["audio"]
            metric_audio_pred_lens = results["audio_len"]

            # resample audio to the asr sampling rate
            metric_audio_pred = resample(metric_audio_pred, self.target_sample_rate, 16000)
            metric_audio_pred_lens = (metric_audio_pred_lens / self.target_sample_rate * 16000).to(torch.long)
            # reshape target audio without prompt
            target_audio_no_prompt_16khz = resample(target_audio_no_prompt, self.target_sample_rate, 16000)
            target_audio_no_prompt_lens_16khz = (target_audio_no_prompt_lens / self.target_sample_rate * 16000).to(
                torch.long
            )
            if self.cfg.get("use_GT_transcriptions_for_metrics", True):
                # use target audio transcription for metrics
                target_asr_texts = self.asr_bleu.asr.transcribe(
                    [
                        audio[:alen]
                        for audio, alen in zip(target_audio_no_prompt_16khz, target_audio_no_prompt_lens_16khz)
                    ],
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
                name=name + "_gt",
                refs=dataset_batch["target_texts"],
                pred_audio=target_audio_no_prompt_16khz,
                pred_audio_lens=target_audio_no_prompt_lens_16khz,
                asr_hyps=(
                    metric_text if self.cfg.get("use_GT_transcriptions_for_metrics", True) else None
                ),  # reuse GT transcription
            )

            self.secs.update(
                name=name,
                target_audio=resample(dataset_batch["target_audio"], self.target_sample_rate, 16000),
                target_audio_lens=(dataset_batch["target_audio_lens"] / self.target_sample_rate * 16000).to(
                    torch.long
                ),
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
                reference_audio=dataset_batch["audio_prompt"].float(),
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
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            B = len(dataset_batch['sample_id'])

            # run inference for a custom speaker reference
            if self.cfg.get("inference_speaker_reference", None):
                new_dataset_batch = copy.deepcopy(dataset_batch)
                speaker_audio, sr = load_audio_librosa(self.cfg.inference_speaker_reference)
                speaker_audio = resample(speaker_audio, sr, self.target_sample_rate)
                speaker_audio = speaker_audio.repeat(B, 1).to(self.device)
                # lengths -> [B]
                speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long().repeat(B)
                new_dataset_batch["audio_prompt"] = speaker_audio
                new_dataset_batch["audio_prompt_lens"] = speaker_audio_lens
                self.run_evaluation_one_batch(name, new_dataset_batch)

            # run inference using dataloader speaker references
            else:
                self.run_evaluation_one_batch(name, dataset_batch, use_dataloader_init=False)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def set_init_inputs(self, speaker_audio, speaker_audio_lens, system_prompt=None, user_prompt=None):
        """
        Registers and prepares initial input buffers for text/audio prompt and context, to warm up AR inference.

        Args:
            speaker_audio (torch.Tensor): Batch of prompt audio, (B, T).
            speaker_audio_lens (torch.Tensor): Lengths for each sample in speaker_audio, (B,).
            system_prompt (str, optional): System prompt for context.
            user_prompt (str, optional): User message for context.

        Returns:
            dict: Dictionary of input tensors to be passed to inference, with registered buffers.
        """
        # compute prompt audio size and slice it
        with fp32_precision():
            # compute the exact number of samples for the prompt duration
            prompt_audio_size = int(
                ((self.data_cfg.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame)
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
        prompt_audio[:, -int(self.target_samples_per_frame * 2) :] = 0

        # get prompt audio size
        with fp32_precision():
            prompt_audio_text_pad_size = int(prompt_audio_size // self.target_samples_per_frame)

        # create a eos token id
        first_text_frame = torch.tensor([self.tokenizer.eos], dtype=torch.long, device=self.device)

        # create a padding tensor
        prompt_audio_text_pad = (
            torch.ones(prompt_audio_text_pad_size, device=self.device, dtype=first_text_frame.dtype) * self.text_pad_id
        )
        prompt_audio_text_pad[-1] = self.tokenizer.eos

        # Prepend an initial text EOS token followed by padding tokens that match
        # the number of audio-prompt frames (in text-token units).
        target_text_tokens = torch.cat([first_text_frame, prompt_audio_text_pad.to(first_text_frame.dtype)])

        # create pad audio for the description
        pad_size = first_text_frame.size(-1) * self.target_samples_per_frame
        pad_audio = (
            torch.zeros(pad_size, device=prompt_audio.device, dtype=prompt_audio.dtype)
            .unsqueeze(0)
            .repeat(prompt_audio.size(0), 1)
        )

        # repeat to reaches the batch size
        target_text_tokens = target_text_tokens.unsqueeze(0).repeat(prompt_audio.size(0), 1)
        target_audio = torch.cat([pad_audio, prompt_audio], dim=1)

        # extract code codes
        target_audio_len = torch.tensor(
            [target_audio.size(-1)] * target_audio.size(0), dtype=torch.long, device=self.device
        )
        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            code, _ = self.audio_codec.encode(target_audio.unsqueeze(1), target_audio_len)

        # get context hidden
        if self.cfg.tts_config.context_hidden_size is not None:
            context_hidden_state = self.embed_tokens(target_text_tokens)
        else:
            context_hidden_state = None

        # create masks
        # non_prompt_mask is all zeros, because all processed is prompt
        non_prompt_mask = torch.zeros_like(target_text_tokens)
        non_prompt_mask[:, -2:] = 1  # set last valid prompt frame as 1 to allow the addition of BOS in the right place
        subword_mask = torch.zeros_like(
            target_text_tokens
        )  # subword_mask is almost all zeros because on the warmup there is only the prompt
        subword_mask[:, -3:] = (
            1  # -3 because of the it start right after the first valid prompt token and it is shifted by 1
        )

        # set the pad token for the first BOS frame
        code[:, 0] = self.speech_pad_id

        # shift subword_ids
        subword_ids = F.pad(target_text_tokens[:, 1:], [0, 1], value=0.0)

        # set special token in the last audio prompt (it will works as a BOS token)
        pos = non_prompt_mask.float().argmax(dim=1)  # shape: [B]
        row_idx = torch.arange(B, device=self.device)
        # set the extra self.speech_pad_id at first 1 position in non_prompt_mask
        code[row_idx, pos] = self.speech_pad_id

        init_inputs = {
            "code": code[:, :-1],
            "audio_mask": non_prompt_mask.bool()[
                :, :-1
            ],  # set audio_mask as non_prompt_mask to avoid the audio prompt in loss computation
            "context_hidden_state": context_hidden_state[:, :-1] if context_hidden_state is not None else None,
            "subword_ids": subword_ids[:, :-1],
            "subword_mask": subword_mask.bool()[:, :-1],
            "non_prompt_mask": non_prompt_mask.bool()[:, :-1],
        }
        # register to acess later
        for k, v in init_inputs.items():
            name = f"init_input_{k}"
            if v is not None:
                self.register_buffer(name, v)

        return init_inputs

    def get_init_inputs(
        self,
        B: int,
        init_inputs_names=[
            "code",
            "audio_mask",
            "context_hidden_state",
            "subword_ids",
            "subword_mask",
            "non_prompt_mask",
        ],
    ):
        """
        Returns a dictionary of initial inputs for inference, using registered buffers.

        Args:
            B (int): Required batch size.
            init_inputs_names (List[str], optional): Names of input buffers to fetch.

        Returns:
            dict: Each key is name from init_inputs_names, and value is tensor of appropriate shape (B, ...).

        Notes:
            Expands batch-1 buffers to B if necessary.
        """
        if init_inputs_names is None:
            init_inputs_names = [
                "code",
                "audio_mask",
                "context_hidden_state",
                "subword_ids",
                "subword_mask",
                "non_prompt_mask",
            ]

        init_inputs = {}
        for name in init_inputs_names:
            buf_name = f"init_input_{name}"
            buf = getattr(self, buf_name, None)

            if buf is None:
                init_inputs[name] = None
                continue

            # Use as-is if batch matches
            if buf.shape[0] == B:
                init_inputs[name] = buf
            else:
                # Otherwise, assume batch=1 and expand to target B
                init_inputs[name] = buf[:1].expand(B, *buf.shape[1:])

        return init_inputs

    @torch.no_grad()
    def infer_codes_one_step(
        self,
        current_subword_id,
        prev_subword_id,
        current_subword_mask,
        prev_audio_tokens,
        past_key_values,
        guidance_enabled=True,
        generation_config=None,
        ignore_eos_flag_stop=True,
    ):
        """
        Runs a single autoregressive prediction step to infer audio codec codes.

        Args:
            current_subword_id (torch.Tensor): Current text token IDs, shape (B, 1).
            prev_subword_id (torch.Tensor): Previous text token IDs, shape (B, 1).
            current_subword_mask (torch.Tensor): Current mask, shape (B, 1).
            prev_audio_tokens (torch.Tensor): Previously generated audio tokens, shape (B, 1, C).
            past_key_values: Key-value cache for transformer decoder state.
            guidance_enabled (bool, optional): Enables classifier-free guidance.
            generation_config (dict, optional): Generation hyperparameters.
            ignore_eos_flag_stop (bool): If True, ignore EOS flag for stopping.

        Returns:
            Tuple[torch.Tensor, Any]:
                - Predicted audio codec token(s), shape (B, 1, C)
                - Updated past_key_values for the next step.
        """

        if self.cfg.tts_config.context_hidden_size is not None:
            # get context_hidden_state it is always one step behind current_subword_id
            # for the first step uses the last step from warmup
            context_hidden_state = self.embed_tokens(prev_subword_id)
        else:
            context_hidden_state = None

        # force silence as next token
        if self.cfg.get('inference_force_speech_silence_on_eos', True):
            silence_codes = self.codec_silence_tokens.view(1, 1, -1).expand(prev_audio_tokens.shape)
            prev_audio_tokens = torch.where(
                current_subword_id.unsqueeze(-1) == self.text_eos_id,
                silence_codes,  # silence
                prev_audio_tokens,  # keep original
            )

        # get subword_ids
        inputs = {
            "code": prev_audio_tokens,
            "context_hidden_state": context_hidden_state,
            "subword_ids": current_subword_id,
            "subword_mask": current_subword_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "guidance_enabled": guidance_enabled,
            "generation_config": generation_config,
            "ignore_eos_flag_stop": ignore_eos_flag_stop,
        }

        outputs = self.tts_model(**inputs)

        return outputs["codes"], outputs["past_key_values"]

    @torch.no_grad()
    def decode_one_audio_step(self, gen_audio_codes_history, number_prev_tokens=None):
        """
        Decodes one step of generated audio codec tokens to raw waveform.

        Args:
            gen_audio_codes_history (torch.Tensor): Audio tokens history, shape (B, T, C).
            number_prev_tokens (int, optional): Number of previous tokens to decode, for incremental decoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - audio_pred_cur_step: Latest decoded waveform chunk, shape (B, wav_to_token_ratio).
                - audio_len: Lengths (number of samples), shape (B,).
        """
        with fp32_precision(), torch.no_grad():
            if number_prev_tokens:
                gen_audio_codes_history = gen_audio_codes_history[:, -number_prev_tokens:]

            gen_audio_codes_history = replace_control_speech_codes(
                gen_audio_codes_history, self._control_codes, self.codec_silence_tokens
            )
            gen_audio_codes_lens = torch.tensor(
                [gen_audio_codes_history.size(1)] * gen_audio_codes_history.size(0), device=self.device
            )
            audio_pred, audio_len = self.audio_codec.decode(gen_audio_codes_history, gen_audio_codes_lens)

        # return only the current/lastest audio chunk
        audio_pred_cur_step = audio_pred.squeeze(1)[:, -self.audio_codec.config.wav_to_token_ratio :]
        audio_len[:] = self.audio_codec.config.wav_to_token_ratio
        return audio_pred_cur_step, audio_len

    @torch.no_grad()
    def offline_inference(
        self,
        next_subword_ids: torch.Tensor,
        init_inputs: dict,
        task: str = "",
        guidance_enabled: bool = True,
        generation_config: dict = None,
        incremental_audio_decoding: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Runs offline autoregressive inference for the Duplex EAR-TTS speech decoder.

        This method performs **text-to-speech (TTS)** generation: given subword/text
        tokens and prompt-initialization states, it autoregressively generates
        audio codec tokens and decodes them into a waveform.

        Args:
            next_subword_ids (torch.Tensor):
                Conditioning subword/text token IDs for the speech decoder.
                Shape: (B, T_text).

            init_inputs (dict):
                Dictionary of prompt-dependent initial states produced by
                ``get_init_inputs()``. May include:

                    • "code"                 — initial audio tokens (e.g., prompt audio)
                    • "audio_mask"           — mask for prompt audio positions
                    • "context_hidden_state" — decoder hidden state at t = 0
                    • "subword_ids"          — prompt text tokens
                    • "subword_mask"         — mask for prompt text
                    • "non_prompt_mask"      — mask marking positions to be generated

                ``get_init_inputs()`` automatically expands batch-1 buffers to
                batch size B.

            task (str, optional):
                Optional task identifier used to customize the prompt structure.

            guidance_enabled (bool, optional):
                Whether classifier-free guidance (CFG) is enabled.
                If enabled and ``generation_config`` is ``None``, guidance parameters
                are taken from ``_get_generation_config()``.

            generation_config (dict, optional):
                Settings controlling autoregressive generation, including sampling
                strategy, noise scale, refinement iterations, and EOS rules.
                If ``None``, defaults are taken from
                ``_get_generation_config(guidance_enabled)``.

            incremental_audio_decoding (bool, optional):
                If True, codec-to-waveform decoding is performed incrementally during
                autoregressive generation.
                If False, waveform decoding occurs only after all audio tokens are produced.

        Returns:
            dict[str, torch.Tensor]:
                Contains:

                • **"audio"**:
                Generated waveform of shape ``(B, T_audio)``, obtained via
                ``audio_pred.squeeze(1)``.

                • **"audio_len"**:
                Length of each generated waveform in samples, shape ``(B,)``.
        """
        B = next_subword_ids.size(0)

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

        # use the text tokens to stop generation
        max_steps = next_subword_ids.size(-1)
        # create variable to store the audios
        gen_audio_codes = torch.zeros(
            B, max_steps, self.tts_model.config.num_quantizers, device=self.device, dtype=torch.long
        )

        # init subwork as all ones
        subword_mask = torch.ones(B, max_steps, device=self.device, dtype=torch.bool)
        # get first context subword_id, that is the last subword_ids from the warmup
        first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)

        # initialize variables used to save the output audio
        audio_pred = None
        audio_pred_len = torch.zeros(B, device=self.device, dtype=torch.long)

        for i in range(max_steps):
            step_start = time.time()
            # current subword id is always seem
            current_subword_id = next_subword_ids[:, i].unsqueeze(-1)

            if i == 0:
                prev_subword_id = first_context_subword_id
            else:
                prev_subword_id = next_subword_ids[:, i - 1].unsqueeze(-1)

            # create subword_mask
            current_subword_mask = subword_mask[:, i].unsqueeze(-1)

            code, past_key_values = self.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=guidance_enabled,
                generation_config=generation_config,
                ignore_eos_flag_stop=True,
            )

            # cache audio tokens
            gen_audio_codes[:, i] = code.squeeze(1)

            if incremental_audio_decoding:
                audio_pred_i, audio_pred_i_len = self.decode_one_audio_step(
                    gen_audio_codes[:, : i + 1],
                    number_prev_tokens=self.cfg.get("inference_codec_decoding_prev_tokens_number", None),
                )
                if audio_pred is None:
                    audio_pred = audio_pred_i
                else:
                    audio_pred = torch.cat([audio_pred, audio_pred_i], dim=1)
                audio_pred_len += audio_pred_i_len

            step_time = time.time() - step_start
            logging.info(f"Autoregressive inference step: {i} of {max_steps} take around {step_time}s")

        if not incremental_audio_decoding:
            gen_audio_codes_lens = torch.tensor([gen_audio_codes.shape[1]] * gen_audio_codes.shape[0]).to(self.device)
            # decode audio. Note that it is not necessary because the prompt is removed, so no special token should be on the output, but lets do it for safety
            gen_audio_codes = replace_control_speech_codes(
                gen_audio_codes, self._control_codes, self.codec_silence_tokens
            )
            with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
                audio_pred, audio_pred_len = self.audio_codec.decode(gen_audio_codes, gen_audio_codes_lens)

        return audio_pred.squeeze(1), audio_pred_len

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
        raise NotImplementedError

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

            for m in (
                self.tts_model.mog_head,
                self.tts_model.embed_subword,
                self.tts_model.embed_context,
                self.tts_model.embed_code,
                self.tts_model.null_emb,
                self.tts_model.bos_emb,
                self.tts_model.lm_head,
            ):
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
                self.tts_model.local_transformer_in_projection = fully_shard(
                    self.tts_model.local_transformer_in_projection, **fsdp_config
                )

            self.embed_text_tokens = fully_shard(self.embed_text_tokens, **fsdp_config)
            self.tts_model.mog_head = fully_shard(self.tts_model.mog_head, **fsdp_config)
            self.tts_model.embed_subword = fully_shard(self.tts_model.embed_subword, **fsdp_config)
            self.tts_model.embed_context = fully_shard(self.tts_model.embed_context, **fsdp_config)
            self.tts_model.embed_code = fully_shard(self.tts_model.embed_code, **fsdp_config)
            self.tts_model.lm_head = fully_shard(self.tts_model.lm_head, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.tts_model = fully_shard(self.tts_model, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info("Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)


def load_audio_librosa(path, sr=None):
    """
    Load audio using librosa with torchaudio-like behavior.

    Returns:
        audio_tensor: torch.FloatTensor of shape [channels, time]
        sr: sampling rate
    """
    # Load with librosa (preserve original sampling rate)
    audio, sr = librosa.load(path, sr=sr, mono=False)

    # Ensure shape is [channels, time]
    if audio.ndim == 1:
        # Mono: (time,) -> (1, time)
        audio = audio[None, :]

    # Convert to torch float32 (torchaudio behavior)
    audio_tensor = torch.from_numpy(audio).float()
    return audio_tensor, sr


def maybe_to(x, dtype):
    if x is None:
        return None
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype)
    return x


@contextmanager
def ensures_target_precision(target_dtype):
    """
    Workaround for precision related issues when training with bf16-true PyTorch Lightning precision setting.
    In bf16-true, PTL changes PyTorch's default dtype, which may break implicit assumptions for some models.
    This context manager restores default float32 precision and runs the computation in float32 autocast context.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=target_dtype):
            yield
    finally:
        torch.set_default_dtype(default_dtype)


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
    device = input_ids.device
    bos_mask = (input_ids == bos_token_id).to(torch.int32).to(device)
    eos_mask = (input_ids == eos_token_id).to(torch.int32).to(device)
    bos_cumsum = torch.cumsum(bos_mask, dim=1)
    eos_cumsum = torch.cumsum(eos_mask, dim=1)
    speaking_mask = (bos_cumsum > eos_cumsum).to(torch.float32)
    return speaking_mask.long()


def replace_control_speech_codes(
    speech_codes: torch.Tensor, control_codes: torch.Tensor, silence_tokens: torch.Tensor = None
) -> torch.Tensor:
    """
    Replaces control codes (speech BOS, EOS, etc) in `speech_codes` with the first frame which is
    assumed to consist of 'valid' codes representing silence.
    """
    if silence_tokens is not None:
        # Expand to [B, 1, 74]
        silence_tokens_expanded = silence_tokens.unsqueeze(0).unsqueeze(1).expand(speech_codes.shape[0], 1, -1)
        return torch.where(torch.isin(speech_codes, control_codes), silence_tokens_expanded, speech_codes)

    if torch.isin(speech_codes[:, :1], control_codes).any():
        return torch.where(
            torch.isin(speech_codes, control_codes), torch.zeros_like(speech_codes[:, :1]), speech_codes
        )
    else:
        return torch.where(torch.isin(speech_codes, control_codes), speech_codes[:, :1], speech_codes)


def ensures_codec_target_dtype(model):
    """
    Ensures the audio codec is instantiated with the target dtype.

    This function checks whether `model.audio_codec` exists and whether its
    parameters match `model.audio_codec_run_dtype`. If the codec is missing
    or is running with the wrong dtype (e.g., due to PTL auto-downcasting),
    the codec is reloaded by calling `setup_audio_codec()`.

    Intended to be called at runtime boundaries such as:
      - `on_train_epoch_start`
      - `on_validation_epoch_start`

    Args:
        model: Model instance of DuplexEARTTS

    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == model.audio_codec_run_dtype:
        return  # already correct precision → no-op

    setup_audio_codec(model)


def setup_audio_codec(model):
    """
    Instantiates the RVQ audio codec and injects codec embeddings into the TTS model.

    This function is responsible only for:
      - Instantiating the codec model (`RVQVAEModel`).
      - Loading pretrained codec weights (if configured).
      - Freezing codec parameters.
      - Registering RVQ embeddings inside the TTS model via `set_rvq_embs`.

    Args:
        model: Model instance of DuplexEARTTS
    """
    with ensures_target_precision(model.audio_codec_run_dtype):
        model.audio_codec = RVQVAEModel(DictConfig(model.cfg.codec_config))
        # load pretrained codec checkpoint
        if model.cfg.get("pretrained_codec_model", None):
            checkpoint_state = load_checkpoint(model.cfg.pretrained_codec_model)
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.audio_codec.state_dict())
            model.audio_codec.load_state_dict(checkpoint_state, strict=True)

    for p in model.audio_codec.parameters():
        p.requires_grad = False

    assert callable(model.tts_model.set_rvq_embs)

    model.tts_model.set_rvq_embs(torch.stack([x.detach() for x in model.audio_codec.prvq.mus_list], 0))
    model.tts_model.rvq_embs = model.tts_model.rvq_embs.to(next(model.tts_model.parameters()).dtype)
    # compute target fps
    model.target_fps = model.target_sample_rate / model.audio_codec.config.wav_to_token_ratio
    model.target_samples_per_frame = model.audio_codec.config.wav_to_token_ratio


def rescale_state_dict(state_dict, target_std=0.02, first_n_layers=None, layer_prefix="tts_model.backbone.layers."):
    """
    Rescale trainable weights in a state_dict for BF16/FP16 stability.

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
            logging.info(f"No weights found for first {first_n_layers} layers with prefix '{layer_prefix}'.")
        else:
            logging.info("No weights found to rescale in state_dict.")
        return state_dict

    # Compute global std across selected weights (on CPU)
    cpu_weights = [p.detach().cpu() for p in weight_tensors]
    flat = torch.cat([p.flatten() for p in cpu_weights])
    current_std = float(torch.std(flat))
    scale = target_std / (current_std + 1e-8)

    logging.info(
        f"Rescaling state_dict "
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

    logging.info("Done: weights rescaled.")
    return new_state_dict
