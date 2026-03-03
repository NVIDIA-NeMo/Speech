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
"""StreamingSALM — Streaming Speech-Augmented Language Model with latency control."""

from __future__ import annotations

import random
import warnings
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import loss_parallel

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder
from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner
from nemo.collections.speechlm2.parts.context_biasing import maybe_apply_context_biasing
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.interleaving import build_interleaved_sequence
from nemo.collections.speechlm2.parts.kv_cache import maybe_evict_cache
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf


@dataclass
class StreamingState:
    """State for functional streaming inference."""

    kv_cache: tuple | None
    cache_length: int
    abs_position: int  # true absolute token index (for correct RoPE after cache eviction)
    latency: int
    sink_size: int
    window_size: int
    num_processed_frames: int
    num_emitted_tokens: int
    last_prediction_was_text: bool
    raw_code_buffer: Tensor | None = None  # last (num_codebooks-1) raw code frames for delay pattern continuity


class StreamingSALM(LightningModule, HFHubMixin):
    """
    Streaming Speech-Augmented Language Model with latency control.

    Architecture:
    - Pretrained LLM decoder (e.g., Qwen3-1.7B)
    - Mimi audio codec encoder (frozen, all codebooks with delay pattern)
    - Per-codebook audio token embeddings (trainable)
    - Qwen Forced Aligner (frozen, training only)
    - <blank> token for non-emission predictions

    Training: on-the-fly forced alignment -> interleaved audio+text sequences
    Inference: frame-by-frame autoregressive generation with KV cache
    """

    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to StreamingSALM as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)

        # --- LLM ---
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        self.llm = load_pretrained_hf(
            self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights
        )

        # Add <blank> token
        self.blank_token = self.cfg.get("blank_token", "<blank>")
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.blank_token]}
        )
        self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))

        # Separate embedding layer (same pattern as SALM)
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # --- Mimi Encoder (frozen) ---
        self.mimi = MimiEncoder(
            pretrained_model=self.cfg.get("pretrained_mimi", "kyutai/mimi"),
            num_codebooks=self.cfg.get("num_codebooks", 8),
        )

        # --- Audio Token Embeddings (trainable) ---
        self.num_codebooks = self.mimi.num_codebooks
        self.audio_codebook_size = self.mimi.codebook_size
        llm_hidden = self.llm.config.hidden_size
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    self.audio_codebook_size + 1,  # +1 for delay pattern padding
                    llm_hidden,
                    padding_idx=self.audio_codebook_size,
                )
                for _ in range(self.num_codebooks)
            ]
        )

        # --- Learned padding audio embedding for flush phase ---
        # Used when audio has ended but text tokens remain to be emitted.
        # Learned (not zero) so the model can distinguish flush from batch-padding.
        self.pad_audio_embed = nn.Parameter(torch.zeros(llm_hidden))

        # --- Forced Aligner (frozen, training only, late-initialized) ---
        self.forced_aligner = None

        # --- Delay pattern (disable for non-generative audio use) ---
        self.use_delay_pattern = self.cfg.get("use_delay_pattern", True)

        # --- Latency and context biasing config ---
        self.min_latency = self.cfg.get("min_latency", 1)
        self.max_latency = self.cfg.get("max_latency", 10)
        self.context_biasing_prob = self.cfg.get("context_biasing_prob", 0.2)

        # --- Streaming cache config ---
        self.cache_sink_size = self.cfg.get("cache_sink_size", 64)
        self.cache_window_size = self.cfg.get("cache_window_size", 2048)

        # --- Offline mode (no forced aligner, all audio before text) ---
        self.offline = self.cfg.get("offline", False)

        # --- FSDP / TP flags ---
        self._use_fsdp = False
        self._use_tp = False

        maybe_install_lora(self)

    @property
    def sample_rate(self) -> int:
        return MimiEncoder.SAMPLE_RATE

    @property
    def blank_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.blank_token)

    @property
    def text_vocab_size(self):
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, "
                "using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    def on_fit_start(self) -> None:
        """Late-initialize the forced aligner (0.6B model) before fit (incl. sanity check)."""
        if self.offline:
            return
        if self.forced_aligner is None:
            self.forced_aligner = QwenForcedAligner(
                pretrained_model=self.cfg.get(
                    "pretrained_forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"
                ),
                language=self.cfg.get("forced_aligner_language", "English"),
            )

    def embed_audio_codes(self, delayed_codes: Tensor) -> Tensor:
        """
        Embed multi-codebook audio codes using per-codebook embeddings with sum pooling.

        Args:
            delayed_codes: (B, num_codebooks, T) with delay pattern applied

        Returns:
            audio_embeds: (B, T, H) frame-level audio embeddings
        """
        return torch.stack(
            [self.audio_embeddings[k](delayed_codes[:, k, :]) for k in range(self.num_codebooks)]
        ).sum(dim=0)

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
    ) -> dict[str, Tensor]:
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        ans = {"logits": out["logits"]}
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict) -> dict:
        """
        Build interleaved audio+text sequences with on-the-fly forced alignment.

        Expects ``batch["audios"]`` at Mimi's native 24 kHz sample rate.
        If a ``sample_rate`` key is present in the batch, it is validated.
        """
        device = batch["audios"].device

        # Validate sample rate if provided by the dataloader
        if "sample_rate" in batch:
            sr = batch["sample_rate"]
            sr_val = sr.item() if isinstance(sr, Tensor) else sr
            assert sr_val == MimiEncoder.SAMPLE_RATE, (
                f"StreamingSALM requires audio at {MimiEncoder.SAMPLE_RATE} Hz, "
                f"but batch has sample_rate={sr_val} Hz. "
                "Resample your data to 24 kHz or set data.train_ds.sample_rate=24000."
            )

        audio_lens = batch["audio_lens"]
        # Cast audio to model dtype for Mimi (e.g. bfloat16 when trainer.precision=bf16-true)
        audios_for_mimi = batch["audios"].to(dtype=next(self.mimi.parameters()).dtype)

        # 1-2. Audio encoding + optional delay pattern (audio already at 24 kHz)
        with torch.cuda.nvtx.range("mimi_encode"):
            with torch.no_grad():
                codes, code_lens = self.mimi.encode(audios_for_mimi, audio_lens)
            if self.use_delay_pattern:
                codes = MimiEncoder.apply_delay_pattern(codes, code_lens)

            # 3. Audio frame embeddings
            audio_embeds = self.embed_audio_codes(codes)

        if self.offline:
            return self._prepare_inputs_offline(batch, audio_embeds, code_lens)

        # 4. Forced alignment
        with torch.cuda.nvtx.range("qfa_align"):
            with torch.no_grad():
                if "audios_16k" in batch:
                    # Fast path: dataloader already resampled to 16 kHz numpy arrays
                    alignments = self.forced_aligner.align_numpy(
                        batch["audios_16k"],
                        batch["transcripts"],
                    )
                else:
                    # Legacy path: resample on GPU + GPU→CPU transfer
                    alignments = self.forced_aligner.align(
                        batch["audios"],
                        audio_lens,
                        batch["transcripts"],
                        source_sample_rate=MimiEncoder.SAMPLE_RATE,
                    )

        # 5-6. Build interleaved sequences per example (batched embedding)
        with torch.cuda.nvtx.range("interleaving"):
            frame_shift = self.mimi.token_equivalent_duration

            pad_embed = self.pad_audio_embed

            # --- Phase 1: Build sequences, collect ALL text token IDs ---
            all_token_ids: list[int] = []  # flat list of ALL token IDs to embed
            per_sample: list[tuple] = []   # (input_parts, label_parts, num_prompt, num_text)

            for i in range(len(batch["transcripts"])):
                K = random.randint(self.min_latency, self.max_latency)
                T = code_lens[i].item()
                audio_embs_i = audio_embeds[i, :T]

                # Optional context biasing
                context_text, audio_embs_i, alignment_i, T = maybe_apply_context_biasing(
                    audio_embs_i,
                    alignments[i],
                    batch["transcripts"][i],
                    T,
                    self.context_biasing_prob,
                    frame_shift,
                )

                # Build text prompt
                prompt = f"Latency: {K}"
                if context_text:
                    prompt += f" Context: >>{context_text}<<"
                prompt_ids = self.tokenizer.text_to_ids(prompt)

                # Build interleaved audio+text sequence (no embedding yet)
                input_parts, label_parts, text_token_ids = build_interleaved_sequence(
                    audio_embs_i,
                    alignment_i,
                    K,
                    self.blank_token_id,
                    self.tokenizer,
                    frame_shift,
                    pad_embed=pad_embed,
                )

                # Accumulate token IDs: prompt first, then interleaved text
                all_token_ids.extend(prompt_ids)
                all_token_ids.extend(text_token_ids)
                per_sample.append((input_parts, label_parts, len(prompt_ids), len(text_token_ids)))

            # --- Phase 2: Single batched embed_tokens call ---
            if all_token_ids:
                all_embeds = self.embed_tokens(
                    torch.tensor(all_token_ids, dtype=torch.long, device=device)
                )  # (N_total, H) — ONE autograd node
            else:
                all_embeds = torch.empty(0, dtype=audio_embeds.dtype, device=device)

            # --- Phase 3: Distribute embeddings back ---
            all_input_embeds = []
            all_labels = []
            idx = 0

            for input_parts, label_parts, num_prompt, num_text in per_sample:
                # Extract prompt embeddings
                prompt_embeds = all_embeds[idx : idx + num_prompt]
                idx += num_prompt

                # Fill None slots in input_parts with text embeddings
                text_idx = idx
                for j, part in enumerate(input_parts):
                    if part is None:
                        input_parts[j] = all_embeds[text_idx]
                        text_idx += 1
                idx += num_text

                # Concatenate prompt + interleaved
                interleaved_embeds = torch.stack(input_parts, dim=0)
                labels = torch.tensor(label_parts, dtype=torch.long, device=device)

                input_embs = torch.cat([prompt_embeds, interleaved_embeds], dim=0)
                full_labels = torch.cat(
                    [
                        torch.full(
                            (num_prompt,), -100, dtype=torch.long, device=device
                        ),
                        labels,
                    ]
                )

                all_input_embeds.append(input_embs)
                all_labels.append(full_labels)

            # 7. Left-pad and stack
            input_embeds, attention_mask = _left_pad_embeds(all_input_embeds)
            labels = left_collate_vectors(all_labels, padding_value=-100)

        return {
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _prepare_inputs_offline(
        self, batch: dict, audio_embeds: Tensor, code_lens: Tensor
    ) -> dict:
        """
        Offline mode: ``[audio_0 ... audio_{T-1} | text_0 ... text_{N-2}]``
        with labels ``[-100*(T-1) | text_0 ... text_{N-1}]`` (unshifted loss).

        ``text_ids`` includes EOS so the last predicted label is EOS.
        """
        device = audio_embeds.device
        all_token_ids: list[int] = []
        per_sample: list[tuple[int, int]] = []  # (T, num_text)

        for i, transcript in enumerate(batch["transcripts"]):
            T = code_lens[i].item()
            text_ids = self.tokenizer.text_to_ids(transcript) + [self.text_eos_id]
            all_token_ids.extend(text_ids)
            per_sample.append((T, len(text_ids)))

        if all_token_ids:
            all_embeds = self.embed_tokens(
                torch.tensor(all_token_ids, dtype=torch.long, device=device)
            )
        else:
            all_embeds = torch.empty(
                0, audio_embeds.shape[-1], dtype=audio_embeds.dtype, device=device
            )

        all_input_embeds: list[Tensor] = []
        all_labels: list[Tensor] = []
        idx = 0

        for i, (T, num_text) in enumerate(per_sample):
            audio_embs_i = audio_embeds[i, :T]
            text_embeds = all_embeds[idx : idx + num_text]
            text_ids = all_token_ids[idx : idx + num_text]
            idx += num_text

            # Input: [audio_0..audio_{T-1} | text_0..text_{N-2}]  (drop EOS embed from input)
            # Labels: [-100 * (T-1) | text_0 | text_1 | ... | text_{N-1}=EOS]
            input_embs = torch.cat([audio_embs_i, text_embeds[:-1]])
            labels = torch.cat([
                torch.full((T - 1,), -100, dtype=torch.long, device=device),
                torch.tensor(text_ids, dtype=torch.long, device=device),
            ])
            all_input_embeds.append(input_embs)
            all_labels.append(labels)

        input_embeds, attention_mask = _left_pad_embeds(all_input_embeds)
        labels = left_collate_vectors(all_labels, padding_value=-100)
        return {"input_embeds": input_embeds, "attention_mask": attention_mask, "labels": labels}

    def training_step(self, batch: dict, batch_idx: int):
        # Freeze modules
        if is_frozen(self.mimi):
            self.mimi.eval()
        if is_frozen(self.llm):
            self.llm.eval()

        with torch.cuda.nvtx.range("prepare_inputs"):
            inputs = self.prepare_inputs(batch)

        with torch.cuda.nvtx.range("llm_forward"):
            forward_outputs = self(
                inputs["input_embeds"], attention_mask=inputs["attention_mask"]
            )

        # UNSHIFTED loss — logits[i] predicts labels[i]
        with torch.cuda.nvtx.range("loss"):
            labels = inputs["labels"]
            valid_mask = labels != -100
            num_valid = torch.clamp(valid_mask.long().sum(), min=1)
            with loss_parallel():
                loss = (
                    F.cross_entropy(
                        forward_outputs["logits"].view(
                            -1, forward_outputs["logits"].size(-1)
                        ),
                        labels.view(-1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_valid
                )

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]["lr"]
                if self._trainer is not None
                else 0
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_valid_positions": num_valid.to(torch.float32),
        }
        self.log_dict(ans, on_step=True)
        return ans

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(
                inputs["input_embeds"], attention_mask=inputs["attention_mask"]
            )
            labels = inputs["labels"]
            valid_mask = labels != -100
            num_valid = torch.clamp(valid_mask.long().sum(), min=1)
            with loss_parallel():
                loss = (
                    F.cross_entropy(
                        forward_outputs["logits"].view(
                            -1, forward_outputs["logits"].size(-1)
                        ),
                        labels.view(-1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_valid
                )

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = labels.reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(loss)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        if val_losses:
            self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        if accuracies:
            self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_accuracies.clear()

    # ------------------------------------------------------------------
    # Backward + OOMptimizer
    # ------------------------------------------------------------------

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @property
    def oomptimizer_schema(self) -> dict:
        from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType

        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
            ],
        }

    @torch.no_grad()
    def generate(
        self,
        audio: Tensor,
        audio_lens: Tensor,
        latency: int = 1,
        context: str | list[str] | None = None,
    ) -> list[str]:
        """
        Batched full-utterance offline inference.

        All B samples are processed in parallel through the LLM.  At each
        decoding step every sample contributes exactly one input token
        (audio frame, text feedback, flush pad, or a no-op pad for finished
        samples), keeping the KV cache length uniform across the batch.

        Args:
            audio: (B, T_samples) raw audio waveform at 24 kHz.
            audio_lens: (B,) sample counts.
            latency: emission latency K in audio frames.
            context: optional context biasing text.  Can be a single string
                (shared across the batch), a list of B strings (per-sample),
                or ``None``.
        """
        device = audio.device
        B = audio.shape[0]

        # --- Encode audio (batched) ---
        audio = audio.to(dtype=next(self.mimi.parameters()).dtype)
        codes, code_lens = self.mimi.encode(audio, audio_lens)
        if self.use_delay_pattern:
            codes = MimiEncoder.apply_delay_pattern(codes, code_lens)
        audio_embeds = self.embed_audio_codes(codes)  # (B, T_max, H)

        if self.offline:
            return self._generate_offline(audio_embeds, code_lens)

        T = [code_lens[b].item() for b in range(B)]

        pad_embed = self.pad_audio_embed

        # --- Build per-sample prompts ---
        # Normalise context to a list of B items (None or str each).
        if context is None or isinstance(context, str):
            ctx_list: list[str | None] = [context] * B
        else:
            assert len(context) == B
            ctx_list = list(context)

        prompt_id_lists: list[list[int]] = []
        for ctx in ctx_list:
            prompt = f"Latency: {latency}"
            if ctx:
                prompt += f" Context: >>{ctx}<<"
            prompt_id_lists.append(self.tokenizer.text_to_ids(prompt))

        prompt_lens = [len(ids) for ids in prompt_id_lists]
        max_prompt_len = max(prompt_lens)

        # Left-pad prompt token ids so the last (real) token aligns.
        padded_prompt_ids = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        for b, ids in enumerate(prompt_id_lists):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids[b, pad_len:] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_mask[b, pad_len:] = 1

        prompt_embeds = self.embed_tokens(padded_prompt_ids)  # (B, max_prompt_len, H)

        # Process prompt with attention mask to ignore left-padding.
        prompt_positions = torch.arange(max_prompt_len, device=device)
        out = self.llm(
            inputs_embeds=prompt_embeds,
            attention_mask=attn_mask,
            cache_position=prompt_positions,
            use_cache=True,
            return_dict=True,
        )
        cache = out["past_key_values"]
        sink_size = max(self.cache_sink_size, max_prompt_len)
        abs_pos = max_prompt_len

        # Grow attention mask by 1 column at each decode step.
        # Shape maintained as (B, total_seq_len) to mask left-padding in prompt.
        # attn_mask is currently (B, max_prompt_len) — it grows as we decode.

        # --- Per-sample decoding state ---
        audio_ptr = [0] * B                  # next audio frame index
        flush_count = [0] * B                # flush frames sent so far
        generated: list[list[int]] = [[] for _ in range(B)]
        need_text_feedback = [False] * B     # feed predicted text next step
        pending_text = torch.zeros(B, dtype=torch.long, device=device)
        done = [False] * B

        # Upper bound: each audio frame can emit at most 1 text token (+ feedback),
        # plus flush frames with text feedback.
        max_steps = 2 * (max(T) + latency) + 16

        for _step in range(max_steps):
            if all(done):
                break

            # --- Build (B, 1, H) input for this step ---
            step_embeds = []
            is_feedback_step = [False] * B
            for b in range(B):
                if done[b]:
                    step_embeds.append(pad_embed)
                elif need_text_feedback[b]:
                    step_embeds.append(
                        self.embed_tokens(pending_text[b : b + 1]).squeeze(0)
                    )
                    need_text_feedback[b] = False
                    is_feedback_step[b] = True
                elif audio_ptr[b] < T[b]:
                    step_embeds.append(audio_embeds[b, audio_ptr[b]])
                    audio_ptr[b] += 1
                elif flush_count[b] < latency:
                    step_embeds.append(pad_embed)
                    flush_count[b] += 1
                else:
                    done[b] = True
                    step_embeds.append(pad_embed)

            if all(done):
                break

            # Extend attention mask: all samples attend to this new position.
            attn_mask = torch.cat(
                [attn_mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1
            )

            input_embeds = torch.stack(step_embeds).unsqueeze(1)  # (B, 1, H)
            cur_pos = torch.tensor([abs_pos], device=device)

            out = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                past_key_values=cache,
                cache_position=cur_pos,
                use_cache=True,
                return_dict=True,
            )
            cache = out["past_key_values"]
            abs_pos += 1
            cache, new_cache_len = maybe_evict_cache(
                cache, cache.get_seq_length(), sink_size, self.cache_window_size
            )
            if new_cache_len < abs_pos:
                # Eviction occurred — cache now has sink_size + window_size
                # entries.  Trim the attention mask to match: keep the first
                # sink_size columns and the last window_size columns.
                attn_mask = torch.cat(
                    [attn_mask[:, :sink_size], attn_mask[:, -self.cache_window_size:]],
                    dim=1,
                )

            # --- Decode predictions per sample ---
            # Only check predictions from audio/flush steps, not text-feedback
            # steps.  In training, text-feedback labels are always blank; the
            # model's output there is not an emission signal.  Matching this
            # in inference avoids (a) premature flush termination and (b)
            # hallucination cascades from text-feedback logits.
            pred_ids = out["logits"][:, -1, :].argmax(dim=-1)  # (B,)
            for b in range(B):
                if done[b] or is_feedback_step[b]:
                    continue
                pred_id = pred_ids[b].item()
                if pred_id != self.blank_token_id and pred_id != self.text_eos_id:
                    generated[b].append(pred_id)
                    need_text_feedback[b] = True
                    pending_text[b] = pred_id
                elif audio_ptr[b] >= T[b]:
                    # In flush phase and got blank/EOS from pad frame → done
                    done[b] = True

        return [self.tokenizer.ids_to_text(g) for g in generated]

    def _generate_offline(
        self, audio_embeds: Tensor, code_lens: Tensor
    ) -> list[str]:
        """Offline generation: process all audio, then greedy-decode text."""
        B = audio_embeds.shape[0]
        device = audio_embeds.device

        # Left-pad audio to match training convention (_left_pad_embeds).
        # This ensures position -1 is always the last valid audio frame.
        max_T = code_lens.max().item()
        padded_audio = torch.zeros(B, max_T, audio_embeds.shape[-1], dtype=audio_embeds.dtype, device=device)
        attn_mask = torch.zeros(B, max_T, dtype=torch.long, device=device)
        for b in range(B):
            T = code_lens[b].item()
            padded_audio[b, max_T - T :] = audio_embeds[b, :T]
            attn_mask[b, max_T - T :] = 1

        # Prefill: process all audio frames at once
        positions = torch.arange(max_T, device=device)
        out = self.llm(
            inputs_embeds=padded_audio,
            attention_mask=attn_mask,
            cache_position=positions,
            use_cache=True,
            return_dict=True,
        )
        cache = out["past_key_values"]
        abs_pos = max_T

        # Greedy autoregressive text decode
        generated: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        max_new_tokens = 512

        pred_ids = out["logits"][:, -1, :].argmax(dim=-1)  # (B,)
        for _step in range(max_new_tokens):
            for b in range(B):
                if done[b]:
                    continue
                pid = pred_ids[b].item()
                if pid == self.text_eos_id:
                    done[b] = True
                else:
                    generated[b].append(pid)
            if all(done):
                break

            next_embeds = self.embed_tokens(pred_ids).unsqueeze(1)  # (B, 1, H)
            attn_mask = torch.cat(
                [attn_mask, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1
            )
            cur_pos = torch.tensor([abs_pos], device=device)
            out = self.llm(
                inputs_embeds=next_embeds,
                attention_mask=attn_mask,
                past_key_values=cache,
                cache_position=cur_pos,
                use_cache=True,
                return_dict=True,
            )
            cache = out["past_key_values"]
            abs_pos += 1
            pred_ids = out["logits"][:, -1, :].argmax(dim=-1)

        return [self.tokenizer.ids_to_text(g) for g in generated]

    @torch.no_grad()
    def generate_streaming(
        self,
        audio_codes: Tensor | None,
        state: StreamingState | None,
        latency: int = 1,
        context: str | None = None,
    ) -> tuple[list[list[int]], StreamingState]:
        """
        Functional streaming inference (single-stream, B=1).

        Each stream maintains its own KV cache and state.  Cross-stream
        batching (B > 1) is not supported because text-feedback steps
        would cause cache-length divergence between batch elements.

        Note:
            ``latency`` and ``context`` are only used on the **first** call
            (when ``state is None``) to build the prompt and initialise the
            session.  On subsequent calls they are ignored — the values are
            captured in ``StreamingState``.

        Call pattern::

            state = None
            while has_audio:
                new_codes = get_next_audio_chunk()   # (1, K, T_chunk)
                tokens, state = model.generate_streaming(new_codes, state, latency=K)
                emit(tokens)
            tokens, state = model.generate_streaming(None, state)
        """
        device = self.device

        if state is None:
            # Initialize new session: process prompt
            prompt = f"Latency: {latency}"
            if context:
                prompt += f" Context: >>{context}<<"
            prompt_ids = torch.tensor(
                self.tokenizer.text_to_ids(prompt), dtype=torch.long, device=device
            )
            prompt_len = len(prompt_ids)
            prompt_embeds = self.embed_tokens(prompt_ids).unsqueeze(0)
            prompt_positions = torch.arange(prompt_len, device=device)
            out = self.llm(
                inputs_embeds=prompt_embeds,
                cache_position=prompt_positions,
                use_cache=True,
                return_dict=True,
            )
            state = StreamingState(
                kv_cache=out["past_key_values"],
                cache_length=prompt_len,
                abs_position=prompt_len,
                latency=latency,
                sink_size=prompt_len,
                window_size=self.cache_window_size,
                num_processed_frames=0,
                num_emitted_tokens=0,
                last_prediction_was_text=False,
            )

        if audio_codes is None:
            # Flush: run up to `latency` extra decoding steps to emit any
            # remaining tokens that were delayed by the latency window.
            flushed: list[int] = []
            cache = state.kv_cache
            cache_pos = state.cache_length
            abs_pos = state.abs_position
            for _ in range(state.latency):
                # Feed the learned padding audio embedding to trigger pending text
                pad_emb = self.pad_audio_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
                cur_pos = torch.tensor([abs_pos], device=device)
                out = self.llm(
                    inputs_embeds=pad_emb,
                    past_key_values=cache,
                    cache_position=cur_pos,
                    use_cache=True,
                    return_dict=True,
                )
                cache = out["past_key_values"]
                cache_pos += 1
                abs_pos += 1
                cache, cache_pos = maybe_evict_cache(
                    cache, cache_pos, state.sink_size, state.window_size
                )
                pred_id = out["logits"][:, -1, :].argmax(dim=-1).item()
                if pred_id != self.blank_token_id and pred_id != self.text_eos_id:
                    flushed.append(pred_id)
                    text_emb = self.embed_tokens(
                        torch.tensor([[pred_id]], device=device)
                    )
                    cur_pos = torch.tensor([abs_pos], device=device)
                    out = self.llm(
                        inputs_embeds=text_emb,
                        past_key_values=cache,
                        cache_position=cur_pos,
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = out["past_key_values"]
                    cache_pos += 1
                    abs_pos += 1
                    cache, cache_pos = maybe_evict_cache(
                        cache, cache_pos, state.sink_size, state.window_size
                    )
                else:
                    break  # blank or EOS = nothing more pending

            new_state = StreamingState(
                kv_cache=cache,
                cache_length=cache_pos,
                abs_position=abs_pos,
                latency=state.latency,
                sink_size=state.sink_size,
                window_size=state.window_size,
                num_processed_frames=state.num_processed_frames,
                num_emitted_tokens=state.num_emitted_tokens + len(flushed),
                last_prediction_was_text=len(flushed) > 0,
                raw_code_buffer=state.raw_code_buffer,
            )
            return [flushed], new_state

        assert audio_codes.shape[0] == 1, (
            f"generate_streaming only supports B=1 (got B={audio_codes.shape[0]}). "
            "Each stream must maintain its own state; use separate calls per stream."
        )

        if self.use_delay_pattern:
            # Apply delay pattern with cross-chunk continuity (BUG 5 fix):
            # Prepend buffered codes from the previous chunk so that codebook k's
            # delay of k frames is computed relative to the utterance start, not
            # the chunk start.
            raw_codes = audio_codes  # (1, K, T_new)
            if state.raw_code_buffer is not None:
                combined = torch.cat([state.raw_code_buffer, raw_codes], dim=2)
                overlap = state.raw_code_buffer.shape[2]
            else:
                combined = raw_codes
                overlap = 0

            combined_lens = torch.full(
                (1,), combined.shape[2], dtype=torch.long, device=device
            )
            delayed = MimiEncoder.apply_delay_pattern(combined, combined_lens)
            # Only embed the NEW frames (skip the overlap prefix)
            audio_embeds = self.embed_audio_codes(delayed[:, :, overlap:])

            # Save last (num_codebooks - 1) raw code frames for the next chunk.
            # Use ``combined`` (buffer + current chunk) rather than ``raw_codes``
            # alone so the buffer accumulates enough context for higher codebooks
            # when individual chunks are smaller than num_codebooks - 1 frames.
            buffer_size = min(self.num_codebooks - 1, combined.shape[2])
            new_raw_code_buffer = combined[:, :, -buffer_size:] if buffer_size > 0 else None
        else:
            # No delay pattern — embed raw codes directly, no cross-chunk buffer needed.
            audio_embeds = self.embed_audio_codes(audio_codes)
            new_raw_code_buffer = None

        emitted: list[int] = []
        cache = state.kv_cache
        cache_pos = state.cache_length
        abs_pos = state.abs_position

        for f in range(audio_embeds.shape[1]):
            frame_emb = audio_embeds[:, f : f + 1, :]
            cur_pos = torch.tensor([abs_pos], device=device)
            out = self.llm(
                inputs_embeds=frame_emb,
                past_key_values=cache,
                cache_position=cur_pos,
                use_cache=True,
                return_dict=True,
            )
            cache = out["past_key_values"]
            cache_pos += 1
            abs_pos += 1

            pred_id = out["logits"][:, -1, :].argmax(dim=-1).item()

            if pred_id != self.blank_token_id and pred_id != self.text_eos_id:
                emitted.append(pred_id)
                # Feed predicted text token back as input
                text_emb = self.embed_tokens(
                    torch.tensor([[pred_id]], device=device)
                )
                cur_pos = torch.tensor([abs_pos], device=device)
                out = self.llm(
                    inputs_embeds=text_emb,
                    past_key_values=cache,
                    cache_position=cur_pos,
                    use_cache=True,
                    return_dict=True,
                )
                cache = out["past_key_values"]
                cache_pos += 1
                abs_pos += 1

            # Cache eviction
            cache, cache_pos = maybe_evict_cache(
                cache, cache_pos, state.sink_size, state.window_size
            )

        new_state = StreamingState(
            kv_cache=cache,
            cache_length=cache_pos,
            abs_position=abs_pos,
            latency=state.latency,
            sink_size=state.sink_size,
            window_size=state.window_size,
            num_processed_frames=state.num_processed_frames + audio_embeds.shape[1],
            num_emitted_tokens=state.num_emitted_tokens + len(emitted),
            last_prediction_was_text=len(emitted) > 0 and emitted[-1] != self.blank_token_id,
            raw_code_buffer=new_raw_code_buffer,
        )

        return [emitted], new_state

    def configure_model(self) -> None:
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (
            dp_mesh := device_mesh.get("data_parallel")
        ) is not None and dp_mesh.size() > 1:
            self._use_fsdp = True
            fsdp_config = {"mesh": dp_mesh}
            for idx, layer in enumerate(llm.model.layers):
                llm.model.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            llm.lm_head = fully_shard(llm.lm_head, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            for k in range(self.num_codebooks):
                self.audio_embeddings[k] = fully_shard(
                    self.audio_embeddings[k], **fsdp_config
                )

    def configure_optimizers(self):
        return configure_optimizers(self)


def _left_pad_embeds(
    embed_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Left-pad a list of (T_i, H) embedding tensors to a batch of (B, T_max, H).

    Returns:
        padded: (B, T_max, H)
        attention_mask: (B, T_max) — True for real positions
    """
    max_len = max(e.shape[0] for e in embed_list)
    H = embed_list[0].shape[1]
    device = embed_list[0].device
    dtype = embed_list[0].dtype
    B = len(embed_list)

    padded = torch.zeros(B, max_len, H, device=device, dtype=dtype)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

    for i, emb in enumerate(embed_list):
        T = emb.shape[0]
        padded[i, max_len - T :] = emb
        attention_mask[i, max_len - T :] = True

    return padded, attention_mask
