# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import atexit
import copy
import os
import random
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.modules.magpietts_modules import SpecialAudioToken, add_special_tokens
from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    get_speaker_embeddings_from_filepaths,
    print_grad_weight_summary,
    process_text_for_cer,
)
from nemo.collections.tts.parts.utils.reward_asr import RewardASRRouter
from nemo.utils import logging

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    Normalizer = None
    PYNINI_AVAILABLE = False


class EasyMagpieTTSModelOnlinePO(EasyMagpieTTSModel):
    """
    EasyMagpie-TTS online preference optimization model (GRPO / DR-GRPO).

    Training flow:
    1. Sample multiple generations per prompt.
    2. Compute rewards (CER/SSIM/UTMOSv2).
    3. Compute group-normalized advantages.
    4. Run teacher-forced policy forward on the generated audio and phoneme tokens.
    5. Optimize both parts of the sampled trajectory with the same GRPO advantage.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        """Initialize the online PO model, including the frozen reference model, reward ASR/speaker
        verification models, optional UTMOSv2 scorer, and all PO hyper-parameters from ``cfg``.
        """
        super().__init__(cfg, trainer)

        self.run_val_inference = True  # Always run validation inference in PO.
        self.automatic_optimization = False

        ref_model_cfg = copy.deepcopy(cfg)
        with open_dict(ref_model_cfg):
            ref_model_cfg.train_ds = None
            ref_model_cfg.validation_ds = None

        self.reference_free = self.cfg.get('reference_free', False)
        if not self.reference_free:
            self._reference_model = EasyMagpieTTSModel(cfg=ref_model_cfg)
            logging.info("Loading EasyMagpie reference model from checkpoint")
            self._reference_model.load_state_dict(
                torch.load(cfg.reference_model_ckpt_path, map_location="cpu")['state_dict']
            )
            self._reference_model.freeze()
            self._reference_model._no_state_dict = True
            logging.info("Reference model loaded and frozen")

        reward_asr_cfg = cfg.get('reward_asr')
        if reward_asr_cfg is None:
            # Backward-compatible translation for existing recipes. New recipes
            # should use reward_asr.default_backend and language_routes directly.
            legacy_backend = str(cfg.get("reward_asr_model", "nemo"))
            model_name = cfg.get("reward_asr_model_name", "nvidia/parakeet-ctc-0.6b")
            if legacy_backend in {"nemo", "nemotron"}:
                backend_type = "nemo_process" if legacy_backend == "nemotron" else "nemo"
                reward_asr_cfg = {
                    "default_backend": legacy_backend,
                    "language_routes": {},
                    "log_samples": cfg.get("reward_asr_log_samples", 0),
                    "backends": {
                        legacy_backend: {
                            "type": backend_type,
                            "model_name": model_name,
                            "batch_size": cfg.get("reward_asr_batch_size", 16),
                            "attention_context": cfg.get("reward_asr_att_context_size"),
                        }
                    },
                }
            elif legacy_backend == "whisper":
                reward_asr_cfg = {
                    "default_backend": "whisper",
                    "language_routes": {},
                    "log_samples": cfg.get("reward_asr_log_samples", 0),
                    "backends": {
                        "whisper": {
                            "type": "whisper",
                            "model_name": "openai/whisper-large-v3",
                        }
                    },
                }
            elif legacy_backend == "qwen_whisper":
                whisper_languages = list(cfg.get("qwen_asr_whisper_languages", ["hi"]))
                qwen_cfg = {
                    "type": "qwen",
                    "model_name": cfg.get("qwen_asr_model_name", "Qwen/Qwen3-ASR-0.6B"),
                    "batch_size": cfg.get("qwen_asr_batch_size", 4),
                    "max_new_tokens": cfg.get("qwen_asr_max_new_tokens", 256),
                }
                if cfg.get("qwen_asr_python") is not None:
                    qwen_cfg["python_executable"] = cfg.qwen_asr_python
                reward_asr_cfg = {
                    "default_backend": "qwen",
                    "language_routes": {language: "whisper" for language in whisper_languages},
                    "log_samples": cfg.get("reward_asr_log_samples", 0),
                    "backends": {
                        "qwen": qwen_cfg,
                        "whisper": {
                            "type": "whisper",
                            "model_name": "openai/whisper-large-v3",
                        },
                    },
                }
            else:
                raise ValueError(
                    f"Unknown legacy reward_asr_model={legacy_backend!r}; configure model.reward_asr instead"
                )
        self._reward_asr_router = RewardASRRouter(reward_asr_cfg, device_getter=lambda: self.device)
        self.reward_asr_log_samples = max(int(reward_asr_cfg.get("log_samples", 0)), 0)
        self.use_multilingual_asr = True

        self._eval_speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=cfg.get('speaker_verification_model_name', 'titanet_large')
        )
        self._eval_speaker_verification_model.freeze()

        self.use_utmos = self.cfg.get('use_utmos', False)
        if self.use_utmos:
            logging.info(f"UTMOSv2 naturalness reward will run on {self.utmos_device}")

        self.loss_type = self.cfg.get('loss_type', 'grpo')
        if self.loss_type not in ['grpo', 'dr_grpo']:
            raise ValueError(f"Received loss_type={self.loss_type}. Supported values: ['grpo', 'dr_grpo'].")
        self.scale_rewards = self.cfg.get('scale_rewards', True)
        self.max_decoder_steps = self.cfg.get('max_decoder_steps', 220)
        self.aux_phoneme_loss_weight = self.cfg.get('aux_phoneme_loss_weight', 1.0)
        self.phoneme_po_loss_weight = self.cfg.get('phoneme_po_loss_weight', 0.0)
        if self.phoneme_po_loss_weight < 0.0:
            raise ValueError(f"phoneme_po_loss_weight must be non-negative, got {self.phoneme_po_loss_weight}.")
        train_dataset_cfg = self.cfg.get("train_ds", {}).get("dataset", {})
        challenging_text_end_prob = float(train_dataset_cfg.get("challenging_text_end_prob", 0.0))
        if challenging_text_end_prob > 0.0 and float(self.cfg.get("gt_phoneme_input_prob", 0.0)) > 0.0:
            raise ValueError(
                "Challenging text replacement changes text without updating IPA. "
                "Set gt_phoneme_input_prob=0 to use predicted phonemes."
            )
        self.audio_sampling_temperature = float(self.cfg.get('inference_temperature', 0.7))
        self.audio_sampling_topk = int(self.cfg.get('inference_topk', 80))
        self.rollout_cfg_mode = str(self.cfg.get('rollout_cfg_mode', 'off'))
        if self.rollout_cfg_mode not in {'off', 'alternate'}:
            raise ValueError(f"rollout_cfg_mode must be one of ['off', 'alternate'], got {self.rollout_cfg_mode!r}.")
        self.inference_cfg_scale = float(self.cfg.get('inference_cfg_scale', 2.5))
        if self.audio_sampling_topk <= 0:
            self.audio_sampling_topk = self.num_all_tokens_per_codebook
        if self.audio_sampling_topk > self.num_all_tokens_per_codebook:
            raise ValueError(
                f"inference_topk={self.audio_sampling_topk} exceeds "
                f"num_all_tokens_per_codebook={self.num_all_tokens_per_codebook}."
            )
        self.phoneme_sampling_temperature = float(
            self.cfg.get('inference_phoneme_temperature', self.audio_sampling_temperature)
        )
        self.phoneme_sampling_topk = int(
            self.cfg.get(
                'inference_phoneme_topk',
                self.phoneme_vocab_size if self.phoneme_po_loss_weight > 0.0 else self.audio_sampling_topk,
            )
        )
        self.po_groups_per_subbatch = max(int(self.cfg.get('po_groups_per_subbatch', 1)), 1)
        self.batch_size_for_chunked_tf = self.cfg.get('batch_size_for_chunked_tf', 4)

        phoneme_sampling_method = self.cfg.get('inference_phoneme_sampling_method', 'argmax')
        if self.phoneme_po_loss_weight > 0.0:
            if (
                phoneme_sampling_method != 'sample'
                or self.phoneme_sampling_temperature <= 0.0
                or self.phoneme_sampling_topk < 2
            ):
                raise ValueError(
                    "Phoneme PO requires stochastic phoneme trajectories: set "
                    "inference_phoneme_sampling_method='sample', inference_phoneme_temperature > 0, "
                    "and inference_phoneme_topk >= 2."
                )
            if self.phoneme_sampling_topk > self.phoneme_vocab_size:
                raise ValueError(
                    f"inference_phoneme_topk={self.phoneme_sampling_topk} exceeds "
                    f"phoneme_vocab_size={self.phoneme_vocab_size}."
                )
            if self.phoneme_confidence_unk_threshold > 0.0:
                raise ValueError(
                    "Phoneme PO requires phoneme_confidence_unk_threshold=0. Confidence-based UNK replacement "
                    "changes sampled actions without a corresponding policy likelihood."
                )

        self._normalize_reward_transcript = self.cfg.get(
            'normalize_reward_transcript', self.cfg.get('normalize_whisper_transcript', True)
        )
        if self._normalize_reward_transcript:
            self._normalizer_cache = {}

        # Entropy bonus coefficient – encourages exploration and prevents mode collapse.
        # Set to 0.0 to disable. Typical range: 0.001–0.01.
        self.entropy_coeff = self.cfg.get('entropy_coeff', 0.0)

        # Filter out poor groups for stable optimization.
        self.best_cer_threshold = self.cfg.get('best_cer_threshold', 1.0)
        self.worst_cer_threshold = self.cfg.get('worst_cer_threshold', 1.0)

        if self.multiturn_grpo_enabled:
            if not self.cfg.get("use_multiturn_dataset", False):
                raise ValueError("multiturn_grpo_enabled requires use_multiturn_dataset=true.")
            if self.phoneme_po_loss_weight != 0.0 or self.aux_phoneme_loss_weight != 0.0:
                raise ValueError(
                    "The minimal multi-turn GRPO path optimizes audio actions only. "
                    "Set phoneme_po_loss_weight=0 and aux_phoneme_loss_weight=0."
                )
            if not self.reference_free or float(self.cfg.get("grpo_beta", 0.0)) != 0.0:
                raise ValueError(
                    "The minimal multi-turn GRPO path currently requires reference_free=true and grpo_beta=0."
                )
            if self.rollout_cfg_mode != 'off':
                raise ValueError("The minimal multi-turn GRPO path currently requires rollout_cfg_mode='off'.")
            if challenging_text_end_prob != 0.0:
                raise ValueError("Disable challenging-text replacement for the minimal multi-turn GRPO path.")
            context_shuffle_prob = float(train_dataset_cfg.get("context_audio_shuffle_batch_prob", 0.0))
            if context_shuffle_prob != 0.0:
                raise ValueError("Disable context-audio shuffle for the minimal multi-turn GRPO path.")

        if self.trainer is not None and str(self.trainer.precision) in ("32", "32-true"):
            self.decoder.float()

    def _get_trainable_module_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Return a dict mapping module-group name → list of trainable parameters."""
        modules_to_exclude = {
            '_speaker_verification_model',
            '_codec_model',
            '_eval_asr_model',
            '_eval_speaker_verification_model',
            '_reference_model',
            'whisper_model',
            'whisper_processor',
            '_utmos_calculator',
        }
        groups: Dict[str, List[torch.nn.Parameter]] = {}
        for name, module in self.named_children():
            if name in modules_to_exclude:
                continue
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                groups[name] = params
        return groups

    @torch.no_grad()
    def _compute_grad_and_weight_metrics(self) -> Dict[str, float]:
        """Compute per-module grad_norm, weight_norm, and global aggregates."""
        module_groups = self._get_trainable_module_groups()
        metrics: Dict[str, float] = {}
        all_grad_norms, all_weight_norms = [], []

        for group_name, params in module_groups.items():
            grad_norms, weight_norms = [], []
            for p in params:
                weight_norms.append(p.data.norm(2).item())
                if p.grad is not None:
                    grad_norms.append(p.grad.data.norm(2).item())

            module_weight_norm = float(np.sqrt(sum(w**2 for w in weight_norms)))
            metrics[f'weight_norm/{group_name}'] = module_weight_norm
            all_weight_norms.extend(weight_norms)

            if grad_norms:
                module_grad_norm = float(np.sqrt(sum(g**2 for g in grad_norms)))
                metrics[f'grad_norm/{group_name}'] = module_grad_norm
                all_grad_norms.extend(grad_norms)
            else:
                metrics[f'grad_norm/{group_name}'] = 0.0

        if all_grad_norms:
            metrics['grad_norm/global'] = float(np.sqrt(sum(g**2 for g in all_grad_norms)))
        if all_weight_norms:
            metrics['weight_norm/global'] = float(np.sqrt(sum(w**2 for w in all_weight_norms)))
        return metrics

    @torch.no_grad()
    def _compute_weight_update_metrics(self, prev_weights: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """Compute per-module weight delta norms (how much weights changed after optimizer step)."""
        metrics: Dict[str, float] = {}
        module_groups = self._get_trainable_module_groups()
        all_deltas = []
        for group_name, params in module_groups.items():
            deltas = []
            for p in params:
                pid = id(p)
                if pid in prev_weights:
                    deltas.append((p.data - prev_weights[pid]).norm(2).item())
            if deltas:
                metrics[f'weight_delta/{group_name}'] = float(np.sqrt(sum(d**2 for d in deltas)))
                all_deltas.extend(deltas)
        if all_deltas:
            metrics['weight_delta/global'] = float(np.sqrt(sum(d**2 for d in all_deltas)))
        return metrics

    @torch.no_grad()
    def _snapshot_trainable_weights(self) -> Dict[int, torch.Tensor]:
        """Take a snapshot of all trainable parameter values (by param id)."""
        snapshot = {}
        for params in self._get_trainable_module_groups().values():
            for p in params:
                snapshot[id(p)] = p.data.clone()
        return snapshot

    def setup_optimizer_param_groups(self):
        """
        Exclude frozen eval/reference modules and output modules that receive no
        active training signal. Including them would apply weight decay without a
        learning signal, slowly degrading their weights.
        """
        modules_to_exclude = {
            '_speaker_verification_model',
            '_codec_model',
            '_eval_asr_model',
            '_eval_speaker_verification_model',
            '_reference_model',
            'whisper_model',
            'whisper_processor',
            '_utmos_calculator',
            # These modules are not used by the PO loss and receive no gradients.
            # Including them would only apply weight decay, degrading their weights.
            'final_proj',
            'lm_text_head',
        }
        # Keep legacy GT-phoneme runs optimizer-compatible: the final phoneme
        # projection is trainable only when reward directly optimizes phoneme actions.
        if self.phoneme_po_loss_weight <= 0.0:
            modules_to_exclude.add('phoneme_final_proj')

        excluded_param_ids = set()
        for name, module in self.named_children():
            if name in modules_to_exclude and hasattr(module, "parameters"):
                for param in module.parameters():
                    excluded_param_ids.add(id(param))

        trainable_params = [p for p in self.parameters() if id(p) not in excluded_param_ids]
        self._optimizer_param_groups = [{"params": trainable_params}]

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return the model state dict, excluding reference model and UTMOSv2 calculator weights."""
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        keys_substrings_to_exclude = ['_reference_model', '_utmos_calculator']
        for key in list(state_dict.keys()):
            if any(substring in key for substring in keys_substrings_to_exclude):
                del state_dict[key]
        return state_dict

    def _get_state_dict_keys_to_exclude(self):
        return super()._get_state_dict_keys_to_exclude() + ['_reference_model']

    def _get_cached_normalizer(self, lang_key: Optional[str]):
        """Return a cached ``Normalizer`` for the given language, creating one on first access.

        Returns ``None`` if pynini is not installed or normalizer creation fails.
        """
        if not PYNINI_AVAILABLE:
            return None
        lang_key = lang_key if lang_key else "en"
        if lang_key not in self._normalizer_cache:
            logging.info(f"Creating normalizer for language: {lang_key}")
            try:
                self._normalizer_cache[lang_key] = Normalizer(input_case="cased", lang=lang_key)
            except Exception as e:
                logging.warning(f"Failed to create normalizer for language: {lang_key}. Error: {e}")
                self._normalizer_cache[lang_key] = None
        return self._normalizer_cache[lang_key]

    def _get_per_token_logps(
        self, logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log-probabilities in fp32, masked by ``loss_mask``.

        Args:
            logits: Unnormalized logits of shape ``[B, T, V]``.
            labels: Ground-truth token ids of shape ``[B, T]``.
            loss_mask: Binary mask of shape ``[B, T]`` indicating valid positions.

        Returns:
            Masked per-token log-probabilities of shape ``[B, T]``.
        """
        # Force fp32 for log_softmax to avoid bf16 precision issues that sever the
        # gradient path through the GRPO "exp(logps - logps.detach())" trick.
        # Under bf16 autocast, the tiny gradient signal through this identity-like
        # expression gets rounded to zero, disconnecting local_transformer_out_projections.
        with torch.cuda.amp.autocast(enabled=False):
            logits_fp32 = logits.float()
            per_token_logps = torch.gather(logits_fp32.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            # Masked tokens have -inf log-probability. Padded labels may select
            # one of those tokens, so multiplication by a zero
            # mask would produce `-inf * 0 = NaN`. Select masked values instead.
            per_token_logps = torch.where(loss_mask.bool(), per_token_logps, torch.zeros_like(per_token_logps))
        return per_token_logps

    @staticmethod
    def _apply_teacher_forced_transform(
        logits: torch.Tensor,
        temperature: float,
        forbidden_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Apply sanitization, token masking, and temperature scaling for policy likelihoods."""
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        if forbidden_token_ids:
            logits = logits.clone()
            logits[..., forbidden_token_ids] = float('-inf')
        return logits / temperature

    def compute_local_transformer_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """
        Override parent to force fp32 computation for the entire local transformer logits path.

        Under bf16-mixed autocast, the nn.Linear out_projections execute in bf16 and insert
        ToCopyBackward0 nodes in the autograd graph. The GRPO loss formula
        ``exp(logps - logps.detach())`` produces an identity in the forward pass, but the
        gradient signal through this expression is extremely small. The bf16 ToCopyBackward0
        nodes round these tiny gradients to zero, completely severing the gradient path to
        local_transformer_out_projections. Running the full computation in fp32 preserves
        the gradient fidelity.
        """
        with torch.cuda.amp.autocast(enabled=False):
            # Cast dec_out to fp32 if it's in a lower precision (e.g. bf16 from autocast)
            dec_out_fp32 = dec_out.float()
            return super().compute_local_transformer_logits(
                dec_out_fp32, audio_codes_target, targets_offset_by_one=targets_offset_by_one
            )

    def repeat_items_in_batch(self, batch: Dict, num_repeats: int) -> Dict:
        """Repeat every item in ``batch`` ``num_repeats`` times along the batch dimension.

        Tensors are repeated via ``repeat_interleave``; lists are element-wise duplicated.
        """
        repeated_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                repeated_batch[key] = value.repeat_interleave(num_repeats, dim=0)
            elif isinstance(value, list):
                repeated_value = []
                for item in value:
                    repeated_value.extend([item] * num_repeats)
                repeated_batch[key] = repeated_value
            else:
                repeated_batch[key] = value
        return repeated_batch

    @property
    def multiturn_grpo_enabled(self) -> bool:
        """Return whether target-turn GRPO with a ground-truth conversation prefix is enabled."""
        return bool(self.cfg.get("multiturn_grpo_enabled", False))

    @staticmethod
    def _is_user_turn(turn: Dict) -> bool:
        """Return ``True`` when dataset turn metadata identifies a user turn."""
        return str(turn.get("role", "")).lower() == "user"

    @staticmethod
    def _is_assistant_turn(turn: Dict) -> bool:
        """Return ``True`` for either supported assistant role spelling."""
        return str(turn.get("role", "")).lower() in {"assistant", "agent"}

    def _select_multiturn_target(self, batch: Dict, mode: str) -> Dict:
        """Choose the assistant turn whose audio actions will receive the GRPO loss.

        The v1 path accepts one strictly alternating conversation per rank. During
        training it samples uniformly from assistant turns starting before
        ``multiturn_grpo_max_prefix_seconds``; evaluation deterministically selects
        the last eligible turn. The returned copy is augmented with timeline and
        turn-count indices used to validate prefix traversal.
        """
        turn_metadata = batch.get("turn_metadata")
        if not isinstance(turn_metadata, list) or len(turn_metadata) != 1:
            raise ValueError(
                "The minimal multi-turn GRPO path requires exactly one conversation per rank "
                "(train batch size 1) and dataset-provided turn_metadata."
            )

        turns = turn_metadata[0]
        if not turns:
            raise ValueError("Cannot select a multi-turn GRPO target from an empty conversation.")
        roles = [str(turn.get("role", "")).lower() for turn in turns]
        if roles[0] != "user" or any(left == right for left, right in zip(roles, roles[1:])):
            raise ValueError(
                "The minimal multi-turn GRPO path currently supports strictly alternating "
                "conversations that start with a user turn."
            )

        max_prefix_seconds = float(self.cfg.get("multiturn_grpo_max_prefix_seconds", float("inf")))
        candidates = [
            idx
            for idx, turn in enumerate(turns)
            if self._is_assistant_turn(turn)
            and idx > 0
            and self._is_user_turn(turns[idx - 1])
            and float(turn["start"]) <= max_prefix_seconds
        ]
        if not candidates:
            raise ValueError(
                "No assistant turn satisfies the multi-turn GRPO target constraints. "
                f"max_prefix_seconds={max_prefix_seconds}."
            )

        if mode == "train":
            target_metadata_idx = random.choice(candidates)
        else:
            target_metadata_idx = candidates[-1]
        target_turn = copy.deepcopy(turns[target_metadata_idx])
        target_turn["metadata_idx"] = target_metadata_idx
        target_turn["assistant_turn_idx"] = sum(self._is_assistant_turn(turn) for turn in turns[:target_metadata_idx])
        target_turn["user_turn_idx"] = sum(self._is_user_turn(turn) for turn in turns[:target_metadata_idx])
        return target_turn

    def _encode_turn_text(self, text: str, num_repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize one assistant transcript and replicate it for grouped rollouts.

        Returns padded-free token rows containing a terminal text EOS and their
        identical lengths, both with batch size ``num_repeats``.
        """
        tokenizer_name = list(self.cfg.text_tokenizers.keys())[0]
        token_ids = self.tokenizer.encode(text, tokenizer_name=tokenizer_name) + [self.eos_id]
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        tokens = tokens.repeat(num_repeats, 1)
        lens = torch.full((num_repeats,), len(token_ids), dtype=torch.long, device=self.device)
        return tokens, lens

    def _encode_turn_phonemes(self, ipa: str, num_repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode GT IPA conditioning for one assistant turn and grouped rollouts.

        GT phonemes are part of the rollout state in the minimal implementation;
        they are not sampled actions and receive no phoneme PO loss.
        """
        if self.phoneme_tokenizer is None:
            raise RuntimeError("The minimal multi-turn GRPO path requires a phoneme tokenizer.")
        token_ids = (
            [self.phoneme_tokenizer.bos_token_id]
            + self.phoneme_tokenizer.encode(ipa or "")
            + [self.phoneme_tokenizer.eos_token_id]
        )
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        tokens = tokens.repeat(num_repeats, 1)
        lens = torch.full((num_repeats,), len(token_ids), dtype=torch.long, device=self.device)
        return tokens, lens

    @staticmethod
    def _reset_streaming_turn_state(state) -> None:
        """Reset turn-local streaming flags while retaining the causal prefix cache.

        Transformer/Mamba cache state, absolute audio progress, and the last audio
        input remain intact so the next user/assistant pair continues the same
        conversation. EOS, text, and phoneme completion flags are per-turn and are
        therefore cleared.
        """
        state.finished.zero_()
        state.text_finished.zero_()
        state.audio_prediction_start_idx.fill_(-1)
        state.audio_prediction_end_idx.fill_(-1)
        state.phoneme_prediction_start_idx.fill_(-1)
        state.phoneme_prediction_end_idx.fill_(-1)
        for attr in (
            "turn_text_tokens_seen",
            "phoneme_steps",
            "phoneme_stream_ended",
            "phoneme_eos_detected",
        ):
            if hasattr(state, attr):
                getattr(state, attr).zero_()
        state.last_phoneme_tokens = None
        state.gt_audio_embeddings = None
        state.gt_audio_lens = None
        state.gt_phoneme_embeddings = None
        state.gt_phoneme_lens = None

    def _set_gt_turn_phonemes(self, state, phoneme_tokens: torch.Tensor, phoneme_lens: torch.Tensor) -> None:
        """Install stacked GT phoneme embeddings for the current assistant turn.

        ``streaming_step`` consumes these embeddings instead of predicted phoneme
        feedback. This keeps the first multi-turn objective audio-only and makes
        rollout/replay conditioning deterministic.
        """
        stacked, stacked_lens = self.stack_codes(
            phoneme_tokens.unsqueeze(1),
            phoneme_lens,
            self.phoneme_tokenizer.bos_token_id,
            self.phoneme_tokenizer.eos_token_id,
            self.phoneme_stacking_factor,
            1,
        )
        state.gt_phoneme_embeddings = self.embed_phoneme_tokens(stacked)
        state.gt_phoneme_lens = stacked_lens

    def _set_gt_assistant_audio(self, state, audio: torch.Tensor, audio_lens: torch.Tensor) -> None:
        """Teacher-force one prior assistant waveform into the streaming prefix.

        The waveform is codec-encoded, converted, wrapped with audio BOS/EOS, and
        frame-stacked exactly like model inference. Its embeddings are offset by
        the state's absolute audio progress so subsequent streaming steps consume
        the GT assistant actions while extending the existing conversation cache.
        """
        audio_codes, audio_codes_lens = self._codec_helper.audio_to_codes(audio, audio_lens)
        if self._codec_converter is not None:
            audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes, audio_lens=audio_codes_lens
            ).long()
        audio_codes, audio_codes_lens = add_special_tokens(
            codes=audio_codes,
            codes_len=audio_codes_lens,
            bos_id=self.audio_bos_id,
            eos_id=self.audio_eos_id,
        )
        audio_codes, audio_codes_lens = self.stack_codes(
            audio_codes,
            audio_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        audio_codes = audio_codes[:, :, :-1]
        audio_codes_lens = audio_codes_lens - 1
        turn_embeddings = self.embed_audio_tokens(audio_codes)

        if not torch.equal(state.audio_steps, state.audio_steps[:1].expand_as(state.audio_steps)):
            raise RuntimeError("Repeated GRPO states diverged before GT-prefix audio prefill.")
        offset = int(state.audio_steps[0].item())
        if offset > 0:
            prefix = turn_embeddings.new_zeros(turn_embeddings.size(0), offset, turn_embeddings.size(2))
            turn_embeddings = torch.cat([prefix, turn_embeddings], dim=1)
        state.gt_audio_embeddings = turn_embeddings
        state.gt_audio_lens = audio_codes_lens + offset

    def _prepare_user_turn_embedding(self, audio: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
        """Codec-encode one user turn for timeline-aligned speech conditioning.

        A leading zero embedding matches the BOS alignment used by supervised
        multi-turn training. The returned tensor is ready for
        :meth:`streaming_prefill`.
        """
        user_codes, user_codes_lens = self._codec_helper.audio_to_codes(audio, audio_lens)
        if self._codec_converter is not None:
            user_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=user_codes, audio_lens=user_codes_lens
            ).long()
        user_codes, user_codes_lens = self.stack_codes(
            user_codes,
            user_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        user_embeddings = self.embed_audio_tokens(user_codes)
        max_len = int(user_codes_lens.max().item())
        user_embeddings = user_embeddings[:, :max_len]
        bos_pad = user_embeddings.new_zeros(user_embeddings.size(0), 1, user_embeddings.size(2))
        return torch.cat([bos_pad, user_embeddings], dim=1)

    def _prefill_user_turn(
        self,
        state,
        user_audio: torch.Tensor,
        user_audio_lens: torch.Tensor,
        turn_text: torch.Tensor,
        turn_text_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append a user turn to the cache before generating its assistant response.

        User speech embeddings (or duration-only silence when disabled) fill the
        conversation timeline. The final ``streaming_speech_delay`` positions also
        consume the first assistant text tokens, matching the streaming training
        alignment. Remaining text tokens and lengths are returned for generation.
        """
        batch_size = turn_text.size(0)
        delay_tokens = int(state.config.training_mode.streaming_speech_delay)
        if self.cfg.get("condition_on_user_speech", False):
            user_embeddings = self._prepare_user_turn_embedding(user_audio, user_audio_lens)
            if user_embeddings.size(1) < delay_tokens:
                pad = user_embeddings.new_zeros(
                    batch_size, delay_tokens - user_embeddings.size(1), user_embeddings.size(2)
                )
                user_embeddings = torch.cat([pad, user_embeddings], dim=1)
            user_prefill_steps = user_embeddings.size(1)
            user_prefill_tokens = torch.full(
                (batch_size, user_prefill_steps),
                self.pad_id,
                dtype=turn_text.dtype,
                device=turn_text.device,
            )
        else:
            samples_per_step = self.codec_model_samples_per_frame * self.frame_stacking_factor
            user_prefill_steps = max(int(round(user_audio.size(-1) / samples_per_step)), 1)
            user_prefill_tokens = torch.full(
                (batch_size, user_prefill_steps),
                self.pad_id,
                dtype=turn_text.dtype,
                device=turn_text.device,
            )
            user_embeddings = None
            delay_tokens = min(delay_tokens, user_prefill_steps)

        num_warmup = min(delay_tokens, int(turn_text_lens.min().item()), turn_text.size(1))
        warmup_tokens = torch.full(
            (batch_size, delay_tokens),
            self.pad_id,
            dtype=turn_text.dtype,
            device=turn_text.device,
        )
        if num_warmup > 0:
            warmup_tokens[:, :num_warmup] = turn_text[:, :num_warmup]
        remaining_text = turn_text[:, num_warmup:]
        remaining_lens = torch.clamp(turn_text_lens - num_warmup, min=0)

        if user_embeddings is not None and delay_tokens > 0:
            warmup_user = user_embeddings[:, -delay_tokens:]
            user_embeddings = user_embeddings[:, :-delay_tokens]
            user_prefill_tokens = user_prefill_tokens[:, :-delay_tokens]
        else:
            warmup_user = None

        if user_prefill_tokens.size(1) > 0:
            state = self.streaming_prefill(
                state=state,
                text_tokens=user_prefill_tokens,
                use_inference_mode=True,
                user_audio_channel_embedding=user_embeddings,
            )
        for step_idx in range(delay_tokens):
            user_step = warmup_user[:, step_idx] if warmup_user is not None else None
            state.finished.zero_()
            state, _, _ = self.streaming_step(
                state=state,
                text_tokens=warmup_tokens[:, step_idx],
                user_audio_channel_embedding=user_step,
                prefill_like_step=True,
                prefill_like_is_last_step=step_idx == delay_tokens - 1,
                use_inference_mode=True,
            )
        return remaining_text, remaining_lens

    def _run_streaming_assistant_turn(
        self,
        state,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        max_steps: int,
    ) -> None:
        """Advance one assistant turn until every rollout emits audio EOS or hits its limit.

        Text is streamed relative to the turn's starting position. Depending on
        state configuration, audio feedback is either GT (prefix turns) or sampled
        policy output (the selected target turn).
        """
        turn_offset = state.text_tokens_seen.clone()
        batch_size = text.size(0)
        device = text.device
        for _ in range(max_steps):
            relative_position = state.text_tokens_seen - turn_offset
            text_exhausted = relative_position >= text_lens
            if text.size(1) == 0:
                current_tokens = torch.full((batch_size,), self.eos_id, dtype=torch.long, device=device)
            else:
                position = relative_position.clamp(min=0, max=text.size(1) - 1)
                current_tokens = text[torch.arange(batch_size, device=device), position]
                current_tokens = torch.where(
                    text_exhausted,
                    torch.full_like(current_tokens, self.eos_id),
                    current_tokens,
                )
            state, _, _ = self.streaming_step(
                state=state,
                text_tokens=current_tokens,
                use_inference_mode=True,
            )
            if bool(state.finished.all().item()):
                return

    @torch.inference_mode()
    def _generate_multiturn_target_rollouts(
        self,
        batch: Dict,
        target_turn: Dict,
        num_generations: int,
        use_local_transformer_for_inference: bool,
    ):
        """Generate grouped audio rollouts for one selected assistant turn.

        The method initializes ``num_generations`` identical streaming states, then
        walks the conversation in chronological user/assistant pairs. Every prior
        user turn is prefixed as speech conditioning and every prior assistant turn
        is teacher-forced with GT phonemes/audio. Only the selected assistant turn
        samples audio. Prediction bookkeeping is reset after its user prefill so
        finalization returns target-turn audio only, never prefix or future audio.

        Runtime checks verify that exactly N user turns and N-1 assistant turns were
        consumed for assistant turn N and that generated codec lengths fit the
        source turn's fixed timeline window.
        """
        if batch["audio"].size(0) != 1:
            raise ValueError("The minimal multi-turn GRPO rollout requires train batch size 1.")
        if num_generations < 2:
            raise ValueError("Multi-turn GRPO requires at least two generations per target turn.")
        if not hasattr(self, "_codec_sil_codes_buffer"):
            self._generate_codec_silence_buffer()

        if "context_audio_codes" in batch:
            context_codes = batch["context_audio_codes"].repeat_interleave(num_generations, dim=0)
            context_lens = batch["context_audio_codes_lens"].repeat_interleave(num_generations, dim=0)
        else:
            context_codes, context_lens = self._codec_helper.audio_to_codes(
                batch["context_audio"], batch["context_audio_lens"]
            )
            context_codes = context_codes.repeat_interleave(num_generations, dim=0)
            context_lens = context_lens.repeat_interleave(num_generations, dim=0)
        context_text = batch["context_text_tokens"].repeat_interleave(num_generations, dim=0)
        context_text_lens = batch["context_text_tokens_lens"].repeat_interleave(num_generations, dim=0)

        state = self.streaming_init(
            context_audio_codes=context_codes,
            context_audio_codes_lens=context_lens,
            context_text_tokens=context_text,
            context_text_tokens_lens=context_text_lens,
            use_cfg=False,
            cfg_scale=1.0,
            use_local_transformer=use_local_transformer_for_inference,
            temperature=self.audio_sampling_temperature,
            topk=self.audio_sampling_topk,
            phoneme_temperature=self.phoneme_sampling_temperature,
            phoneme_topk=self.phoneme_sampling_topk,
            phoneme_input_type="gt",
            phoneme_sampling_method="argmax",
            use_inference_mode=True,
        )

        turns = batch["turn_metadata"][0]
        user_turn_row = 0
        prefix_assistant_turns = 0
        target_seen = False
        target_metadata_idx = int(target_turn["metadata_idx"])
        for metadata_idx, turn in enumerate(turns[: target_metadata_idx + 1]):
            if self._is_user_turn(turn):
                continue
            if (
                not self._is_assistant_turn(turn)
                or metadata_idx == 0
                or not self._is_user_turn(turns[metadata_idx - 1])
            ):
                raise ValueError("Expected every assistant turn to be immediately preceded by a user turn.")

            self._reset_streaming_turn_state(state)
            turn_text, turn_text_lens = self._encode_turn_text(str(turn["text"]), num_generations)
            turn_phonemes, turn_phoneme_lens = self._encode_turn_phonemes(str(turn.get("ipa", "")), num_generations)
            self._set_gt_turn_phonemes(state, turn_phonemes, turn_phoneme_lens)

            user_audio = batch["user_audio_turn_splitted"][user_turn_row : user_turn_row + 1]
            user_audio_lens = batch["user_audio_turn_splitted_lens"][user_turn_row : user_turn_row + 1]
            user_turn_row += 1
            user_audio = user_audio.repeat_interleave(num_generations, dim=0)
            user_audio_lens = user_audio_lens.repeat_interleave(num_generations, dim=0)

            is_target = metadata_idx == target_metadata_idx
            turn_text, turn_text_lens = self._prefill_user_turn(
                state, user_audio, user_audio_lens, turn_text, turn_text_lens
            )
            if is_target:
                # User prefill may append one tensor containing many silence frames, while
                # streaming EOS bookkeeping assumes one frame-stack per list entry. Keep
                # the populated cache/last-audio token but start target output indexing at 0.
                state.all_predictions = []
                state.audio_prediction_start_idx.fill_(-1)
                state.audio_prediction_end_idx.fill_(-1)

            turn_steps = max(int(turn["end_frame"]) - int(turn["start_frame"]), 1)
            if not is_target:
                prefix_assistant_turns += 1
                start_sample = int(round(float(turn["start"]) * self.sample_rate))
                end_sample = int(round(float(turn["end"]) * self.sample_rate))
                gt_audio = batch["audio"][:, start_sample:end_sample].repeat_interleave(num_generations, dim=0)
                gt_audio_lens = torch.full(
                    (num_generations,),
                    gt_audio.size(1),
                    dtype=torch.long,
                    device=gt_audio.device,
                )
                self._set_gt_assistant_audio(state, gt_audio, gt_audio_lens)
                max_steps = turn_steps + int(turn_text_lens.max().item()) + 8
            else:
                state.gt_audio_embeddings = None
                state.gt_audio_lens = None
                max_steps = min(self.max_decoder_steps, turn_steps)
                target_seen = True

            self._run_streaming_assistant_turn(state, turn_text, turn_text_lens, max_steps)
            if is_target:
                break
            state.gt_audio_embeddings = None
            state.gt_audio_lens = None
            state.all_predictions = []

        if not target_seen:
            raise RuntimeError(f"Selected assistant turn metadata_idx={target_metadata_idx} was not generated.")
        expected_user_turns = int(target_turn["user_turn_idx"])
        if user_turn_row != expected_user_turns:
            raise RuntimeError(
                "GT-prefix traversal consumed an unexpected number of user turns: "
                f"consumed={user_turn_row}, expected={expected_user_turns}."
            )
        expected_prefix_assistant_turns = int(target_turn["assistant_turn_idx"])
        if prefix_assistant_turns != expected_prefix_assistant_turns:
            raise RuntimeError(
                "GT-prefix traversal consumed an unexpected number of assistant turns: "
                f"consumed={prefix_assistant_turns}, expected={expected_prefix_assistant_turns}."
            )

        output = self.streaming_finalize(state, use_inference_mode=True)
        target_capacity = (
            int(target_turn["end_frame"]) - int(target_turn["start_frame"])
        ) * self.frame_stacking_factor
        if (output.audio_codes_len > target_capacity).any():
            raise RuntimeError(
                "Selected-turn rollout exceeded its target window: "
                f"lengths={output.audio_codes_len.tolist()}, capacity={target_capacity}."
            )
        target_turn["prefix_user_turns"] = user_turn_row
        target_turn["prefix_assistant_turns"] = prefix_assistant_turns
        return output

    def _prepare_user_audio_embeddings_for_po(self, batch: Dict) -> Optional[torch.Tensor]:
        """Restore packed user-turn embeddings onto the full replay timeline.

        The multi-turn dataset packs user waveforms across turns and records each
        turn's conversation index and sample boundaries. This method codec-encodes
        the packed waveforms and scatters their real (non-special) embeddings into
        a ``[1, timeline, hidden]`` tensor. Training-time silence/trim augmentation
        is intentionally omitted so rollout and teacher-forced PO replay share the
        same deterministic GT prefix.
        """
        if not self.cfg.get("condition_on_user_speech", False) or batch.get("user_audio_turn_splitted") is None:
            return None
        if batch["text"].size(0) != 1:
            raise ValueError("The minimal multi-turn GRPO replay requires train batch size 1.")

        user_audio = batch["user_audio_turn_splitted"]
        user_audio_lens = batch["user_audio_turn_splitted_lens"]
        user_codes, user_codes_lens = self._codec_helper.audio_to_codes(user_audio, user_audio_lens)
        if self._codec_converter is not None:
            user_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=user_codes, audio_lens=user_codes_lens
            ).long()
        user_codes, user_codes_lens = self.stack_codes(
            user_codes,
            user_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        packed_embeddings = self.embed_audio_tokens(user_codes)

        timeline_len = batch["text"].size(1)
        restored = packed_embeddings.new_zeros(1, timeline_len, packed_embeddings.size(-1))
        samples_per_step = self.codec_model_samples_per_frame * self.frame_stacking_factor
        indices = batch["user_audio_turn_splitted_indices"].to(packed_embeddings.device)
        for turn_idx, (batch_idx, start_sample, end_sample) in enumerate(indices):
            if int(batch_idx.item()) != 0:
                raise ValueError("Unexpected packed user-turn batch index in batch-size-1 replay.")
            start_frame = int(torch.ceil(start_sample.float() / samples_per_step).item())
            end_frame = int(end_sample.item()) // samples_per_step
            start_frame = max(0, min(start_frame, timeline_len))
            end_frame = max(start_frame, min(end_frame, timeline_len))
            copy_len = end_frame - start_frame
            if copy_len <= 0:
                continue
            turn_len = int(user_codes_lens[turn_idx].item())
            real_start = 1
            real_end = max(real_start, turn_len - 1)
            turn_embedding = packed_embeddings[turn_idx, real_start:real_end]
            copy_len = min(copy_len, turn_embedding.size(0))
            if copy_len > 0:
                restored[0, start_frame : start_frame + copy_len] = turn_embedding[:copy_len]
        return restored

    def _build_multiturn_po_replay(
        self,
        batch: Dict,
        target_turn: Dict,
        target_codes: torch.Tensor,
        target_codes_lens: torch.Tensor,
        num_generations: int,
    ) -> tuple[Dict, torch.Tensor, torch.Tensor]:
        """Build teacher-forced PO inputs for GT prefix plus sampled target audio.

        GT codec frames are copied only up to the target start. Each rollout's
        sampled target codes are inserted into the source turn's fixed frame window;
        shorter generations leave zero-filled future positions, which are causally
        after every optimized action. ``audio_action_mask`` marks only inserted
        sampled target timesteps, excluding GT prefix, user regions, padding, and
        unsampled suffixes from GRPO/KL/entropy terms.

        The replay batch copies only fields consumed by ``process_batch`` to avoid
        multiplying full conversation waveforms and packed-turn tensors by the
        rollout group size. User-prefix embeddings are detached before chunking
        because the prefix is conditioning state and would otherwise share one
        autograd graph across independent backward calls.
        """
        if batch["text"].size(0) != 1:
            raise ValueError("The minimal multi-turn GRPO replay requires train batch size 1.")

        if "audio_codes" in batch:
            gt_codes = batch["audio_codes"]
            gt_codes_lens = batch["audio_codes_lens"]
        else:
            gt_codes, gt_codes_lens = self._codec_helper.audio_to_codes(batch["audio"], batch["audio_lens"])
        del gt_codes_lens

        start_frame = int(target_turn["start_frame"])
        end_frame = int(target_turn["end_frame"])
        if end_frame <= start_frame:
            raise ValueError(f"Invalid target turn frame range: [{start_frame}, {end_frame}).")
        stacking_factor = int(self.frame_stacking_factor)
        prefix_raw_len = start_frame * stacking_factor
        total_raw_len = end_frame * stacking_factor
        target_raw_capacity = total_raw_len - prefix_raw_len

        replay_codes = gt_codes.new_zeros(num_generations, gt_codes.size(1), total_raw_len)
        prefix_copy_len = min(prefix_raw_len, gt_codes.size(-1))
        if prefix_copy_len > 0:
            replay_codes[:, :, :prefix_copy_len] = gt_codes[:, :, :prefix_copy_len].repeat(num_generations, 1, 1)

        action_mask = torch.zeros(
            num_generations,
            end_frame,
            dtype=torch.bool,
            device=target_codes.device,
        )
        for item_idx in range(num_generations):
            generated_len = min(int(target_codes_lens[item_idx].item()), target_raw_capacity)
            if generated_len > 0:
                replay_codes[
                    item_idx,
                    :,
                    prefix_raw_len : prefix_raw_len + generated_len,
                ] = target_codes[item_idx, :, :generated_len]
            generated_steps = min(
                (generated_len + stacking_factor - 1) // stacking_factor,
                end_frame - start_frame,
            )
            action_mask[item_idx, start_frame : start_frame + generated_steps] = True

        replay_lens = torch.full(
            (num_generations,),
            total_raw_len,
            dtype=torch.long,
            device=replay_codes.device,
        )
        replay_keys = {
            "text",
            "text_lens",
            "context_text_tokens",
            "context_text_tokens_lens",
            "context_audio",
            "context_audio_lens",
            "context_audio_codes",
            "context_audio_codes_lens",
            "phoneme_tokens",
            "phoneme_tokens_lens",
            "phoneme_turn_dropout",
            "task",
            "agent_mask",
            "agent_mask_lens",
            "raw_texts",
            "languages",
            "sample_id",
            "dataset_names",
        }
        replay_batch = self.repeat_items_in_batch(
            {key: value for key, value in batch.items() if key in replay_keys},
            num_generations,
        )
        replay_batch["raw_texts"] = [str(target_turn["text"])] * num_generations
        replay_batch["text"] = replay_batch["text"][:, : end_frame + 1]
        replay_batch["text_lens"] = torch.full(
            (num_generations,),
            replay_batch["text"].size(1),
            dtype=torch.long,
            device=replay_batch["text"].device,
        )
        if "phoneme_tokens" in replay_batch:
            replay_batch["phoneme_tokens"] = replay_batch["phoneme_tokens"][:, : end_frame + 1]
            replay_batch["phoneme_tokens_lens"] = torch.full(
                (num_generations,),
                replay_batch["phoneme_tokens"].size(1),
                dtype=torch.long,
                device=replay_batch["phoneme_tokens"].device,
            )
        replay_batch["agent_mask"] = replay_batch["agent_mask"][:, :end_frame]
        replay_batch["agent_mask_lens"] = torch.full(
            (num_generations,),
            end_frame,
            dtype=torch.long,
            device=replay_batch["agent_mask"].device,
        )
        replay_batch["audio_action_mask"] = action_mask

        user_audio_embedded = self._prepare_user_audio_embeddings_for_po(batch)
        if user_audio_embedded is not None:
            # The GT prefix is conditioning state, not a sampled action. Detach it before
            # repeating so independent teacher-forced chunks do not share one autograd graph.
            replay_batch["user_audio_embedded"] = user_audio_embedded.detach().repeat_interleave(
                num_generations, dim=0
            )
        return replay_batch, replay_codes, replay_lens

    def _unstack_rollout_phoneme_tokens(self, output) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert inference's offset stacked phoneme timeline into padded token sequences.

        ``infer_batch`` records predictions in a shared stacked timeline because items can
        enter and leave phoneme generation at different decoder steps. ``process_batch``
        instead expects one unstacked sequence per item, including BOS and EOS. Preserve
        exactly the actions sampled for each item so the teacher-forced likelihood is
        evaluated on the rollout trajectory rather than on GT or recomputed phonemes.
        """
        predicted = output.predicted_phoneme_tokens
        step_lens = output.predicted_phoneme_tokens_lens
        start_indices = output.phoneme_prediction_start_idx
        if predicted is None or step_lens is None or start_indices is None:
            raise RuntimeError("Predicted-phoneme rollouts did not return phoneme tokens, lengths, and start indices.")

        batch_size, stacking_factor, timeline_steps = predicted.shape
        if batch_size != step_lens.numel() or batch_size != start_indices.numel():
            raise RuntimeError(
                "Predicted-phoneme rollout metadata has inconsistent batch dimensions: "
                f"tokens={batch_size}, lengths={step_lens.numel()}, starts={start_indices.numel()}."
            )

        token_lens = step_lens.long() * stacking_factor + 1  # Include the autoregressive BOS input.
        max_token_len = int(token_lens.max().item())
        phoneme_tokens = torch.full(
            (batch_size, max_token_len),
            self.phoneme_tokenizer.eos_token_id,
            dtype=predicted.dtype,
            device=predicted.device,
        )
        phoneme_tokens[:, 0] = self.phoneme_tokenizer.bos_token_id

        for item_idx in range(batch_size):
            start = int(start_indices[item_idx].item())
            num_steps = int(step_lens[item_idx].item())
            end = start + num_steps
            if start < 0 or num_steps <= 0 or end > timeline_steps:
                raise RuntimeError(
                    "Invalid predicted-phoneme rollout span for item "
                    f"{item_idx}: start={start}, length={num_steps}, timeline={timeline_steps}."
                )
            # stack_codes groups consecutive source tokens into channels, so invert
            # that layout by traversing time first and stacked channel second.
            sampled_tokens = predicted[item_idx, :, start:end].transpose(0, 1).reshape(-1)
            phoneme_tokens[item_idx, 1 : 1 + sampled_tokens.numel()] = sampled_tokens

        return phoneme_tokens, token_lens

    def _get_audio_dir(self) -> str:
        """Return (and create if needed) the directory used to store intermediate waveforms during PO."""
        if not self.cfg.get('save_reward_audios', True):
            if not hasattr(self, '_tmp_reward_audio_dir'):
                job_id = os.environ.get('SLURM_JOB_ID', 'local')
                self._tmp_reward_audio_dir = os.path.join(
                    '/dev/shm', f'easy_magpie_rewards_{job_id}_rank{self.global_rank}'
                )
                os.makedirs(self._tmp_reward_audio_dir, exist_ok=True)
                atexit.register(shutil.rmtree, self._tmp_reward_audio_dir, ignore_errors=True)
            return self._tmp_reward_audio_dir

        if self.logger is not None and hasattr(self.logger, "log_dir") and self.logger.log_dir is not None:
            log_dir = self.logger.log_dir
        elif self.trainer is not None and self.trainer.log_dir is not None:
            log_dir = self.trainer.log_dir
        else:
            log_dir = "."
        audio_dir = os.path.join(log_dir, 'online_po_audios')
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir

    def _save_waveforms_to_paths(
        self,
        waveforms: torch.Tensor,
        waveform_lens: torch.Tensor,
        prefix: str,
        sample_rate: int,
    ) -> List[str]:
        """Write each waveform in the batch to a WAV file and return the list of file paths.

        Args:
            waveforms: Audio tensor of shape ``[B, T]``.
            waveform_lens: Per-item lengths of shape ``[B]``.
            prefix: Filename prefix (e.g. ``'generated'``, ``'reference_context_audio'``).
            sample_rate: Sampling rate written into the WAV header.

        Returns:
            List of absolute file paths, one per batch item.
        """
        audio_dir = self._get_audio_dir()
        paths = []
        for idx in range(waveforms.size(0)):
            wav = waveforms[idx].float().detach().cpu().numpy()
            wav = wav[: int(waveform_lens[idx].item())]
            # path = os.path.join(audio_dir, f'{prefix}_rank{self.global_rank}_{time_id}_{idx}.wav')
            path = os.path.join(audio_dir, f'{prefix}_rank{self.global_rank}_{idx}.wav')
            sf.write(path, wav, sample_rate)
            paths.append(path)
        return paths

    def _get_reference_audio_paths(self, batch_repeated: Dict) -> List[str]:
        """
        Build per-item reference audio paths for speaker similarity reward.
        Priority: audio_filepaths -> context_audio -> context_audio_codes.
        """
        if 'context_audio' in batch_repeated and 'context_audio_lens' in batch_repeated:
            # TODO: Handle text context here support here.
            return self._save_waveforms_to_paths(
                waveforms=batch_repeated['context_audio'],
                waveform_lens=batch_repeated['context_audio_lens'],
                prefix='reference_context_audio',
                sample_rate=self.sample_rate,
            )

        if 'context_audio_codes' in batch_repeated and 'context_audio_codes_lens' in batch_repeated:
            context_codes = batch_repeated['context_audio_codes'].clone()
            context_lens = batch_repeated['context_audio_codes_lens'].clone()

            target_codes = batch_repeated['audio_codes'].clone()
            target_lens = batch_repeated['audio_codes_lens'].clone()

            # For items where context_lens < 3, fall back to target_codes/target_lens
            # This is for items with text context
            short_context_mask = context_lens < 3
            if short_context_mask.any():
                # Pad the shorter tensor along the time dimension if needed
                max_len = max(context_codes.shape[-1], target_codes.shape[-1])
                if context_codes.shape[-1] < max_len:
                    pad_size = max_len - context_codes.shape[-1]
                    context_codes = torch.nn.functional.pad(context_codes, (0, pad_size), value=0)
                if target_codes.shape[-1] < max_len:
                    pad_size = max_len - target_codes.shape[-1]
                    target_codes = torch.nn.functional.pad(target_codes, (0, pad_size), value=0)
                context_codes[short_context_mask] = target_codes[short_context_mask]
                context_lens[short_context_mask] = target_lens[short_context_mask]
                # Slice to the actual max length needed
                context_codes = context_codes[..., : context_lens.max()]

            if self._codec_converter is not None:
                context_codes = self._codec_converter.convert_original_to_new(
                    audio_tokens=context_codes, audio_lens=context_lens
                ).long()
            context_codes, context_lens = self._prepare_codes_for_decode(context_codes, context_lens)
            context_audio, context_audio_lens, _ = self._codec_helper.codes_to_audio(
                context_codes,
                context_lens,
            )
            return self._save_waveforms_to_paths(
                waveforms=context_audio,
                waveform_lens=context_audio_lens,
                prefix='reference_context_codes_decoded',
                sample_rate=self.output_sample_rate,
            )

        raise ValueError(
            "Could not construct reference audio for speaker similarity. Need one of: "
            "context_audio/context_audio_lens, or context_audio_codes/context_audio_codes_lens."
        )

    def _run_easy_process_batch(
        self,
        model: EasyMagpieTTSModel,
        batch: Dict,
        audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        mode: str,
    ):
        """Run ``model.process_batch`` with the supplied audio codes, resolving context audio
        codes from the batch (either pre-computed or extracted on-the-fly from raw context audio).
        """
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio_codes, context_audio_codes_lens = model._codec_helper.audio_to_codes(
                batch['context_audio'], batch['context_audio_lens']
            )

        return model.process_batch(
            text=batch['text'],
            text_lens=batch['text_lens'],
            context_text_tokens=batch['context_text_tokens'],
            context_text_tokens_lens=batch['context_text_tokens_lens'],
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            phoneme_tokens=batch.get('phoneme_tokens'),
            phoneme_tokens_lens=batch.get('phoneme_tokens_lens'),
            phoneme_turn_dropout=batch.get('phoneme_turn_dropout'),
            mode=mode,
            task=batch.get('task'),
            agent_mask=batch.get('agent_mask'),
            user_audio_embedded=batch.get('user_audio_embedded'),
        )

    def _format_text_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format ``headers`` and ``rows`` into an aligned, pipe-delimited plain-text table string."""
        col_widths = [len(h) for h in headers]
        for row in rows:
            for col_idx, value in enumerate(row):
                col_widths[col_idx] = max(col_widths[col_idx], len(value))

        header_line = " | ".join(headers[col_idx].ljust(col_widths[col_idx]) for col_idx in range(len(headers)))
        separator = "-+-".join("-" * col_widths[col_idx] for col_idx in range(len(headers)))
        row_lines = [
            " | ".join(row[col_idx].ljust(col_widths[col_idx]) for col_idx in range(len(headers))) for row in rows
        ]
        return "\n".join([header_line, separator] + row_lines)

    def _print_group_cer_wer_table(
        self,
        batch: Dict,
        batch_metrics: List[Dict],
        group_idx: int,
        group_start_idx: int,
        group_end_idx: int,
        is_group_valid: bool,
        mean_reward: float,
        std_reward: float,
    ) -> None:
        """Log a per-generation metrics table (CER, WER, SSIM, UTMOS, reward, advantage) for one
        prompt group. Only runs on rank-zero.
        """
        if not getattr(self.trainer, "is_global_zero", True):
            return

        prompt_text = str(batch['raw_texts'][group_idx]).replace("\n", " ")
        if len(prompt_text) > 120:
            prompt_text = f"{prompt_text[:117]}..."
        languages = batch.get('languages', [])
        language = str(languages[group_idx]) if group_idx < len(languages) else "unknown"

        rows = []
        for local_idx, metric_idx in enumerate(range(group_start_idx, group_end_idx)):
            item_metrics = batch_metrics[metric_idx]
            rows.append(
                [
                    str(local_idx),
                    "on" if item_metrics.get('used_cfg', False) else "off",
                    f"{item_metrics['cer_gt']:.4f}",
                    f"{item_metrics['wer_gt']:.4f}",
                    f"{item_metrics['spk_similarity']:.4f}",
                    f"{item_metrics.get('utmos', 0.0):.4f}",
                    f"{item_metrics['reward']:.4f}",
                    f"{item_metrics.get('advantage', 0.0):.4f}",
                ]
            )

        table = self._format_text_table(
            headers=["item", "cfg", "cer", "wer", "ssim", "utmos", "reward", "advantage"], rows=rows
        )
        logging.info(
            f"[generate_and_reward] group={group_idx} language={language} valid={is_group_valid} "
            f"mean_reward={mean_reward:.4f} std_reward={std_reward:.4f}\n"
            f"prompt: {prompt_text}\n{table}\n"
        )

    def _compute_pred_transcripts(self, predicted_audio_paths: List[str], batch_repeated: Dict) -> List[str]:
        """Route reward transcription by language, then normalize consistently for CER/WER."""
        languages = list(batch_repeated.get('languages', ['en'] * len(predicted_audio_paths)))
        raw_transcripts = self._reward_asr_router.transcribe(predicted_audio_paths, languages)
        pred_transcripts = []
        logged_languages: Dict[str, int] = {}
        for item_idx, (language, transcript) in enumerate(zip(languages, raw_transcripts)):
            normalizer = self._get_cached_normalizer(language) if self._normalize_reward_transcript else None
            normalized = normalizer.normalize(transcript) if normalizer is not None else transcript
            pred_transcripts.append(process_text_for_cer(normalized))

            num_logged = logged_languages.get(language, 0)
            if num_logged < self.reward_asr_log_samples and getattr(self.trainer, "is_global_zero", True):
                gt_text = str(batch_repeated['raw_texts'][item_idx]).replace("\n", " ")
                logging.info(
                    f"[reward_asr_transcript] language={language} gt={gt_text[:240]!r} "
                    f"raw_pred={transcript[:240]!r} normalized_pred={pred_transcripts[-1][:240]!r}"
                )
                logged_languages[language] = num_logged + 1
        return pred_transcripts

    def _compute_speaker_embeddings_parallel(
        self, predicted_audio_paths: List[str], batch: Dict, num_generations_per_item: int
    ):
        """Extract speaker embeddings for both predicted and reference audio and align their batch
        dimensions so that cosine similarity can be computed element-wise.
        """
        reference_audio_paths = self._get_reference_audio_paths(batch)
        pred_speaker_embeddings = get_speaker_embeddings_from_filepaths(
            predicted_audio_paths, self._eval_speaker_verification_model, self.device
        )
        gt_speaker_embeddings = get_speaker_embeddings_from_filepaths(
            reference_audio_paths, self._eval_speaker_verification_model, self.device
        )
        if num_generations_per_item > 1:
            gt_speaker_embeddings = gt_speaker_embeddings.repeat_interleave(num_generations_per_item, dim=0)

        if gt_speaker_embeddings.size(0) != pred_speaker_embeddings.size(0):
            raise RuntimeError(
                f"Speaker embedding size mismatch. GT={gt_speaker_embeddings.size(0)}, "
                f"Pred={pred_speaker_embeddings.size(0)}."
            )
        return pred_speaker_embeddings, gt_speaker_embeddings

    def _compute_utmos_scores_batched(self, predicted_audio_paths: List[str]) -> List[float]:
        """Compute UTMOSv2 naturalness scores for the given audio files.

        Returns a list of zeros if UTMOS is disabled.
        """
        if not self.use_utmos:
            return [0.0] * len(predicted_audio_paths)
        if len(predicted_audio_paths) == 0:
            return []
        utmos_batch_size = max(int(self.cfg.get('utmos_batch_size', len(predicted_audio_paths))), 1)
        utmos_num_workers = max(int(self.cfg.get('utmos_num_workers', 0)), 0)
        target_device = self.device if self.utmos_device == 'cuda' else self.utmos_device
        self._utmos_calculator.to(target_device)
        audio_dir = self._get_audio_dir()
        val_list = [os.path.basename(p) for p in predicted_audio_paths]
        batch_results = self._utmos_calculator.process_directory(
            audio_dir, batch_size=utmos_batch_size, num_workers=utmos_num_workers, val_list=val_list
        )
        return [float(item['predicted_mos']) for item in batch_results]

    def generate_and_reward(
        self,
        batch: Dict,
        num_generations_per_item: int,
        mode: str = 'train',
        use_local_transformer_for_inference: bool = False,
    ):
        """Run autoregressive inference on the batch, compute multi-signal rewards
        (CER, speaker similarity, UTMOSv2), and return per-item advantages.

        This is the core rollout-then-reward step of the online PO pipeline.

        Returns:
            Dict containing mean/std rewards, per-item metrics, predicted codes,
            advantages, group validities, and timing information.
        """
        source_batch = batch
        multiturn_target = None
        if self.multiturn_grpo_enabled:
            multiturn_target = self._select_multiturn_target(batch, mode=mode)
            batch = dict(batch)
            batch["raw_texts"] = [str(multiturn_target["text"])]
            total_assistant_turns = sum(self._is_assistant_turn(turn) for turn in source_batch["turn_metadata"][0])
            sample_id = source_batch.get("sample_id", ["unknown"])[0]
            logging.info(
                "[multiturn_grpo_target] "
                f"sample_id={sample_id} "
                f"assistant_turn={int(multiturn_target['assistant_turn_idx']) + 1}/{total_assistant_turns} "
                f"metadata_idx={multiturn_target['metadata_idx']} "
                f"prefix_user_turns={multiturn_target['user_turn_idx']} "
                f"prefix_assistant_turns={multiturn_target['assistant_turn_idx']} "
                f"window={float(multiturn_target['start']):.2f}-{float(multiturn_target['end']):.2f}s "
                f"text={str(multiturn_target['text'])[:160]!r}"
            )
            batch_repeated = {
                "raw_texts": batch["raw_texts"] * num_generations_per_item,
                "languages": batch.get("languages", ["en"]) * num_generations_per_item,
                "sample_id": batch.get("sample_id", ["unknown"]) * num_generations_per_item,
            }
        else:
            batch_repeated = self.repeat_items_in_batch(batch, num_generations_per_item)
        rollout_cfg_mode = self.rollout_cfg_mode if mode == 'train' else 'off'

        phoneme_input_type = 'gt' if self.multiturn_grpo_enabled else 'pred'
        if not self.multiturn_grpo_enabled:
            gt_phoneme_input_prob = self.cfg.get('gt_phoneme_input_prob', 0.0)
            can_use_gt_phonemes = ('phoneme_tokens' in batch_repeated) and ('phoneme_tokens_lens' in batch_repeated)
            if can_use_gt_phonemes and gt_phoneme_input_prob > 0.0 and mode == 'train':
                phoneme_input_type = 'gt' if random.random() < gt_phoneme_input_prob else 'pred'

        generation_start_time = time.perf_counter()
        logging.info(f"Inference started (rollout_cfg_mode={rollout_cfg_mode})")
        if self.multiturn_grpo_enabled:
            output = self._generate_multiturn_target_rollouts(
                batch=source_batch,
                target_turn=multiturn_target,
                num_generations=num_generations_per_item,
                use_local_transformer_for_inference=use_local_transformer_for_inference,
            )
            rollout_cfg_mask = torch.zeros(len(batch_repeated['raw_texts']), device=self.device, dtype=torch.bool)
        else:

            def run_inference(inference_batch, use_cfg):
                return self.infer_batch(
                    batch=inference_batch,
                    max_decoder_steps=self.max_decoder_steps,
                    temperature=self.audio_sampling_temperature,
                    topk=self.audio_sampling_topk,
                    phoneme_temperature=self.phoneme_sampling_temperature,
                    phoneme_topk=self.phoneme_sampling_topk,
                    use_cfg=use_cfg,
                    cfg_scale=self.inference_cfg_scale,
                    use_local_transformer_for_inference=use_local_transformer_for_inference,
                    phoneme_input_type=phoneme_input_type,
                    phoneme_sampling_method=self.cfg.get('inference_phoneme_sampling_method', 'argmax'),
                    force_dropout_text=False,
                    use_teacher_forced=False,
                    use_inference_mode=False,
                )

            use_cfg = rollout_cfg_mode == 'alternate' and int(self.global_step) % 2 == 1
            output = run_inference(batch_repeated, use_cfg=use_cfg)
            rollout_cfg_mask = torch.full(
                (len(batch_repeated['raw_texts']),),
                fill_value=use_cfg,
                device=self.device,
                dtype=torch.bool,
            )
        logging.info("Inference ended")
        audio_generation_time_sec = time.perf_counter() - generation_start_time

        if self.multiturn_grpo_enabled:
            predicted_audio = output.audio
            predicted_audio_lens = output.audio_len
            predicted_codes = output.audio_codes
            predicted_codes_lens = output.audio_codes_len
        else:
            predicted_audio = output.predicted_audio
            predicted_audio_lens = output.predicted_audio_lens
            predicted_codes = output.predicted_codes
            predicted_codes_lens = output.predicted_codes_lens
        predicted_phoneme_tokens = None
        predicted_phoneme_tokens_lens = None
        if phoneme_input_type == 'pred':
            predicted_phoneme_tokens, predicted_phoneme_tokens_lens = self._unstack_rollout_phoneme_tokens(output)
        save_start_time = time.perf_counter()
        predicted_audio_paths = self._save_waveforms_to_paths(
            waveforms=predicted_audio,
            waveform_lens=predicted_audio_lens,
            prefix='generated',
            sample_rate=self.output_sample_rate,
        )
        audio_save_time_sec = time.perf_counter() - save_start_time
        audio_durations = [
            int(predicted_audio_lens[idx].item()) / self.output_sample_rate for idx in range(predicted_audio.size(0))
        ]

        rewarding_start_time = time.perf_counter()
        pred_transcripts = self._compute_pred_transcripts(predicted_audio_paths, batch_repeated)
        try:
            pred_speaker_embeddings, gt_speaker_embeddings = self._compute_speaker_embeddings_parallel(
                predicted_audio_paths, batch, num_generations_per_item
            )
        except Exception as e:
            logging.warning(f"Speaker-embedding reward failed. Falling back to zero SSIM reward. Error: {e}")
            pred_speaker_embeddings = None
            gt_speaker_embeddings = None
        utmos_scores = self._compute_utmos_scores_batched(predicted_audio_paths)

        batch_metrics = []
        cer_reward_weight = self.cfg.get('cer_reward_weight', 0.5)
        ssim_reward_weight = self.cfg.get('ssim_reward_weight', 0.5)
        utmos_reward_weight = self.cfg.get('utmos_reward_weight', 0.0)
        min_valid_codes_len = self.cfg.get('min_valid_codes_len', 4)
        max_valid_codes_len = self.cfg.get(
            'max_valid_codes_len', self.max_decoder_steps * self.frame_stacking_factor - 1
        )

        # UTMOSv2 reward shaping parameters (MOS scale is 1–5).
        mean_utmos_dataset = self.cfg.get('mean_utmos_dataset', 3.5)
        best_utmos_achievable = self.cfg.get('best_utmos_achievable', 4.5)
        languages = batch_repeated.get('languages', [])

        for idx in range(predicted_audio.size(0)):
            pred_transcript = pred_transcripts[idx]
            gt_transcript = process_text_for_cer(batch_repeated['raw_texts'][idx])
            cer_gt = min(max(word_error_rate([pred_transcript], [gt_transcript], use_cer=True), 0.0), 1.0)
            wer_gt = min(max(word_error_rate([pred_transcript], [gt_transcript], use_cer=False), 0.0), 1.0)

            if pred_speaker_embeddings is not None and gt_speaker_embeddings is not None:
                spk_embedding_pred = pred_speaker_embeddings[idx].cpu().float().numpy()
                spk_embedding_gt = gt_speaker_embeddings[idx].cpu().float().numpy()
                denom = max(np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt), 1e-8)
                spk_similarity = float(np.dot(spk_embedding_pred, spk_embedding_gt) / denom)
            else:
                spk_similarity = 0.0

            utmos_score = utmos_scores[idx]

            item_metrics = {
                'cer_gt': float(cer_gt),
                'wer_gt': float(wer_gt),
                'duration': float(audio_durations[idx]),
                'spk_similarity': float(spk_similarity),
                'pred_transcript': pred_transcript,
                'gt_transcript': gt_transcript,
                'codes_len': int(predicted_codes_lens[idx].item()),
                'utmos': float(utmos_score),
                'language': str(languages[idx]) if idx < len(languages) else "unknown",
                'used_cfg': bool(rollout_cfg_mask[idx].item()),
            }
            if multiturn_target is not None:
                item_metrics.update(
                    {
                        'target_assistant_turn': int(multiturn_target['assistant_turn_idx']) + 1,
                        'target_metadata_idx': int(multiturn_target['metadata_idx']),
                        'prefix_user_turns': int(multiturn_target['prefix_user_turns']),
                        'prefix_assistant_turns': int(multiturn_target['prefix_assistant_turns']),
                        'target_start_sec': float(multiturn_target['start']),
                        'target_end_sec': float(multiturn_target['end']),
                    }
                )

            best_ssim_achievable = self.cfg.get('best_ssim_achievable', 0.9)
            mean_cer_dataset = self.cfg.get('mean_cer_dataset', 0.1)
            mean_ssim_dataset = self.cfg.get('mean_ssim_dataset', 0.6)

            item_cer = item_metrics['cer_gt']
            item_ssim = max(min(item_metrics['spk_similarity'], best_ssim_achievable), 0.0)
            if item_cer <= mean_cer_dataset:
                cer_reward = 0.5 + 0.5 * (mean_cer_dataset - item_cer) / max(mean_cer_dataset, 1e-8)
            else:
                cer_reward = 0.5 - 0.5 * (item_cer - mean_cer_dataset) / max(1.0 - mean_cer_dataset, 1e-8)

            if item_ssim >= mean_ssim_dataset:
                spk_similarity_reward = 0.5 + 0.5 * (item_ssim - mean_ssim_dataset) / max(
                    best_ssim_achievable - mean_ssim_dataset, 1e-8
                )
            else:
                spk_similarity_reward = 0.5 - 0.5 * (mean_ssim_dataset - item_ssim) / max(mean_ssim_dataset, 1e-8)

            # UTMOSv2 reward: piecewise linear shaping centered on mean_utmos_dataset,
            # analogous to the CER and SSIM reward shaping.
            if self.use_utmos:
                item_utmos = max(min(utmos_score, best_utmos_achievable), 1.0)
                if item_utmos >= mean_utmos_dataset:
                    utmos_reward = 0.5 + 0.5 * (item_utmos - mean_utmos_dataset) / max(
                        best_utmos_achievable - mean_utmos_dataset, 1e-8
                    )
                else:
                    utmos_reward = 0.5 - 0.5 * (mean_utmos_dataset - item_utmos) / max(mean_utmos_dataset - 1.0, 1e-8)
            else:
                utmos_reward = 0.0

            reward = (
                cer_reward * cer_reward_weight
                + spk_similarity_reward * ssim_reward_weight
                + utmos_reward * utmos_reward_weight
            )
            if (item_metrics['codes_len'] >= max_valid_codes_len) or (
                item_metrics['codes_len'] <= min_valid_codes_len
            ):
                item_metrics['_needs_group_min_reward'] = True
            else:
                item_metrics['_needs_group_min_reward'] = False

            item_metrics['cer_reward'] = float(cer_reward)
            item_metrics['spk_similarity_reward'] = float(spk_similarity_reward)
            item_metrics['utmos_reward'] = float(utmos_reward)
            item_metrics['reward'] = float(reward)
            batch_metrics.append(item_metrics)

        # Second pass: replace rewards for items with invalid code lengths with the group minimum reward
        num_groups = len(batch['raw_texts'])
        for group_idx in range(num_groups):
            group_start_idx = group_idx * num_generations_per_item
            group_end_idx = group_start_idx + num_generations_per_item
            group_rewards = [batch_metrics[idx]['reward'] for idx in range(group_start_idx, group_end_idx)]
            group_min_reward = min(group_rewards)
            for idx in range(group_start_idx, group_end_idx):
                if batch_metrics[idx]['_needs_group_min_reward']:
                    batch_metrics[idx]['reward'] = float(group_min_reward)

        all_groups_mean_reward = 0.0
        all_groups_std_reward = 0.0
        group_validities = []
        for group_idx in range(num_groups):
            group_start_idx = group_idx * num_generations_per_item
            group_end_idx = group_start_idx + num_generations_per_item
            group_rewards = [batch_metrics[idx]['reward'] for idx in range(group_start_idx, group_end_idx)]
            group_cers = [batch_metrics[idx]['cer_gt'] for idx in range(group_start_idx, group_end_idx)]
            mean_reward = float(np.mean(group_rewards))
            std_reward = float(np.std(group_rewards))
            is_group_valid = True
            if min(group_cers) > self.best_cer_threshold:
                is_group_valid = False
            if max(group_cers) > self.worst_cer_threshold:
                is_group_valid = False

            for idx in range(group_start_idx, group_end_idx):
                advantage = batch_metrics[idx]['reward'] - mean_reward
                if self.scale_rewards:
                    advantage = advantage / (std_reward + 1e-4)
                batch_metrics[idx]['advantage'] = float(advantage)
                group_validities.append(is_group_valid)

            self._print_group_cer_wer_table(
                batch=batch,
                batch_metrics=batch_metrics,
                group_idx=group_idx,
                group_start_idx=group_start_idx,
                group_end_idx=group_end_idx,
                is_group_valid=is_group_valid,
                mean_reward=mean_reward,
                std_reward=std_reward,
            )

            all_groups_mean_reward += mean_reward
            all_groups_std_reward += std_reward

        all_groups_mean_reward = all_groups_mean_reward / max(num_groups, 1)
        all_groups_std_reward = all_groups_std_reward / max(num_groups, 1)
        advantages = torch.tensor([x['advantage'] for x in batch_metrics], device=self.device, dtype=torch.float32)
        group_validities = torch.tensor(group_validities, device=self.device, dtype=torch.float32)
        cfg_rewards = [item['reward'] for item in batch_metrics if item['used_cfg']]
        no_cfg_rewards = [item['reward'] for item in batch_metrics if not item['used_cfg']]
        rewarding_time_sec = time.perf_counter() - rewarding_start_time

        return {
            'mean_reward': torch.tensor(all_groups_mean_reward, device=self.device, dtype=torch.float32),
            'std_reward': torch.tensor(all_groups_std_reward, device=self.device, dtype=torch.float32),
            'batch_repeated': batch_repeated,
            'metrics': batch_metrics,
            'predicted_codes': predicted_codes,
            'predicted_codes_lens': predicted_codes_lens,
            'predicted_phoneme_tokens': predicted_phoneme_tokens,
            'predicted_phoneme_tokens_lens': predicted_phoneme_tokens_lens,
            'advantages': advantages,
            'group_validities': group_validities,
            'cfg_fraction': rollout_cfg_mask.float().mean(),
            'mean_reward_cfg': (
                torch.tensor(np.mean(cfg_rewards), device=self.device, dtype=torch.float32) if cfg_rewards else None
            ),
            'mean_reward_no_cfg': (
                torch.tensor(np.mean(no_cfg_rewards), device=self.device, dtype=torch.float32)
                if no_cfg_rewards
                else None
            ),
            'rollout_phoneme_input_type': phoneme_input_type,
            'multiturn_target': multiturn_target,
            'timings': {
                'audio_generation_time_sec': float(audio_generation_time_sec),
                'audio_save_time_sec': float(audio_save_time_sec),
                'rewarding_time_sec': float(rewarding_time_sec),
            },
        }

    def process_batch_online_po(self, batch: Dict, n_generations_per_item: int, mode: str = 'train'):
        """End-to-end online PO forward pass: generate rollouts, score rewards, and compute PO +
        auxiliary losses *without* performing a backward pass (useful for validation).
        """
        generated_codes_and_metrics, batch_repeated, predicted_codes, predicted_codes_lens = (
            self._prepare_online_po_inputs(
                batch=batch,
                n_generations_per_item=n_generations_per_item,
                mode=mode,
            )
        )
        chunked_outputs = self._run_teacher_forced_chunked_po(
            generated_codes_and_metrics=generated_codes_and_metrics,
            batch_repeated=batch_repeated,
            predicted_codes=predicted_codes,
            predicted_codes_lens=predicted_codes_lens,
            n_generations_per_item=n_generations_per_item,
            do_backward=False,
        )
        return {
            'mean_reward': generated_codes_and_metrics['mean_reward'],
            'std_reward': generated_codes_and_metrics['std_reward'],
            'loss': chunked_outputs['loss'],
            'po_loss': chunked_outputs['po_loss'],
            'audio_po_loss': chunked_outputs['audio_po_loss'],
            'phoneme_po_loss': chunked_outputs['phoneme_po_loss'],
            'phoneme_aux_loss': chunked_outputs['phoneme_aux_loss'],
            'kl_loss': chunked_outputs['kl_loss'],
            'used_gt_phoneme_input': chunked_outputs['used_gt_phoneme_input'],
            'batch_metrics': generated_codes_and_metrics['metrics'],
        }

    def _slice_batch_range(self, batch: Dict, start_idx: int, end_idx: int) -> Dict:
        """Slice ``batch`` along the batch dimension from ``start_idx`` to ``end_idx``, and trim
        temporal tensors to the local maximum length to reduce memory during chunked processing.
        """
        sliced_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                sliced_batch[key] = value[start_idx:end_idx]
            elif isinstance(value, list):
                sliced_batch[key] = value[start_idx:end_idx]
            else:
                sliced_batch[key] = value

        # Keep explicit keys only to avoid accidental slicing of non-temporal tensors.
        temporal_key_pairs = [
            ('text', 'text_lens'),
            ('context_text_tokens', 'context_text_tokens_lens'),
            ('audio_codes', 'audio_codes_lens'),
            ('context_audio_codes', 'context_audio_codes_lens'),
            ('phoneme_tokens', 'phoneme_tokens_lens'),
            ('context_audio', 'context_audio_lens'),
            ('audio', 'audio_lens'),
        ]
        for tensor_key, lens_key in temporal_key_pairs:
            tensor_value = sliced_batch.get(tensor_key)
            lens = sliced_batch.get(lens_key)
            if not isinstance(tensor_value, torch.Tensor) or not isinstance(lens, torch.Tensor):
                continue
            if tensor_value.dim() < 2 or tensor_value.size(0) != lens.size(0):
                continue

            local_max_len = int(lens.max().item()) if lens.numel() > 0 else 0
            local_max_len = min(local_max_len, tensor_value.size(-1))
            sliced_batch[tensor_key] = tensor_value[..., :local_max_len]

        return sliced_batch

    def _iter_group_ranges(self, num_groups: int, groups_per_subbatch: int):
        """Yield ``(start, end)`` index pairs that partition ``num_groups`` into sub-batches."""
        for group_start in range(0, num_groups, groups_per_subbatch):
            yield group_start, min(group_start + groups_per_subbatch, num_groups)

    def _prepare_online_po_inputs(self, batch: Dict, n_generations_per_item: int, mode: str):
        """Generate rollouts with rewards and prepare the inputs needed for teacher-forced PO.

        Runs ``generate_and_reward`` in eval / no-grad mode, converts the predicted codes back
        to the original codec format, and returns the metrics dict alongside the repeated batch,
        predicted codes, and their lengths.
        """
        use_local_transformer_for_inference = False
        use_local_transformer_prob = self.cfg.get('use_local_transformer_prob', 0.0)
        if use_local_transformer_prob > 0.0 and mode == 'train':
            use_local_transformer_for_inference = random.random() < use_local_transformer_prob

        with torch.no_grad():
            self.eval()
            generated_codes_and_metrics = self.generate_and_reward(
                batch=batch,
                num_generations_per_item=n_generations_per_item,
                mode=mode,
                use_local_transformer_for_inference=use_local_transformer_for_inference,
            )
            self.train()

        batch_repeated = generated_codes_and_metrics['batch_repeated']
        predicted_codes = generated_codes_and_metrics['predicted_codes']
        predicted_codes_lens = generated_codes_and_metrics['predicted_codes_lens']
        predicted_codes = predicted_codes[:, :, : predicted_codes_lens.max()]
        predicted_codes = self._codec_converter.convert_new_to_original(
            audio_tokens=predicted_codes, audio_lens=predicted_codes_lens
        )
        if self.multiturn_grpo_enabled:
            target_turn = generated_codes_and_metrics.get("multiturn_target")
            if target_turn is None:
                raise RuntimeError("Multi-turn rollout did not return selected target metadata.")
            batch_repeated, predicted_codes, predicted_codes_lens = self._build_multiturn_po_replay(
                batch=batch,
                target_turn=target_turn,
                target_codes=predicted_codes,
                target_codes_lens=predicted_codes_lens,
                num_generations=n_generations_per_item,
            )
        batch_repeated['audio_codes'] = predicted_codes
        batch_repeated['audio_codes_lens'] = predicted_codes_lens
        if generated_codes_and_metrics.get('rollout_phoneme_input_type') == 'pred':
            batch_repeated['phoneme_tokens'] = generated_codes_and_metrics['predicted_phoneme_tokens']
            batch_repeated['phoneme_tokens_lens'] = generated_codes_and_metrics['predicted_phoneme_tokens_lens']
        if 'audio' in batch_repeated:
            del batch_repeated['audio']
        if 'audio_lens' in batch_repeated:
            del batch_repeated['audio_lens']

        return generated_codes_and_metrics, batch_repeated, predicted_codes, predicted_codes_lens

    def _compute_action_po_components(
        self,
        logits: torch.Tensor,
        reference_logits: Optional[torch.Tensor],
        targets: torch.Tensor,
        target_lens: torch.Tensor,
        vocab_size: int,
        advantages: torch.Tensor,
        group_validities: torch.Tensor,
        sampling_temperature: Optional[float] = None,
        forbidden_token_ids: Optional[List[int]] = None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        """Compute normalized GRPO, exact forward KL, and entropy for parallel action streams.

        KL is evaluated over the complete policy/reference distributions rather than
        estimated from only the sampled action. The sampled-action ``k3`` estimator
        (``exp(log p_ref - log p_policy) - log p_ref + log p_policy - 1``) is
        non-negative, but its exponential importance ratio has an unbounded
        heavy tail. A single rare token can therefore dominate an otherwise
        healthy batch. Since this method already materializes the full policy
        distribution for entropy, exact ``KL(policy || reference)`` only requires
        one additional reference log-softmax per stream and avoids that variance.

        Both the GRPO and KL terms are masked by ``group_validities``. A group
        rejected as reward-uninformative must not update the policy through a
        hidden KL-only path.
        """
        loss_mask = get_mask_from_lengths(target_lens, x=targets[:, 0, :])
        if action_mask is not None:
            if action_mask.dim() != 2 or action_mask.size(0) != targets.size(0):
                raise ValueError(
                    f"Expected action_mask [B,T], got {tuple(action_mask.shape)} for targets "
                    f"{tuple(targets.shape)}."
                )
            if action_mask.size(1) < targets.size(-1):
                action_mask = torch.nn.functional.pad(
                    action_mask, (0, targets.size(-1) - action_mask.size(1)), value=False
                )
            action_mask = action_mask[:, : targets.size(-1)].to(device=targets.device, dtype=torch.bool)
            loss_mask = loss_mask.bool() & action_mask
        loss_mask = loss_mask.float()
        group_mask = group_validities.float().unsqueeze(1)
        num_streams = targets.size(1)
        total_po_loss = logits.new_zeros((), dtype=torch.float32)
        total_kl = logits.new_zeros((), dtype=torch.float32)
        total_entropy = logits.new_zeros((), dtype=torch.float32)

        for stream_idx in range(num_streams):
            si = stream_idx * vocab_size
            ei = si + vocab_size
            stream_logits = logits[:, :, si:ei]
            if sampling_temperature is not None:
                stream_logits = self._apply_teacher_forced_transform(
                    stream_logits,
                    temperature=sampling_temperature,
                    forbidden_token_ids=forbidden_token_ids,
                )
            stream_labels = targets[:, stream_idx, :].long()
            per_token_logps = self._get_per_token_logps(stream_logits, stream_labels, loss_mask)
            valid_logps = per_token_logps[loss_mask.bool()]
            if not torch.isfinite(valid_logps).all():
                raise FloatingPointError(
                    "A valid sampled action has a non-finite teacher-forced log-probability. "
                    "The rollout action may be outside the recomputed policy support."
                )

            # This on-policy surrogate is value-identical to -advantage, while its
            # gradient is -advantage * grad(log pi(action)).
            with torch.cuda.amp.autocast(enabled=False):
                per_token_po_loss = -(
                    torch.exp(per_token_logps.float() - per_token_logps.float().detach())
                    * advantages.float().unsqueeze(1)
                )
                per_token_po_loss = per_token_po_loss * group_mask

                logits_fp32 = stream_logits.float()
                log_probs = logits_fp32.log_softmax(-1)
                probs = log_probs.exp()
                per_token_entropy = -torch.xlogy(probs, probs).sum(-1)

            stream_entropy = ((per_token_entropy * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1e-8)).mean()

            grpo_beta = float(self.cfg.get('grpo_beta', 0.0))
            if not self.reference_free and reference_logits is not None and grpo_beta > 0.0:
                with torch.no_grad():
                    ref_stream_logits = reference_logits[:, :, si:ei]
                    if sampling_temperature is not None:
                        # Policy and reference must use identical temperature and
                        # token support for a meaningful distribution-level KL.
                        ref_stream_logits = self._apply_teacher_forced_transform(
                            ref_stream_logits,
                            temperature=sampling_temperature,
                            forbidden_token_ids=forbidden_token_ids,
                        )
                    ref_log_probs = ref_stream_logits.float().log_softmax(-1)

                with torch.cuda.amp.autocast(enabled=False):
                    # Exact forward KL: sum_a p_policy(a) *
                    # (log p_policy(a) - log p_reference(a)). Forbidden tokens
                    # are -inf under both distributions; replace their undefined
                    # (-inf - -inf) log-ratio with zero before multiplying by the
                    # policy probability.
                    log_ratio = torch.nan_to_num(
                        log_probs - ref_log_probs,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    per_token_kl = (probs * log_ratio).sum(dim=-1).clamp_min(0.0)
                    per_token_kl = per_token_kl * group_mask

                stream_kl = ((per_token_kl * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1e-8)).mean()
            else:
                stream_kl = logits.new_zeros((), dtype=torch.float32)

            if self.loss_type == "grpo":
                stream_po_loss = (
                    (per_token_po_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1e-8)
                ).mean()
            elif self.loss_type == "dr_grpo":
                total_tokens = per_token_po_loss.shape[0] * self.max_decoder_steps
                stream_po_loss = (per_token_po_loss * loss_mask).sum() / max(total_tokens, 1)
            else:
                raise ValueError(f"Unknown loss function: {self.loss_type}")

            total_po_loss = total_po_loss + stream_po_loss
            total_kl = total_kl + stream_kl
            total_entropy = total_entropy + stream_entropy

        return total_po_loss / num_streams, total_kl / num_streams, total_entropy / num_streams

    def _compute_po_losses_from_outputs(
        self,
        policy_output,
        reference_output,
        advantages: torch.Tensor,
        group_validities: torch.Tensor,
        rollout_phoneme_input_type: str,
        audio_action_mask: Optional[torch.Tensor] = None,
    ):
        """Compute joint audio/phoneme GRPO and optional GT-phoneme auxiliary loss."""
        audio_logits = policy_output.local_transformer_logits
        if audio_logits is None:
            audio_logits = policy_output.logits
        reference_audio_logits = None
        if reference_output is not None:
            reference_audio_logits = reference_output.local_transformer_logits
            if reference_audio_logits is None:
                reference_audio_logits = reference_output.logits

        audio_po_loss, audio_kl, audio_entropy = self._compute_action_po_components(
            logits=audio_logits,
            reference_logits=reference_audio_logits,
            targets=policy_output.audio_codes_target,
            target_lens=policy_output.audio_codes_lens_target,
            vocab_size=self.num_all_tokens_per_codebook,
            advantages=advantages,
            group_validities=group_validities,
            sampling_temperature=self.audio_sampling_temperature,
            forbidden_token_ids=SpecialAudioToken.get_forbidden_tokens(self.codebook_size, forbid_audio_eos=False),
            action_mask=audio_action_mask,
        )

        zero = audio_logits.new_zeros((), dtype=torch.float32)
        phoneme_po_loss = zero
        phoneme_kl = zero
        phoneme_entropy = zero
        if rollout_phoneme_input_type == 'pred' and self.phoneme_po_loss_weight > 0.0:
            if (
                policy_output.phoneme_logits is None
                or policy_output.phoneme_tokens_target is None
                or policy_output.phoneme_tokens_lens_target is None
            ):
                raise RuntimeError("Predicted-phoneme PO requires phoneme logits, targets, and target lengths.")
            reference_phoneme_logits = reference_output.phoneme_logits if reference_output is not None else None
            phoneme_po_loss, phoneme_kl, phoneme_entropy = self._compute_action_po_components(
                logits=policy_output.phoneme_logits,
                reference_logits=reference_phoneme_logits,
                targets=policy_output.phoneme_tokens_target,
                target_lens=policy_output.phoneme_tokens_lens_target,
                vocab_size=self.phoneme_vocab_size,
                advantages=advantages,
                group_validities=group_validities,
                sampling_temperature=self.phoneme_sampling_temperature,
            )

        # GT phonemes are supervised targets, not sampled policy actions. Retain the
        # existing auxiliary objective only for GT-conditioned rollouts.
        phoneme_aux_loss = policy_output.phoneme_loss if rollout_phoneme_input_type == 'gt' else zero
        if phoneme_aux_loss is None:
            phoneme_aux_loss = zero

        po_loss = audio_po_loss + self.phoneme_po_loss_weight * phoneme_po_loss
        kl_loss = audio_kl + self.phoneme_po_loss_weight * phoneme_kl
        entropy = audio_entropy + self.phoneme_po_loss_weight * phoneme_entropy
        # Keep policy and KL losses separate for logging. ``train_kl_loss`` is
        # the raw exact KL; only ``grpo_beta * kl_loss`` contributes to the
        # optimized objective.
        grpo_beta = float(self.cfg.get('grpo_beta', 0.0))
        total_loss = po_loss + grpo_beta * kl_loss + self.aux_phoneme_loss_weight * phoneme_aux_loss
        if self.entropy_coeff > 0:
            total_loss = total_loss - self.entropy_coeff * entropy

        return {
            'loss': total_loss,
            'po_loss': po_loss,
            'audio_po_loss': audio_po_loss,
            'phoneme_po_loss': phoneme_po_loss,
            'phoneme_aux_loss': phoneme_aux_loss,
            'kl_loss': kl_loss,
            'audio_kl_loss': audio_kl,
            'phoneme_kl_loss': phoneme_kl,
            'entropy': entropy,
            'audio_entropy': audio_entropy,
            'phoneme_entropy': phoneme_entropy,
            'used_gt_phoneme_input': float(rollout_phoneme_input_type == 'gt'),
        }

    def _run_teacher_forced_chunked_po(
        self,
        generated_codes_and_metrics: Dict,
        batch_repeated: Dict,
        predicted_codes: torch.Tensor,
        predicted_codes_lens: torch.Tensor,
        n_generations_per_item: int,
        do_backward: bool,
    ):
        """Run teacher-forced PO forward (and optionally backward) in memory-friendly chunks.

        The batch is split into sub-batches of size ``batch_size_for_chunked_tf``. Each chunk's
        loss is weighted proportionally and, when ``do_backward`` is ``True``, gradients are
        accumulated via ``manual_backward``.

        Returns:
            Dict of accumulated (weighted-average) loss components across all chunks.
        """
        total_items = len(batch_repeated['raw_texts'])
        if self.batch_size_for_chunked_tf is not None:
            chunk_size = self.batch_size_for_chunked_tf
        else:
            # Backward compatibility: preserve previous effective item-chunk size
            # when the new explicit batch-size chunking config is not set.
            chunk_size = max(self.po_groups_per_subbatch, 1) * max(n_generations_per_item, 1)
        chunk_size = max(int(chunk_size), 1)

        accumulated_loss = torch.tensor(0.0, device=self.device)
        accumulated_po_loss = torch.tensor(0.0, device=self.device)
        accumulated_audio_po_loss = torch.tensor(0.0, device=self.device)
        accumulated_phoneme_po_loss = torch.tensor(0.0, device=self.device)
        accumulated_phoneme_aux_loss = torch.tensor(0.0, device=self.device)
        accumulated_kl_loss = torch.tensor(0.0, device=self.device)
        accumulated_audio_kl_loss = torch.tensor(0.0, device=self.device)
        accumulated_phoneme_kl_loss = torch.tensor(0.0, device=self.device)
        accumulated_entropy = torch.tensor(0.0, device=self.device)
        accumulated_audio_entropy = torch.tensor(0.0, device=self.device)
        accumulated_phoneme_entropy = torch.tensor(0.0, device=self.device)
        used_gt_phoneme_input = 0.0

        for item_start_idx in range(0, total_items, chunk_size):
            item_end_idx = min(item_start_idx + chunk_size, total_items)
            chunk_weight = float(item_end_idx - item_start_idx) / max(float(total_items), 1.0)

            batch_sub = self._slice_batch_range(batch_repeated, item_start_idx, item_end_idx)
            predicted_codes_sub = predicted_codes[item_start_idx:item_end_idx]
            predicted_codes_lens_sub = predicted_codes_lens[item_start_idx:item_end_idx]
            predicted_codes_sub = predicted_codes_sub[:, :, : predicted_codes_lens_sub.max()]
            advantages_sub = generated_codes_and_metrics['advantages'][item_start_idx:item_end_idx]
            group_validities_sub = generated_codes_and_metrics['group_validities'][item_start_idx:item_end_idx]
            rollout_phoneme_input_type = generated_codes_and_metrics.get('rollout_phoneme_input_type', 'pred')

            # Use mode='val' intentionally for stable PO optimization:
            # no random input dropout, no CFG unconditional dropout, no random phoneme corruption.
            policy_output = self._run_easy_process_batch(
                model=self,
                batch=batch_sub,
                audio_codes=predicted_codes_sub,
                audio_codes_lens=predicted_codes_lens_sub,
                mode='val',
            )

            reference_output = None
            if not self.reference_free:
                with torch.no_grad():
                    reference_output = self._run_easy_process_batch(
                        model=self._reference_model,
                        batch=batch_sub,
                        audio_codes=predicted_codes_sub,
                        audio_codes_lens=predicted_codes_lens_sub,
                        mode='val',
                    )

            chunk_outputs = self._compute_po_losses_from_outputs(
                policy_output=policy_output,
                reference_output=reference_output,
                advantages=advantages_sub,
                group_validities=group_validities_sub,
                rollout_phoneme_input_type=rollout_phoneme_input_type,
                audio_action_mask=batch_sub.get("audio_action_mask"),
            )

            if do_backward:
                if not torch.isfinite(chunk_outputs['loss']).all():
                    raise FloatingPointError(
                        f"Non-finite PO loss before backward for items [{item_start_idx}, {item_end_idx})."
                    )
                self.manual_backward(chunk_outputs['loss'] * chunk_weight)

            accumulated_loss = accumulated_loss + chunk_outputs['loss'].detach() * chunk_weight
            accumulated_po_loss = accumulated_po_loss + chunk_outputs['po_loss'].detach() * chunk_weight
            accumulated_audio_po_loss = (
                accumulated_audio_po_loss + chunk_outputs['audio_po_loss'].detach() * chunk_weight
            )
            accumulated_phoneme_po_loss = (
                accumulated_phoneme_po_loss + chunk_outputs['phoneme_po_loss'].detach() * chunk_weight
            )
            accumulated_phoneme_aux_loss = (
                accumulated_phoneme_aux_loss + chunk_outputs['phoneme_aux_loss'].detach() * chunk_weight
            )
            accumulated_kl_loss = accumulated_kl_loss + chunk_outputs['kl_loss'].detach() * chunk_weight
            accumulated_audio_kl_loss = (
                accumulated_audio_kl_loss + chunk_outputs['audio_kl_loss'].detach() * chunk_weight
            )
            accumulated_phoneme_kl_loss = (
                accumulated_phoneme_kl_loss + chunk_outputs['phoneme_kl_loss'].detach() * chunk_weight
            )
            accumulated_entropy = accumulated_entropy + chunk_outputs['entropy'].detach() * chunk_weight
            accumulated_audio_entropy = (
                accumulated_audio_entropy + chunk_outputs['audio_entropy'].detach() * chunk_weight
            )
            accumulated_phoneme_entropy = (
                accumulated_phoneme_entropy + chunk_outputs['phoneme_entropy'].detach() * chunk_weight
            )
            used_gt_phoneme_input = max(used_gt_phoneme_input, chunk_outputs['used_gt_phoneme_input'])

        return {
            'loss': accumulated_loss,
            'po_loss': accumulated_po_loss,
            'audio_po_loss': accumulated_audio_po_loss,
            'phoneme_po_loss': accumulated_phoneme_po_loss,
            'phoneme_aux_loss': accumulated_phoneme_aux_loss,
            'kl_loss': accumulated_kl_loss,
            'audio_kl_loss': accumulated_audio_kl_loss,
            'phoneme_kl_loss': accumulated_phoneme_kl_loss,
            'entropy': accumulated_entropy,
            'audio_entropy': accumulated_audio_entropy,
            'phoneme_entropy': accumulated_phoneme_entropy,
            'used_gt_phoneme_input': used_gt_phoneme_input,
        }

    def _update_dataset_training_step(self) -> None:
        dataset = getattr(getattr(self, "_train_dl", None), "dataset", None)
        while dataset is not None:
            if hasattr(dataset, "set_training_step"):
                dataset.set_training_step(int(self.global_step))
                return
            dataset = getattr(dataset, "dataset", None)

    def training_step(self, batch, batch_idx):
        """Execute one full online PO training iteration: rollout generation, reward computation,
        chunked teacher-forced forward/backward, gradient clipping, optimizer step, LR scheduling,
        and logging of all training metrics and diagnostics.
        """
        self._update_dataset_training_step()
        n_generations_per_item = self.cfg.get('n_generations_per_item', 6)
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != 1:
                raise ValueError(f"Expected a single optimizer, got {len(optimizer)}.")
            optimizer = optimizer[0]
        optimizer.zero_grad(set_to_none=True)

        # Snapshot weights before optimizer step to measure weight deltas.
        prev_weights = self._snapshot_trainable_weights()

        generated_codes_and_metrics, batch_repeated, predicted_codes, predicted_codes_lens = (
            self._prepare_online_po_inputs(
                batch=batch,
                n_generations_per_item=n_generations_per_item,
                mode='train',
            )
        )
        teacher_forced_start_time = time.perf_counter()
        po_outputs = self._run_teacher_forced_chunked_po(
            generated_codes_and_metrics=generated_codes_and_metrics,
            batch_repeated=batch_repeated,
            predicted_codes=predicted_codes,
            predicted_codes_lens=predicted_codes_lens,
            n_generations_per_item=n_generations_per_item,
            do_backward=True,
        )
        teacher_forced_time_sec = time.perf_counter() - teacher_forced_start_time

        # Clip gradients to prevent catastrophic updates from outlier batches.
        # Even when clipping is disabled, compute the norm with error_if_nonfinite
        # so an invalid update cannot corrupt the policy or its next checkpoint.
        max_grad_norm = self.cfg.get('max_grad_norm', 0.0)
        params_with_grad = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
        torch.nn.utils.clip_grad_norm_(
            params_with_grad,
            max_norm=max_grad_norm if max_grad_norm > 0 else float('inf'),
            error_if_nonfinite=True,
        )

        # Compute gradient/weight metrics AFTER clipping but BEFORE optimizer.step() clears them.
        grad_weight_metrics = self._compute_grad_and_weight_metrics()

        optimizer.step()

        # Step the LR scheduler (required in manual optimization mode).
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            if isinstance(lr_schedulers, (list, tuple)):
                for sched in lr_schedulers:
                    sched.step()
            else:
                lr_schedulers.step()

        # Compute weight delta metrics AFTER optimizer.step().
        grad_weight_metrics.update(self._compute_weight_update_metrics(prev_weights))

        # Log learning rate.
        self.log('learning_rate', optimizer.param_groups[0]['lr'], prog_bar=False, sync_dist=True)

        # Core training metrics.
        self.log('train_loss', po_outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train_po_loss', po_outputs['po_loss'], prog_bar=True, sync_dist=True)
        self.log('train_audio_po_loss', po_outputs['audio_po_loss'], prog_bar=False, sync_dist=True)
        self.log('train_phoneme_po_loss', po_outputs['phoneme_po_loss'], prog_bar=True, sync_dist=True)
        self.log('train_phoneme_aux_loss', po_outputs['phoneme_aux_loss'], prog_bar=True, sync_dist=True)
        self.log('train_kl_loss', po_outputs['kl_loss'], prog_bar=True, sync_dist=True)
        self.log('train_audio_kl_loss', po_outputs['audio_kl_loss'], prog_bar=False, sync_dist=True)
        self.log('train_phoneme_kl_loss', po_outputs['phoneme_kl_loss'], prog_bar=False, sync_dist=True)
        self.log('train_entropy', po_outputs['entropy'], prog_bar=True, sync_dist=True)
        self.log('train_audio_entropy', po_outputs['audio_entropy'], prog_bar=False, sync_dist=True)
        self.log('train_phoneme_entropy', po_outputs['phoneme_entropy'], prog_bar=False, sync_dist=True)
        self.log('train_used_gt_phoneme_input', po_outputs['used_gt_phoneme_input'], prog_bar=True, sync_dist=True)
        self.log('train_mean_reward', generated_codes_and_metrics['mean_reward'], prog_bar=True, sync_dist=True)
        self.log('train_std_reward', generated_codes_and_metrics['std_reward'], prog_bar=True, sync_dist=True)
        self.log('train_cfg_fraction', generated_codes_and_metrics['cfg_fraction'], prog_bar=False, sync_dist=True)
        if generated_codes_and_metrics['mean_reward_cfg'] is not None:
            self.log(
                'train_mean_reward_cfg',
                generated_codes_and_metrics['mean_reward_cfg'],
                prog_bar=False,
                sync_dist=True,
            )
        if generated_codes_and_metrics['mean_reward_no_cfg'] is not None:
            self.log(
                'train_mean_reward_no_cfg',
                generated_codes_and_metrics['mean_reward_no_cfg'],
                prog_bar=False,
                sync_dist=True,
            )

        # Gradient / weight diagnostics to wandb.
        for metric_name, metric_value in grad_weight_metrics.items():
            self.log(f'train_{metric_name}', metric_value, prog_bar=False, sync_dist=True)

        # Compact summary to stdout / log file.
        print_grad_weight_summary(
            metrics=grad_weight_metrics,
            step=self.global_step,
            is_global_zero=getattr(self.trainer, "is_global_zero", True),
        )

        # Timing metrics.
        timings = generated_codes_and_metrics.get('timings', {})
        for tkey in ('audio_generation_time_sec', 'audio_save_time_sec', 'rewarding_time_sec'):
            self.log(f'train_{tkey}', float(timings.get(tkey, 0.0)), prog_bar=False, sync_dist=True)
        self.log('train_teacher_forced_time_sec', teacher_forced_time_sec, prog_bar=False, sync_dist=True)

    def teardown(self, stage: str) -> None:
        self._reward_asr_router.close()
        super().teardown(stage)
