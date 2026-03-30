# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import random
import time
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    get_speaker_embeddings_from_filepaths,
    print_grad_weight_summary,
    process_text_for_cer,
    transcribe_with_whisper_from_filepaths,
)
from nemo.utils import logging

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    Normalizer = None
    PYNINI_AVAILABLE = False

try:
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    HAVE_UTMOSV2 = True
except (ImportError, ModuleNotFoundError):
    HAVE_UTMOSV2 = False


class EasyMagpieTTSModelOnlinePO(EasyMagpieTTSModel):
    """
    EasyMagpie-TTS online preference optimization model (GRPO / DR-GRPO).

    Training flow:
    1. Sample multiple generations per prompt.
    2. Compute rewards (CER/SSIM/UTMOSv2).
    3. Compute group-normalized advantages.
    4. Run teacher-forced policy forward on generated codes and optimize GRPO objective.
    5. Add auxiliary phoneme loss from the same forward pass with GT phoneme tokens.
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
                torch.load(cfg.reference_model_ckpt_path, map_location="cpu", weights_only=False)['state_dict']
            )
            self._reference_model.freeze()
            self._reference_model._no_state_dict = True
            logging.info("Reference model loaded and frozen")

        reward_asr_model = cfg.get('reward_asr_model', 'nemo')
        if reward_asr_model == 'nemo':
            self._eval_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=cfg.get('reward_asr_model_name', "nvidia/parakeet-ctc-0.6b")
            )
            self._eval_asr_model.freeze()
            self.whisper_processor = None
            self.whisper_model = None
        elif reward_asr_model == 'whisper':
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self._eval_asr_model = None
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            self.whisper_model.eval()
            for param in self.whisper_model.parameters():
                param.requires_grad = False
            self.use_multilingual_asr = True
        else:
            raise ValueError(f"Unknown reward_asr_model: {reward_asr_model}")

        self._eval_speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=cfg.get('speaker_verification_model_name', 'titanet_large')
        )
        self._eval_speaker_verification_model.freeze()

        self.use_utmos = self.cfg.get('use_utmos', False)
        if self.use_utmos:
            assert HAVE_UTMOSV2, (
                "UTMOSv2 is required for the UTMOS reward but is not installed. "
                "Install it with: pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1"
            )
            # Initialize on CPU; we score from saved wav files so no GPU needed.
            self._utmos_calculator = UTMOSv2Calculator(device='cpu')
            logging.info("UTMOSv2 calculator initialized for naturalness reward")

        self.loss_type = self.cfg.get('loss_type', 'grpo')
        if self.loss_type not in ['grpo', 'dr_grpo']:
            raise ValueError(f"Received loss_type={self.loss_type}. Supported values: ['grpo', 'dr_grpo'].")
        self.scale_rewards = self.cfg.get('scale_rewards', True)
        self.max_decoder_steps = self.cfg.get('max_decoder_steps', 220)
        self.aux_phoneme_loss_weight = self.cfg.get('aux_phoneme_loss_weight', 1.0)
        self.po_groups_per_subbatch = max(int(self.cfg.get('po_groups_per_subbatch', 1)), 1)
        self.batch_size_for_chunked_tf = self.cfg.get('batch_size_for_chunked_tf', 4)

        self._normalize_whisper_transcript = self.cfg.get('normalize_whisper_transcript', True)
        if reward_asr_model == 'whisper' and self._normalize_whisper_transcript:
            self._normalizer_cache = {}

        # Entropy bonus coefficient – encourages exploration and prevents mode collapse.
        # Set to 0.0 to disable. Typical range: 0.001–0.01.
        self.entropy_coeff = self.cfg.get('entropy_coeff', 0.0)

        # Filter out poor groups for stable optimization.
        self.best_cer_threshold = self.cfg.get('best_cer_threshold', 1.0)
        self.worst_cer_threshold = self.cfg.get('worst_cer_threshold', 1.0)

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
        Exclude frozen eval/reference modules AND modules that receive no gradients
        from the PO loss (final_proj, lm_text_head, phoneme_final_proj) from the
        optimizer. Including them would subject their weights to weight decay without
        any learning signal, slowly degrading them.
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
            'phoneme_final_proj',
        }

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
            per_token_logps = per_token_logps * loss_mask.float()
        return per_token_logps

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

    def _get_audio_dir(self) -> str:
        """Return (and create if needed) the directory used to store intermediate waveforms during PO."""
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
            mode=mode,
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

        rows = []
        for local_idx, metric_idx in enumerate(range(group_start_idx, group_end_idx)):
            item_metrics = batch_metrics[metric_idx]
            rows.append(
                [
                    str(local_idx),
                    f"{item_metrics['cer_gt']:.4f}",
                    f"{item_metrics['wer_gt']:.4f}",
                    f"{item_metrics['spk_similarity']:.4f}",
                    f"{item_metrics.get('utmos', 0.0):.4f}",
                    f"{item_metrics['reward']:.4f}",
                    f"{item_metrics.get('advantage', 0.0):.4f}",
                ]
            )

        table = self._format_text_table(
            headers=["item", "cer", "wer", "ssim", "utmos", "reward", "advantage"], rows=rows
        )
        logging.info(
            f"[generate_and_reward] group={group_idx} valid={is_group_valid} "
            f"mean_reward={mean_reward:.4f} std_reward={std_reward:.4f}\n"
            f"prompt: {prompt_text}\n{table}\n"
        )

    def _compute_pred_transcripts(
        self, predicted_audio_paths: List[str], batch_repeated: Dict, reward_asr_model: str
    ) -> List[str]:
        """Transcribe predicted audio files using either the NeMo ASR model or Whisper.

        Returns a list of processed transcript strings (one per audio file), ready for CER/WER
        computation.
        """
        if reward_asr_model == 'nemo':
            pred_transcripts = self._eval_asr_model.transcribe(
                predicted_audio_paths,
                batch_size=len(predicted_audio_paths),
                override_config=TranscribeConfig(
                    use_lhotse=False, batch_size=len(predicted_audio_paths), num_workers=0
                ),
            )
            return [process_text_for_cer(transcript.text) for transcript in pred_transcripts]

        self.whisper_model.to(self.device)
        pred_transcripts = [""] * len(predicted_audio_paths)
        langs = batch_repeated.get('languages', ['en'] * len(predicted_audio_paths))
        language_groups = {}
        for item_idx, audio_path in enumerate(predicted_audio_paths):
            language = langs[item_idx] if item_idx < len(langs) else 'en'
            language_groups.setdefault(language, []).append((item_idx, audio_path))

        for language, grouped_items in language_groups.items():
            normalizer = self._get_cached_normalizer(language) if self._normalize_whisper_transcript else None
            grouped_paths = [audio_path for _, audio_path in grouped_items]
            group_transcripts = transcribe_with_whisper_from_filepaths(
                audio_filepaths=grouped_paths,
                language=language,
                whisper_processor=self.whisper_processor,
                whisper_model=self.whisper_model,
                device=self.device,
                normalizer=normalizer,
            )
            for (item_idx, _), transcript in zip(grouped_items, group_transcripts):
                pred_transcripts[item_idx] = process_text_for_cer(transcript)
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
        batch_repeated = self.repeat_items_in_batch(batch, num_generations_per_item)
        reward_asr_model = self.cfg.get('reward_asr_model', 'nemo')

        use_cfg = False
        cfg_scale = 1.0
        inference_cfg_prob = self.cfg.get('inference_cfg_prob', 0.0)
        if (inference_cfg_prob == 1.0) or (inference_cfg_prob > 0.0 and mode == 'train'):
            use_cfg = random.random() < inference_cfg_prob
            cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)

        phoneme_input_type = 'pred'
        gt_phoneme_input_prob = self.cfg.get('gt_phoneme_input_prob', 0.0)
        can_use_gt_phonemes = ('phoneme_tokens' in batch_repeated) and ('phoneme_tokens_lens' in batch_repeated)
        if can_use_gt_phonemes and gt_phoneme_input_prob > 0.0 and mode == 'train':
            phoneme_input_type = 'gt' if random.random() < gt_phoneme_input_prob else 'pred'

        generation_start_time = time.perf_counter()
        logging.info("Inference started")
        output = self.infer_batch(
            batch=batch_repeated,
            max_decoder_steps=self.max_decoder_steps,
            temperature=self.cfg.get('inference_temperature', 0.7),
            topk=self.cfg.get('inference_topk', 80),
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            use_local_transformer_for_inference=use_local_transformer_for_inference,
            phoneme_input_type=phoneme_input_type,
            phoneme_sampling_method=self.cfg.get('inference_phoneme_sampling_method', 'argmax'),
            force_dropout_text=False,
            use_teacher_forced=False,
            use_inference_mode=False,
        )
        logging.info("Inference ended")
        audio_generation_time_sec = time.perf_counter() - generation_start_time

        predicted_audio = output.predicted_audio
        predicted_audio_lens = output.predicted_audio_lens
        predicted_codes = output.predicted_codes
        predicted_codes_lens = output.predicted_codes_lens
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
        pred_transcripts = self._compute_pred_transcripts(predicted_audio_paths, batch_repeated, reward_asr_model)
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
            }

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
        rewarding_time_sec = time.perf_counter() - rewarding_start_time

        return {
            'mean_reward': torch.tensor(all_groups_mean_reward, device=self.device, dtype=torch.float32),
            'std_reward': torch.tensor(all_groups_std_reward, device=self.device, dtype=torch.float32),
            'batch_repeated': batch_repeated,
            'metrics': batch_metrics,
            'predicted_codes': predicted_codes,
            'predicted_codes_lens': predicted_codes_lens,
            'advantages': advantages,
            'group_validities': group_validities,
            'rollout_phoneme_input_type': phoneme_input_type,
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
        batch_repeated['audio_codes'] = predicted_codes
        batch_repeated['audio_codes_lens'] = predicted_codes_lens
        if 'audio' in batch_repeated:
            del batch_repeated['audio']
        if 'audio_lens' in batch_repeated:
            del batch_repeated['audio_lens']

        return generated_codes_and_metrics, batch_repeated, predicted_codes, predicted_codes_lens

    def _compute_po_losses_from_outputs(
        self,
        policy_output,
        reference_output,
        advantages: torch.Tensor,
        group_validities: torch.Tensor,
        rollout_phoneme_input_type: str,
    ):
        """Compute the GRPO (or DR-GRPO) policy-optimization loss, KL divergence against the
        reference model, per-token entropy, and the optional auxiliary phoneme loss.

        Returns:
            Dict with keys ``loss``, ``po_loss``, ``phoneme_aux_loss``, ``kl_loss``,
            ``entropy``, and ``used_gt_phoneme_input``.
        """
        logits = policy_output.local_transformer_logits
        if logits is None:
            logits = policy_output.logits
        ref_logits = None
        if reference_output is not None:
            ref_logits = reference_output.local_transformer_logits
            if ref_logits is None:
                ref_logits = reference_output.logits

        audio_codes_target = policy_output.audio_codes_target.long()
        audio_codes_lens_target = policy_output.audio_codes_lens_target
        audio_loss_mask = get_mask_from_lengths(audio_codes_lens_target).float()

        n_codebooks = audio_codes_target.size(1)
        total_loss = None
        total_kl = None
        total_entropy = None
        for codebook_idx in range(n_codebooks):
            si = codebook_idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]
            codebook_labels = audio_codes_target[:, codebook_idx, :]
            per_token_logps = self._get_per_token_logps(codebook_logits, codebook_labels, audio_loss_mask)
            # Ensure the GRPO policy gradient trick stays in fp32 to preserve gradient signal
            with torch.cuda.amp.autocast(enabled=False):
                per_token_loss = -(
                    torch.exp(per_token_logps.float() - per_token_logps.float().detach())
                    * advantages.float().unsqueeze(1)
                )
                per_token_loss = per_token_loss * group_validities.float().unsqueeze(1)

            # Per-token entropy of the policy distribution (always computed for logging).
            with torch.cuda.amp.autocast(enabled=False):
                logits_fp32 = codebook_logits.float()
                log_probs = logits_fp32.log_softmax(-1)  # [B, T, V]
                probs = log_probs.exp()  # [B, T, V]
                per_token_entropy = -(probs * log_probs).sum(-1)  # [B, T]
            codebook_entropy = (
                (per_token_entropy * audio_loss_mask).sum(dim=1) / audio_loss_mask.sum(dim=1).clamp_min(1e-8)
            ).mean()

            if not self.reference_free and ref_logits is not None:
                with torch.no_grad():
                    ref_codebook_logits = ref_logits[:, :, si:ei]
                    per_token_ref_logps = self._get_per_token_logps(
                        ref_codebook_logits, codebook_labels, audio_loss_mask
                    )
                with torch.cuda.amp.autocast(enabled=False):
                    per_token_kl = (
                        torch.exp(per_token_ref_logps.float() - per_token_logps.float())
                        - (per_token_ref_logps.float() - per_token_logps.float())
                        - 1
                    )
                    per_token_loss = per_token_loss + self.cfg.get('grpo_beta', 0.0) * per_token_kl
                codebook_kl_loss_mean = (
                    (per_token_kl * audio_loss_mask).sum(dim=1) / audio_loss_mask.sum(dim=1).clamp_min(1e-8)
                ).mean()
            else:
                codebook_kl_loss_mean = torch.tensor(0.0, device=self.device)

            if self.loss_type == "grpo":
                codebook_loss = (
                    (per_token_loss * audio_loss_mask).sum(dim=1) / audio_loss_mask.sum(dim=1).clamp_min(1e-8)
                ).mean()
            elif self.loss_type == "dr_grpo":
                total_tokens = per_token_loss.shape[0] * self.max_decoder_steps
                codebook_loss = (per_token_loss * audio_loss_mask).sum() / max(total_tokens, 1)
            else:
                raise ValueError(f"Unknown loss function: {self.loss_type}")

            if total_loss is None:
                total_loss = codebook_loss
                total_kl = codebook_kl_loss_mean
                total_entropy = codebook_entropy
            else:
                total_loss += codebook_loss
                total_kl += codebook_kl_loss_mean
                total_entropy += codebook_entropy

        total_po_loss = total_loss / n_codebooks
        total_kl = total_kl / n_codebooks
        total_entropy = total_entropy / n_codebooks

        phoneme_aux_loss = policy_output.phoneme_loss if rollout_phoneme_input_type == 'gt' else None
        if phoneme_aux_loss is None:
            phoneme_aux_loss = torch.tensor(0.0, device=self.device)

        # Subtracting entropy encourages higher entropy (more exploration / prevents mode collapse).
        total_loss = total_po_loss + self.aux_phoneme_loss_weight * phoneme_aux_loss
        if self.entropy_coeff > 0:
            total_loss = total_loss - self.entropy_coeff * total_entropy

        return {
            'loss': total_loss,
            'po_loss': total_po_loss,
            'phoneme_aux_loss': phoneme_aux_loss,
            'kl_loss': total_kl,
            'entropy': total_entropy,
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
        accumulated_phoneme_aux_loss = torch.tensor(0.0, device=self.device)
        accumulated_kl_loss = torch.tensor(0.0, device=self.device)
        accumulated_entropy = torch.tensor(0.0, device=self.device)
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
            )

            if do_backward:
                self.manual_backward(chunk_outputs['loss'] * chunk_weight)

            accumulated_loss = accumulated_loss + chunk_outputs['loss'].detach() * chunk_weight
            accumulated_po_loss = accumulated_po_loss + chunk_outputs['po_loss'].detach() * chunk_weight
            accumulated_phoneme_aux_loss = (
                accumulated_phoneme_aux_loss + chunk_outputs['phoneme_aux_loss'].detach() * chunk_weight
            )
            accumulated_kl_loss = accumulated_kl_loss + chunk_outputs['kl_loss'].detach() * chunk_weight
            accumulated_entropy = accumulated_entropy + chunk_outputs['entropy'].detach() * chunk_weight
            used_gt_phoneme_input = max(used_gt_phoneme_input, chunk_outputs['used_gt_phoneme_input'])

        return {
            'loss': accumulated_loss,
            'po_loss': accumulated_po_loss,
            'phoneme_aux_loss': accumulated_phoneme_aux_loss,
            'kl_loss': accumulated_kl_loss,
            'entropy': accumulated_entropy,
            'used_gt_phoneme_input': used_gt_phoneme_input,
        }

    def training_step(self, batch, batch_idx):
        """Execute one full online PO training iteration: rollout generation, reward computation,
        chunked teacher-forced forward/backward, gradient clipping, optimizer step, LR scheduling,
        and logging of all training metrics and diagnostics.
        """
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
        max_grad_norm = self.cfg.get('max_grad_norm', 0.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.parameters() if p.requires_grad and p.grad is not None],
                max_norm=max_grad_norm,
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
        self.log('train_phoneme_aux_loss', po_outputs['phoneme_aux_loss'], prog_bar=True, sync_dist=True)
        self.log('train_kl_loss', po_outputs['kl_loss'], prog_bar=True, sync_dist=True)
        self.log('train_entropy', po_outputs['entropy'], prog_bar=True, sync_dist=True)
        self.log('train_used_gt_phoneme_input', po_outputs['used_gt_phoneme_input'], prog_bar=True, sync_dist=True)
        self.log('train_mean_reward', generated_codes_and_metrics['mean_reward'], prog_bar=True, sync_dist=True)
        self.log('train_std_reward', generated_codes_and_metrics['std_reward'], prog_bar=True, sync_dist=True)

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
