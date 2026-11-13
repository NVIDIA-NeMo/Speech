# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from types import SimpleNamespace

import torch
from omegaconf import open_dict

from nemo.collections.tts.models.easy_magpietts_preference_optimization import OnlinePreferenceOptimizationMixin
from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths


class NemotronTTSModelOnlinePO(OnlinePreferenceOptimizationMixin, NemotronTTSModel):
    """Online GRPO using the V2 acoustic predictor and sampled phoneme policy."""

    def __init__(self, cfg, trainer=None):
        if cfg.get('inference_cfg_prob', 0.0) != 0.0:
            raise ValueError('V2 online preference optimization requires inference_cfg_prob=0.')
        if cfg.get('inference_temperature', 0.7) <= 0.0:
            raise ValueError('V2 online preference optimization requires a positive inference_temperature.')
        if cfg.get('inference_topk', 80) < 1:
            raise ValueError('V2 online preference optimization requires inference_topk >= 1.')
        if cfg.get('feature_masking') is not None:
            raise ValueError('V2 online preference optimization requires feature_masking=null.')
        train_data = (cfg.get('train_ds') or {}).get('dataset') or {}
        if (
            train_data.get('challenging_text_replacement_prob', 0.0) > 0.0
            and cfg.get('gt_phoneme_input_prob', 0.0) > 0.0
        ):
            raise ValueError('Challenging text replacement requires gt_phoneme_input_prob=0.')
        if (
            cfg.get('inference_phoneme_sampling_method', 'argmax') == 'sample'
            and cfg.get('phoneme_confidence_unk_threshold', 0.0) > 0.0
        ):
            raise ValueError('Sampled phoneme optimization requires phoneme_confidence_unk_threshold=0.')

        model_cfg = copy.deepcopy(cfg)
        with open_dict(model_cfg):
            model_cfg.run_val_inference = False
        self.utmos_device = cfg.get('utmos_device', 'cpu')
        super().__init__(model_cfg, trainer)
        if self.mode_name_to_mode[self.default_inference_mode].text_input_mode != 'streaming':
            raise ValueError('V2 online preference optimization requires a streaming inference mode.')
        self.acoustic_codes_predictor.sampling_in_fp32 = True
        self.aux_phoneme_po_loss_weight = cfg.get('aux_phoneme_po_loss_weight', 0.2)
        self._capture_policy_rollout = False
        self._policy_rollout = None
        self._vllm_rollout_client = None
        for name in ('final_proj', 'audio_out_projection', 'text_aligner'):
            module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(False)

    def _create_reference_model(self, cfg):
        return NemotronTTSModel(cfg=cfg)

    def get_lhotse_dataloader(self, dataset_cfg, mode='train'):
        """Use single-turn prompts without changing the checkpoint vocabulary."""
        return super().get_lhotse_dataloader(dataset_cfg, mode=mode, use_multiturn_dataset=False)

    def infer_batch(self, *args, **kwargs):
        if self._capture_policy_rollout and self.cfg.get('rollout_backend', 'native') == 'vllm':
            if kwargs.get('use_cfg', False) or kwargs.get('phoneme_input_type') == 'gt':
                raise ValueError('vLLM GRPO rollouts require CFG disabled and predicted phonemes.')
            if kwargs.get('use_teacher_forced', False) or kwargs.get('force_dropout_text', False):
                raise ValueError('vLLM GRPO rollouts require generated audio and complete text conditioning.')
            if self._vllm_rollout_client is None:
                from nemo.collections.tts.modules.nemotron_tts.v2.vllm_rollout import VLLMRolloutClient

                self._vllm_rollout_client = VLLMRolloutClient(self)
            return self._vllm_rollout_client.generate(
                self,
                args[0] if args else kwargs['batch'],
                max_audio_steps=kwargs.get('max_decoder_steps', self.max_decoder_steps),
                temperature=kwargs.get('temperature', 0.7),
                top_k=kwargs.get('topk', 80),
                phoneme_sampling_method=kwargs.get('phoneme_sampling_method', 'argmax'),
            )
        return super().infer_batch(*args, **kwargs)

    def teardown(self, stage):
        client = getattr(self, '_vllm_rollout_client', None)
        if client is not None:
            client.close()
        return super().teardown(stage)

    def _get_state_dict_keys_to_exclude(self):
        return super()._get_state_dict_keys_to_exclude() + ['_reference_model']

    def save_to(self, save_path: str):
        """Export an ordinary V2 inference archive while preserving the training configuration."""
        training_cfg = self._cfg
        self._cfg = copy.deepcopy(training_cfg)
        with open_dict(self._cfg):
            self._cfg.target = f'{NemotronTTSModel.__module__}.{NemotronTTSModel.__name__}'
            self._cfg.run_val_inference = False
            self._cfg.use_utmos = False
        try:
            return super().save_to(save_path)
        finally:
            self._cfg = training_cfg

    def sample_codes_from_logits_phoneme(self, all_code_logits_t, temperature=0.7, topk=80):
        return super().sample_codes_from_logits_phoneme(
            all_code_logits_t.float(), temperature=temperature, topk=min(topk, self.phoneme_vocab_size)
        )

    def setup_optimizer_param_groups(self):
        """Optimize the sampled policy, preserving auxiliary and reward networks."""
        excluded = {
            '_speaker_verification_model',
            '_codec_model',
            '_eval_asr_model',
            '_eval_speaker_verification_model',
            '_reference_model',
            'whisper_model',
            'whisper_processor',
            '_utmos_calculator',
            'final_proj',
            'audio_out_projection',
            'text_aligner',
        }
        phoneme_pg = (
            self.cfg.get('inference_phoneme_sampling_method', 'argmax') == 'sample'
            and self.aux_phoneme_po_loss_weight > 0.0
        )
        phoneme_ce = self.cfg.get('gt_phoneme_input_prob', 0.0) > 0.0 and self.aux_phoneme_loss_weight > 0.0
        if not phoneme_pg and not phoneme_ce:
            excluded.add('phoneme_final_proj')
        excluded_ids = {
            id(parameter)
            for name, module in self.named_children()
            if name in excluded
            for parameter in module.parameters()
        }
        self._optimizer_param_groups = [
            {'params': [p for p in self.parameters() if p.requires_grad and id(p) not in excluded_ids]}
        ]

    def generate_and_reward(self, *args, **kwargs):
        self._capture_policy_rollout = True
        try:
            return super().generate_and_reward(*args, **kwargs)
        finally:
            self._capture_policy_rollout = False
            self._policy_rollout = None

    def streaming_finalize(self, state, **kwargs):
        """Retain complete sampled actions before waveform EOS cropping."""
        if self._capture_policy_rollout:
            rollout = {}
            if not state.all_predictions:
                raise RuntimeError('A GRPO rollout produced no acoustic-code actions.')
            codes = torch.cat(state.all_predictions, dim=-1)
            lengths = state.audio_steps.long().clone()
            stacked = codes.new_full(
                (codes.size(0), self.num_audio_codebooks * self.frame_stacking_factor, int(lengths.max())),
                self.mask_token_id,
            )
            for item, length in enumerate(lengths.tolist()):
                if length == 0:
                    continue
                start = max(int(state.audio_prediction_start_idx[item]), 0)
                frames = codes[item, :, start : start + length * self.frame_stacking_factor]
                stacked[item, :, :length] = (
                    frames.reshape(self.num_audio_codebooks, length, self.frame_stacking_factor)
                    .permute(0, 2, 1)
                    .reshape(-1, length)
                )
            rollout['_po_audio_codes_stacked'] = stacked
            rollout['_po_audio_codes_lens'] = lengths

            if self.phoneme_tokenizer is not None and state.all_phoneme_predictions:
                phonemes = torch.stack(state.all_phoneme_predictions, dim=-1)
                starts = state.phoneme_prediction_start_idx.clamp_min(0)
                ends = torch.where(
                    state.phoneme_prediction_end_idx >= 0,
                    state.phoneme_prediction_end_idx,
                    torch.full_like(starts, phonemes.size(-1)),
                )
                phoneme_lens = (ends - starts) * self.phoneme_stacking_factor + 1
                tokens = phonemes.new_full(
                    (phonemes.size(0), int(phoneme_lens.max())), self.phoneme_tokenizer.eos_token_id
                )
                tokens[:, 0] = self.phoneme_tokenizer.bos_token_id
                for item, length in enumerate(phoneme_lens.tolist()):
                    tokens[item, 1:length] = phonemes[item, :, int(starts[item]) : int(ends[item])].T.reshape(-1)
                rollout['_po_phoneme_tokens'] = tokens
                rollout['_po_phoneme_tokens_lens'] = phoneme_lens
            self._policy_rollout = rollout
        return super().streaming_finalize(state, **kwargs)

    def _prepare_rollout_batch(self, output, batch, phoneme_input_type):
        if self._policy_rollout is None:
            raise RuntimeError('The V2 rollout is missing its sampled policy actions.')
        batch.update(self._policy_rollout)
        if phoneme_input_type != 'gt' and '_po_phoneme_tokens' in batch:
            batch['phoneme_tokens'] = batch['_po_phoneme_tokens']
            batch['phoneme_tokens_lens'] = batch['_po_phoneme_tokens_lens']

    @staticmethod
    def _sampling_logits(logits, temperature, topk, sanitize=False):
        """Reproduce the rollout sampling distribution in FP32."""
        logits = logits.float()
        if sanitize:
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        cutoff = logits.topk(min(topk, logits.size(-1)), dim=-1).values[..., -1:]
        return (logits / temperature).masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)

    def _run_easy_process_batch(self, model, batch, audio_codes, audio_codes_lens, mode):
        if 'context_audio_codes' in batch:
            context_codes, context_lens = batch['context_audio_codes'], batch['context_audio_codes_lens']
        else:
            context_codes, context_lens = model._codec_helper.audio_to_codes(
                batch['context_audio'], batch['context_audio_lens']
            )
        lengths = batch['_po_audio_codes_lens']
        sampled_codes = batch['_po_audio_codes_stacked'][..., : int(lengths.max())]
        was_training = model.training
        model.eval()
        try:
            result = model.process_batch(
                text=batch['text'].clone(),
                text_lens=batch['text_lens'],
                context_text_tokens=batch['context_text_tokens'],
                context_text_tokens_lens=batch['context_text_tokens_lens'],
                audio_codes=sampled_codes,
                audio_codes_lens=lengths,
                context_audio_codes=context_codes,
                context_audio_codes_lens=context_lens,
                phoneme_tokens=batch.get('phoneme_tokens'),
                phoneme_tokens_lens=batch.get('phoneme_tokens_lens'),
                mode='val',
                training_mode=model.mode_name_to_mode[model.default_inference_mode],
                return_policy_logits=True,
                audio_codes_are_stacked=True,
            )
        finally:
            model.train(was_training)

        compact_logits = self._sampling_logits(
            result.acoustic_codes_predictor_logits,
            self.cfg.get('inference_temperature', 0.7),
            self.cfg.get('inference_topk', 80),
            sanitize=True,
        )
        logits = compact_logits.new_full(
            (*compact_logits.shape[:-1], self.num_all_tokens_per_codebook),
            torch.finfo(compact_logits.dtype).min,
        )
        logits[..., : self.codebook_size] = compact_logits[..., : self.codebook_size]
        logits[..., self.audio_eos_id] = compact_logits[..., self.codebook_size]
        return SimpleNamespace(
            local_transformer_logits=logits.flatten(2),
            audio_codes_target=result.audio_codes_target,
            audio_codes_lens_target=result.audio_codes_lens_target,
            phoneme_loss=result.phoneme_loss,
            phoneme_logits=result.phoneme_logits,
            phoneme_tokens_target=result.phoneme_tokens_target,
            phoneme_tokens_lens_target=result.phoneme_tokens_lens_target,
        )

    def _compute_po_losses_from_outputs(
        self, policy_output, reference_output, advantages, group_validities, rollout_phoneme_input_type
    ):
        losses = super()._compute_po_losses_from_outputs(
            policy_output, reference_output, advantages, group_validities, rollout_phoneme_input_type
        )
        if (
            rollout_phoneme_input_type == 'gt'
            or self.cfg.get('inference_phoneme_sampling_method', 'argmax') != 'sample'
            or policy_output.phoneme_logits is None
            or self.aux_phoneme_po_loss_weight == 0.0
        ):
            return losses

        logits = policy_output.phoneme_logits.reshape(
            *policy_output.phoneme_logits.shape[:2], self.phoneme_stacking_factor, self.phoneme_vocab_size
        )
        logits = self._sampling_logits(
            logits, self.cfg.get('inference_temperature', 0.7), self.cfg.get('inference_topk', 80), sanitize=True
        )
        targets = policy_output.phoneme_tokens_target.transpose(1, 2).long()
        mask = get_mask_from_lengths(policy_output.phoneme_tokens_lens_target, x=targets[:, :, 0]).unsqueeze(-1)
        log_probs = logits.log_softmax(-1)
        logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).masked_fill(~mask, 0.0)
        token_loss = -torch.exp(logps - logps.detach()) * advantages.float()[:, None, None]
        token_loss = token_loss * group_validities.float()[:, None, None]
        kl = torch.zeros_like(token_loss)
        if reference_output is not None:
            with torch.no_grad():
                ref_logits = reference_output.phoneme_logits.reshape_as(logits)
                ref_logits = self._sampling_logits(
                    ref_logits,
                    self.cfg.get('inference_temperature', 0.7),
                    self.cfg.get('inference_topk', 80),
                    sanitize=True,
                )
                ref_logps = ref_logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                ref_logps = ref_logps.masked_fill(~mask, 0.0)
            delta = ref_logps - logps
            kl = delta.exp() - delta - 1.0
            token_loss = token_loss + self.cfg.get('grpo_beta', 0.0) * kl

        denominators = mask.sum(1).clamp_min(1)
        if self.loss_type == 'dr_grpo':
            phoneme_loss = (token_loss * mask).sum() / max(
                token_loss.size(0) * self.max_decoder_steps * self.phoneme_stacking_factor, 1
            )
        else:
            phoneme_loss = ((token_loss * mask).sum(1) / denominators).mean()
        entropy = (-(log_probs.exp() * log_probs).sum(-1) * mask).sum(1) / denominators
        losses['loss'] = losses['loss'] + self.aux_phoneme_po_loss_weight * (
            phoneme_loss - self.entropy_coeff * entropy.mean()
        )
        return losses
