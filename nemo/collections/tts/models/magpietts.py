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
import json
import os
import string
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import get_worker_info
from transformers import AutoTokenizer, T5Tokenizer

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths, plot_alignment_to_numpy
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


def setup_tokenizers(all_tokenizers_config, use_text_conditioning_tokenizer, mode='train'):
    # Being used in both model and worker_init_fn, so it is defined here
    # Returns two tokenizers: one for TTS transcript and one for conditioning text (if needed)
    tokenizers = []
    tokenizer_names = []
    for tokenizer_name in all_tokenizers_config:
        tokenizer_config = all_tokenizers_config[tokenizer_name]
        if tokenizer_config._target_ == 'AutoTokenizer':
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.pretrained_model)
        else:
            text_tokenizer_kwargs = {}
            if "g2p" in tokenizer_config:
                text_tokenizer_kwargs["g2p"] = instantiate(tokenizer_config.g2p)
            tokenizer = instantiate(tokenizer_config, **text_tokenizer_kwargs)
            if mode == 'test' and hasattr(tokenizer, "set_phone_prob"):
                tokenizer.set_phone_prob(1.0)
        tokenizers.append(tokenizer)
        tokenizer_names.append(tokenizer_name)

    aggregated_tokenizer = AggregatedTTSTokenizer(tokenizers, tokenizer_names)  # TTS Transcript tokenizer
    text_conditioning_tokenizer = None

    if use_text_conditioning_tokenizer:
        # TODO: make this configurable
        # Conditioning text tokenizer
        text_conditioning_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

    return aggregated_tokenizer, text_conditioning_tokenizer


def worker_init_fn(worker_id):
    # For mp.set_start_method("spawn", force=True)
    # The dataset class should be picklable, so we initialize non-picklable objects here
    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # Get the dataset instance in this worker
    tokenizer, text_conditioning_tokenizer = setup_tokenizers(
        dataset.tokenizer_config, dataset.use_text_conditioning_tokenizer, mode=dataset.dataset_type
    )
    dataset.text_tokenizer = tokenizer
    dataset.text_conditioning_tokenizer = text_conditioning_tokenizer


class MagpieTTS_Model(ModelPT):
    """
    Magpie-TTS Model Base Class used for training a TTS model that can generate audio codes from transcript and a context
    audio/text

    Supports multiple model types:

    - single_encoder_sv_tts: Transcript goes into the encoder and target audio goes to the decoder. Additionally,
    speaker_embedding of target audio (or context audio if provided) from TitaNet gets added to encoder
    output(all timesteps).

    - multi_encoder_context_tts: Transcript and context audio go to different encoders. Transcript encoding feeds to
    layers given by cfg.model.transcript_decoder_layers and the context encoding feeds into the layers given by
    context_decoder_layers .Also supports text context which gets encoded by the same encoder as context audio.
    Only one of context audio or contex text is supported.

    - decoder_context_tts: Text goes into the encoder; context & target audio go to the decoder. Also supports text
    context. Supports fixed sized context so we set context_duration_min and context_duration_max to the same
    value (5 seconds). Text context, which is usually shorter than number of codec frames of 5 second of audio, is
    padded to the max context duration in this model.

    - decoder_pretrain_synthesizer: This is the model type used for pretraining the decoder only on audio data using
    next frame prediction loss.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # Setup tokenizer
        if hasattr(cfg, 'text_tokenizer'):
            # For backward compatibility for English-only models
            with open_dict(cfg):
                cfg.text_tokenizers = {"english_phoneme": cfg.text_tokenizer}
                del cfg['text_tokenizer']

        self.use_text_conditioning_encoder = cfg.get('use_text_conditioning_encoder', False)
        tokenizer, text_conditioning_tokenizer = self._setup_tokenizers(cfg)
        self.tokenizer = tokenizer
        self.text_conditioning_tokenizer = text_conditioning_tokenizer

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        num_tokens = num_tokens_tokenizer + 2  # +2 for BOS and EOS
        self.bos_id = num_tokens - 2
        self.eos_id = num_tokens - 1

        self.audio_bos_id = cfg.num_audio_tokens_per_codebook - 2
        self.audio_eos_id = cfg.num_audio_tokens_per_codebook - 1
        self.context_audio_bos_id = cfg.num_audio_tokens_per_codebook - 2  # For backward compatibility
        self.context_audio_eos_id = cfg.num_audio_tokens_per_codebook - 1  # For backward compatibility
        self.model_type = cfg.get('model_type', 'single_encoder_sv_tts')

        if self.model_type == 'decoder_context_tts':
            self.context_audio_bos_id = (
                cfg.num_audio_tokens_per_codebook - 4
            )  # Changing these to make them different from target audio bos and eos
            self.context_audio_eos_id = cfg.num_audio_tokens_per_codebook - 3

        self._tb_logger = None

        self.pad_context_text_to_max_duration = self.model_type == 'decoder_context_tts'
        self.use_kv_cache_for_inference = cfg.get('use_kv_cache_for_inference', False)

        super().__init__(cfg=cfg, trainer=trainer)

        audio_embeddings = []
        for _ in range(cfg.num_audio_codebooks):
            audio_embeddings.append(nn.Embedding(cfg.num_audio_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        if self.model_type != 'decoder_pretrain_synthesizer':
            # Decoder pretrain synthesizer doesn't have transcript encoder/text embeddings
            self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)
            self.encoder = transformer_2501.Transformer(**dict(cfg.encoder))

        self.decoder = transformer_2501.Transformer(**dict(cfg.decoder))

        self.final_proj = nn.Linear(cfg.decoder.d_model, cfg.num_audio_codebooks * cfg.num_audio_tokens_per_codebook)

        codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'), strict=False)
        # del codec discriminator to free memory
        del codec_model.discriminator
        codec_model.eval()
        self.freeze_model(codec_model)
        self._codec_model = codec_model

        if self.model_type == 'single_encoder_sv_tts':
            speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name='titanet_large'
            )
            speaker_verification_model.eval()
            self.freeze_model(speaker_verification_model)
            self._speaker_verification_model = speaker_verification_model
            self.speaker_projection_layer = nn.Linear(cfg.speaker_emb_dim, cfg.embedding_dim)
            self.transcript_decoder_layers = [
                idx for idx in range(cfg.decoder.n_layers)
            ]  # All layers are used for text
        elif self.model_type == 'multi_encoder_context_tts':
            self.transcript_decoder_layers = cfg.get('transcript_decoder_layers', [3, 4, 5, 6, 7, 8])
            self.context_decoder_layers = cfg.get(
                'context_decoder_layers', [0, 1, 2, 9, 10, 11]
            )  # For backward compatibility
            multi_encoder_mapping = [None for _ in range(cfg.decoder.n_layers)]
            for layer in self.transcript_decoder_layers:
                multi_encoder_mapping[layer] = 0  # 0 means text goes to this layer, 1 means context goes to this layer
            for layer in self.context_decoder_layers:
                multi_encoder_mapping[layer] = 1
            self.multi_encoder_mapping = multi_encoder_mapping
            self.context_encoder = transformer_2501.Transformer(**dict(cfg.context_encoder))
        elif self.model_type == 'decoder_context_tts':
            self.transcript_decoder_layers = [
                idx for idx in range(cfg.decoder.n_layers)
            ]  # All layers are used for text
        elif self.model_type == 'decoder_pretrain_synthesizer':
            assert cfg.alignment_loss_scale == 0.0, "Alignment loss is not supported for decoder pretrain synthesizer"
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        if self.use_text_conditioning_encoder:
            self.context_text_embedding = nn.Embedding(self.text_conditioning_tokenizer.vocab_size, cfg.embedding_dim)

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        alignment_loss_scale = cfg.get('alignment_loss_scale', 0.0)
        if alignment_loss_scale > 0.0:
            self.alignment_loss = ForwardSumLoss(loss_scale=alignment_loss_scale)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if hasattr(self, '_no_state_dict') and self._no_state_dict:
            return {}
        # Don't save the speaker verification and codec model in the state dict
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model']
        for key in list(state_dict.keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Override to load all the keys except _speaker_verification_model and _codec_model
        super().load_state_dict(state_dict, strict=False)

    def _setup_tokenizers(self, cfg, mode='test'):
        tokenizer, text_conditioning_tokenizer = setup_tokenizers(
            cfg.text_tokenizers, cfg.use_text_conditioning_encoder, mode=mode
        )
        return tokenizer, text_conditioning_tokenizer

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            self._tb_logger = tb_logger
        return self._tb_logger

    def audio_to_codes(self, audio, audio_len, audio_type='target'):
        # audio: (B, T)
        # audio_len: (B,)
        if audio_type == 'target':
            audio_eos_id = self.audio_eos_id
            audio_bos_id = self.audio_bos_id
        elif audio_type == 'context':
            audio_eos_id = self.context_audio_eos_id
            audio_bos_id = self.context_audio_bos_id
        else:
            raise ValueError(f"Received audio_type of {audio_type}. Must be `target` or `context`")

        self._codec_model.eval()
        with torch.no_grad():
            codes, codes_len = self._codec_model.encode(audio=audio, audio_len=audio_len)
            # Add a timestep to begining and end of codes tensor
            bos_tensor = torch.full(
                (codes.size(0), codes.size(1), 1), audio_bos_id, dtype=codes.dtype, device=codes.device
            )
            pad_tensor = torch.full(
                (codes.size(0), codes.size(1), 1), 0, dtype=codes.dtype, device=codes.device
            )  # 0 is the padding token in the audio codebook
            codes = torch.cat([bos_tensor, codes, pad_tensor], dim=-1)
            # codes: (B, C, T')
            # codes_len: (B,)
            for idx in range(codes.size(0)):
                codes[idx, :, codes_len[idx] + 1] = audio_eos_id
            codes_len = codes_len + 2

            return codes.long(), codes_len.long()

    def codes_to_audio(self, codes, codes_len):
        # codes: (B, C, T')
        # codes_len: (B,)
        self._codec_model.eval()
        with torch.no_grad():
            # Replace eos and bos tokens with padding in codes tensor
            codes[codes == self.audio_bos_id] = 0  # zero is the padding token in the audio codebook
            codes[codes == self.audio_eos_id] = 0
            # self.additional_models['codec'] = self.additional_models['codec'].to(codes.device)
            audio, audio_len = self._codec_model.decode(tokens=codes, tokens_len=codes_len)
            # audio: (B, T)
            # audio_len: (B,)
            return audio, audio_len

    def embed_audio_tokens(self, audio_tokens):
        # audio_tokens: (B, C, T')
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(audio_tokens.size(1)):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def get_speaker_embeddings(self, audio_16khz, audio_len_16khz):
        # audio_16khz: (B, T)
        # audio_len_16khz: (B,)
        self._speaker_verification_model.eval()
        with torch.no_grad():
            _, speaker_embeddings = self._speaker_verification_model.forward(
                input_signal=audio_16khz, input_signal_length=audio_len_16khz
            )
            return speaker_embeddings

    def compute_loss(self, logits, audio_codes, audio_codes_lens):
        # logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes: (B, C, T')
        # audio_codes_lens: (B,)
        loss_mask = get_mask_from_lengths(audio_codes_lens)
        total_codebook_loss = None
        for codebook in range(audio_codes.size(1)):
            si = codebook * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]  # (B, T', num_tokens_per_codebook)
            codebook_targets = audio_codes[:, codebook]  # (B, T')
            codebook_loss = self.cross_entropy_loss(
                codebook_logits.permute(0, 2, 1), codebook_targets  # (B, num_tokens_per_codebook, T')
            )  # (B, T')
            codebook_loss = codebook_loss * loss_mask
            codebook_loss = codebook_loss.sum() / loss_mask.sum()
            if total_codebook_loss is None:
                total_codebook_loss = codebook_loss
            else:
                total_codebook_loss = total_codebook_loss + codebook_loss

        total_codebook_loss = total_codebook_loss / audio_codes.size(1)
        return total_codebook_loss, loss_mask

    def forward(self, dec_input_embedded, dec_input_mask, cond, cond_mask, attn_prior, multi_encoder_mapping):
        decoder_out = self.decoder(
            dec_input_embedded,
            dec_input_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=multi_encoder_mapping,
        )
        attn_probabilities = decoder_out['attn_probabilities']
        all_code_logits = self.final_proj(decoder_out['output'])  # (B, T', num_codebooks * num_tokens_per_codebook)
        return all_code_logits, attn_probabilities

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = []
        for idx in range(self.cfg.num_audio_codebooks):
            si = idx * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits[:, :, si:ei]
            codebook_probs = torch.softmax(codebook_logits, dim=-1)  # (B, T', num_tokens_per_codebook)
            # argmax to get the tokens
            codebook_preds = torch.argmax(codebook_probs, dim=-1)  # (B, T')
            all_preds.append(codebook_preds)

        all_preds = torch.stack(all_preds, dim=1)  # (B, C, T')
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.cfg.num_audio_codebooks):
            si = idx * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(codebook_logits / temperature, dim=-1)  # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def log_attention_probs(self, attention_prob_matrix, audio_codes_lens, text_lens, prefix="", dec_context_size=0):
        # attention_prob_matrix List of (B, C, audio_timesteps, text_timesteps)
        with torch.no_grad():
            attention_prob_matrix = torch.cat(attention_prob_matrix, dim=1)  # (B, C, audio_timesteps, text_timesteps)
            attention_prob_matrix_mean = attention_prob_matrix.mean(dim=1)  # (B, audio_timesteps, text_timesteps)
            for idx in range(min(3, attention_prob_matrix_mean.size(0))):
                item_attn_matrix = attention_prob_matrix_mean[idx][
                    dec_context_size : dec_context_size + audio_codes_lens[idx], : text_lens[idx]
                ]
                item_attn_matrix = item_attn_matrix.detach().cpu().numpy()
                attn_np = plot_alignment_to_numpy(item_attn_matrix.T)
                self.tb_logger.add_image(
                    f'{prefix}attention_matrix_{idx}',
                    attn_np,
                    global_step=self.global_step,
                    dataformats="HWC",
                )

    def log_train_val_example(
        self,
        logits,
        target_audio_codes,
        audio_codes_lens_target,
        context_audio_codes=None,
        context_audio_codes_lens=None,
    ):
        pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens_target)
        pred_audio, pred_audio_lens = self.codes_to_audio(pred_audio_codes, audio_codes_lens_target)
        target_audio, target_audio_lens = self.codes_to_audio(target_audio_codes, audio_codes_lens_target)
        context_audio, context_audio_lens = None, None
        if context_audio_codes is not None and context_audio_codes.shape[2] > 3:
            # > 3 ensures, it is a valid context audio tensor (and not dummy tensor used in text context)
            context_audio, context_audio_lens = self.codes_to_audio(context_audio_codes, context_audio_codes_lens)
        for idx in range(min(3, pred_audio.size(0))):
            pred_audio_np = pred_audio[idx].float().detach().cpu().numpy()
            target_audio_np = target_audio[idx].float().detach().cpu().numpy()
            pred_audio_np = pred_audio_np[: pred_audio_lens[idx]]
            target_audio_np = target_audio_np[: target_audio_lens[idx]]
            self.tb_logger.add_audio(
                f'pred_audio_{idx}',
                pred_audio_np,
                global_step=self.global_step,
                sample_rate=self.cfg.sample_rate,
            )
            self.tb_logger.add_audio(
                f'target_audio_{idx}',
                target_audio_np,
                global_step=self.global_step,
                sample_rate=self.cfg.sample_rate,
            )
            if context_audio is not None:
                context_audio_np = context_audio[idx].float().detach().cpu().numpy()
                context_audio_np = context_audio_np[: context_audio_lens[idx]]
                self.tb_logger.add_audio(
                    f'context_audio_{idx}',
                    context_audio_np,
                    global_step=self.global_step,
                    sample_rate=self.cfg.sample_rate,
                )

    def scale_prior(self, prior, global_step):
        if prior is None:
            return None
        prior_end_step = self.cfg.prior_end_step
        prior_scaledown_start_step = self.cfg.prior_scaledown_start_step
        if global_step < prior_scaledown_start_step:
            return prior
        elif global_step >= prior_end_step:
            return None
        else:
            with torch.no_grad():
                # Interpolate between all ones and the prior
                residual = 1.0 - prior
                new_prior = prior + (
                    residual
                    * (global_step - prior_scaledown_start_step)
                    / (prior_end_step - prior_scaledown_start_step)
                )
                return new_prior

    def compute_alignment_loss(self, attention_scores, text_lens, audio_lens, dec_context_size=0):
        # attention scores: List of (B, C, audio_timesteps, text_timesteps)
        attention_scores_combined = torch.cat(attention_scores, dim=1)  # (B, C, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_combined.mean(
            dim=1, keepdim=True
        )  # (B, 1, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_mean[
            :, :, dec_context_size:, :
        ]  # Remove the context audio embeddings from the attention scores
        alignment_loss = self.alignment_loss(
            attn_logprob=attention_scores_mean, in_lens=text_lens, out_lens=audio_lens
        )
        return alignment_loss

    def prepare_context_tensors(self, batch):
        dec_context_size = 0
        additional_decoder_input = None
        addtional_decoder_mask = None
        context_audio_codes = None
        context_audio_codes_lens = None
        _attn_prior = None
        attn_prior = None
        cond = None
        cond_mask = None
        multi_encoder_mapping = None
        text = None
        text_lens = None

        # self.model_type must be one of
        # [single_encoder_sv_tts, multi_encoder_context_tts, decoder_context_tts, decoder_pretrain_synthesizer]
        if self.model_type != 'decoder_pretrain_synthesizer':
            text = batch['text']
            text_lens = batch['text_lens']
            text_embedded = self.text_embedding(text)  # (B, T, E)
            text_mask = get_mask_from_lengths(text_lens)  # (B, T)
            text_encoder_out = self.encoder(text_embedded, text_mask, cond=None, cond_mask=None)['output']  # (B, T, E)
            _attn_prior = batch.get('align_prior_matrix', None)
            _attn_prior = self.scale_prior(_attn_prior, self.global_step)

        if self.model_type == 'single_encoder_sv_tts':
            target_audio_16khz = batch['audio_16khz']
            target_audio_lens_16khz = batch['audio_lens_16khz']
            speaker_embeddings = self.get_speaker_embeddings(target_audio_16khz, target_audio_lens_16khz)
            speaker_embeddings_projected = self.speaker_projection_layer(speaker_embeddings)
            cond = text_encoder_out + speaker_embeddings_projected.unsqueeze(1)
            cond_mask = text_mask
            multi_encoder_mapping = None
            attn_prior = _attn_prior
        elif self.model_type in ['multi_encoder_context_tts', 'decoder_context_tts']:
            if 'context_audio_codes' in batch:
                context_audio_codes = batch['context_audio_codes']
                context_audio_codes_lens = batch['context_audio_codes_lens']
            else:
                context_audio_codes, context_audio_codes_lens = self.audio_to_codes(
                    batch['context_audio'], batch['context_audio_lens'], audio_type='context'
                )
            context_audio_embedded = self.embed_audio_tokens(context_audio_codes)  # (B, T', E)

            if self.use_text_conditioning_encoder:
                context_text_tokens = batch['context_text_tokens']
                context_text_lens = batch['context_text_tokens_lens']
                context_text_embedded = self.context_text_embedding(context_text_tokens)  # (B, L, E)
                # Pad context_audio_embedded or context_text_embedded so that they have same number of timesteps
                if context_audio_embedded.size(1) < context_text_embedded.size(1):
                    padding = torch.zeros(
                        context_audio_embedded.size(0),
                        context_text_embedded.size(1) - context_audio_embedded.size(1),
                        context_audio_embedded.size(2),
                        device=context_audio_embedded.device,
                    )
                    context_audio_embedded = torch.cat([context_audio_embedded, padding], dim=1)
                elif context_audio_embedded.size(1) > context_text_embedded.size(1):
                    padding = torch.zeros(
                        context_text_embedded.size(0),
                        context_audio_embedded.size(1) - context_text_embedded.size(1),
                        context_text_embedded.size(2),
                        device=context_text_embedded.device,
                    )
                    context_text_embedded = torch.cat([context_text_embedded, padding], dim=1)  # (B, T, E)
                has_text_context = batch['has_text_context'].unsqueeze(-1).unsqueeze(-1).float()  # (B, 1, 1)
                context_input_embedded = (
                    has_text_context * context_text_embedded + (1 - has_text_context) * context_audio_embedded
                )
                context_input_lens = (
                    batch['has_text_context'].float() * context_text_lens
                    + (1 - batch['has_text_context'].float()) * context_audio_codes_lens
                )  # (B,)
            else:
                context_input_embedded = context_audio_embedded
                context_input_lens = context_audio_codes_lens

            context_mask = get_mask_from_lengths(context_input_lens)

            if self.model_type == 'multi_encoder_context_tts':
                context_embeddings = self.context_encoder(
                    context_input_embedded, context_mask, cond=None, cond_mask=None
                )['output']
                cond = [text_encoder_out, context_embeddings]
                cond_mask = [text_mask, context_mask]
                multi_encoder_mapping = self.multi_encoder_mapping
                attn_prior = [_attn_prior, None]

            elif self.model_type == 'decoder_context_tts':
                dec_context_size = context_mask.size(1)
                context_embeddings = context_input_embedded
                attn_prior = _attn_prior
                if attn_prior is not None:
                    # B, audio_timesteps, text_timesteps
                    padding_zeros = torch.zeros(
                        attn_prior.size(0), dec_context_size, attn_prior.size(2), device=attn_prior.device
                    )
                    attn_prior = torch.cat([padding_zeros, attn_prior], dim=1)
                cond = text_encoder_out
                cond_mask = text_mask
                multi_encoder_mapping = None
                additional_decoder_input = context_embeddings
                addtional_decoder_mask = context_mask
        elif self.model_type == 'decoder_pretrain_synthesizer':
            pass
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        return {
            'cond': cond,
            'cond_mask': cond_mask,
            'attn_prior': attn_prior,
            'multi_encoder_mapping': multi_encoder_mapping,
            'additional_decoder_input': additional_decoder_input,
            'addtional_decoder_mask': addtional_decoder_mask,
            'dec_context_size': dec_context_size,
            'text': text,
            'text_lens': text_lens,
            'context_audio_codes': context_audio_codes,
            'context_audio_codes_lens': context_audio_codes_lens,
        }

    def prepare_dummy_cond_for_cfg(self, cond, cond_mask, additional_decoder_input, additional_dec_mask):
        dummy_additional_decoder_input = None
        dummy_additional_dec_mask = None
        if additional_decoder_input is not None:
            dummy_additional_decoder_input = torch.zeros_like(additional_decoder_input)
            # all ones mask means dont ignore any timesteps (so that it is consistent with usual decoder mask)
            dummy_additional_dec_mask = torch.ones_like(additional_dec_mask)

        if isinstance(cond, list):
            # multi encoder conditioning
            dummy_cond = [torch.zeros_like(cond_item) for cond_item in cond]
            attn_prior = [None for _ in cond]
            dummy_mask = []
            for mask_item in cond_mask:
                # ignore all timesteps except the first one
                mask = torch.zeros_like(mask_item)
                mask[:, 0] = 1  # Make first timestep all zeros
                dummy_mask.append(mask)

        elif isinstance(cond, torch.Tensor):
            # single encoder conditioning
            dummy_cond = torch.zeros_like(cond)
            dummy_mask = torch.zeros_like(cond_mask)
            dummy_mask[:, 0] = 1  # ignore all timesteps except the first one
            attn_prior = None
        else:
            raise ValueError(f"Unsupported type for cond {type(cond)}")

        return dummy_cond, dummy_mask, dummy_additional_decoder_input, dummy_additional_dec_mask, attn_prior

    def process_batch(self, batch, mode="train"):
        context_tensors = self.prepare_context_tensors(batch)
        disable_alignment_loss = False
        if 'audio_codes' not in batch:
            audio_codes, audio_codes_lens = self.audio_to_codes(batch['audio'], batch['audio_lens'])
        else:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']

        audio_codes_input = audio_codes[:, :, :-1]  # B, C, T'
        audio_codes_target = audio_codes[:, :, 1:]
        audio_codes_lens_input = audio_codes_lens_target = audio_codes_lens - 1

        audio_codes_mask = get_mask_from_lengths(audio_codes_lens_input)
        use_cfg = (
            (self.cfg.get('cfg_unconditional_prob', 0.0) > 0.0)
            and (mode == "train")
            and (context_tensors['cond'] is not None)
        )
        if use_cfg and torch.rand(1).item() < self.cfg.cfg_unconditional_prob:
            cond, cond_mask, additional_decoder_input, additional_decoder_mask, attn_prior = (
                self.prepare_dummy_cond_for_cfg(
                    context_tensors['cond'],
                    context_tensors['cond_mask'],
                    context_tensors['additional_decoder_input'],
                    context_tensors['addtional_decoder_mask'],
                )
            )
            disable_alignment_loss = True
        else:
            cond = context_tensors['cond']
            cond_mask = context_tensors['cond_mask']
            additional_decoder_input = context_tensors['additional_decoder_input']
            additional_decoder_mask = context_tensors['addtional_decoder_mask']
            attn_prior = context_tensors['attn_prior']

            if (
                mode == "train"
                and self.cfg.get('decoder_input_dropout_prob', 0.0) > 0.0
                and torch.rand(1).item() < 0.5
            ):
                # For some batches (half of them), replace decoder_input_dropout_prob of the timesteps with random tokens
                max_codebook_val = self.cfg.get('dec_random_input_max', self.cfg.num_audio_tokens_per_codebook)
                # @pneekhara: Keeping dec_random_input_max configurable since num_audio_tokens_per_codebook usually has padding tokens
                # which can cause errors when doing codes_to_audio for audio_codes_input. We are not currently calling codes_to_audio on
                # audio_codes_input so should not matter if we dont supply dec_random_input_max.
                random_audio_tokens = torch.randint(
                    0, max_codebook_val, audio_codes_input.size(), device=audio_codes_input.device
                )
                random_audio_tokens = random_audio_tokens * audio_codes_mask.unsqueeze(1)
                dec_dropout_mask = (
                    torch.rand((1, 1, audio_codes_input.size(2)), device=audio_codes_input.device)
                    > self.cfg.decoder_input_dropout_prob
                )
                # timestep_mask is True for timesteps to be kept
                audio_codes_input = audio_codes_input * dec_dropout_mask + random_audio_tokens * (~dec_dropout_mask)

        audio_codes_embedded = self.embed_audio_tokens(audio_codes_input)  # (B, T', E)
        if context_tensors['additional_decoder_input'] is not None:
            dec_input_embedded = torch.cat([additional_decoder_input, audio_codes_embedded], dim=1)
            dec_input_mask = torch.cat([additional_decoder_mask, audio_codes_mask], dim=1)
        else:
            dec_input_embedded = audio_codes_embedded
            dec_input_mask = audio_codes_mask

        logits, attn_info = self.forward(
            dec_input_embedded=dec_input_embedded,
            dec_input_mask=dec_input_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=context_tensors['multi_encoder_mapping'],
        )
        # logits: (B, T', num_codebooks * num_tokens_per_codebook)
        dec_context_size = context_tensors['dec_context_size']
        logits = logits[:, dec_context_size:, :]  # Remove the context audio embeddings from the logits

        codebook_loss, loss_mask = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        alignment_loss = None
        if self.cfg.alignment_loss_scale > 0.0 and not disable_alignment_loss:
            text_lens = context_tensors['text_lens']
            cross_attention_scores = [
                attn['cross_attn_probabilities'][1]
                for layer_idx, attn in enumerate(attn_info)
                if layer_idx in self.transcript_decoder_layers
            ]
            alignment_loss = self.compute_alignment_loss(
                cross_attention_scores, text_lens, audio_codes_lens_target, dec_context_size
            )
            loss = codebook_loss + alignment_loss
        else:
            loss = codebook_loss

        return {
            'logits': logits,
            'attn_info': attn_info,
            'loss': loss,
            'codebook_loss': codebook_loss,
            'loss_mask': loss_mask,
            'alignment_loss': alignment_loss,
            'audio_codes_target': audio_codes_target,
            'audio_codes_lens_target': audio_codes_lens_target,
            'text': context_tensors['text'],
            'text_lens': context_tensors['text_lens'],
            'context_audio_codes': context_tensors['context_audio_codes'],
            'context_audio_codes_lens': context_tensors['context_audio_codes_lens'],
            'dec_context_size': dec_context_size,
        }

    def training_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        self.log('train_codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        if self.cfg.get('cfg_unconditional_prob', 0.0) == 0.0:
            # Only log alignment loss when not using cfg to avoid sync issues when
            # alignment loss is None on some ranks
            alignment_loss = batch_output['alignment_loss']
            if alignment_loss is not None:
                self.log('train_alignment_loss', alignment_loss, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch, mode="val")
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        alignment_loss = batch_output['alignment_loss']
        logits = batch_output['logits']
        audio_codes_target = batch_output['audio_codes_target']
        audio_codes_lens_target = batch_output['audio_codes_lens_target']
        context_audio_codes = batch_output['context_audio_codes']
        context_audio_codes_lens = batch_output['context_audio_codes_lens']
        attn_info = batch_output['attn_info']
        text_lens = batch_output['text_lens']
        dec_context_size = batch_output['dec_context_size']
        if alignment_loss is None:
            alignment_loss = torch.tensor(0.0, device=loss.device)

        if batch_idx == 0 and self.global_rank == 0:
            self.log_train_val_example(
                logits, audio_codes_target, audio_codes_lens_target, context_audio_codes, context_audio_codes_lens
            )
            if (
                self.model_type != 'decoder_pretrain_synthesizer'
                and len(attn_info[self.transcript_decoder_layers[0]]['cross_attn_probabilities']) > 1
            ):
                # cross_attn_probabilities only returned when not using flash attention
                cross_attention_probs = [
                    attn['cross_attn_probabilities'][0]
                    for layer_idx, attn in enumerate(attn_info)
                    if layer_idx in self.transcript_decoder_layers
                ]
                self.log_attention_probs(
                    cross_attention_probs,
                    audio_codes_lens_target,
                    text_lens,
                    prefix="val_",
                    dec_context_size=dec_context_size,
                )

        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
            'val_alignment_loss': alignment_loss,
        }
        self.validation_step_outputs.append(val_output)

        return val_output

    def infer_batch(self, batch, max_decoder_steps=500, temperature=0.7, topk=80, use_cfg=False, cfg_scale=1.0):
        with torch.no_grad():
            self.decoder.reset_cache(use_cache=self.use_kv_cache_for_inference)

            context_tensors = self.prepare_context_tensors(batch)
            text = context_tensors['text']
            audio_codes_bos = torch.full(
                (text.size(0), self.cfg.num_audio_codebooks, 1), self.audio_bos_id, device=text.device
            ).long()
            audio_codes_lens = torch.full((text.size(0),), 1, device=text.device).long()
            audio_codes_input = audio_codes_bos
            audio_codes_mask = get_mask_from_lengths(audio_codes_lens)

            all_predictions = []
            end_indices = {}

            if use_cfg:
                dummy_cond, dummy_cond_mask, dummy_additional_decoder_input, dummy_addition_dec_mask, _ = (
                    self.prepare_dummy_cond_for_cfg(
                        context_tensors['cond'],
                        context_tensors['cond_mask'],
                        context_tensors['additional_decoder_input'],
                        context_tensors['addtional_decoder_mask'],
                    )
                )

            for idx in range(max_decoder_steps):
                if idx % 20 == 0:
                    print(f"Decoding timestep {idx}")
                audio_codes_embedded = self.embed_audio_tokens(audio_codes_input)
                if context_tensors['additional_decoder_input'] is not None:
                    _audio_codes_embedded = torch.cat(
                        [context_tensors['additional_decoder_input'], audio_codes_embedded], dim=1
                    )
                    _audio_codes_mask = torch.cat([context_tensors['addtional_decoder_mask'], audio_codes_mask], dim=1)
                else:
                    _audio_codes_embedded = audio_codes_embedded
                    _audio_codes_mask = audio_codes_mask

                if use_cfg:
                    batch_size = audio_codes_embedded.size(0)
                    # Combine conditional and unconditional inputs into one batch
                    if isinstance(context_tensors['cond'], list):
                        cfg_cond = [
                            torch.cat([cond_item, dummy_cond_item], dim=0)
                            for cond_item, dummy_cond_item in zip(context_tensors['cond'], dummy_cond)
                        ]
                        cfg_cond_mask = [
                            torch.cat([cond_mask_item, dummy_cond_mask_item], dim=0)
                            for cond_mask_item, dummy_cond_mask_item in zip(
                                context_tensors['cond_mask'], dummy_cond_mask
                            )
                        ]
                    else:
                        cfg_cond = torch.cat([context_tensors['cond'], dummy_cond], dim=0)
                        cfg_cond_mask = torch.cat([context_tensors['cond_mask'], dummy_cond_mask], dim=0)
                    cfg_audio_codes_embedded = torch.cat([_audio_codes_embedded, _audio_codes_embedded], dim=0)
                    cfg_audio_codes_mask = torch.cat([_audio_codes_mask, _audio_codes_mask], dim=0)
                    if dummy_additional_decoder_input is not None:
                        cfg_audio_codes_embedded[batch_size:, : dummy_additional_decoder_input.size(1)] = (
                            dummy_additional_decoder_input
                        )
                        cfg_audio_codes_mask[batch_size:, : dummy_additional_decoder_input.size(1)] = (
                            dummy_addition_dec_mask
                        )

                    combined_logits, _ = self.forward(
                        dec_input_embedded=cfg_audio_codes_embedded,
                        dec_input_mask=cfg_audio_codes_mask,
                        cond=cfg_cond,
                        cond_mask=cfg_cond_mask,
                        attn_prior=None,
                        multi_encoder_mapping=context_tensors['multi_encoder_mapping'],
                    )

                    cond_logits = combined_logits[:batch_size]
                    uncond_logits = combined_logits[batch_size:]
                    all_code_logits = (1 - cfg_scale) * uncond_logits + cfg_scale * cond_logits
                else:
                    all_code_logits, _ = self.forward(
                        dec_input_embedded=_audio_codes_embedded,
                        dec_input_mask=_audio_codes_mask,
                        cond=context_tensors['cond'],
                        cond_mask=context_tensors['cond_mask'],
                        attn_prior=None,
                        multi_encoder_mapping=context_tensors['multi_encoder_mapping'],
                    )
                all_code_logits_t = all_code_logits[:, -1, :]  # (B, num_codebooks * num_tokens_per_codebook)
                audio_codes_next = self.sample_codes_from_logits(
                    all_code_logits_t, temperature=temperature, topk=topk
                )  # (B, num_codebooks)
                all_codes_next_argmax = self.sample_codes_from_logits(
                    all_code_logits_t, temperature=0.01
                )  # (B, num_codebooks)

                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices:
                        pred_token = all_codes_next_argmax[item_idx][0].item()
                        pred_token_multinomial = audio_codes_next[item_idx][0].item()
                        if (pred_token == self.audio_eos_id) or (pred_token_multinomial == self.audio_eos_id):
                            print("End detected for item {} at timestep {}".format(item_idx, idx))
                            end_indices[item_idx] = idx

                all_predictions.append(audio_codes_next)
                audio_codes_input = torch.cat(
                    [audio_codes_input, audio_codes_next.unsqueeze(-1)], dim=-1
                )  # (B, C, T')
                audio_codes_lens = audio_codes_lens + 1
                audio_codes_mask = get_mask_from_lengths(audio_codes_lens)
                if len(end_indices) == text.size(0):
                    print("All ends reached")
                    break

            predicted_codes = torch.stack(all_predictions, dim=-1)  # (B, num_codebooks, T')
            predicted_lens = [end_indices.get(idx, max_decoder_steps) for idx in range(text.size(0))]
            predicted_codes_lens = torch.tensor(predicted_lens, device=text.device).long()

            predicted_audio, predicted_audio_lens = self.codes_to_audio(predicted_codes, predicted_codes_lens)

            torch.cuda.empty_cache()
            return predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            temperature = self.cfg.get('inference_temperature', 0.7)
            topk = self.cfg.get('inference_topk', 80)
            use_cfg = self.cfg.get('inference_use_cfg', False)
            cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)
            predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens = self.infer_batch(
                batch,
                max_decoder_steps=self.cfg.get('max_decoder_steps', 500),
                temperature=temperature,
                topk=topk,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
            )
            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                predicted_audio_np = predicted_audio_np[: predicted_audio_lens[idx]]
                item_idx = batch_idx * test_dl_batch_size + idx
                self.tb_logger.add_audio(
                    'predicted_audio',
                    predicted_audio_np,
                    global_step=item_idx,
                    sample_rate=self.cfg.sample_rate,
                )
                # Save the predicted audio
                log_dir = self.logger.log_dir
                audio_dir = os.path.join(log_dir, 'audios')
                if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir)
                audio_path = os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}.wav')
                sf.write(audio_path, predicted_audio_np, self.cfg.sample_rate)

    def on_validation_epoch_end(self):
        collect = lambda key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        val_loss = collect("val_loss")
        val_codebook_loss = collect("val_codebook_loss")
        val_alignment_loss = collect("val_alignment_loss")
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_codebook_loss", val_codebook_loss, prog_bar=True, sync_dist=True)
        self.log("val_alignment_loss", val_alignment_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def get_dataset(self, cfg, dataset_type):
        dataset = instantiate(
            cfg.dataset,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            audio_bos_id=self.audio_bos_id,
            audio_eos_id=self.audio_eos_id,
            context_audio_bos_id=self.context_audio_bos_id,
            context_audio_eos_id=self.context_audio_eos_id,
            num_audio_codebooks=self.cfg.num_audio_codebooks,
            codec_model_downsample_factor=self.cfg.codec_model_downsample_factor,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
        )
        dataset.load_16khz_audio = self.model_type == 'single_encoder_sv_tts'
        dataset.tokenizer_config = (
            self.cfg.text_tokenizers
        )  # This will be used in worker_init_fn for instantiating tokenizer
        return dataset

    def _setup_train_dataloader(self, cfg):
        dataset = self.get_dataset(cfg, dataset_type='train')
        sampler = dataset.get_sampler(cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
        persistent_workers = True
        if cfg.dataloader_params.num_workers == 0:
            persistent_workers = False
            # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
            dataset.text_tokenizer, dataset.text_conditioning_tokenizer = self._setup_tokenizers(self.cfg)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            **cfg.dataloader_params,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
        )
        return data_loader

    def _setup_test_dataloader(self, cfg):
        dataset = self.get_dataset(cfg, dataset_type='test')
        persistent_workers = True
        if cfg.dataloader_params.num_workers == 0:
            persistent_workers = False
            # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
            dataset.text_tokenizer, dataset.text_conditioning_tokenizer = self._setup_tokenizers(self.cfg, mode='test')

        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            **cfg.dataloader_params,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
        )
        return data_loader

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_test_dataloader(cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []


class MagpieTTS_ModelInference(MagpieTTS_Model):
    """Small override of MagpieTTS_Model for parallel multi-GPU inference and metrics calculation.
    This class is used in 'test' mode and leverages trainer.test() for multi-GPU/multi-node inference.
    Saves the predicted audio files and logs the CER/WER metrics as individual json files for each audio.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg, trainer)
        if cfg.get('pref_set_language', "en") == "en":
            self.eval_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-1.1b"
            )
            self.eval_asr_model.freeze()
            self.eval_asr_model.eval()

        self.eval_speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name='titanet_large'
        )
        self.eval_speaker_verification_model.freeze()
        self.eval_speaker_verification_model.eval()

        if cfg.get('load_whisper_model', False):
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            self.whisper_model.eval()

    def transcribe_with_whisper(self, audio_filepath, language):
        speech_array, sampling_rate = librosa.load(audio_filepath, sr=16000)
        forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language=language) if language else None
        inputs = self.whisper_processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
        inputs = inputs.to(self.device)
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        result = transcription[0]
        return result

    def process_text(self, input_text):
        """
        Normalizes text for CER/WER calculation.
        Taken from hallucination_eval.py
        """
        # Convert text to lowercase
        lower_case_text = input_text.lower()

        # Remove commas from text
        no_comma_text = lower_case_text.replace(",", "")
        # Replace "-" with spaces
        no_dash_text = no_comma_text.replace("-", " ")
        no_dash_text = no_dash_text.replace("'", "")
        no_dash_text = no_dash_text.replace(";", "")
        no_dash_text = no_dash_text.replace(".", "")

        # Replace double spaces with single space
        single_space_text = " ".join(no_dash_text.split())

        single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))

        # @shehzeen: Added this to handle some common errors in ASR transcripts
        single_space_text.replace("h t t p", "http")
        single_space_text.replace("w w w", "www")

        return single_space_text

    def get_speaker_embeddings_from_filepaths(self, filepaths):
        audio_batch = []
        audio_lengths = []
        for filepath in filepaths:
            audio, sr = sf.read(filepath)
            if sr != 16000:
                audio = librosa.core.resample(audio, orig_sr=sr, target_sr=16000)
            audio_tensor = torch.tensor(audio, dtype=torch.float32, device=self.device)
            audio_batch.append(audio_tensor)
            audio_lengths.append(audio_tensor.size(0))

        batch_audio_lens = torch.tensor(audio_lengths, device=self.device).long()
        max_audio_len = int(batch_audio_lens.max().item())
        audio_batch = stack_tensors(audio_batch, max_lens=[max_audio_len])

        _, speaker_embeddings = self.eval_speaker_verification_model.forward(
            input_signal=audio_batch, input_signal_length=batch_audio_lens
        )

        return speaker_embeddings

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            temperature = self.cfg.get('inference_temperature', 0.7)
            topk = self.cfg.get('inference_topk', 80)
            use_cfg = self.cfg.get('inference_use_cfg', False)
            cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)
            predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens = self.infer_batch(
                batch,
                max_decoder_steps=self.cfg.get('max_decoder_steps', 500),
                temperature=temperature,
                topk=topk,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
            )
            predicted_audio_paths = []
            audio_durations = []
            batch_invalid = False
            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                predicted_audio_np = predicted_audio_np[: predicted_audio_lens[idx]]
                item_idx = batch_idx * test_dl_batch_size + idx
                # Save the predicted audio
                log_dir = self.logger.log_dir
                audio_dir = os.path.join(log_dir, 'audios')
                if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir)
                audio_path = os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}.wav')
                audio_durations.append(len(predicted_audio_np) / self.cfg.sample_rate)
                sf.write(audio_path, predicted_audio_np, self.cfg.sample_rate)

                predicted_codes_torch = predicted_codes[idx].cpu().type(torch.int16)
                predicted_codes_torch = predicted_codes_torch[:, : predicted_codes_lens[idx]]
                torch.save(
                    predicted_codes_torch,
                    os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}_codes.pt'),
                )
                predicted_audio_paths.append(audio_path)

                if not batch_invalid:
                    with torch.no_grad():
                        try:
                            if self.cfg.get("pref_set_language", "en") == "en":
                                pred_transcripts = self.eval_asr_model.transcribe(
                                    predicted_audio_paths, batch_size=len(predicted_audio_paths)
                                )[0]
                                pred_transcripts = [self.process_text(transcript) for transcript in pred_transcripts]
                            else:
                                pred_transcripts = [
                                    self.transcribe_with_whisper(audio_path, self.cfg.pref_set_language)
                                    for audio_path in predicted_audio_paths
                                ]
                                pred_transcripts = [self.process_text(transcript) for transcript in pred_transcripts]
                        except Exception as e:
                            assert (
                                predicted_audio_lens[idx] < 1000
                            ).any(), f"Expected short audio file to be the only cause of ASR errors, but got error with lengths {predicted_audio_lens}"
                            logging.warning(f"Exception during ASR transcription: {e}")
                            logging.warning(
                                "Skipping processing of the batch; generating metrics indicating a WER of 100% and "
                                "Speaker Similarity of 0.0"
                            )
                            batch_invalid = True
                            continue  # don't break since we want to continue building audio durations list
                        pred_speaker_embeddings = self.get_speaker_embeddings_from_filepaths(predicted_audio_paths)
                        gt_speaker_embeddings = self.get_speaker_embeddings_from_filepaths(batch['audio_filepaths'])

            for idx in range(predicted_audio.size(0)):
                if not batch_invalid:
                    item_idx = batch_idx * test_dl_batch_size + idx
                    pred_transcript = pred_transcripts[idx]
                    gt_transcript = self.process_text(batch['raw_texts'][idx])

                    cer_gt = word_error_rate([pred_transcript], [gt_transcript], use_cer=True)
                    wer_gt = word_error_rate([pred_transcript], [gt_transcript], use_cer=False)

                    spk_embedding_pred = pred_speaker_embeddings[idx].cpu().numpy()
                    spk_embedding_gt = gt_speaker_embeddings[idx].cpu().numpy()

                    spk_similarity = np.dot(spk_embedding_pred, spk_embedding_gt) / (
                        np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt)
                    )
                else:
                    # Create an entry indicating invalid metrics
                    cer_gt = 1.0
                    wer_gt = 1.0
                    spk_similarity = 0.0
                    pred_transcript = "<INVALID>"  # do not change this string; subsequent processing relies on it
                    gt_transcript = self.process_text(batch['raw_texts'][idx])

                item_metrics = {
                    'cer_gt': float(cer_gt),
                    'wer_gt': float(wer_gt),
                    'duration': audio_durations[idx],
                    'spk_similarity': float(spk_similarity),
                    'pred_transcript': pred_transcript,
                    'gt_transcript': gt_transcript,
                }

                with open(
                    os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}_metrics.json'), 'w'
                ) as f:
                    json.dump(item_metrics, f)


class MagpieTTS_ModelDPO(MagpieTTS_Model):
    """Extends MagpieTTS_Model to support Direct Preference Optimization (DPO) training.
    This class is used for training the model with preference-based losses, including DPO, RPO, and IPO losses.
    It maintains a frozen reference model to compare log probabilities between policy and reference outputs.

    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        """Initialize the MagpieTTS_ModelDPO class.

        Args:
            cfg (DictConfig): Configuration object containing model hyperparameters.
            trainer (Trainer, optional): Trainer instance for model training.
        """
        super().__init__(cfg, trainer)
        # Create a copy of the configuration for the reference model
        ref_model_cfg = copy.deepcopy(cfg)
        with open_dict(ref_model_cfg):
            ref_model_cfg.train_ds = None
            ref_model_cfg.validation_ds = None

        # Initialize the frozen reference model
        self._reference_model = MagpieTTS_Model(cfg=ref_model_cfg)
        print("Loading reference model from checkpoint")
        self._reference_model.load_state_dict(
            torch.load(cfg.reference_model_ckpt_path, map_location="cpu")['state_dict']
        )
        self.freeze_model(self._reference_model)
        self._reference_model.eval()
        self._reference_model._no_state_dict = True
        print("Reference model loaded and frozen")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return the state dictionary excluding non-trainable components.

        Excludes state keys related to `_speaker_verification_model`, `_codec_model`, and `_reference_model`.

        Args:
            destination (dict, optional): The destination dictionary for the state_dict.
            prefix (str, optional): Prefix to prepend to keys.
            keep_vars (bool, optional): If True, tensors in the returned dictionary will not be detached.

        Returns:
            dict: Filtered state dictionary.
        """
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model', '_reference_model']
        for key in list(state_dict.keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del state_dict[key]
        return state_dict

    def _get_batch_logps(self, logits, labels, loss_mask, average_log_prob=False):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored.
                Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return
                the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under
            the given logits.
        """
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def preference_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        chosen_gt_rewards=None,
        rejected_gt_rewards=None,
        beta=0.2,
        gt_reward_scale=1.0,
        label_smoothing=0,
        loss_type="dpo",
        reference_free=False,
    ):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses.
                Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses.
                Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses.
                Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses.
                Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore
                the reference model as beta -> 0.
            label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with
                probability label_smoothing)
            ipo: If True, use the IPO loss instead of the DPO loss.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model
                that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}
        # logits = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
        # logits = (policy_chosen_logps - reference_chosen_logps) - (policy_rejected_logps - reference_rejected_logps)
        # logits is the same as rewards_delta in NeMo aligner
        # https://github.com/NVIDIA/NeMo-Aligner/blob/0b5bffeb78a8316dd57e0816a2a9544540f0c8dd/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L241

        if loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        elif loss_type == "rpo":
            # https://github.com/NVIDIA/NeMo-Aligner/blob/0b5bffeb78a8316dd57e0816a2a9544540f0c8dd/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L241
            logbeta_hat_chosen = torch.nn.functional.logsigmoid(beta * logits)
            logbeta_hat_rejected = torch.nn.functional.logsigmoid(-beta * logits)
            gt_rewards_delta = gt_reward_scale * (chosen_gt_rewards - rejected_gt_rewards)
            logalpha_hat_chosen = torch.nn.functional.logsigmoid(gt_rewards_delta)
            logalpha_hat_rejected = torch.nn.functional.logsigmoid(-gt_rewards_delta)
            losses = torch.exp(logalpha_hat_chosen) * (logalpha_hat_chosen - logbeta_hat_chosen) + torch.exp(
                logalpha_hat_rejected
            ) * (logalpha_hat_rejected - logbeta_hat_rejected)
        elif loss_type == "rpo_sq":
            gt_rewards_delta = gt_reward_scale * (chosen_gt_rewards - rejected_gt_rewards)
            losses = (beta * logits - gt_rewards_delta) ** 2
        elif loss_type == "dpo":
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf;
            # label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            F = torch.nn.functional
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
            )
        else:
            raise NotImplementedError("loss type {} is not implemented".format(loss_type))

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def process_batch_dpo(self, batch_chosen_rejected):
        """Process a batch for Direct Preference Optimization (DPO) training.

        This method computes the preference loss by comparing the model's policy outputs with a frozen reference model.
        It processes chosen and rejected samples, extracts log probabilities for each codebook, and calculates the
        preference loss based on the difference in likelihoods between chosen and rejected responses.

        Args:
            batch_chosen_rejected (dict): A dictionary containing two keys:
                - 'chosen': The batch of chosen responses.
                - 'rejected': The batch of rejected responses.

        Returns:
            dict: A dictionary containing:
                - 'loss': The total computed loss.
                - 'pref_loss': The preference loss.
                - 'sft_loss': The supervised fine-tuning loss.
                - 'alignment_loss': The alignment loss, if applicable.
        """
        batch_chosen = batch_chosen_rejected['chosen']
        batch_rejected = batch_chosen_rejected['rejected']

        model_output_chosen = self.process_batch(batch_chosen)
        model_output_rejected = self.process_batch(batch_rejected)
        with torch.no_grad():
            reference_model_output_chosen = self._reference_model.process_batch(batch_chosen)
            reference_model_output_rejected = self._reference_model.process_batch(batch_rejected)

        chosen_policy_logprobs = None
        rejected_policy_logprobs = None
        chosen_ref_logprobs = None
        rejected_ref_logprobs = None
        for codebook_idx in range(self.cfg.num_audio_codebooks):
            si = codebook_idx * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits_chosen = model_output_chosen['logits'][:, :, si:ei]
            codebook_logits_rejected = model_output_rejected['logits'][:, :, si:ei]

            ref_codebook_logits_chosen = reference_model_output_chosen['logits'][:, :, si:ei]
            ref_codebook_logits_rejected = reference_model_output_rejected['logits'][:, :, si:ei]

            codebook_labels_chosen = model_output_chosen['audio_codes_target'][:, codebook_idx]
            codebook_labels_rejected = model_output_rejected['audio_codes_target'][:, codebook_idx]

            codebook_log_probs_chosen = self._get_batch_logps(
                codebook_logits_chosen, codebook_labels_chosen, model_output_chosen['loss_mask']
            )
            codebook_log_probs_rejected = self._get_batch_logps(
                codebook_logits_rejected, codebook_labels_rejected, model_output_rejected['loss_mask']
            )
            with torch.no_grad():
                ref_codebook_log_probs_chosen = self._get_batch_logps(
                    ref_codebook_logits_chosen, codebook_labels_chosen, reference_model_output_chosen['loss_mask']
                )
                ref_codebook_log_probs_rejected = self._get_batch_logps(
                    ref_codebook_logits_rejected,
                    codebook_labels_rejected,
                    reference_model_output_rejected['loss_mask'],
                )

            if chosen_policy_logprobs is None:
                chosen_policy_logprobs = codebook_log_probs_chosen
                rejected_policy_logprobs = codebook_log_probs_rejected
                chosen_ref_logprobs = ref_codebook_log_probs_chosen
                rejected_ref_logprobs = ref_codebook_log_probs_rejected
            else:
                chosen_policy_logprobs += codebook_log_probs_chosen
                rejected_policy_logprobs += codebook_log_probs_rejected
                chosen_ref_logprobs += ref_codebook_log_probs_chosen
                rejected_ref_logprobs += ref_codebook_log_probs_rejected

        rewards_chosen = batch_chosen['rewards']
        rewards_rejected = batch_rejected['rewards']

        assert torch.all(rewards_chosen == 1)
        assert torch.all(rewards_rejected < 1)

        pref_loss, chosen_rewards, rejected_rewards = self.preference_loss(
            chosen_policy_logprobs,
            rejected_policy_logprobs,
            chosen_ref_logprobs,
            rejected_ref_logprobs,
            chosen_gt_rewards=rewards_chosen,
            rejected_gt_rewards=rewards_rejected,
            beta=self.cfg.get('dpo_beta', 0.01),
            loss_type=self.cfg.get('dpo_loss_type', 'dpo'),
        )

        pref_loss = pref_loss.mean()
        sft_loss = -chosen_policy_logprobs.mean()

        pref_loss_weight = self.cfg.get('dpo_pref_loss_weight', 1.0)
        sft_loss_weight = self.cfg.get('dpo_sft_loss_weight', 0.0)
        loss = pref_loss_weight * pref_loss + sft_loss * sft_loss_weight

        alignment_loss = model_output_chosen['alignment_loss']
        if alignment_loss is not None:
            loss += alignment_loss

        return {
            'loss': loss,
            'pref_loss': pref_loss,
            'sft_loss': sft_loss,
            'alignment_loss': alignment_loss,
        }

    def training_step(self, batch, batch_idx):
        """Perform a training step using DPO loss.

        Args:
            batch (dict): Batch data containing chosen and rejected samples.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Training loss.
        """
        dpo_outputs = self.process_batch_dpo(batch)
        self.log('train_loss', dpo_outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train_pref_loss', dpo_outputs['pref_loss'], prog_bar=True, sync_dist=True)
        self.log('train_sft_loss', dpo_outputs['sft_loss'], prog_bar=True, sync_dist=True)
        return dpo_outputs['loss']

    def validation_step(self, batch, batch_idx):
        """Perform a validation step using DPO loss.

        Args:
            batch (dict): Validation batch data.
            batch_idx (int): Batch index.
        """
        dpo_outputs = self.process_batch_dpo(batch)

        val_loss = dpo_outputs['loss']
        val_pref_loss = dpo_outputs['pref_loss']
        val_sft_loss = dpo_outputs['sft_loss']
        val_alignment_loss = dpo_outputs['alignment_loss']

        self.validation_step_outputs.append(
            {
                'val_loss': val_loss,
                'val_pref_loss': val_pref_loss,
                'val_sft_loss': val_sft_loss,
                'val_alignment_loss': val_alignment_loss,
            }
        )

    def on_validation_epoch_end(self):
        """Aggregate validation losses at the end of the validation epoch."""

        def collect(key):
            values = []
            for x in self.validation_step_outputs:
                if x[key] is not None:
                    values.append(x[key])
                else:
                    values.append(torch.tensor(0.0, device=self.device))
            stacked_values = torch.stack(values)
            return stacked_values.mean()

        val_loss = collect("val_loss")
        val_pref_loss = collect("val_pref_loss")
        val_sft_loss = collect("val_sft_loss")
        val_alignment_loss = collect("val_alignment_loss")
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_pref_loss", val_pref_loss, prog_bar=True, sync_dist=True)
        self.log("val_sft_loss", val_sft_loss, prog_bar=True, sync_dist=True)
        if val_alignment_loss is not None:
            self.log("val_alignment_loss", val_alignment_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
