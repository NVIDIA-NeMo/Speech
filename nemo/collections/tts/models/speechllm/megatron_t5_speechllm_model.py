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
#
# flake8: noqa

import itertools
import json
import os
import random
import string
from functools import partial
from typing import Any, List

import editdistance
import imageio
import numpy as np
import soundfile as sf
import torch
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceSpeechLLMTTSTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import (
    MegatronTokenLevelEncoderDecoderSpeechLLMModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    init_method_normal,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.tts.data.speechllm.t5_speechllm_dataset import Lang, T5SpeechLMDataset
from nemo.collections.tts.data.speechllm.t5_speechllm_tarred_dataset import T5SpeechLMTarredDataset
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.models.speechllm.megatron_base_speechllm_prompt_model import MegatronBaseSpeechLM
from nemo.collections.tts.parts.utils.helpers import plot_alignment_to_numpy_for_speechllm, plot_codec_to_numpy
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


import time

import librosa
from torchaudio.pipelines import SQUIM_SUBJECTIVE
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

__all__ = ['MegatronT5SpeechLMModel']


class MegatronT5OverrideModel(MegatronT5Model):
    def _build_tokenizer(self):
        if self._cfg.tokenizer.library == "sentencepiece":
            if hasattr(self._cfg.tokenizer, "sentencepiece_legacy"):
                legacy = self._cfg.tokenizer.sentencepiece_legacy
            else:
                legacy = True if self._cfg.tokenizer.library == 'sentencepiece' else False
            self.tokenizer = SentencePieceSpeechLLMTTSTokenizer(
                model_path=self.register_artifact("tokenizer.model", self._cfg.tokenizer.get('model', None)),
                legacy=legacy,
            )

            if self._cfg.tokenizer.get('additional_special_tokens', None) is not None:
                tokens_list = OmegaConf.to_object(self._cfg.tokenizer.additional_special_tokens)
                self.tokenizer.add_special_tokens(tokens_list)
        else:
            super()._build_tokenizer()

    def model_provider_func(self, pre_process, post_process, add_encoder, add_decoder):
        if not hasattr(self.cfg, 'encoder') or not hasattr(self.cfg, 'decoder'):
            logging.warning(
                'Could not find encoder or decoder in config. This is probably because of restoring an old checkpoint. Copying shared model configs to encoder and decoder configs.'
            )
            # After the call below, self.cfg.encoder and self.cfg.decoder will be populated with the cfg.model configs from old checkpoints.
            self._populate_encoder_decoder_configs_for_backward_compatibility(self.cfg)

        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and self.cfg.encoder.arch == 'perceiver':
            raise ValueError(f"Perceivers with pipeline parallel > 1 is not supported yet.")

        if not hasattr(self.cfg, 'embedding_init_method_std'):
            embedding_init_method_std = self.cfg.encoder.init_method_std
        else:
            embedding_init_method_std = self.cfg.embedding_init_method_std

        if not hasattr(self.cfg, 'embedding_dropout'):
            embedding_dropout = self.cfg.encoder.hidden_dropout
        else:
            embedding_dropout = self.cfg.embedding_dropout

        model = MegatronTokenLevelEncoderDecoderSpeechLLMModule(
            config=self.model_parallel_config,
            encoder_cfg=self.cfg.encoder,
            decoder_cfg=self.cfg.decoder,
            vocab_size=self.padded_vocab_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            fp16_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            precision=self.cfg.get('precision', 16),
            embedding_init_method_std=embedding_init_method_std,
            embedding_dropout=embedding_dropout,
            label_smoothing=self.cfg.get('label_smoothing', 0.0),
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            share_token_embeddings=self.cfg.get('share_token_embeddings', True),
            share_decoder_tokens_head_embeddings=self.cfg.get('share_decoder_tokens_head_embeddings', True),
            tokens_head_bias=self.cfg.get('tokens_head_bias', True),
            hiddens_cfg=self.cfg.get('hiddens', None),
        )
        return model


class MegatronT5SpeechLMModel(MegatronBaseSpeechLM):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron T5 model.

    Prompt Tuning initializes virtual prompt embeddings directly from a copy of
    certain token embeddings from the pretrained T5 model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in
    initialization are specified by the user in the config file. The model can
    be prompt-tuned for multiple tasks at once. Virtual prompts are stored in a
    prompt table and can be added or deleted without disrupting virtual prompts
    for other tasks.

    P-tuning initializes an LSTM encoder model that generates virtual prompt
    embeddings for every task. Each task shares the same encoder. After p-tuning
    is complete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a
    new virtual prompt via p-tuning, they do not need to retrain on all previous
    tasks. This gives p-tuning the same task flexibility as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.model_type = ModelType.encoder_or_decoder
        speech_codebook_size = cfg.data.get('speech_codebook_size', 1024)
        num_speech_codebooks = cfg.data.get('num_speech_codebooks', 8)
        speech_offset = cfg.data.get('speech_offset', 30000)
        codecmodel_type = cfg.get('codecmodel_type', 'nemo_codec')
        attn_prior_scaledown_start_step = cfg.get('attn_prior_scaledown_start_step', 10000)
        attn_prior_end_step = cfg.get('attn_prior_end_step', 11000)
        num_cross_attention_heads = cfg.get('num_cross_attention_heads', 12)
        self.lm_vocab_size = cfg.get('lm_vocab_size', 30000)
        self.context_pattern = cfg.data.get('context_pattern', 'parallel')
        self.context_conditioning = cfg.get('context_conditioning', "decoder")
        self.context_duration_min = cfg.data.get('context_duration_min', 2.9)
        self.context_duration_max = cfg.data.get('context_duration_max', 2.9)
        self.codebook_fps = cfg.data.get('codebook_fps', 86)
        self.decoder_context_len = 0
        if self.context_conditioning == "decoder":
            assert self.context_duration_min == self.context_duration_max, "Decoder context duration must be fixed"
            self.decoder_context_len = int(self.codebook_fps * self.context_duration_min)

        self.speech_offset = speech_offset
        self.speech_codebook_size = speech_codebook_size
        self.num_speech_codebooks = num_speech_codebooks
        self.codecmodel_type = codecmodel_type
        self.enc_output_to_layers = cfg.get('enc_output_to_layers', None)
        if self.enc_output_to_layers is not None:
            # Convert from listconfig to list
            self.enc_output_to_layers = [[l for l in encoder_layer] for encoder_layer in self.enc_output_to_layers]

        self.frozen_model.enc_dec_model.speech_offset = speech_offset
        self.frozen_model.enc_dec_model.speech_codebook_size = speech_codebook_size
        self.frozen_model.enc_dec_model.num_speech_codebooks = num_speech_codebooks
        self.frozen_model.enc_dec_model.seq_pattern = cfg.get('seq_pattern', 'parallel')
        self.frozen_model.enc_dec_model.attn_prior_scaledown_start_step = attn_prior_scaledown_start_step
        self.frozen_model.enc_dec_model.attn_prior_end_step = attn_prior_end_step
        self.frozen_model.enc_dec_model.alignment_decoder_layerids = cfg.get(
            'alignment_decoder_layerids', list(range(0, 12))
        )
        self.frozen_model.enc_dec_model.return_all_crossattention_probs = cfg.get(
            'return_all_crossattention_probs', False
        )
        self.frozen_model.enc_dec_model.num_cross_attention_heads = num_cross_attention_heads
        self.frozen_model.enc_dec_model.context_conditioning = self.context_conditioning
        self.frozen_model.enc_dec_model.decoder_context_len = self.decoder_context_len
        self.frozen_model.enc_dec_model.enc_output_to_layers = self.enc_output_to_layers

        self.alignment_loss_start_step = 0
        self.alignment_loss_end_step = float('inf')
        self.use_alignment_loss = cfg.get('use_alignment_loss', False)
        if self.use_alignment_loss:
            alignment_loss_scale = cfg.get('alignment_loss_scale', 1.0)
            self.frozen_model.enc_dec_model.use_alignment_loss = True
            self.frozen_model.enc_dec_model.forward_sum_loss = ForwardSumLoss(loss_scale=alignment_loss_scale)
            self.frozen_model.enc_dec_model.alignment_text_end_offset = cfg.get('alignment_text_end_offset', 0)
            self.frozen_model.enc_dec_model.align_every_n_head = cfg.get('align_every_n_head', 1)
            self.alignment_loss_start_step = cfg.get('alignment_loss_start_step', 0)
            self.alignment_loss_end_step = cfg.get('alignment_loss_end_step', float('inf'))

        # Need to explicitly set this since it is already initialized
        self.frozen_model.enc_dec_model.tokens_head.parallel_output = self.frozen_model.enc_dec_model.parallel_output

        list_of_speech_heads = []
        list_of_speech_tokens_embeddings = []
        for _ in range(self.num_speech_codebooks - 1):
            # init is NOT used since we overwrite the weight below anyways
            _speech_head_embedding = tensor_parallel.VocabParallelEmbedding(
                speech_codebook_size,
                embedding_dim=self.word_embeddings.embedding_dim,
                init_method=lambda x: x.data.fill_(0),
                config=self.model_parallel_config,
            )
            _speech_head_embedding.weight.data.fill_(0)
            _speech_head_embedding.shared = True
            list_of_speech_tokens_embeddings.append(_speech_head_embedding)
            # Linear layer that maps from hidden size to speech codebook size
            hidden_size = self.frozen_model.enc_dec_model.decoder_cfg.hidden_size
            init_method_std = self.frozen_model.enc_dec_model.decoder_cfg.init_method_std
            # Changing to ColumnParallelLinear instead of Linear to support 3b Tensor Parallelism
            _speech_head = tensor_parallel.ColumnParallelLinear(
                input_size=hidden_size,
                output_size=speech_codebook_size,
                bias=True,
                gather_output=not self.frozen_model.enc_dec_model.parallel_output,
                init_method=init_method_normal(init_method_std),
                config=self.model_parallel_config,
            )
            list_of_speech_heads.append(_speech_head)

        self.frozen_model.enc_dec_model.speech_tokens_heads = torch.nn.ModuleList(list_of_speech_heads)
        self.frozen_model.enc_dec_model.speech_tokens_embeddings = torch.nn.ModuleList(
            list_of_speech_tokens_embeddings
        )

        self.sample_rate = 24000
        if codecmodel_type == 'nemo_codec':
            codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'))
            codec_model.to('cuda')
            codec_model.eval()
            self.sample_rate = 22050
        else:
            raise NotImplementedError()

        self.additional_models = {'codec': codec_model}
        self.train_check_interval = self.cfg.get('train_check_interval', 500)
        self.plot_alignments_sliced = self.cfg.get('plot_alignments_sliced', True)
        app_state = AppState()
        self.is_rank_zero = app_state.global_rank == 0
        self.predict_step_outputs = []
        self.phoneme_tokenizer = None

        # classifier-free guidance (CFG) option during training. The probability (0.0 <= ε <= 1.0) is used to trigger the action that the
        # text or audio tokens in a batch are replaced by [UNK], such that mimicking the text- or audio-free scenario.
        # If a random number is greater than ε, then keep text or audio tokens as-is, otherwise, the text or audio tokens are
        # replaced by [UNK]. Default to 0.0, meaning CFG is disabled.
        self.train_text_cfg_prob = cfg.get('train_text_cfg_prob', 0.0)
        self.train_audio_cfg_prob = cfg.get('train_audio_cfg_prob', 0.0)
        self._rng = random.Random()

        # control the strength of the classifier guidance during inference, Logits_cfg = w*Logits_cond + (1-w)*Logits_uncond,
        # equivalent to Logits_cfg = Logits_cond + alpha*(Logits_cond - Logits_uncond) where alpha=w-1.
        # Default w to 1.O, indicating no interpolation is applied.
        self.inference_cfg_interpolation_scale = cfg.get('inference_cfg_interpolation_scale', 1.0)
        self.inference_apply_text_cfg = cfg.get('inference_apply_text_cfg', False)
        self.inference_apply_audio_cfg = cfg.get('inference_apply_audio_cfg', False)
        if self.inference_cfg_interpolation_scale == 1.0:
            self.inference_apply_text_cfg = False
            self.inference_apply_audio_cfg = False

        # whether to apply cfg filter to address faster speech rate.
        self.inference_apply_cfg_filter = cfg.get("inference_apply_cfg_filter", False)

        # this scale is suggested to be smaller than `self.question_guidance_scale` and it is used to balance the weights
        # between the conditioned logits after applying cfg filter and the original unconditioned logits. Default to 1.0,
        # indicating only conditioned logits are used.
        if not self.inference_apply_cfg_filter:
            self.inference_cfg_filter_interpolation_scale = None
        else:
            self.inference_cfg_filter_interpolation_scale = cfg.get('inference_cfg_filter_interpolation_scale', 1.0)

        # whether to estimate MOS in predict_step.
        self.estimate_mos = cfg.get('estimate_mos', True)
        if self.estimate_mos:
            # requires to specify a non-matching high-quality and clean reference audio file. It is used to estimate MOS.
            self.non_matching_ref_audio_filepath = cfg.get('non_matching_ref_audio_filepath', None)
            if self.non_matching_ref_audio_filepath is None:
                raise ValueError(
                    f"Please provide a high-quality reference audio to estimate the MOS. Alternatively, "
                    f"set `model.estimate_mos=False` to disable MOS estimation."
                )
            if not os.path.exists(self.non_matching_ref_audio_filepath):
                raise FileNotFoundError(
                    f"Please provide a valid file path for a high-quality reference audio to estimate"
                    f" the MOS. Alternatively, set `model.estimate_mos=False` to disable MOS estimation."
                )

    def decode_wav_from_codec_model(self, codes):
        codec_model = self.additional_models['codec']
        if self.codecmodel_type == 'nemo_codec':
            codec_len = torch.Tensor([codes.shape[1]]).long().cuda()
            if codec_len < 10:
                # return a one-second silence
                return torch.zeros(24000).cuda()
            wav, _ = codec_model.decode(tokens=codes.unsqueeze(0), tokens_len=codec_len)
            wav = wav[0]
        else:
            raise NotImplementedError()
        return wav

    def first_stage_of_pipeline(self):
        if self.frozen_model.enc_dec_model.pre_process and parallel_state.get_pipeline_model_parallel_rank() == 0:
            return True
        return False

    def forward(
        self,
        virtual_tokens,
        context_and_question_tokens,
        enc_mask,
        dec_input,
        dec_mask,
        position_ids,
        taskname_ids,
        labels=None,
        speech_mask=None,
        inference=False,
        inference_step=0,
        cross_attention_prior=None,
        text_limits=None,
        decoder_max_sequence_len=None,
        encoder_max_sequence_len=None,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        T5 style models.
        """
        if isinstance(context_and_question_tokens, list):
            multi_encoder = True
            assert isinstance(enc_mask, list)
            assert isinstance(position_ids, list)
            if cross_attention_prior is None:
                cross_attention_prior = [None for _ in range(len(context_and_question_tokens))]
            assert isinstance(cross_attention_prior, list)
            assert len(context_and_question_tokens) == len(enc_mask) == len(position_ids) == len(cross_attention_prior)
        else:
            multi_encoder = False
            context_and_question_tokens = [context_and_question_tokens]
            enc_mask = [enc_mask]
            position_ids = [position_ids]
            cross_attention_prior = [cross_attention_prior]

        enc_output = None
        logging.debug(
            f"self.first_stage_of_pipeline()={self.first_stage_of_pipeline()}\tinference_step={inference_step}"
        )
        if self.first_stage_of_pipeline() and inference_step == 0:
            # Get embeddings for text tokens and insert virtual token embeddings
            encoder_input_list = []
            for ei in range(len(context_and_question_tokens)):
                input_embeds = self.get_embeddings_and_combine(
                    [virtual_tokens, context_and_question_tokens[ei]], taskname_ids, inference
                )
                # TODO: This check needs to be revisited with PP support.
                if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                    position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                        position_ids[ei]
                    )
                    encoder_input = input_embeds + position_embeddings
                else:
                    encoder_input = input_embeds
                encoder_input_list.append(encoder_input)
        else:
            encoder_input_list = None
            encoder_input = None
            if inference_step != 0:
                enc_output = context_and_question_tokens if multi_encoder else context_and_question_tokens[0]

        # If the decoder input starts with <pad> instead of <bos>, which is the case for huggingface T5 models, we don't want to mask the first token.
        # For NeMo-Megatron, the sequence starts with <bos>, which is never masked so we can always set index 0 to be unmasked.
        dec_mask[:, 0] = 1

        if not self.cfg.data.get('use_attention_prior', False):
            cross_attention_prior = [None for _ in range(len(cross_attention_prior))]

        _encoder_input = encoder_input_list
        if not multi_encoder:
            enc_mask = enc_mask[0]
            cross_attention_prior = cross_attention_prior[0]
            _encoder_input = encoder_input_list[0] if encoder_input_list is not None else None

        # Call forward on T5 model with preprocessed embeddings
        if inference and inference_step == 0:
            set_inference_key_value_memory = True
        else:
            set_inference_key_value_memory = False

        if self.autocast_dtype == torch.float32:
            output, out_logits = self.frozen_model.enc_dec_model(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=_encoder_input,
                enc_output=enc_output,
                speech_mask=speech_mask,
                cross_attention_prior=cross_attention_prior,
                text_limits=text_limits,
                global_step=self.global_step,
                set_inference_key_value_memory=set_inference_key_value_memory,
                decoder_max_sequence_len=decoder_max_sequence_len,
                encoder_max_sequence_len=encoder_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output, out_logits = self.frozen_model.enc_dec_model(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=_encoder_input,
                    enc_output=enc_output,
                    speech_mask=speech_mask,
                    cross_attention_prior=cross_attention_prior,
                    text_limits=text_limits,
                    global_step=self.global_step,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    decoder_max_sequence_len=decoder_max_sequence_len,
                    encoder_max_sequence_len=encoder_max_sequence_len,
                )

        return output, encoder_input, out_logits

    def load_frozen_model(self, cfg, trainer):
        self.megatron_amp_O2 = cfg.get('megatron_amp_o2', False)

        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        cfg_language_model_path = cfg.get('language_model_path', None)
        cfg_frozen_model = cfg.get('frozen_model', None)
        if not (bool(cfg_language_model_path) ^ bool(cfg_frozen_model)):
            raise ValueError(
                "T5-TTS requires either 'language_model_path' or 'frozen_model' in its config, but not both."
            )

        if cfg_language_model_path:
            t5_cfg = MegatronT5Model.restore_from(cfg_language_model_path, trainer=trainer, return_config=True)
        else:
            t5_cfg = cfg_frozen_model

        OmegaConf.set_struct(t5_cfg, True)
        with open_dict(t5_cfg):
            if hasattr(t5_cfg, 'encoder') and hasattr(t5_cfg, 'decoder'):
                t5_cfg.encoder.masked_softmax_fusion = False
                t5_cfg.decoder.masked_softmax_fusion = False
            else:
                t5_cfg.masked_softmax_fusion = False
            t5_cfg.megatron_amp_O2 = self.megatron_amp_O2
            # hack to make the _GLOBAL_NUM_MICROBATCHES_CALCULATOR initialize
            t5_cfg.micro_batch_size = cfg.get('micro_batch_size', 4)
            t5_cfg.global_batch_size = cfg.get('global_batch_size', 4)
            t5_cfg.precision = trainer.precision
            t5_cfg.tokenizer.num_sentinel_tokens = cfg.get('num_sentinel_tokens', 39184 - 29056)
            t5_cfg.seq_length = cfg.data.max_seq_length
            if cfg.get('max_position_embeddings', None) is None:
                t5_cfg.max_position_embeddings = cfg.data.max_seq_length
            else:
                t5_cfg.max_position_embeddings = cfg.get('max_position_embeddings')
            t5_cfg.use_flash_attention = cfg.get('use_flash_attention', False)
            if cfg.get('override_token_model', None):
                t5_cfg.tokenizer.model = cfg['override_token_model']
            if cfg.get('override_tokenizer_vocab_file', None):
                t5_cfg.tokenizer.vocab_file = cfg['override_tokenizer_vocab_file']

        if cfg.get('train_from_scratch', False):
            print("Training from scratch!")
            # Defaults for 220m model
            # To override any of these, add +model.override_<key>=<value> to the config file.
            # Eg. +model.override_hidden_size=1024
            overide_keys = [
                'hidden_size',  # 768
                'num_layers',  # 12
                'num_attention_heads',  # 12
                'hidden_dropout',  # 0.1
                'attention_dropout',  # 0.1
                'kv_channels',  # 64
                'ffn_hidden_size',  # 2048
            ]
            # Defaults for 220m model
            for k in overide_keys:
                if cfg.get(f'override_{k}') is not None:
                    t5_cfg[k] = cfg.get(f'override_{k}')

            self.frozen_model = MegatronT5OverrideModel(t5_cfg, trainer=trainer)
            num_params = sum(p.numel() for p in self.frozen_model.parameters() if p.requires_grad)
            print(f"Number of parameters: {num_params}")
        else:
            print(f"Loading from pretrained checkpoint: {cfg_language_model_path}")
            if cfg_language_model_path is None:
                raise ValueError(
                    "T5-TTS SFT on pretrained model checkpoint requires `langauge_model_path` in its config."
                )

            self.frozen_model = MegatronT5OverrideModel.restore_from(
                cfg_language_model_path,
                trainer=trainer,
                override_config_path=t5_cfg,
                save_restore_connector=NLPSaveRestoreConnector(),
            )

        if not cfg.get('english_only_model', False):
            self.frozen_model.tokenizer.add_phone_tokens_to_special_tokens()

        logging.info(f"self.frozen_model {self.frozen_model}")

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        """
        Dataloader produces a global batch which is turned into a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Get seq length of batch
        batch = next(dataloader_iter)
        _, seq_length = batch[0].shape
        if batch[4].dim() > 2:
            _, _, dec_seq_length = batch[4].shape
        else:
            _, dec_seq_length = batch[4].shape
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=data_iter,
            model=[self],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
            decoder_seq_length=dec_seq_length,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def convert_tokens_to_range(self, tokens, apply_offset_correction=True, pattern=None):
        # convert tokens to range [0, 1024]
        output_tokens = tokens.clone()
        if apply_offset_correction:
            output_tokens[0] = output_tokens[0] - self.speech_offset
        output_tokens = torch.clamp(output_tokens, min=0, max=self.speech_codebook_size - 1)
        if pattern is None:
            pattern = self.cfg.get('seq_pattern', 'delay_parallel')
        if pattern == "delay_parallel":
            output_tokens_new = []
            for _c in range(output_tokens.shape[0]):
                si = _c
                ei = _c + output_tokens.shape[1] - self.num_speech_codebooks
                output_tokens_new.append(output_tokens[_c, si:ei])
            output_tokens_new = torch.stack(output_tokens_new)
            output_tokens = output_tokens_new

        return output_tokens

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            _batch = []
            for x in batch:
                if isinstance(x, torch.Tensor):
                    x = x.cuda(non_blocking=True)
                elif isinstance(x, list):
                    if isinstance(x[0], torch.Tensor):
                        x = [y.cuda(non_blocking=True) for y in x]
                _batch.append(x)
            batch = _batch
            # batch = [x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch]
            (
                virtual_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                labels,
                loss_mask,
                position_ids,
                taskname_ids,
                speech_mask,
                context_and_question_tokens_lens,
                cross_attention_prior,
                text_limits,
                _,  # TODO: text limit and lang not in tarred dataset
                _,
            ) = batch

            if self.trainer.global_step % self.train_check_interval == 0 and not validation_step and self.is_rank_zero:
                self.frozen_model.enc_dec_model.logging_step = True

            _cross_attention_prior = cross_attention_prior
            if isinstance(context_and_question_tokens, list):
                # None for context and prior for question
                _cross_attention_prior = [None, cross_attention_prior]

            output_tensor, encoder_input, out_logits = model(
                virtual_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                position_ids,
                taskname_ids,
                labels=labels,
                speech_mask=speech_mask,
                cross_attention_prior=_cross_attention_prior,
                text_limits=text_limits,
                inference=False,
            )
            output_tensor = output_tensor.contiguous()

            alignment_loss = out_logits[3]
            if alignment_loss is not None:
                self.logger.experiment.add_scalar('train_alignment_loss', alignment_loss, self.global_step)

            if self.trainer.global_step % self.train_check_interval == 0 and not validation_step and self.is_rank_zero:
                self.frozen_model.enc_dec_model.logging_step = False
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        if torch.count_nonzero(speech_mask) == 0:
                            text_labels = labels[:, 0, :]  # [B, 8, T] -> [B, T]
                            token_logits = out_logits[0] * 1  # [T, B, V]
                            if self.frozen_model.enc_dec_model.parallel_output:
                                # Gather from tensor parallel region
                                token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                            token_logits = token_logits.argmax(dim=2)  # [T, B]
                            token_logits = token_logits.t()  # [B, T]
                            score = 0
                            for i in range(text_labels.size()[0]):
                                r = text_labels[i].long()
                                nzm = r != 0
                                r = r.tolist()
                                h = token_logits[i].long() * nzm
                                h = h.tolist()
                                score += editdistance.eval(r, h)
                            score /= text_labels.size()[0]
                            logging.info(f"wer score : {score}")
                            self.logger.experiment.add_scalar('WER', score, self.global_step)
                        else:
                            audio_len = (
                                self.decoder_context_len + (labels[0][0][self.decoder_context_len :] != 0).sum().item()
                            )
                            labels_to_1024 = self.convert_tokens_to_range(labels[0, :, 0:audio_len])
                            label_wav = self.decode_wav_from_codec_model(labels_to_1024)
                            dec_input_to_1024 = self.convert_tokens_to_range(dec_input[0, :, 0:audio_len])
                            dec_input_wav = self.decode_wav_from_codec_model(dec_input_to_1024)
                            self.logger.experiment.add_audio(
                                "train_label_wav", label_wav, self.global_step, self.sample_rate
                            )
                            self.logger.experiment.add_audio(
                                "train_dec_input_wav", dec_input_wav, self.global_step, self.sample_rate
                            )
                            if isinstance(context_and_question_tokens, list):
                                context_tokens = context_and_question_tokens[0]
                                question_tokens = context_and_question_tokens[1]
                                input_token_list_all = [
                                    question_tokens[0, 0, i].item() for i in range(question_tokens.shape[2])
                                ]
                                input_token_list = [
                                    (ti, t)
                                    for ti, t in enumerate(input_token_list_all)
                                    if t != 0 and t < self.speech_offset
                                ]
                                context_end_step = context_and_question_tokens_lens[0][0].item()
                                _context_tokens = context_tokens[0, :, :context_end_step]
                            else:
                                input_token_list_all = [
                                    context_and_question_tokens[0, 0, i].item()
                                    for i in range(context_and_question_tokens.shape[2])
                                ]
                                input_token_list = [
                                    (ti, t)
                                    for ti, t in enumerate(input_token_list_all)
                                    if t != 0 and t < self.speech_offset
                                ]
                                context_end_step = input_token_list[0][0]
                                _context_tokens = context_and_question_tokens[0, :, :context_end_step]

                            if context_end_step > 1:
                                is_speech_context = _context_tokens[1, :].sum().item() > 0
                                if is_speech_context:
                                    _context_tokens = self.convert_tokens_to_range(
                                        _context_tokens, pattern=self.context_pattern
                                    )
                                    _context_wav = self.decode_wav_from_codec_model(_context_tokens)
                                    self.logger.experiment.add_audio(
                                        "train_context_wav", _context_wav, self.global_step, self.sample_rate
                                    )
                                else:
                                    _context_token_list = [v.item() for v in _context_tokens[0, :]]
                                    _context_text = self.frozen_model.tokenizer.ids_to_text(
                                        [v for v in _context_token_list if v < self.lm_vocab_size]
                                    )
                                    self.logger.experiment.add_text(
                                        "train_context_text", _context_text, self.global_step
                                    )

                            question_si = text_limits[0, 0].item() - virtual_tokens.shape[1]
                            question_ei = text_limits[0, 1].item() - virtual_tokens.shape[1]
                            text_si = text_limits[0, 0].item()
                            text_ei = text_limits[0, 1].item()
                            input_text = self.frozen_model.tokenizer.ids_to_text(
                                [v for v in input_token_list_all[question_si:question_ei] if v < self.lm_vocab_size]
                            )
                            self.logger.experiment.add_text("Train Input Text", input_text, self.global_step)

                            input_phoneme_tokens = [
                                v - self.lm_vocab_size
                                for v in input_token_list_all[question_si:question_ei]
                                if v >= self.lm_vocab_size
                            ]

                            if len(input_phoneme_tokens) > 0:
                                phoneme_text = self.phoneme_tokenizer.decode(input_phoneme_tokens)
                                self.logger.experiment.add_text(
                                    "Train Input Phoneme Text", phoneme_text, self.global_step
                                )

                            token_logits = out_logits[0]
                            speech_logits_list = out_logits[1]

                            attention_probs_list = out_logits[2]  # list of (BS, 12, out_length, in_length)
                            if attention_probs_list is not None:
                                attention_sliced_list = []
                                for lidx in range(len(attention_probs_list)):
                                    attention_probs = attention_probs_list[lidx]
                                    for _i in range(attention_probs.shape[1]):
                                        name = f"Attention Probs Layer {lidx} Head {_i}"
                                        attention_to_plot = attention_probs[0, _i, :audio_len, :text_ei]
                                        if self.plot_alignments_sliced:
                                            attention_to_plot = attention_probs[0, _i, 0:audio_len, text_si:text_ei]
                                            # 4 to offset "Text to Speech this"
                                            name += " Sliced"
                                        alignment_image = plot_alignment_to_numpy_for_speechllm(
                                            attention_to_plot.cpu().float().numpy().T,
                                            phoneme_ver=0 if self.plot_alignments_sliced else 1,
                                            phoneme_seq=None if self.plot_alignments_sliced else [text_si],
                                        )
                                        self.logger.experiment.add_image(
                                            name,
                                            alignment_image,
                                            self.global_step,
                                            dataformats="HWC",
                                        )
                                        attention_sliced_list.append(
                                            attention_probs[
                                                0, _i, self.decoder_context_len : audio_len, text_si:text_ei
                                            ]
                                        )
                                attention_sliced = torch.stack(attention_sliced_list)
                                attention_sliced = torch.mean(attention_sliced, 0)
                                text = None
                                if len(input_text) > 0:
                                    text = self.frozen_model.tokenizer.ids_to_tokens(
                                        [
                                            v
                                            for v in input_token_list_all[question_si:question_ei]
                                            if v < self.lm_vocab_size
                                        ]
                                    )
                                if len(input_phoneme_tokens) > 0:
                                    text = phoneme_text.split("|")
                                alignment_image_sliced = plot_alignment_to_numpy_for_speechllm(
                                    attention_sliced.cpu().float().numpy().T,
                                    phoneme_seq=text,
                                    phoneme_ver=2,
                                    vmin=0.0,
                                    phone_offset=0,
                                    h_offset=False,
                                )
                                self.logger.experiment.add_image(
                                    f"Attention Probs Average Sliced",
                                    alignment_image_sliced,
                                    self.global_step,
                                    dataformats="HWC",
                                )
                            if self.frozen_model.enc_dec_model.parallel_output:
                                # Gather from tensor parallel region
                                token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                                for _i in range(len(speech_logits_list)):
                                    speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                                        speech_logits_list[_i]
                                    )
                            speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t, b, 1024, 7)
                            token_logits_example = token_logits[:, 0, :] * 1
                            speech_logits_example = speech_logits[:, 0, :, :] * 1
                            first_layer_tokens = token_logits_example.argmax(dim=1) - self.speech_offset
                            other_layer_tokens = []
                            for _i in range(speech_logits_example.shape[2]):
                                other_layer_tokens.append(speech_logits_example[:, :, _i].argmax(dim=1))

                            all_layer_tokens = torch.stack([first_layer_tokens] + other_layer_tokens)  # (8, t)
                            all_layer_tokens = self.convert_tokens_to_range(
                                all_layer_tokens, apply_offset_correction=False
                            )
                            # all_layer_tokens = torch.clip(all_layer_tokens, 0, 1023)
                            predicted_wav = self.decode_wav_from_codec_model(all_layer_tokens)
                            self.logger.experiment.add_audio(
                                "train_tf_pred_wav", predicted_wav, self.global_step, self.sample_rate
                            )

            def loss_func(loss_args):
                output_tensor, out_logits, curr_step = loss_args
                alignment_loss = out_logits[3]
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                if (
                    (alignment_loss is not None)
                    and (curr_step > self.alignment_loss_start_step)
                    and (curr_step < self.alignment_loss_end_step)
                ):
                    logging.debug(f"Adding alignment loss. cur:{curr_step} start:{self.alignment_loss_start_step}")
                    loss = loss + alignment_loss
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return [output_tensor, out_logits, self.global_step], loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        """Used in inference / predict"""

        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            _batch = []
            for x in batch:
                if isinstance(x, torch.Tensor):
                    x = x.cuda(non_blocking=True)
                elif isinstance(x, list):
                    if isinstance(x[0], torch.Tensor):
                        x = [y.cuda(non_blocking=True) for y in x]
                _batch.append(x)
            batch = _batch
            # batch = [x.cuda(non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch]
            (
                decoder_max_sequence_len,
                encoder_max_sequence_len,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                position_ids,
                taskname_ids,
                speech_mask,
            ) = batch

            output_logits, _, token_and_speech_logits = model(
                context_and_question_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input,
                dec_input_mask,
                position_ids,
                taskname_ids,
                labels=None,
                speech_mask=speech_mask,
                inference=True,
                inference_step=1,
                decoder_max_sequence_len=decoder_max_sequence_len,
                encoder_max_sequence_len=encoder_max_sequence_len,
            )
            output_tensor = [output_logits, token_and_speech_logits]

            def id_func(output_tensor):
                return 0, {'output_logits': output_tensor[0], 'token_and_speech_logits': output_tensor[1]}

            return output_tensor, id_func

        return fwd_output_only_func

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
        No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When using pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.frozen_model.enc_dec_model.set_input_tensor(input_tensor)

    def on_train_epoch_start(self) -> None:
        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        mbs = self.cfg.get('validation_micro_batch_size', self.cfg.micro_batch_size)
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_validation_epoch_start()

    def training_step(self, dataloader_iter, batch_idx):
        self._optimizer.zero_grad()
        batch = next(dataloader_iter)

        # apply text classifier-free guidance by replacing input question tokens with [UNK].
        if self.train_text_cfg_prob > 0.0:
            if self._rng.random() < self.train_text_cfg_prob:
                logging.info(f"Text Classifier-Free Guidance is triggered for the {batch_idx}-th batch.")

                # temporally disable computing CTC alignment loss.
                if self.use_alignment_loss:
                    self.frozen_model.enc_dec_model.use_alignment_loss = False

                # make cross-attention prior to None to remove the prior.
                batch[11] = None

                # replace question token IDs with [UNK]'s id. No speech offset for Phoneme's [UNK]. Same op as train.
                # instruction token IDs are bpe token IDs directly obtained from self.tokenizer without any offset.
                # question token IDs are phoneme and grapheme token IDs and are offset by self.lm_vocab_size
                #   if under "Phoneme TTS" instruction, so existing no overlaps between instruction and question token IDs.
                # question token IDs are bpe token IDs without any offset
                #   if under "Text to speech this" instruction, so existing overlaps between instruction and question token IDs.
                context_and_question_tokens = batch[
                    1
                ]  # (batch_size, self.num_speech_codebooks, max_context_question_tokens_len)
                text_limits = batch[12]
                virtual_tokens = batch[0]
                question_limits = text_limits - virtual_tokens.size(
                    1
                )  # (b, 2), reset question range to start from [pad] context, same start position as context_and_question_tokens.
                question_start = question_limits[:, 0].unsqueeze(1)  # (b, 1)
                question_end = question_limits[:, 1].unsqueeze(1)  # (b, 1)

                if isinstance(context_and_question_tokens, list):  # indicate self.encoder_type=multi_transformers.
                    context_tokens, question_tokens = context_and_question_tokens
                    question_tokens_unconditioned = question_tokens.clone()
                    time_range = torch.arange(
                        question_tokens_unconditioned.size(2), device=question_tokens_unconditioned.device
                    ).unsqueeze(0)
                    question_mask = (time_range >= question_start) & (
                        time_range < question_end
                    )  # create a mask for question only tokens.
                    question_tokens_unconditioned[:, 0][
                        question_mask
                    ] = self.tokenizer.unk_id  # only the first layer has non-zero IDs.
                    batch[1] = [context_tokens, question_tokens_unconditioned]
                else:
                    context_and_question_tokens_unconditioned = (
                        context_and_question_tokens.clone()
                    )  # (batch_size, self.num_speech_codebooks, max_context_question_tokens_len)
                    time_range = torch.arange(
                        context_and_question_tokens_unconditioned.size(2),
                        device=context_and_question_tokens_unconditioned.device,
                    ).unsqueeze(
                        0
                    )  # (1, max_context_question_tokens_len)
                    question_mask = (time_range >= question_start) & (
                        time_range < question_end
                    )  # create a mask for question only tokens.
                    context_and_question_tokens_unconditioned[:, 0][
                        question_mask
                    ] = self.tokenizer.unk_id  # only the first layer has non-zero IDs.
                    batch[1] = context_and_question_tokens_unconditioned

                del question_limits, question_start, question_end, time_range, question_mask
            else:
                # recover to original alignment loss config.
                self.frozen_model.enc_dec_model.use_alignment_loss = self.use_alignment_loss

        # apply audio context classifier-free guidance by replacing audio codec with [UNK]
        if self.train_audio_cfg_prob > 0.0:
            if self._rng.random() < self.train_audio_cfg_prob:
                logging.info(f"Audio Classifier-Free Guidance is triggered for the {batch_idx}-th batch.")

                context_and_question_tokens = batch[
                    1
                ]  # (batch_size, self.num_speech_codebooks, max_context_question_tokens_len)

                if isinstance(context_and_question_tokens, list):  # indicate self.encoder_type=multi_transformers.
                    context_tokens, question_tokens = context_and_question_tokens
                    context_tokens_unconditioned = context_tokens.clone()
                    context_tokens_unconditioned[:, :, :] = (
                        self.tokenizer.unk_id
                    )  # TODO @xueyang: verify if extra tokens other than audio codec tokens are appended.
                    batch[1] = [context_tokens_unconditioned, question_tokens]
                else:
                    # dec_input
                    dec_input = batch[3]
                    dec_input_unconditioned = dec_input.clone()
                    dec_input_unconditioned[:, :, 1 : self.decoder_context_len + 1] = (
                        self.tokenizer.unk_id
                    )  # TODO @xueyang: switch to other token id if this one is conflict with text unk.
                    batch[3] = dec_input_unconditioned

        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        return loss_mean

    def get_predictions(self, input_ids, enc_mask, encoder_input, labels):
        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=input_ids,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
            bos_id=(
                self.tokenizer.pad_id if self.cfg.data.get('decoder_starts_with_pad', False) else self.tokenizer.bos_id
            ),
        )
        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(input_ids, self.tokenizer)
        return {
            'predicted_token_ids': preds_text,
            'labels': labels_text,
            'enc_inputs': input_text,
        }

    def get_embeddings(self, tokens, taskname_ids, inference=False):
        out = None
        if tokens.dim() > 2:
            for i in range(tokens.size()[1]):  # for 8 channels
                if i == 0:
                    # Embed first layer using word embeddings
                    out = self.embed_input(tokens[:, i, :], taskname_ids, inference)  # (B, T, D)
                else:
                    # Embed other layers using speech embeddings
                    cur = self.frozen_model.enc_dec_model.speech_tokens_embeddings[i - 1](tokens[:, i, :])
                    # do not add embeddings of zero tokens of other channels (except the first channel)
                    non_zero_flag = tokens[:, i, :] != 0  # (B, T)
                    cur = cur * non_zero_flag.unsqueeze(2)
                    out = out + cur
        else:
            out = self.embed_input(tokens, taskname_ids, inference)
        return out

    def get_embeddings_and_combine(self, token_list, taskname_ids, inference):
        embedding_list = []
        for tokens in token_list:
            embedding_list.append(self.get_embeddings(tokens, taskname_ids, inference))
        return torch.cat(embedding_list, dim=1)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (
            virtual_tokens,
            context_and_question_tokens,
            enc_mask,
            dec_input,
            dec_input_mask,
            labels,
            loss_mask,
            position_ids,
            taskname_ids,
            speech_mask,
            context_and_question_tokens_lens,
            cross_attention_prior,
            text_limits,
            _,
            _,
        ) = batch
        # loss_mask (b, t)
        # does not use dataloader_iter due to device placement issues arising from PTL

        mode = self.training
        self.eval()
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(virtual_tokens.size(0), gbs)

        loss_mean = self.fwd_bwd_step(
            itertools.chain([batch]), batch_idx, forward_only=True
        )  # comment this out and add custom forward function to calculate WER
        # # logging.info (f'loss_mean {loss_mean}')

        if batch_idx == 0 and self.is_rank_zero:
            self.frozen_model.enc_dec_model.logging_step = True
            self.predict_step_outputs = []
            # log_scalars=False avoids logging scalar TTS metrics in the predict_step
            # Images, audio and texts will still be logged
            self.predict_step(batch=batch, batch_idx=batch_idx, log_scalars=False, global_step=self.global_step)
            for inf_key in self.predict_step_outputs[0]:
                if self.predict_step_outputs[0][inf_key] is not None:
                    self.logger.experiment.add_scalar(
                        f'Val_{inf_key}', self.predict_step_outputs[0][inf_key], self.global_step
                    )

        labels_original = labels.clone()  # (b, 8, t)

        _cross_attention_prior = cross_attention_prior
        if isinstance(context_and_question_tokens, list):
            _cross_attention_prior = [None, cross_attention_prior]

        output_loss, _, output_logits = self.forward(
            virtual_tokens,
            context_and_question_tokens,
            enc_mask,
            dec_input,
            dec_input_mask,
            position_ids,
            taskname_ids,
            labels=labels,
            speech_mask=speech_mask,
            cross_attention_prior=_cross_attention_prior,
            text_limits=text_limits,
            inference=False,
        )

        if batch_idx == 0 and self.is_rank_zero:
            self.frozen_model.enc_dec_model.logging_step = False
            with torch.cuda.amp.autocast(enabled=False):
                if torch.count_nonzero(speech_mask) == 0:
                    text_labels = labels[:, 0, :]  # [B, 8, T] -> [B, T]
                    token_logits = output_logits[0] * 1  # [T, B, V]
                    if self.frozen_model.enc_dec_model.parallel_output:
                        # Gather from tensor parallel region
                        token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                    token_logits = token_logits.argmax(dim=2)  # [T, B]
                    token_logits = token_logits.t()  # [B, T]
                    score = 0
                    for i in range(text_labels.size()[0]):
                        r = text_labels[i].long()
                        nzm = r != 0
                        r = r.tolist()
                        h = token_logits[i].long() * nzm
                        h = h.tolist()
                        score += editdistance.eval(r, h)
                    score /= text_labels.size()[0]
                    logging.info(f"wer score : {score}")
                    self.logger.experiment.add_scalar('WER', score, self.global_step)
                else:
                    audio_len = self.decoder_context_len + (labels[0][0][self.decoder_context_len :] != 0).sum().item()
                    labels_to_1024 = self.convert_tokens_to_range(labels[0, :, 0:audio_len])
                    label_wav = self.decode_wav_from_codec_model(labels_to_1024)
                    dec_input_to_1024 = self.convert_tokens_to_range(dec_input[0, :, 0:audio_len])
                    dec_input_wav = self.decode_wav_from_codec_model(dec_input_to_1024)
                    self.logger.experiment.add_audio("val_label_wav", label_wav, self.global_step, self.sample_rate)
                    self.logger.experiment.add_audio(
                        "val_dec_input_wav", dec_input_wav, self.global_step, self.sample_rate
                    )

                    if isinstance(context_and_question_tokens, list):
                        context_tokens = context_and_question_tokens[0]
                        question_tokens = context_and_question_tokens[1]
                        input_token_list_all = [
                            question_tokens[0, 0, i].item() for i in range(question_tokens.shape[2])
                        ]
                        input_token_list = [
                            (ti, t) for ti, t in enumerate(input_token_list_all) if t != 0 and t < self.speech_offset
                        ]
                        context_end_step = context_and_question_tokens_lens[0][0].item()
                        _context_tokens = context_tokens[0, :, :context_end_step]

                    else:
                        input_token_list_all = [
                            context_and_question_tokens[0, 0, i].item()
                            for i in range(context_and_question_tokens.shape[2])
                        ]
                        input_token_list = [
                            (ti, t) for ti, t in enumerate(input_token_list_all) if t != 0 and t < self.speech_offset
                        ]
                        context_end_step = input_token_list[0][0]
                        _context_tokens = context_and_question_tokens[0, :, :context_end_step]
                    if context_end_step > 1:
                        is_speech_context = _context_tokens[1, :].sum().item() > 0
                        if is_speech_context:
                            _context_tokens = self.convert_tokens_to_range(
                                _context_tokens, pattern=self.context_pattern
                            )
                            _context_wav = self.decode_wav_from_codec_model(_context_tokens)
                            self.logger.experiment.add_audio(
                                "val_context_wav", _context_wav, self.global_step, self.sample_rate
                            )
                        else:
                            _context_token_list = [v.item() for v in _context_tokens[0, :]]
                            _context_text = self.frozen_model.tokenizer.ids_to_text(
                                [v for v in _context_token_list if v < self.lm_vocab_size]
                            )
                            self.logger.experiment.add_text("val_context_text", _context_text, self.global_step)

                    question_si = text_limits[0, 0].item() - virtual_tokens.shape[1]
                    question_ei = text_limits[0, 1].item() - virtual_tokens.shape[1]

                    text_si = text_limits[0, 0].item()
                    text_ei = text_limits[0, 1].item()

                    input_text = self.frozen_model.tokenizer.ids_to_text(
                        [v for v in input_token_list_all[question_si:question_ei] if v < self.lm_vocab_size]
                    )
                    self.logger.experiment.add_text("Val Input Text", input_text, self.global_step)

                    input_phoneme_tokens = [
                        v - self.lm_vocab_size
                        for v in input_token_list_all[question_si:question_ei]
                        if v >= self.lm_vocab_size
                    ]
                    if len(input_phoneme_tokens) > 0:
                        phoneme_text = self.phoneme_tokenizer.decode(input_phoneme_tokens)
                        self.logger.experiment.add_text("Val Input Phoneme Text", phoneme_text, self.global_step)

                    token_logits = output_logits[0]
                    speech_logits_list = output_logits[1]

                    # if self.trainer.global_step % 500 == 0:
                    attention_probs_list = output_logits[2]  # list of (BS, 12, out_length, in_length)
                    if attention_probs_list is not None:
                        attention_sliced_list = []
                        for lidx in range(len(attention_probs_list)):
                            attention_probs = attention_probs_list[lidx]
                            for _i in range(attention_probs.shape[1]):
                                attention_sliced_list.append(
                                    attention_probs[0, _i, self.decoder_context_len : audio_len, text_si:text_ei]
                                )
                        attention_sliced = torch.stack(attention_sliced_list)
                        attention_sliced = torch.mean(attention_sliced, 0)
                        text = None
                        if len(input_text) > 0:
                            text = self.frozen_model.tokenizer.ids_to_tokens(
                                [v for v in input_token_list_all[question_si:question_ei] if v < self.lm_vocab_size]
                            )
                        if len(input_phoneme_tokens) > 0:
                            text = phoneme_text.split("|")
                        alignment_image_sliced = plot_alignment_to_numpy_for_speechllm(
                            attention_sliced.cpu().float().numpy().T,
                            phoneme_seq=text,
                            phoneme_ver=2,
                            vmin=0.0,
                            phone_offset=0,
                            h_offset=False,
                        )
                        self.logger.experiment.add_image(
                            f"Val Attention Probs Average Sliced",
                            alignment_image_sliced,
                            self.global_step,
                            dataformats="HWC",
                        )
                    if self.frozen_model.enc_dec_model.parallel_output:
                        # Gather from tensor parallel region
                        token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
                        for _i in range(len(speech_logits_list)):
                            speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                                speech_logits_list[_i]
                            )
                    speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t, b, 1024, 7)
                    token_logits_example = token_logits[:, 0, :] * 1
                    speech_logits_example = speech_logits[:, 0, :, :] * 1
                    first_layer_tokens = token_logits_example.argmax(dim=1) - self.speech_offset
                    other_layer_tokens = []
                    for _i in range(speech_logits_example.shape[2]):
                        other_layer_tokens.append(speech_logits_example[:, :, _i].argmax(dim=1))

                    all_layer_tokens = torch.stack([first_layer_tokens] + other_layer_tokens)  # (8, t)
                    all_layer_tokens = self.convert_tokens_to_range(all_layer_tokens, apply_offset_correction=False)
                    all_layer_tokens = torch.clip(all_layer_tokens, 0, self.speech_codebook_size - 1)
                    predicted_wav = self.decode_wav_from_codec_model(all_layer_tokens)
                    self.logger.experiment.add_audio(
                        "val_tf_pred_wav", predicted_wav, self.global_step, self.sample_rate
                    )

        first_layer_logits = output_logits[0]
        speech_logits_list = output_logits[1]

        if self.frozen_model.enc_dec_model.parallel_output:
            # Gather from tensor parallel region
            first_layer_logits = tensor_parallel.gather_from_tensor_model_parallel_region(first_layer_logits)
            if torch.count_nonzero(speech_mask) > 0:
                for _i in range(len(speech_logits_list)):
                    speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                        speech_logits_list[_i]
                    )
                speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t, b, 1024, 7)
        first_layer_preds = first_layer_logits.argmax(dim=2)  # (t,bs)
        first_layer_preds = first_layer_preds.transpose(0, 1)  # (bs,t)
        labels_first_layer = labels_original[:, 0, :]  # (bs,t)
        correct_predictions = first_layer_preds == labels_first_layer  # (bs,t)
        correct_predictions = correct_predictions * loss_mask  # (bs,t)
        total_correct_predictions = torch.sum(correct_predictions)
        total_predictions = torch.sum(loss_mask)
        first_layer_accuracy = total_correct_predictions / total_predictions
        first_layer_loss = torch.nn.functional.cross_entropy(
            first_layer_logits.permute(1, 2, 0), labels_first_layer, reduction='none'
        )  # (bs,t)
        first_layer_loss = torch.sum(first_layer_loss * loss_mask) / total_predictions

        metrics = {
            'loss': loss_mean,
            'first_layer_accuracy': first_layer_accuracy,
            'first_layer_loss': first_layer_loss,
        }
        loss_total = first_layer_loss
        for i in range(self.num_speech_codebooks - 1):
            if torch.count_nonzero(speech_mask) > 0:
                speech_logits_i = speech_logits[:, :, :, i]
                speech_preds_i = speech_logits_i.argmax(dim=2)  # (t,bs)
                speech_preds_i = speech_preds_i.transpose(0, 1)  # (bs,t)
                labels_i = labels_original[:, i + 1, :]  # (bs,t)
                correct_predictions_i = speech_preds_i == labels_i  # (bs,t)
                correct_predictions_i = correct_predictions_i * loss_mask * speech_mask  # (bs,t)
                total_correct_predictions_i = torch.sum(correct_predictions_i)
                total_predictions_i = torch.sum(loss_mask * speech_mask)
                speech_accuracy_i = total_correct_predictions_i / total_predictions_i
                loss_i = torch.nn.functional.cross_entropy(
                    speech_logits_i.permute(1, 2, 0), labels_i, reduction='none'
                )  # (bs,t)
                loss_i = torch.sum(loss_i * loss_mask * speech_mask) / total_predictions_i
            else:
                speech_accuracy_i = torch.tensor(0.0)
                loss_i = torch.tensor(0.0)
            metrics[f'speech_accuracy_{i+1}'] = speech_accuracy_i
            metrics[f'speech_loss_{i+1}'] = loss_i
            loss_total += loss_i

        metrics['loss_total_check'] = loss_total
        self.validation_step_outputs.append(metrics)
        self.train(mode=mode)
        self.frozen_model.train()
        return metrics['loss']

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            assert (
                self.cfg.get("virtual_pipeline_model_parallel_size", None) is None
            ), "Virtual pipeline model parallel size is no longer supported for nemo 1.0"
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
                averaged_loss_total_check = torch.stack([item['loss_total_check'] for item in outputs]).mean()
                averaged_first_layer_accuracy = torch.stack([item['first_layer_accuracy'] for item in outputs]).mean()

                self.log(
                    'val_loss_total_check',
                    averaged_loss_total_check,
                    prog_bar=False,
                    rank_zero_only=True,
                    batch_size=1,
                )
                self.log(
                    'val_first_layer_accuracy',
                    averaged_first_layer_accuracy,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                )
                logging.info(f'Validation first_layer_accuracy: {averaged_first_layer_accuracy}')
                logging.info(f'Validation loss_total_check: {averaged_loss_total_check}')

                for i in range(1, self.num_speech_codebooks):
                    averaged_speech_accuracy = torch.stack([item[f'speech_accuracy_{i}'] for item in outputs]).mean()
                    averaged_speech_loss = torch.stack([item[f'speech_loss_{i}'] for item in outputs]).mean()
                    self.log(
                        f'val_speech_accuracy_{i}',
                        averaged_speech_accuracy,
                        prog_bar=True,
                        rank_zero_only=True,
                        batch_size=1,
                    )
                    self.log(
                        f'val_speech_loss_{i}', averaged_speech_loss, prog_bar=True, rank_zero_only=True, batch_size=1
                    )
                    logging.info(f'Validation speech_accuracy_{i}: {averaged_speech_accuracy}')
                    logging.info(f'Validation speech_loss_{i}: {averaged_speech_loss}')
            else:
                averaged_loss = torch.tensor(0.0).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(averaged_loss, get_last_rank())

            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
            logging.info(f'Validation loss: {averaged_loss}')

        else:
            if len(outputs) > 0:
                averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
                averaged_loss_total_check = torch.stack([item['loss_total_check'] for item in outputs]).mean()
                logging.info(f'Validation loss: {averaged_loss}')
                self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
                self.log(
                    'val_loss_total_check',
                    averaged_loss_total_check,
                    prog_bar=False,
                    rank_zero_only=True,
                    batch_size=1,
                )

                averaged_first_layer_accuracy = torch.stack([item['first_layer_accuracy'] for item in outputs]).mean()
                logging.info(f'Validation first_layer_accuracy: {averaged_first_layer_accuracy}')
                self.log(
                    'val_first_layer_accuracy',
                    averaged_first_layer_accuracy,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                )

                for i in range(1, self.num_speech_codebooks):
                    averaged_speech_accuracy = torch.stack([item[f'speech_accuracy_{i}'] for item in outputs]).mean()
                    averaged_speech_loss = torch.stack([item[f'speech_loss_{i}'] for item in outputs]).mean()
                    logging.info(f'Validation speech_accuracy_{i}: {averaged_speech_accuracy}')
                    logging.info(f'Validation speech_loss_{i}: {averaged_speech_loss}')
                    self.log(
                        f'val_speech_accuracy_{i}',
                        averaged_speech_accuracy,
                        prog_bar=True,
                        rank_zero_only=True,
                        batch_size=1,
                    )
                    self.log(
                        f'val_speech_loss_{i}', averaged_speech_loss, prog_bar=True, rank_zero_only=True, batch_size=1
                    )

        if self.cfg.get("report_validation_metric", False):
            gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]

            all_preds = list(itertools.chain(*[item['predicted_token_ids'] for item in outputs]))
            all_labels = list(itertools.chain(*[item['labels'] for item in outputs]))
            all_inputs = list(itertools.chain(*[item['enc_inputs'] for item in outputs]))

            assert len(all_preds) == len(all_labels)
            assert len(all_preds) == len(all_inputs)

            # Gather inputs, preds, labels from all workers
            torch.distributed.all_gather_object(
                gather_results,
                [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
                group=parallel_state.get_data_parallel_group(),
            )

            # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
            if parallel_state.get_data_parallel_rank() == 0:

                gather_results_dedup = list(set(itertools.chain(*gather_results)))

                val_metric_dict = self.validation_metric.get_score(
                    [i[2] for i in gather_results_dedup],
                    [i[1] for i in gather_results_dedup],
                )

                for metric, val in val_metric_dict.items():
                    logging.info(f'Validation {metric}: {val}')
                val_metric = list(val_metric_dict.items())[0][1]
                metric_name = list(val_metric_dict.items())[0][0]
            else:
                val_metric = torch.tensor(0.0).cuda()
                metric_name = ''

            self.log(f'val_{metric_name}', val_metric, prog_bar=True, rank_zero_only=True, batch_size=1)

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        result = self.predict_step(batch, batch_idx)
        return result

    def on_test_epoch_end(self):
        """
        This might still be broken for lightning 2.0. to fix: see
        https://github.com/NVIDIA/NeMo/blob/9bdf4d12276ee8f95a340cf2f7f340e9b5b74a7e/docs/source/starthere/migration-guide.rst
        """
        outputs = self.predict_step_outputs
        average_metrics = {}
        for output in outputs:
            for key in output:
                if key not in average_metrics:
                    average_metrics[key] = []
                if isinstance(output[key], torch.Tensor):
                    average_metrics[key].append(output[key].item())
                elif output[key] is None:
                    continue
                else:
                    average_metrics[key].append(output[key])

        for key in average_metrics:
            average_metrics[key] = np.mean(average_metrics[key]).item()
            logging.info(f'Test {key}: {average_metrics[key]}')
            self.log(f'test_{key}', average_metrics[key], prog_bar=True, rank_zero_only=True, batch_size=1)
            self.logger.experiment.add_scalar(f'Inf Cumulative {key}', average_metrics[key], 0)

        # save average metrics into json file
        with open(os.path.join(self.logger.log_dir, 'output_metrics.json'), 'w') as f:
            json.dump(average_metrics, f)

    def build_virtual_prompt_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = T5SpeechLMDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            sample_rate=self.cfg.data.get('sample_rate', 24000),
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
            add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
            add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
            ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
            for_train=for_train,
            segment_max_duration=self.cfg.data.get('segment_max_duration', None),
            trim=self.cfg.data.get('trim', None),
            trim_ref=self.cfg.data.get('trim_ref', None),
            trim_top_db=self.cfg.data.get('trim_top_db', None),
            trim_frame_length=self.cfg.data.get('trim_frame_length', None),
            trim_hop_length=self.cfg.data.get('trim_hop_length', None),
            pad_multiple=self.cfg.data.get('pad_multiple', 1),
            pitch_augment=self.cfg.data.get('pitch_augment', None),
            sup_data_path=self.cfg.data.get('sup_data_path', None),
            codec_folder=self.cfg.data.get('codec_folder', None),
            speech_offset=self.cfg.data.get('speech_offset', None),
            train_task=self.cfg.data.get('train_task', "tts"),
            seq_pattern=self.cfg.get('seq_pattern', 'delay_parallel'),
            use_attention_prior=self.cfg.data.get('use_attention_prior', False),
            attention_prior_scaling_factor=self.cfg.data.get('attention_prior_scaling_factor', 1.0),
            cross_attention_epsilon=self.cfg.data.get('cross_attention_epsilon', 0.0),
            lm_vocab_size=self.lm_vocab_size,
            num_speech_codebooks=self.num_speech_codebooks,
            codebook_fps=self.cfg.data.get('codebook_fps', 86),
            add_special_tokens_to_only_first_codebook=self.cfg.data.get(
                'add_special_tokens_to_only_first_codebook', False
            ),
            context_pattern=self.cfg.data.get('context_pattern', 'parallel'),
            context_duration_min=self.cfg.data.get('context_duration_min', 3.0),
            context_duration_max=self.cfg.data.get('context_duration_max', 5.0),
            g2p=self.cfg.data.get('g2p', None),
            skip_datasets=self.cfg.data.get('skip_datasets', []),
            english_only_model=self.cfg.get('english_only_model', False),
            use_ipa=self.cfg.data.get('use_ipa', False),
            context_conditioning=self.cfg.get('context_conditioning', "decoder"),
            use_beta_binomial_interpolator=self.cfg.get('use_beta_binomial_interpolator', False),
            context_slice_method=self.cfg.data.get('context_slice_method', 'random'),
            phoneme_probability=self.cfg.data.get('phoneme_probability', 0.5),
            encoder_type=self.cfg.data.get('encoder_type', 'single_transformer'),
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                True if num_workers > 0 else False
            ),  # (@adithyare and @eharper) We need to set this to True to get around issues with spawn=True
        )
        logging.info(f'build success: {len(dataloader)} {dataset_paths}')
        if self.phoneme_tokenizer is None:
            self.phoneme_tokenizer = dataset.phoneme_tokenizer
        return dataset, dataloader

    def build_virtual_prompt_tarred_dataset(
        self, dataset_paths, audio_path, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = T5SpeechLMTarredDataset(
            audio_tar_filepaths=audio_path,
            manifest_filepath=dataset_paths,
            tokenizer=self.tokenizer,
            sample_rate=self.cfg.data.get('sample_rate', 24000),
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            shuffle_n=shuffle,
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
            add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
            add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
            ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
            for_train=for_train,
            segment_max_duration=self.cfg.data.get('segment_max_duration', None),
            trim=self.cfg.data.get('trim', None),
            trim_ref=self.cfg.data.get('trim_ref', None),
            trim_top_db=self.cfg.data.get('trim_top_db', None),
            trim_frame_length=self.cfg.data.get('trim_frame_length', None),
            trim_hop_length=self.cfg.data.get('trim_hop_length', None),
            pad_multiple=self.cfg.data.get('pad_multiple', 1),
            pitch_augment=self.cfg.data.get('pitch_augment', None),
            speech_offset=self.cfg.data.get('speech_offset', None),
            train_task=self.cfg.data.get('train_task', "tts"),
            seq_pattern=self.cfg.get('seq_pattern', 'delay_parallel'),
            use_attention_prior=self.cfg.data.get('use_attention_prior', False),
            attention_prior_scaling_factor=self.cfg.data.get('attention_prior_scaling_factor', 1.0),
            cross_attention_epsilon=self.cfg.data.get('cross_attention_epsilon', 0.0),
            lm_vocab_size=self.lm_vocab_size,
            num_speech_codebooks=self.num_speech_codebooks,
        )
        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                True if num_workers > 0 else False
            ),  # (@adithyare and @eharper) We need to set this to True to get around issues with spawn=True
        )
        logging.info(f'build success: {len(dataloader)} {dataset_paths}')

        return dataset, dataloader

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

        # Replace double spaces with single space
        single_space_text = " ".join(no_dash_text.split())

        single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))

        return single_space_text

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_scalars=True, global_step=None
    ) -> Any:

        with torch.no_grad():
            (
                virtual_tokens,
                context_and_question_tokens,
                enc_mask,
                dec_input_raw,
                dec_input_mask_raw,
                labels,
                loss_mask,
                position_ids,
                taskname_ids,
                speech_mask,
                context_and_question_tokens_lens,
                cross_attention_prior,
                text_limits,  # [start of question token, question token len) in [0, enc_mask.size(1))
                lang,
                question_texts,
            ) = batch

            batch_size = virtual_tokens.size(0)
            dec_input = (
                dec_input_raw * 1
            )  # (B, 8, T)  # TODO @xueyang: apply clone() method bypasses this unnecessary computation.
            dec_input_mask = dec_input_mask_raw * 1  # (B, T)
            dec_input_mask[:, :] = 1  # Does not really matter
            output_token_list = []

            end_indices = {}
            # pad dec_input (B, 8, T) to 1000 timesteps
            max_inference_timesteps = self.cfg.get('max_inference_timesteps', 2000)
            # TODO @xueyang: potential bug when max_inference_timesteps < dec_input.shape[2], then dec_input is clipped.
            dec_input = torch.nn.functional.pad(dec_input, (0, max_inference_timesteps - dec_input.shape[2]), value=0)
            dec_input[:, :, self.decoder_context_len + 1 :].zero_()
            # TODO @xueyang: why not just declare torch.ones(dec_input_raw.size(0), max_inference_timesteps)?
            dec_input_mask = torch.nn.functional.pad(
                dec_input_mask, (0, max_inference_timesteps - dec_input_mask.shape[1]), value=1
            )

            if self.inference_apply_text_cfg and self.inference_apply_audio_cfg:
                question_limits = text_limits - virtual_tokens.size(
                    1
                )  # (b, 2), reset question range to start from [pad] context, same start position as context_and_question_tokens.
                question_start = question_limits[:, 0].unsqueeze(1)  # (b, 1)
                question_end = question_limits[:, 1].unsqueeze(1)  # (b, 1)

                # duplicate and glue two batches into a single one.
                virtual_tokens = torch.cat((virtual_tokens, virtual_tokens), dim=0)
                taskname_ids = torch.cat((taskname_ids, taskname_ids), dim=0)
                speech_mask = torch.cat((speech_mask, speech_mask), dim=0)
                dec_input_mask = torch.cat((dec_input_mask, dec_input_mask), dim=0)

                if isinstance(context_and_question_tokens, list):  # indicate self.encoder_type = "multi_transformers".
                    context_tokens, question_tokens = context_and_question_tokens

                    # text
                    question_tokens_unconditioned = question_tokens.clone()
                    time_range = torch.arange(
                        question_tokens_unconditioned.size(2), device=question_tokens_unconditioned.device
                    ).unsqueeze(0)
                    question_mask = (time_range >= question_start) & (
                        time_range < question_end
                    )  # create a mask for question only tokens.
                    question_tokens_unconditioned[:, 0][
                        question_mask
                    ] = self.tokenizer.unk_id  # only the first layer has non-zero IDs.

                    # audio
                    context_tokens_unconditioned = context_tokens.clone()
                    context_tokens_unconditioned[:, :, :] = self.tokenizer.unk_id

                    # concatenate both conditioned and unconditioned batches as a single one.
                    context_and_question_tokens = [
                        torch.cat((context_tokens, context_tokens_unconditioned), dim=0),
                        torch.cat((question_tokens, question_tokens_unconditioned), dim=0),
                    ]
                    enc_mask = [torch.cat((mask, mask), dim=0) for mask in enc_mask]
                    dec_input = torch.cat((dec_input, dec_input), dim=0)
                    position_ids = [torch.cat((pos_ids, pos_ids), dim=0) for pos_ids in position_ids]
                else:
                    assert (
                        self.context_conditioning == "decoder"
                    ), f"The encoder_type is single_transformer. We expect context_condition is decoder: context_condition={self.context_conditioning}"

                    # text
                    context_and_question_tokens_unconditioned = context_and_question_tokens.clone()
                    time_range = torch.arange(
                        context_and_question_tokens_unconditioned.size(2),
                        device=context_and_question_tokens_unconditioned.device,
                    ).unsqueeze(
                        0
                    )  # (1, max_context_question_tokens_len)
                    question_mask = (time_range >= question_start) & (
                        time_range < question_end
                    )  # create a mask for question only tokens.
                    context_and_question_tokens_unconditioned[:, 0][
                        question_mask
                    ] = self.tokenizer.unk_id  # only the first layer has non-zero IDs.

                    # audio
                    dec_input_unconditioned = dec_input.clone()
                    dec_input_unconditioned[:, :, 1 : self.decoder_context_len + 1] = (
                        self.tokenizer.unk_id
                    )  # TODO @xueyang: switch to other token id if this one is conflict with text unk.

                    # concatenate both conditioned and unconditioned batches as a single one.
                    context_and_question_tokens = torch.cat(
                        (context_and_question_tokens, context_and_question_tokens_unconditioned), dim=0
                    )
                    enc_mask = torch.cat((enc_mask, enc_mask), dim=0)
                    dec_input = torch.cat((dec_input, dec_input_unconditioned), dim=0)
                    position_ids = torch.cat((position_ids, position_ids), dim=0)

                # clean up useless variables.
                del question_limits, question_start, question_end, time_range, question_mask
            elif self.inference_apply_text_cfg:
                # replace question token IDs with [UNK]'s id. No speech offset for Phoneme's [UNK]. Same op as train.
                # instruction token IDs are bpe token IDs directly obtained from self.tokenizer without any offset.
                # question token IDs are phoneme and grapheme token IDs and are offset by self.lm_vocab_size
                #   if under "Phoneme TTS" instruction, so exising no overlaps between instruction and question token IDs.
                # question token IDs are bpe token IDs without any offset
                #   if under "Text to speech this" instruction, so existing overlaps between instruction and question token IDs.
                question_limits = text_limits - virtual_tokens.size(
                    1
                )  # (b, 2), reset question range to start from [pad] context, same start position as context_and_question_tokens.
                question_start = question_limits[:, 0].unsqueeze(1)  # (b, 1)
                question_end = question_limits[:, 1].unsqueeze(1)  # (b, 1)

                # duplicate and glue two batches into a single one.
                virtual_tokens = torch.cat((virtual_tokens, virtual_tokens), dim=0)
                taskname_ids = torch.cat((taskname_ids, taskname_ids), dim=0)
                speech_mask = torch.cat((speech_mask, speech_mask), dim=0)
                dec_input_mask = torch.cat((dec_input_mask, dec_input_mask), dim=0)

                if isinstance(context_and_question_tokens, list):  # indicate self.encoder_type = "multi_transformers".
                    context_tokens, question_tokens = context_and_question_tokens
                    question_tokens_unconditioned = question_tokens.clone()

                    time_range = torch.arange(
                        question_tokens_unconditioned.size(2), device=question_tokens_unconditioned.device
                    ).unsqueeze(0)
                    question_mask = (time_range >= question_start) & (
                        time_range < question_end
                    )  # create a mask for question only tokens.
                    question_tokens_unconditioned[:, 0][
                        question_mask
                    ] = self.tokenizer.unk_id  # only the first layer has non-zero IDs.

                    # concatenate both conditioned and unconditioned batches as a single one.
                    context_and_question_tokens = [
                        torch.cat((context_tokens, context_tokens), dim=0),
                        torch.cat((question_tokens, question_tokens_unconditioned), dim=0),
                    ]
                    enc_mask = [torch.cat((mask, mask), dim=0) for mask in enc_mask]
                    dec_input = torch.cat((dec_input, dec_input), dim=0)
                    position_ids = [torch.cat((pos_ids, pos_ids), dim=0) for pos_ids in position_ids]
                else:
                    assert (
                        self.context_conditioning == "decoder"
                    ), f"The encoder_type is single_transformer. We expect context_condition is decoder: context_condition={self.context_conditioning}"
                    context_and_question_tokens_unconditioned = context_and_question_tokens.clone()
                    time_range = torch.arange(
                        context_and_question_tokens_unconditioned.size(2),
                        device=context_and_question_tokens_unconditioned.device,
                    ).unsqueeze(
                        0
                    )  # (1, max_context_question_tokens_len)
                    question_mask = (time_range >= question_start) & (
                        time_range < question_end
                    )  # create a mask for question only tokens.
                    context_and_question_tokens_unconditioned[:, 0][
                        question_mask
                    ] = self.tokenizer.unk_id  # only the first layer has non-zero IDs.

                    # concatenate both conditioned and unconditioned batches as a single one.
                    context_and_question_tokens = torch.cat(
                        (context_and_question_tokens, context_and_question_tokens_unconditioned), dim=0
                    )
                    enc_mask = torch.cat((enc_mask, enc_mask), dim=0)
                    dec_input = torch.cat((dec_input, dec_input), dim=0)
                    position_ids = torch.cat((position_ids, position_ids), dim=0)

                # clean up useless variables.
                del question_limits, question_start, question_end, time_range, question_mask
            elif self.inference_apply_audio_cfg:
                # duplicate and glue two batches into a single one.
                virtual_tokens = torch.cat((virtual_tokens, virtual_tokens), dim=0)
                taskname_ids = torch.cat((taskname_ids, taskname_ids), dim=0)
                speech_mask = torch.cat((speech_mask, speech_mask), dim=0)
                dec_input_mask = torch.cat((dec_input_mask, dec_input_mask), dim=0)

                if isinstance(
                    context_and_question_tokens, list
                ):  # indicate that self.encoder_type = "multi_transformers"
                    context_tokens, question_tokens = context_and_question_tokens
                    context_tokens_unconditioned = context_tokens.clone()
                    context_tokens_unconditioned[:, :, :] = (
                        self.tokenizer.unk_id
                    )  # TODO @xueyang: verify if extra tokens other than audio codec tokens are appended.

                    # concatenate both conditioned and unconditioned batches as a single one.
                    context_and_question_tokens = [
                        torch.cat((context_tokens, context_tokens_unconditioned), dim=0),
                        torch.cat((question_tokens, question_tokens), dim=0),
                    ]
                    enc_mask = [torch.cat((mask, mask), dim=0) for mask in enc_mask]
                    dec_input = torch.cat((dec_input, dec_input), dim=0)
                    position_ids = [torch.cat((pos_ids, pos_ids), dim=0) for pos_ids in position_ids]
                else:
                    assert (
                        self.context_conditioning == "decoder"
                    ), f"The encoder_type is single_transformer. We expect context_condition is decoder: context_condition={self.context_conditioning}"
                    dec_input_unconditioned = dec_input.clone()
                    dec_input_unconditioned[:, :, 1 : self.decoder_context_len + 1] = (
                        self.tokenizer.unk_id
                    )  # TODO @xueyang: switch to other token id if this one is conflict with text unk.

                    # concatenate both conditioned and unconditioned batches as a single one.
                    context_and_question_tokens = torch.cat(
                        (context_and_question_tokens, context_and_question_tokens), dim=0
                    )
                    enc_mask = torch.cat((enc_mask, enc_mask), dim=0)
                    dec_input = torch.cat((dec_input, dec_input_unconditioned), dim=0)
                    position_ids = torch.cat((position_ids, position_ids), dim=0)
            else:
                logging.debug(
                    f"Neither text or audio cfg logits are applied:"
                    f" self.inference_apply_text_cfg={self.inference_apply_text_cfg},"
                    f" self.inference_apply_audio_cfg={self.inference_apply_audio_cfg}"
                )

            end_inference_loop_at = None
            fwd_bwd_function = get_forward_backward_func()
            encoder_output = None
            attention_probs_all = []
            start_time = time.time()
            for t in range(self.decoder_context_len + 1, dec_input.shape[2] - 1):
                # Start at 0 if encoder context, else context_len
                if t % 100 == 0:
                    logging.info("Timestep {}".format(t))
                if t == end_inference_loop_at:
                    print("All ends detected")
                    break

                if isinstance(enc_mask, list):
                    encoder_max_sequence_len = [e.size(1) for e in enc_mask]
                else:
                    encoder_max_sequence_len = enc_mask.size(1)

                # if context_condition is decoder, then t starts at [PAD] token represented as [0] * 8.
                # if context_condition is encoder, then t starts at [CLS].
                if t == self.decoder_context_len + 1:
                    # Run first step manually
                    output_logits, _, token_and_speech_logits = self.forward(
                        virtual_tokens,
                        context_and_question_tokens,
                        enc_mask,
                        dec_input[
                            :, :, : t + 1
                        ],  # tensors representing [CLS] + context audio tokens + [PAD] if context_condition is decoder, otherwise, tensors representing [CLS].
                        dec_input_mask[:, : t + 1],  # doesn't matter because of all ones.
                        position_ids,
                        taskname_ids,
                        labels=None,
                        speech_mask=speech_mask,
                        inference=True,
                        inference_step=0,
                        decoder_max_sequence_len=max_inference_timesteps,
                        encoder_max_sequence_len=encoder_max_sequence_len,
                    )
                    encoder_output = token_and_speech_logits[-1]

                    if isinstance(encoder_output, list):
                        encoder_output = [e.transpose(0, 1) for e in encoder_output]
                    else:
                        encoder_output = encoder_output.transpose(0, 1)

                else:
                    # Prepare batch
                    batch = [
                        max_inference_timesteps,
                        encoder_max_sequence_len,
                        encoder_output,
                        enc_mask,
                        dec_input[:, :, : t + 1],
                        dec_input_mask[:, : t + 1],
                        position_ids,
                        taskname_ids,
                        speech_mask,
                    ]

                    output_tensor = fwd_bwd_function(
                        forward_step_func=self.get_forward_output_only_func(),
                        data_iterator=iter(
                            [
                                batch,
                            ]
                        ),
                        model=[self],
                        num_microbatches=get_num_microbatches(),
                        forward_only=True,
                        seq_length=t,
                        micro_batch_size=dec_input.shape[0],
                    )
                    output_logits = output_tensor[0]['output_logits']  # (B, T, V, 8) or (2B, T, V, 8)
                    token_and_speech_logits = output_tensor[0]['token_and_speech_logits']

                    # when return_all_crossattention is False, attention_probs is None.
                    if self.frozen_model.enc_dec_model.return_all_crossattention_probs:
                        attention_probs = token_and_speech_logits[2]
                        attention_probs_mean = torch.stack(attention_probs).mean(dim=0)  # B, 12, 1, enc_timesteps
                        attention_probs_all.append(attention_probs_mean)

                if self.inference_apply_text_cfg or self.inference_apply_audio_cfg:
                    # interpolate conditioned and unconditioned logits
                    token_logits = (
                        self.inference_cfg_interpolation_scale * token_and_speech_logits[0][:batch_size]
                        + (1 - self.inference_cfg_interpolation_scale) * token_and_speech_logits[0][batch_size:]
                    )
                    output_speech_logits = (
                        self.inference_cfg_interpolation_scale * output_logits[:batch_size]
                        + (1 - self.inference_cfg_interpolation_scale) * output_logits[batch_size:]
                    )
                else:
                    token_logits = token_and_speech_logits[0]  # (B, T, V)
                    output_speech_logits = output_logits

                token_logits_currtimestep = token_logits[:, -1, :]  # (B, V)
                token_preds = token_logits_currtimestep.argmax(dim=1)  # (B,)

                if torch.count_nonzero(speech_mask) > 0:
                    output_logits_currtimestep = (
                        output_speech_logits[:, -1, :, :]
                        .permute(0, 2, 1)
                        .contiguous()
                        .view(-1, self.speech_codebook_size)
                    )  # (B*8, V)
                    output_logits_currtimestep_conditioned = (
                        output_logits[:batch_size][:, -1, :, :]
                        .permute(0, 2, 1)
                        .contiguous()
                        .view(-1, self.speech_codebook_size)
                    )
                    output_logits_currtimestep_unconditioned = (
                        output_logits[batch_size:][:, -1, :, :]
                        .permute(0, 2, 1)
                        .contiguous()
                        .view(-1, self.speech_codebook_size)
                    )
                else:
                    output_logits_currtimestep = token_logits_currtimestep  # (B, V)
                    output_logits_currtimestep_conditioned = token_logits_currtimestep
                    output_logits_currtimestep_unconditioned = token_logits_currtimestep

                top_k = self.cfg.get('top_k', 80)

                # (B*8, 80) or (B, 80)
                output_logits_currtimestep_topk = torch.topk(output_logits_currtimestep, top_k, dim=1)[0]

                # find indices which are not top k
                indices_to_remove = output_logits_currtimestep < output_logits_currtimestep_topk[:, -1].unsqueeze(1)
                # (B*8, 1024) or (B, 1024)

                if self.inference_apply_cfg_filter:
                    output_logits_currtimestep_rescored = output_logits_currtimestep_conditioned.clone()
                else:
                    output_logits_currtimestep_rescored = output_logits_currtimestep.clone()

                output_logits_currtimestep_rescored[indices_to_remove] = -float('Inf')

                # logits interpolation between conditioned and unconditioned logits.
                if (
                    self.inference_apply_text_cfg or self.inference_apply_audio_cfg
                ) and self.inference_apply_cfg_filter:
                    output_logits_currtimestep_rescored = (
                        self.inference_cfg_filter_interpolation_scale * output_logits_currtimestep_rescored
                        + (1 - self.inference_cfg_filter_interpolation_scale)
                        * output_logits_currtimestep_unconditioned
                    )

                temperature = self.cfg.get('temperature', 0.85)  # Set temp 0.01 for greedy decoding
                output_logits_currtimestep_rescored = output_logits_currtimestep_rescored / temperature
                output_logits_currtimestep_rescored = torch.nn.functional.softmax(
                    output_logits_currtimestep_rescored, dim=1
                )

                output_tokens_curr_timestep = torch.multinomial(
                    output_logits_currtimestep_rescored, num_samples=1
                )  # (B*8, 1)

                if torch.count_nonzero(speech_mask) > 0:
                    # Convert back to (B, 8)
                    output_tokens_curr_timestep = output_tokens_curr_timestep.view(
                        batch_size, self.num_speech_codebooks
                    )

                for _b in range(token_preds.shape[0]):
                    if t > self.decoder_context_len + 10 and token_preds[_b] == self.tokenizer.eos_id:
                        if _b not in end_indices:
                            logging.info("End detected for item {}".format(_b) + " at timestep {}".format(t))
                            end_indices[_b] = t
                            if len(end_indices) == token_preds.shape[0]:
                                end_inference_loop_at = t + self.num_speech_codebooks

                output_token_list.append(output_tokens_curr_timestep)

                # duplicate to 2b dim as input for the next iteration if enabling cfg.
                if self.inference_apply_text_cfg or self.inference_apply_audio_cfg:
                    output_tokens_curr_timestep = torch.cat(
                        (output_tokens_curr_timestep, output_tokens_curr_timestep), dim=0
                    )

                if torch.count_nonzero(speech_mask) > 0:
                    dec_input_next_timestep = output_tokens_curr_timestep * 1  # (B,8)
                    dec_input_next_timestep[:, 0] = (
                        dec_input_next_timestep[:, 0] + self.speech_offset
                    )  # add offset to first codebook
                    dec_input[:, :, t + 1] = dec_input_next_timestep * 1
                else:
                    dec_input[:, 0, t + 1] = output_tokens_curr_timestep.squeeze(1)

            # end of for loop
            output_tokens_combined = torch.stack(output_token_list)  # (T, B, 8) if speech else (T, B)
            if torch.count_nonzero(speech_mask) > 0:
                output_tokens_combined = output_tokens_combined.permute(1, 2, 0)  # (B, 8, T)
            else:
                output_tokens_combined = output_tokens_combined.squeeze(2)
                output_tokens_combined = output_tokens_combined.permute(1, 0)  # (B, T)

            # consider only autoregressive time, disconsider loading eval models for RTF time
            total_process_time = time.time() - start_time

            # Layerwise token error rate
            ter_dict = {}
            for i in range(self.num_speech_codebooks):
                ter_dict[i] = {'hypothesis': [], 'gt': []}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if 'nemo_sv_model' not in self.additional_models:
                nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
                nemo_sv_model = nemo_sv_model.to(device)
                nemo_sv_model.encoder.disable_torch_distributed = True  # For multi-gpu training validation
                nemo_sv_model.eval()
                self.additional_models['nemo_sv_model'] = nemo_sv_model
                logging.info(f"Loaded SV Model: {nemo_sv_model}")
            else:
                nemo_sv_model = self.additional_models['nemo_sv_model']

            if 'asr_model' not in self.additional_models:
                asr_model = self.cfg.get("asr_model_name", "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu")

                if "hybrid" in asr_model:
                    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
                else:
                    model = nemo_asr.models.EncDecRNNTBPEModel
                asr_model = model.from_pretrained(model_name=asr_model)
                asr_model = asr_model.to(device)
                asr_model.encoder.disable_torch_distributed = True  # For multi-gpu training validation
                asr_model.eval()
                self.additional_models['asr_model'] = asr_model
                logging.info(f"Loaded ASR Model: {asr_model}")
            else:
                asr_model = self.additional_models['asr_model']

            asr_model_zh = None
            if Lang.zh.value in lang:
                if 'asr_model_zh' not in self.additional_models:
                    asr_model_zh = nemo_asr.models.EncDecRNNTModel.from_pretrained(
                        model_name="stt_zh_conformer_transducer_large"
                    )
                    asr_model_zh = asr_model_zh.to(device)
                    asr_model_zh.eval()
                    self.additional_models['asr_model_zh'] = asr_model_zh
                else:
                    asr_model_zh = self.additional_models['asr_model_zh']

            if 'wavlm_sv_model' not in self.additional_models:
                wavlm_sv_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
                wavlm_sv_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
                wavlm_sv_model = wavlm_sv_model.to(device)
                wavlm_sv_model = wavlm_sv_model.eval()
                self.additional_models['wavlm_sv_model'] = wavlm_sv_model
                self.additional_models['wavlm_sv_extractor'] = wavlm_sv_extractor
                logging.info(f"Loaded SV Model: {wavlm_sv_model}")
            else:
                wavlm_sv_model = self.additional_models['wavlm_sv_model']
                wavlm_sv_extractor = self.additional_models['wavlm_sv_extractor']

            # load MOS estimator model only if True.
            if self.estimate_mos:
                # load mos estimator.
                if 'squim_mos_model' not in self.additional_models:
                    squim_mos_model_full = SQUIM_SUBJECTIVE.get_model().to(device)
                    self.additional_models['squim_mos_model'] = squim_mos_model_full
                else:
                    squim_mos_model_full = self.additional_models['squim_mos_model']

                # load non-matching reference clean audio.
                ref_16khz_wav, _ = librosa.load(self.non_matching_ref_audio_filepath, sr=16000)

                # prepare MOS estimator by taking a single audio example as an input.
                squim_mos_model = partial(
                    squim_mos_model_full, reference=torch.from_numpy(ref_16khz_wav).to(device).unsqueeze(0)
                )

            _exp_dir_path = self.logger.log_dir
            _exp_dir_path = _exp_dir_path + '/Sample_Audios'
            if not os.path.exists(_exp_dir_path):
                os.mkdir(_exp_dir_path)

            squim_mos_list_pred = []
            squim_mos_list_context = []
            squim_mos_list_gt = []
            similarity_list = []
            similarity_list_wavlm = []
            pred_context_similarity_list = []
            pred_context_similarity_list_wavlm = []
            gt_context_similarity_list = []
            gt_context_similarity_list_wavlm = []
            question_type = []

            # predicting audio
            batch_size = output_tokens_combined.shape[0]
            test_dataloader_batch_size = batch_size
            # self.test_dataloader() is not defined during validation
            if isinstance(self.test_dataloader(), torch.utils.data.DataLoader):
                test_dataloader_batch_size = self.test_dataloader().batch_size

            # logging attention maps.
            # empty attention_probs_all indicates self.frozen_model.enc_dec_model.return_all_crossattention_probs is False.
            if len(attention_probs_all) != 0:
                attention_probs_all = torch.cat(attention_probs_all, dim=2)  # B, 12, dec_timesteps, enc_timesteps
                attention_probs_all = attention_probs_all.mean(dim=1)  # B, dec_timesteps, enc_timesteps

                for i in range(batch_size):
                    text_end_step = text_limits[i, 1].item()
                    text_start_step = text_limits[i, 0].item()
                    end_index = end_indices.get(i, output_tokens_combined.shape[2])
                    if len(attention_probs_all) != 0:
                        attention_probs_example = attention_probs_all[i][
                            : end_index - (1 + self.decoder_context_len), text_start_step:text_end_step
                        ]  # T, enc_timesteps
                        attention_map = attention_probs_example.float().cpu().numpy().T
                        alignment_image = plot_alignment_to_numpy_for_speechllm(
                            attention_map,
                            phoneme_ver=1,
                            phoneme_seq=None,
                        )

                        if global_step is not None:
                            # During validation, step is simply global_step + i
                            step = global_step + i
                        else:
                            # During inference, step is the index of the sample
                            step = batch_idx * test_dataloader_batch_size + i

                        self.logger.experiment.add_image(
                            "Inf Attention Map",
                            alignment_image,
                            step,
                            dataformats="HWC",
                        )
                        # Save attention image to file
                        alignment_fp = os.path.join(_exp_dir_path, f'attention_map_{step}.png')
                        imageio.imwrite(alignment_fp, alignment_image)

            wer_score = 0
            audio_to_pred = []
            audio_to_pred_zh = []
            total_audio_seconds = 0
            for i in range(batch_size):
                if global_step is not None:
                    # During validation, step is simply global_step + i
                    step = global_step + i
                else:
                    # During inference, step is the index of the sample
                    step = batch_idx * test_dataloader_batch_size + i

                audio_len = self.decoder_context_len + (labels[i][0][self.decoder_context_len :] != 0).sum().item()

                if torch.count_nonzero(speech_mask) > 0:
                    dec_input_to_1024 = self.convert_tokens_to_range(dec_input_raw[i, :, 0:audio_len])
                    dec_input_to_1024_answer = dec_input_to_1024[:, self.decoder_context_len + 1 :]
                    dec_input_wav = self.decode_wav_from_codec_model(dec_input_to_1024_answer)
                    self.logger.experiment.add_audio("Inf Dec Input Wav", dec_input_wav, step, self.sample_rate)

                    predicted_tokens = output_tokens_combined[i]  # Should not contain context even if decoder context
                    if i in end_indices:
                        logging.info(f"Clipping until end index for audio {i}")
                        if self.cfg.get('seq_pattern', 'parallel') == 'delay_parallel':
                            predicted_tokens = predicted_tokens[
                                :, 0 : end_indices[i] - (1 + self.decoder_context_len) + self.num_speech_codebooks
                            ]  # trim to audio length
                        else:
                            predicted_tokens = predicted_tokens[
                                :, 0 : end_indices[i] - (1 + self.decoder_context_len)
                            ]  # trim to audio length

                    pred_img = predicted_tokens.data.cpu().float().numpy()
                    dec_inp_img = dec_input_to_1024.data.cpu().float().numpy()
                    start_time = time.time()
                    predicted_tokens = self.convert_tokens_to_range(predicted_tokens, apply_offset_correction=False)
                    predicted_wav = self.decode_wav_from_codec_model(predicted_tokens)
                    # accumulate audio length in seconds and process time in seconds to the RTF
                    total_process_time = total_process_time + (time.time() - start_time)
                    total_audio_seconds = total_audio_seconds + predicted_wav.size(-1) / self.sample_rate

                    self.logger.experiment.add_audio("Inf Pred Wav", predicted_wav, step, self.sample_rate)
                    self.logger.experiment.add_image(
                        "Inf Pred Tokens",
                        plot_codec_to_numpy(pred_img),
                        step,
                        dataformats="HWC",
                    )
                    self.logger.experiment.add_image(
                        "Inf Dec Input Tokens",
                        plot_codec_to_numpy(dec_inp_img),
                        step,
                        dataformats="HWC",
                    )

                    # save predicted_wav and gt_wav to a wav files in dir_path
                    if global_step is not None:
                        # During training, overwrite the wav file from the previous validation
                        wav_num = i
                    else:
                        wav_num = step

                    audio_fp_pred = os.path.join(_exp_dir_path, f'predicted_wav_{wav_num}.wav')
                    sf.write(audio_fp_pred, predicted_wav.cpu().numpy(), self.sample_rate)
                    audio_fp_gt = os.path.join(_exp_dir_path, f'dec_input_wav_{wav_num}.wav')
                    sf.write(audio_fp_gt, dec_input_wav.cpu().numpy(), self.sample_rate)

                    # speaker verification evaluation using nemo model
                    spk_embedding_pred = nemo_sv_model.get_embedding(audio_fp_pred)
                    spk_embedding_pred = spk_embedding_pred.cpu().detach().numpy().flatten()
                    spk_embedding_gt = nemo_sv_model.get_embedding(audio_fp_gt)
                    spk_embedding_gt = spk_embedding_gt.cpu().detach().numpy().flatten()
                    similarity = np.dot(spk_embedding_pred, spk_embedding_gt) / (
                        np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt)
                    )

                    if log_scalars:
                        self.logger.experiment.add_scalar(f'Inf SV Cossim Individual Sample', similarity, step)
                    similarity_list.append(similarity)

                    # speaker verification evaluation using wavlm model
                    gt_16khz_wav, _ = librosa.load(audio_fp_gt, sr=16000)
                    pred_16khz_wav, _ = librosa.load(audio_fp_pred, sr=16000)
                    inputs_wavlm = wavlm_sv_extractor(
                        [pred_16khz_wav, gt_16khz_wav], padding=True, return_tensors="pt", sampling_rate=16000
                    )
                    for key in inputs_wavlm.keys():
                        inputs_wavlm[key] = inputs_wavlm[key].to(device)

                    with torch.no_grad():
                        wavlm_embeddings = wavlm_sv_model(**inputs_wavlm).embeddings
                        wavlm_embeddings = torch.nn.functional.normalize(wavlm_embeddings, dim=-1).cpu()

                    spk_embedding_pred_wavlm = wavlm_embeddings[0].cpu().detach().numpy().flatten()
                    spk_embedding_gt_wavlm = wavlm_embeddings[1].cpu().detach().numpy().flatten()
                    similarity_wavlm = np.dot(spk_embedding_pred_wavlm, spk_embedding_gt_wavlm) / (
                        np.linalg.norm(spk_embedding_pred_wavlm) * np.linalg.norm(spk_embedding_gt_wavlm)
                    )
                    similarity_list_wavlm.append(similarity_wavlm)

                    if lang[i] == Lang.zh.value:
                        audio_to_pred_zh.append({"step": i, "audio": audio_fp_pred})
                        audio_to_pred_zh.append({"step": i, "audio": audio_fp_gt})
                    else:
                        audio_to_pred.append({"step": i, "audio": audio_fp_pred})
                        audio_to_pred.append({"step": i, "audio": audio_fp_gt})

                    if isinstance(context_and_question_tokens, list):
                        context_tokens, question_tokens = context_and_question_tokens
                        input_token_list = [
                            question_tokens[i, 0, j].item()
                            for j in range(context_and_question_tokens_lens[1][i].item())
                        ]
                        input_token_list = [
                            (ti, t) for ti, t in enumerate(input_token_list) if t != 0 and t < self.speech_offset
                        ]
                        context_end_step = context_and_question_tokens_lens[0][i]
                        context_tokens = context_tokens[i][:, :context_end_step]
                    else:
                        input_token_list = [
                            context_and_question_tokens[i, 0, j].item()
                            for j in range(context_and_question_tokens.shape[2])
                        ]
                        input_token_list = [
                            (ti, t) for ti, t in enumerate(input_token_list) if t != 0 and t < self.speech_offset
                        ]
                        context_end_step = input_token_list[0][0]
                        context_tokens = context_and_question_tokens[i][:, :context_end_step]

                    spk_embedding_context = spk_embedding_gt
                    spk_embedding_context_wavlm = spk_embedding_gt_wavlm
                    if self.decoder_context_len > 0:
                        context_tokens = dec_input_to_1024[:, : self.decoder_context_len + 1]
                        context_wav = self.decode_wav_from_codec_model(context_tokens)
                    elif context_end_step > 1:
                        is_speech_context = context_tokens[1, :].sum().item() > 0
                        if is_speech_context:
                            context_tokens = self.convert_tokens_to_range(context_tokens, pattern=self.context_pattern)
                            context_wav = self.decode_wav_from_codec_model(context_tokens)
                        else:
                            context_wav = None
                            _context_token_list = [v.item() for v in context_tokens[0, :]]
                            _context_text = self.frozen_model.tokenizer.ids_to_text(
                                [v for v in _context_token_list if v < self.lm_vocab_size]
                            )
                            self.logger.experiment.add_text("Context Text", _context_text, self.global_step)

                    else:
                        context_wav = None

                    if context_wav is not None:
                        self.logger.experiment.add_audio("Context Wav", context_wav, step, self.sample_rate)
                        context_wav_fp = os.path.join(_exp_dir_path, f'context_wav_{wav_num}.wav')
                        sf.write(context_wav_fp, context_wav.cpu().numpy(), self.sample_rate)
                        # titanet
                        spk_embedding_context = nemo_sv_model.get_embedding(context_wav_fp)
                        spk_embedding_context = spk_embedding_context.cpu().detach().numpy().flatten()
                        # wavlm
                        context_wavlm_wav, _ = librosa.load(context_wav_fp, sr=16000)
                        inputs_wavlm = wavlm_sv_extractor(
                            [context_wavlm_wav], padding=True, return_tensors="pt", sampling_rate=16000
                        )
                        for key in inputs_wavlm.keys():
                            inputs_wavlm[key] = inputs_wavlm[key].to(device)

                        with torch.no_grad():
                            wavlm_embeddings = wavlm_sv_model(**inputs_wavlm).embeddings
                            wavlm_embeddings = torch.nn.functional.normalize(wavlm_embeddings, dim=-1).cpu()

                        spk_embedding_context_wavlm = wavlm_embeddings[0].cpu().detach().numpy().flatten()

                    pred_similarity_context = np.dot(spk_embedding_context, spk_embedding_pred) / (
                        np.linalg.norm(spk_embedding_context) * np.linalg.norm(spk_embedding_pred)
                    )
                    gt_similarity_context = np.dot(spk_embedding_context, spk_embedding_gt) / (
                        np.linalg.norm(spk_embedding_context) * np.linalg.norm(spk_embedding_gt)
                    )

                    pred_similarity_context_wavlm = np.dot(spk_embedding_context_wavlm, spk_embedding_pred_wavlm) / (
                        np.linalg.norm(spk_embedding_context_wavlm) * np.linalg.norm(spk_embedding_pred_wavlm)
                    )
                    gt_similarity_context_wavlm = np.dot(spk_embedding_context_wavlm, spk_embedding_gt_wavlm) / (
                        np.linalg.norm(spk_embedding_context_wavlm) * np.linalg.norm(spk_embedding_gt_wavlm)
                    )

                    if log_scalars:
                        self.logger.experiment.add_scalar(f'Inf SV Cossim Context Pred', pred_similarity_context, step)
                        self.logger.experiment.add_scalar(f'Inf SV Cossim Context GT', gt_similarity_context, step)
                    pred_context_similarity_list.append(pred_similarity_context)
                    gt_context_similarity_list.append(gt_similarity_context)
                    pred_context_similarity_list_wavlm.append(pred_similarity_context_wavlm)
                    gt_context_similarity_list_wavlm.append(gt_similarity_context_wavlm)

                    task_question = self.frozen_model.tokenizer.ids_to_text(
                        [v[1] for v in input_token_list if v[1] < self.lm_vocab_size]
                    )
                    self.logger.experiment.add_text("Inf Task Question", task_question, step)
                    if "Phoneme TTS" in task_question:
                        question_type.append("Phoneme TTS")
                    elif "Text to speech this" in task_question:
                        question_type.append("Text to speech this")
                    else:
                        question_type.append("Other")

                    task_question_phoneme_tokens = [
                        v[1] - self.lm_vocab_size for v in input_token_list if v[1] >= self.lm_vocab_size
                    ]
                    if len(task_question_phoneme_tokens) > 0:
                        phoneme_text = self.phoneme_tokenizer.decode(task_question_phoneme_tokens)
                        self.logger.experiment.add_text("Inf Task Question Phoneme Text", phoneme_text, step)

                    # store predicted_tokens for each layer to compute token error rate
                    for layer_idx in range(self.num_speech_codebooks):
                        ter_dict[layer_idx]['hypothesis'].append(predicted_tokens[layer_idx].cpu().numpy().tolist())
                        ter_dict[layer_idx]['gt'].append(dec_input_to_1024_answer[layer_idx].cpu().numpy().tolist())

                    # estimate MOS scores.
                    if self.estimate_mos:
                        squim_mos_score_pred = squim_mos_model(
                            torch.from_numpy(pred_16khz_wav).to(device).unsqueeze(0)
                        ).item()
                        squim_mos_score_gt = squim_mos_model(
                            torch.from_numpy(gt_16khz_wav).to(device).unsqueeze(0)
                        ).item()
                        if context_wav is not None:
                            squim_mos_score_context = squim_mos_model(context_wav.to(device).unsqueeze(0)).item()
                            squim_mos_list_context.append(squim_mos_score_context)
                        squim_mos_list_pred.append(squim_mos_score_pred)
                        squim_mos_list_gt.append(squim_mos_score_gt)
                else:
                    r = labels[i, 0].long()
                    nzm = r != 0
                    r = r.tolist()[:-1]
                    nzm = nzm[:-1]
                    h = output_tokens_combined[i].long() * nzm
                    h = h.tolist()
                    cur_wer_score = editdistance.eval(r, h)
                    if log_scalars:
                        self.logger.experiment.add_scalar('WER', cur_wer_score, step)
                        logging.info(f"current wer score : {cur_wer_score}")
                    wer_score += cur_wer_score
            if wer_score > 0:
                wer_score /= batch_size
                if log_scalars:
                    self.logger.experiment.add_scalar('AVG WER', wer_score, step)
                    logging.info(f"average wer score : {wer_score}")

            # compute token error rate for each layer
            if log_scalars:
                for layer_idx in range(self.num_speech_codebooks):
                    wer = word_error_rate(ter_dict[layer_idx]['hypothesis'], ter_dict[layer_idx]['gt'], use_cer=True)
                    self.logger.experiment.add_scalar(f'Inf TER Layer {layer_idx}', wer, 0)

            greedy_transcripts = []
            if len(audio_to_pred) > 0:
                greedy_transcripts.extend(asr_model.transcribe([i["audio"] for i in audio_to_pred])[0])
            if len(audio_to_pred_zh) > 0:
                greedy_transcripts.extend(asr_model_zh.transcribe([i["audio"] for i in audio_to_pred_zh])[0])

            all_audio_to_pred = audio_to_pred + audio_to_pred_zh
            # Note WER over the batch is not equal to WER(sample) / batch_size, but approx. here

            # These are between ASR outputs of GT audio and predicted audio
            wer_batch = []
            cer_batch = []
            cer_phoneme = []
            wer_phoneme = []
            cer_tts = []
            wer_tts = []

            # These are between ASR output of Pred audio and GT text
            wer_batch_gt = []
            cer_batch_gt = []
            cer_phoneme_gt = []
            wer_phoneme_gt = []
            cer_tts_gt = []
            wer_tts_gt = []

            for i in range(0, len(greedy_transcripts) - 1, 2):
                assert all_audio_to_pred[i]["step"] == all_audio_to_pred[i + 1]["step"]
                step = batch_idx * test_dataloader_batch_size + all_audio_to_pred[i]["step"]
                question_text = question_texts[i // 2]

                # No need to process text since both are ASR outputs
                cer_sample = word_error_rate([greedy_transcripts[i]], [greedy_transcripts[i + 1]], use_cer=True)
                wer_sample = word_error_rate([greedy_transcripts[i]], [greedy_transcripts[i + 1]], use_cer=False)

                # Processing text since one is ASR output and the other is the GT text
                cer_gt = word_error_rate(
                    [self.process_text(greedy_transcripts[i])], [self.process_text(question_text)], use_cer=True
                )
                wer_gt = word_error_rate(
                    [self.process_text(greedy_transcripts[i])], [self.process_text(question_text)], use_cer=False
                )

                self.logger.experiment.add_text("Inf Predicted Text", greedy_transcripts[i], step)
                self.logger.experiment.add_text("Inf GT Text", greedy_transcripts[i + 1], step)
                self.logger.experiment.add_text("Inf Question Text", question_text, step)
                if log_scalars:
                    self.logger.experiment.add_scalar(f'Inf CER Transcript', cer_sample, step)
                    self.logger.experiment.add_scalar(f'Inf WER Transcript', wer_sample, step)
                    self.logger.experiment.add_scalar(f'Inf CER GT Transcript', cer_gt, step)
                cer_batch.append(cer_sample)
                wer_batch.append(wer_sample)
                cer_batch_gt.append(cer_gt)
                wer_batch_gt.append(wer_gt)
                if question_type[all_audio_to_pred[i]["step"]] == "Phoneme TTS":
                    if log_scalars:
                        self.logger.experiment.add_scalar(f'Inf CER Phoneme Task', cer_sample, step)
                        self.logger.experiment.add_scalar(f'Inf WER Phoneme Task', wer_sample, step)
                        self.logger.experiment.add_scalar(f'Inf CER GT Phoneme Task', cer_gt, step)
                    cer_phoneme.append(cer_sample)
                    wer_phoneme.append(wer_sample)
                    cer_phoneme_gt.append(cer_gt)
                    wer_phoneme_gt.append(wer_gt)
                elif question_type[all_audio_to_pred[i]["step"]] == "Text to speech this":
                    if log_scalars:
                        self.logger.experiment.add_scalar(f'Inf CER TTS Task', cer_sample, step)
                        self.logger.experiment.add_scalar(f'Inf WER TTS Task', wer_sample, step)
                        self.logger.experiment.add_scalar(f'Inf CER GT TTS Task', cer_gt, step)
                    cer_tts.append(cer_sample)
                    wer_tts.append(wer_sample)
                    cer_tts_gt.append(cer_gt)
                    wer_tts_gt.append(wer_gt)

            # compute average similarity
            similarity_avg = np.mean(similarity_list)
            pred_context_similarity_avg = np.mean(pred_context_similarity_list)
            gt_context_similarity_avg = np.mean(gt_context_similarity_list)
            similarity_avg_wavlm = np.mean(similarity_list_wavlm)
            pred_context_similarity_avg_wavlm = np.mean(pred_context_similarity_list_wavlm)
            gt_context_similarity_avg_wavlm = np.mean(gt_context_similarity_list_wavlm)

            if log_scalars:
                self.logger.experiment.add_scalar(f'Inf SV Avg Cossim', similarity_avg, batch_idx)
            self.predict_step_outputs.append(
                {
                    'titanet_avg_cossim': similarity_avg,
                    'titanet_avg_cossim_context_pred': pred_context_similarity_avg,
                    'titanet_avg_cossim_context_gt': gt_context_similarity_avg,
                    'wavlm_avg_cossim': similarity_avg_wavlm,
                    'wavlm_avg_cossim_context_pred': pred_context_similarity_avg_wavlm,
                    'wavlm_avg_cossim_context_gt': gt_context_similarity_avg_wavlm,
                    'squim_mos_pred': np.mean(squim_mos_list_pred) if len(squim_mos_list_pred) > 0 else None,
                    'squim_mos_context': np.mean(squim_mos_list_context) if len(squim_mos_list_context) > 0 else None,
                    'squim_mos_gt': np.mean(squim_mos_list_gt) if len(squim_mos_list_gt) > 0 else None,
                    'cer_transcript': np.mean(cer_batch),
                    'wer_transcript': np.mean(wer_batch),
                    'cer_phoneme': np.mean(cer_phoneme) if len(cer_phoneme) > 0 else None,
                    'wer_phoneme': np.mean(wer_phoneme) if len(wer_phoneme) > 0 else None,
                    'cer_tts': np.mean(cer_tts) if len(cer_tts) > 0 else None,
                    'wer_tts': np.mean(wer_tts) if len(wer_tts) > 0 else None,
                    'cer_transcript_gt': np.mean(cer_batch_gt),
                    'wer_transcript_gt': np.mean(wer_batch_gt),
                    'cer_phoneme_gt': np.mean(cer_phoneme_gt) if len(cer_phoneme_gt) > 0 else None,
                    'wer_phoneme_gt': np.mean(wer_phoneme_gt) if len(wer_phoneme_gt) > 0 else None,
                    'cer_tts_gt': np.mean(cer_tts_gt) if len(cer_tts_gt) > 0 else None,
                    'wer_tts_gt': np.mean(wer_tts_gt) if len(wer_tts_gt) > 0 else None,
                    "RTF": total_process_time / total_audio_seconds,
                }
            )

    # TODO @xueyang: PTL 2.0+ patch. Signature of method `on_predict_epoch_end` does not match signature of the base method in PTL class 'ModelHooks'.
    # Remove the `outputs` param and choose `self.predict_step_output` instead.
    def on_predict_epoch_end(self, outputs: List[Any]) -> None:

        gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        all_preds = list(itertools.chain(*[item['preds_text'] for item in outputs[0]]))
        all_labels = list(itertools.chain(*[item['labels_text'] for item in outputs[0]]))
        all_inputs = list(itertools.chain(*[item['input_text'] for item in outputs[0]]))

        assert len(all_preds) == len(all_labels)
        assert len(all_preds) == len(all_inputs)

        # Gather inputs, predictions, and ground truths from all workers
        torch.distributed.all_gather_object(
            gather_results,
            [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
            group=parallel_state.get_data_parallel_group(),
        )

        # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
        if parallel_state.get_data_parallel_rank() == 0:
            gather_results_dedup = list(set(itertools.chain(*gather_results)))

            input_prediction_pair = []
            correct = 0
            for input, pred, label in gather_results_dedup:
                input_prediction_pair.append((input, pred))
                if label:
                    if pred == label:
                        correct += 1

            acc = correct / len(gather_results_dedup) if all_labels[0] else None
            logging.info(f'Prediction results: {acc}')
            logging.info(f'Test finish')
