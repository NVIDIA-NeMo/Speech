# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

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

import torch

from nemo.collections.nlp.modules.common.megatron.language_model import Embedding
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.mup.layer import MuReadout
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts import utils_funcs

try:
    from apex.transformer.enums import ModelType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()

try:
    from megatron.core import ModelParallelConfig, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = True


__all__ = ["MegatronRetrievalTokenLevelEncoderDecoderModule"]


class MegatronRetrievalTokenLevelEncoderDecoderModule(MegatronModule):
    """Token-based (input/output is tokens) retrieval encoder-decoder model"""

    def __init__(
        self,
        config: ModelParallelConfig,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_cross_entropy=False,
        megatron_amp_O2=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        layernorm_epsilon=1e-5,
        persist_layer_norm=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        openai_gelu=False,
        activation='gelu',
        onnx_safe=False,
        bias=True,
        normalization='layernorm',
        headscale=False,
        transformer_block_type='pre_ln',
        hidden_steps=-1,
        add_encoder=True,
        add_decoder=True,
        chunk_size=64,
        enc_num_layers=4,  # total number of encoder layers
        dec_num_layers=6,  # total number of decoder layers
        enc_cross_attention=[3],  # layer numbers for cross attention
        dec_cross_attention=[3, 5],  # layer numbers for chunked cross attention
        add_position_embedding=False,
        tokenizer=None,  # tokenizer
        normalize_attention_scores=True,
        activations_checkpoint_granularity=None,
        megatron_lm_compatible=False,
        version=1,
    ):
        super(MegatronRetrievalTokenLevelEncoderDecoderModule, self).__init__()
        if megatron_lm_compatible:
            assert (
                apply_query_key_layer_scaling
            ), "megatron lm compatible model has to set apply_query_key_layer_scaling"
            assert add_position_embedding, "megatron lm compatible model has to set add_position_embedding"
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_cross_entropy = fp16_cross_entropy
        self.precision = precision
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.add_abs_position_embedding = add_position_embedding  # whether use absolute position embedding
        self.tokenizer = tokenizer
        self.eod_id = tokenizer.eos_id
        self.transformer_block_type = transformer_block_type
        self.num_chunked_cross_attention = len(dec_cross_attention)
        self.megatron_lm_compatible = megatron_lm_compatible

        self.dtype = utils_funcs.torch_dtype_from_precision(precision, megatron_amp_O2)

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        if pre_process:
            self.encoder_embedding = Embedding(
                config=config,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                max_sequence_length=max_position_embeddings,
                init_method=init_method_normal(init_method_std),
                num_tokentypes=num_tokentypes,
                embedding_dropout_prob=hidden_dropout,
                position_embedding_type='learned_absolute' if add_position_embedding else '',
                transpose_batch_sequence=False,
                dtype=self.dtype,
            )
            self._embedding_key = "embedding"

        encoder_init = init_method_normal(init_method_std)
        encoder_scaled_init = scaled_init_method_normal(init_method_std, dec_num_layers)
        pre_decoder_init = init_method_normal(init_method_std)
        pre_decoder_scaled_init = scaled_init_method_normal(init_method_std, dec_num_layers)
        post_decoder_init = init_method_normal(init_method_std)
        post_decoder_scaled_init = scaled_init_method_normal(init_method_std, dec_num_layers)

        if add_encoder:
            enc_layer_types = []
            for i in range(enc_num_layers):
                if i in enc_cross_attention:
                    enc_layer_types.append(LayerType.retrieval_encoder)
                else:
                    enc_layer_types.append(LayerType.encoder)

            self.encoder = get_encoder_model(
                config=config,
                arch="retro",
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_layers=enc_num_layers,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                init_method=encoder_init,
                scaled_init_method=encoder_scaled_init,
                pre_process=pre_process,
                post_process=False
                if megatron_lm_compatible
                else post_process,  # megatron lm model has no final layer_norm
                init_method_std=init_method_std,
                megatron_amp_O2=megatron_amp_O2,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                layernorm_epsilon=layernorm_epsilon,
                bias_activation_fusion=bias_gelu_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                hidden_steps=hidden_steps,
                activation=activation,
                bias=bias,
                normalization=normalization,
                transformer_block_type=transformer_block_type,
                headscale=headscale,
                parent_model_type=ModelType.encoder_and_decoder,
                layer_type=enc_layer_types,
                chunk_size=chunk_size,
                layer_number_offset=0,
                normalize_attention_scores=normalize_attention_scores,
                turn_off_rop=megatron_lm_compatible,
                version=version,
            )
            self._encoder_key = "encoder"

        if add_decoder:
            pre_decoder_num_layers = min(dec_cross_attention)
            pre_decoder_layer_types = []
            for i in range(pre_decoder_num_layers):
                pre_decoder_layer_types.append(LayerType.encoder)
            pre_decoder_layer_types.append(LayerType.decoder_pre_mlp)

            post_decoder_num_layers = dec_num_layers - pre_decoder_num_layers
            post_decoder_layer_types = []
            # the first layer in post decoder has to be chunked cross attention without self attention
            assert pre_decoder_num_layers in dec_cross_attention
            for i in range(post_decoder_num_layers):
                if i == 0:
                    post_decoder_layer_types.append(LayerType.retrieval_decoder_after_self_attn)
                elif i + pre_decoder_num_layers in dec_cross_attention:
                    post_decoder_layer_types.append(LayerType.retrieval_decoder)
                else:
                    post_decoder_layer_types.append(LayerType.encoder)

            # it is used to process the inputs for encoder to use as context (H in the paper)
            self.pre_decoder = get_decoder_model(
                config=config,
                arch="retro",
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_layers=pre_decoder_num_layers + 1,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                init_method=pre_decoder_init,
                scaled_init_method=pre_decoder_scaled_init,
                pre_process=pre_process,
                post_process=False,  # no need for post process
                init_method_std=init_method_std,
                megatron_amp_O2=megatron_amp_O2,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                layernorm_epsilon=layernorm_epsilon,
                bias_activation_fusion=bias_gelu_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                hidden_steps=hidden_steps,
                activation=activation,
                bias=bias,
                normalization=normalization,
                transformer_block_type=transformer_block_type,
                headscale=headscale,
                parent_model_type=ModelType.encoder_and_decoder,
                layer_type=pre_decoder_layer_types,
                chunk_size=chunk_size,
                layer_number_offset=0,
                normalize_attention_scores=normalize_attention_scores,
                turn_off_rop=megatron_lm_compatible,
                version=version,
            )

            # it is where the chunked cross attention happens
            self.post_decoder = get_decoder_model(
                config=config,
                arch="retro",
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_layers=post_decoder_num_layers,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                init_method=post_decoder_init,
                scaled_init_method=post_decoder_scaled_init,
                pre_process=False,  # directly take the pre_decoder output, skip preprocess
                post_process=post_process,
                init_method_std=init_method_std,
                megatron_amp_O2=megatron_amp_O2,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                layernorm_epsilon=layernorm_epsilon,
                bias_activation_fusion=bias_gelu_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                hidden_steps=hidden_steps,
                activation=activation,
                bias=bias,
                normalization=normalization,
                headscale=headscale,
                transformer_block_type=transformer_block_type,
                parent_model_type=ModelType.encoder_and_decoder,
                layer_type=post_decoder_layer_types,
                chunk_size=chunk_size,
                layer_number_offset=pre_decoder_num_layers + 1,
                normalize_attention_scores=normalize_attention_scores,
                turn_off_rop=megatron_lm_compatible,
                version=version,
            )
            self._pre_decoder_key = "pre_decoder"
            self._post_decoder_key = "post_decoder"

        self.initialize_word_embeddings(
            init_method=init_method_normal(init_method_std), vocab_size=vocab_size, hidden_size=hidden_size
        )

        if add_decoder and post_process:
            self.tokens_head = MuReadout(self.word_embeddings_weight().size(0), parallel_output)
            self._tokens_head_key = 'tokens_head'

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        input_ids,
        input_attn_mask,
        retrieved_ids,
        retrieved_attn_mask,
        token_type_ids=None,
        labels=None,
        input_emb=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        neighbors=None,
        position_ids=None,
    ):
        """
        Return value is per token / per dimension (i.e., non collapsed loss value)
        """
        eod_positions = None
        retrieved_emb = None
        if input_ids is not None and self.eod_id is not None and not self.megatron_lm_compatible:
            # do not reset attention for megatron lm compatible model
            eod_positions = torch.where(input_ids == self.eod_id)

        if input_emb is None:
            if self.pre_process and self.add_encoder:
                # encoder embeddings
                if self.add_abs_position_embedding:
                    input_position_ids = position_ids
                else:
                    input_position_ids = None
                input_emb = self.encoder_embedding(input_ids, input_position_ids, token_type_ids=token_type_ids)
            else:
                input_emb = None

        if retrieved_ids is not None:
            if self.add_abs_position_embedding:
                seq_length = retrieved_ids.size(-1)
                retrieved_position_ids = torch.arange(seq_length, dtype=torch.long, device=retrieved_ids.device)
                retrieved_position_ids = retrieved_position_ids.unsqueeze(0).expand_as(retrieved_ids).clone()
            else:
                retrieved_position_ids = None
            retrieved_emb = self.encoder_embedding(retrieved_ids, retrieved_position_ids)

        if self.add_decoder:
            hidden = self.pre_decoder(
                input_emb,
                input_attn_mask,
                eod_positions=eod_positions,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
            # hidden is a tuple, (layernorm_input, layernorm_output)
            self.post_decoder.set_input_tensor(hidden)
            encoder_output = hidden[1].transpose(0, 1).contiguous()

        if self.add_encoder:
            if retrieved_emb is not None and neighbors is None:
                neighbors = retrieved_emb.shape[2]
            retrieved_emb = self.encoder(
                retrieved_emb,
                retrieved_attn_mask,
                context_attn_mask=input_attn_mask,
                encoder_output=encoder_output,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                neighbors=neighbors,
            )

        if self.add_decoder:
            dec_output = self.post_decoder(
                hidden,
                input_attn_mask,
                retrieved_attn_mask=retrieved_attn_mask,
                retrieved_emb=retrieved_emb,
                eod_positions=eod_positions,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
            # only transpose it for post_ln
            token_logits = self.tokens_head(dec_output, self.word_embeddings_weight())

            if labels is not None:
                # [b, s] -> [s, b]
                labels = labels.transpose(0, 1).contiguous()

                # tensor_parallel.vocab_parallel_cross_entropy performs log_softmax and return log p(x_i|z) per token i
                if self.fp16_cross_entropy:
                    assert token_logits.dtype == torch.half
                    tokens_loss = tensor_parallel.vocab_parallel_cross_entropy(token_logits, labels)
                else:
                    tokens_loss = tensor_parallel.vocab_parallel_cross_entropy(token_logits.float(), labels)
                # [s, b] -> [b, s]
                tokens_loss = tokens_loss.transpose(0, 1).contiguous()
                return tokens_loss
            else:
                # [s, b, h] -> [b, s, h]
                token_logits = token_logits.transpose(0, 1).contiguous()
                return token_logits

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}

        state_dict_[self._encoder_embedding_key] = self.encoder_embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._pre_decoder_key] = self.pre_decoder.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._post_decoder_key] = self.post_decoder.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._tokens_head_key] = self.tokens_head.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.encoder_embedding.encoder_embeddingload_state_dict(state_dict[self._encoder_embedding_key], strict=strict)
        self.encoder.load_state_dict(state_dict[self._encoder_key], strict=strict)
        self.pre_decoder.load_state_dict(state_dict[self._pre_decoder_key], strict=strict)
        self.post_decoder.load_state_dict(state_dict[self._post_decoder_key], strict=strict)
        self.tokens_head.load_state_dict(state_dict[self._tokens_head_key], strict=strict)
