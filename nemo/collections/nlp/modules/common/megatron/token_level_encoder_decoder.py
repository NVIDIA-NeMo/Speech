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

# pylint: disable=C0301,C0115,C0116
import torch
from omegaconf import DictConfig

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    PromptEncoderAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.hiddens import get_hiddens_module
from nemo.collections.nlp.modules.common.megatron.language_model import Embedding
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_decoder import (
    MegatronTransformerEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.position_embedding import (
    ALiBiRelativePositionEmbedding,
    KERPLERelativePositionEmbedding,
    T5RelativePositionEmbedding,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
    init_method_normal,
    parallel_lm_logits,
    scaled_init_method_normal,
)
from nemo.collections.nlp.modules.common.megatron.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging

try:
    from apex.transformer.enums import AttnMaskType, ModelType

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()

    HAVE_APEX = False

try:
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

__all__ = [
    "MegatronTokenLevelHead",
    "MegatronTokenLevelEncoderDecoderModule",
    "MegatronTokenLevelEncoderDecoderSpeechLLMModule",
]


class MegatronTokenLevelHead(MegatronModule):
    """Masked LM head for token-based encoder-decoder models (e.g., T5)

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output, bias=True):
        super(MegatronTokenLevelHead, self).__init__()

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = 1
        else:
            self.bias = None
        self.parallel_output = parallel_output

    def forward(self, hidden_states, word_embeddings_weight):

        async_tensor_model_parallel_allreduce = parallel_state.get_tensor_model_parallel_world_size() > 1
        output = parallel_lm_logits(
            hidden_states,
            word_embeddings_weight,
            self.parallel_output,
            bias=self.bias,
            async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
        )
        return output


# TODO: add soft prompts as an Embedding sub-class


class MegatronTokenLevelEncoderDecoderModule(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """Token-based (input/output is tokens) encoder-decoder model (e.g. T5 Language model.)"""

    def __init__(
        self,
        config: ModelParallelConfig,
        encoder_cfg: DictConfig,
        decoder_cfg: DictConfig,
        vocab_size: int,  # TODO: This should eventually go inside encoder_cfg and decoder_cfg when separate enc/dec tokenizers are supported.
        max_position_embeddings,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        fp16_cross_entropy=False,
        precision=16,
        embedding_init_method_std=0.02,
        embedding_dropout=0.1,
        label_smoothing=0.0,
        add_encoder=True,
        add_decoder=True,
        share_token_embeddings=True,
        share_decoder_tokens_head_embeddings=True,
        tokens_head_bias=True,
        hiddens_cfg: DictConfig = None,  # allows for hidden state transformations before the decoder
    ):
        super(MegatronTokenLevelEncoderDecoderModule, self).__init__(config=config)

        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_cross_entropy = fp16_cross_entropy
        self.precision = precision
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.label_smoothing = label_smoothing
        self.share_token_embeddings = share_token_embeddings
        self.share_decoder_tokens_head_embeddings = share_decoder_tokens_head_embeddings
        self.tokens_head_bias = tokens_head_bias
        self.hiddens_cfg = hiddens_cfg

        encoder_kv_channels, decoder_kv_channels = self._validate_config()

        encoder, decoder = None, None
        if add_encoder:
            if pre_process:
                self.encoder_embedding = Embedding(
                    config=self.config,
                    hidden_size=encoder_cfg.hidden_size,
                    vocab_size=vocab_size,
                    max_sequence_length=max_position_embeddings,
                    init_method=init_method_normal(embedding_init_method_std),
                    num_tokentypes=num_tokentypes,
                    embedding_dropout_prob=embedding_dropout,
                    position_embedding_type=encoder_cfg.get('position_embedding_type', 'learned_absolute'),
                )
                self._encoder_embedding_key = "encoder_embedding"
            if self.encoder_cfg.get('position_embedding_type', 'learned_absolute') == 'relative':
                self.encoder_relative_position_embedding = T5RelativePositionEmbedding(
                    init_method=init_method_normal(embedding_init_method_std),
                    num_attention_heads=encoder_cfg.num_attention_heads,
                    relative_position_num_buckets=encoder_cfg.relative_attention_num_buckets,
                    relative_position_max_distance=encoder_cfg.relative_attention_max_distance,
                    bidirectional=True,
                    layer_type=LayerType.encoder,
                )
                self._encoder_relative_position_embedding_key = "encoder_relative_position_embedding"
                # Pipeline model parallel rank 0 will have the actual RPE weights. We zero it out on all other ranks and then sync them on setup.
                if parallel_state.get_pipeline_model_parallel_rank() != 0:
                    self.encoder_relative_position_embeddings_weight().data.fill_(0)
                    self.encoder_relative_position_embeddings_weight().shared = True
            elif self.encoder_cfg.get('position_embedding_type', 'learned_absolute') == 'alibi':
                self.encoder_relative_position_embedding = ALiBiRelativePositionEmbedding(
                    bidirectional=True,
                    num_attention_heads=encoder_cfg.num_attention_heads,
                    layer_type=LayerType.encoder,
                    num_attention_heads_alibi=None,
                    max_seq_len=max_position_embeddings,
                )
                self._encoder_relative_position_embedding_key = "encoder_alibi_position_embedding"
            elif self.encoder_cfg.get('position_embedding_type', 'learned_absolute') == 'kerple':
                self.encoder_relative_position_embedding = KERPLERelativePositionEmbedding(
                    bidirectional=True,
                    num_attention_heads=encoder_cfg.num_attention_heads,
                    layer_type=LayerType.encoder,
                    num_attention_heads_kerple=None,
                    max_seq_len=max_position_embeddings,
                )
                self._encoder_relative_position_embedding_key = "encoder_kerple_position_embedding"
            else:
                self.encoder_relative_position_embedding = None

            if encoder_cfg.get('use_flash_attention', False) and encoder_cfg.get(
                'position_embedding_type', 'learned_absolute'
            ) in ['relative', 'kerple']:
                raise ValueError('flash-attention not supported with relative or kerple at this point')

            encoder = get_encoder_model(
                config=config,
                arch=encoder_cfg.arch,
                hidden_size=encoder_cfg.hidden_size,
                ffn_hidden_size=encoder_cfg.ffn_hidden_size,
                num_layers=encoder_cfg.num_layers,
                num_attention_heads=encoder_cfg.num_attention_heads,
                apply_query_key_layer_scaling=encoder_cfg.get('apply_query_key_layer_scaling', True),
                kv_channels=encoder_kv_channels,
                init_method=init_method_normal(encoder_cfg.get('init_method_std', 0.02)),
                scaled_init_method=scaled_init_method_normal(
                    encoder_cfg.get('init_method_std', 0.02), encoder_cfg.num_layers
                ),
                encoder_attn_mask_type=AttnMaskType.padding,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=encoder_cfg.get('init_method_std', 0.02),
                hidden_dropout=encoder_cfg.get('hidden_dropout', 0.1),
                attention_dropout=encoder_cfg.get('attention_dropout', 0.1),
                ffn_dropout=encoder_cfg.get('ffn_dropout', 0.0),
                precision=precision,
                fp32_residual_connection=encoder_cfg.get('fp32_residual_connection', False),
                activations_checkpoint_method=encoder_cfg.get('activations_checkpoint_method', None),
                activations_checkpoint_num_layers=encoder_cfg.get('activations_checkpoint_num_layers', 1),
                activations_checkpoint_granularity=encoder_cfg.get('activations_checkpoint_granularity', None),
                layernorm_epsilon=encoder_cfg.get('layernorm_epsilon', 1e-5),
                bias_activation_fusion=encoder_cfg.get('bias_activation_fusion', True),
                bias_dropout_add_fusion=encoder_cfg.get('bias_dropout_add_fusion', True),
                masked_softmax_fusion=encoder_cfg.get('masked_softmax_fusion', True),
                persist_layer_norm=encoder_cfg.get('persist_layer_norm', True),
                openai_gelu=encoder_cfg.get('openai_gelu', False),
                onnx_safe=encoder_cfg.get('onnx_safe', False),
                hidden_steps=encoder_cfg.get('hidden_steps', -1),
                activation=encoder_cfg.get('activation', 'gelu'),
                bias=encoder_cfg.get('bias', True),
                normalization=encoder_cfg.get('normalization', 'layernorm'),
                transformer_block_type=encoder_cfg.get('transformer_block_type', 'pre_ln'),
                headscale=encoder_cfg.get('headscale', False),
                parent_model_type=ModelType.encoder_or_decoder,
                num_self_attention_per_cross_attention=encoder_cfg.get('num_self_attention_per_cross_attention', 1),
                megatron_legacy=encoder_cfg.get('megatron_legacy', False),
                normalize_attention_scores=encoder_cfg.get('normalize_attention_scores', True),
                num_moe_experts=encoder_cfg.get('num_moe_experts', 1),
                moe_frequency=encoder_cfg.get('moe_frequency', 1),
                moe_dropout=encoder_cfg.get('moe_dropout', 0.0),
                position_embedding_type=encoder_cfg.get('position_embedding_type', 'learned_absolute'),
                use_flash_attention=encoder_cfg.get('use_flash_attention', False),
                n_transformers=encoder_cfg.get('n_transformers', 1),
            )

        if add_decoder:
            # If this is the decoder first stage
            if pre_process:
                # If the encoder also lies on this rank (PP = 1), then just assign embeddings directly.
                if hasattr(self, 'encoder_embedding') and share_token_embeddings:
                    self.decoder_embedding = self.encoder_embedding
                else:
                    # This is the case where PP > 1 and first decoder first stage, or when not sharing embeddings with encoder
                    self.decoder_embedding = Embedding(
                        config=self.config,
                        hidden_size=decoder_cfg.hidden_size,
                        vocab_size=vocab_size,
                        max_sequence_length=max_position_embeddings,
                        init_method=init_method_normal(embedding_init_method_std),
                        num_tokentypes=num_tokentypes,
                        embedding_dropout_prob=embedding_dropout,
                        position_embedding_type=decoder_cfg.get('position_embedding_type', 'learned_absolute'),
                    )
                    # We initialize decoder embeddings, but set them to zero since we they're tied with the encoder embeddings.
                    # A later initialize_embedding call will synchronize the embeddings.
                    if share_token_embeddings:
                        self.decoder_embedding.zero_parameters()

                self._decoder_embedding_key = "decoder_embedding"

            if self.decoder_cfg.get('position_embedding_type', 'learned_absolute') == 'relative':
                self.decoder_relative_position_embedding = T5RelativePositionEmbedding(
                    init_method=init_method_normal(embedding_init_method_std),
                    num_attention_heads=decoder_cfg.num_attention_heads,
                    relative_position_num_buckets=decoder_cfg.relative_attention_num_buckets,
                    relative_position_max_distance=decoder_cfg.relative_attention_max_distance,
                    bidirectional=False,
                    layer_type=LayerType.decoder,
                )
                self._decoder_relative_position_embedding_key = "decoder_relative_position_embedding"
                # Pipeline model parallel rank == split_rank will have the actual RPE weights. We zero it out on all other ranks and then sync them on setup.
                if (
                    parallel_state.get_pipeline_model_parallel_rank()
                    != parallel_state.get_pipeline_model_parallel_split_rank()
                ):
                    self.decoder_relative_position_embeddings_weight().data.fill_(0)
                    self.decoder_relative_position_embeddings_weight().shared = True

                if not self.decoder_cfg.relative_position_bias_self_attention_only:
                    self.decoder_cross_attention_relative_position_embedding = T5RelativePositionEmbedding(
                        init_method=init_method_normal(embedding_init_method_std),
                        num_attention_heads=decoder_cfg.num_attention_heads,
                        relative_position_num_buckets=decoder_cfg.relative_attention_num_buckets,
                        relative_position_max_distance=decoder_cfg.relative_attention_max_distance,
                        bidirectional=True,
                        layer_type=LayerType.decoder,
                    )
                    self._decoder_cross_attention_relative_position_embedding_key = (
                        "decoder_cross_attention_relative_position_embedding"
                    )
                    if (
                        parallel_state.get_pipeline_model_parallel_rank()
                        != parallel_state.get_pipeline_model_parallel_split_rank()
                    ):
                        self.decoder_cross_attention_relative_position_embeddings_weight().data.fill_(0)
                        self.decoder_cross_attention_relative_position_embeddings_weight().shared = True

            elif self.decoder_cfg.get('position_embedding_type', 'learned_absolute') == 'alibi':
                self.decoder_relative_position_embedding = ALiBiRelativePositionEmbedding(
                    bidirectional=False,
                    num_attention_heads=decoder_cfg.num_attention_heads,
                    layer_type=LayerType.decoder,
                    num_attention_heads_alibi=None,
                    max_seq_len=max_position_embeddings,
                )
                self._decoder_relative_position_embedding_key = "decoder_alibi_position_embedding"
            elif self.decoder_cfg.get('position_embedding_type', 'learned_absolute') == 'kerple':
                self.decoder_relative_position_embedding = KERPLERelativePositionEmbedding(
                    bidirectional=False,
                    num_attention_heads=decoder_cfg.num_attention_heads,
                    layer_type=LayerType.decoder,
                    num_attention_heads_kerple=None,
                    max_seq_len=max_position_embeddings,
                )
                self._decoder_relative_position_embedding_key = "decoder_kerple_position_embedding"
            else:
                self.decoder_relative_position_embedding = None

            if decoder_cfg.get('use_flash_attention', False) and decoder_cfg.get(
                'position_embedding_type', 'learned_absolute'
            ) in ['relative', 'kerple']:
                raise ValueError('flash-attention not supported with relative or kerple at this point')

            decoder = get_decoder_model(
                config=config,
                arch=decoder_cfg.arch,
                hidden_size=decoder_cfg.hidden_size,
                ffn_hidden_size=decoder_cfg.ffn_hidden_size,
                num_layers=decoder_cfg.num_layers,
                num_attention_heads=decoder_cfg.num_attention_heads,
                apply_query_key_layer_scaling=decoder_cfg.get('apply_query_key_layer_scaling', True),
                kv_channels=decoder_kv_channels,
                init_method=init_method_normal(decoder_cfg.get('init_method_std', 0.02)),
                scaled_init_method=scaled_init_method_normal(
                    decoder_cfg.get('init_method_std', 0.02), decoder_cfg.num_layers
                ),
                decoder_attn_mask_type=AttnMaskType.causal,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=decoder_cfg.get('init_method_std', 0.02),
                hidden_dropout=decoder_cfg.get('hidden_dropout', 0.1),
                attention_dropout=decoder_cfg.get('attention_dropout', 0.1),
                ffn_dropout=decoder_cfg.get('ffn_dropout', 0.0),
                precision=precision,
                fp32_residual_connection=decoder_cfg.get('fp32_residual_connection', False),
                activations_checkpoint_method=decoder_cfg.get('activations_checkpoint_method', None),
                activations_checkpoint_num_layers=decoder_cfg.get('activations_checkpoint_num_layers', 1),
                activations_checkpoint_granularity=decoder_cfg.get('activations_checkpoint_granularity', None),
                layernorm_epsilon=decoder_cfg.get('layernorm_epsilon', 1e-5),
                bias_activation_fusion=decoder_cfg.get('bias_activation_fusion', True),
                bias_dropout_add_fusion=decoder_cfg.get('bias_dropout_add_fusion', True),
                masked_softmax_fusion=decoder_cfg.get('masked_softmax_fusion', True),
                persist_layer_norm=decoder_cfg.get('persist_layer_norm', True),
                openai_gelu=decoder_cfg.get('openai_gelu', False),
                onnx_safe=decoder_cfg.get('onnx_safe', False),
                hidden_steps=decoder_cfg.get('hidden_steps', -1),
                activation=decoder_cfg.get('activation', 'gelu'),
                bias=decoder_cfg.get('bias', True),
                normalization=decoder_cfg.get('normalization', 'layernorm'),
                transformer_block_type=decoder_cfg.get('transformer_block_type', 'pre_ln'),
                headscale=decoder_cfg.get('headscale', False),
                parent_model_type=ModelType.encoder_or_decoder,
                megatron_legacy=decoder_cfg.get('megatron_legacy', False),
                normalize_attention_scores=decoder_cfg.get('normalize_attention_scores', True),
                num_moe_experts=decoder_cfg.get('num_moe_experts', 1),
                moe_frequency=decoder_cfg.get('moe_frequency', 1),
                moe_dropout=decoder_cfg.get('moe_dropout', 0.0),
                position_embedding_type=decoder_cfg.get('position_embedding_type', 'learned_absolute'),
                use_flash_attention=decoder_cfg.get('use_flash_attention', False),
                layer_type=decoder_cfg.get('layer_type', LayerType.decoder),
            )

        hiddens_module = get_hiddens_module(hiddens_cfg, model_parallel_cfg=config)
        self.enc_dec_model = MegatronTransformerEncoderDecoderModule(
            config=config,
            encoder=encoder,
            decoder=decoder,
            hidden_steps=encoder_cfg.get('hidden_steps', -1),
            hiddens_module=hiddens_module,
        )
        self._enc_dec_model_key = "enc_dec_model"

        if self.share_token_embeddings:
            # This is only relevant for PP > 1.
            self.initialize_word_embeddings(
                init_method=init_method_normal(embedding_init_method_std),
                vocab_size=vocab_size,
                hidden_size=encoder_cfg.hidden_size,
            )

        if add_decoder and post_process:
            if share_decoder_tokens_head_embeddings:
                # parallel_output is True if TP > 1 (3b model)
                self.tokens_head = MegatronTokenLevelHead(
                    self.word_embeddings_weight().size(0), parallel_output, bias=tokens_head_bias
                )
            else:
                self.tokens_head = tensor_parallel.ColumnParallelLinear(
                    input_size=decoder_cfg.hidden_size,
                    output_size=vocab_size,
                    config=config,
                    bias=tokens_head_bias,
                    gather_output=not self.parallel_output,
                    init_method=init_method_normal(decoder_cfg.init_method_std),
                )

            self._tokens_head_key = 'tokens_head'
        self.set_accepted_adapter_types([PromptEncoderAdapterConfig._target_])

    def _validate_kv_channels(self, cfg):
        kv_channels = cfg.kv_channels
        if cfg.kv_channels is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads

        return kv_channels

    def _validate_enc_dec_hidden_size(self, encoder_cfg, decoder_cfg):
        if encoder_cfg.hidden_size != decoder_cfg.hidden_size:
            raise ValueError(
                f"Encoder and decoder hidden_size must be equal, but got encoder: {encoder_cfg.hidden_size} and decoder: {decoder_cfg.hidden_size}"
            )

    def _validate_perceiver_config(self, cfg):
        if (
            cfg.get("position_embedding_type", "learned_absolute") == "relative"
            and cfg.get("arch", "transformer") == "perceiver"
        ):
            raise ValueError("Perceivers with relative position embeddings are not supported")

    def _validate_config(self):
        encoder_kv_channels = self._validate_kv_channels(self.encoder_cfg)
        decoder_kv_channels = self._validate_kv_channels(self.decoder_cfg)
        self._validate_enc_dec_hidden_size(self.encoder_cfg, self.decoder_cfg)
        self._validate_perceiver_config(self.encoder_cfg)
        self._validate_perceiver_config(self.decoder_cfg)
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            assert (
                self.share_token_embeddings
            ), "Token embeddings must be shared when using pipeline model parallel size > 1"
            assert (
                self.share_decoder_tokens_head_embeddings
            ), "Decoder token embeddings and the outputlayer must be shared when using pipeline model parallel size > 1"
            assert (
                self.hiddens_cfg is None
            ), "Hiddens module must not be enabled when using pipeline model parallel size > 1"

        return encoder_kv_channels, decoder_kv_channels

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None

        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert (
                len(input_tensor) == 1
            ), 'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.enc_dec_model.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, 'input_tensor should only be length 1 for stage with only encoder'
            self.enc_dec_model.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.enc_dec_model.decoder.set_input_tensor(input_tensor[0])
                self.enc_dec_model.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.enc_dec_model.decoder.set_input_tensor(None)
                self.enc_dec_model.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')

    def forward(
        self,
        enc_input_ids=None,
        enc_attn_mask=None,
        dec_input_ids=None,
        dec_attn_mask=None,
        token_type_ids=None,
        labels=None,
        batch_data=None,  # additional data to be passed to hiddens module
        enc_output=None,  # Result of running the entire encoder
        enc_output_attn_mask=None,
        enc_input=None,  # Result of running encoder embedding only
        output_enc_hidden_only=False,
    ):
        """
        Return value is per token / per dimension (i.e., non collapsed loss value)
        """
        (
            encoder_self_attention_relative_position_bias,
            decoder_self_attention_relative_position_bias,
            decoder_cross_attention_relative_position_bias,
        ) = (None, None, None)

        if enc_input is not None and enc_output is not None:
            raise ValueError(
                """Both enc_input and enc_output are not None.
                You should only be passing one of them.
                enc_input is the result of the encoder embedding layer
                enc_output is the result of running the entire transformer encoder."""
            )

        # In order of precedence, we use enc_output, enc_input, and then enc_input_ids to determine the encoder sequence length.
        if enc_output is not None:
            # If enc_output is provided in `batch_for_pipeline`, we need to transpose it from [B x S x H] -> [S x B x H].
            enc_output = enc_output.transpose(0, 1)
            enc_seq_length = enc_output.size(0)
        elif enc_input is not None:
            # If enc_input is provided, we need to transpose it from [B x S x H] -> [S x B x H].
            enc_input = enc_input.transpose(0, 1)
            enc_seq_length = enc_input.size(0)
        # Only need to run encoder embedding and position ids if enc_input or enc_output is not provided.
        elif enc_input_ids is not None:
            enc_seq_length = enc_input_ids.size(1)
            if self.pre_process and self.add_encoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.encoder_relative_position_embedding is None:
                    enc_position_ids = build_position_ids(enc_input_ids)
                else:
                    enc_position_ids = None
                enc_input = self.encoder_embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
                if self.is_adapter_available():
                    _sq, _bs, _hs = enc_input.size()
                    ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
                    v = ptuning_adapter.virtual_tokens
                    if (
                        ptuning_adapter and self.adapter_cfg[AdapterName.PTUNING_ADAPTER]['enabled'] and _sq >= v
                    ):  # The sequence should be longer the v to insert virtual embeddings.
                        virtual_embeddings = ptuning_adapter(_bs)
                        enc_input = enc_input[
                            v:, :, :
                        ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                        enc_input = torch.concat([virtual_embeddings, enc_input], dim=0)
            else:
                enc_input = None
        else:
            # This should only happen with PP > 1 for enc-dec prompt learning models
            enc_seq_length = enc_attn_mask.size(1)

        if self.add_encoder and self.encoder_relative_position_embedding is not None:
            encoder_self_attention_relative_position_bias = self.encoder_relative_position_embedding(
                query_seq_length=enc_seq_length,
                key_seq_length=enc_seq_length,
            )

        if output_enc_hidden_only:
            # When pipeline parallel > 1 we need to make sure encoder exist (will be missing in decoder)
            if enc_output is None and self.enc_dec_model.encoder is not None:
                enc_output = self.enc_dec_model.encode(
                    enc_input=enc_input,
                    enc_attn_mask=enc_attn_mask,
                    enc_layer_past=None,
                    enc_get_key_value=False,
                    enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                    batch_data=batch_data,
                )
            else:
                enc_output = self.enc_dec_model.encoder_hidden_state

            return enc_output
        else:
            if enc_output_attn_mask is None:
                enc_output_attn_mask = enc_attn_mask

            if self.pre_process and self.add_decoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.decoder_relative_position_embedding is None:
                    dec_position_ids = build_position_ids(dec_input_ids)
                else:
                    dec_position_ids = None
                dec_input = self.decoder_embedding(dec_input_ids, dec_position_ids, token_type_ids=token_type_ids)
            else:
                # Note: This is when the decoder itself is split across PP ranks.
                dec_input = None

            if self.add_decoder and self.decoder_relative_position_embedding is not None:
                decoder_self_attention_relative_position_bias = self.decoder_relative_position_embedding(
                    query_seq_length=dec_input_ids.size(1), key_seq_length=dec_input_ids.size(1)
                )
                if not self.decoder_cfg.relative_position_bias_self_attention_only:
                    decoder_cross_attention_relative_position_bias = (
                        self.decoder_cross_attention_relative_position_embedding(
                            query_seq_length=dec_input_ids.size(1),
                            key_seq_length=enc_seq_length,
                        )
                    )
                else:
                    decoder_cross_attention_relative_position_bias = None

            output = self.enc_dec_model(
                enc_input=enc_input,
                enc_attn_mask=enc_attn_mask,
                dec_input=dec_input,
                dec_attn_mask=dec_attn_mask,
                enc_layer_past=None,
                enc_get_key_value=False,
                enc_output=enc_output,
                enc_output_attn_mask=enc_output_attn_mask,
                dec_layer_past=None,
                dec_get_key_value=False,
                enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                dec_self_attention_relative_position_bias=decoder_self_attention_relative_position_bias,
                dec_cross_attention_relative_position_bias=decoder_cross_attention_relative_position_bias,
                batch_data=batch_data,
            )

            if self.post_process and self.add_decoder:
                dec_output, enc_output = output  # [s, b, h], enc_output might be a dict if hiddens_module is used
                # project decoder output to vocabulary-size dimensions
                if self.share_decoder_tokens_head_embeddings:
                    token_logits = self.tokens_head(dec_output, self.word_embeddings_weight())
                else:
                    token_logits = self.tokens_head(dec_output)[0]

                if labels is not None:
                    # compute loss here
                    # [b, s] -> [s, b]
                    labels = labels.transpose(0, 1).contiguous()

                    # Set label smoothing to 0 if in eval mode.
                    label_smoothing = self.label_smoothing if self.training else 0.0

                    # tensor_parallel.vocab_parallel_cross_entropy performs log_softmax and return log p(x_i|z) per token i
                    if self.fp16_cross_entropy:
                        assert token_logits.dtype == torch.half
                        tokens_loss = vocab_parallel_cross_entropy(token_logits, labels, label_smoothing)
                    else:
                        tokens_loss = vocab_parallel_cross_entropy(token_logits.float(), labels, label_smoothing)

                    # [s, b] -> [b, s]
                    tokens_loss = tokens_loss.transpose(0, 1).contiguous()

                    # check if hiddens is used
                    if self.hiddens_cfg is not None:
                        loss_dict = self.enc_dec_model.hiddens_module.apply_loss_transforms(
                            outputs=enc_output,
                            batch_data=batch_data,
                        )
                        loss_dict["tokens_loss"] = tokens_loss
                        # We need to store default output in a known key, so that we can mimic default behaviour
                        loss_dict["output"] = tokens_loss
                        return loss_dict
                    else:
                        return tokens_loss
                else:
                    # else return token logits (and hiddens if needed)
                    # [s, b, h] -> [b, s, h]
                    token_logits = token_logits.transpose(0, 1).contiguous()
                    if self.hiddens_cfg is not None:
                        # return all hiddens and token logits
                        hiddens_dict = enc_output
                        hiddens_dict["token_logits"] = token_logits
                        # We need to store default output in a known key, so that we can mimic default behaviour
                        hiddens_dict["output"] = token_logits
                        return hiddens_dict
                    else:
                        return token_logits

            elif self.add_decoder and not self.add_encoder:
                decoder_output, _ = output
                return decoder_output
            else:
                encoder_output = output
                return encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}

        state_dict_[self._encoder_embedding_key] = self.encoder_embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._decoder_embedding_key] = self.decoder_embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._enc_dec_model_key] = self.enc_dec_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._tokens_head_key] = self.tokens_head.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        self.encoder_embedding.load_state_dict(state_dict[self._encoder_embedding_key], strict=strict)
        self.decoder_embedding.load_state_dict(state_dict[self._decoder_embedding_key], strict=strict)
        self.enc_dec_model.load_state_dict(state_dict[self._enc_dec_model_key], strict=strict)
        self.tokens_head.load_state_dict(state_dict[self._tokens_head_key], strict=strict)


class MegatronTokenLevelEncoderDecoderSpeechLLMModule(MegatronTokenLevelEncoderDecoderModule):
    def __init__(self, *args, **kwargs):
        super(MegatronTokenLevelEncoderDecoderSpeechLLMModule, self).__init__(*args, **kwargs)
        # Overridden in MegatronT5SpeechLMModel constructor
        self.seq_pattern = "parallel"
        self.speech_head_type = "token_level"
        self.attn_prior_scaledown_start_step = 10000
        self.attn_prior_end_step = 11000
        self.use_alignment_loss = False
        self.return_all_crossattention_probs = False
        self.logging_step = False
        self.num_cross_attention_heads = 12  # 12 for 220m T5, 16 for 11b T5
        self.enc_output_to_layers = None

    def get_decoder_embeddings(self, dec_input_ids, dec_position_ids, token_type_ids):
        if dec_input_ids.dim() <= 2:
            dec_input = self.decoder_embedding(dec_input_ids, dec_position_ids, token_type_ids=token_type_ids)
        else:
            dec_input = None
            for i in range(dec_input_ids.size()[1]):
                if i == 0:
                    # For the first channel (text + first layer of speech), use the decoder embedding layer
                    dec_input = self.decoder_embedding(
                        dec_input_ids[:, i, :], dec_position_ids, token_type_ids=token_type_ids
                    )
                else:
                    # For the rest of the channels (speech), use the speech embedding layer. No need for position, since already added in first layer.
                    current = self.speech_tokens_embeddings[i - 1](dec_input_ids[:, i, :]).permute(1, 0, 2)
                    # @pneekhara - Commenting the below because we always want to include all channels for speech.
                    # @pneekhara - include_channel_flag can become 0 when doing autoregressive inference and the first timestep is zeros
                    # For text inputs, only include 1st channel embeddings. Zero-out others.
                    # include_channel_flag = (torch.sum(dec_input_ids[:, i, :], dim=1) > 0).float()  # [B]
                    # current = current * include_channel_flag.unsqueeze(0).unsqueeze(2)
                    dec_input = dec_input + current

        return dec_input

    def forward(
        self,
        enc_input_ids=None,
        enc_attn_mask=None,
        dec_input_ids=None,
        dec_attn_mask=None,
        token_type_ids=None,
        labels=None,
        batch_data=None,  # additional data to be passed to hiddens module
        enc_output=None,  # Result of running the entire encoder
        enc_output_attn_mask=None,
        enc_input=None,  # Result of running encoder embedding only
        output_enc_hidden_only=False,
        speech_mask=None,
        cross_attention_prior=None,
        text_limits=None,
        global_step=None,
        set_inference_key_value_memory=False,
        decoder_max_sequence_len=None,
        encoder_max_sequence_len=None,
    ):
        """
        Return value is per token / per dimension (i.e., non collapsed loss value)
        """
        (
            encoder_self_attention_relative_position_bias,
            decoder_self_attention_relative_position_bias,
            decoder_cross_attention_relative_position_bias,
        ) = (None, None, None)

        if enc_input is not None and enc_output is not None:
            raise ValueError(
                """Both enc_input and enc_output are not None.
                You should only be passing one of them.
                enc_input is the result of the encoder embedding layer
                enc_output is the result of running the entire transformer encoder."""
            )

        # In order of precedence, we use enc_output, enc_input, and then enc_input_ids to determine the encoder sequence length.
        if enc_output is not None:
            # If enc_output is provided in `batch_for_pipeline`, we need to transpose it from [B x S x H] -> [S x B x H].
            if isinstance(enc_output, list):
                encoder_self_attention_relative_position_bias = [None for _ in enc_output]
                enc_output = [x.transpose(0, 1) for x in enc_output]
                enc_seq_length = [x.size(0) for x in enc_output]
            else:
                enc_output = enc_output.transpose(0, 1)
                enc_seq_length = enc_output.size(0)
        elif enc_input is not None:
            # If enc_input is provided, we need to transpose it from [B x S x H] -> [S x B x H].
            if isinstance(enc_input, list):
                encoder_self_attention_relative_position_bias = [None for _ in enc_input]
                enc_input = [x.transpose(0, 1) for x in enc_input]
                enc_seq_length = [x.size(0) for x in enc_input]
            else:
                enc_input = enc_input.transpose(0, 1)
                enc_seq_length = enc_input.size(0)
        # Only need to run encoder embedding and position ids if enc_input or enc_output is not provided.
        elif enc_input_ids is not None:
            enc_seq_length = enc_input_ids.size(1)
            if self.pre_process and self.add_encoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.encoder_relative_position_embedding is None:
                    enc_input_ids_p = enc_input_ids[:, 0, :] if enc_input_ids.dim() == 3 else enc_input_ids
                    enc_position_ids = build_position_ids(enc_input_ids_p)
                else:
                    enc_position_ids = None
                enc_input = self.encoder_embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
                if self.is_adapter_available():
                    _sq, _bs, _hs = enc_input.size()
                    ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
                    v = ptuning_adapter.virtual_tokens
                    if (
                        ptuning_adapter and _sq >= v
                    ):  # The sequence should be longer the v to insert virtual embeddings.
                        virtual_embeddings = ptuning_adapter(_bs)
                        enc_input = enc_input[
                            v:, :, :
                        ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                        enc_input = torch.concat([virtual_embeddings, enc_input], dim=0)
            else:
                enc_input = None
        else:
            # This should only happen with PP > 1 for enc-dec prompt learning models
            enc_seq_length = enc_attn_mask.size(1)

        if self.add_encoder and self.encoder_relative_position_embedding is not None:
            encoder_self_attention_relative_position_bias = self.encoder_relative_position_embedding(
                query_seq_length=enc_seq_length,
                key_seq_length=enc_seq_length,
            )

        if output_enc_hidden_only:
            # When pipeline parallel > 1 we need to make sure encoder exist (will be missing in decoder)
            # SpeechT5 should not go here for inference
            if enc_output is None and self.enc_dec_model.encoder is not None:
                enc_output = self.enc_dec_model.encode(
                    enc_input=enc_input,
                    enc_attn_mask=enc_attn_mask,
                    enc_layer_past=None,
                    enc_get_key_value=False,
                    enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                    batch_data=batch_data,
                )
            else:
                enc_output = self.enc_dec_model.encoder_hidden_state

            return enc_output
        else:
            if enc_output_attn_mask is None:
                enc_output_attn_mask = enc_attn_mask

            if self.pre_process and self.add_decoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.decoder_relative_position_embedding is None:
                    dec_input_ids_p = dec_input_ids[:, 0, :] if dec_input_ids.dim() == 3 else dec_input_ids
                    dec_position_ids = build_position_ids(dec_input_ids_p)
                else:
                    dec_position_ids = None
                dec_input = self.get_decoder_embeddings(dec_input_ids, dec_position_ids, token_type_ids)
                if not set_inference_key_value_memory and (decoder_max_sequence_len or encoder_max_sequence_len):
                    # In inference
                    # On step 0 when set_inference_key_value_memory is True, we need all inputs in case
                    # we are using decoder context
                    # Else on step >= 1, only need last input
                    logging.debug("Clipping dec_input and only keep the last input.")
                    dec_input = dec_input[-1, :, :].unsqueeze(0)  # shape (b, embed_dim)
            else:
                # Note: This is when the decoder itself is split across PP ranks.
                dec_input = None

            if self.add_decoder and self.decoder_relative_position_embedding is not None:
                decoder_self_attention_relative_position_bias = self.decoder_relative_position_embedding(
                    query_seq_length=dec_input_ids.size(1), key_seq_length=dec_input_ids.size(1)
                )
                if not self.decoder_cfg.relative_position_bias_self_attention_only:
                    decoder_cross_attention_relative_position_bias = (
                        self.decoder_cross_attention_relative_position_embedding(
                            query_seq_length=dec_input_ids.size(1),
                            key_seq_length=enc_seq_length,
                        )
                    )
                else:
                    decoder_cross_attention_relative_position_bias = None

            return_all_crossattention_probs = self.return_all_crossattention_probs
            single_encoder = False
            if not isinstance(cross_attention_prior, list):
                single_encoder = True
                cross_attention_prior = [cross_attention_prior]

            decoder_cross_attention_relative_position_bias = []
            for _cross_attention_prior in cross_attention_prior:
                _decoder_cross_attention_relative_position_bias = None
                if _cross_attention_prior is not None:
                    # cross_attention_prior shape [B, dec_len, enc_len]
                    # Repeat it to make it [B, 12, dec_len, enc_len]
                    attn_prior_end_step = self.attn_prior_end_step
                    attn_prior_scaledown_start_step = self.attn_prior_scaledown_start_step
                    num_attention_heads = self.num_cross_attention_heads
                    assert attn_prior_scaledown_start_step <= attn_prior_end_step
                    logging.debug(
                        f"attn_prior_scaledown_start_step: {attn_prior_scaledown_start_step}, attn_prior_scaledown_start_step: {attn_prior_end_step}"
                    )
                    if global_step >= attn_prior_end_step:
                        _decoder_cross_attention_relative_position_bias = None
                    elif global_step > attn_prior_scaledown_start_step and global_step < attn_prior_end_step:
                        total_annealing_steps = attn_prior_end_step - attn_prior_scaledown_start_step
                        curr_annealing_step = global_step - attn_prior_scaledown_start_step
                        curr_cross_attention_prior = _cross_attention_prior + (
                            (1.0 - _cross_attention_prior) * curr_annealing_step / total_annealing_steps
                        )
                        _decoder_cross_attention_relative_position_bias = curr_cross_attention_prior.unsqueeze(
                            1
                        ).repeat(1, num_attention_heads, 1, 1)
                        _decoder_cross_attention_relative_position_bias = torch.log(
                            _decoder_cross_attention_relative_position_bias + 1e-8
                        )
                    else:
                        _decoder_cross_attention_relative_position_bias = _cross_attention_prior.unsqueeze(1).repeat(
                            1, num_attention_heads, 1, 1
                        )
                        _decoder_cross_attention_relative_position_bias = torch.log(
                            _decoder_cross_attention_relative_position_bias + 1e-8
                        )
                decoder_cross_attention_relative_position_bias.append(_decoder_cross_attention_relative_position_bias)

            return_all_crossattention_probs = return_all_crossattention_probs or self.logging_step

            if single_encoder:
                decoder_cross_attention_relative_position_bias = decoder_cross_attention_relative_position_bias[0]

            output = self.enc_dec_model(
                enc_input=enc_input,
                enc_attn_mask=enc_attn_mask,
                dec_input=dec_input,
                dec_attn_mask=dec_attn_mask,
                enc_layer_past=None,
                enc_get_key_value=False,
                enc_output=enc_output,
                enc_output_attn_mask=enc_output_attn_mask,
                dec_layer_past=None,
                dec_get_key_value=False,
                enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                dec_self_attention_relative_position_bias=decoder_self_attention_relative_position_bias,
                dec_cross_attention_relative_position_bias=decoder_cross_attention_relative_position_bias,
                return_all_crossattention_probs=return_all_crossattention_probs,
                batch_data=batch_data,
                set_inference_key_value_memory=set_inference_key_value_memory,
                decoder_max_sequence_len=decoder_max_sequence_len,
                encoder_max_sequence_len=encoder_max_sequence_len,
                enc_output_to_layers=self.enc_output_to_layers,
            )

            alignment_loss = None
            if self.post_process and self.add_decoder:
                dec_output, enc_output = output  # [s, b, h]
                if return_all_crossattention_probs:
                    dec_output, attention_scores = dec_output
                    attention_probs = [
                        torch.softmax(attention_score, dim=-1)
                        for lidx, attention_score in enumerate(attention_scores)
                        if lidx in self.alignment_decoder_layerids
                    ]

                    if text_limits is not None and self.use_alignment_loss and hasattr(self, "forward_sum_loss"):
                        attention_scores_filtered = [
                            attention_scores[lidx] for lidx in self.alignment_decoder_layerids
                        ]
                        attention_scores_combined = torch.cat(attention_scores_filtered, dim=1)
                        text_start_idx = text_limits[0, 0].item()
                        assert torch.all(
                            text_limits[:, 0] == text_start_idx
                        )  # all texts should start at the same index
                        end_offset = self.alignment_text_end_offset
                        # align_every_n_head: eg if set to 2, will skip every other head
                        # if set to 12, will select 1 head from every layer
                        align_every_n_head = self.align_every_n_head
                        dec_start_idx = self.decoder_context_len + 1  # +1 to remove bos
                        attention_scores_sliced = attention_scores_combined[
                            :, ::align_every_n_head, dec_start_idx:, text_start_idx : -(2 + end_offset)
                        ]  # -2 to remove eos and pad
                        attention_logprobs = (
                            attention_scores_sliced  # not taking log_softmax, since we will do that in loss function
                        )
                        attention_logprobs = torch.mean(attention_logprobs, dim=1, keepdim=True)
                        dec_len = torch.sum(dec_attn_mask, dim=1) - dec_start_idx
                        enc_len = text_limits[:, 1] - text_limits[:, 0] - end_offset
                        alignment_loss = self.forward_sum_loss(
                            attn_logprob=attention_logprobs, in_lens=enc_len, out_lens=dec_len
                        )
                else:
                    attention_probs = None
                # project decoder output to vocabulary-size dimensions
                if self.share_decoder_tokens_head_embeddings:
                    first_layer_vocabsize = (
                        self.speech_offset + self.speech_codebook_size
                    )  # variables set in __init__ of speechlm model
                    token_logits = self.tokens_head(dec_output, self.word_embeddings_weight())  # s, b, vocab
                    if self.seq_pattern in ["parallel", "delay_parallel"]:
                        # For flat seq_pattern we need all the logits
                        token_logits = token_logits[:, :, :first_layer_vocabsize]
                    speech_layers = self.num_speech_codebooks - 1

                    # speech_logits_list will be used in loss calculation (parallel output)
                    speech_logits_list = []
                    if self.seq_pattern in ["parallel", "delay_parallel"] and torch.count_nonzero(speech_mask) > 0:
                        for i in range(speech_layers):
                            last_layer_logits = self.speech_tokens_heads[i](dec_output)[0]  # T, B, 1024
                            speech_logits_list.append(last_layer_logits)  # T, B, 1024
                else:
                    token_logits = self.tokens_head(dec_output)[0]  # T, B, WordEmbSize

                if labels is not None:
                    if labels.dim() == 2:
                        # [b, s] -> [s, b]
                        labels = labels.transpose(0, 1).contiguous()
                    elif labels.dim() == 3:
                        # [b, c, s] -> [c, s, b]
                        labels = labels.permute(1, 2, 0).contiguous()

                    # Set label smoothing to 0 if in eval mode.
                    label_smoothing = self.label_smoothing if self.training else 0.0

                    # tensor_parallel.vocab_parallel_cross_entropy performs log_softmax and return log p(x_i|z) per token i
                    if self.fp16_cross_entropy:
                        assert token_logits.dtype == torch.half
                        if labels.dim() == 3:
                            raise NotImplementedError("fp16_cross_entropy is not support for labels of dimension 3")
                        tokens_loss = vocab_parallel_cross_entropy(token_logits, labels, label_smoothing)
                    else:
                        if labels.dim() == 2:
                            tokens_loss = vocab_parallel_cross_entropy(token_logits.float(), labels, label_smoothing)
                        elif labels.dim() == 3:
                            if token_logits.size()[0] != labels[0, :, :].size()[0]:
                                raise Exception("TODO: add a permute")
                            tokens_loss = vocab_parallel_cross_entropy(
                                token_logits.float(), labels[0, :, :], label_smoothing
                            )
                            logging.debug(f"token_loss: {tokens_loss}")
                            logging.debug(f"token_loss: {torch.all(torch.isfinite(tokens_loss))}")
                            if (
                                self.seq_pattern in ["parallel", "delay_parallel"]
                                and torch.count_nonzero(speech_mask) > 0
                            ):
                                for i in range(speech_layers):
                                    if speech_logits_list[i].size()[0] != labels[i + 1, :, :].size()[0]:
                                        raise Exception("TODO: add a permute")
                                    curr_codebook_loss = (
                                        vocab_parallel_cross_entropy(
                                            speech_logits_list[i].float(), labels[i + 1, :, :], label_smoothing
                                        )
                                        * speech_mask.T
                                    )
                                    tokens_loss += curr_codebook_loss
                                    logging.debug(f"token_loss_{i}: {tokens_loss}")
                                    logging.debug(f"token_loss_{i}: {torch.all(torch.isfinite(tokens_loss))}")

                    # [s, b] -> [b, s]
                    tokens_loss = tokens_loss.transpose(0, 1).contiguous()

                    # check if hiddens is used
                    if self.hiddens_cfg is not None:
                        raise NotImplementedError("Not currently implemented for speechllm")
                    else:
                        return tokens_loss, [token_logits, speech_logits_list, attention_probs, alignment_loss]
                else:
                    # else return token logits (and hiddens if needed)
                    # [s, b, h] -> [b, s, h]
                    # If labels is None then we are in inference mode and we return the gathered logits
                    if self.parallel_output:
                        # Gather logits from tensor parallel if in parallel_output mode
                        token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                            token_logits
                        )  # T, B, 30208
                        for _i in range(len(speech_logits_list)):
                            speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                                speech_logits_list[_i]
                            )  # T, B, 1024

                    token_logits = token_logits.transpose(0, 1).contiguous()  # (B, T, 30208)
                    speech_logits = torch.stack(speech_logits_list, dim=-1)  # T, B, 1024, 7
                    speech_logits = speech_logits.transpose(0, 1).contiguous()  # (B, T, 1024, 7)

                    _si = self.speech_offset
                    _ei = _si + self.speech_codebook_size
                    first_layer_speech_logits = token_logits[:, :, _si:_ei].unsqueeze(-1)  # (b, s, 1023, 1)

                    all_speech_logits = torch.cat(
                        [first_layer_speech_logits, speech_logits], dim=-1
                    )  # (b, s, 1024, 8)

                    if self.hiddens_cfg is not None:
                        raise NotImplementedError("Not currently implemented for speechllm")
                    else:
                        # all_speech_logits: tensor, (b, s, 1024, 8), all layers of speech.
                        # token_logits: tensor, (b, s, vocab_size), text token logits.
                        # speech_logits: tensor, (b, s, 1024, 7), 1-7 layers of speech.
                        # attention_probs: tensor or None, (b, s, )
                        # enc_output: tensor, (virtual_token_len+context_token_len+question_token_len+extra_id_0+[SEP], b, )
                        return all_speech_logits, [token_logits, speech_logits, attention_probs, enc_output]

            elif self.add_decoder and not self.add_encoder:
                decoder_output, _ = output
                return decoder_output
            else:
                encoder_output = output
                return encoder_output

    def state_dict(self):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._encoder_embedding_key] = self.encoder_embedding.state_dict()
        state_dict_[self._decoder_embedding_key] = self.decoder_embedding.state_dict()
        state_dict_[self._enc_dec_model_key] = self.enc_dec_model.state_dict()
        state_dict_[self._tokens_head_key] = self.tokens_head.state_dict()
        if hasattr(self, "speech_tokens_heads"):
            state_dict_["speech_tokens_heads"] = self.speech_tokens_heads.state_dict()
        if hasattr(self, "speech_tokens_embeddings"):
            state_dict_["speech_tokens_embeddings"] = self.speech_tokens_embeddings.state_dict()
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        super().load_state_dict(state_dict, strict=strict)
        if hasattr(self, "speech_tokens_heads"):
            self.speech_tokens_heads.load_state_dict(state_dict["speech_tokens_heads"], strict=strict)
        if hasattr(self, "speech_tokens_embeddings"):
            self.speech_tokens_embeddings.load_state_dict(state_dict["speech_tokens_embeddings"], strict=strict)
