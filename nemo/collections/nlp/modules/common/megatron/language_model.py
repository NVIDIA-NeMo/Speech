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

"""Transformer based language model."""
from ast import Mod

import torch

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    PromptEncoderAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.position_embedding import (
    ALiBiRelativePositionEmbedding,
    KERPLERelativePositionEmbedding,
    RotaryEmbedding,
    SandwichRelativePositionEmbedding,
)
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts import utils_funcs
from nemo.core import adapter_mixins

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    LayerType = ApexGuardDefaults()

try:
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


def get_language_model(
    config: ModelParallelConfig,
    hidden_size,
    ffn_hidden_size,
    num_layers,
    max_position_embeddings,
    num_tokentypes,
    add_pooler,
    vocab_size,
    num_attention_heads,
    encoder_attn_mask_type,
    apply_query_key_layer_scaling=False,
    kv_channels=None,
    init_method=None,
    scaled_init_method=None,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
    init_method_std=0.02,
    megatron_amp_O2=False,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    ffn_dropout=0.0,
    precision=16,
    fp32_residual_connection=False,
    activations_checkpoint_method=None,
    activations_checkpoint_num_layers=1,
    normalization='layernorm',
    layernorm_epsilon=1e-5,
    bias_activation_fusion=True,
    masked_softmax_fusion=True,
    activation='gelu',
    headscale=False,
    transformer_block_type='pre_ln',
    normalize_attention_scores=True,
    position_embedding_type='learned_absolute',
    attention_type='multihead',
    share_embeddings_and_output_weights=True,
    rotary_percentage=1.0,
    multi_query_attention=False,
    bias_dropout_add_fusion=True,
    bias=True,
    persist_layer_norm=False,
    openai_gelu=False,
    onnx_safe=False,
    megatron_legacy=False,
    activations_checkpoint_granularity=None,
    activations_checkpoint_layers_per_pipeline=None,
    transformer_engine=False,
    fp8=False,
    fp8_e4m3=False,
    fp8_hybrid=False,
    fp8_margin=0,
    fp8_interval=1,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo='max',
    reduce_amax=True,
    use_emha=False,
    ub_tp_comm_overlap=False,
    use_flash_attention=False,
    seq_len_interpolation_factor=None,
    rotary_base=10000,
):
    """Build language model and return along with the key to save."""

    if kv_channels is None:
        assert (
            hidden_size % num_attention_heads == 0
        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
        kv_channels = hidden_size // num_attention_heads

    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        config=config,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        encoder_attn_mask_type=encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        kv_channels=kv_channels,
        ffn_hidden_size=ffn_hidden_size,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        megatron_amp_O2=megatron_amp_O2,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        precision=precision,
        fp32_residual_connection=fp32_residual_connection,
        activations_checkpoint_method=activations_checkpoint_method,
        activations_checkpoint_num_layers=activations_checkpoint_num_layers,
        normalization=normalization,
        layernorm_epsilon=layernorm_epsilon,
        bias_activation_fusion=bias_activation_fusion,
        bias_dropout_add_fusion=bias_dropout_add_fusion,
        bias=bias,
        rotary_percentage=rotary_percentage,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        masked_softmax_fusion=masked_softmax_fusion,
        activation=activation,
        headscale=headscale,
        transformer_block_type=transformer_block_type,
        normalize_attention_scores=normalize_attention_scores,
        position_embedding_type=position_embedding_type,
        multi_query_attention=multi_query_attention,
        persist_layer_norm=persist_layer_norm,
        openai_gelu=openai_gelu,
        onnx_safe=onnx_safe,
        megatron_legacy=megatron_legacy,
        activations_checkpoint_granularity=activations_checkpoint_granularity,
        activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
        transformer_engine=transformer_engine,
        fp8=fp8,
        fp8_e4m3=fp8_e4m3,
        fp8_hybrid=fp8_hybrid,
        fp8_margin=fp8_margin,
        fp8_interval=fp8_interval,
        fp8_amax_history_len=fp8_amax_history_len,
        fp8_amax_compute_algo=fp8_amax_compute_algo,
        reduce_amax=reduce_amax,
        use_emha=use_emha,
        ub_tp_comm_overlap=ub_tp_comm_overlap,
        use_flash_attention=use_flash_attention,
        seq_len_interpolation_factor=seq_len_interpolation_factor,
        rotary_base=rotary_base,
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method, sequence_parallel=False):
        super(Pooler, self).__init__()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.sequence_parallel = sequence_parallel

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h] prompt_embeddings
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.mappings.gather_from_sequence_parallel_region(hidden_states)

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        position_embedding_type: position embedding type determines whether we instantiate a learnable position embedding table.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        dtype=torch.float32,
        fp32_residual_connection=False,
        position_embedding_type='learned_absolute',
        transpose_batch_sequence=True,
    ):
        super(Embedding, self).__init__(config=config)

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.position_embedding_type = position_embedding_type
        self.transpose_batch_sequence = transpose_batch_sequence

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, init_method=self.init_method, config=config,
        )
        self._word_embeddings_key = 'word_embeddings'

        if self.position_embedding_type == 'learned_absolute':
            # Position embedding (serial).
            self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size, dtype=dtype)
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            self.init_method(self.position_embeddings.weight)

        if self.position_embedding_type == 'learned_parameters':
            # Position embedding (learn parameters directly).
            self.position_embeddings = torch.nn.Parameter(torch.empty(max_sequence_length, self.hidden_size))
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            self.init_method(self.position_embeddings)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size, dtype=dtype)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = fp32_residual_connection
        self.sequence_parallel = config.sequence_parallel

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.position_embedding_type == 'learned_absolute':
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == 'learned_absolute':
            assert position_ids is not None
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        elif self.position_embedding_type == 'learned_parameters':
            embeddings = words_embeddings + self.position_embeddings
        else:
            embeddings = words_embeddings
        if token_type_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(token_type_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        if self.transpose_batch_sequence:
            embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.mappings.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        if self.position_embedding_type == 'learned_absolute':
            state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(
                destination, prefix, keep_vars
            )
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        if self.position_embedding_type == 'learned_absolute':
            # Position embedding.
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print(
                    '***WARNING*** expected tokentype embeddings in the ' 'checkpoint but could not find it',
                    flush=True,
                )


class TransformerLanguageModel(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        encoder_attn_mask_type,
        vocab_size,
        max_position_embeddings,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_tokentypes,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        add_decoder=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        add_pooler=False,
        pre_process=True,
        post_process=True,
        megatron_amp_O2=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        normalization='layernorm',
        layernorm_epsilon=1e-5,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        bias=True,
        masked_softmax_fusion=True,
        activation='gelu',
        headscale=False,
        transformer_block_type='pre_ln',
        normalize_attention_scores=True,
        position_embedding_type='learned_absolute',
        rotary_percentage=1.0,
        multi_query_attention=False,
        share_embeddings_and_output_weights=True,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        megatron_legacy=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_layers_per_pipeline=None,
        transformer_engine=False,
        fp8=False,
        fp8_e4m3=False,
        fp8_hybrid=False,
        fp8_margin=0,
        fp8_interval=1,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo='max',
        reduce_amax=True,
        use_emha=False,
        ub_tp_comm_overlap=False,
        use_flash_attention=False,
        seq_len_interpolation_factor=None,
        rotary_base=10000,
    ):
        super(TransformerLanguageModel, self).__init__(
            config=config, share_token_embeddings=share_embeddings_and_output_weights
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.sequence_parallel = config.sequence_parallel
        self.dtype = utils_funcs.torch_dtype_from_precision(precision, megatron_amp_O2)
        if kv_channels is None:

            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(
                config=config,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_position_embeddings,
                init_method=self.init_method,
                num_tokentypes=self.num_tokentypes,
                embedding_dropout_prob=self.hidden_dropout,
                position_embedding_type=position_embedding_type,
                fp32_residual_connection=fp32_residual_connection,
                dtype=self.dtype,
            )
            self._embedding_key = 'embedding'

        if position_embedding_type == 'rope':
            rotary_dim = self.hidden_size // num_attention_heads if kv_channels is None else kv_channels
            assert 0 < rotary_percentage <= 1
            if rotary_percentage < 1:
                rotary_dim = int(rotary_dim * rotary_percentage)
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_dim,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                pretrained_max_position_embeddings=max_position_embeddings,
                rotary_base=rotary_base,
            )

        elif position_embedding_type == 'alibi':
            # TODO: If this is used for encoder-decodemax_position_embeddingsr model, implement proper logic and following
            # addition for decoder. Currently it is only used for decoder model only.
            # Encoder-decoder model, such as T5 is implemented in token_level_encoder_decoder.py
            self.encoder_relative_position_embedding = ALiBiRelativePositionEmbedding(
                bidirectional=encoder_attn_mask_type != AttnMaskType.causal,
                num_attention_heads=num_attention_heads,
                layer_type=LayerType.encoder,
                num_attention_heads_alibi=None,
                max_seq_len=max_position_embeddings,
            )

        elif position_embedding_type == 'kerple':
            # TODO: If this is used for encoder-decodemax_position_embeddingsr model, implement proper logic and following
            # addition for decoder. Currently it is only used for decoder model only.
            # Encoder-decoder model, such as T5 is implemented in token_level_encoder_decoder.py
            self.encoder_relative_position_embedding = KERPLERelativePositionEmbedding(
                bidirectional=encoder_attn_mask_type != AttnMaskType.causal,
                num_attention_heads=num_attention_heads,
                layer_type=LayerType.encoder,
                num_attention_heads_kerple=None,
                max_seq_len=max_position_embeddings,
            )
            assert use_flash_attention == False  # flash-attention not supported with kerple at this point

        elif position_embedding_type == 'sandwich':
            self.encoder_relative_position_embedding = SandwichRelativePositionEmbedding(
                bidirectional=encoder_attn_mask_type != AttnMaskType.causal,
                num_attention_heads=num_attention_heads,
                layer_type=LayerType.encoder,
                hidden_size=self.hidden_size // num_attention_heads if kv_channels is None else kv_channels,
                max_seq_len=max_position_embeddings,
            )

        # Transformer.
        self.encoder = ParallelTransformer(
            config=config,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            megatron_amp_O2=megatron_amp_O2,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            bias=bias,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            activation=activation,
            headscale=headscale,
            transformer_block_type=transformer_block_type,
            normalize_attention_scores=normalize_attention_scores,
            multi_query_attention=multi_query_attention,
            megatron_legacy=megatron_legacy,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            transformer_engine=transformer_engine,
            fp8=fp8,
            fp8_e4m3=fp8_e4m3,
            fp8_hybrid=fp8_hybrid,
            fp8_margin=fp8_margin,
            fp8_interval=fp8_interval,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            reduce_amax=reduce_amax,
            use_emha=use_emha,
            ub_tp_comm_overlap=ub_tp_comm_overlap,
            position_embedding_type=position_embedding_type,
            use_flash_attention=use_flash_attention,
        )
        self._encoder_key = 'encoder'

        # Decoder
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                config=config,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                init_method=self.init_method,
                output_layer_init_method=self.output_layer_init_method,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                ffn_hidden_size=ffn_hidden_size,
                pre_process=self.pre_process,
                post_process=self.post_process,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                normalization=normalization,
                layernorm_epsilon=layernorm_epsilon,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                megatron_amp_O2=megatron_amp_O2,
                bias_activation_fusion=bias_activation_fusion,
                bias_dropout_add_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                megatron_legacy=megatron_legacy,
                activations_checkpoint_granularity=activations_checkpoint_granularity,
                activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
                transformer_engine=transformer_engine,
                position_embedding_type=position_embedding_type,
                use_flash_attention=use_flash_attention,
            )
            self._decoder_key = 'decoder'

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method, sequence_parallel=self.sequence_parallel)
                self._pooler_key = 'pooler'

            if not self.share_embeddings_and_output_weights:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    self.hidden_size,
                    self.vocab_size,
                    config=config,
                    bias=False,  # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                    init_method=self.init_method,
                )
                self._output_layer_key = 'output_layer'
        self.set_accepted_adapter_types([PromptEncoderAdapterConfig._target_])

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        self.encoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        enc_input_ids,
        enc_position_ids,
        enc_attn_mask,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attn_mask=None,
        enc_dec_attn_mask=None,
        token_type_ids=None,
        layer_past=None,
        get_key_value=False,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
        encoder_input=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # Embeddings.
        if self.pre_process and encoder_input is None:

            encoder_input = self.embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
            if self.is_adapter_available():
                _sq, _bs, _hs = encoder_input.size()
                ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
                v = ptuning_adapter.virtual_tokens
                if ptuning_adapter and _sq >= v:  # The sequence should be longer the v to insert virtual embeddings.
                    virtual_embeddings = ptuning_adapter(_bs)
                    encoder_input = encoder_input[
                        v:, :, :
                    ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                    encoder_input = torch.concat([virtual_embeddings, encoder_input], dim=0)
        else:
            pass

        # enc_attn_mask: [1, 1, s, s]
        if inference_max_sequence_len is not None:
            enc_seq_length = inference_max_sequence_len
        elif self.encoder.input_tensor is not None:
            if self.sequence_parallel:
                enc_seq_length = (
                    self.encoder.input_tensor.size(0) * parallel_state.get_tensor_model_parallel_world_size()
                )
            else:
                enc_seq_length = self.encoder.input_tensor.size(0)
        else:
            if self.sequence_parallel:
                enc_seq_length = encoder_input.size(0) * parallel_state.get_tensor_model_parallel_world_size()
            else:
                enc_seq_length = encoder_input.size(0)

        rotary_pos_emb = None
        encoder_self_attention_relative_position_bias = None
        if self.position_embedding_type == 'rope':
            rotary_pos_emb = self.rotary_pos_emb(enc_seq_length)
        elif (
            self.position_embedding_type == 'alibi'
            or self.position_embedding_type == 'sandwich'
            or self.position_embedding_type == 'kerple'
        ):
            encoder_self_attention_relative_position_bias = self.encoder_relative_position_embedding(
                query_seq_length=enc_seq_length, key_seq_length=enc_seq_length,
            )
            # causal attention bias: [1, head, 1, k]
            # non-causal attention bias: [1, head, q, k]

        # encoder.
        if enc_hidden_states is None:
            encoder_output = self.encoder(
                encoder_input,
                enc_attn_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                rotary_pos_emb=(rotary_pos_emb, None, None)
                if rotary_pos_emb is not None
                else None,  # This assumes that this being used as a GPT/BERT model only (no cross-attention)
                self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
            )
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output, pooling_sequence_index)

        # output_enc_hidden_only refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden_only:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder Embedding
        dec_embedding_output = self.embedding(dec_input_ids, dec_position_ids)
        # decoder
        decoder_output = self.decoder(
            dec_embedding_output,
            dec_attn_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] = self.embedding.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] = self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars
                )
        if self.add_decoder:
            state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self._encoder_key in state_dict:
            state_dict_ = state_dict[self._encoder_key]

        # for backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if self._encoder_key + '.' in key:
                    state_dict_[key.split(self._encoder_key + '.')[1]] = state_dict[key]
                elif 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # for backward compatibility.
        state_dict_self_attention = {}
        for key in state_dict_.keys():
            if '.attention.' in key:
                state_dict_self_attention[key.replace(".attention.", ".self_attention.")] = state_dict_[key]
            else:
                state_dict_self_attention[key] = state_dict_[key]
        state_dict_ = state_dict_self_attention

        self.encoder.load_state_dict(state_dict_, strict=strict)

        if self.post_process:
            # pooler
            if self.add_pooler:
                assert 'pooler' in state_dict, 'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key], strict=strict)
            if not self.share_embeddings_and_output_weights:
                # import pdb; pdb.set_trace()
                assert (
                    self._output_layer_key in state_dict
                ), 'could not find data for output embedding layer in the checkpoint'
                self.output_layer.load_state_dict(state_dict[self._output_layer_key], strict=strict)

        # decoder
        if self.add_decoder:
            assert 'decoder' in state_dict, 'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)
