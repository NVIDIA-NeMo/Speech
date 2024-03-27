# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""BERT model."""

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import get_layer_norm
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
    erf_gelu,
    get_linear_layer,
    init_method_normal,
    openai_gelu,
    parallel_lm_logits,
    scaled_init_method_normal,
)

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.tensor_parallel.layers import set_tensor_model_parallel_attributes

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()

try:
    from megatron.core import InferenceParams, ModelParallelConfig, parallel_state, tensor_parallel
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    from megatron.core.models.bert.bert_lm_head import BertLMHead as MCoreBertLMHead
    from megatron.core.models.bert.bert_model import BertModel as MCoreBert
    from megatron.core.models.bert.pooler import Pooler
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.transformer.utils import get_linear_layer as mcore_get_linear_layer
    from megatron.core.utils import make_viewless_tensor

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False


def bert_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # HF Masking is equivalent to the one below
    # extended_attention_mask = (attention_mask.unsqueeze(1) * torch.ones_like(attention_mask).unsqueeze(2)).unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = extended_attention_mask < 0.5

    return extended_attention_mask


class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        mpu_vocab_size,
        hidden_size,
        init_method,
        layernorm_epsilon,
        parallel_output,
        use_openai_gelu,
        onnx_safe,
    ):

        super(BertLMHead, self).__init__(config=config)

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.parallel_output = parallel_output
        self.sequence_parallel = config.sequence_parallel

        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.layernorm = get_layer_norm(hidden_size, eps=layernorm_epsilon)
        self.gelu = torch.nn.functional.gelu
        if use_openai_gelu:
            self.gelu = openai_gelu
        elif onnx_safe:
            self.gelu = erf_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        async_tensor_model_parallel_allreduce = self.config.async_tensor_model_parallel_allreduce
        output = parallel_lm_logits(
            hidden_states,
            word_embeddings_weight,
            self.parallel_output,
            sequence_parallel=self.sequence_parallel,
            bias=self.bias,
            async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
        )
        return output


def post_language_model_processing(
    lm_output, pooled_output, lm_head, binary_head, lm_labels, logit_weights, fp16_lm_cross_entropy,
):
    # lm_logits: [s, b, vocab_size]
    lm_logits = lm_head(lm_output, logit_weights)

    binary_logits = None
    if binary_head is not None:
        # binary_logits: [s, b, 2] or [s, b, vocab_size] if binary_head is Identity
        binary_logits = binary_head(pooled_output)

    if lm_labels is None:
        return lm_logits, binary_logits
    else:
        # match shape of labels to lm_logits
        # lm_labels: [b, s] -> [s, b]
        lm_labels = lm_labels.transpose(0, 1).contiguous()
        if fp16_lm_cross_entropy:
            assert lm_logits.dtype == torch.half
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
        else:
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(), lm_labels)
        # lm_loss: [s, b]
        return lm_loss, binary_logits


@dataclass
class TransformerLayerSubmodulesWithPostLNSupport(TransformerLayerSubmodules):
    def __init__(self, post_att_layernorm, post_mlp_layernorm, **kwargs):
        super(TransformerLayerSubmodulesWithPostLNSupport, self).__init__(**kwargs)
        self.post_att_layernorm = post_att_layernorm
        self.post_mlp_layernorm = post_mlp_layernorm


class TransformerLayerWithPostLNSupport(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerLayerWithPostLNSupport, self).__init__(*args, **kwargs)
        ## [Module add: Post attention LN]
        self.post_att_layernorm = build_module(
            self.submodules_config.post_att_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        ## [Module add: Post MLP LN]
        self.post_mlp_layernorm = build_module(
            self.submodules_config.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Post-LN after Self Attention
        hidden_states = self.post_att_layernorm(hidden_states)

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Post-LN after MLP
        hidden_states = self.post_mlp_layernorm(hidden_states)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context


class TransformerBlockWithPostLNSupport(TransformerBlock):
    def __init__(self, transformer_block_type='post_ln', *args, **kwargs):

        super(TransformerBlockWithPostLNSupport, self).__init__(*args, **kwargs)
        self.transformer_block_type = transformer_block_type
        if self.transformer_block_type == 'post_ln':
            self.initial_layernorm = FusedLayerNorm(
                config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor
        if self.transformer_block_type == 'post_ln':
            hidden_states = self.initial_layernorm(hidden_states)
        return super(TransformerBlockWithPostLNSupport, self).forward(
            hidden_states, attention_mask, context, context_mask, rotary_pos_emb, inference_params, packed_seq_params
        )


'''
This class is used for working with HF Bert Checkpoints. These checkpoints
by default have post layer norm, while the vanilla mcore bert model does not support it.
'''


class MCoreBertModelWrapperWithPostLNSupport(MCoreBert):
    def __init__(self, transformer_block_type='pre-ln', add_pooler=True, *args, **kwargs):

        super(MCoreBertModelWrapperWithPostLNSupport, self).__init__(*args, **kwargs)
        self.add_pooler = add_pooler
        self.transformer_block_type = transformer_block_type

        # Transformer.
        self.encoder = TransformerBlockWithPostLNSupport(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            transformer_block_type=self.transformer_block_type,
        )

        if self.add_pooler:
            self.pooler = Pooler(
                self.config.hidden_size, self.config.init_method, self.config, self.config.sequence_parallel
            )

        # Output
        if self.post_process:
            # TODO: Make sure you are passing in the mpu_vocab_size properly
            self.lm_head = MCoreBertLMHead(
                self.config.hidden_size,
                self.config,
                self.parallel_output,
                self.vocab_size,
                self.pre_process,
                self.share_embeddings_and_output_weights,
            )

            self.output_layer = self.lm_head.output_layer

            self.binary_head = None
            if self.add_binary_head:
                # TODO: Shoudl switch this to TE ?
                self.binary_head = mcore_get_linear_layer(
                    self.config.hidden_size, 2, self.config.init_method, self.config.perform_initialization
                )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tokentype_ids: Tensor = None,
        lm_labels: Tensor = None,
        inference_params=None,
    ):
        """Forward function of BERT model

        Forward function of the BERT Model This function passes the input tensors
        through the embedding layer, and then the encoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        original_post_process = self.post_process

        # We set this to false since we just want to get the hidden states from the encoder
        self.post_process = False
        hidden_states = super().forward(input_ids, attention_mask, tokentype_ids, lm_labels, inference_params)
        self.post_process = original_post_process

        if not self.post_process:
            return hidden_states

        if self.add_pooler:
            pooled_output = self.pooler(hidden_states, 0)

        if self.return_embeddings:
            embeddings = torch.transpose(hidden_states, 0, 1)
            masks = torch.sum(attention_mask, dim=1)
            # Collect masked embeddings.
            output = torch.zeros(
                size=(embeddings.shape[0], embeddings.shape[2]),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                output[i, :] = torch.mean(embedding[1 : mask - 1], dim=0)
            return output

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits = self.lm_head(hidden_states=hidden_states, word_embeddings_weight=output_weight)

        binary_logits = None
        if self.binary_head is not None and self.add_pooler:
            binary_logits = self.binary_head(pooled_output)

        if lm_labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous(), binary_logits

        loss = self.compute_language_model_loss(lm_labels, logits)

        return loss, binary_logits


class NeMoBertModel(MegatronModule):
    """
    Bert Language model.
    Model returns [seq, batch, hidden] shape
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_lm_cross_entropy=False,
        hidden_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        layernorm_epsilon=1e-5,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        masked_softmax_fusion=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        add_binary_head=True,
        add_pooler=True,
        add_lm_head=True,
        megatron_legacy=False,
        sequence_parallel=False,
        position_embedding_type='learned_absolute',
    ):
        warnings.warn(
            "NeMoBertModel will be deprecated mid 2024. Use MCoreBertModelWrapperWithPostLNSupport instead.",
            DeprecationWarning,
        )
        super(NeMoBertModel, self).__init__(config=config)
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel = sequence_parallel
        self.add_lm_head = add_lm_head
        self.add_pooler = add_pooler

        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)
        if self.add_binary_head:
            assert self.add_pooler, "Binary head requires pooler."
        self.language_model, self._language_model_key = get_language_model(
            config=config,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            hidden_dropout=hidden_dropout,
            num_tokentypes=num_tokentypes,
            max_position_embeddings=max_position_embeddings,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            add_pooler=self.add_pooler,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            init_method_std=init_method_std,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            layernorm_epsilon=layernorm_epsilon,
            normalization=normalization,
            transformer_block_type=transformer_block_type,
            masked_softmax_fusion=masked_softmax_fusion,
            bias_activation_fusion=bias_gelu_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            megatron_legacy=megatron_legacy,
            position_embedding_type=position_embedding_type,
        )

        self.initialize_word_embeddings(
            init_method=init_method_normal(init_method_std), vocab_size=vocab_size, hidden_size=hidden_size
        )

        if self.post_process and self.add_lm_head:
            self.lm_head = BertLMHead(
                config,
                self.word_embeddings_weight().size(0),
                hidden_size,
                init_method,
                layernorm_epsilon,
                parallel_output,
                openai_gelu,
                onnx_safe,
            )
            self._lm_head_key = 'lm_head'
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(hidden_size, 2, init_method)
                self._binary_head_key = 'binary_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        bert_model_input,
        attention_mask,
        token_type_ids=None,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
    ):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            input_ids = bert_model_input
            position_ids = build_position_ids(input_ids)
        else:
            position_ids = None
            input_ids = None

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            token_type_ids=token_type_ids,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.post_process and self.add_binary_head and self.add_lm_head:
            lm_output, pooled_output = lm_output
        else:
            pooled_output = None

        if self.post_process and self.add_lm_head:
            return post_language_model_processing(
                lm_output,
                pooled_output,
                self.lm_head,
                self.binary_head,
                lm_labels,
                self.word_embeddings_weight(),
                self.fp16_lm_cross_entropy,
            )
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        if self.post_process and self.add_lm_head:
            state_dict_[self._lm_head_key] = self.lm_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )
        if self.post_process and self.add_binary_head and self.add_lm_head:
            state_dict_[self._binary_head_key] = self.binary_head.state_dict(destination, prefix, keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process and self.add_lm_head:
            state_dict_[self._word_embeddings_for_head_key] = self.word_embeddings.state_dict(
                destination, prefix, keep_vars
            )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            self.lm_head.load_state_dict(state_dict[self._lm_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(state_dict[self._word_embeddings_for_head_key], strict=strict)
