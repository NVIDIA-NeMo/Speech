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

import torch

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
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel

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


class BertModel(MegatronModule):
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
        megatron_amp_O2=False,
        hidden_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        layernorm_epsilon=1e-5,
        masked_softmax_fusion=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        add_binary_head=True,
        megatron_legacy=False,
        sequence_parallel=False,
        position_embedding_type='learned_absolute',
    ):
        super(BertModel, self).__init__(config=config)
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel = sequence_parallel

        init_method = init_method_normal(init_method_std)
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

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
            add_pooler=self.add_binary_head,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            init_method_std=init_method_std,
            megatron_amp_O2=megatron_amp_O2,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            layernorm_epsilon=layernorm_epsilon,
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

        if self.post_process:
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

        if self.post_process and self.add_binary_head:
            lm_output, pooled_output = lm_output
        else:
            pooled_output = None

        if self.post_process:
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
        if self.post_process:
            state_dict_[self._lm_head_key] = self.lm_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )
        if self.post_process and self.add_binary_head:
            state_dict_[self._binary_head_key] = self.binary_head.state_dict(destination, prefix, keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
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
