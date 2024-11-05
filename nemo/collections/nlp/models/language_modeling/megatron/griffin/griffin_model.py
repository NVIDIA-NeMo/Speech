# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import math
import torch
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from megatron.core import tensor_parallel
    from megatron.core.jit import jit_fuser
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    from torch import Tensor, nn

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False

from nemo.collections.nlp.models.language_modeling.megatron.griffin.griffin_block import GriffinStack


class GriffinModel(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int = 256000,
        logits_soft_cap: float = 30.0,
        position_embedding_type: str = 'rope',
        max_sequence_length: int = 1024,
        rotary_percent: float = 0.5,
        rotary_base: int = 10000,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = True,
    ):

        super().__init__(config)
        self.config = config
        self.vocab_size = vocab_size
        self.logits_soft_cap = logits_soft_cap
        self.position_embedding_type = position_embedding_type
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        if pre_process:
            self.embedding = LanguageModelEmbedding(
                config,
                vocab_size=self.vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=None,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=None,
                rotary_base=rotary_base,
            )

        self.decoder = GriffinStack(self.config)

        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=None,
                grad_output_buffer=None,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the emedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre processing it returns the input embeddings weight while during post processing it returns the final output layers weight
        """
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def griffin_position_ids(self, token_ids):
        # Create position ids
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        return position_ids

    def embedding_forward(self, input_ids):

        position_ids = self.griffin_position_ids(input_ids)
        embeddings = self.embedding(input_ids, position_ids)
        embeddings = embeddings * torch.tensor(math.sqrt(self.config.hidden_size)).type_as(embeddings)

        return embeddings

    @jit_fuser
    def _embedding_decode_(self, logits, transpose):
        logits = nn.functional.tanh(logits / self.logits_soft_cap) * self.logits_soft_cap
        if transpose:
            logits = logits.transpose(0, 1)
        return logits.contiguous()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        **extra_arg,
    ):
        if input_ids is None:
            input_ids = self.input_tensor

        hidden_states = self.embedding_forward(input_ids)

        rotary_pos_emb = None
        self.decoder.input_tensor = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                None, self.decoder, hidden_states, self.config, None
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        hidden_states = self.decoder(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)
        logits = self._embedding_decode_(logits, labels is None)

        if labels is None:
            # [b s h]
            return logits

        loss = self.compute_language_model_loss(labels, logits)

        return loss
