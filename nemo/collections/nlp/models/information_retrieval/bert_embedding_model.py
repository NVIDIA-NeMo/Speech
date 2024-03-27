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


import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nemo.collections.nlp.models.language_modeling.megatron.bert.bert_model import (
    MCoreBertModelWrapperWithPostLNSupport,
    NeMoBertModel,
)
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


class BertEmbeddingHead(nn.Module):
    """Performs mean pooling on the token embeddings.
    """

    def __init__(
        self, word_embedding_dimension: int, pooling_mode_mean_tokens: bool = True,
    ):
        super(BertEmbeddingHead, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_mean_tokens",
        ]
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor):

        token_embeddings = token_embeddings.permute(1, 0, 2)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        output_vector = sum_embeddings / sum_mask

        output_vector = F.normalize(output_vector, p=2, dim=1)

        return output_vector

    def __repr__(self):
        return "Pooling({}) and Normalize".format(self.get_config_dict())

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}


class MCoreBertEmbeddingModel(MCoreBertModelWrapperWithPostLNSupport):
    def __init__(self, *args, **kwargs):

        super(MCoreBertEmbeddingModel, self).__init__(*args, **kwargs)
        # Changing the default settings of the original Bert model to make it compatible with the embedding model.
        self.post_process = False
        self.binary_head = None
        self.lm_head = None
        self.output_layer = None
        self.encoder.final_layernorm = None
        self.encoder.post_process = False
        self.embedding_head = BertEmbeddingHead(
            word_embedding_dimension=self.config.hidden_size, pooling_mode_mean_tokens=True,
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
        hidden_states = super(MCoreBertEmbeddingModel, self).forward(
            input_ids, attention_mask, tokentype_ids, lm_labels, inference_params
        )
        embeddings_out = self.embedding_head(hidden_states, attention_mask)
        return embeddings_out


class NeMoBertEmbeddingModel(NeMoBertModel):
    """
    Bert Language model.
    Model returns [seq, batch, hidden] shape
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "NeMoBertModel will be deprecated mid 2024. Use MCoreBertEmbeddingModel instead.", DeprecationWarning
        )
        super().__init__(*args, **kwargs)
        self.embedding_head = BertEmbeddingHead(
            word_embedding_dimension=self.config.hidden_size, pooling_mode_mean_tokens=True,
        )

    def forward(
        self,
        bert_model_input,
        attention_mask,
        token_type_ids=None,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
    ):

        lm_output = super(NeMoBertEmbeddingModel, self).forward(
            bert_model_input, attention_mask, token_type_ids, lm_labels, checkpoint_activations_all_layers
        )
        embeddings_out = self.embedding_head(lm_output[0], attention_mask)
        return embeddings_out
