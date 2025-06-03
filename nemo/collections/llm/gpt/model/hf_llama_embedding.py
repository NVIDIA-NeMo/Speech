# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification, LlamaModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


def pool(last_hidden_states: Tensor, attention_mask: Tensor, pool_type: str) -> Tensor:
    """Pooling on last_hidden_states without pad tokens."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weighted_avg":
        emb = last_hidden.sum(dim=1)
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


class LlamaBidirectionalConfig(LlamaConfig):
    """LLamaBidirectionalConfig for LlamaBidirectionalModel."""

    model_type = "llama_bidirec"

    def __init__(
        self,
        pooling="avg",
        temperature=1.0,
        **kwargs,
    ):
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(
            **kwargs,
        )


class LlamaBidirectionalModel(LlamaModel):
    """LlamaBidirectionalModel.
    Attention has been adjusted to bidirectional.
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # Generates bi-directional attention.
        causal_mask = _prepare_4d_attention_mask(attention_mask, input_tensor.dtype)
        return causal_mask


class LlamaBidirectionalForSequenceClassification(LlamaForSequenceClassification):
    """The LLaMa Model transformer with a sequence classification head on top (linear layer)."""

    config_class = LlamaBidirectionalConfig

    def __init__(self, config):
        super().__init__(config)
        # Releasing the parameters of LlamaModel
        # created by parent LlamaForSequenceClassification
        del self.model

        self.model = LlamaBidirectionalModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        pooled_hidden_states = pool(
            last_hidden_states=hidden_states,
            attention_mask=attention_mask,
            pool_type=self.config.pooling,
        )

        pooled_logits = self.score(pooled_hidden_states)
        pooled_logits = pooled_logits / self.config.temperature

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class LlamaBidirectionalHFAdapter(torch.nn.Module):
    """Wraps a Text embedding model with pooling and normalization."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalize: bool,
        pooling_module: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.pooling_module = pooling_module

    @property
    def device(self) -> torch.device:
        """Returns the device"""

        return self.model.device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        dimensions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference for the adapted Llama model"""

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        outputs = self.model(**inputs)
        hidden_states = outputs["last_hidden_state"].to(torch.float32)
        embeddings = self.pooling_module(hidden_states, inputs["attention_mask"])

        if dimensions is not None:
            if not torch.all(dimensions > 0):
                raise ValueError("Dimensions must be positive")

            fill_value = torch.tensor(float("-inf"), dtype=embeddings.dtype, device=embeddings.device)

            clipped_dimensions = torch.clamp(dimensions, max=int(embeddings.shape[1]))

            embeddings = embeddings.masked_fill(
                torch.arange(embeddings.shape[1], device=embeddings.device) >= clipped_dimensions.unsqueeze(-1),
                fill_value,
            )[:, : dimensions.max()]

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class Pooling(torch.nn.Module):
    """Pooling layer for the adapter."""

    def __init__(self, pooling_mode: str):
        super().__init__()
        self.pooling_mode = pooling_mode

    def forward(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward function of the Pooling layer."""

        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        pool_type = self.pooling_mode
        if pool_type == "avg":
            epsilon = 1e-9  # A small value to avoid division by zero
            emb = last_hidden.sum(dim=1) / (attention_mask.sum(dim=1)[..., None] + epsilon)
        elif pool_type == "cls":  # tokenizer padding right
            emb = last_hidden[:, 0]
        elif pool_type == "cls__left":  # tokenizer padding left
            seq_idxs = (1 - attention_mask).sum(dim=1).to(dtype=torch.long)
            batch_size = last_hidden.shape[0]
            batch_idxs = torch.arange(batch_size, device=last_hidden.device)
            emb = last_hidden[batch_idxs, seq_idxs]
        elif pool_type == "last":  # tokenizer padding left
            emb = last_hidden[:, -1]
        elif pool_type == "last__right":  # tokenizer padding right
            sequence_lengths = (attention_mask.sum(dim=1) - 1).to(dtype=torch.long)
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        return emb


def get_llama_bidirectional_hf_model(
    model_name_or_path: Union[str, os.PathLike[str]],
    normalize: bool,
    pooling_mode: Optional[Literal["avg", "cls", "last"]] = None,
    torch_dtype: Optional[Union[torch.dtype, str]] = None,
    trust_remote_code: bool = False,
):
    """Returns the adapter for the Llama bidirectional HF model."""

    # check that the tokenizer matches the requirements of the pooling mode
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    pooling_mode = pooling_mode or "avg"
    if pooling_mode == "last" and tokenizer.padding_side == "right":
        pooling_mode = "last__right"  # type: ignore
    if pooling_mode == "cls" and tokenizer.padding_side == "left":
        pooling_mode = "cls__left"  # type: ignore

    # load the model
    model = AutoModel.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
    ).eval()

    # configure pooling
    pooling_module = Pooling(pooling_mode=pooling_mode)

    # NV-Embed-v1 model has seperate embedding model and a built-in pooling module
    if (
        model.__class__.__name__ == "NVEmbedModel"
        and hasattr(model, "latent_attention_model")
        and hasattr(model, "embedding_model")
    ):
        pooling_module = model.latent_attention_model
        model = model.embedding_model

    adapted_model = LlamaBidirectionalHFAdapter(model=model, normalize=normalize, pooling_module=pooling_module)
    return adapted_model, tokenizer
