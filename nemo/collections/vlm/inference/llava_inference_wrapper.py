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

from typing import Any, Dict, List

import torch
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference_params import InferenceParams
from torch.utils.data import default_collate


class LlavaInferenceWrapper(AbstractModelInferenceWrapper):
    """Constructor for the model inference wrapper

    The wrapper prepares the model for inference, provides the required input
    data, and runs the forward pass

    Args:
        model (NevaModel): The Neva model
        inference_wrapper_config (InferenceWrapperConfig): the config of inference wrapper
    """

    def __init__(self, model, inference_wrapper_config: InferenceWrapperConfig):
        super().__init__(model, inference_wrapper_config)
        self._img_seq_len = self.model.module._img_seq_len

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        image_dict: List[Dict] = None,
    ):
        # pylint: disable=C0115,C0116
        media = default_collate(image_dict)['pixel_values'].cuda(non_blocking=True)
        media = media.reshape(media.size(0), 3, 336, 336)

        batch_size = prompts_tokens.size(0)
        seq_length = prompts_tokens.size(1)
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=prompts_tokens.device)
            .unsqueeze(0)
            .expand_as(prompts_tokens)
        )

        self.inference_params = InferenceParams(batch_size, seq_length + self._img_seq_len)

        return {
            "input_ids": prompts_tokens,
            "position_ids": position_ids,
            "images": media,
        }

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        # pylint: disable=C0115,C0116
        tokens2use = inference_input["input_ids"][:, context_start_position:context_end_position]
        positions2use = inference_input["position_ids"][:, context_start_position:context_end_position]
        self.img_token_offset = (context_start_position == 0) * (self._img_seq_len - 1)

        return {
            "input_ids": tokens2use,
            "position_ids": positions2use,
            "images": inference_input['images'],
        }

    def forward_pass_without_pipeline_parallel(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used  in the case of models without
        any parallelism or only tensor parallelism.

        Args:
            inference_input (List): A list containg the inputs for the neva
                model [input ids, position ids, media]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        logits = self.model(
            attention_mask=None, inference_params=self.inference_params, runtime_gather_output=True, **inference_input
        )
        self.inference_params.sequence_len_offset += inference_input["input_ids"].size(1) + self.img_token_offset

        return logits
