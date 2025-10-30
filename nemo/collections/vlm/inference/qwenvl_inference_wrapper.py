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


class QwenVLInferenceWrapper(AbstractModelInferenceWrapper):
    """Constructor for the model inference wrapper

    The wrapper prepares the model for inference, provides the required input
    data, and runs the forward pass

    Args:
        model (Qwen2VLModel): The Qwen2VL model
        inference_wrapper_config (InferenceWrapperConfig): the config of inference wrapper
    """

    def __init__(self, model, inference_wrapper_config: InferenceWrapperConfig):
        super().__init__(model, inference_wrapper_config)

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        image_dict: List[Dict] = None,
    ):
        # pylint: disable=C0115,C0116
        batch_size = prompts_tokens.size(0)
        seq_length = prompts_tokens.size(1)

        self.inference_params = InferenceParams(batch_size, seq_length)

        return {
            "input_ids": prompts_tokens,
            "pixel_values": image_dict[0]['pixel_values'].cuda(non_blocking=True),
            "image_grid_thw": image_dict[0]['image_grid_thw'].cuda(non_blocking=True),
        }

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        # pylint: disable=C0115,C0116
        tokens2use = inference_input["input_ids"][:, :context_end_position]

        return {
            "input_ids": tokens2use,
            "pixel_values": inference_input['pixel_values'],
            "image_grid_thw": inference_input['image_grid_thw'],
        }

    def forward_pass_without_pipeline_parallel(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used in the case of models without
        any parallelism or only tensor parallelism.

        Args:
            inference_input (Dict): A dictionary containing the inputs for the qwen
                model [input_ids, position_ids, pixel_values, image_grid_thw]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        # TODO: add kv cache support
        logits = self.model(attention_mask=None, **inference_input)

        return logits
