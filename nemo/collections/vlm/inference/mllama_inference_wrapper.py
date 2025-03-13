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

from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference_params import InferenceParams
from torch.utils.data import default_collate

from nemo.collections.vlm.mllama.model.utils import create_vision_mask_tensor


class MllamaInferenceWrapper(AbstractModelInferenceWrapper):
    """Constructor for the model inference wrapper

    The wrapper prepares the model for inference, provides the required input
    data, and runs the forward pass

    Args:
        model (MllamaModel): The Mllama model
        args (Namespace): The command line arguments that were passed
    """

    def __init__(self, model, inference_wrapper_config: InferenceWrapperConfig):
        super().__init__(model, inference_wrapper_config)

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        image_dict: List[Dict] = None,
    ):
        # pylint: disable=C0115,C0116
        max_num_concurrent_media = max(instance['pixel_values'].shape[0] for instance in image_dict)
        for instance in image_dict:
            pad_num_images = max_num_concurrent_media - instance['pixel_values'].shape[0]
            instance['pixel_values'] = F.pad(
                instance['pixel_values'], (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_num_images), 'constant', 0
            )
            instance['aspect_ratio_ids'] = F.pad(
                instance['aspect_ratio_ids'], (0, max(pad_num_images - 1, 0)), 'constant', 0
            )
            instance['num_tiles'] = F.pad(
                torch.tensor(instance['num_tiles']), (0, max(pad_num_images - 1, 0)), 'constant', 0
            )
        batch = default_collate(image_dict)

        batch_size = prompts_tokens.size(0)
        seq_length = prompts_tokens.size(1)
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=prompts_tokens.device)
            .unsqueeze(0)
            .expand_as(prompts_tokens)
        )

        # Clear xattn caches
        self.inference_params = InferenceParams(batch_size, seq_length)
        self.inference_params.xattn_caches = None
        self.inference_params.cross_attention_masks = None
        self.inference_params.full_text_row_masked_out_mask = None

        return {
            "prompts_tokens": prompts_tokens,
            "position_ids": position_ids,
            "pixel_values": batch['pixel_values'].cuda(non_blocking=True),
            "num_tiles": batch['num_tiles'],
            "aspect_ratio_ids": batch['aspect_ratio_ids'].cuda(non_blocking=True),
        }

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        # pylint: disable=C0115,C0116
        tokens2use = inference_input["prompts_tokens"][:, context_start_position:context_end_position]
        positions2use = inference_input["position_ids"][:, context_start_position:context_end_position]

        return {
            "prompts_tokens": tokens2use,
            "position_ids": positions2use,
            "pixel_values": inference_input['pixel_values'],
            "num_tiles": inference_input['num_tiles'],
            "aspect_ratio_ids": inference_input['aspect_ratio_ids'],
        }

    def forward_pass_without_pipeline_parallel(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used  in the case of models without
        any parallelism or only tensor parallelism.

        Args:
            inference_input (List): A list containg the inputs for the vlm
                model [tokens, position ids]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        tokens2use = inference_input["prompts_tokens"]
        batch_masks = [create_vision_mask_tensor(tokens2use[0], 128256)] * tokens2use.size(0)
        logits = self.model(
            batch_images=inference_input["pixel_values"],
            batch_masks=batch_masks,
            num_chunks=inference_input["num_tiles"],
            aspect_ratio_ids=inference_input["aspect_ratio_ids"],
            tokens=tokens2use,
            position_ids=inference_input["position_ids"],
            xattn_caches=self.inference_params.xattn_caches,
            cross_attention_masks=self.inference_params.cross_attention_masks,
            full_text_row_masked_out_mask=self.inference_params.full_text_row_masked_out_mask,
            inference_params=self.inference_params,
        )
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        self.inference_params.sequence_len_offset += tokens2use.size(1)

        return logits
