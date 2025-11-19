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

from typing import List, Union

import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.inference_request import InferenceRequest
from PIL.Image import Image


class VLMEngine(MCoreEngine):
    # pylint: disable=C0115,C0116
    def generate(
        self,
        prompts: List[str],
        images: List[Union[Image, List[Image]]] = None,
        common_inference_params: CommonInferenceParams = None,
    ) -> dict:
        # pylint: disable=C0115,C0116
        request_ids: List[str] = []

        if self.random_seed:
            torch.random.manual_seed(self.random_seed)

        for i in range(len(prompts)):
            prompt = prompts[i]
            image = images[i] if images is not None else None
            prompt_tokens, image_dict = self.text_generation_controller.tokenize_prompt(prompt, image)

            # Reuse encoder_prompt from scheduler to pass image
            request_id = self.scheduler.add_request(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                encoder_prompt=image_dict,
                inference_parameters=common_inference_params,
            )
            request_ids.append(request_id)

        self.run_engine()

        result: List[InferenceRequest] = [
            self.scheduler.completed_request_pool[request_id] for request_id in request_ids
        ]
        return result
