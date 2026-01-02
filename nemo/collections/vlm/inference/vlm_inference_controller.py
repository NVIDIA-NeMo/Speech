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

from typing import OrderedDict

import torch

from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)


class TokenizerWrapper:
    # pylint: disable=C0115,C0116
    def __init__(self, tokenizer):
        self.eod = tokenizer.eos_token_id
        self.vocab_size = None
        self._tokenizer = tokenizer

    def detokenize(self, tokens):
        # pylint: disable=C0115,C0116
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

    def tokenize(self, prompt):
        # pylint: disable=C0115,C0116
        return self._tokenizer.encode(prompt, add_special_tokens=False)


class VLMTextGenerationController(SimpleTextGenerationController):
    # pylint: disable=C0115,C0116
    def __init__(self, inference_wrapped_model, tokenizer, image_processor):
        super().__init__(inference_wrapped_model, TokenizerWrapper(tokenizer))
        self.image_processor = image_processor

    def tokenize_prompt(self, prompt: str, image):
        # pylint: disable=C0115,C0116
        tokens = self.tokenizer.tokenize(prompt)
        if image is None:
            image_dict = dict(
                pixel_values=torch.zeros(
                    1, 4, 3, self.image_processor.size['height'], self.image_processor.size['width']
                ),
                aspect_ratio_ids=torch.tensor([0], dtype=torch.long),
                num_tiles=[0],
            )
        else:
            image_dict = self.image_processor.preprocess(image, return_tensors='pt')
            image_dict = {
                k: v[0] for k, v in image_dict.items() if k in ["pixel_values", "aspect_ratio_ids", "num_tiles"]
            }
        return tokens, image_dict

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[str, InferenceRequest],
        use_attention_mask: bool = False,
    ):
        """Preparing input data for inference, using respective wrapper's prep_inference_input method

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests
            use_attention_mask (bool): Whether to use an attention mask. Should be set to True only
                when exclusively doing prefill (no decode) with variable prompt lengths.
                Currently unused and added to match an expected interface in Mcore.
        """
        images = list(map(lambda request: request.encoder_prompt, active_requests.values()))

        return self.inference_wrapped_model.prep_inference_input(
            prompts_tokens=prompts_tokens,
            image_dict=images,
        )


class QwenVLTextGenerationController(VLMTextGenerationController):
    """Text generation controller for QwenVL model"""

    def __init__(self, inference_wrapped_model, tokenizer, image_processor, processor):
        super().__init__(inference_wrapped_model, tokenizer, image_processor)

        class QwenVLTokenizer(TokenizerWrapper):
            # pylint: disable=C0115,C0116
            def detokenize(self, tokens):
                new_tokens = []
                for token in tokens:
                    if token == 151652:
                        new_tokens.append(token)
                        new_tokens.append(151655)
                    elif token != -200:
                        new_tokens.append(token)
                return self._tokenizer.decode(new_tokens, skip_special_tokens=False)

        self.tokenizer = QwenVLTokenizer(tokenizer)
        self.processor = processor

    def tokenize_prompt(self, prompt: str, image):
        """Tokenize prompt and process image following the VLM controller API"""
        inputs = self.processor(
            text=[prompt],
            images=image,
            padding=True,
            return_tensors="pt",
        )

        tokens = inputs['input_ids'][0]
        tokens[tokens == 151655] = -200

        image_dict = {
            'pixel_values': inputs['pixel_values'],
            'image_grid_thw': inputs['image_grid_thw'],
        }
        return tokens.tolist(), image_dict
