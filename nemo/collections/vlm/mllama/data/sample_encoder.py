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

import re
from dataclasses import field
from typing import Dict

import torch
from megatron.energon import VQASample

from nemo.collections.multimodal.data.energon.config import ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import VQASampleEncoder
from nemo.collections.vlm.mllama.model.utils import create_vision_mask_tensor
from nemo.utils import logging


class LlamaImageTextSample(ImageTextSample):
    """Llama Image Text Sample"""

    vision_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    aspect_ratio_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    aspect_ratio_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    num_tiles: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))


class Llama3SampleEncoder(VQASampleEncoder):
    """MLlama Sample Encoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the VQASampleEncoder.

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.conversation_template_config = multimodal_sample_config.conversation_template_config

    def process_image(self, image) -> Dict[str, torch.Tensor]:
        image_dict = self.image_processor.preprocess(image, return_tensors='pt')
        return image_dict

    def apply_prompt_template(self, input_text: VQASample, use_plain=False):
        if self.conversation_template_config.chat_template:
            self.tokenizer.chat_template = self.conversation_template_config.chat_template
        elif self.tokenizer.chat_template is None:
            raise ValueError(
                "Both tokenizer and conversation template does not have chat template defined. Refer to "
                "https://huggingface.co/docs/transformers/main/en/chat_templating "
            )
        logging.debug(f"apply_conversation_template context {input_text.context} answer {input_text.answers}")

        messages = []
        if self.conversation_template_config.system:
            messages.append(
                {'role': 'system', 'content': [{'type': 'text', 'text': self.conversation_template_config.system}]}
            )

        if isinstance(input_text.context, list) and isinstance(input_text.answers, list):
            # Ensure both lists are the same length or adjust based on your specific needs
            min_length = min(len(input_text.context), len(input_text.answers))
            for i in range(min_length):
                messages.append(
                    {
                        'role': self.conversation_template_config.roles[0],
                        'content': [{'type': 'text', 'text': input_text.context[i]}],
                    }
                )
                messages.append(
                    {
                        'role': self.conversation_template_config.roles[1],
                        'content': [{'type': 'text', 'text': input_text.answers[i]}],
                    }
                )
        elif isinstance(input_text.context, str) and isinstance(input_text.answers, str):
            # Handle single context and answer as strings
            messages.append(
                {
                    'role': self.conversation_template_config.roles[0],
                    'content': [{'type': 'text', 'text': input_text.context}],
                }
            )
            messages.append(
                {
                    'role': self.conversation_template_config.roles[1],
                    'content': [{'type': 'text', 'text': input_text.answers}],
                }
            )
        else:
            raise ValueError(
                "VQA Sample context/answers should either be a List[str] or str. Other types not supported"
            )

        templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        logging.debug(f"apply prompt template templated_prompt {templated_prompt}")
        return templated_prompt

    def tokenize(self, prompt: str) -> torch.Tensor:
        regex_pattern = '(' + '|'.join(re.escape(token) for token in [self.image_token.token_str]) + ')'
        chunks = re.split(regex_pattern, prompt)
        # Tokenize each chunk and replace special tokens with their indices
        tokenized_chunks = []
        for chunk in chunks:
            if chunk == self.image_token.token_str:
                tokenized_chunks.append(self.image_token.token_id)
            elif len(chunk) > 0:
                tokenized_chunks.extend(self.tokenizer(chunk, add_special_tokens=False).input_ids)

        return torch.tensor(tokenized_chunks, dtype=torch.long)

    def encode(self, input_sample: VQASample, output_sample: LlamaImageTextSample):
        conversation_prompt = self.apply_prompt_template(input_sample)
        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        tokens = self.tokenize(conversation_prompt)
        labels = self.compute_labels(tokens, input_sample)

        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"[Energon] task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"[Energon] task encoder encode_sample labels {labels}")
        loss_mask = self.compute_loss_mask(labels)
        vision_mask = create_vision_mask_tensor(tokens=tokens, vision_token_id=self.image_token.token_id)
        processed_image_dict = self.process_image(input_sample.image)
        output_sample.__key__ = input_sample.__key__
        output_sample.images = processed_image_dict['pixel_values'][0]
        output_sample.aspect_ratio_ids = processed_image_dict['aspect_ratio_ids'][0]
        output_sample.aspect_ratio_mask = processed_image_dict['aspect_ratio_mask'][0]
        output_sample.num_tiles = processed_image_dict['num_tiles'][0]
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.vision_mask = vision_mask
        return output_sample

    def process_answer_str(self, answer, stop_str):
        return answer + ("" if stop_str is None else stop_str)
