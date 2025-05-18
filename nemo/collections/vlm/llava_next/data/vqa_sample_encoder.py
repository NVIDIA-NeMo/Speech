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
import torch
from megatron.energon import VQASample

from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import VQASampleEncoder
from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextSample
from nemo.collections.vlm.llava_next.model.utils import get_number_of_features
from nemo.utils import logging


class LlavaNextSampleEncoder(VQASampleEncoder):
    """LlavaNextSampleEncoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the LlavaNextSampleEncoder, inherited from VQASampleEncoder for multimodal samples
        focused on VQA-style data to support LLaVANeXT

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        from transformers import LlavaNextConfig

        self.hf_config = LlavaNextConfig()

    def process_image(self, image):
        """
        Process and prepare an image sample for encoding.

        This method preprocesses the image using the HF image_processor, converting it to
        a tensor.

        Parameters:
        image: The input image to be processed.

        Returns:
        torch.Tensor: The processed image tensor.
        """
        image_array = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image_array

    def encode(self, input_sample: VQASample, output_sample: LlavaNextTextSample):
        """
        Encode a single sample into a format suitable for model input.

        This method prepares the conversation prompt, tokenizes it, and processes
        the associated image. It fills the output sample with tokens, labels, loss mask,
        and other required fields for multimodal processing.

        Parameters:
        input_sample (VQASample): The input VQA sample containing an image and conversation text.
        output_sample (LlavaNextTextSample): The output sample structure where encoded results are stored.

        Returns:
        LlavaNextTextSample: The encoded output sample, containing processed tokens, labels,
            images, loss masks, and metadata.
        """
        conversation_prompt = self.apply_prompt_template(input_sample)

        height = input_sample.image.shape[1]
        width = input_sample.image.shape[2]

        text = [conversation_prompt]
        image_sizes = iter([[height, width]])
        resized_height, resized_width = (
            self.hf_config.vision_config.image_size,
            self.hf_config.vision_config.image_size,
        )
        prompt_strings = []
        for sample in text:
            while self.image_token.token_str in sample:
                image_size = next(image_sizes)
                if not isinstance(image_size, (list, tuple)):
                    # cast to list to avoid numerical precision errors when calculating unpadding
                    image_size = image_size.tolist()
                orig_height, orig_width = image_size
                num_image_tokens = get_number_of_features(
                    orig_height,
                    orig_width,
                    resized_height,
                    resized_width,
                    self.hf_config.image_grid_pinpoints,
                    self.hf_config.vision_config.patch_size,
                )
                sample = sample.replace(self.image_token.token_str, "<placeholder>" * num_image_tokens, 1)
            prompt_strings.append(sample)
        prompt_strings = [sample.replace("<placeholder>", self.image_token.token_str) for sample in prompt_strings]
        conversation_prompt = prompt_strings[0]

        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        tokens = self.tokenize(conversation_prompt)
        labels = self.compute_labels(tokens, input_sample)
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"[Energon] task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"[Energon] task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)

        processed_image = self.process_image(input_sample.image)
        output_sample.__key__ = input_sample.__key__
        output_sample.images = processed_image

        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.num_media_tiles = processed_image.shape[0]
        output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)

        output_sample.image_sizes = torch.tensor([[height, width]], dtype=torch.long)
        return output_sample
