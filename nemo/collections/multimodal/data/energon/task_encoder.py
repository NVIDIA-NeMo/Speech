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

import dataclasses
from typing import Dict, List, Union

import torch
from megatron.energon import (
    CaptioningSample,
    DefaultTaskEncoder,
    InterleavedSample,
    SimilarityInterleavedSample,
    VQASample,
    batch_list,
    batch_pad_stack,
)

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample
from nemo.collections.multimodal.data.energon.sample_encoder import (
    InterleavedSampleEncoder,
    SampleEncoder,
    SimilarityInterleavedEncoder,
    VQASampleEncoder,
)


class MultiModalTaskEncoder(
    DefaultTaskEncoder[
        Union[VQASample, CaptioningSample, InterleavedSample, SimilarityInterleavedSample],
        ImageTextSample,
        ImageTextRawBatch,
        dict,
    ]
):
    """
    A task encoder that handles multiple modalities including VQA, captioning, interleaved samples,
    and similarity interleaved samples.

    This class extends the DefaultTaskEncoder and provides a flexible mechanism to handle and encode
    different types of multimodal data. Support for VQA, captioning and interleaved samples is provided by default. It supports registering custom encoders for each sample type
    and provides methods for encoding individual samples, batching them, and further processing the batch
    for model input.
    """

    def __init__(self, tokenizer, image_processor, multimodal_sample_config):
        """
        Initialize the MultiModalTaskEncoder with specific encoders for different sample types.

        Parameters:
        tokenizer (Tokenizer): The tokenizer used for processing text across different sample types.
        image_processor (ImageProcessor): The image processor used for preprocessing images across different sample types.
        multimodal_sample_config (MultiModalSampleConfig): Configuration object for multimodal samples, including tokens and placeholders.
        """

        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: VQASampleEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=multimodal_sample_config,
            ),
            InterleavedSample.__name__: InterleavedSampleEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=multimodal_sample_config,
            ),
            SimilarityInterleavedSample.__name__: SimilarityInterleavedEncoder(
                tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
            ),
        }

    def register_encoder(self, sample_type: str, encoder: SampleEncoder) -> None:
        """
        Registers a custom encoder for a specific sample type.

        This method allows adding or overriding encoders for specific sample types.

        Parameters:
        sample_type (str): The name of the sample type for which the encoder is being registered.
        encoder (SampleEncoder): The custom encoder instance that will handle the specified sample type.
        """
        self.encoders[sample_type] = encoder

    def encode_sample(
        self, sample: Union[VQASample, InterleavedSample, SimilarityInterleavedSample, CaptioningSample]
    ) -> ImageTextSample:
        """
        Encode an individual sample based on its type.

        This method selects the appropriate encoder based on the sample type and encodes the sample
        into a format suitable for further processing.

        Parameters:
        sample (Union[VQASample, InterleavedSample, SimilarityInterleavedSample, CaptioningSample]):
            The sample to be encoded. The sample type is used to determine the appropriate encoder.

        Returns:
        ImageTextSample: The encoded sample.

        Raises:
        NotImplementedError: If no encoder is registered for the sample type.
        """
        sample_type = type(sample).__name__
        encoder = self.encoders.get(sample_type)
        if not encoder:
            raise NotImplementedError(f"No encoder implemented for sample type {sample_type}")
        encoded_sample = encoder.encode(input_sample=sample, output_sample=ImageTextSample())
        return encoded_sample

    def batch(self, samples: List[ImageTextSample]) -> ImageTextRawBatch:
        """
        Batch a list of encoded samples into a single raw batch.

        This method collates individual encoded samples into a batch, preparing them for model input.

        Parameters:
        samples (List[ImageTextSample]): A list of encoded samples to be batched.

        Returns:
        ImageTextRawBatch: The batched data, including images, tokens, labels, and loss masks.
        """

        keys, images, tokens, labels, loss_mask = [], [], [], [], []
        for sample in samples:
            keys.append(sample.__key__)
            images.append(sample.images)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)

        batch_keys = batch_list(keys)
        batch_images = batch_pad_stack(images)
        batch_prompt_tokens = batch_pad_stack(tokens)
        batch_labels = batch_pad_stack(labels)
        batch_loss_mask = batch_pad_stack(loss_mask)
        return ImageTextRawBatch(
            __keys__=batch_keys,
            images=batch_images,
            tokens=batch_prompt_tokens,
            labels=batch_labels,
            loss_mask=batch_loss_mask,
        )

    def encode_batch(self, batch_data: ImageTextRawBatch) -> dict:
        """
        Encode a batched set of samples for model input.

        This method transforms the raw batched data into a format ready for model input, including
        generating position IDs and other necessary fields.

        Parameters:
        batch_data (ImageTextRawBatch): The raw batch of data to be encoded.

        Returns:
        dict: A dictionary containing the encoded batch data, ready for model input.
        """
        batch_dict = dataclasses.asdict(batch_data)
        if 'images' in batch_dict:
            batch_dict['media'] = batch_dict['images']
            del batch_dict['images']
        micro_batch_size, seq_length = batch_dict['tokens'].size()
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        batch_dict['position_ids'] = position_ids
        batch_dict['attention_mask'] = None
        return batch_dict
