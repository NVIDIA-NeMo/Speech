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
from megatron.energon.task_encoder.base import stateless

from nemo.collections.multimodal.data.energon.config import (
    ImageTextRawBatch,
    ImageTextSample,
    PackedImageTextRawBatch,
    PackedImageTextSample,
)
from nemo.collections.multimodal.data.energon.sample_encoder import (
    InterleavedSampleEncoder,
    SampleEncoder,
    SimilarityInterleavedEncoder,
    VQASampleEncoder,
)
from nemo.utils import logging


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
    different types of multimodal data. Support for VQA, captioning and interleaved samples is provided by default.
    It supports registering custom encoders for each sample type
    and provides methods for encoding individual samples, batching them, and further processing the batch
    for model input.
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
        multimodal_sample_config,
        packed_sequence=False,
        packed_sequence_size=-1,
        num_image_embeddings_per_tile=576,
        image_tag_type=None,
    ):
        """
        Initialize the MultiModalTaskEncoder with specific encoders for different sample types.

        Parameters:
        tokenizer (Tokenizer): The tokenizer used for processing textual components across sample types.
        image_processor (ImageProcessor): The image processor responsible for preprocessing image data.
        multimodal_sample_config (MultiModalSampleConfig): Configuration object defining properties and
            requirements for multimodal samples.
        packed_sequence (bool, optional): Flag indicating whether packed sequences are used. Default is False.
        packed_sequence_size (int, optional): The size of packed sequences, used when `packed_sequence` is True.
            Default is -1.
        num_image_embeddings_per_tile (int, optional): Number of image embeddings per image tile. Determines
            the granularity of image features. Default is 576.
        """
        self.tokenizer = tokenizer
        self.sample_config = multimodal_sample_config
        self.packed_sequence = packed_sequence
        self.num_image_embeddings_per_tile = num_image_embeddings_per_tile  # only used with seq packing
        self.image_tag_type = image_tag_type
        self.packed_sequence_size = packed_sequence_size
        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: VQASampleEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=multimodal_sample_config,
                image_tag_type=image_tag_type,
            ),
            InterleavedSample.__name__: InterleavedSampleEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=multimodal_sample_config,
                image_tag_type=image_tag_type,
            ),
            SimilarityInterleavedSample.__name__: SimilarityInterleavedEncoder(
                tokenizer=tokenizer,
                image_processor=image_processor,
                multimodal_sample_config=multimodal_sample_config,
                image_tag_type=image_tag_type,
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

    @stateless
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

    def batch(
        self, samples: List[Union[ImageTextSample, PackedImageTextSample]]
    ) -> Union[ImageTextRawBatch, PackedImageTextRawBatch]:
        """
        Batch a list of encoded samples into a single raw batch.

        This method collates individual encoded samples into a batch, preparing them for model input.

        Parameters:
        samples (List[ImageTextSample]): A list of encoded samples to be batched.

        Returns:
        ImageTextRawBatch: The batched data, including images, tokens, labels, and loss masks.
        """

        if self.packed_sequence:
            if len(samples) > 1:
                raise ValueError(
                    "Micro batch size should be 1 when training with packed sequence, but your micro batch size "
                    f"is {len(samples)}. \nThe following config is equivalent to your current setting for "
                    f"a packed dataset. Please update your config to the following: \n"
                    f"Set micro batch size to 1 (currently {len(samples)})\n"
                    f"Set global batch size to `global_batch_size // {len(samples)}` "
                    f"Set packed sequence length to `original_sample_seq_len * {len(samples)}` "
                    f"(currently {self.packed_sequence_size}) \n"
                    f"For details please visit "
                    f"https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html"
                )
            # The batching are taken care by packing.
            sample = samples[0]
            return PackedImageTextRawBatch(
                __keys__=sample.__key__,
                images=sample.images,
                tokens=sample.tokens,
                labels=sample.labels,
                loss_mask=sample.loss_mask,
                position_ids=sample.position_ids,
                packed_seq_params=sample.packed_seq_params,
                num_image_tiles=sample.num_image_tiles,
            )
        else:
            keys, images, tokens, labels, loss_mask = [], [], [], [], []
            batch_num_image_tiles = []
            for sample in samples:
                keys.append(sample.__key__)
                images.append(sample.images)
                tokens.append(sample.tokens)
                labels.append(sample.labels)
                loss_mask.append(sample.loss_mask)
                if sample.num_image_tiles is not None:
                    batch_num_image_tiles.extend(sample.num_image_tiles)
            if len(batch_num_image_tiles) == 0:
                batch_num_image_tiles = None

            batch_keys = batch_list(keys)
            batch_images = batch_pad_stack(images)
            if batch_images.ndim == 5:
                batch_images = batch_images.reshape(-1, *batch_images.shape[2:])
            batch_prompt_tokens = batch_pad_stack(tokens)
            batch_labels = batch_pad_stack(labels)
            batch_loss_mask = batch_pad_stack(loss_mask)
            return ImageTextRawBatch(
                __keys__=batch_keys,
                images=batch_images,
                tokens=batch_prompt_tokens,
                labels=batch_labels,
                loss_mask=batch_loss_mask,
                num_image_tiles=batch_num_image_tiles,
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
        batch_dict = batch_data.__dict__
        if 'images' in batch_dict:
            batch_dict['media'] = batch_dict['images']
            del batch_dict['images']
        is_num_image_tiles_present = (
            'num_image_tiles' in batch_dict
            and batch_dict['num_image_tiles'] is not None
            and batch_dict['num_image_tiles'] != 0
        )
        is_num_media_tiles_present = (
            'num_media_tiles' in batch_dict
            and batch_dict['num_media_tiles'] is not None
            and batch_dict['num_media_tiles'] != 0
        )

        # Assert both should not be present
        assert not (
            is_num_image_tiles_present and is_num_media_tiles_present
        ), "num_image_tiles and num_media_tiles should not be present at the same time"
        if is_num_image_tiles_present:
            batch_dict['num_media_tiles'] = batch_dict['num_image_tiles']
            del batch_dict['num_image_tiles']
        micro_batch_size, seq_length = batch_dict['tokens'].size()
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        batch_dict['position_ids'] = position_ids
        if 'attention_mask' not in batch_dict:
            batch_dict['attention_mask'] = None
        return batch_dict

    def select_samples_to_pack(self, samples):
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        from nemo.collections.vlm.neva.data.sequence_packing import greedy_knapsack, predict_seq_len

        media_token_id = self.sample_config.image_token.token_id
        lengths = [
            predict_seq_len(
                sample.tokens,
                media_token_index=media_token_id,
                num_image_embeddings_per_tile=self.num_image_embeddings_per_tile,
            )
            for sample in samples
        ]
        packed_samples = greedy_knapsack(lengths, samples, self.packed_sequence_size)
        avg_samples_per_bin = round(len(lengths) / len(packed_samples))
        logging.info(
            f"[Seq Packing Info] - Packing seq len: {self.packed_sequence_size}, "
            f"Buffered samples: {len(lengths)}, Total number of bins: {len(packed_samples)}, "
            f"Average samples per bin: {avg_samples_per_bin}"
        )
        return packed_samples

    @stateless
    def pack_selected_samples(self, samples):
        """
        Function to pack a list of ImageTaskSample into a single ImageTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

        Args:
            samples: List of ImageTaskSample instances to pack into one sample.

        Returns:
            ImageTaskSamplePacked instance.
        """
        from nemo.collections.vlm.neva.data.sequence_packing import convert_to_packed

        packed_images = torch.cat([sample.images for sample in samples], dim=0)
        media_token_id = self.sample_config.image_token.token_id
        packed_tokens, packed_labels, packed_position_ids, packed_loss_mask, packed_seq_params = convert_to_packed(
            tokens=[sample.tokens for sample in samples],
            labels=[sample.labels for sample in samples],
            num_image_embeddings_per_tile=self.num_image_embeddings_per_tile,
            media_token_index=media_token_id,
            ignore_index=self.sample_config.ignore_place_holder,
        )
        batch_num_image_tiles = []
        for sample in samples:
            if sample.num_image_tiles is not None:
                batch_num_image_tiles.extend(sample.num_image_tiles)
        if len(batch_num_image_tiles) == 0:
            batch_num_image_tiles = None

        return PackedImageTextSample(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            tokens=packed_tokens,
            labels=packed_labels,
            images=packed_images,
            position_ids=packed_position_ids,
            loss_mask=packed_loss_mask,
            packed_seq_params=packed_seq_params,
            num_image_tiles=batch_num_image_tiles,
        )
