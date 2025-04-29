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

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    InterleavedSample,
    Sample,
    SimilarityInterleavedSample,
    VQASample,
    generic_batch,
)
from megatron.energon.task_encoder.base import stateless
from transformers import AutoImageProcessor, AutoProcessor

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.vlm.data.utils import _find_pattern_indices, convert_to_packed, greedy_knapsack, predict_seq_len
from nemo.utils import logging


@dataclass
class DataSample(Sample):
    """DataSample for multimodal data.

    This class represents a single data sample in a multimodal dataset, containing
    both image and text data along with their associated labels and masks.

    Attributes:
        images (torch.Tensor): Input images with shape (N, C, H, W), where N is typically 1
            for a single sample, C is channels, H is height, and W is width.
        tokens (torch.Tensor): Input token IDs for text data.
        labels (torch.Tensor): Target labels for the tokens.
        loss_mask (torch.Tensor): Mask indicating which tokens should contribute to the loss.
        position_ids (torch.Tensor): Position embeddings for the tokens.
        packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence processing.
        seqlen (int): Length of the sequence before padding.
    """

    images: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    packed_seq_params: Optional[PackedSeqParams] = None
    seqlen: int = field(default_factory=lambda: 0)


@dataclass
class DataBatch(Batch):
    """DataBatch for multimodal data.

    This class represents a batch of data samples in a multimodal dataset. It maintains
    the same structure as DataSample but with an additional batch dimension.

    Attributes:
        images (torch.Tensor): Batched input images with shape (B, N, C, H, W), where B is
            batch size, N is typically 1, C is channels, H is height, and W is width.
        tokens (torch.Tensor): Batched input token IDs with shape (B, L), where L is sequence length.
        labels (torch.Tensor): Batched target labels with shape (B, L).
        loss_mask (torch.Tensor): Batched loss masks with shape (B, L).
        position_ids (torch.Tensor): Batched position embeddings with shape (B, L).
        packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence processing.
    """

    images: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float))
    packed_seq_params: Optional[PackedSeqParams] = None


@dataclass
class TaskEncoderConfig:
    """Configuration for multimodal processing.

    This class consolidates all configuration needed for multimodal processing,
    including model paths, tokenization, image processing, and sequence packing parameters.

    Args:
        hf_path (str, optional): HuggingFace model path used to load tokenizer and image processor.
        tokenizer (AutoTokenizer, optional): Pre-initialized tokenizer instance.
        image_processor (AutoImageProcessor, optional): Pre-initialized image processor instance.

    Note:
        Either hf_path or both tokenizer and image_processor must be provided.
    """

    # Model loading configuration
    hf_path: Optional[str] = None
    tokenizer: Optional[AutoTokenizer] = None
    image_processor: Optional[AutoImageProcessor] = None

    # Token configuration
    image_token_str: str = "<image>"
    image_token_id: int = -200
    ignore_place_holder: int = -100
    stop_string: Optional[str] = "</s>"

    # Chat configuration
    roles: List[str] = field(default_factory=lambda: ['user', 'assistant'])
    chat_template: Optional[str] = None
    image_following_text: bool = True  # For similarity interleaved samples

    # Sequence packing configuration
    packed_sequence: bool = False
    num_image_embeddings_per_tile: int = 576
    packed_sequence_size: int = -1

    # Processing parameters
    pad_to_multiple_of: Optional[int] = 64

    def __post_init__(self):
        """Initialize tokenizer and image processor if not provided.

        Raises:
            ValueError: If neither hf_path nor both tokenizer and image_processor are provided.
        """
        if not self.hf_path and (not self.tokenizer or not self.image_processor):
            raise ValueError("Either hf_path or both tokenizer and image_processor must be provided")

        if self.hf_path:
            self.hf_processor = AutoProcessor.from_pretrained(self.hf_path)

            if not self.tokenizer:
                # We are using AutoTokenizer from nemo.collections.common.tokenizers
                self.tokenizer = AutoTokenizer(self.hf_path)
                logging.info(f"Loaded tokenizer from {self.hf_path}")

            if not self.image_processor:
                self.image_processor = AutoImageProcessor.from_pretrained(self.hf_path)
                logging.info(f"Loaded image processor from {self.hf_path}")

            # Go over all the attributes in the hf_processor and set them as attributes if they are not functions
            # Also give preference to the attributes in the task encoder config
            for attr in dir(self.hf_processor):
                if (
                    not callable(getattr(self.hf_processor, attr))
                    and not attr.startswith('_')
                    and not hasattr(self, attr)
                ):
                    setattr(self, attr, getattr(self.hf_processor, attr))

        if self.stop_string is None:
            self.stop_string = ""

        if self.packed_sequence and self.pad_to_multiple_of is None:
            raise ValueError("pad_to_multiple_of must be provided when using packed sequence. We recommend 64.")


class TaskEncoder(
    DefaultTaskEncoder[
        Union[VQASample, CaptioningSample, InterleavedSample, SimilarityInterleavedSample],
        DataSample,
        DataBatch,
        dict,
    ]
):
    """TaskEncoder for multimodal data processing.

    This class handles the processing of different types of multimodal samples,
    including Visual Question Answering (VQA), Captioning, and Interleaved samples.
    It provides functionality for encoding individual samples, batching them together,
    and handling packed sequences for efficient processing.

    The encoder supports:
    - VQA samples: Processing image-question pairs with corresponding answers
    - [In progress] Interleaved samples: Processing alternating image and text content
    - [In progress] Similarity interleaved samples: Processing image-text pairs for similarity tasks
    - [In progress] Packed sequences: Efficient processing of multiple samples in a single sequence

    Args:
        config (TaskEncoderConfig): Configuration object containing processing parameters

    Note:
        When using packed sequences, the micro batch size must be 1, and the global batch
        size and sequence length must be adjusted accordingly.
    """

    def __init__(self, config: TaskEncoderConfig):
        """Initialize the multimodal processor.

        Args:
            config (TaskEncoderConfig): Configuration for processing
        """
        self.config = config
        self.hf_processor = self.config.hf_processor
        self.tokenizer = self.config.tokenizer

        # Initialize encoders with the config
        self.encoders = {
            "VQASample": self.encode_vqa_sample,
            # "InterleavedSample": self.encode_interleaved_sample,
            # "SimilarityInterleavedSample": self.encode_similarity_interleaved_sample,
        }

    @stateless
    def encode_sample(self, sample: Union[dict, object]) -> DataSample:
        """Process a sample based on its type.

        Args:
            sample: Input sample to process. Can be a dictionary or an object.

        Returns:
            dict: Processed sample in a standardized format
        """
        # Get sample type
        sample_type = type(sample).__name__

        # Get appropriate encoder
        encoder = self.encoders.get(sample_type)
        if not encoder:
            raise ValueError(f"No encoder implemented for sample type {sample_type}")

        # Encode the sample
        encoded_sample: DataSample = encoder(input_sample=sample)

        return encoded_sample

    def batch(self, samples: List[DataSample]) -> DataBatch:
        """
        Batch a list of samples.
        """
        if self.config.packed_sequence:
            if len(samples) > 1:
                raise ValueError(
                    "Micro batch size should be 1 when training with packed sequence, but your micro batch size "
                    f"is {len(samples)}. \nThe following config is equivalent to your current setting for "
                    f"a packed dataset. Please update your config to the following: \n"
                    f"Set micro batch size to 1 (currently {len(samples)})\n"
                    f"Set global batch size to `global_batch_size // {len(samples)}` "
                    f"Set packed sequence length to `original_sample_seq_len * {len(samples)}` "
                    f"(currently {self.config.packed_sequence_size}) \n"
                    f"For details please visit "
                    f"https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html"
                )
            # The batching are taken care by packing.
            batch_sample = samples[0]
        else:
            batch_sample = generic_batch(samples)

        # Copy attributes from sample to batch
        batch = DataBatch()
        for attr in batch_sample.__dict__:
            setattr(batch, attr, getattr(batch_sample, attr))
        return batch

    def encode_batch(self, batch_data: DataBatch) -> dict:
        """
        Encode a batched set of samples for model input.

        This method transforms the raw batched data into a format ready for model input, including
        generating position IDs and other necessary fields.

        Parameters:
        batch_data (DataBatch): The raw batch of data to be encoded.

        Returns:
        dict: A dictionary containing the encoded batch data, ready for model input.
        """
        batch_dict = batch_data.__dict__
        # TODO: Change media to images in the model code and remove this
        if 'images' in batch_dict:
            batch_dict['media'] = batch_dict['images']
            del batch_dict['images']
        micro_batch_size, seq_length = batch_dict['tokens'].size()
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        batch_dict['position_ids'] = position_ids
        if 'attention_mask' not in batch_dict:
            batch_dict['attention_mask'] = None
        # If all the packed_seq_params are None, then we need to set it to an empty tuple
        if all(param is None for param in batch_dict['packed_seq_params']):
            batch_dict['packed_seq_params'] = None
        return batch_dict

    def select_samples_to_pack(self, samples):
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """

        media_token_id = self.config.image_token_id
        lengths = [
            predict_seq_len(
                sample.tokens,
                media_token_index=media_token_id,
                num_image_embeddings_per_tile=self.config.num_image_embeddings_per_tile,
            )
            for sample in samples
        ]
        packed_samples = greedy_knapsack(lengths, samples, self.packed_sequence_size)
        avg_samples_per_bin = round(len(lengths) / len(packed_samples))
        logging.info(
            f"[Seq Packing Info] - Packing seq len: {self.config.packed_sequence_size}, "
            f"Buffered samples: {len(lengths)}, Total number of bins: {len(packed_samples)}, "
            f"Average samples per bin: {avg_samples_per_bin}"
        )
        return packed_samples

    @stateless
    def pack_selected_samples(self, samples):
        """
        Function to pack a list of DataSample into a single DataBatch.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

        Args:
            samples: List of DataSample instances to pack into one sample.

        Returns:
            DataBatch instance.
        """

        packed_images = torch.stack([sample.images for sample in samples])
        packed_tokens, packed_labels, packed_position_ids, packed_loss_mask, packed_seq_params = convert_to_packed(
            tokens=[sample.tokens for sample in samples],
            labels=[sample.labels for sample in samples],
            seqlens=[sample.seqlen for sample in samples],
        )

        return DataBatch(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__,
            tokens=packed_tokens,
            labels=packed_labels,
            images=packed_images,
            position_ids=packed_position_ids,
            loss_mask=packed_loss_mask,
            packed_seq_params=packed_seq_params,
        )

    def encode_vqa_sample(self, input_sample: VQASample) -> DataSample:
        """Encode a VQA sample into a DataSample format.

        Args:
            input_sample (VQASample): Input VQA sample containing image, context and answers

        Returns:
            DataSample: Encoded sample with processed image, tokens, labels and loss mask
        """
        # Apply conversation template
        messages = []
        if self.config.system_prompt:
            messages.append({'role': 'system', 'content': self.config.system_prompt})

        # Ensure context and answers are lists for consistent processing
        contexts = input_sample.context if isinstance(input_sample.context, list) else [input_sample.context]
        answers = input_sample.answers if isinstance(input_sample.answers, list) else [input_sample.answers]

        # Build the conversation messages, replacing image placeholder
        min_length = min(len(contexts), len(answers))
        for i in range(min_length):
            context_with_placeholder = contexts[i].replace("<image>", self.config.image_token)
            messages.append({'role': self.config.roles[0], 'content': context_with_placeholder})
            messages.append({'role': self.config.roles[1], 'content': answers[i]})

        # Generate templated prompt
        conversation_prompt = self.config.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize prompt
        regex_pattern = f'({re.escape(self.config.image_token_str)})'
        chunks = re.split(regex_pattern, conversation_prompt)

        tokenized_chunks = []
        for chunk in chunks:
            if chunk == self.config.image_token_str:
                # Todo (abhi): Expand this with number of image_id_tokens
                tokenized_chunks.append(self.config.image_token_id)
            elif len(chunk) > 0:
                tokenized_chunks.extend(self.config.tokenizer(chunk, add_special_tokens=False).input_ids)

        tokens = torch.tensor(tokenized_chunks, dtype=torch.long)

        # Compute labels
        labels = torch.ones_like(tokens) * self.config.ignore_place_holder
        answers = input_sample.answers if isinstance(input_sample.answers, list) else [input_sample.answers]

        search_start_index = 0
        for answer in answers:
            # Tokenize the answer, including the stop string if provided
            answer_with_stop = answer + (self.config.stop_string or "")
            answer_tokens = self.tokenizer.tokenizer(answer_with_stop, add_special_tokens=False)["input_ids"]
            answer_tokens_tensor = torch.tensor(answer_tokens, device=tokens.device)  # Ensure same device

            # Find answer pattern in tokens
            answer_start, answer_end = _find_pattern_indices(tokens, answer_tokens_tensor, search_start_index)

            if answer_start >= 0:
                labels[answer_start:answer_end] = tokens[answer_start:answer_end]
                search_start_index = answer_end
            else:
                logging.warning(
                    "Unable to find answer segment in the tokenized conversation. "
                    "Skipping labeling for this and subsequent answers. Details: "
                    "\n- Processed Text: %s"
                    "\n- Tokens: %s"
                    "\n- Target Answer Tokens: %s"
                    "\n- Search Start Index: %d",
                    conversation_prompt,
                    tokens,
                    answer_tokens,
                    search_start_index,
                )
                break

        # Prepare final tensors
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()

        seqlen = predict_seq_len(tokens, self.config.num_image_embeddings_per_tile, self.config.image_token_id)
        # Pad tokens
        if self.config.pad_to_multiple_of:
            seqlen_padded = (
                (seqlen + self.config.pad_to_multiple_of - 1)
                // self.config.pad_to_multiple_of
                * self.config.pad_to_multiple_of
            )
            pad_len = seqlen_padded - seqlen

            if pad_len > 0:
                tokens = F.pad(tokens, (0, pad_len), 'constant', 0)
                labels = F.pad(labels, (0, pad_len), 'constant', self.config.ignore_place_holder)

        # Compute loss mask
        loss_mask = torch.ones_like(labels, dtype=torch.float)
        loss_mask[labels == self.config.ignore_place_holder] = 0.0

        # Process image
        image = self.config.image_processor.preprocess(input_sample.image, return_tensors='pt', do_rescale=False)[
            'pixel_values'
        ][0]
        processed_image = image.unsqueeze(0).unsqueeze(0)  # Add T, F dimensions

        return DataSample(
            __key__=input_sample.__key__,
            __restore_key__=input_sample.__restore_key__,
            __subflavor__=input_sample.__subflavor__,
            __subflavors__=input_sample.__subflavors__,
            images=processed_image,
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            seqlen=seqlen,
        )
