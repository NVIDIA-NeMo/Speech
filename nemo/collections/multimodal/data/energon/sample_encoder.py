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
from abc import ABC, abstractmethod

import torch
from einops import rearrange
from megatron.energon import InterleavedSample, SimilarityInterleavedSample, VQASample

from nemo.collections.multimodal.data.energon.config import ImageTextSample, MultiModalSampleConfig
from nemo.utils import logging


class SampleEncoder(ABC):
    def __init__(self):
        """
        Initialize the SampleEncoder class.

        This class serves as an abstract base class for encoding samples. It provides a common interface for
        different types of sample encoders. Subclasses should implement the encode method to perform the actual
        encoding process.

        Parameters:
        None

        Returns:
        None
        """
        return None

    @abstractmethod
    def encode(self, input_sample, output_sample):
        """
        Abstract method to encode a sample. Must be implemented by subclasses.

        This method is responsible for encoding a given input sample into a format suitable for further processing.
        The encoded sample is then stored in the output_sample object.

        Parameters:
        input_sample (object): The input sample to be encoded. The type and structure of this object depend on the specific subclass.
        output_sample (object): The object where the encoded sample will be stored. The type and structure of this object depend on the specific subclass.

        Returns:
        None: The method does not return any value.

        Raises:
        NotImplementedError: If the method is called directly on the abstract class, it will raise this exception. Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the encode method.")


class BaseSampleEncoder(SampleEncoder):
    """
    Base class for encoding multimodal samples, specifically for handling text and image data.

    This class provides basic functionality for preprocessing images, computing loss masks,
    and managing sample configuration. It serves as a base class for more specialized encoders.

    Attributes:
    tokenizer (Tokenizer): The HF tokenizer used for tokenizing input text.
    image_processor (ImageProcessor): The HF image processor used for preprocessing input images.
    multimodal_sample_config (MultiModalSampleConfig): Configuration for multimodal samples, including tokens and placeholders.
    ignore_place_holder (int): Token ID used to ignore certain tokens during loss computation.
    image_token (Token): Token dataclass representing image placeholders in the tokenized sequence.
    """

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the BaseSampleEncoder.

        Parameters:
        tokenizer (Tokenizer): The tokenizer used for processing text.
        image_processor (ImageProcessor): The image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.multimodal_sample_config = multimodal_sample_config
        self.ignore_place_holder = multimodal_sample_config.ignore_place_holder
        self.image_token = multimodal_sample_config.image_token

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess and reshape an input image for encoding.

        The function preprocesses an image using the specified image processor and reshapes it
        to the expected format for further processing.

        Parameters:
        image (torch.Tensor): A tensor representing the input image with dimensions (channels, height, width).

        Returns:
        torch.Tensor: A preprocessed and reshaped image tensor with dimensions (1, 1, channels, height, width).
        """
        image = self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        assert isinstance(image, torch.Tensor)
        image = rearrange(image, "c h w -> 1 1 c h w")  # T F c h w
        return image

    def compute_loss_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute a binary loss mask based on the provided labels.

        The function generates a mask where the loss is computed only for tokens that are not
        equal to the `ignore_place_holder` token.

        Parameters:
        labels (torch.Tensor): A tensor containing labels for which the loss mask needs to be generated.

        Returns:
        torch.Tensor: A binary mask tensor with the same shape as the input labels. The mask has ones
            for tokens where loss should be computed and zeros for `ignore_place_holder` tokens.
        """
        loss_mask = torch.ones(labels.size(), dtype=torch.float)
        loss_mask[labels == self.ignore_place_holder] = 0.0  # loss computed only for answer and non image tokens
        return loss_mask

    def encode(self, input_sample: ImageTextSample, output_sample: ImageTextSample) -> None:
        """
        Abstract method to encode an input sample.

        Subclasses must implement this method to encode input samples into the desired format.

        Parameters:
        input_sample (ImageTextSample): The sample to be encoded.
        output_sample (ImageTextSample): The object to store the encoded sample.

        Returns:
        None

        Raises:
        NotImplementedError: If the method is called directly on the abstract class.
        """
        raise NotImplementedError("Subclasses must implement the encode method.")


class VQASampleEncoder(BaseSampleEncoder):
    """
    Encoder specifically designed for Visual Question Answering (VQA) samples.

    This class extends the BaseSampleEncoder to handle VQA tasks, applying a specific prompt
    template and computing labels and loss masks based on the VQA input.

    Attributes:
    conversation_template_config (ConversationTemplateConfig): Configuration for conversation templates used in VQA.
    """

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

    def apply_prompt_template(self, input_text: VQASample, use_plain=False):
        """
        Apply a conversation template to the input text for VQA.

        This method generates a templated prompt by combining system, user, and assistant messages.

        Parameters:
        input_text (VQASample): The VQA sample containing the context and answer.
        use_plain (bool, optional): Whether to use a plain format for the prompt. Defaults to False.

        Returns:
        str: The generated templated prompt as a string.
        """
        logging.debug(f"apply_conversation_template context {input_text.context} answer {input_text.answers}")

        messages = []

        # Add system message if it exists
        if self.conversation_template_config.system:
            messages.append({'role': 'system', 'content': self.conversation_template_config.system})

        # Handle cases where context and answers are lists
        if isinstance(input_text.context, list) and isinstance(input_text.answers, list):
            # Ensure both lists are the same length or adjust based on your specific needs
            min_length = min(len(input_text.context), len(input_text.answers))
            for i in range(min_length):
                messages.append({'role': self.conversation_template_config.roles[0], 'content': input_text.context[i]})
                messages.append({'role': self.conversation_template_config.roles[1], 'content': input_text.answers[i]})
        elif isinstance(input_text.context, str) and isinstance(input_text.answers, str):
            # Handle single context and answer as strings
            messages.append({'role': self.conversation_template_config.roles[0], 'content': input_text.context})
            messages.append({'role': self.conversation_template_config.roles[1], 'content': input_text.answers})
        else:
            raise ValueError(
                f"VQA Sample context/answers should either be a List[str] or str. Other types not supported"
            )
        # Set the chat template if defined
        if self.conversation_template_config.chat_template:
            self.tokenizer.chat_template = self.conversation_template_config.chat_template
        elif self.tokenizer.chat_template is None:
            raise ValueError(
                "Both tokenizer and conversation template does not have chat template defined. Refer to https://huggingface.co/docs/transformers/main/en/chat_templating"
            )

        # Apply the conversation template to generate the prompt
        templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        logging.debug(f"apply prompt template templated_prompt {templated_prompt}")
        return templated_prompt

    def tokenize(self, prompt: str) -> torch.Tensor:
        """
        Tokenize the input prompt, replacing special tokens with their IDs.

        This method splits the prompt into chunks based on the presence of special tokens (like <image>)
        and tokenizes each chunk. Special tokens are replaced with their corresponding token IDs.

        Parameters:
        prompt (str): The prompt string to tokenize.

        Returns:
        torch.Tensor: A tensor containing the tokenized prompt.
        """
        # Split the prompt into chunks and track special tokens
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

    def compute_labels(self, tokens: torch.Tensor, sample: VQASample) -> torch.Tensor:
        """
        Compute labels for the tokenized prompt based on the answers in the VQA sample.

        This method generates a label tensor where the tokens corresponding to the answers are marked
        with their token IDs, while other tokens are marked with the `ignore_place_holder` ID.

        Parameters:
        tokens (torch.Tensor): A tensor containing the tokenized prompt.
        sample (VQASample): The VQA sample containing the answers.

        Returns:
        torch.Tensor: A tensor containing the labels for the tokenized prompt.
        """
        # Initialize labels with ignore index
        labels = torch.ones_like(tokens) * self.ignore_place_holder
        search_start_index = 0  # Initialize search index to start labeling answers sequentially

        stop_str = getattr(self.conversation_template_config, "stop_string", None)

        # Check if answers is a single string or a list
        answers = sample.answers if isinstance(sample.answers, list) else [sample.answers]

        # Iterate through the answers and compute labels for each answer
        for answer in answers:
            # Encode the answer with the stop string
            answer_tokens = self.tokenizer.encode(
                answer + ("" if stop_str is None else stop_str), add_special_tokens=False, return_tensors="pt"
            )[0]

            # Find the start and end indices of the answer tokens in the prompt
            answer_start, answer_end = _find_pattern_indices(tokens, answer_tokens, search_start_index)

            # Label the answer tokens
            labels[answer_start:answer_end] = tokens[answer_start:answer_end]

            # Update the search start index to the end of the current answer tokens
            search_start_index = answer_end
        return labels

    def encode(self, input_sample: VQASample, output_sample: ImageTextSample):
        """
        Encode a VQA sample into a format suitable for further processing.

        This method applies a prompt template, tokenizes the prompt, computes labels and a loss mask,
        and processes the image. The encoded sample is then stored in the output_sample object.

        Parameters:
        input_sample (VQASample): The VQA sample to be encoded.
        output_sample (ImageTextSample): The object to store the encoded sample.

        Returns:
        ImageTextSample: The encoded sample stored in output_sample.
        """
        # apply prompt template
        conversation_prompt = self.apply_prompt_template(input_sample)
        logging.debug(f"task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        tokens = self.tokenize(conversation_prompt)
        labels = self.compute_labels(tokens, input_sample)

        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)
        processed_image = self.process_image(input_sample.image)
        processed_image = processed_image.squeeze()
        output_sample.__key__ = input_sample.__key__
        output_sample.images = processed_image
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        return output_sample


class InterleavedSampleEncoder(BaseSampleEncoder):
    """
    Encoder for handling interleaved sequences of text and images (InterleavedSample from energon).

    This class extends the BaseSampleEncoder to handle interleaved samples, where the input
    consists of a sequence of text strings and image tensors. The text and images are processed
    and encoded into a format suitable for further processing.

    Attributes:
    tokenizer (Tokenizer): The tokenizer used for processing text.
    image_processor (ImageProcessor): The image processor used for preprocessing images.
    multimodal_sample_config (MultiModalSampleConfig): Configuration for multimodal samples, including tokens and placeholders.
    """

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the InterleavedSampleEncoder.

        Parameters:
        tokenizer (Tokenizer): The tokenizer used for processing text.
        image_processor (ImageProcessor): The image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)

    def tokenize(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the input sequence and process images in an interleaved sample.

        This method processes a sequence that consists of text strings and image tensors.
        The text is tokenized, and the images are processed. The method returns a tensor
        of tokenized text and a concatenated tensor of processed images.

        Parameters:
        sample (InterleavedSample): The interleaved sample containing a sequence of text strings and image tensors.

        Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor with tokenized text and image token IDs.
            - A concatenated tensor of processed images.
        """
        images = []
        # sample.sequence is a list consisting of text string or image tensor (only image modality supported for now)
        tokenized_chunks = []
        images = []
        for chunk in sample.sequence:
            if isinstance(chunk, torch.Tensor):
                tokenized_chunks.append(self.image_token.token_id)
                processed_image = self.process_image(chunk)
                images.append(processed_image)
            elif len(chunk) > 0:
                logging.debug(f"Multimodal datalaoder encoder interleaved sample text chunk {chunk}")
                tokenized_chunks.extend(self.tokenizer(chunk, add_special_tokens=False).input_ids)
            else:
                raise ValueError(f"Unsupported type in interleaved sequence: {type(chunk)}")
        tokens = torch.tensor(tokenized_chunks, dtype=torch.long)
        logging.debug(f"Multimodal dataloader encode interleaved sample tokenized chunks {tokenized_chunks}")
        image_tensor = torch.concatenate(images, dim=1)  # T F(no of images) c h w
        return tokens, image_tensor

    def compute_labels(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute labels for an interleaved sample, ignoring image token IDs.

        This method generates a label tensor where the tokens corresponding to images are marked
        with the `ignore_place_holder` ID, and other tokens retain their original IDs.

        Parameters:
        tokens (torch.Tensor): A tensor containing the tokenized sequence.

        Returns:
        torch.Tensor: A tensor containing the labels for the tokenized sequence.
        """
        labels = tokens.clone()
        labels[labels == self.image_token.token_id] = self.ignore_place_holder
        labels = labels[1:].contiguous()
        return labels

    def encode(self, input_sample: InterleavedSample, output_sample: ImageTextSample):
        """
        Encode an interleaved sample into a format suitable for further processing.

        This method tokenizes the input sequence, computes labels and a loss mask, and processes
        the images. The encoded sample is then stored in the output_sample object.

        Parameters:
        input_sample (InterleavedSample): The interleaved sample to be encoded.
        output_sample (ImageTextSample): The object to store the encoded sample.

        Returns:
        ImageTextSample: The encoded sample stored in output_sample.
        """
        # tokenize prompt
        tokens, images = self.tokenize(input_sample)
        labels = self.compute_labels(tokens)
        tokens = tokens[:-1]
        logging.debug(f"task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)
        output_sample.__key__ = input_sample.__key__
        output_sample.images = images
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        return output_sample


class SimilarityInterleavedEncoder(InterleavedSampleEncoder):
    """
    Encoder for handling similarity-based interleaved sequences of text and images.

    This class extends the InterleavedSampleEncoder to handle samples where images and text
    are interleaved based on a similarity matrix. The images are inserted into the text sequence
    based on the similarity scores (matched_text_indices), allowing for flexible interleaving of media types.

    Attributes:
    image_following_text (bool): A flag indicating whether images should follow the text they are related to.
    """

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the SimilarityInterleavedEncoder.

        Parameters:
        tokenizer (Tokenizer): The tokenizer used for processing text.
        image_processor (ImageProcessor): The image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.image_following_text = multimodal_sample_config.image_following_text

    def tokenize(self, sample: SimilarityInterleavedSample) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the input sequence and process images based on similarity indices.

        This method processes a sequence of text strings and images, interleaving them based
        on similarity indices (matched_text_indices). The text is tokenized, and the images are processed. The method
        returns a tensor of tokenized text and a concatenated tensor of processed images.

        Parameters:
        sample (SimilarityInterleavedSample): The sample containing a sequence of text strings and images,
            along with similarity indices that determine the interleaving order.

        Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor with tokenized text and image token IDs.
            - A concatenated tensor of processed images.
        """
        images = sample.images
        texts = sample.texts
        matched_text_indices = sample.matched_text_indices
        # Initialize a list with placeholders for interleaving images and texts
        interleaved_list = []
        image_idx = 0
        # Sort images according to matched_text_indices
        sorted_images = [img for _, img in sorted(zip(matched_text_indices, images))]
        sorted_images = [self.process_image(chunk) for chunk in sorted_images]
        sorted_indices = sorted(matched_text_indices)
        # Traverse texts, and interleave images based on sorted_indices
        for text_idx in range(len(texts)):
            if image_idx < len(sorted_indices) and sorted_indices[image_idx] == text_idx:
                if not self.image_following_text:
                    # Add image before the text
                    interleaved_list.append(self.image_token.token_id)
                interleaved_list.append(texts[text_idx])
                if self.image_following_text:
                    # Add image after the text
                    interleaved_list.append(self.image_token.token_id)
                image_idx += 1
            else:
                interleaved_list.append(texts[text_idx])

        # Merge consecutve text entries with a space between them
        final_sequence = []
        for item in interleaved_list:
            if final_sequence and isinstance(final_sequence[-1], str) and isinstance(item, str):
                final_sequence[-1] += " " + item
            else:
                final_sequence.append(item)
        tokenized_chunks = []
        for chunk in final_sequence:
            if chunk == self.image_token.token_id:
                tokenized_chunks.append(chunk)
            else:
                tokenized_chunks.extend(self.tokenizer(chunk, add_special_tokens=False).input_ids)
        tokens = torch.tensor(tokenized_chunks, dtype=torch.long)
        logging.debug(
            f"Multimodal dataloader encode similarity interleaved sample tokenized chunks {tokenized_chunks}"
        )
        image_tensor = torch.concatenate(sorted_images, dim=1)  # T F(no of images) c h w
        return tokens, image_tensor


def _find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = template[i : i + pattern_len] == pattern
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1
