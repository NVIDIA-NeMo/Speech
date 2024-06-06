# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import json
import logging
import os
import re
import tarfile
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import decord
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset, default_collate
from transformers import CLIPImageProcessor, SiglipImageProcessor

import nemo.collections.multimodal.data.neva.conversation as conversation_lib
from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.data.neva.conversation import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_LABELS_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

MAX_NUM_IMAGES = 1
IGNORE_INDEX = -1

try:
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class TarOrFolderImageLoader:
    """
    A class for loading images from a tar archive or a regular folder.

    This class provides functionality to open and read images from either a tar archive
    (.tar file) or a standard directory with image files. It builds an index of images
    if the source is a tar archive for efficient access.

    Attributes:
        image_folder (str): The path to the tar archive or image folder.
        tar_index (dict): A dictionary that maps file names to their tarfile member
                          objects if the image source is a tar archive.

    Methods:
        __init__(self, image_folder): Initializes the loader with the specified image folder.
        build_index(self): Builds an index of image file names and their corresponding
                           tarfile member objects for a tar archive.
        open_image(self, file_name): Opens and returns an image by its file name. The image
                                     is returned as an RGB PIL Image object.
    """

    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.tar_index = {}
        if self.image_folder.endswith('.tar'):
            self.build_index()

    def build_index(self):
        with tarfile.open(self.image_folder, 'r') as tar:
            for member in tar.getmembers():
                self.tar_index[member.name] = member

    def open_image(self, file_name):
        if self.image_folder.endswith('.tar'):
            with tarfile.open(self.image_folder, 'r') as tar:
                member = self.tar_index.get(file_name)
                if member:
                    f = tar.extractfile(member)
                    return Image.open(f).convert('RGB')
        else:
            return Image.open(os.path.join(self.image_folder, file_name)).convert('RGB')
        return None


class TarOrFolderVideoLoader:
    """
    A class for loading videos from a tar archive or a regular folder.

    This class provides functionality to open and read videos from either a tar archive
    (.tar file) or a standard directory with video files. It builds an index of videos
    if the source is a tar archive for efficient access.

    Attributes:
        video_folder (str): The path to the tar archive or video folder.
        data_cfg (dict): A dictionary of configuration options for video decoding to frames
        tar_index (dict): A dictionary that maps file names to their tarfile member
                          objects if the video source is a tar archive.

    Methods:
        __init__(self, video_folder): Initializes the loader with the specified video folder.
        build_index(self): Builds an index of image file names and their corresponding
                           tarfile member objects for a tar archive.
        open_video(self, file_name): Opens and returns an video by its file name. The video
                                     is returned as a list of RGB PIL Image objects.
        flatten_frames(self, cap): Converts decord VideoReader video object to list of frame
                                   images based on data config information.
    """

    def __init__(self, video_folder, data_cfg):
        self.video_folder = video_folder
        self.data_cfg = data_cfg
        self.tar_index = {}
        if self.video_folder.endswith('.tar'):
            self.build_index()

    def build_index(self):
        with tarfile.open(self.video_folder, 'r') as tar:
            for member in tar.getmembers():
                self.tar_index[member.name] = member

    def open_video(self, file_name):
        if self.video_folder.endswith('.tar'):
            with tarfile.open(self.video_folder, 'r') as tar:
                member = self.tar_index.get(file_name)
                if member:
                    f = tar.extractfile(member)
                    cap = decord.VideoReader(f)
                    return self.flatten_frames(cap)
        else:
            decord.bridge.set_bridge("torch")
            cap = decord.VideoReader(os.path.join(self.video_folder, file_name))
            return self.flatten_frames(cap)
        return None

    def flatten_frames(self, cap):
        if self.data_cfg['splice_single_frame'] == 'first':
            frame = cap[0].asnumpy()
            return Image.fromarray(frame).convert('RGB')
        elif self.data_cfg['splice_single_frame'] == 'middle':
            frame = cap[len(cap) // 2].asnumpy()
            return Image.fromarray(frame).convert('RGB')
        elif self.data_cfg['splice_single_frame'] == 'last':
            frame = cap[-1].asnumpy()
            return Image.fromarray(frame).convert('RGB')
        else:
            if self.data_cfg['num_frames'] == -1:
                frames = []
                for frame in cap:
                    rgb_frame = frame.asnumpy()
                    img = Image.fromarray(rgb_frame).convert('RGB')
                    frames.append(img)
                return frames
            else:
                num_frames = min(len(cap), self.data_cfg['num_frames'])
                indices = np.linspace(0, len(cap) - 1, num_frames, dtype=int)
                frames = []
                frames = cap.get_batch(indices)

                while len(frames) < self.data_cfg['num_frames']:
                    frames.append(frames[-1])
                return frames


def tokenize(
    texts: Union[str, List[str]],
    tokenizer: Any,
    context_length: int,
    add_extra_token: int,
) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s). If the list of tokens exceeds the context
    length plus the number of extra tokens, it gets truncated. If it's smaller, it gets padded with zeros.

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize.
    tokenizer : Any
        A tokenizer to be used for tokenization.
    context_length : int
        The context length to be used for the output tensor.
    add_extra_token : int
        Number of extra tokens to add, should be either 0 or 1.

    Returns
    -------
    torch.LongTensor
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length + add_extra_token].
    """
    assert add_extra_token == 0 or add_extra_token == 1, "`add_extra_token` should be either 0 or 1."

    texts_is_str = False
    if isinstance(texts, str):
        texts = [texts]
        texts_is_str = True
    tokens = [tokenizer.text_to_ids(t) for t in texts]
    max_len = max([len(token) for token in tokens])
    context_length = min(max_len - add_extra_token, context_length)
    # truncate and padding
    result = torch.zeros(len(tokens), context_length + add_extra_token, dtype=torch.long)

    for i, token in enumerate(tokens):
        if len(token) > context_length + add_extra_token:
            token = token[: context_length + add_extra_token]  # Truncate
        result[i, : len(token)] = torch.tensor(token)
    if texts_is_str:
        result = result[0]
    return result


def preprocess_multimodal(sources: dict, multimodal_cfg: dict, cur_token_len: int, use_plain: bool = False) -> Dict:
    """
    Preprocesses multimodal sources based on the provided configuration.

    This function modifies the sources for multimodal data processing. It checks if the data is multimodal and
    adjusts the token lengths accordingly. It also handles the start and end tokens for images and replaces
    image tokens in conversations.

    Parameters:
    - sources (dict): A dictionary containing the multimodal sources to be processed.
    - multimodal_cfg (dict): A configuration dictionary specifying various options for multimodal processing.
      It includes keys like 'is_multimodal', 'use_im_start_end', and 'sep_image_conv_front'.
    - cur_token_len (int): The current length of tokens to be considered for image processing.
    - use_plain (bool, optional): A boolean flag to use plain image token replacement without additional processing.
      Defaults to False.

    Returns:
    - dict: The processed sources dictionary after applying multimodal preprocessing steps.
    """
    is_multimodal = multimodal_cfg['is_multimodal']
    model_type = multimodal_cfg['model_type']
    media_type = multimodal_cfg['media_type']
    image_token_len = cur_token_len
    if media_type == 'image':
        default_token = DEFAULT_IMAGE_TOKEN
    elif media_type == 'video':
        default_token = DEFAULT_VIDEO_TOKEN
    else:
        return sources

    if not is_multimodal:
        return sources

    num_patches = image_token_len

    if media_type == 'video':
        num_patches *= multimodal_cfg['num_frames']

    if multimodal_cfg['mm_mlp_adapter_type'] == 'mlp_downsample':
        num_patches //= 4

    if multimodal_cfg['use_im_start_end']:
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN[model_type] * num_patches
    else:
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN[model_type] * (num_patches - 2)
    replace_token = DEFAULT_IM_START_TOKEN[model_type] + replace_token + DEFAULT_IM_END_TOKEN[model_type]

    for source in sources:
        conversation = source['conversations']
        if multimodal_cfg['sep_image_conv_front']:
            assert default_token in conversation[0]['value']
            conversation[0]['value'] = conversation[0]['value'].replace(default_token, '').strip()
            conversation[0]['value'] = (
                default_token
                + conversation_lib.default_conversation.sep
                + conversation_lib.default_conversation.roles[0]
                + ": "
                + conversation[0]['value']
            )
        if use_plain:
            assert default_token in conversation[0]['value']
            conversation[0]['value'] = default_token
        for turn in conversation:
            turn["value"] = turn["value"].replace(default_token, replace_token)

    return sources


def process_image(processor, image, image_aspect_ratio="square"):
    if isinstance(processor, CLIPImageProcessor) or isinstance(processor, SiglipImageProcessor):
        # image processor from HF
        if image_aspect_ratio == 'keep':
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = processor.preprocess(
                image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge}
            )['pixel_values'][0]
        elif image_aspect_ratio == 'pad':

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    else:
        assert image_aspect_ratio == 'square', 'NeMo image transform with setting `image_aspect_ratio` to `square`.'
        image = processor(image)
    return image


def preprocess_llama_3(
    sources: dict,
    tokenizer,
    cfg,
) -> Dict:
    """
    Preprocesses sources for the LLaMA 3 model configuration.

    The function applies prompt templates and tokenizes the conversations according to the LLaMA 2 model specifications.
    It involves special handling of tokens, masking of labels, and adjustments based on configuration settings.

    Parameters:
    - sources (dict): A dictionary of sources containing conversations to be processed.
    - tokenizer: The tokenizer to be used for processing the text.
    - cfg: Configuration settings for preprocessing, including context length and additional tokens.

    Returns:
    - Dict: A dictionary containing tokenized and labeled data suitable for the LLaMA 2 model.
      This includes tokens, labels, and any special processing as defined in the configuration.
    """
    conv = conversation_lib.conv_llava_llama_3.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        source = source['conversations']
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    add_extra_token = cfg.get("add_extra_token")

    # Tokenize conversations
    tokens = tokenize(
        texts=conversations,
        tokenizer=tokenizer,
        context_length=cfg.get("context_length"),
        add_extra_token=add_extra_token,
    )
    labels = tokens.clone().detach()
    # Mask labels
    sep = "<|start_header_id|>assistant<|end_header_id|>\n\n"  # part sep
    round_sep = "<|start_header_id|>user<|end_header_id|>\n\n"
    for conversation, target in zip(conversations, labels):
        # the first match of round sep is going to be the one after system, which is not the intended behavior
        rounds = conversation.split(round_sep)
        rounds = [round_sep.join(rounds[:2])] + rounds[2:]
        cur_len = 0
        for i, rou in enumerate(rounds):

            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if i == 0:
                round_len = len(tokenizer.text_to_ids(rou))
                instruction_len = len(tokenizer.text_to_ids(parts[0]))
            else:
                round_len = len(tokenizer.text_to_ids(round_sep + rou))
                instruction_len = len(tokenizer.text_to_ids(round_sep + parts[0]))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    # Check if masking working correctly
    # print([x for x in zip(tokens[0].numpy().tolist(), labels[0].numpy().tolist())])

    if add_extra_token:
        tokens = tokens[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
    else:
        labels = torch.roll(labels, shifts=-1, dims=-1)
        labels[:, -1] = IGNORE_INDEX

    return dict(
        tokens=tokens,
        labels=labels,
    )


def preprocess_llama_2(
    sources: dict,
    tokenizer,
    cfg,
) -> Dict:
    """
    Preprocesses sources for the LLaMA 2 model configuration.

    The function applies prompt templates and tokenizes the conversations according to the LLaMA 2 model specifications.
    It involves special handling of tokens, masking of labels, and adjustments based on configuration settings.

    Parameters:
    - sources (dict): A dictionary of sources containing conversations to be processed.
    - tokenizer: The tokenizer to be used for processing the text.
    - cfg: Configuration settings for preprocessing, including context length and additional tokens.

    Returns:
    - Dict: A dictionary containing tokenized and labeled data suitable for the LLaMA 2 model.
      This includes tokens, labels, and any special processing as defined in the configuration.
    """
    conv = conversation_lib.conv_llava_llama_2.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        source = source['conversations']
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    add_extra_token = cfg.get("add_extra_token")

    # Tokenize conversations
    tokens = tokenize(
        texts=conversations,
        tokenizer=tokenizer,
        context_length=cfg.get("context_length"),
        add_extra_token=add_extra_token,
    )

    # llama tricks
    tokens[tokens == 32003] = 0  # DEFAULT_IMAGE_PATCH_TOKEN
    tokens[tokens == 32006] = 1  # <s>
    tokens[tokens == 32007] = 2  # </s>
    labels = tokens.clone().detach()

    # Mask labels
    sep = "[/INST] "
    for conversation, target in zip(conversations, labels):
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        for i, rou in enumerate(rounds):

            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer.text_to_ids(rou + conv.sep2))
            instruction_len = len(tokenizer.text_to_ids(parts[0])) - 2
            if i > 0:
                round_len -= 1  # Remove extra token added by sp tokenizer
            else:
                instruction_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    # Check if masking working correctly
    # print([x for x in zip(tokens[0].numpy().tolist(), labels[0].numpy().tolist())])

    if add_extra_token:
        tokens = tokens[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
    else:
        labels = torch.roll(labels, shifts=-1, dims=-1)
        labels[:, -1] = IGNORE_INDEX

    return dict(
        tokens=tokens,
        labels=labels,
    )


def preprocess_v1(
    sources: dict,
    tokenizer,
    cfg,
) -> Dict:
    """
    Preprocesses sources for the Vicuna V1 model configuration.

    Similar to `preprocess_llama_2`, this function applies prompt templates and performs tokenization, but it is tailored
    for the Vicuna V1 model. It includes specific handling for token translations, label masking, and tokenizer configuration.

    Parameters:
    - sources (dict): A dictionary of sources containing conversations to be processed.
    - tokenizer: The tokenizer to be used for processing the text.
    - cfg: Configuration settings for preprocessing, which may include context length and additional tokens.

    Returns:
    - Dict: A dictionary containing the processed data, including tokens and labels, formatted for the Vicuna V1 model.
    """
    conv = conversation_lib.conv_vicuna_v1.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        source = source['conversations']
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    add_extra_token = cfg.get("add_extra_token")
    # Tokenize conversations
    tokens = tokenize(
        texts=conversations,
        tokenizer=tokenizer,
        context_length=cfg.get("context_length"),
        add_extra_token=add_extra_token,
    )

    # llama tricks
    tokens[tokens == 32003] = 0  # DEFAULT_IMAGE_PATCH_TOKEN
    tokens[tokens == 32006] = 1  # <s>
    tokens[tokens == 32007] = 2  # </s>
    labels = tokens.clone().detach()

    # Mask labels
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, labels):

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer.text_to_ids(rou + conv.sep2))
            instruction_len = len(tokenizer.text_to_ids(parts[0])) - 1
            if i > 0:
                round_len -= 1  # Remove extra token added by sp tokenizer
                instruction_len -= 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    if add_extra_token:
        tokens = tokens[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
    else:
        labels = torch.roll(labels, shifts=-1, dims=-1)
        labels[:, -1] = IGNORE_INDEX

    return dict(
        tokens=tokens,
        labels=labels,
    )


def preprocess_nvgpt(
    sources: dict,
    tokenizer,
    cfg,
) -> Dict:
    """
    Preprocess a given set of conversational sources using nvgpt conversation template

    This function processes conversations by first ensuring the conversation starts with a 'human' role, then tokenizes the conversations, applies specific token replacements, and finally masks labels for training purposes.

    Parameters:
    - sources: A dictionary containing conversational data. Expected format is a dict of conversations, where each conversation is a list of messages, and each message is a dict with 'from' (role) and 'value' (message text).
    - tokenizer: A tokenizer from the Hugging Face Transformers library used for tokenizing the conversations.
    - cfg: Configuration settings which include 'add_extra_token' (bool) to determine if an extra token should be added to the tokenized output, and 'context_length' for specifying the tokenization context length.

    Returns:
    - Dict: A dictionary containing two keys:
        - 'tokens': A tensor of tokenized conversation data.
        - 'labels': A tensor of labels for the conversation data, used for training models. Labels are masked based on the conversation structure.

    Note:
    - The function includes specific token replacements (e.g., DEFAULT_IMAGE_PATCH_TOKEN, <s>, </s>) and masking techniques for labels.
    - It is designed to work with conversational data where messages alternate between a 'human' and a 'gpt' role.
    - The function asserts that each message in a conversation alternates between the defined roles and skips messages not starting with the 'human' role.
    """

    """<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n<extra_id_1>User\n{user input}\n<extra_id_1>Assistant\n<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:4\n"""

    conv = conversation_lib.conv_nvgpt.copy()

    # Apply prompt templates
    conversations = []
    for source in sources:
        conv.messages = []
        conv.system = source.get('system', conv.system)

        strip_end_for_inference = False
        for i, turn in enumerate(source['conversations']):

            if i % 2 == 1:
                turn['from'] = conv.roles[1]
                if 'label' not in turn:
                    turn['label'] = (
                        "quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:4"
                    )
                value = DEFAULT_LABELS_TOKEN + turn['label'] + '\n' + turn['value']
                conv.append_message(turn['from'], value)
                if not turn["value"]:
                    strip_end_for_inference = (
                        True  # in inference, current turn is empty, thus end tokens need to striped.
                    )
            else:
                turn['from'] = conv.roles[0]
                conv.append_message(turn['from'], turn['value'])
        context = conv.get_prompt()
        if strip_end_for_inference:
            context = context.rstrip("\n<extra_id_1>") + "\n"
        conversations.append(context)

    add_extra_token = cfg.get("add_extra_token")
    # Tokenize conversations
    tokens = tokenize(
        texts=conversations,
        tokenizer=tokenizer,
        context_length=cfg.get("context_length"),
        add_extra_token=add_extra_token,
    )

    labels = tokens.clone().detach()

    # Mask targets
    sep = conv.sep + conv.roles[1] + "\n"
    labels_str_regexp = re.compile(f"{DEFAULT_LABELS_TOKEN}quality:.*\n")
    for conversation, target in zip(conversations, labels):
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt

        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt

        cur_len = 0
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break

            # Match the pattern
            match = labels_str_regexp.search(parts[1])
            labels_str = match.group() if match else ""

            instruction_len = len(tokenizer.text_to_ids(parts[0] + sep + labels_str))
            round_len = len(tokenizer.text_to_ids(rou + conv.sep))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    if add_extra_token:
        tokens = tokens[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
    else:
        labels = torch.roll(labels, shifts=-1, dims=-1)
        labels[:, -1] = IGNORE_INDEX

    return dict(
        tokens=tokens,
        labels=labels,
    )


def preprocess_nv_dpo(
    sources: dict,
    tokenizer,
    cfg,
) -> Dict:
    """
    Preprocess a given set of conversational sources using nvgpt conversation template

    This function processes conversations by first ensuring the conversation starts with a 'human' role, then tokenizes the conversations, applies specific token replacements, and finally masks labels for training purposes.

    Parameters:
    - sources: A dictionary containing conversational data. Expected format is a dict of conversations, where each conversation is a list of messages, and each message is a dict with 'from' (role) and 'value' (message text).
    - tokenizer: A tokenizer from the Hugging Face Transformers library used for tokenizing the conversations.
    - cfg: Configuration settings which include 'add_extra_token' (bool) to determine if an extra token should be added to the tokenized output, and 'context_length' for specifying the tokenization context length.

    Returns:
    - Dict: A dictionary containing two keys:
        - 'tokens': A tensor of tokenized conversation data.
        - 'labels': A tensor of labels for the conversation data, used for training models. Labels are masked based on the conversation structure.

    Note:
    - The function includes specific token replacements (e.g., DEFAULT_IMAGE_PATCH_TOKEN, <s>, </s>) and masking techniques for labels.
    - It is designed to work with conversational data where messages alternate between a 'human' and a 'gpt' role.
    - The function asserts that each message in a conversation alternates between the defined roles and skips messages not starting with the 'human' role.
    """

    """<extra_id_0>System\n\n<extra_id_1>User\n{user input}\n<extra_id_1>Assistant\n"""

    conv = conversation_lib.conv_nv_dpo.copy()

    # Apply prompt templates
    conversations = []
    for source in sources:
        conv.messages = []
        conv.system = source.get('system', conv.system)

        strip_end_for_inference = False
        for i, turn in enumerate(source['conversations']):

            if i % 2 == 1:
                turn['from'] = conv.roles[1]
                if "label" in turn:
                    value = DEFAULT_LABELS_TOKEN + turn['label'] + '\n' + turn['value']
                else:
                    value = turn["value"]
                conv.append_message(turn['from'], value)
                if not turn["value"]:
                    strip_end_for_inference = (
                        True  # in inference, current turn is empty, thus end tokens need to striped.
                    )
            else:
                turn['from'] = conv.roles[0]
                conv.append_message(turn['from'], turn['value'])
        context = conv.get_prompt()
        if strip_end_for_inference:
            if context.endswith("\n<extra_id_1>"):
                context = context[: -len("\n<extra_id_1>")] + "\n"
        conversations.append(context)

    add_extra_token = cfg.get("add_extra_token")
    # Tokenize conversations
    tokens = tokenize(
        texts=conversations,
        tokenizer=tokenizer,
        context_length=cfg.get("context_length"),
        add_extra_token=add_extra_token,
    )

    labels = tokens.clone().detach()

    # Mask targets
    sep = conv.sep + conv.roles[1] + "\n"
    for conversation, target in zip(conversations, labels):
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt

        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt

        cur_len = 0
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break

            # handle label if exists
            labels_match = re.search(rf"{re.escape(DEFAULT_LABELS_TOKEN)}.*?\n", parts[1])
            instruction_len = len(
                tokenizer.text_to_ids(parts[0] + sep + (parts[1][: labels_match.end()] if labels_match else ""))
            )
            round_len = len(tokenizer.text_to_ids(rou + conv.sep))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    # Check if masking working correctly
    # print([x for x in zip(tokens[0].numpy().tolist(), labels[0].numpy().tolist())])

    if add_extra_token:
        tokens = tokens[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
    else:
        labels = torch.roll(labels, shifts=-1, dims=-1)
        labels[:, -1] = IGNORE_INDEX

    return dict(
        tokens=tokens,
        labels=labels,
    )


def preprocess_plain(
    sources,
    tokenizer,
    cfg,
) -> Dict:
    """
    Preprocesses plain text sources (no template) for tokenization and label generation.

    This function concatenates conversations with an end signal, tokenizes them, and prepares labels for training.
    It handles sources with a specific structure (expecting two elements in 'conversations') and includes the
    option to add an extra token as specified in the configuration. The function also applies masking to the labels.

    Parameters:
    - sources: A list of source dictionaries. Each source dictionary should have a key 'conversations'
      containing a list of conversation parts.
    - tokenizer: The tokenizer to be used for converting text to tokens.
    - cfg: Configuration dictionary which may include 'context_length' and 'add_extra_token' settings.

    Returns:
    - Dict: A dictionary containing tokenized data and corresponding labels. This includes 'tokens' which are the
      tokenized representations of the conversations, and 'labels' which are used for training the model. The labels
      have specific indices masked with IGNORE_INDEX as per the preprocessing logic.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        source = source['conversations']
        assert len(source) == 2
        # This line is different from LLaVA repo, we inserted '\n' after <image>.
        conversation = source[0]['value'] + source[1]['value'] + '\n'
        conversations.append(conversation)
    # tokenize conversations
    add_extra_token = cfg.get("add_extra_token")
    tokens = tokenize(
        texts=conversations,
        tokenizer=tokenizer,
        context_length=cfg.get("context_length"),
        add_extra_token=add_extra_token,
    )
    labels = tokens.clone().detach()
    for target, source in zip(labels, sources):
        source = source['conversations']
        tokenized_len = len(tokenizer.text_to_ids(source[0]['value']))
        target[:tokenized_len] = IGNORE_INDEX

    if add_extra_token:
        tokens = tokens[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
    else:
        labels = torch.roll(labels, shifts=-1, dims=-1)
        labels[:, -1] = IGNORE_INDEX

    return dict(
        tokens=tokens,
        labels=labels,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, multimodal_cfg: dict, data_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        if data_path is not None:
            with open(data_path, "r") as file:
                list_data_dict = json.load(file)
        else:
            list_data_dict = []

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        self.conv_template = multimodal_cfg["conv_template"]
        self.image_folder = multimodal_cfg['image_folder']
        self.video_folder = multimodal_cfg['video_folder']
        self.processor = multimodal_cfg["image_processor"]

        self.image_loader = TarOrFolderImageLoader(self.image_folder) if self.image_folder else None
        self.video_loader = TarOrFolderVideoLoader(self.video_folder, data_cfg) if self.video_folder else None

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            if not isinstance(self.list_data_dict[i]['image'], list):
                self.list_data_dict[i]['image'] = [self.list_data_dict[i]['image']]

            images = []
            for image_file in self.list_data_dict[i]['image']:
                image = self.image_loader.open_image(image_file)
                if image is None:
                    logging.warning(f"Image {image_file} could not be found!")
                image = process_image(self.processor, image, self.multimodal_cfg['image_aspect_ratio'])
                images.append(image)
            media_tensors = torch.tensor([])
            if images:
                media_tensors = torch.stack(images)
                patch_dim = self.multimodal_cfg['patch_dim']

                height_num_patches = media_tensors[0].shape[1] // patch_dim
                width_num_patches = media_tensors[0].shape[2] // patch_dim

                if self.multimodal_cfg['mm_mlp_adapter_type'] == 'mlp_downsample':
                    if height_num_patches % 2 != 0:
                        height_num_patches += 1
                    if width_num_patches % 2 != 0:
                        width_num_patches += 1

                cur_token_len = height_num_patches * width_num_patches

                sources = preprocess_multimodal(
                    copy.deepcopy(sources),
                    self.multimodal_cfg,
                    cur_token_len,
                    use_plain=(self.conv_template == "plain"),
                )
        elif 'video' in sources[0]:
            if not isinstance(self.list_data_dict[i]['video'], list):
                self.list_data_dict[i]['video'] = [self.list_data_dict[i]['video']]

            videos = []
            for video_file in self.list_data_dict[i]['video']:
                frames = self.video_loader.open_video(video_file)
                if frames is None:
                    logging.warning(f"Video {video_file} could not be found!")
                if isinstance(self.processor, CLIPImageProcessor):
                    # image processor from HF
                    if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                        max_hw, min_hw = max(frames.size), min(frames.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 448, 224
                        shortest_edge = int(min(max_len / aspect_ratio, min_len))
                        frames = self.processor.preprocess(
                            frames, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge}
                        )['pixel_values']
                    elif self.multimodal_cfg['image_aspect_ratio'] == 'pad':

                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        frames = expand2square(frames, tuple(int(x * 255) for x in self.processor.image_mean))
                        frames = self.processor.preprocess(frames, return_tensors='pt')['pixel_values']
                    else:
                        frames = self.processor.preprocess(frames, return_tensors='pt')['pixel_values']
                else:
                    assert (
                        self.multimodal_cfg['image_aspect_ratio'] == 'square'
                    ), 'NeMo image transform with setting `image_aspect_ratio` to `square`.'
                    frames = self.processor(frames)
                videos.append(frames)
            media_tensors = frames
            if videos:
                media_tensors = torch.stack(videos)
                patch_dim = self.multimodal_cfg['patch_dim']

                height_num_patches = media_tensors[0].shape[-2] // patch_dim
                width_num_patches = media_tensors[0].shape[-1] // patch_dim

                if self.multimodal_cfg['mm_mlp_adapter_type'] == 'mlp_downsample':
                    if height_num_patches % 2 != 0:
                        height_num_patches += 1
                    if width_num_patches % 2 != 0:
                        width_num_patches += 1

                cur_token_len = height_num_patches * width_num_patches

                sources = preprocess_multimodal(
                    copy.deepcopy(sources),
                    self.multimodal_cfg,
                    cur_token_len,
                    use_plain=(self.conv_template == "plain"),
                )

        else:
            media_tensors = torch.tensor([])
            sources = copy.deepcopy(sources)

        if self.conv_template in ["nvgpt", "nv_steerlm"]:
            data_dict = preprocess_nvgpt(
                sources,
                self.tokenizer,
                self.multimodal_cfg,
            )
        elif self.conv_template == "nv_dpo":
            data_dict = preprocess_nv_dpo(
                sources,
                self.tokenizer,
                self.multimodal_cfg,
            )
        elif self.conv_template == "v1":
            data_dict = preprocess_v1(
                sources,
                self.tokenizer,
                self.multimodal_cfg,
            )
        elif self.conv_template == "llama_2":
            data_dict = preprocess_llama_2(
                sources,
                self.tokenizer,
                self.multimodal_cfg,
            )
        elif self.conv_template == "llama_3":
            data_dict = preprocess_llama_3(
                sources,
                self.tokenizer,
                self.multimodal_cfg,
            )
        elif self.conv_template == "plain":
            data_dict = preprocess_plain(
                sources,
                self.tokenizer,
                self.multimodal_cfg,
            )
        else:
            raise ValueError(f"Conversation template `{self.conv_template}` is not supported in Neva now.")

        if isinstance(i, int):
            data_dict = dict(tokens=data_dict["tokens"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if self.multimodal_cfg['is_multimodal']:
            if isinstance(self.processor, CLIPImageProcessor):
                crop_size = [self.processor.crop_size['height'], self.processor.crop_size['width']]
            else:
                crop_size = self.multimodal_cfg['crop_size']

            # Image does not exist in the data, but the model is multimodal
            # TODO, if there are different videos on T dimensions.
            if media_tensors.shape[0] < MAX_NUM_IMAGES:
                padding_size = MAX_NUM_IMAGES - media_tensors.shape[0]
                zero_padding = torch.zeros((padding_size, 3, crop_size[0], crop_size[1]), dtype=torch.float)
                media_tensors = torch.cat((media_tensors, zero_padding), dim=0)

            if self.multimodal_cfg['media_type'] == 'image':
                data_dict['image'] = media_tensors
            elif self.multimodal_cfg['media_type'] == 'video':
                data_dict['video'] = media_tensors

        return data_dict


class NevaDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, multimodal_cfg: dict, data_cfg: dict):

        if data_path.endswith(".json"):
            super(NevaDataset, self).__init__(data_path, tokenizer, multimodal_cfg, data_cfg)

        elif data_path.endswith(".jsonl"):
            super(NevaDataset, self).__init__(None, tokenizer, multimodal_cfg, data_cfg)
            logging.warning("Loading image inputs from SteerLM Dataset")
            if multimodal_cfg['media_type'] == 'image':
                image_folder = multimodal_cfg['image_folder']
                for line in open(data_path, "r"):
                    record = json.loads(line)

                    # This currently supports only a single image
                    # search for <img src="/absolute/path/to/image" in the conversation
                    #   add it as record['image'], remove src tag from the <img> tag

                    record['image'] = []
                    for turn in record['conversations']:
                        matches = re.finditer('<img src="([^"]+)"', turn['value'])
                        for match in matches:
                            image_name = match.group(1).split("/")[-1]
                            image_path = os.path.join(image_folder, image_name)
                            if not os.path.isfile(image_path):
                                logging.warning(f"Image not found: {image_path}")
                                continue
                            record['image'].append(image_name)  # url
                        turn['value'] = re.sub('<img src="([^"]+)">', DEFAULT_IMAGE_TOKEN, turn['value'])

                    self.list_data_dict.append(record)

        else:
            raise ValueError(f"Formatting of {data_path} is not supported in Neva.")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    model_cfg: DictConfig
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        packed_sequence = "cu_seqlens" in instances[0]
        max_len = max(instance['tokens'].shape[0] for instance in instances)
        max_len = (max_len - 1) // 64 * 64 + 64
        for instance in instances:
            pad_len = max_len - instance['tokens'].shape[0]
            instance['tokens'] = F.pad(instance['tokens'], (0, pad_len), 'constant', 0)
            instance['labels'] = F.pad(instance['labels'], (0, pad_len), 'constant', -1)
            if packed_sequence and instance["cu_seqlens"][-1] != max_len:
                instance["cu_seqlens"] = torch.cat((instance["cu_seqlens"], torch.IntTensor([max_len])), 0)

        if packed_sequence:
            max_len_cu = max(instance['cu_seqlens'].shape[0] for instance in instances)
            max_len_image = max(instance['image'].shape[0] for instance in instances)
            for instance in instances:
                pad_len_cu = max_len_cu - instance['cu_seqlens'].shape[0]
                instance['cu_seqlens'] = F.pad(instance['cu_seqlens'], (0, pad_len_cu), 'constant', max_len)

                x = instance['image']
                num_pad = max_len_image - x.shape[0]
                pad_tensor = torch.zeros(num_pad, *x.shape[1:], dtype=x.dtype, device=x.device)
                instance['image'] = torch.cat((x, pad_tensor), dim=0)

        batch = default_collate(instances)
        tokenizer = self.tokenizer
        model_cfg = self.model_cfg

        tokens = batch['tokens']
        labels = batch['labels']
        media_type = model_cfg.data.get('media_type', 'image')
        if media_type == 'image':
            media = batch.get('image')
        elif media_type == 'video':
            media = batch.get('video')
        else:
            raise ValueError(f"Unsupported media type {media_type}")

        if packed_sequence:
            cu_seqlens = batch["cu_seqlens"]
            position_ids = []
            for cu_seqlen in cu_seqlens:
                position_ids.append([])
                for ind in range(0, len(cu_seqlen) - 1):
                    seqlen = cu_seqlen[ind + 1] - cu_seqlen[ind]
                    position_ids[-1].extend(list(range(seqlen)))
            position_ids = torch.LongTensor(position_ids)
            loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
            attention_mask = torch.ones(tokens.size(), dtype=torch.long, device=tokens.device)
        else:
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=tokenizer.eos_id,
                eod_mask_loss=model_cfg.data.get("eod_mask_loss", False),
                reset_attention_mask=False,
                reset_position_ids=False,
            )

        loss_mask[labels == -1] = 0.0
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        if media is None:
            raise NotImplementedError
        else:
            if media_type == 'image':
                media = rearrange(media, "b T c h w -> b T 1 c h w")
            elif media_type == 'video':
                media = rearrange(media, "b T F c h w -> b T F c h w")

        batch = {
            'tokens': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'media': media,
        }
        if packed_sequence:
            batch["cu_seqlens"] = cu_seqlens
        return batch


def make_supervised_data_module(tokenizer, image_processor, model_cfg) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_cfg = model_cfg.data
    mm_cfg = model_cfg.mm_cfg
    add_extra_token = 1
    if getattr(model_cfg, 'no_seqlen_plus_one_input_tokens', False):
        add_extra_token = 0
    crop_size = mm_cfg.vision_encoder.get("crop_size", (224, 224))

    train_dataset = NevaDataset(
        tokenizer=tokenizer,
        data_path=data_cfg.data_path,
        multimodal_cfg=dict(
            is_multimodal=data_cfg.is_multimodal,
            sep_image_conv_front=data_cfg.sep_image_conv_front,
            model_type=mm_cfg.llm.get("model_type", "nvgpt"),
            conv_template=data_cfg.get("conv_template", "nvgpt"),
            patch_dim=model_cfg.mm_cfg.vision_encoder.patch_dim,
            crop_size=crop_size,
            image_folder=data_cfg.get('image_folder', None),
            video_folder=data_cfg.get('video_folder', None),
            image_aspect_ratio=data_cfg.image_aspect_ratio,
            use_im_start_end=getattr(model_cfg.mm_cfg, 'use_im_start_end', False),
            image_processor=image_processor,
            add_extra_token=add_extra_token,
            context_length=model_cfg.encoder_seq_length,
            media_type=data_cfg.get('media_type', 'image'),
            num_frames=data_cfg.get('num_frames', -1),
            mm_mlp_adapter_type=model_cfg.mm_cfg.get('mm_mlp_adapter_type', 'linear'),
        ),
        data_cfg=dict(
            splice_single_frame=data_cfg.get('splice_single_frame', None),
            num_frames=data_cfg.get('num_frames', -1),
            sep_token_between_frames=data_cfg.get('sep_token_between_frames', False),
        ),
    )

    return dict(train_dataset=train_dataset, eval_dataset=train_dataset)


class NevaPackedSeqDatatset(Dataset):
    def __init__(self, data_path: str, crop_size: Tuple[int, int] = (224, 224)):
        self.ds = IndexedDataset(data_path)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.ds.document_indices) - 1

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        doc_start = self.ds.document_indices[i]
        batch = {
            "cu_seqlens": torch.IntTensor(self.ds[doc_start]),
            "tokens": torch.LongTensor(self.ds[doc_start + 1]),
            "labels": torch.LongTensor(self.ds[doc_start + 2]),
            "image": torch.FloatTensor(self.ds[doc_start + 3]).reshape(-1, 3, *self.crop_size),
        }

        return batch
