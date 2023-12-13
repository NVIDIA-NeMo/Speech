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
from typing import Any, Dict, List, Sequence, Union

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset, default_collate
from transformers import CLIPImageProcessor

import nemo.collections.multimodal.data.neva.conversation as conversation_lib
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

MAX_NUM_IMAGES = 4
IGNORE_INDEX = -1
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_BOS_TOKEN = "<extra_id_6>"
DEFAULT_EOS_TOKEN = "<extra_id_7>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SYSTEM_TOKEN = "<extra_id_0>"
DEFAULT_SEPARATOR_TOKEN = "<extra_id_1>"
DEFAULT_LABELS_TOKEN = "<extra_id_2>"
DEFAULT_IMAGE_PATCH_TOKEN = "<extra_id_3>"
DEFAULT_IM_START_TOKEN = "<extra_id_4>"
DEFAULT_IM_END_TOKEN = "<extra_id_5>"


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


def tokenize(
    texts: Union[str, List[str]], tokenizer: Any, context_length: int, add_extra_token: int,
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
    tokens = tokenizer.text_to_ids(texts)
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


def preprocess_multimodal(sources: dict, multimodal_cfg: dict, cur_token_len: int,) -> Dict:
    """
    Preprocesses a given multimodal input based on specified configurations.

    This function modifies the 'sources' dictionary, primarily focusing on conversations. It checks if the input
    is multimodal based on 'multimodal_cfg'. If not, it returns the 'sources' unmodified. For multimodal inputs,
    it processes each conversation in 'sources'. If 'sep_image_conv_front' is set in 'multimodal_cfg', the function
    asserts the presence of 'DEFAULT_IMAGE_TOKEN' at the beginning of each conversation, removes it, and restructures
    the conversation's first turn with this token and other formatting details. Furthermore, the function replaces
    'DEFAULT_IMAGE_TOKEN' with a series of 'DEFAULT_IMAGE_PATCH_TOKEN' tokens, the count of which depends on
    'image_token_len' and 'use_im_start_end' configuration.

    Parameters:
    - sources (dict): A dictionary containing the source data to be processed. Each source is expected to have
      'conversations' as one of its keys.
    - multimodal_cfg (dict): A configuration dictionary specifying how the multimodal data should be processed.
      Key configurations include 'is_multimodal', 'sep_image_conv_front', and 'use_im_start_end'.
    - cur_token_len (int): The current length of image tokens, used to determine the number of patch tokens
      to replace the 'DEFAULT_IMAGE_TOKEN'.

    Returns:
    - dict: The modified 'sources' dictionary after applying the multimodal preprocessing.
    """
    is_multimodal = multimodal_cfg['is_multimodal']
    image_token_len = cur_token_len
    if not is_multimodal:
        return sources

    for source in sources:
        conversation = source['conversations']
        if multimodal_cfg['sep_image_conv_front']:
            assert DEFAULT_IMAGE_TOKEN in conversation[0]['value']
            conversation[0]['value'] = conversation[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            conversation[0]['value'] = (
                DEFAULT_IMAGE_TOKEN
                + conversation_lib.default_conversation.sep
                + conversation_lib.default_conversation.roles[0]
                + ": "
                + conversation[0]['value']
            )
        for turn in conversation:
            if multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            else:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * (image_token_len - 2)
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            turn["value"] = turn["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(sources: dict, tokenizer, cfg,) -> Dict:
    """
    Preprocess a given set of conversational sources using llama_2 chat conversation template

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
            if i > 0:
                round_len -= 1  # Remove extra token added by sp tokenizer
            instruction_len = len(tokenizer.text_to_ids(parts[0])) - 1
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

    return dict(tokens=tokens, labels=labels,)


def preprocess_v1(sources: dict, tokenizer, cfg,) -> Dict:
    """
    Preprocess a given set of conversational sources using vicuna v1 conversation template

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

    return dict(tokens=tokens, labels=labels,)


def preprocess_nvgpt(sources: dict, tokenizer, cfg,) -> Dict:
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

    conv = conversation_lib.conv_nvgpt.copy()

    # Apply prompt templates
    conversations = []
    for source in sources:
        conv.messages = []
        conv.system = source.get('system', conv.system)
        if len(source['conversations']) >= 2:
            conv.roles = (source['conversations'][0]['from'], source['conversations'][1]['from'])

        strip_end_for_inference = False
        for turn in source['conversations']:
            if 'label' in turn:
                value = DEFAULT_LABELS_TOKEN + turn['label'] + '\n' + turn['value']
                conv.append_message(turn['from'], value)
                if not turn["value"]:
                    strip_end_for_inference = (
                        True  # in inference, current turn is empty, thus end tokens need to striped.
                    )
            else:
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

    return dict(tokens=tokens, labels=labels,)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if data_path is not None:
            logging.warning("Loading data...")
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
        self.processor = multimodal_cfg["image_processor"]

        self.image_loader = TarOrFolderImageLoader(self.image_folder)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        processor = self.processor
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
                if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 448, 224
                    shortest_edge = int(min(max_len / aspect_ratio, min_len))
                    image = processor.preprocess(
                        image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge}
                    )['pixel_values'][0]
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

                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images.append(image)
            images_tensors = torch.tensor([])
            if images:
                images_tensors = torch.stack(images)
                cur_token_len = (images_tensors[0].shape[1] // 14) * (
                    images_tensors[0].shape[2] // 14
                )  # FIXME: 14 is hardcoded patch size
                sources = preprocess_multimodal(copy.deepcopy(sources), self.multimodal_cfg, cur_token_len)
        else:
            images_tensors = torch.tensor([])
            sources = copy.deepcopy(sources)

        if self.conv_template == "nvgpt":
            data_dict = preprocess_nvgpt(sources, self.tokenizer, self.multimodal_cfg,)
        elif self.conv_template == "v1":
            data_dict = preprocess_v1(sources, self.tokenizer, self.multimodal_cfg,)
        elif self.conv_template == "llama_2":
            data_dict = preprocess_llama_2(sources, self.tokenizer, self.multimodal_cfg,)
        else:
            raise ValueError(f"Conversation template `{self.conv_template}` is not supported in Neva now.")

        if isinstance(i, int):
            data_dict = dict(tokens=data_dict["tokens"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if self.multimodal_cfg['is_multimodal']:
            crop_size = self.processor.crop_size
            # image does not exist in the data, but the model is multimodal
            zero_padding = torch.zeros(
                (MAX_NUM_IMAGES - len(images_tensors), 3, crop_size['height'], crop_size['width']), dtype=torch.float
            )
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
            data_dict['image'] = images_tensors
        return data_dict


class NevaDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, multimodal_cfg: dict):

        if data_path.endswith(".json"):
            super(NevaDataset, self).__init__(data_path, tokenizer, multimodal_cfg)

        elif data_path.endswith(".jsonl"):
            super(NevaDataset, self).__init__(None, tokenizer, multimodal_cfg)
            logging.warning("Loading image inputs from SteerLM Dataset")
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
        max_len = max(instance['tokens'].shape[0] for instance in instances)
        max_len = (max_len - 1) // 4 * 4 + 4
        for instance in instances:
            pad_len = max_len - instance['tokens'].shape[0]
            instance['tokens'] = F.pad(instance['tokens'], (0, pad_len), 'constant', 0)
            instance['labels'] = F.pad(instance['labels'], (0, pad_len), 'constant', -1)

        batch = default_collate(instances)
        tokenizer = self.tokenizer
        model_cfg = self.model_cfg

        tokens = batch['tokens']
        labels = batch['labels']
        media = batch.get('image')

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
            media = rearrange(media, "b T c h w -> b T 1 c h w")

        batch = {
            'tokens': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'media': media,
        }
        return batch


def make_supervised_data_module(tokenizer, model_cfg) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_cfg = model_cfg.data
    mm_cfg = model_cfg.mm_cfg
    add_extra_token = 1
    if getattr(model_cfg, 'no_seqlen_plus_one_input_tokens', False):
        add_extra_token = 0
    if mm_cfg.vision_encoder.from_hf:
        image_processor = CLIPImageProcessor.from_pretrained(
            mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16
        )
    else:
        # TODO(yuya): Fix this hard-code for our own CLIP
        image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16
        )
    train_dataset = NevaDataset(
        tokenizer=tokenizer,
        data_path=data_cfg.data_path,
        multimodal_cfg=dict(
            is_multimodal=data_cfg.is_multimodal,
            sep_image_conv_front=data_cfg.sep_image_conv_front,
            conv_template=data_cfg.get("conv_template", "nvgpt"),
            image_token_len=data_cfg.image_token_len,
            image_folder=data_cfg.image_folder,
            image_aspect_ratio=data_cfg.image_aspect_ratio,
            use_im_start_end=getattr(model_cfg.mm_cfg, 'use_im_start_end', False),
            image_processor=image_processor,
            add_extra_token=add_extra_token,
            context_length=model_cfg.encoder_seq_length,
        ),
    )
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=train_dataset)
