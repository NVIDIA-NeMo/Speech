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

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    pass

import copy

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from qwen_vl_utils import fetch_image, fetch_video
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, Qwen2VLImageProcessor

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.vlm.qwen2vl.data.config import Qwen2VLDataConfig
from nemo.collections.vlm.qwen2vl.data.conversation import conv_templates as supported_conv_templates
from nemo.collections.vlm.qwen2vl.data.multimodal_tokens import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SPECIAL_TOKEN_MAP,
    VIDEO_TOKEN_INDEX,
    VISION_END_TOKEN_INDEX,
)
from nemo.lightning.pytorch.plugins import MegatronDataSampler


def process_vision(processor, images, videos, fps=None, model_version="qwen2-vl"):
    # pylint: disable=C0115,C0116
    assert isinstance(processor, Qwen2VLImageProcessor), "processor needs to be Qwen2VLImageProcessor"
    if images is not None:
        image_inputs = processor(images=images, videos=None, return_tensors='pt')
        image_grid_thw = image_inputs["image_grid_thw"]
    else:
        image_inputs = {}
        image_grid_thw = None

    if videos is not None:
        videos_inputs = processor(images=None, videos=videos, return_tensors='pt')
        video_grid_thw = videos_inputs["video_grid_thw"]
        if model_version == "qwen25-vl":
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length "
                    f"of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            second_per_grid_ts = torch.tensor(
                second_per_grid_ts,
                dtype=videos_inputs['pixel_values_videos'].dtype,
                device=videos_inputs['pixel_values_videos'].device,
            )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})
    else:
        videos_inputs = {}
        video_grid_thw = None

    return {
        "image_inputs": image_inputs,
        "image_grid_thw": image_grid_thw,
        "video_inputs": videos_inputs,
        "video_grid_thw": video_grid_thw,
    }


def tokenize_special_token(prompt, tokenizer, vision_tensors, merge_length=2, special_token_map=None):
    """
    Tokenizes a given prompt with special handling for multiple special tokens.

    This function splits the prompt at special tokens, tokenizes each chunk separately,
    and then reassembles the chunks with the corresponding special token inserted in place of the placeholders.

    Parameters:
    prompt (str): The input prompt containing text and special token placeholders.
    tokenizer: The tokenizer object used to tokenize the prompt chunks.
    special_token_map (list, optional): A list containing tuples of special token strings
                                        and their corresponding token indices. Defaults to SPECIAL_TOKEN_MAP.

    Returns:
    torch.Tensor: A tensor of token IDs representing the tokenized prompt with special tokens.
    """
    if vision_tensors['image_grid_thw'] is not None:
        image_token_length = vision_tensors['image_grid_thw'].prod(dim=1) // (merge_length**2)
    else:
        image_token_length = 0

    if vision_tensors['video_grid_thw'] is not None:
        video_token_length = vision_tensors['video_grid_thw'].prod(dim=1) // (merge_length**2)
    else:
        video_token_length = 0
    # Use the default special token map if none is provided
    if special_token_map is None:
        special_token_map = SPECIAL_TOKEN_MAP

    # Create a mapping of special tokens to their indices
    special_token_dict = {token: index for token, index in special_token_map}

    # Split the prompt into chunks and track special tokens
    regex_pattern = '(' + '|'.join(re.escape(token) for token in special_token_dict.keys()) + ')'
    chunks = re.split(regex_pattern, prompt)

    # Tokenize each chunk and replace special tokens with their indices
    tokenized_chunks = []
    image_index = 0
    video_index = 0
    for chunk in chunks:
        if chunk in special_token_dict and chunk == "<|image_pad|>":
            tokenized_chunks.extend([special_token_dict[chunk]] * image_token_length[image_index])
            image_index += 1
        elif chunk in special_token_dict and chunk == "<|video_pad|>":
            tokenized_chunks.extend([special_token_dict[chunk]] * video_token_length[video_index])
            video_index += 1
        elif chunk in special_token_dict:
            tokenized_chunks.append(special_token_dict[chunk])
        elif len(chunk) > 0:
            tokenized_chunk = tokenizer(chunk, add_special_tokens=False)
            tokenized_chunks.extend(tokenized_chunk.input_ids)

    assert vision_tensors["image_grid_thw"] is None or image_index == len(
        vision_tensors['image_grid_thw']
    ), f"{image_index=} != {len(vision_tensors['image_grid_thw'])=}"
    assert vision_tensors["video_grid_thw"] is None or video_index == len(
        vision_tensors['video_grid_thw']
    ), f"{video_index=} != {len(vision_tensors['video_grid_thw'])=}"
    return tokenized_chunks


def find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    # pylint: disable=C0115,C0116
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = torch.tensor([template[i + j] == pattern[j] for j in range(pattern_len)])
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int):
    """
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    if new_target_len == 0 and new_source_len > 0:
        new_target_len += 1
        new_source_len -= 1
    return new_source_len, new_target_len


def extract_dialogue_pairs(tokens, decoded_tokens):
    """
    Extract user-assistant dialogue pairs from a sequence of tokens.

    Args:
        tokens: List of token ids
        decoded_tokens: List of decoded tokens (strings)

    Returns:
        List of dialogue pairs, where each pair is [user_tokens, assistant_tokens]
    """
    dialogue_pairs = []

    # State flags
    in_user = False
    in_assistant = False

    user_start_idx = -1
    user_end_idx = -1
    assistant_start_idx = -1
    assistant_end_idx = -1

    # Track the first user dialogue start position
    first_user_start = -1

    i = 0
    while i < len(decoded_tokens):
        # Detect user dialogue start
        if (
            i + 2 < len(decoded_tokens)
            and decoded_tokens[i] == '<|im_start|>'
            and decoded_tokens[i + 1] == 'user'
            and decoded_tokens[i + 2] == '\n'
        ):
            if first_user_start == -1:
                first_user_start = i
                # If this is the first user dialogue and there's content before it, include that content
                if i > 0:
                    user_start_idx = 0
                else:
                    user_start_idx = i
            else:
                user_start_idx = i

            in_user = True
            i += 3  # Skip '<|im_start|>', 'user', '\n'
            continue

        # Detect user dialogue end and assistant dialogue start
        if (
            in_user
            and i + 4 < len(decoded_tokens)
            and decoded_tokens[i] == '<|im_end|>'
            and decoded_tokens[i + 1] == '\n'
            and decoded_tokens[i + 2] == '<|im_start|>'
            and decoded_tokens[i + 3] == 'assistant'
            and decoded_tokens[i + 4] == '\n'
        ):
            user_end_idx = i + 4
            assistant_start_idx = i + 5
            in_user = False
            in_assistant = True
            i += 5  # Skip '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n'
            continue

        # Detect assistant dialogue end
        if (
            in_assistant
            and decoded_tokens[i] == '<|im_end|>'
            and i + 1 < len(decoded_tokens)
            and decoded_tokens[i + 1] == '\n'
        ):
            assistant_end_idx = i + 1
            in_assistant = False

            # Found a complete dialogue pair
            if user_start_idx != -1 and user_end_idx != -1 and assistant_start_idx != -1 and assistant_end_idx != -1:
                user_tokens = tokens[user_start_idx : user_end_idx + 1]
                assistant_tokens = tokens[assistant_start_idx : assistant_end_idx + 1]
                dialogue_pairs.append([user_tokens, assistant_tokens])

                # Reset indices for the next pair
                user_start_idx = -1
                user_end_idx = -1
                assistant_start_idx = -1
                assistant_end_idx = -1

            i += 2  # Skip '<|im_end|>', '\n'
            continue

        # Normal advancement
        i += 1
    assert len(tokens) == sum(
        len(pair[0]) + len(pair[1]) for pair in dialogue_pairs
    ), f"Tokens length mismatch: {len(tokens)} != {sum(len(pair[0]) + len(pair[1]) for pair in dialogue_pairs)}"

    return dialogue_pairs


def truncate_tokens(tokens, labels, max_sequence_length, tokenizer):
    """truncate tokens"""
    vision_token_num = len([i for i in tokens if i == VISION_END_TOKEN_INDEX])
    special_index_map = {index: token for token, index in SPECIAL_TOKEN_MAP}
    decoded_tokens = []
    for _id in tokens:
        if _id == IMAGE_TOKEN_INDEX or _id == VIDEO_TOKEN_INDEX:
            decoded_tokens.append(special_index_map[int(_id)])
        else:
            decoded_tokens.append(tokenizer.decode([_id]))
    assert len(decoded_tokens) == len(
        tokens
    ), f"Decoded tokens length mismatch: {len(decoded_tokens)} != {len(tokens)}"
    truncated_tokens = []
    truncated_labels = []
    remain_labels = labels[:]
    # Extract dialogue pairs from tokens
    dialogue_pairs = extract_dialogue_pairs(tokens, decoded_tokens)
    total_length = 0
    for n, (source_ids, target_ids) in enumerate(dialogue_pairs):

        if total_length >= max_sequence_length:
            break

        current_labels = remain_labels[: len(source_ids) + len(target_ids)]
        remain_labels = remain_labels[len(source_ids) + len(target_ids) :]
        source_labels = current_labels[: len(source_ids)]
        target_labels = current_labels[len(source_ids) :]

        # infer the length of source and target
        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), max_sequence_length - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        truncated_tokens += source_ids + target_ids

        source_labels = source_labels[:source_len]
        target_labels = target_labels[:target_len]
        truncated_labels += source_labels + target_labels

        total_length += source_len + target_len

    if len([i for i in truncated_tokens if i == VISION_END_TOKEN_INDEX]) != vision_token_num:
        raise ValueError(
            f"Image/video tokens was truncated. This will cause training to fail. "
            f"Please increase max_sequence_length {max_sequence_length=} to accommodate "
            f"the full image/video token sequence."
        )

    return torch.tensor(truncated_tokens, dtype=torch.long), torch.tensor(truncated_labels, dtype=torch.long)


class PreloadedSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        data_config,
        tokenizer,
        image_processor,
        model_version,
        sequence_length=None,
    ):
        super().__init__()
        if data_path is not None:
            with open(data_path, "r") as file:
                list_data_dict = json.load(file)
        else:
            list_data_dict = []

        logging.warning("Formatting inputs...Skip in preloaded mode")
        self.data_config = data_config
        self.tokenizer = tokenizer
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        if isinstance(self.tokenizer, AutoTokenizer):
            self.tokenizer = self.tokenizer.tokenizer

        self.image_processor = image_processor
        self.sequence_length = sequence_length

        self.conv_template = data_config.conv_template
        self.image_process_mode = data_config.image_process_mode
        self.list_data_dict = list_data_dict

        self.image_folder = getattr(data_config, "image_folder", None)
        self.video_folder = getattr(data_config, "video_folder", None) or self.image_folder
        self.model_version = model_version

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]
        # To prevent multiple threads from modifying conv at the same time
        conv = copy.deepcopy(supported_conv_templates[self.conv_template])
        chatml = self._apply_prompt_templates(conv, source, use_plain=self.conv_template == "plain")

        vision_tensors = self._process_vision(source, self.image_folder, self.video_folder, self.model_version)
        tokens, labels = self._tokenize_and_label(conv, chatml, vision_tensors)

        data_dict = dict(
            tokens=tokens,
            labels=labels,
            **vision_tensors['image_inputs'],
            **vision_tensors['video_inputs'],
        )
        return data_dict

    def _normalize_vision_paths(self, source, image_folder, video_folder):
        """
        Normalize image and video paths, converting relative paths to absolute paths.

        Args:
            source: Dictionary containing image and video paths
            image_folder: Base directory for image files
            video_folder: Base directory for video files

        Returns:
            Source dictionary with normalized image and video paths
        """

        def normalize_paths(paths, base_folder):
            """Convert relative paths to absolute paths"""
            if base_folder is None or not paths:
                return paths

            for i, path in enumerate(paths):
                # Skip non-string paths
                if not isinstance(path, str):
                    continue

                # Skip URLs and absolute paths
                if any(prefix in path for prefix in ["http:", "https:", "file:"]) or os.path.isabs(path):
                    continue

                # Convert relative path to absolute path
                paths[i] = os.path.normpath(os.path.join(base_folder, path))

            return paths

        # Get image and video paths
        images = source.get('images', [])
        videos = source.get('videos', [])

        # Process image and video paths
        images = normalize_paths(images, image_folder)
        videos = normalize_paths(videos, video_folder)

        return images, videos

    def _fetch_vision_content(self, images, videos):
        image_inputs = []
        for image in images:
            image_inputs.append(fetch_image({"image": image}))
        video_inputs = []
        video_sample_fps_list = []
        for video in videos:
            video_input, video_sample_fps = fetch_video({"video": video}, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        return image_inputs, video_inputs, video_sample_fps_list

    def _process_vision(self, source, image_folder, video_folder, model_version):
        # normalize image and video paths
        images, videos = self._normalize_vision_paths(source, image_folder, video_folder)
        # leave the I/O and smart_resize to qwen_vl_utils, which is maintained on github by Qwen Team.
        image_inputs, video_inputs, video_sample_fps_list = self._fetch_vision_content(images, videos)
        # call Huggingface processor to get patches and size info, which is maintained by Qwen Team as well.
        vision_tensors = process_vision(
            self.image_processor, image_inputs, video_inputs, video_sample_fps_list, model_version
        )
        return vision_tensors

    def _apply_prompt_templates(self, conv, source, use_plain=False):
        """
        According to https://github.com/QwenLM/Qwen2-VL#data-preparation
        [
          {
            "messages": [
              {
                "content": "<image>Who are they?",
                "role": "user"
              },
              {
                "content": "They're Kane and Gretzka from Bayern Munich.",
                "role": "assistant"
              },
              {
                "content": "What are they doing?<image>",
                "role": "user"
              },
              {
                "content": "They are celebrating on the soccer field.",
                "role": "assistant"
              }
            ],
            "images": [
              "demo_data1/1.jpg",
              "demo_data2/1.jpg"
            ]
          },
        ]
        """

        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

        # FIXME: Current implementation does not support system prompt in the data.
        # FIXME: Current implementation does not support tools in the data.
        messages = source['messages']
        if source.get('system', None) is not None:
            conv.system = source['system']

        def _fix_roles(roles):
            if len(messages) < 2:
                return roles
            return {messages[0]["role"]: conv.roles[0], messages[1]["role"]: conv.roles[1]}

        roles = _fix_roles(roles)

        conv.messages = []
        for j, sentence in enumerate(messages):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["content"])

        prompt = conv.get_prompt()

        images = source.get('images', [])
        videos = source.get('videos', [])
        assert prompt.count("<image>") == len(images), f"{prompt.count('<image>')=} != {len(images)=}"
        assert prompt.count("<video>") == len(videos), f"{prompt.count('<video>')=} != {len(videos)=}"

        image_block = "<|vision_start|><|image_pad|><|vision_end|>"
        video_block = "<|vision_start|><|video_pad|><|vision_end|>"

        prompt = prompt.replace("<image>", image_block).replace("<video>", video_block)
        return prompt

    def _tokenize_and_label(self, conv, chatml, vision_tensors):
        tokens = tokenize_special_token(
            chatml, self.tokenizer, vision_tensors, merge_length=self.image_processor.merge_size
        )

        labels = [IGNORE_INDEX for _ in range(len(tokens))]
        search_start_index = 0
        messages = conv.messages
        for i in range(len(messages)):
            role = messages[i][0]
            if role == 'assistant':
                stop_str = getattr(conv, "stop_str", None)

                answer = messages[i][1]
                # Be aware that "\n" is added after EOS intensionally, might remove it in future
                answer_tokens = self.tokenizer.encode(
                    answer + ("" if stop_str is None else stop_str) + "\n",
                    add_special_tokens=False,
                )
                answer_start, answer_end = find_pattern_indices(tokens, answer_tokens, search_start_index)
                assert answer_start > 0, "Not found valid answer in conversation."
                labels[answer_start:answer_end] = tokens[answer_start:answer_end]
                search_start_index = answer_end

        if len(tokens) - 1 > self.sequence_length:
            logging.warning(
                f"Token indices sequence length is longer than the specified maximum sequence length "
                f"for this model ({len(tokens) - 1} > {self.sequence_length}). "
                f"Running this sequence through the model will result in indexing errors."
            )
            tokens, labels = truncate_tokens(tokens, labels, self.sequence_length, self.tokenizer)
        else:
            tokens, labels = torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        tokens = tokens[:-1]
        labels = labels[1:]
        return tokens, labels

    def _get_crop_size(self):
        if isinstance(self.image_processor, CLIPImageProcessor):
            return [self.image_processor.crop_size['height'], self.image_processor.crop_size['width']]
        else:
            raise NotImplementedError


class Qwen2VLDataset(PreloadedSupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        data_config,
        tokenizer,
        image_processor,
        model_version,
        sequence_length=None,
    ):

        if data_path.endswith(".json"):
            super().__init__(data_path, data_config, tokenizer, image_processor, model_version, sequence_length)
        elif data_path.endswith(".jsonl"):
            # FIXME: implement support for more data formats
            super().__init__(None, data_config, tokenizer, image_processor, model_version, sequence_length)
            logging.warning("Loading image inputs from Dataset...")
            if data_config.image_folder is not None:
                image_folder = data_config.image_folder
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
                        turn['value'] = re.sub('<img src="([^"]+)">', "<image>", turn['value'])

                    self.list_data_dict.append(record)

        else:
            raise ValueError(f"Formatting of {data_path} is not supported in Qwen2VL.")

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function to bundle multiple samples into a single batch.
        """
        data_config = self.data_config
        # FIXME: packed_sequence is not supported yet.
        packed_sequence = "cu_seqlens" in instances[0]
        max_len = max(instance['tokens'].shape[0] for instance in instances)
        max_len = (max_len - 1) // 64 * 64 + 64
        for instance in instances:
            pad_len = max_len - instance['tokens'].shape[0]
            instance['tokens'] = F.pad(instance['tokens'], (0, pad_len), 'constant', 0)
            instance['labels'] = F.pad(instance['labels'], (0, pad_len), 'constant', IGNORE_INDEX)
            # FIXME: packed_sequence is not supported yet.
            if packed_sequence and instance["cu_seqlens"][-1] != max_len:
                instance["cu_seqlens"] = torch.cat((instance["cu_seqlens"], torch.IntTensor([max_len])), 0)

        # FIXME: packed_sequence is not supported yet.
        if packed_sequence:
            max_len_cu = max(instance['cu_seqlens'].shape[0] for instance in instances)
            max_len_image = max(instance['image'].shape[0] for instance in instances)
            for instance in instances:
                pad_len_cu = max_len_cu - instance['cu_seqlens'].shape[0]
                instance['cu_seqlens'] = F.pad(instance['cu_seqlens'], (0, pad_len_cu), 'constant', max_len)

                x = instance['pixel_values']
                num_pad = max_len_image - x.shape[0]
                pad_tensor = torch.zeros(num_pad, *x.shape[1:], dtype=x.dtype, device=x.device)
                instance['pixel_values'] = torch.cat((x, pad_tensor), dim=0)

        batch = {
            'input_ids': torch.stack([instance['tokens'] for instance in instances]),
            'labels': torch.stack([instance['labels'] for instance in instances]),
        }

        if 'pixel_values' in instances[0]:
            batch['pixel_values'] = torch.cat([instance['pixel_values'] for instance in instances], dim=0)
        else:
            batch['pixel_values'] = None
        if 'image_grid_thw' in instances[0]:
            batch['image_grid_thw'] = torch.cat([instance['image_grid_thw'] for instance in instances], dim=0)
        else:
            batch['image_grid_thw'] = None

        if 'pixel_values_videos' in instances[0]:
            batch['pixel_values_videos'] = torch.cat(
                [instance['pixel_values_videos'] for instance in instances], dim=0
            )
        else:
            batch['pixel_values_videos'] = None
        if 'video_grid_thw' in instances[0]:
            batch['video_grid_thw'] = torch.cat([instance['video_grid_thw'] for instance in instances], dim=0)
            if self.model_version == "qwen25-vl":
                batch['second_per_grid_ts'] = torch.cat(
                    [instance['second_per_grid_ts'] for instance in instances], dim=0
                )
            else:
                batch['second_per_grid_ts'] = None
        else:
            batch['video_grid_thw'] = None
            batch['second_per_grid_ts'] = None

        tokenizer = self.tokenizer

        tokens = batch['input_ids']
        labels = batch['labels']

        if packed_sequence:
            # FIXME: packed_sequence is not supported yet.
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
                eod_token=tokenizer.eos_token_id,
                eod_mask_loss=data_config.eod_mask_loss,
                reset_attention_mask=data_config.reset_attention_mask,
                reset_position_ids=data_config.reset_position_ids,
            )

        loss_mask[labels < 0] = 0.0

        batch['loss_mask'] = loss_mask

        if packed_sequence:
            batch["cu_seqlens"] = cu_seqlens
        return batch


class Qwen2VLPreloadedDataModule(pl.LightningDataModule):
    """Preloaded DataModule for Qwen2VL."""

    def __init__(
        self,
        model_version,
        paths: str | List[str],
        weights: Optional[List[float]] = None,
        data_config: Optional[Qwen2VLDataConfig] = Qwen2VLDataConfig,
        seq_length: int = 2048,
        decoder_seq_length: Optional[int] = None,
        tokenizer: Optional = None,
        image_processor: Optional = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        use_packed_sequence: bool = False,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        if weights is not None:
            assert len(weights) == len(paths)
            if len(weights) == 1:
                # weights must be None if there is only one dataset
                weights = None

        self.model_version = model_version
        self.paths = paths
        self.weights = weights
        self.data_config = data_config
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.use_packed_sequence = use_packed_sequence
        self.init_global_step = 0

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="cyclic",
        )

    def setup(self, stage: str = "") -> None:
        # pylint: disable=C0115,C0116
        assert len(self.paths) == 1, "not yet support blend dataset in Qwen 2.0!"
        if self.use_packed_sequence:
            pass  # TODO
        else:
            # TODO:
            # rng = torch.Generator().manual_seed(self.seed)
            self._train_ds = Qwen2VLDataset(
                self.paths[0],
                self.data_config,
                self.tokenizer,
                self.image_processor,
                self.model_version,
                sequence_length=self.seq_length,
            )
            self._validation_ds = Qwen2VLDataset(
                self.paths[0],
                self.data_config,
                self.tokenizer,
                self.image_processor,
                self.model_version,
                sequence_length=self.seq_length,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # pylint: disable=C0115,C0116
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0115,C0116
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        # pylint: disable=C0115,C0116
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=getattr(dataset, 'collate_fn', data.dataloader.default_collate),
            **kwargs,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        consumed_samples = self.data_sampler.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        return {'consumed_samples': consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        try:
            from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR
        except ModuleNotFoundError:
            from nemo.lightning.apex_utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR
        consumed_samples = state_dict['consumed_samples']
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples
        self.if_first_step = 1

        if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is not None:
            num_microbatch_calculator = _GLOBAL_NUM_MICROBATCHES_CALCULATOR  # noqa: SLF001

            num_microbatch_calculator.update(
                consumed_samples=consumed_samples,
                consistency_check=False,
            )
