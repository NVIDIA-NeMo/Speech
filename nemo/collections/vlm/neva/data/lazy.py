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

from typing import TYPE_CHECKING, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader

from nemo.collections.vlm.neva.data.config import DataConfig, ImageDataConfig
from nemo.collections.vlm.neva.data.conversation import conv_templates as supported_conv_templates
from nemo.lightning.pytorch.plugins import MegatronDataSampler

if TYPE_CHECKING:
    pass

import json
import logging
import os
import re
import tarfile
from typing import Any, Dict, List, Sequence

import decord
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, default_collate
from transformers import CLIPImageProcessor, SiglipImageProcessor

from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.vlm.neva.data.multimodal_tokens import IGNORE_INDEX, SPECIAL_TOKEN_MAP


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
        data_config (dict): A dictionary of configuration options for video decoding to frames
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

    def __init__(self, video_folder, data_config):
        self.video_folder = video_folder
        self.data_config = data_config
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
            # decord.bridge.set_bridge("torch")
            cap = decord.VideoReader(os.path.join(self.video_folder, file_name))
            return self.flatten_frames(cap)
        return None

    def flatten_frames(self, cap):
        if self.data_config.splice_single_frame == 'first':
            frame = cap[0].asnumpy()
            return Image.fromarray(frame).convert('RGB')
        elif self.data_config.splice_single_frame == 'middle':
            frame = cap[len(cap) // 2].asnumpy()
            return Image.fromarray(frame).convert('RGB')
        elif self.data_config.splice_single_frame == 'last':
            frame = cap[-1].asnumpy()
            return Image.fromarray(frame).convert('RGB')
        else:
            if self.data_config.num_frames == -1:
                frames = []
                for frame in cap:
                    rgb_frame = frame.asnumpy()
                    img = Image.fromarray(rgb_frame).convert('RGB')
                    frames.append(img)
                return frames
            else:
                num_frames = min(len(cap), self.data_config.num_frames)
                indices = np.linspace(0, len(cap) - 1, num_frames, dtype=int)
                frames = [Image.fromarray(cap[i].asnumpy()).convert('RGB') for i in indices]
                while len(frames) < self.data_config.num_frames:
                    frames.append(frames[-1])
                return frames


def process_image(processor, image, image_process_mode="square"):  # this needs to be merged with conv's process image
    if isinstance(processor, CLIPImageProcessor) or isinstance(processor, SiglipImageProcessor):
        # image processor from HF
        if image_process_mode == 'keep':
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = processor.preprocess(
                image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge}
            )['pixel_values'][0]
        elif image_process_mode == 'pad':

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
        assert image_process_mode == 'square', 'NeMo image transform with setting `image_process_mode` to `square`.'
        image = processor(image)
    return image


def tokenize_special_token(prompt, tokenizer, special_token_map=None):
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
    for chunk in chunks:
        if chunk in special_token_dict:
            tokenized_chunks.append(special_token_dict[chunk])
        elif len(chunk) > 0:
            tokenized_chunks.extend(tokenizer(chunk, add_special_tokens=False).input_ids)

    return torch.tensor(tokenized_chunks, dtype=torch.long)


def find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = template[i : i + pattern_len] == pattern
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1


class LazySupervisedDataset(Dataset):

    def __init__(
        self,
        data_path,
        data_config,
        tokenizer,
        image_processor,
    ):
        super().__init__()
        if data_path is not None:
            with open(data_path, "r") as file:
                list_data_dict = json.load(file)
        else:
            list_data_dict = []

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.conv_template = data_config.conv_template
        self.conv = supported_conv_templates[self.conv_template]
        self.image_process_mode = data_config.image_process_mode
        self.list_data_dict = list_data_dict

        image_folder = getattr(data_config, "image_folder", None)
        video_folder = getattr(data_config, "video_folder", None)

        self.image_loader = TarOrFolderImageLoader(image_folder) if image_folder else None
        self.video_loader = TarOrFolderVideoLoader(video_folder, data_config) if video_folder else None

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]
        conversations = self._apply_prompt_templates(source, use_plain=self.conv_template == "plain")
        tokens, labels = self._tokenize_and_label(conversations)

        media_tensors = self._process_images(source)
        data_dict = dict(
            image=media_tensors,
            tokens=tokens,
            labels=labels,
        )
        return data_dict

    def _process_images(self, source):
        media_tensors = torch.tensor([])
        if 'image' in source:
            if not isinstance(source['image'], list):
                source['image'] = [source['image']]

            images = []
            for image_file in source['image']:
                image = self.image_loader.open_image(image_file)
                if image is None:
                    logging.warning(f"Image {image_file} could not be found!")
                image = process_image(self.image_processor, image, self.image_process_mode)
                images.append(image)

            if images:
                media_tensors = torch.stack(images)
        return media_tensors

    def _apply_prompt_templates(self, source, use_plain=False):
        conv = self.conv

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        source = source['conversations']
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])

        if use_plain:
            assert len(conv.messages) == 2, "Plain template requires image-caption pairs."
            assert "<image>" in conv.messages[0][1]
            conv.messages[0][1] = "<image>"

        return conv.get_prompt()

    def _tokenize_and_label(self, conversations):
        tokens = tokenize_special_token(conversations, self.tokenizer)
        labels = torch.ones_like(tokens) * IGNORE_INDEX
        search_start_index = 0
        for i in range(1, len(self.conv.messages), 2):
            stop_str = getattr(self.conv, "stop_str", None)
            assert (
                stop_str is not None
            ), "If `stop_str` is not provided, issues might occur in labeling the answer tokens."
            answer_tokens = self.tokenizer.encode(
                self.conv.messages[i][1] + ("" if stop_str is None else stop_str),
                add_special_tokens=False,
                return_tensors="pt",
            )[0]
            answer_start, answer_end = find_pattern_indices(tokens, answer_tokens, search_start_index)
            labels[answer_start:answer_end] = tokens[answer_start:answer_end]
            search_start_index = answer_end
        tokens = tokens[:-1]
        labels = labels[1:]
        return tokens, labels

    def _get_crop_size(self):
        if isinstance(self.image_processor, CLIPImageProcessor):
            return [self.image_processor.crop_size['height'], self.image_processor.crop_size['width']]
        else:
            raise NotImplementedError


class NevaDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        data_config,
        tokenizer,
        image_processor,
    ):

        if data_path.endswith(".json"):
            super().__init__(data_path, data_config, tokenizer, image_processor)

        elif data_path.endswith(".jsonl"):
            super().__init__(None, data_config, tokenizer, image_processor)
            logging.warning("Loading image inputs from SteerLM Dataset...")
            if data_config.media_type == 'image':
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
            raise ValueError(f"Formatting of {data_path} is not supported in Neva.")

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        data_config = self.data_config
        packed_sequence = "cu_seqlens" in instances[0]
        max_len = max(instance['tokens'].shape[0] for instance in instances)
        for instance in instances:
            pad_len = max_len - instance['tokens'].shape[0]
            instance['tokens'] = F.pad(instance['tokens'], (0, pad_len), 'constant', 0)
            instance['labels'] = F.pad(instance['labels'], (0, pad_len), 'constant', IGNORE_INDEX)
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

        media_type = data_config.media_type
        if media_type == 'image':
            media = [instance.pop('image') for instance in instances]
            media = torch.cat(media, dim=0)
            if media.size(0) == 0:
                media = None
        elif media_type == 'video':
            media = [instance.pop('video', None) for instance in instances]
        else:
            raise ValueError(f"Unsupported media type {media_type}")

        batch = default_collate(instances)
        tokenizer = self.tokenizer

        tokens = batch['tokens']
        labels = batch['labels']

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
                eod_token=tokenizer.eos_token_id,
                eod_mask_loss=data_config.eod_mask_loss,
                reset_attention_mask=data_config.reset_attention_mask,
                reset_position_ids=data_config.reset_position_ids,
            )

        loss_mask[labels < 0] = 0.0

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


class NevaLazyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        paths: str | List[str],
        weights: Optional[List[float]] = None,
        data_config: Optional[DataConfig] = ImageDataConfig,
        seq_length: int = 2048,
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

        self.paths = paths
        self.weights = weights
        self.data_config = data_config
        self.seq_length = seq_length
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

        if tokenizer is None or image_processor is None:
            logging.warning(f"Processor and tokenizer are not provided! Fall back to `llava-hf/llava-1.5-7b-hf`.")
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.tokenizer = tokenizer or processor.tokenizer
            self.image_processor = image_processor or processor.image_processor

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="cyclic",
        )

    def setup(self, stage: str = "") -> None:
        assert len(self.paths) == 1, "not yet support blend dataset in Neva 2.0!"
        if self.use_packed_sequence:
            pass  # TODO
        else:
            # TODO:
            # rng = torch.Generator().manual_seed(self.seed)
            # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=rng)
            self._train_ds = NevaDataset(self.paths[0], self.data_config, self.tokenizer, self.image_processor)
            self._validation_ds = NevaDataset(self.paths[0], self.data_config, self.tokenizer, self.image_processor)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
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
