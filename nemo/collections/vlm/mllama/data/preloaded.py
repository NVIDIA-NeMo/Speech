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
# pylint: disable=C0115,C0116

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, default_collate

from nemo.collections.vlm.mllama.model.utils import create_vision_mask_tensor
from nemo.collections.vlm.neva.data.config import DataConfig, ImageDataConfig
from nemo.collections.vlm.neva.data.preloaded import IGNORE_INDEX, LazySupervisedDataset
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils.megatron_utils import get_ltor_masks_and_position_ids


class MLlamaDataset(LazySupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        data_config,
        tokenizer,
        image_processor,
        sequence_length,
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
                        matches = re.finditer(r'<img src=["\']([^"\']+)["\']', turn['value'])
                        for match in matches:
                            image_name = match.group(1).split("/")[-1]
                            image_path = os.path.join(image_folder, image_name)
                            if not os.path.isfile(image_path):
                                logging.warning(f"Image not found: {image_path}")
                                continue
                            record['image'].append(image_name)  # url
                        turn['value'] = re.sub('<img src=["\']([^"\']+)["\']', "<image>", turn['value'])

                    self.list_data_dict.append(record)

        else:
            raise ValueError(f"Formatting of {data_path} is not supported in MLlama.")
        self.sequence_length = sequence_length

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]
        conversations = self._apply_prompt_templates(source, use_plain=self.conv_template == "plain")
        conversations = conversations.replace("<image>", "<|image|>")
        tokens, labels = self._tokenize_and_label(conversations)

        image_dict = self._process_images(source)
        data_dict = dict(
            **image_dict,
            tokens=tokens,
            labels=labels,
        )
        return data_dict

    def _process_images(self, source):
        images = []
        if 'image' in source:
            if not isinstance(source['image'], list):
                source['image'] = [source['image']]
            for image_file in source['image']:
                image = self.image_loader.open_image(image_file)
                if image is None:
                    logging.warning(f"Image {image_file} could not be found!")
                images.append(image)

        if len(images) > 0:
            image_dict = self.image_processor.preprocess(images, return_tensors='pt')
            image_dict = {
                k: v[0] for k, v in image_dict.items() if k in ["pixel_values", "aspect_ratio_ids", "num_tiles"]
            }  # remove batch dim
        else:
            image_dict = dict(
                pixel_values=torch.zeros(
                    1, 4, 3, self.image_processor.size['height'], self.image_processor.size['width']
                ),
                aspect_ratio_ids=torch.tensor([0], dtype=torch.long),
                num_tiles=[0],
            )

        return image_dict

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        data_config = self.data_config
        max_len = (max(instance['tokens'].shape[0] for instance in instances) - 1) // 64 * 64 + 64
        if max_len > self.sequence_length:
            logging.warning(f"Truncating sequence length {max_len} to {self.seq_length}.")
            max_len = self.sequence_length
        max_num_concurrent_media = max(instance['pixel_values'].shape[0] for instance in instances)
        for instance in instances:
            pad_len = max_len - instance['tokens'].shape[0]
            instance['tokens'] = F.pad(instance['tokens'], (0, pad_len), 'constant', 0)
            instance['labels'] = F.pad(instance['labels'], (0, pad_len), 'constant', IGNORE_INDEX)
            pad_num_images = max_num_concurrent_media - instance['pixel_values'].shape[0]
            instance['pixel_values'] = F.pad(
                instance['pixel_values'], (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_num_images), 'constant', 0
            )
            instance['aspect_ratio_ids'] = F.pad(
                instance['aspect_ratio_ids'], (0, max(pad_num_images - 1, 0)), 'constant', 0
            )
            instance['num_tiles'] = F.pad(
                torch.tensor(instance['num_tiles']), (0, max(pad_num_images - 1, 0)), 'constant', 0
            )

        batch_masks = [create_vision_mask_tensor(instance['tokens'], 128256) for instance in instances]
        batch = default_collate(instances)

        tokenizer = self.tokenizer

        tokens = batch['tokens']
        labels = batch['labels']

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
            'batch_images': batch['pixel_values'],
            'batch_masks': batch_masks,
            'num_chunks': batch['num_tiles'],
            'attention_mask': attention_mask,
            "aspect_ratio_ids": batch['aspect_ratio_ids'],
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }
        return batch


class MLlamaPreloadedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        paths: str | List[str],
        weights: Optional[List[float]] = None,
        data_config: Optional[DataConfig] = ImageDataConfig,
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
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if tokenizer is None or image_processor is None:
            logging.warning(
                "Processor and tokenizer are not provided! Fall back to `meta-llama/Llama-3.2-11B-Vision-Instruct`."
            )
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
            self.tokenizer = tokenizer or processor.tokenizer
            self.image_processor = image_processor or processor.image_processor

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="cyclic",
        )

    def setup(self, stage: str = "") -> None:
        assert len(self.paths) == 1, "not yet support blend dataset in MLlama 2.0!"
        if self.use_packed_sequence:
            pass  # TODO
        else:
            # TODO:
            # rng = torch.Generator().manual_seed(self.seed)
            # train_dataset, val_dataset, test_dataset =
            # random_split(dataset, [train_size, val_size, test_size], generator=rng)
            self._train_ds = MLlamaDataset(
                self.paths[0], self.data_config, self.tokenizer, self.image_processor, self.seq_length
            )
            self._validation_ds = MLlamaDataset(
                self.paths[0], self.data_config, self.tokenizer, self.image_processor, self.seq_length
            )

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
