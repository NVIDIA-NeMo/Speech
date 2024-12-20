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

import logging
from typing import Any, Dict, Literal

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from megatron.core import parallel_state
from megatron.energon import DefaultTaskEncoder, WorkerConfig, get_savable_loader, get_train_dataset

from nemo.collections.multimodal.data.energon.base import EnergonMultiModalDataModule


class DiffusionDataModule(EnergonMultiModalDataModule):
    """
    A PyTorch Lightning DataModule for handling multimodal datasets with images and text.

    This data module is designed to work with multimodal datasets that involve both images and text.
    It provides a seamless interface to load training and validation data, manage batching, and handle
    the state of the data pipeline across training epochs. The module integrates with the Megatron-Energon
    framework for efficient data handling in large-scale distributed training.

    Attributes:
    path (str): Path to the energon dataset.
    tokenizer (Tokenizer): The tokenizer used for processing text.
    image_processor (ImageProcessor): The image processor used for preprocessing images.
    seq_length (int): The maximum sequence length for tokenized text.
    micro_batch_size (int): The batch size for training and validation.
    num_workers (int): Number of workers for data loading.
    pin_memory (bool): Whether to pin memory in the DataLoader.
    multimodal_sample_config (MultiModalSampleConfig): Configuration object for multimodal samples.
    task_encoder (MultiModalTaskEncoder): Encoder responsible for encoding and batching samples.
    init_global_step (int): The initial global step for the trainer, used for resuming training.
    data_sampler (SequentialMegatronSampler): Sampler responsible for generating sequential samples.
    train_dataloader_object (Optional): The DataLoader object for training data.
    val_dataloader_object (Optional): The DataLoader object for validation data.
    """

    def __init__(
        self,
        path: str,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = 1,
        pin_memory: bool = True,
        task_encoder: DefaultTaskEncoder = None,
        use_train_split_for_val: bool = False,
        virtual_epoch_length: int = 1_000_000_000,  # a hack to avoid energon end of epoch warning
        packing_buffer_size: int | None = None,
        max_samples_per_sequence: int | None = None,
    ) -> None:
        """
        Initialize the EnergonMultiModalDataModule.

        Parameters:
        path (str): Path to the dataset.
        tokenizer (Tokenizer): The tokenizer used for processing text.
        image_processor (ImageProcessor): The image processor used for preprocessing images.
        seq_length (int, optional): The maximum sequence length for tokenized text. Defaults to 2048.
        micro_batch_size (int, optional): The batch size for training and validation. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 1.
        pin_memory (bool, optional): Whether to pin memory in the DataLoader. Defaults to True.
        """

        super().__init__(
            path=path,
            tokenizer=None,
            image_processor=None,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            task_encoder=task_encoder,
        )
        self.use_train_split_for_val = use_train_split_for_val
        self.virtual_epoch_length = virtual_epoch_length
        self.num_workers_val = 1
        self.packing_buffer_size = packing_buffer_size
        self.max_samples_per_sequence = max_samples_per_sequence

    def datasets_provider(self, worker_config, split: Literal['train', 'val'] = 'val'):
        """
        Provide the dataset for training or validation.

        This method retrieves the dataset for the specified split (either 'train' or 'val') and configures
        it according to the worker configuration.

        Parameters:
        worker_config: Configuration for the data loader workers.
        split (Literal['train', 'val'], optional): The data split to retrieve ('train' or 'val'). Defaults to 'val'.

        Returns:
        Dataset: The dataset configured for the specified split.
        """
        if split not in {'train', 'val'}:
            raise ValueError("Invalid value for split. Allowed values are 'train' or 'val'.")
        if self.use_train_split_for_val:
            split = 'train'
        _dataset = get_train_dataset(
            self.path,
            batch_size=self.micro_batch_size,
            task_encoder=self.task_encoder,
            worker_config=worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            shuffle_buffer_size=None,
            split_part=split,
            virtual_epoch_length=self.virtual_epoch_length,
            packing_buffer_size=self.packing_buffer_size,
        )
        return _dataset

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Initialize and return the validation DataLoader.

        This method initializes the DataLoader for the validation dataset. It ensures that the parallel state
        is initialized correctly for distributed training and returns a configured DataLoader object.

        Returns:
        EVAL_DATALOADERS: The DataLoader for the validation dataset.
        """
        if self.use_train_split_for_val:
            return self.train_dataloader()
        if self.val_dataloader_object:
            return self.val_dataloader_object

        if not parallel_state.is_initialized():
            message = (
                "Muiltimodal val data loader parallel state is not initialized "
                f"using default worker config with no_workers {self.num_workers}"
            )
            logging.info(message)

            worker_config = WorkerConfig.default_worker_config(self.num_workers_val)
        else:
            rank = parallel_state.get_data_parallel_rank()
            world_size = parallel_state.get_data_parallel_world_size()
            data_parallel_group = parallel_state.get_data_parallel_group()

            logging.info(f"rank {rank} world_size {world_size} data_parallel_group {data_parallel_group}")
            worker_config = WorkerConfig(
                rank=rank,
                world_size=world_size,
                num_workers=self.num_workers_val,
                data_parallel_group=data_parallel_group,
                worker_debug_path=None,
                worker_log_level=0,
            )
        val_dataset = self.datasets_provider(worker_config, split='val')
        energon_loader = get_savable_loader(val_dataset, worker_config=worker_config)
        self.val_dataloader_object = energon_loader
        return self.val_dataloader_object

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state of the data module from a checkpoint.

        This method is called when loading a checkpoint. It restores the state of the data module,
        including the state of the dataloader and the number of consumed samples.

        Parameters:
        state_dict (Dict[str, Any]): The state dictionary containing the saved state of the data module.
        """
        try:
            super().load_state_dict(state_dict)
        except Exception as e:
            logging.warning(f"datamodule.load_state_dict failed  {e}")
