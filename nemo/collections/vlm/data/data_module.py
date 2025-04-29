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

from copy import deepcopy
from typing import Any, Dict, Literal, Optional

import fiddle as fdl
import lightning.pytorch as pl
import torch.distributed
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from megatron.core import parallel_state
from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset
from typing_extensions import Self

from nemo.collections.vlm.data.task_encoder import TaskEncoder
from nemo.lightning.io.mixin import IOMixin, serialization, track_io
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


class EnergonDataModule(pl.LightningDataModule, IOMixin):
    """
    A PyTorch Lightning DataModule for handling Energon datasets.

    It provides a seamless interface to load training and validation data, saving, and sampling strategies.
    The module integrates with the Megatron-Energon framework for efficient data handling
    in large-scale distributed training.
    """

    def __init__(
        self,
        path: str,
        train_encoder: TaskEncoder,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        global_batch_size: int = 1,
        num_workers: int = 1,
        num_val_workers: int | None = None,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 100,
        max_samples_per_sequence: int | None = None,
        decoder_seq_length: Optional[int] = None,
        packing_buffer_size: Optional[int] = None,
        validation_encoder: Optional[TaskEncoder] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the EnergonModule.

        Parameters:
            path (str): Path to the dataset (must be in webdataset format, and prepared using energon prepare).
            train_encoder (TaskEncoder): Encoder for training data.
            seq_length (int, optional): The maximum sequence length for tokenized text. Defaults to 2048.
            micro_batch_size (int, optional): The batch size for training and validation. Defaults to 1.
            global_batch_size (int, optional): The global batch size across all processes. Defaults to 1.
            num_workers (int, optional): Number of workers for data loading. Defaults to 1.
            num_val_workers (int, optional): Number of workers for validation data loading. Defaults to num_workers.
            pin_memory (bool, optional): Whether to pin memory in the DataLoader. Defaults to True.
            shuffle_buffer_size (int, optional): Size of the shuffle buffer. Defaults to 100.
            max_samples_per_sequence (int, optional): Maximum number of samples per sequence to load from memory.
                Defaults to None (loads the whole tar file at once).
            decoder_seq_length (int, optional): The max seq length for the decoder. Used in encoder-decoder models.
            packing_buffer_size (int, optional): Size of the packing buffer for batched samples. Defaults to None.
            validation_encoder (TaskEncoder, optional): Encoder for validation data.
                Defaults to None and will be the same as train_encoder.
            **kwargs: Additional keyword arguments passed to get_train_dataset() of Energon.
        """

        super().__init__()
        self.path = path
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples_per_sequence = max_samples_per_sequence
        self.train_encoder = train_encoder
        self.validation_encoder = validation_encoder if validation_encoder else train_encoder
        self.init_global_step = 0
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
        )
        # Since energon does shuffling and sampling for us, we only use MegatronDataSampler for microbatch calculations
        self.data_sampler.transform_dataloader = lambda dataloader: dataloader

        self.train_dataloader_object = None
        self.val_dataloader_object = None
        self.packing_buffer_size = packing_buffer_size
        self.num_val_workers = num_val_workers or self.num_workers
        self.kwargs = kwargs

        # We don't use tokenizer in data module, but we initialize it here to ensure compatibility with Nemo-Run.
        self.tokenizer = self.train_encoder.tokenizer

    def io_init(self, **kwargs) -> fdl.Config[Self]:

        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items() if k not in ['validation_encoder', 'train_encoder']}

        for val in cfg_kwargs.values():
            if not serialization.find_node_traverser(type(val)):
                track_io(type(val))
        cfg = fdl.Config(type(self), **cfg_kwargs)
        return cfg

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

        task_encoder = self.validation_encoder if split == "val" else self.train_encoder

        _dataset = get_train_dataset(
            self.path,
            batch_size=self.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=self.packing_buffer_size,
            split_part=split,
            shuffle_buffer_size=self.shuffle_buffer_size,
            max_samples_per_sequence=self.max_samples_per_sequence,
            **self.kwargs,
        )

        return _dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Initialize and return the training DataLoader.

        This method initializes the DataLoader for the training dataset. It uses the global step
        from the trainer to configure the data sampler and ensures that the parallel state is initialized
        correctly for distributed training.

        Returns:
        TRAIN_DATALOADERS: The DataLoader for the training dataset.
        """
        if self.trainer:
            self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        logging.info(f"Ttrain dataloader initializing with init_global_step {self.init_global_step}")
        if self.train_dataloader_object:
            return self.train_dataloader_object
        if not parallel_state.is_initialized():
            logging.info(
                f"Data loader parallel state is not initialized,"
                f"using default worker config with no_workers {self.num_workers}"
            )
            worker_config = WorkerConfig.default_worker_config(self.num_workers)
        else:
            rank = parallel_state.get_data_parallel_rank()
            world_size = parallel_state.get_data_parallel_world_size()
            data_parallel_group = parallel_state.get_data_parallel_group()
            logging.info(
                f" Train dataloader initializing with"
                f"rank {rank} world_size {world_size} data_parallel_group {data_parallel_group} ****** "
            )
            worker_config = WorkerConfig(
                rank=rank,
                world_size=world_size,
                num_workers=self.num_workers,
                data_parallel_group=data_parallel_group,
                worker_debug_path=None,
                worker_log_level=0,
            )
        train_dataset = self.datasets_provider(worker_config, split='train')
        energon_dataloader = get_savable_loader(train_dataset, worker_config=worker_config)
        self.train_dataloader_object = energon_dataloader
        return self.train_dataloader_object

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Initialize and return the validation DataLoader.

        This method initializes the DataLoader for the validation dataset. It ensures that the parallel state
        is initialized correctly for distributed training and returns a configured DataLoader object.

        Returns:
        EVAL_DATALOADERS: The DataLoader for the validation dataset.
        """
        if self.val_dataloader_object:
            return self.val_dataloader_object

        if not parallel_state.is_initialized():
            logging.info(
                f"Data loader val data loader parallel state is not initialized,"
                f"using default worker config with no_workers {self.num_workers}"
            )
            worker_config = WorkerConfig.default_worker_config(self.num_val_workers)
        else:
            rank = parallel_state.get_data_parallel_rank()
            world_size = parallel_state.get_data_parallel_world_size()
            data_parallel_group = parallel_state.get_data_parallel_group()

            logging.info(f"rank {rank} world_size {world_size} data_parallel_group {data_parallel_group}")
            worker_config = WorkerConfig(
                rank=rank,
                world_size=world_size,
                num_workers=self.num_workers,
                data_parallel_group=data_parallel_group,
                worker_debug_path=None,
                worker_log_level=0,
            )
        val_dataset = self.datasets_provider(worker_config, split='val')
        energon_loader = get_savable_loader(val_dataset, worker_config=worker_config)
        self.val_dataloader_object = energon_loader
        return self.val_dataloader_object

    def test_dataloader(self) -> None:
        """
        Return None as test dataset split does not exist.

        This method overrides the test_dataloader method and returns None since the test dataset split
        is not defined or used in this module.

        Returns:
        None
        """
        logging.warning("Data loader test dataset split does not exist")
        return None

    def state_dict(self) -> Dict[str, Any]:
        """
        Save the state of the data module.

        This method is called when saving a checkpoint. It generates and saves the state of the data module,
        including the state of the dataloader and the number of consumed samples.

        Returns:
        Dict[str, Any]: A dictionary containing the state of the data module.
        """

        if self.trainer:
            dataloader_obj = self.trainer.train_dataloader

            state = []
            if torch.distributed.get_rank() == parallel_state.get_model_parallel_src_rank():
                # Save_state_global in energon assumes that we call it for only the first rank within each group that
                # shares the same dataloader state. By making sure that current rank is the first rank in a model
                # parallel group, we ensure this.
                state = dataloader_obj.save_state_global(global_dst_rank=0)

            consumed_samples = self.data_sampler.compute_consumed_samples(
                self.trainer.global_step - self.init_global_step
            )

            if state is None:
                state = []  # Megatron core requires all the states on all the ranks to have same python
            # type. Energon sends the state as a list
            logging.info(f"Data loader saving dataloader state dict consumed samples {consumed_samples}")
            return {'dataloader_state': state, 'consumed_samples': consumed_samples}

        logging.warning("trainer object not connected to data module object returning empty state")
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state of the data module from a checkpoint.

        This method is called when loading a checkpoint. It restores the state of the data module,
        including the state of the dataloader and the number of consumed samples.

        Parameters:
        state_dict (Dict[str, Any]): The state dictionary containing the saved state of the data module.
        """
        if not 'dataloader_state' in state_dict:
            logging.warning(
                f"Data loader state cannot be resumed from state_dict, "
                f"it does not have the required key dataloader_state. It has {state_dict.keys()}"
            )
            return

        state = state_dict['dataloader_state']
        try:
            if self.trainer:
                self.trainer.datamodule.train_dataloader().restore_state_global(state)
                logging.info("Data loader state restored")
            else:
                logging.error(f"Cannot restore state from state_dict {state_dict}")
                raise ValueError(
                    "Cannot restore state from state_dict: "
                    "Is the trainer object is initialized and attached to datamodule???"
                )
        except Exception as e:
            logging.warning(
                f"Failed to dataloader restore state due to [Please ensure you are using same version "
                f"of energon while saving and loading, Continuing without restoring data loader] : {e}"
            )

        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        consumed_samples = state_dict['consumed_samples']
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples
        logging.info(f"Data loader load state dict with consumed_samples {consumed_samples}")
        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
