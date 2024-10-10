import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from nemo.collections.llm.t5.data.core import create_sft_dataset
from nemo.lightning.pytorch.plugins import MegatronDataSampler

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec


class FineTuningDataModule(pl.LightningDataModule):
    """Base class for fine-tuning an LLM.

    This class provides a foundation for building custom data modules for fine-tuning Nemo NLP models. It inherits from
    `pl.LightningDataModule` from the PyTorch Lightning library and handles data loading, preprocessing, and batch creation
    for training, validation, and testing.

    Args:
        dataset_root (Union[str, Path]): The root directory containing the training, validation, and test data.
        seq_length (int, optional): The maximum sequence length for the input and output text. Defaults to 2048.
        tokenizer (Optional[TokenizerSpec], optional): The tokenizer to use for preprocessing the text. Defaults to None.
            If not provided, a BertWordPieceCase tokenizer will be used.
        micro_batch_size (int, optional): The micro batch size for training. Defaults to 4.
        global_batch_size (int, optional): The global batch size for training. Defaults to 8.
        rampup_batch_size (Optional[List[int]], optional): A list of batch sizes for ramping up during training. Defaults to None.
        seed (int, optional): The random seed for data shuffling. Defaults to 1234.
        memmap_workers (int, optional): The number of worker processes for loading data using TextMemMapDataset. Defaults to 1.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.
        pin_memory (bool, optional): Whether to pin memory during data loading for faster GPU training. Defaults to True.
        persistent_workers (bool, optional): Whether to keep data loading workers persistent across epochs. Defaults to False.
        max_train_steps (int, optional): Maximum number of steps to train. Used to calculate samples mapping for the mmap dataset
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        seq_length: int = 512,
        seq_length_dec: int = 128,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.seq_length_dec = seq_length_dec
        self.seed = seed
        self.dataset_root = Path(dataset_root)

        # add additional tokens for T5 tokenizer
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        self.tokenizer = tokenizer or get_nmt_tokenizer("megatron", "BertWordPieceCase")
        additional_tokens = {'additional_special_tokens': [f'<extra_id_{i}>' for i in range(100)]}
        self.tokenizer.add_special_tokens(additional_tokens)

        self.memmap_workers = memmap_workers
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.rampup_batch_size = rampup_batch_size
        self.data_sampler = None
        self.max_train_samples = None

    def setup(self, stage: str):
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
            dataloader_type="batch",
        )

        # Follows the calculation in nemo.collections.nlp.data.language_modeling.megatron.
        # base_dataset_utils.get_datasets_weights_and_num_samples
        self.max_train_samples = int(math.ceil(self.global_batch_size * self.trainer.max_steps * 1.005))

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self._create_dataset(
                str(self.train_path),
                max_num_samples=self.max_train_samples,
            )
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self._create_dataset(
                str(self.validation_path),
                is_test=True,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self._create_dataset(
                str(self.test_path),
                tokens_to_generate=32,
                is_test=True,
            )
        )

    @lru_cache
    def _create_dataset(self, path, **kwargs):
        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            seq_length_dec=self.seq_length_dec,
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            **kwargs,
        )

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )

    @property
    def train_path(self) -> Path:
        return self.dataset_root / "training.jsonl"

    @property
    def validation_path(self) -> Path:
        return self.dataset_root / "validation.jsonl"

    @property
    def test_path(self) -> Path:
        return self.dataset_root / "test.jsonl"
