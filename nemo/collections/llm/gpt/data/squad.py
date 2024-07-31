import json
import shutil
from typing import TYPE_CHECKING, List, Optional

from datasets import DatasetDict, load_dataset

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec


class SquadDataModule(FineTuningDataModule, IOMixin):
    """A data module for fine-tuning on the Squad dataset.

    This class inherits from the `FineTuningDataModule` class and is specifically designed for fine-tuning models on the
    Stanford Question Answering Dataset (SQuAD). It handles data download, preprocessing, splitting, and preparing the data
    in a format suitable for training, validation, and testing.

    Args:
        force_redownload (bool, optional): Whether to force re-download the dataset even if it exists locally. Defaults to False.
        delete_raw (bool, optional): Whether to delete the raw downloaded dataset after preprocessing. Defaults to True.
        See FineTuningDataModule for the other args
    """

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        super().__init__(
            dataset_root=get_dataset_root("squad"),
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def prepare_data(self) -> None:
        # if train file is specified, no need to do anything
        if self.train_path.exists() and not self.force_redownload:
            return

        dset = self._download_data()
        self._preprocess_and_split_data(dset)

    def _download_data(self):
        logging.info(f"Downloading {self.__class__.__name__}...")
        return load_dataset(
            "squad",
            cache_dir=str(self.dataset_root),
            download_mode="force_redownload" if self.force_redownload else None,
        )

    def _preprocess_and_split_data(
        self, dset: DatasetDict, split_val_from_train: bool = True, val_proportion: float = 0.05
    ):
        """Preprocesses and splits the downloaded dataset into training, validation, and test sets.

        Args:
            dset (DatasetDict): The downloaded dataset object.
            split_val_from_train (bool, optional): Whether to split the validation set from the training set.
                If False, the validation set is split from the test set. Defaults to True.
            val_proportion (float, optional): The proportion of the training or test set to be used for the validation split.
                Defaults to 0.05.
        """
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")
        save_splits = {}
        train_set = dset.get('train')
        val_set = dset.get('validation')

        if split_val_from_train:
            split_dataset = train_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = split_dataset['train']
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = val_set
        else:
            split_dataset = val_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = train_set
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = split_dataset['train']

        for split_name, dataset in save_splits.items():
            output_file = self.dataset_root / f"{split_name}.jsonl"

            with output_file.open("w", encoding="utf-8") as f:
                for example in dataset:
                    json_line = {}
                    # Write each example as a JSON line in the output file
                    json_line["input"] = (
                        "Context: " + example["context"] + " Question: " + example['question'] + " Answer:"
                    )
                    json_line["output"] = example["answers"]["text"][0]
                    if split_name == "test":
                        json_line["original_answers"] = example["answers"]["text"]
                    f.write(json.dumps(json_line) + "\n")

            logging.info(f"{split_name} split saved to {output_file}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.jsonl' not in str(p.name):
                    p.unlink()

    def reconfigure_limit_batches(self):
        return
