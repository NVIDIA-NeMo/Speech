from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from nemo.lightning.pytorch.plugins import MegatronDataSampler

if TYPE_CHECKING:
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class PreTrainingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: Path,
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        reset_position_ids: bool = False,
        reset_attention_mask: bool = False,
        eod_mask_loss: bool = False,
        seed: int = 1234,
        split: str = "900,50,50",
    ) -> None:
        super().__init__()
        self.path = path
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.seed = seed
        self.split = split

        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        self.tokenizer = tokenizer or get_nmt_tokenizer("megatron", "GPT2BPETokenizer")
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
        from megatron.core.datasets.gpt_dataset import GPTDataset

        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."

        # Trainer API
        max_train_steps = self.trainer.max_steps
        assert max_train_steps > 0, "Please specify trainer.max_steps"
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        num_train_samples = max_train_steps * self.data_sampler.global_batch_size
        num_val_samples = eval_iters * self.data_sampler.global_batch_size
        num_test_samples = test_iters * self.data_sampler.global_batch_size

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            # This is to make sure we only have one epoch on every validation iteration
            num_val_samples = 1

        train_valid_test_num_samples = [num_train_samples, num_val_samples, num_test_samples]
        self._train_ds, self._validation_ds, self._test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_valid_test_num_samples,
            is_built_on_rank=lambda: True,
            config=self.gpt_dataset_config,
        ).build()

    # uncomment once fabric API is merged
    # def fabric_setup(
    #     self,
    #     fabric: fl.Fabric,
    #     num_train_samples: int,
    #     num_val_samples: int,
    #     num_test_samples: int,
    # ) -> None:
    #     from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    #     from megatron.core.datasets.gpt_dataset import GPTDataset
    #
    #     del fabric
    #     train_valid_test_num_samples = [num_train_samples, num_val_samples, num_test_samples]
    #     self._train_ds, self._validation_ds, self._test_ds = BlendedMegatronDatasetBuilder(
    #         GPTDataset, train_valid_test_num_samples, self.gpt_dataset_config,
    #     ).build()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_ds)

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
    def gpt_dataset_config(self) -> "GPTDatasetConfig":
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

        return GPTDatasetConfig(
            blend=[[str(self.path)], [1.0]],
            random_seed=self.seed,
            sequence_length=self.seq_length,
            tokenizer=self.tokenizer,
            split=self.split,
            path_to_cache=None,
            reset_position_ids=self.reset_position_ids,
            reset_attention_mask=self.reset_attention_mask,
            eod_mask_loss=self.eod_mask_loss,
        )
