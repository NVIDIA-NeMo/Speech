import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data

from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.pytorch.plugins import MegatronDataSampler

if TYPE_CHECKING:
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class PreTrainingDataModule(pl.LightningDataModule, IOMixin):
    """PyTorch Lightning-compatible data module for pre-training
       GPT-style models.
    Args:
        paths (Path | List | Dict[str, List]): Paths of the data distributions. Can be either a
            single path, a list of paths, or a dictionary. If a single path or a list of paths,
            the given paths will be used to generate the train, validation and test datasets. If
            providing a list of paths, the format can be either (1) a list of paths, e.g.
                ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"],
            or (2) a flattened, zipped list of weights and paths, e.g.
                ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]
            If a dictionary is provided, it is expected to have the following form:
                {
                    'train': <TRAIN PATHS>,
                    'validation': <VALID PATHS>,
                    'test': <TEST PATHS>
                }
            where each value is either a path or a list of paths as described above.
            In this case, each split will be generated using the given paths.
            Note that if limit_val_batches <= 1, we generate the entire validaton dataset, so
            weights should not be provided for the validation split.
        seq_length (int): Sequence length.
        tokenizer (Optional["TokenizerSpec"]): An instance of a TokenizerSpec object.
        micro_batch_size (int): Batch size per GPU.
        global_batch_size (int): Global batch size.
        rampup_batch_size (Optional[List[int]]): Rampup batch size, should be in format of
            [start_global_batch_size, batch_size_increment, ramup_samples].
        num_workers (int): See ``torch.utils.data.DataLoader`` documentation.
        pin_memory (bool): See ``torch.utils.data.DataLoader`` documentation.
        persistent_workers (bool): See ``torch.utils.data.DataLoader`` documentation.
        reset_position_ids (bool): Option to reset the position IDs in the dataset at an interval.
        reset_attention_mask (bool): Option to reset the attention mask from the dataset.
        eod_mask_loss (int): Option to enable the EOD mask loss.
        seed (int): Seed for generating the GPT dataset.
        split (str): A string of 3 comma-separated integers denoting how much of the distribution
            to allocate to train, validation, and test sets, respectively. Unused if ``paths`` is a dict.
        index_mapping_dir (Optional[str]): Path to a directory to write index mapping files.
    """

    def __init__(
        self,
        paths: Path | List | Dict[str, List],
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        reset_position_ids: bool = False,
        reset_attention_mask: bool = False,
        eod_mask_loss: bool = False,
        seed: int = 1234,
        split: str = "900,50,50",
        index_mapping_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not isinstance(paths, (list, tuple, dict)):
            paths = [paths]

        from megatron.core.datasets.utils import get_blend_from_list

        build_kwargs = {}
        if isinstance(paths, dict):
            if split is not None:
                warnings.warn(
                    f"{split=} will be ignored since datasets are being created " f"from 3 separate distributions."
                )
            build_kwargs["blend_per_split"] = [
                get_blend_from_list(paths["train"]),
                get_blend_from_list(paths["validation"]),
                get_blend_from_list(paths["test"]),
            ]
        else:
            paths, weights = get_blend_from_list(paths)
            if len(paths) == 1:
                weights = None
            build_kwargs["blend"] = [paths, weights]
            build_kwargs["split"] = split

        self.build_kwargs = build_kwargs
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.seed = seed
        self.split = split
        self.index_mapping_dir = index_mapping_dir
        self.init_global_step = 0

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
        num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)
        num_val_samples = int(eval_iters * self.data_sampler.global_batch_size)
        num_test_samples = int(test_iters * self.data_sampler.global_batch_size)

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            assert "blend" not in self.build_kwargs, (
                "When using a single data distribution, limit_val_batches <= 1.0 is not supported. If you'd "
                "like to run with a fractional value of limit_val_batches, please pass in separate datasets for "
                "the train, validation, and test datasets by providing a dictionary of paths, e.g.: \n"
                "    paths={ \n "
                "        'train': [PATHS FOR TRAIN], \n "
                "        'validation': [PATHS FOR VALIDATION], \n "
                "        'test' :[PATHS FOR TEST],  \n"
                "    }"
            )

            # This is to make sure we only have one epoch on every validation iteration
            num_val_samples = None

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
        return self._create_dataloader(self._train_ds, mode='train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._validation_ds, mode='validation')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_ds, mode='test')

    def _create_dataloader(self, dataset, mode, **kwargs) -> WrappedDataLoader:
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        dataloader = WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=getattr(dataset, 'collate_fn', data.dataloader.default_collate),
            **kwargs,
        )
        return dataloader

    @property
    def gpt_dataset_config(self) -> "GPTDatasetConfig":
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

        return GPTDatasetConfig(
            random_seed=self.seed,
            sequence_length=self.seq_length,
            tokenizer=self.tokenizer,
            path_to_cache=self.index_mapping_dir,
            reset_position_ids=self.reset_position_ids,
            reset_attention_mask=self.reset_attention_mask,
            eod_mask_loss=self.eod_mask_loss,
            **self.build_kwargs,
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
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        consumed_samples = state_dict['consumed_samples']
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1

    def reconfigure_limit_batches(self):
        # Override limit_train_batches in terms of num of microbatches
        self._reconfigure_limit_batches(self.trainer.limit_train_batches, self._train_ds, 'train')
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting in between a step
        self._reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_ds, 'val')

    def _reconfigure_limit_batches(self, limit_batches, dataloader, mode):
        """
        Reconfigure trainer.limit_val_batches for pretraining
        """
        # Override limit_batches in terms of num microbatches and so there are limit_batches//num_micro_batches num of global batches
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        if isinstance(limit_batches, int):
            limit_batches *= get_num_microbatches()
        else:
            assert isinstance(limit_batches, float)
            # Don't reconfigure if limit_batches is 0.0 or if there's no dataloader
            if limit_batches == 0.0 or dataloader is None:
                return
            # len(dataloader) returns len as num of microbatches
            dl_len_in_micro_batches = len(dataloader)
            if len(dataloader) != float("inf"):
                if limit_batches == 1.0:
                    limit_batches = dl_len_in_micro_batches
                else:
                    limit_micro_batches = int(dl_len_in_micro_batches * limit_batches)
                    if limit_micro_batches == 0 and limit_batches > 0.0:
                        min_percentage = 1.0 / len(dataloader)
                        raise MisconfigurationException(
                            f"You requested to check {limit_batches} of the val_dataloader but"
                            f" {limit_batches} * {len(dataloader)} < 1. Please increase the"
                            f" `limit_val_batches` argument. Try at least"
                            f" `limit_val_batches={min_percentage}`"
                        )
                    # Make sure trainer.limit_val_batches is a multiple of num of microbatches
                    if limit_micro_batches < get_num_microbatches():
                        limit_batches = get_num_microbatches()
                    else:
                        limit_batches = limit_batches - limit_batches % get_num_microbatches()

        if mode == 'train':
            self.trainer.limit_train_batches = limit_batches
        else:
            self.trainer.limit_val_batches = limit_batches

        # Override num sanity steps to be a multiple of num of microbatches
        self.trainer.num_sanity_val_steps *= get_num_microbatches()
