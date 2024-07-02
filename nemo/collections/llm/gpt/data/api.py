import pytorch_lightning as pl

from nemo.collections.llm.gpt.data.dolly import DollyDataModule
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.utils import factory


@factory
def mock() -> pl.LightningDataModule:
    return MockDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


@factory
def squad() -> pl.LightningDataModule:
    return SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


@factory
def dolly() -> pl.LightningDataModule:
    return DollyDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


__all__ = ["mock", "squad", "dolly"]
