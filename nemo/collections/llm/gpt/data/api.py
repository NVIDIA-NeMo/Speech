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

import lightning.pytorch as pl
import nemo_run as run

from nemo.collections.llm.gpt.data.dolly import DollyDataModule
from nemo.collections.llm.gpt.data.hf_dataset import HFDatasetDataModule
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule


@run.cli.factory
@run.autoconvert
def mock() -> pl.LightningDataModule:
    return MockDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


@run.cli.factory
@run.autoconvert
def squad() -> pl.LightningDataModule:
    return SquadDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


@run.cli.factory
@run.autoconvert
def dolly() -> pl.LightningDataModule:
    return DollyDataModule(seq_length=4096, global_batch_size=16, micro_batch_size=2)


@run.cli.factory
@run.autoconvert
def hf_dataset(path: str) -> pl.LightningDataModule:
    return HFDatasetDataModule(path=path, global_batch_size=16, micro_batch_size=2)


__all__ = ["mock", "squad", "dolly", "hf_dataset"]
