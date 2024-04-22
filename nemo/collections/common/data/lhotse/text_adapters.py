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

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Union

from lhotse.cut.text import TextExample, TextPairExample
from lhotse.dataset.dataloading import resolve_seed
from lhotse.utils import Pathlike

from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths


@dataclass
class LhotseTextAdapter:
    """
    ``LhotseTextAdapter`` is used to read a text file and wrap
    each line into a ``TextExample``.
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)

    def __iter__(self) -> Iterator[TextExample]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            with open(path) as f:
                for line in f:
                    example = TextExample(line)
                    if self.language is not None:
                        example.language = self.language
                    yield example


@dataclass
class LhotseTextPairAdapter:
    """
    ``LhotseTextAdapter`` is used to read a tuple of N text files
    (e.g., a pair of files with translations in different languages)
    and wrap them in a ``TextExample`` object to enable dataloading
    with Lhotse together with training examples in audio modality.
    """

    source_paths: Union[Pathlike, list[Pathlike]]
    target_paths: Union[Pathlike, list[Pathlike]]
    source_language: str | None = None
    target_language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        ASSERT_MSG = "Both source and target must be a single path or lists of paths"
        if isinstance(self.source_paths, (str, Path)):
            assert isinstance(self.target_paths, (str, Path)), ASSERT_MSG
        else:
            assert isinstance(self.source_paths, list) and isinstance(self.target_paths, list), ASSERT_MSG
            assert len(self.source_paths) == len(
                self.target_paths
            ), f"Source ({len(self.source_paths)}) and target ({len(self.target_paths)}) path lists must have the same number of items."
        self.source_paths = expand_sharded_filepaths(self.source_paths)
        self.target_paths = expand_sharded_filepaths(self.target_paths)

    def __iter__(self) -> Iterator[TextPairExample]:
        paths = list(zip(self.source_paths, self.target_paths))
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for source_path, target_path in paths:
            with open(source_path) as fs, open(target_path) as ft:
                for ls, lt in zip(fs, ft):
                    example = TextPairExample(source=TextExample(ls.strip()), target=TextExample(lt.strip()))
                    if self.source_language is not None:
                        example.source.language = self.source_language
                    if self.target_language is not None:
                        example.target.language = self.target_language
                    yield example
