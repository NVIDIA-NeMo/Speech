# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import json
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from nemo.utils import logging


@dataclass
class DataEntry:
    """A wrapper for data entry + any additional metrics."""

    data: Optional[Dict]  # can be None to drop the entry
    metrics: Any = None


class BaseProcessor(ABC):
    """
    Abstract class for SDP processors.

    Args
    output_manifest_file: path of where the output manifest file will be located.
    input_manifest_file: path of where the input manifest file is located. This arg 
        is optional - some processors may not take in an input manifest because they
        need to create an initial manifest from scratch (ie from some transcript file
        that is in a format different to the NeMo manifest format).
    """

    def __init__(self, output_manifest_file, input_manifest_file=None):
        self.output_manifest_file = output_manifest_file
        self.input_manifest_file = input_manifest_file

    @abstractmethod
    def process(self):
        pass

    def test(self):
        """This method can be used to perform "runtime" tests.

        This can be any kind of self-consistency tests, but are usually
        in the form of checking that provided input test data entries match
        provided output test data entries.

        There are not tests by default.
        """


class BaseParallelProcessor(BaseProcessor):
    """
    Processor class which allows operations on each utterance to be parallelized. Parallelization 
    is done using tqdm.contrib.concurrent.process_map.

    Args:
        max_workers: maximum number of workers that will be spawned during parallel processing.
        chunksize: the size of the chunks that will be sent to worker processes. 
    """

    def __init__(self, max_workers: int = -1, chunksize: int = 100, **kwargs):
        super().__init__(**kwargs)
        if max_workers == -1:
            max_workers = multiprocessing.cpu_count()
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.number_of_entries = 0
        self.total_duration = -1

    def read_manifest(self):
        """
        This function should be overridden in the "initial" class creating
        manifest to read from the original source of data.
        """
        if self.input_manifest_file is None:
            raise NotImplementedError("Override this method if the processor creates initial manifest")

        # TODO: should we not assume that manifest can fully fit in memory?
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            dataset_entries = [json.loads(line) for line in fin.readlines()]

        return dataset_entries

    def prepare(self):
        """Can be used in derived classes to prepare processing in any way.

        E.g., download data or compute some aggregates. Will be called before
        starting processing the data.
        """

    def process(self):
        """
        We always going to serialize output into a manifest file that contains
        all information about the dataset.

        This method should also create a new manifest in the end according to
        the `self.output_manifest_file` argument.
        """
        self.prepare()
        dataset_entries = self.read_manifest()

        # this will unroll all inner lists
        data = itertools.chain(
            *process_map(
                self.process_dataset_entry, dataset_entries, max_workers=self.max_workers, chunksize=self.chunksize,
            )
        )
        metrics = []
        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for data_entry in tqdm(data):
                metrics.append(data_entry.metrics)
                if data_entry.data is None:
                    continue
                json.dump(data_entry.data, fout)
                self.number_of_entries += 1
                self.total_duration += data_entry.data.get("duration", 0)
                fout.write("\n")

        self.finalize(metrics)

    def finalize(self, metrics):
        """Can be used to output statistics about processed data.

        By default outputs new number of entries/hours.
        """
        logging.info("Total number of entries after processing: %d", self.number_of_entries)
        if self.total_duration != -1:
            logging.info("Total audio duration (hours) after processing: %.2f", self.total_duration / 3600)

    @abstractmethod
    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """Needs to be implemented in the derived classes.

        Note that this method should always return a list of objects
        to allow one-to-many mapping (many-to-one is not
        supported in this design).

        Each returned value should be a DataEntry object that will hold
        a dictionary (or anything else that can be json-serialized) with
        the actual data + any additional metrics required for statistcs
        reporting. Those metrics can be used in :meth:`finalize` to
        prepare for final reporting.

        TODO: it would be more strightforward to use a generator here, but
            seems that it's not supported with multiprocessing. Is there a
            way to make it work?
        """
