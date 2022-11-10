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

import collections
import os
import tempfile
import urllib.request
from typing import List

import pandas as pd
from sdp.processors.base_processor import DataEntry
from sdp.processors.modify_manifest.modify_manifest import ModifyManifestTextProcessor

from nemo.utils import logging

NUMERAL_DATA_DOWNLOAD_PATH = (
    "https://github.com/NVIDIA/NeMo/releases/download/v1.0.0rc1/1-100_roman_numeral_table_spanish.csv"
)


def download_numeral_data(tgt_path):
    urllib.request.urlretrieve(NUMERAL_DATA_DOWNLOAD_PATH, tgt_path)

    return None


class CleanRomanNumerals(ModifyManifestTextProcessor):
    def __init__(
        self,
        king_triggers,
        queen_triggers,
        ordinal_masc_triggers,
        ordinal_fem_triggers,
        cardinal_triggers,
        numerals_data_path=None,
        save_changes_in=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.numerals_data_path = numerals_data_path
        self.king_triggers = king_triggers
        self.queen_triggers = queen_triggers
        self.ordinal_masc_triggers = ordinal_masc_triggers
        self.ordinal_fem_triggers = ordinal_fem_triggers
        self.cardinal_triggers = cardinal_triggers

        # read csv
        if self.numerals_data_path:
            if not os.path.isfile(self.numerals_data_path):
                download_numeral_data(self.numerals_data_path)
                df = pd.read_csv(self.numerals_data_path, sep="\t", index_col=0)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, "temporary_numeral_data_download.csv")
                download_numeral_data(temp_file)
                df = pd.read_csv(temp_file, sep="\t", index_col=0)

        self.roman_numeral_to_ordinal_masc = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_ordinal_masc[row["roman"]] = row["ordinal_masc"].strip()

        self.roman_numeral_to_ordinal_fem = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_ordinal_fem[row["roman"]] = row["ordinal_fem"].strip()

        self.roman_numeral_to_cardinal = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_cardinal[row["roman"]] = row["cardinal"].strip()

        self.roman_numeral_to_king = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_king[row["roman"]] = row["king"].strip()

        self.roman_numeral_to_queen = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_queen[row["roman"]] = row["queen"].strip()

        self.clean_roman_numerals_count = collections.defaultdict(int)

    def _process_dataset_entry(self, data_entry) -> List:
        data_entry = self.clean_operation(data_entry, self.ordinal_masc_triggers, self.roman_numeral_to_ordinal_masc)
        data_entry = self.clean_operation(data_entry, self.ordinal_fem_triggers, self.roman_numeral_to_ordinal_fem)
        data_entry = self.clean_operation(data_entry, self.cardinal_triggers, self.roman_numeral_to_cardinal)
        data_entry = self.clean_operation(data_entry, self.king_triggers, self.roman_numeral_to_king)
        data_entry = self.clean_operation(data_entry, self.queen_triggers, self.roman_numeral_to_queen)
        return [DataEntry(data=data_entry, metrics=self.clean_roman_numerals_count)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logging.info("Num of roman numeral substitutions")
        total_counter_sorted = dict(sorted(total_counter.items(), key=lambda x: x[1], reverse=True,))
        for word, count in total_counter_sorted.items():
            logging.info(f"{word} {count}")
        super().finalize(metrics)

    def clean_operation(self, data, triggers, roman_numeral_to_num_written):
        for trigger in triggers:
            if trigger in data["text"]:
                for roman_numeral, num_written in roman_numeral_to_num_written.items():
                    noun_roman = f" {trigger} {roman_numeral} "
                    if noun_roman in data["text"]:
                        noun_number = f" {trigger} {num_written} "
                        data["text"] = data["text"].replace(noun_roman, noun_number)
                        self.clean_roman_numerals_count[noun_roman] += 1
        return data
