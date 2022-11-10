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

import json
from typing import List

from sdp.processors.base_processor import BaseProcessor
from tqdm import tqdm


class WriteManifest(BaseProcessor):
    """
    Saves a copy of a manifest but only with the fields specified in fields_to_save.

    Args:
        output_manifest_file: path of where the output file will be saved.
        input_manifest_file: path of where the input file that we will be copying is saved.
        fields_to_save: list of the fields in the input manifest that we want to copy over. 
            The output file will only contain these fields.
    """

    def __init__(self, output_manifest_file: str, input_manifest_file: str, fields_to_save: List[str]):
        self.output_manifest_file = output_manifest_file
        self.input_manifest_file = input_manifest_file
        self.fields_to_save = fields_to_save

    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin, open(
            self.output_manifest_file, "wt", encoding="utf8"
        ) as fout:
            for line in tqdm(fin):
                line = json.loads(line)
                new_line = {field: line[field] for field in self.fields_to_save}
                fout.write(json.dumps(new_line) + "\n")
