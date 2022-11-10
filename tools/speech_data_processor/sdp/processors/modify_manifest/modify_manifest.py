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


from abc import abstractmethod
from typing import Dict, List, Optional

from sdp.processors.base_processor import BaseParallelProcessor
from sdp.utils.edit_spaces import add_start_end_spaces, remove_extra_spaces


class ModifyManifestTextProcessor(BaseParallelProcessor):
    """Base class useful for most "text-only" modifications of the manifest.

    This adds the following functionality on top of BaseParallelProcessor
        - Adds space in the beginning and end of sentence for easier regex-based
          processing.
        - Automatically handles common test cases by comparing input to output
          values.

    Args:
        test_cases: an optional list of dicts containing test cases for checking 
            that the processor makes the changes that we are expecting.
            The dicts must have a key 'input', the value of which is a dictionary
            containing data which is our test input manifest line, and a key 
            'output', the value of which is a dictionary containing data which is
            the expected output manifest line.

    .. note::
        This class only supports one-to-one or one-to-none mappings.
    """

    def __init__(self, test_cases: Optional[List[Dict]] = None, **kwargs):
        super().__init__(**kwargs)
        self.test_cases = test_cases

    def test(self):
        for test_case in self.test_cases:
            generated_outputs = self.process_dataset_entry(test_case["input"])
            # can only return 1 or zero entries
            if len(generated_outputs) == 1:
                generated_output = generated_outputs[0].data
            else:
                generated_output = None
            if generated_output != test_case["output"]:
                raise RuntimeError(
                    "Runtime test failed.\n"
                    f"Test input: {test_case['input']}\n"
                    f"Generated output: {generated_output}\n"
                    f"Expected output: {test_case['output']}"
                )

    @abstractmethod
    def _process_dataset_entry(self, data_entry):
        pass

    def process_dataset_entry(self, data_entry):
        """Wrapper for 'process_dataset_entry' abstract method.

        Before 'process_dataset_entry' is called, the function
        'add_start_end_spaces' is applied to the "text" and "pred_text" in
        the input data.
        After 'process_dataset_entry' is called, the function
        'remove_extra_spaces' is applied to the "text" and "pred_text" to
        the output 'data' variable of the 'process_dataset_entry' method.
        """
        # handle spaces
        if "text" in data_entry:
            data_entry["text"] = add_start_end_spaces(data_entry["text"])
        if "pred_text" in data_entry:
            data_entry["pred_text"] = add_start_end_spaces(data_entry["pred_text"])

        data_entries = self._process_dataset_entry(data_entry)
        if len(data_entries) > 1:
            raise RuntimeError("Cannot handle one-to-many relation")

        if len(data_entries) == 1 and data_entries[0].data is not None:
            # handle spaces
            if "text" in data_entries[0].data:
                data_entries[0].data["text"] = remove_extra_spaces(data_entries[0].data["text"])
            if "pred_text" in data_entries[0].data:
                data_entries[0].data["pred_text"] = remove_extra_spaces(data_entries[0].data["pred_text"])

        return data_entries
