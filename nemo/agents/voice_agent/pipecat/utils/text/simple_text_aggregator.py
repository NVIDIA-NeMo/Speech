# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class SimpleSegmentedTextAggregator(SimpleTextAggregator):
    def __init__(self, punctuation_marks: str | list[str] = ",!?", **kwargs):
        super().__init__(**kwargs)
        if not punctuation_marks:
            self._punctuation_marks = set()
        else:
            self._punctuation_marks = set(punctuation_marks)

    def _find_segment_end(self, text: str) -> Optional[int]:
        for punc in self._punctuation_marks:
            idx = text.find(punc)
            if idx != -1:
                return idx
        return None

    async def aggregate(self, text: str) -> Optional[str]:
        result: Optional[str] = None

        self._text += text

        self._text = self._text.replace("*", "")

        eos_end_marker = match_endofsentence(self._text)

        if not eos_end_marker:
            eos_end_marker = self._find_segment_end(self._text)

        if eos_end_marker:
            result = self._text[:eos_end_marker]
            self._text = self._text[eos_end_marker:]

        return result
