# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from enum import Enum


class PrettyStrEnum(Enum):
    """
    Pretty enum to work with string values for config options with choices
    Provides en automatic error message with possible values, if the value is not in the enum
    Converting to string will show the actual string value, which makes serialization/deserialization straightforward

    Example:
        class ASRModelType(PrettyStrEnum):
            CTC = "ctc"
            RNNT = "rnnt"
        ...
        model_type = ModelType(model_type_string)  # automatically validated
        if model_type == ModelType.CTC:  # more error-prone (to typos) compared to pure string literals
            ...  # do something specific to CTC model
    """

    def __str__(self):
        return self.value

    @classmethod
    def _missing_(cls, value: object):
        choices = ', '.join(map(str, cls))
        raise ValueError(f"{value} is not a valid {cls.__name__}. Possible choices: {choices}")
