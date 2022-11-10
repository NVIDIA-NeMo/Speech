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

# let's import all supported processors here to simplify target specification
from sdp.processors.asr_inference import ASRInference
from sdp.processors.create_initial_manifest.create_initial_manifest_mls import CreateInitialManifestMLS
from sdp.processors.modify_manifest.data_to_data import (
    InsIfASRInsertion,
    SubIfASRSubstitution,
    SubMakeLowercase,
    SubRegex,
    SubSubstringToSpace,
    SubSubstringToSubstring,
)
from sdp.processors.modify_manifest.data_to_dropbool import (
    DropASRErrorBeginningEnd,
    DropHighCER,
    DropHighLowCharrate,
    DropHighLowDuration,
    DropHighLowWordrate,
    DropHighWER,
    DropIfRegexInAttribute,
    DropIfSubstringInAttribute,
    DropIfSubstringInInsertion,
    DropIfTextIsEmpty,
    DropLowWordMatchRate,
    DropNonAlphabet,
)
from sdp.processors.write_manifest import WriteManifest
