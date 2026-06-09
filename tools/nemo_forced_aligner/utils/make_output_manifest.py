# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


def write_manifest_out_line(
    f_manifest_out,
    utt_obj,
):

    data = {"audio_filepath": utt_obj.audio_filepath}
    if not utt_obj.text is None:
        data["text"] = utt_obj.text

    if not utt_obj.pred_text is None:
        data["pred_text"] = utt_obj.pred_text

    for key, val in utt_obj.saved_output_files.items():
        data[key] = val

    new_line = json.dumps(data)
    f_manifest_out.write(f"{new_line}\n")

    return None
