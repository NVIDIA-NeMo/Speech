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

import argparse
import json
import os

"""
Create a dataset with five Lambada test examples for functional testing. Each line
contains a dictionary with a "text_before_last_word" and "last_word" keys.
"""


def create_sample_lambada(output_file: str, overwrite: bool = False):
    """Create JSON file with a few Lambada examples."""
    if os.path.isfile(output_file) and not overwrite:
        print(f"File {output_file} exists and overwrite flag is not set so exiting.")
        return

    texts = [
        {
            "text_before_last_word": "In my palm is a clear stone , and inside it is a small ivory statuette . A guardian angel .\n\n\" Figured if you re going to be out at night getting hit by cars , you might as well have some backup .\"\n\n I look at him , feeling stunned . Like this is some sort of sign . But as I stare at Harlin , his mouth curved in a confident grin , I don t care about",
            "last_word": "signs",
        },
        {
            "text_before_last_word": "Give me a minute to change and I ll meet you at the docks .\" She d forced those words through her teeth .\n\n\" No need to change . We won t be that long .\"\n\n Shane gripped her arm and started leading her to the dock .\n\n\" I can make it there on my own ,",
            "last_word": "Shane",
        },
        {
            "text_before_last_word": "\" Only one source I know of that would be likely to cough up enough money to finance a phony sleep research facility and pay people big bucks to solve crimes in their dreams ,\" Farrell concluded dryly .\n\n\" What can I say ?\" Ellis unfolded his arms and widened his hands . \" Your tax dollars at work .\"\n\n Before Farrell could respond , Leila s voice rose from inside the house .\n\n\" No insurance ?\" she wailed . \" What do you mean you don t have any",
            "last_word": "insurance",
        },
        {
            "text_before_last_word": "Helen s heart broke a little in the face of Miss Mabel s selfless courage . She thought that because she was old , her life was of less value than the others . For all Helen knew , Miss Mabel had a lot more years to live than she did . \" Not going to happen ,\" replied",
            "last_word": "Helen",
        },
        {
            "text_before_last_word": "Preston had been the last person to wear those chains , and I knew what I d see and feel if they were slipped onto my skin the Reaper s unending hatred of me . I d felt enough of that emotion already in the amphitheater . I didn t want to feel anymore .\n\n\" Don t put those on me ,\" I whispered . \" Please .\"\n\n Sergei looked at me , surprised by my low , raspy please , but he put down the",
            "last_word": "chains",
        },
    ]

    print(f"Writing {len(texts)} line(s) to {output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, mode="w", encoding="utf-8") as f:
        json.dump(texts, f)
    print("OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create a sample from Lambada test dataset.")
    parser.add_argument("--output_file", required=True, help="Output file name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite file if it exists")
    args = parser.parse_args()
    create_sample_lambada(args.output_file, args.overwrite)
