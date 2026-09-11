# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import os
from argparse import ArgumentParser
from typing import Dict

from syllabify import syllabify


"""
Usage:
    cd NeMo/scripts && python dataset_processing/g2p/convert_cmu_arpabet_to_ipa.py
"""


def parse_args():
    parser = ArgumentParser("ARPABET to IPA conversion sctipt")
    parser.add_argument(
        '--cmu_arpabet',
        help="Path to CMU ARPABET dictionary file",
        type=str,
        default="tts_dataset_files/cmudict-0.7b_nv22.10",
    )
    parser.add_argument("--ipa_out", help="Path to save IPA version of the dictionary", type=str, required=True)
    parser.add_argument(
        "--mapping",
        help="ARPABET to IPA phoneme mapping file",
        type=str,
        default="tts_dataset_files/cmudict-arpabet_to_ipa_nv22.10.tsv",
    )
    return parser.parse_args()


def convert_arp_to_ipa(arp_to_ipa_dict: Dict[str, str], arp_input: str, remove_space: bool = False) -> str:
    """
    Converts ARPABET phoneme to IPA based on arp_to_ipa_dict mapping

    Args:
        arp_to_ipa_dict: ARPABET to IPA phonemes mapping
        arp_input: ARPABET input
        remove_space: set to TRUE to remove spaces between IPA phonemes

    Returns:
        input word in IPA form
    """

    primary_stress = "ˈ"
    secondary_stress = "ˌ"
    stress_dict = {"0": "", "1": primary_stress, "2": secondary_stress}

    word_ipa = ""
    phonemes = arp_input.split()

    # split ARPABET phoneme input into syllables,
    # e.g. syllabify(["HH", "AH0", "L", "OW1"]) -> [(['HH'], ['AH0'], []), (['L'], ['OW1'], [])]
    syllables = syllabify(phonemes)

    for syl_idx, syll in enumerate(syllables):
        syll_stress = ""
        syll_ipa = ""

        # syll is a tuple of lists of phonemes, here we flatten it and get rid of empty entries,
        # e.g. (['HH'], ['AH0'], []) -> ['HH', 'AH0']
        syll = [x for x in itertools.chain.from_iterable(syll)]
        for phon_idx, phon in enumerate(syll):
            if phon[-1].isdigit():
                syll_stress = phon[-1]
                if syll_stress not in stress_dict:
                    raise ValueError(f"{syll_stress} unknown")
                syll_stress = stress_dict[syll_stress]

            # some phonemes are followed by a digit that represents stress, e.g., `AH0`
            if phon not in arp_to_ipa_dict and phon[-1].isdigit():
                phon = phon[:-1]

            if phon not in arp_to_ipa_dict:
                raise ValueError(f"|{phon}| phoneme not found in |{arp_input}|")
            else:
                ipa_phone = arp_to_ipa_dict[phon]
                syll_ipa += ipa_phone + " "

        word_ipa += " " + syll_stress + syll_ipa.strip()

    word_ipa = word_ipa.strip()
    if remove_space:
        word_ipa = word_ipa.replace(" ", "")
    return word_ipa


def _get_arpabet_to_ipa_mapping(arp_ipa_map_file: str) -> Dict[str, str]:
    """
    arp_ipa_map_file: Arpabet to IPA phonemes mapping
    """
    arp_to_ipa = {}
    with open(arp_ipa_map_file, "r", encoding="utf-8") as f:
        for line in f:
            arp, ipa = line.strip().split("\t")
            arp_to_ipa[arp] = ipa
    return arp_to_ipa


def convert_cmu_arpabet_to_ipa(arp_ipa_map_file: str, arp_dict_file: str, output_ipa_file: str):
    """
    Converts CMU ARPABET-based dictionary to IPA.

    Args:
        arp_ipa_map_file: ARPABET to IPA phoneme mapping file
        arp_dict_file: path to ARPABET version of CMU dictionary
        output_ipa_file: path to output IPA version of CMU dictionary
    """
    arp_to_ipa_dict = _get_arpabet_to_ipa_mapping(arp_ipa_map_file)
    with open(arp_dict_file, "r", encoding="utf-8") as f_arp, open(output_ipa_file, "w", encoding="utf-8") as f_ipa:
        for line in f_arp:
            if line.startswith(";;;"):
                f_ipa.write(line)
            else:
                # First, split the line at " #" if there are comments in the dictionary file following the mapping entries.
                # Next, split at default "  " separator.
                graphemes, phonemes = line.split(" #")[0].strip().split("  ")
                ipa_form = convert_arp_to_ipa(arp_to_ipa_dict, phonemes, remove_space=True)
                f_ipa.write(f"{graphemes}  {ipa_form}\n")

    print(f"IPA version of {os.path.abspath(arp_dict_file)} saved in {os.path.abspath(output_ipa_file)}")


if __name__ == "__main__":
    args = parse_args()
    convert_cmu_arpabet_to_ipa(args.mapping, args.cmu_arpabet, args.ipa_out)
