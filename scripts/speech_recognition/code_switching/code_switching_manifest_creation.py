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

import argparse
import logging
import os
import random
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

# Checks -
# (Recommendation) Please normalize the text for each language (avoid numbers, special characters, punctuation)
# Please ensure that the audio_filepaths are absolute locations


parser = argparse.ArgumentParser(description='Create synthetic code-switching data manifest from monolingual data')

parser.add_argument("--manifest_language1", default=None, type=str, help='Manifest file for language 1', required=True)
parser.add_argument("--manifest_language2", default=None, type=str, help='Manifest file for language 2', required=True)
parser.add_argument(
    "--manifest_save_path", default=None, type=str, help='Path to save created CS indermediate manifest', required=True
)
parser.add_argument(
    "--id_language1", default=None, type=str, help='Identifier for language 1, eg: en, es, hi', required=True
)
parser.add_argument(
    "--id_language2", default=None, type=str, help='Identifier for language 2, eg: en, es, hi', required=True
)
parser.add_argument("--max_sample_duration_sec", default=19, type=int, help='Maximum duration of sample (sec)')
parser.add_argument("--min_sample_duration_sec", default=16, type=int, help='Minimum duration of sample (sec)')
parser.add_argument("--dataset_size_required_hrs", default=1, type=int, help='Duration of dataset required (hrs)')

args = parser.parse_args()


def create_cs_manifest(
    data_lang_0: list,
    data_lang_1: list,
    lid_lang_0: str,
    lid_lang_1: str,
    max_sample_duration_sec: int,
    min_sample_duration_sec: int,
    data_requirement_hrs: int,
):
    """
    Args:
        data_lang_0: Manifest entries from first langauge
        data_lang_1: Manifest entries from second langauge
        lid_lang_0: Language ID marker for first langauge
        lid_lang_1: Language ID marker for second langauge
        max_sample_duration_sec: Maximum permissible duration of generated CS sample in sec
        min_sample_duration_sec: Minimum permissible duration of generated CS sample in sec
        data_requirement_hrs: Required size of generated corpus

    Returns:
        Created synthetic CS manifest as list

    """

    total_duration = 0
    constructed_data = []
    sample_id = 0

    num_samples_lang0 = len(data_lang_0)
    num_samples_lang1 = len(data_lang_1)

    while total_duration < (data_requirement_hrs * 3600):

        created_sample_duration_sec = 0
        created_sample_dict = {}
        created_sample_dict['lang_ids'] = []
        created_sample_dict['texts'] = []
        created_sample_dict['paths'] = []
        created_sample_dict['durations'] = []

        while created_sample_duration_sec < min_sample_duration_sec:

            lang_selection = random.randint(0, 1)

            if lang_selection == 0:
                index = random.randint(0, num_samples_lang0 - 1)
                sample = data_lang_0[index]
                lang_id = lid_lang_0
            else:
                index = random.randint(0, num_samples_lang1 - 1)
                sample = data_lang_1[index]
                lang_id = lid_lang_1

            if (created_sample_duration_sec + sample['duration']) > max_sample_duration_sec:
                continue
            else:
                created_sample_duration_sec += sample['duration']
                created_sample_dict['lang_ids'].append(lang_id)
                created_sample_dict['texts'].append(sample['text'])
                created_sample_dict['paths'].append(sample['audio_filepath'])
                created_sample_dict['durations'].append(sample['duration'])

        created_sample_dict['total_duration'] = created_sample_duration_sec

        # adding a uid which will be used to save the generated audio file later
        created_sample_dict['uid'] = sample_id
        sample_id += 1

        constructed_data.append(created_sample_dict)
        total_duration += created_sample_duration_sec

    return constructed_data


def main():

    manifest0 = args.manifest_language1
    manifest1 = args.manifest_language2
    lid0 = args.id_language1
    lid1 = args.id_language2
    min_sample_duration = args.min_sample_duration_sec
    max_sample_duration = args.max_sample_duration_sec
    dataset_requirement = args.dataset_size_required_hrs
    manifest_save_path = args.manifest_save_path

    # Sanity Checks
    if (manifest0 is None) or (not os.path.exists(manifest0)):
        logging.error('Manifest for language 1 is incorrect')
        exit

    if (manifest1 is None) or (not os.path.exists(manifest1)):
        logging.error('Manifest for language 2 is incorrect')
        exit

    if lid0 is None:
        logging.error('Please provide correct language code for language 1')
        exit

    if lid1 is None:
        logging.error('Please provide correct language code for language 2')
        exit

    if manifest_save_path is None:
        logging.error('Please provide correct manifest save path')
        exit

    if min_sample_duration >= max_sample_duration:
        logging.error('Please ensure max_sample_duration > min_sample_duration')
        exit

    # Reading data
    logging.info('Reading manifests')
    data_language0 = read_manifest(manifest0)
    data_language1 = read_manifest(manifest1)

    # Creating the CS data Manifest
    logging.info('Creating CS manifest')
    constructed_data = create_cs_manifest(
        data_language0, data_language1, lid0, lid1, max_sample_duration, min_sample_duration, dataset_requirement
    )

    # Saving Manifest
    logging.info('saving manifest')
    write_manifest(manifest_save_path, constructed_data)

    print("Synthetic CS manifest saved at :", manifest_save_path)

    logging.info('Done!')


if __name__ == "__main__":
    main()
