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

import argparse
import json
import logging
import os

import librosa
import numpy as np
from joblib import Parallel, delayed
from scipy.io import wavfile
from tqdm import tqdm
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

parser = argparse.ArgumentParser(description='Create synthetic code-switching data audio data from monolingual data')
parser.add_argument("--manifest_path", default=None, type=str, help='Path to CS indermediate manifest', required=True)
parser.add_argument(
    "--audio_save_folder_path",
    default=None,
    type=str,
    help='Path to directory where created synthetic set would be saved',
    required=True,
)
parser.add_argument(
    "--manifest_save_path", default=None, type=str, help='Path to save the created manifest', required=True
)
parser.add_argument(
    "--audio_normalized_amplitude", default=15000, type=int, help='Normalized amplitdue of audio samples'
)
parser.add_argument(
    "--cs_data_sampling_rate",
    default=16000,
    type=int,
    help='Desired sampling rate for the audios in the generated dataset',
)
parser.add_argument(
    "--sample_beginning_pause_msec",
    default=20,
    type=int,
    help='Pause to be added at the beginning of the sample (msec)',
)
parser.add_argument(
    "--sample_joining_pause_msec",
    default=100,
    type=int,
    help='Pause to be added between different phrases of the sample (msec)',
)
parser.add_argument(
    "--sample_end_pause_msec", default=20, type=int, help='Pause to be added at the end of the sample (msec)'
)
parser.add_argument(
    "--is_lid_manifest",
    default=True,
    type=bool,
    help='If true, generate manifest in the multi-sample lid format, else the standard manifest format',
)
parser.add_argument("--workers", default=1, type=int, help='Number of worker processes')

args = parser.parse_args()


def split_list(input_list: list, num_splits: int):
    """
    Args:
        input_list: the input list to split
        num_splits: number of splits required

    Returns:
        iterator of split lists

    """
    k, m = divmod(len(input_list), num_splits)
    return (input_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_splits))


def combine_manifests(manifest_save_path: str, num_split: int):
    """
    Args:
        manifest_save_path: absolute path to save the combined manifest
        num_splits: number of splits of manifest

    Returns:
        num_samples_combined: the total number of samples in the generated dataset
    """
    num_samples_combined = 0
    base_directory = os.path.dirname(manifest_save_path)

    with open(manifest_save_path, 'w') as outfile:
        for i in range(num_split):
            split_manifest_path = base_directory + '/temp_' + str(i) + '.json'
            data_split = read_manifest(split_manifest_path)

            for elem in data_split:
                s = json.dumps(elem)
                outfile.write(s + '\n')
                num_samples_combined += 1

            # removing the intermediate file
            os.remove(split_manifest_path)

    return num_samples_combined


def create_cs_data(
    intermediate_cs_manifest_list: list,
    audio_save_folder: str,
    manfest_save_path: str,
    audio_amplitude_normalization: int,
    pause_beg_msec: int,
    pause_join_msec: int,
    pause_end_msec: int,
    cs_data_sampling_rate: int,
    is_lid_manifest: bool,
):
    """
    Args:
        intermediate_cs_manifest_list: the intermediate cs manifest obtained from code_switching_manifest_creation.py as a list
        audio_save_folder: Absolute path to save the generated audio samples
        manfest_save_path: Absolute path to save the corresponding manifest
        audio_amplitude_normalization: The amplitude to scale to after normalization
        pause_beg_msec: Pause to be added at the beginning of the sample (msec)
        pause_join_msec: Pause to be added between different phrases of the sample (msec)
        pause_end_msec: Pause to be added at the end of the sample (msec)
        cs_data_sampling_rate: Desired sampling rate of the generated samples
        is_lid_manifest: If true, generate manifest in the multi-sample lid format, else the standard manifest format

    Returns:

    """

    fs = cs_data_sampling_rate
    incorrect_sample_flag = 0

    with open(manfest_save_path, 'w') as outfile:
        for data in tqdm(intermediate_cs_manifest_list):

            combined_audio = []

            staring_pause = np.zeros(int(pause_beg_msec * fs / 1000))
            combined_audio += list(staring_pause)

            text_entry_list = []
            for index in range(len(data['lang_ids'])):

                phrase_entry = {}
                # dictionary to store the phrase information which will be added to the complete sentence

                data_sample, fs_sample = librosa.load(data['paths'][index], sr=fs)
                # Alternative-  fs_sample, data_sample = wavfile.read(data['paths'][index])

                if fs_sample != fs:
                    logging.error('Sampling rate error inside create_cs_data function')
                    exit

                # Remove leading and trailing zeros
                data_sample = np.trim_zeros(data_sample)

                # take care of empty arrays: rare
                if data_sample.size == 0:
                    incorrect_sample_flag = 1
                    continue

                # normalizing data
                data_sample_norm = (
                    data_sample
                    / np.maximum(np.abs(data_sample.max()), np.abs(data_sample.min()))
                    * audio_amplitude_normalization
                )

                combined_audio += list(data_sample_norm)

                phrase_entry['str'] = data['texts'][index]
                phrase_entry['lang'] = data['lang_ids'][index]

                text_entry_list.append(phrase_entry)

                # adding small pause between semgments
                if index != (len(data['lang_ids']) - 1):
                    pause = np.zeros(int(pause_join_msec * fs / 1000))
                    combined_audio += list(pause)

            if incorrect_sample_flag == 1:
                incorrect_sample_flag = 0
                continue

            ending_pause = np.zeros(int(pause_end_msec * fs / 1000))
            combined_audio += list(ending_pause)

            sample_id = data['uid']
            audio_file_path = audio_save_folder + '/' + str(sample_id) + ".wav"

            # saving audio
            wavfile.write(audio_file_path, fs, np.array(combined_audio).astype(np.int16))
            # Alternative-  librosa.output.write_wav(audio_file_path, combined_audio, fs)

            metadata_json = {}
            metadata_json['audio_filepath'] = audio_file_path
            metadata_json['duration'] = float(len(combined_audio) / fs)
            if is_lid_manifest:
                metadata_json['text'] = text_entry_list
            else:
                metadata_json['text'] = ' '.join(data['texts'])

            metadata_json['language_ids'] = data['lang_ids']
            metadata_json['original_texts'] = data['texts']
            metadata_json['original_paths'] = data['paths']
            metadata_json['original_durations'] = data['durations']

            s = json.dumps(metadata_json)
            outfile.write(s + '\n')


def main():

    cs_intermediate_manifest_path = args.manifest_path
    audio_save_folder = args.audio_save_folder_path
    manifest_save_path = args.manifest_save_path
    audio_amplitude_normalization = args.audio_normalized_amplitude
    pause_beg_msec = args.sample_beginning_pause_msec
    pause_join_msec = args.sample_joining_pause_msec
    pause_end_msec = args.sample_end_pause_msec
    cs_data_sampling_rate = args.cs_data_sampling_rate
    is_lid_manifest = args.is_lid_manifest
    num_process = args.workers

    # Sanity Checks
    if (cs_intermediate_manifest_path is None) or (not os.path.exists(cs_intermediate_manifest_path)):
        logging.error('Please provide correct CS manifest (obtained from code_switching_manifest_creation.py)')
        exit

    if (audio_save_folder is None) or (not os.path.exists(audio_save_folder)):
        logging.error('audio_save_folder_path is incorrect or does not exist')
        exit

    if manifest_save_path is None:
        logging.error('Please provide valid manifest_save_path')
        exit

    # Reading data
    logging.info('Reading manifests')
    intermediate_cs_manifest = read_manifest(cs_intermediate_manifest_path)

    # Spliting the data
    data_split = split_list(intermediate_cs_manifest, num_process)

    # Creating Audio data
    logging.info('Creating synthetic audio data')
    base_directory = os.path.dirname(manifest_save_path)

    Parallel(n_jobs=num_process)(
        delayed(create_cs_data)(
            split_manifest,
            audio_save_folder,
            base_directory + '/temp_' + str(idx) + '.json',
            audio_amplitude_normalization,
            pause_beg_msec,
            pause_join_msec,
            pause_end_msec,
            cs_data_sampling_rate,
            is_lid_manifest,
        )
        for idx, split_manifest in enumerate(data_split)
    )

    # Combining manifests
    num_samples_combined = combine_manifests(manifest_save_path, num_process)

    print("Synthetic CS audio data saved at :", audio_save_folder)
    print("Synthetic CS manifest saved at :", manifest_save_path)
    print("Total number of samples in the generated dataset :", str(num_samples_combined))

    logging.info('Done!')


if __name__ == "__main__":
    main()
