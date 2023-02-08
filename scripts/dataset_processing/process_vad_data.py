# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
"""
Usage:

python process_vad_data.py \
    --out_dir=<output path to where the generated manifest should be stored> \
    --speech_data_root=<path where the speech data are stored> \
    --background_data_root=<path where the background data are stored> \
    --rebalance_method=<'under' or 'over' or 'fixed'> \ 
    --log
    (Optional --demo (for demonstration in tutorial). If you want to use your own background noise data, make sure to delete --demo)
"""
import argparse
import glob
import json
import logging
import os
import tarfile
import urllib.request

import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

sr = 16000

# google speech command v2
URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource

    Returns:

    """
    if not os.path.exists(destination):
        logging.info(f"{destination} does not exist. Downloading ...")
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info(f"Downloaded {destination}.")
    else:
        logging.info(f"Destination {destination} exists. Skipping.")
    return destination


def extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __extract_all_files(filepath: str, data_root: str, data_dir: str):
    if not os.path.exists(data_dir):
        extract_file(filepath, data_dir)
    else:
        logging.info(f'Skipping extracting. Data already there {data_dir}')


def split_train_val_test(data_dir, file_type, test_size=0.1, val_size=0.1, demo=False):
    X = []
    if file_type == "speech":
        for o in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, o)) and o.split("/")[-1] != "_background_noise_":
                X.extend(glob.glob(os.path.join(data_dir, o) + '/*.wav'))

        if demo:
            logging.info(
                f"For Demonstration, we use {int(len(X)/100)}/{len(X)} speech data. Make sure to remove --demo flag when you actually train your model!"
            )
            X = np.random.choice(X, int(len(X) / 100), replace=False)

    else:
        for o in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, o)):
                X.extend(glob.glob(os.path.join(data_dir, o) + '/*.wav'))
            else:  # for using "_background_noise_" from google speech commands as background data
                if o.endswith(".wav"):
                    X.append(os.path.join(data_dir, o))

    X_train, X_test = train_test_split(X, test_size=test_size, random_state=1)
    val_size_tmp = val_size / (1 - test_size)
    X_train, X_val = train_test_split(X_train, test_size=val_size_tmp, random_state=1)

    with open(os.path.join(data_dir, file_type + "_training_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_train))
    with open(os.path.join(data_dir, file_type + "_testing_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_test))
    with open(os.path.join(data_dir, file_type + "_validation_list.txt"), "w") as outfile:
        outfile.write("\n".join(X_val))

    logging.info(f'Overall: {len(X)}, Train: {len(X_train)}, Validatoin: {len(X_val)}, Test: {len(X_test)}')
    logging.info(f"Finish spliting train, val and test for {file_type}. Write to files!")


def process_google_speech_train(data_dir):
    X = []
    for o in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, o)) and o.split("/")[-1] != "_background_noise_":
            X.extend(glob.glob(os.path.join(data_dir, o) + '/*.wav'))

    short_files = [i.split(data_dir)[1] for i in files]

    with open(os.path.join(data_dir, 'testing_list.txt'), 'r') as allfile:
        testing_list = allfile.read().splitlines()

    with open(os.path.join(data_dir, 'validation_list.txt'), 'r') as allfile:
        validation_list = allfile.read().splitlines()

    exist_set = set(testing_list).copy()
    exist_set.update(set(validation_list))

    training_list = [i for i in short_files if i not in exist_set]

    with open(os.path.join(data_dir, "training_list.txt"), "w") as outfile:
        outfile.write("\n".join(training_list))

    logging.info(
        f'Overall: {len(files)}, Train: {len(training_list)}, Validatoin: {len(validation_list)}, Test: {len(testing_list)}'
    )


def write_manifest(
    out_dir,
    files,
    prefix,
    manifest_name,
    start=0.0,
    end=None,
    duration_stride=1.0,
    duration_max=None,
    duration_limit=100.0,
    filter_long=False,
):
    """
    Given a list of files, segment each file and write them to manifest with restrictions.
    Args:
        out_dir: directory of generated manifest
        files: list of files to be processed
        prefix: label of samples
        manifest_name: name of generated manifest
        start: beginning of audio of generating segment
        end: end of audio of generating segment
        duration_stride: stride for segmenting audio samples
        duration_max: duration for each segment
        duration_limit: duration threshold for filtering out long audio samples
        filter_long: boolean to determine whether to filter out long audio samples
    Returns:
    """
    seg_num = 0
    skip_num = 0
    if duration_max is None:
        duration_max = 1e9

    if not os.path.exists(out_dir):
        logging.info(f'Outdir {out_dir} does not exist. Creat directory.')
        os.mkdir(out_dir)

    output_path = os.path.join(out_dir, manifest_name + '.json')
    with open(output_path, 'w') as fout:
        for file in files:
            label = prefix

            try:
                x, _sr = librosa.load(file, sr=sr)
                duration = librosa.get_duration(y=x, sr=sr)

            except Exception:
                continue

            if filter_long and duration > duration_limit:
                skip_num += 1
                continue

            offsets = []
            durations = []

            if duration > duration_max:
                current_offset = start

                while current_offset < duration:
                    if end is not None and current_offset > end:
                        break

                    difference = duration - current_offset

                    if difference < duration_max:
                        break

                    offsets.append(current_offset)
                    durations.append(duration_max)

                    current_offset += duration_stride

            else:
                # Duration is not long enough! Skip
                skip_num += 1

            for duration, offset in zip(durations, offsets):
                metadata = {
                    'audio_filepath': file,
                    'duration': duration,
                    'label': label,
                    'text': '_',  # for compatibility with ASRAudioText
                    'offset': offset,
                }
                json.dump(metadata, fout)
                fout.write('\n')
                fout.flush()
                seg_num += 1
    return skip_num, seg_num, output_path


def load_list_write_manifest(
    data_dir,
    out_dir,
    filename,
    prefix,
    start,
    end,
    duration_stride=1.0,
    duration_max=1.0,
    duration_limit=100.0,
    filter_long=True,
):

    filename = prefix + '_' + filename
    file_path = os.path.join(data_dir, filename)

    with open(file_path, 'r') as allfile:
        files = allfile.read().splitlines()

    manifest_name = filename.split('_list.txt')[0] + '_manifest'
    skip_num, seg_num, output_path = write_manifest(
        out_dir,
        files,
        prefix,
        manifest_name,
        start,
        end,
        duration_stride,
        duration_max,
        duration_limit,
        filter_long=True,
    )
    return skip_num, seg_num, output_path


def rebalance_json(data_dir, data_json, num, prefix):
    data = []
    seg = 0
    with open(data_json, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    filename = data_json.split('/')[-1]
    fout_path = os.path.join(data_dir, prefix + "_" + filename)

    if len(data) >= num:
        selected_sample = np.random.choice(data, num, replace=False)
    else:
        selected_sample = np.random.choice(data, num, replace=True)

    with open(fout_path, 'a') as fout:
        for i in selected_sample:
            seg += 1
            json.dump(i, fout)
            fout.write('\n')
            fout.flush()

    logging.info(f'Get {seg}/{num} to  {fout_path} from {data_json}')
    return fout_path


def generate_variety_noise(data_dir, filename, prefix):

    curr_dir = data_dir.split("_background_noise_")[0]
    silence_path = os.path.join(curr_dir, "_background_noise_more")

    if not os.path.exists(silence_path):
        os.mkdir(silence_path)

    silence_stride = 1000  # stride = 1/16 seconds
    sampling_rate = 16000

    silence_files = []
    rng = np.random.RandomState(0)

    filename = prefix + '_' + filename
    file_path = os.path.join(data_dir, filename)

    with open(file_path, 'r') as allfile:
        files = allfile.read().splitlines()

    for file in files:
        y, sr = librosa.load(path=file, sr=sampling_rate)

        for i in range(
            0, len(y) - sampling_rate, silence_stride * 100
        ):  # stride * 100 to generate less samples for demo
            file_name = "{}_{}.wav".format(file.split("/")[-1], i)
            y_slice = y[i : i + sampling_rate]
            magnitude = rng.uniform(0.0, 1.0)
            y_slice *= magnitude
            out_file_path = os.path.join(silence_path, file_name)
            sf.write(out_file_path, y_slice, sr)

            silence_files.append(out_file_path)

    new_list_file = os.path.join(silence_path, filename)
    with open(new_list_file, "w") as outfile:
        outfile.write("\n".join(silence_files))

    logging.info(f"Generate {len(out_file_path)} background files for {file_path}. => {new_list_file} !")
    return len(silence_files)


def main():
    parser = argparse.ArgumentParser(description='Speech and backgound data download and preprocess')
    parser.add_argument("--out_dir", required=False, default='./manifest/', type=str)
    parser.add_argument("--speech_data_root", required=True, default=None, type=str)
    parser.add_argument("--background_data_root", required=True, default=None, type=str)
    parser.add_argument('--test_size', required=False, default=0.1, type=float)
    parser.add_argument('--val_size', required=False, default=0.1, type=float)
    parser.add_argument('--window_length_in_sec', required=False, default=0.63, type=float)
    parser.add_argument('--log', required=False, action='store_true')
    parser.add_argument('--rebalance_method', required=False, default=None, type=str)
    parser.add_argument('--demo', required=False, action='store_true')
    parser.set_defaults(log=False, generate=False)
    args = parser.parse_args()

    if not args.rebalance_method:
        rebalance = False
    else:
        if args.rebalance_method != 'over' and args.rebalance_method != 'under' and args.rebalance_method != 'fixed':
            raise NameError("Please select a valid sampling method: over/under/fixed.")
        else:
            rebalance = True

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    # Download speech data
    speech_data_root = args.speech_data_root
    data_set = "google_speech_recognition_v2"
    speech_data_folder = os.path.join(speech_data_root, data_set)

    background_data_folder = args.background_data_root
    logging.info(f"Working on: {data_set}")

    # Download and extract speech data
    if not os.path.exists(speech_data_folder):
        file_path = os.path.join(speech_data_root, data_set + ".tar.bz2")
        logging.info(f"Getting {data_set}")
        __maybe_download_file(file_path, URL)
        logging.info(f"Extracting {data_set}")
        __extract_all_files(file_path, speech_data_root, speech_data_folder)

    logging.info(f"Split speech data!")
    # dataset provide testing.txt and validation.txt feel free to split data using that with process_google_speech_train
    split_train_val_test(speech_data_folder, "speech", args.test_size, args.val_size, args.demo)

    logging.info(f"Split background data!")
    split_train_val_test(background_data_folder, "background", args.test_size, args.val_size)

    out_dir = args.out_dir

    # Process Speech manifest
    logging.info(f"=== Write speech data to manifest!")
    skip_num_val, speech_seg_num_val, speech_val = load_list_write_manifest(
        speech_data_folder,
        out_dir,
        'validation_list.txt',
        'speech',
        0.2,
        0.8,
        args.window_length_in_sec,
        args.window_length_in_sec,
    )
    skip_num_test, speech_seg_num_test, speech_test = load_list_write_manifest(
        speech_data_folder, out_dir, 'testing_list.txt', 'speech', 0.2, 0.8, 0.01, args.window_length_in_sec
    )
    skip_num_train, speech_seg_num_train, speech_train = load_list_write_manifest(
        speech_data_folder,
        out_dir,
        'training_list.txt',
        'speech',
        0.2,
        0.8,
        args.window_length_in_sec,
        args.window_length_in_sec,
    )

    logging.info(f'Val: Skip {skip_num_val} samples. Get {speech_seg_num_val} segments! => {speech_val} ')
    logging.info(f'Test: Skip {skip_num_test} samples. Get {speech_seg_num_test} segments! => {speech_test}')
    logging.info(f'Train: Skip {skip_num_train} samples. Get {speech_seg_num_train} segments!=> {speech_train}')

    # Process background manifest
    # if we select to generate more background noise data
    if args.demo:
        logging.info("Start generating more background noise data")
        generate_variety_noise(background_data_folder, 'validation_list.txt', 'background')
        generate_variety_noise(background_data_folder, 'training_list.txt', 'background')
        generate_variety_noise(background_data_folder, 'testing_list.txt', 'background')
        background_data_folder = os.path.join(
            background_data_folder.split("_background_noise_")[0], "_background_noise_more"
        )

    logging.info(f"=== Write background data to manifest!")
    skip_num_val, background_seg_num_val, background_val = load_list_write_manifest(
        background_data_folder, out_dir, 'validation_list.txt', 'background', 0, None, 0.15, args.window_length_in_sec
    )
    skip_num_test, background_seg_num_test, background_test = load_list_write_manifest(
        background_data_folder, out_dir, 'testing_list.txt', 'background', 0, None, 0.01, args.window_length_in_sec
    )
    skip_num_train, background_seg_num_train, background_train = load_list_write_manifest(
        background_data_folder, out_dir, 'training_list.txt', 'background', 0, None, 0.15, args.window_length_in_sec
    )

    logging.info(f'Val: Skip {skip_num_val} samples. Get {background_seg_num_val} segments! => {background_val}')
    logging.info(f'Test: Skip {skip_num_test} samples. Get {background_seg_num_test} segments! => {background_test}')
    logging.info(
        f'Train: Skip {skip_num_train} samples. Get {background_seg_num_train} segments! => {background_train}'
    )
    min_val, max_val = min(speech_seg_num_val, background_seg_num_val), max(speech_seg_num_val, background_seg_num_val)
    min_test, max_test = (
        min(speech_seg_num_test, background_seg_num_test),
        max(speech_seg_num_test, background_seg_num_test),
    )
    min_train, max_train = (
        min(speech_seg_num_train, background_seg_num_train),
        max(speech_seg_num_train, background_seg_num_train),
    )

    logging.info('Finish generating manifest!')

    if rebalance:
        # Random Oversampling: Randomly duplicate examples in the minority class.
        # Random Undersampling: Randomly delete examples in the majority class.
        if args.rebalance_method == 'under':
            logging.info(f"Rebalancing number of samples in classes using {args.rebalance_method} sampling.")
            logging.info(f'Val: {min_val} Test: {min_test} Train: {min_train}!')

            rebalance_json(out_dir, background_val, min_val, 'balanced')
            rebalance_json(out_dir, background_test, min_test, 'balanced')
            rebalance_json(out_dir, background_train, min_train, 'balanced')

            rebalance_json(out_dir, speech_val, min_val, 'balanced')
            rebalance_json(out_dir, speech_test, min_test, 'balanced')
            rebalance_json(out_dir, speech_train, min_train, 'balanced')

        if args.rebalance_method == 'over':
            logging.info(f"Rebalancing number of samples in classes using {args.rebalance_method} sampling.")
            logging.info(f'Val: {max_val} Test: {max_test} Train: {max_train}!')

            rebalance_json(out_dir, background_val, max_val, 'balanced')
            rebalance_json(out_dir, background_test, max_test, 'balanced')
            rebalance_json(out_dir, background_train, max_train, 'balanced')

            rebalance_json(out_dir, speech_val, max_val, 'balanced')
            rebalance_json(out_dir, speech_test, max_test, 'balanced')
            rebalance_json(out_dir, speech_train, max_train, 'balanced')

        if args.rebalance_method == 'fixed':
            fixed_test, fixed_val, fixed_train = 200, 100, 500
            logging.info(f"Rebalancing number of samples in classes using {args.rebalance_method} sampling.")
            logging.info(f'Val: {fixed_val} Test: {fixed_test} Train: {fixed_train}!')

            rebalance_json(out_dir, background_val, fixed_val, 'balanced')
            rebalance_json(out_dir, background_test, fixed_test, 'balanced')
            rebalance_json(out_dir, background_train, fixed_train, 'balanced')

            rebalance_json(out_dir, speech_val, fixed_val, 'balanced')
            rebalance_json(out_dir, speech_test, fixed_test, 'balanced')
            rebalance_json(out_dir, speech_train, fixed_train, 'balanced')
    else:
        logging.info("Don't rebalance number of samples in classes.")


if __name__ == '__main__':
    main()
