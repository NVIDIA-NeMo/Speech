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

"""
Usage:

python process_speech_commands_data.py \
    --data_root=<absolute path to where the data should be stored> \
    --data_version=<either 1 or 2, indicating version of the dataset> \
    --class_split=<either "all" or "sub", indicates whether all 30/35 classes should be used, or the 10+2 split should be used> \
    --num_processes=<number of processes to use for data preprocessing> \
    --rebalance \
    --log
"""

import argparse
import glob
import json
import logging
import os
import re
import tarfile
import urllib.request
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

from nemo.utils.tar_utils import safe_extract

URL_v1 = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
URL_v2 = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'


def __maybe_download_file(destination: str, source: str) -> str:
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download

    Args:
        destination: local filepath
        source: url of resource

    Returns:
        Local filepath of the downloaded file
    """
    if not os.path.exists(destination):
        logging.info(f'{destination} does not exist. Downloading ...')
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info(f'Downloaded {destination}.')
    else:
        logging.info(f'Destination {destination} exists. Skipping.')
    return destination


def __extract_all_files(filepath: str, data_dir: str):
    if not os.path.exists(data_dir):
        extract_file(filepath, data_dir)
    else:
        logging.info(f'Skipping extracting. Data already there {data_dir}')


def extract_file(filepath: str, data_dir: str):
    try:
        with tarfile.open(filepath) as tar:
            safe_extract(tar, data_dir)
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __get_mp_chunksize(dataset_size: int, num_processes: int) -> int:
    """
    Returns the number of chunks to split the dataset into for multiprocessing.

    Args:
        dataset_size: size of the dataset
        num_processes: number of processes to use for multiprocessing

        Returns:
            Number of chunks to split the dataset into for multiprocessing
    """
    chunksize = dataset_size // num_processes
    return chunksize if chunksize > 0 else 1


def __construct_filepaths(
    all_files: List[str],
    valset_uids: Set[str],
    testset_uids: Set[str],
    class_split: str,
    class_subset: List[str],
    pattern: str,
) -> Tuple[Dict[str, int], Dict[str, List[tuple]], List[tuple], List[tuple], List[tuple], List[tuple], List[tuple]]:
    """
    Prepares the filepaths for the dataset.

    Args:
        all_files: list of all files in the dataset
        valset_uids: set of uids of files in the validation set
        testset_uids: set of uids of files in the test set
        class_split: whether to use all classes as distinct labels, or to use
            10 classes subset and rest of the classes as noise or background
        class_subset: list of classes to consider if `class_split` is set to `sub`
        pattern: regex pattern to match the file names in the dataset
    """

    label_count = defaultdict(int)
    label_filepaths = defaultdict(list)
    unknown_val_filepaths = []
    unknown_test_filepaths = []

    train, val, test = [], [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)

            if label == '_background_noise_' or label == 'silence':
                continue

        if class_split == 'sub' and label not in class_subset:
            label = 'unknown'

            if uid in valset_uids:
                unknown_val_filepaths.append((label, entry))
            elif uid in testset_uids:
                unknown_test_filepaths.append((label, entry))

        if uid not in valset_uids and uid not in testset_uids:
            label_count[label] += 1
            label_filepaths[label].append((label, entry))

        if label == 'unknown':
            continue

        if uid in valset_uids:
            val.append((label, entry))
        elif uid in testset_uids:
            test.append((label, entry))
        else:
            train.append((label, entry))

    return {
        'label_count': label_count,
        'label_filepaths': label_filepaths,
        'unknown_val_filepaths': unknown_val_filepaths,
        'unknown_test_filepaths': unknown_test_filepaths,
        'train': train,
        'val': val,
        'test': test,
    }


def __construct_silence_set(
    rng: np.random.RandomState, sampling_rate: int, silence_stride: int, data_folder: str, background_noise: str
) -> List[str]:
    """
    Creates silence files given a background noise.

    Args:
        rng: Random state for random number generator
        sampling_rate: sampling rate of the audio
        silence_stride: stride for creating silence files
        data_folder: folder containing the silence directory
        background_noise: filepath of the background noise

    Returns:
        List of filepaths of silence files
    """
    silence_files = []
    if '.wav' in background_noise:
        y, sr = librosa.load(background_noise, sr=sampling_rate)

        for i in range(0, len(y) - sampling_rate, silence_stride):
            file_path = f'silence/{os.path.basename(background_noise)[:-4]}_{i}.wav'
            y_slice = y[i : i + sampling_rate] * rng.uniform(0.0, 1.0)
            out_file_path = os.path.join(data_folder, file_path)
            soundfile.write(out_file_path, y_slice, sr)

            silence_files.append(('silence', out_file_path))

    return silence_files


def __rebalance_files(max_count: int, label_filepath: str) -> Tuple[str, List[str], int]:
    """
    Rebalance the number of samples for a class.

    Args:
        max_count: maximum number of samples for a class
        label_filepath: list of filepaths for a class

    Returns:
        Rebalanced list of filepaths along with the label name and the number of samples
    """
    command, samples = label_filepath
    filepaths = [sample[1] for sample in samples]

    rng = np.random.RandomState(0)
    filepaths = np.asarray(filepaths)
    num_samples = len(filepaths)

    if num_samples < max_count:
        difference = max_count - num_samples
        duplication_ids = rng.choice(num_samples, difference, replace=True)
        filepaths = np.append(filepaths, filepaths[duplication_ids], axis=0)

    return command, filepaths, num_samples


def __prepare_metadata(skip_duration, sample: Tuple[str, str]) -> dict:
    """
    Creates the manifest entry for a file.

    Args:
        skip_duration: Whether to skip the computation of duration
        sample: Tuple of label and filepath

    Returns:
        Manifest entry of the file
    """
    label, audio_path = sample
    return json.dumps(
        {
            'audio_filepath': audio_path,
            'duration': 0.0 if skip_duration else librosa.core.get_duration(filename=audio_path),
            'command': label,
        }
    )


def __process_data(
    data_folder: str,
    dst_folder: str,
    num_processes: int = 1,
    rebalance: bool = False,
    class_split: str = 'all',
    skip_duration: bool = False,
):
    """
    Processes the data and generates the manifests.

    Args:
        data_folder: source with wav files and validation / test lists
        dst_folder: where manifest files will be stored
        num_processes: number of processes
        rebalance: rebalance the classes to have same number of samples
        class_split: whether to use all classes as distinct labels, or to use
            10 classes subset and rest of the classes as noise or background
        skip_duration: Bool whether to skip duration computation. Use this only for
            colab notebooks where knowing duration is not necessary for demonstration
    """

    os.makedirs(dst_folder, exist_ok=True)

    # Used for 10 classes + silence + unknown class setup - Only used when class_split is 'sub'
    class_subset = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    pattern = re.compile(r'(.+\/)?(\w+)\/([^_]+)_.+wav')
    all_files = glob.glob(os.path.join(data_folder, '*/*wav'))

    # Get files in the validation set
    valset_uids = set()
    with open(os.path.join(data_folder, 'validation_list.txt')) as fin:
        for line in fin:
            r = re.match(pattern, line)
            if r:
                valset_uids.add(r.group(3))

    # Get files in the test set
    testset_uids = set()
    with open(os.path.join(data_folder, 'testing_list.txt')) as fin:
        for line in fin:
            r = re.match(pattern, line)
            if r:
                testset_uids.add(r.group(3))

    logging.info('Validation and test set lists extracted')

    filepath_info = __construct_filepaths(all_files, valset_uids, testset_uids, class_split, class_subset, pattern)
    label_count = filepath_info['label_count']
    label_filepaths = filepath_info['label_filepaths']
    unknown_val_filepaths = filepath_info['unknown_val_filepaths']
    unknown_test_filepaths = filepath_info['unknown_test_filepaths']
    train = filepath_info['train']
    val = filepath_info['val']
    test = filepath_info['test']

    logging.info('Prepared filepaths for dataset')

    pool = Pool(num_processes)

    # Add silence and unknown class label samples
    if class_split == 'sub':
        logging.info('Perforiming 10+2 class subsplit')

        silence_path = os.path.join(data_folder, 'silence')
        os.makedirs(silence_path, exist_ok=True)

        silence_stride = 1000  # 0.0625 second stride
        sampling_rate = 16000
        folder = os.path.join(data_folder, '_background_noise_')

        silence_files = []
        rng = np.random.RandomState(0)

        background_noise_files = [os.path.join(folder, x) for x in os.listdir(folder)]
        silence_set_fn = partial(__construct_silence_set, rng, sampling_rate, silence_stride, data_folder)
        for silence_flist in tqdm(
            pool.imap(
                silence_set_fn, background_noise_files, __get_mp_chunksize(len(background_noise_files), num_processes)
            ),
            total=len(background_noise_files),
            desc='Constructing silence set',
        ):
            silence_files.extend(silence_flist)

        rng = np.random.RandomState(0)
        rng.shuffle(silence_files)
        logging.info(f'Constructed silence set of {len(silence_files)}')

        # Create the splits
        rng = np.random.RandomState(0)
        silence_split = 0.1
        unknown_split = 0.1

        # train split
        num_total_samples = sum([label_count[cls] for cls in class_subset])
        num_silence_samples = int(np.ceil(silence_split * num_total_samples))

        # initialize sample
        label_count['silence'] = 0
        label_filepaths['silence'] = []

        for silence_id in range(num_silence_samples):
            label_count['silence'] += 1
            label_filepaths['silence'].append(silence_files[silence_id])

        train.extend(label_filepaths['silence'])

        # Update train unknown set
        unknown_train_samples = label_filepaths['unknown']

        rng.shuffle(unknown_train_samples)
        unknown_size = int(np.ceil(unknown_split * num_total_samples))

        label_count['unknown'] = unknown_size
        label_filepaths['unknown'] = unknown_train_samples[:unknown_size]

        train.extend(label_filepaths['unknown'])

        logging.info('Train set prepared')

        # val set silence
        num_val_samples = len(val)
        num_silence_samples = int(np.ceil(silence_split * num_val_samples))

        val_idx = label_count['silence'] + 1
        for silence_id in range(num_silence_samples):
            val.append(silence_files[val_idx + silence_id])

        # Update val unknown set
        rng.shuffle(unknown_val_filepaths)
        unknown_size = int(np.ceil(unknown_split * num_val_samples))

        val.extend(unknown_val_filepaths[:unknown_size])

        logging.info('Validation set prepared')

        # test set silence
        num_test_samples = len(test)
        num_silence_samples = int(np.ceil(silence_split * num_test_samples))

        test_idx = val_idx + num_silence_samples + 1
        for silence_id in range(num_silence_samples):
            test.append(silence_files[test_idx + silence_id])

        # Update test unknown set
        rng.shuffle(unknown_test_filepaths)
        unknown_size = int(np.ceil(unknown_split * num_test_samples))

        test.extend(unknown_test_filepaths[:unknown_size])

        logging.info('Test set prepared')

    max_command = None
    max_count = -1
    for command, count in label_count.items():
        if command == 'unknown':
            continue

        if count > max_count:
            max_count = count
            max_command = command

    if rebalance:
        logging.info(f'Command with maximum number of samples = {max_command} with {max_count} samples')
        logging.info(f'Rebalancing dataset by duplicating classes with less than {max_count} samples...')

        rebalance_fn = partial(__rebalance_files, max_count)
        for command, filepaths, num_samples in tqdm(
            pool.imap(rebalance_fn, label_filepaths.items(), __get_mp_chunksize(len(label_filepaths), num_processes)),
            total=len(label_filepaths),
            desc='Rebalancing dataset',
        ):
            if num_samples < max_count:
                logging.info(f'Extended class label {command} from {num_samples} samples to {len(filepaths)} samples')
                label_filepaths[command] = [(command, filepath) for filepath in filepaths]

        del train
        train = []
        for label, samples in label_filepaths.items():
            train.extend(samples)

    manifests = [
        ('train_manifest.json', train),
        ('validation_manifest.json', val),
        ('test_manifest.json', test),
    ]

    metadata_fn = partial(__prepare_metadata, skip_duration)
    for manifest_filename, dataset in manifests:
        num_files = len(dataset)

        logging.info(f'Preparing manifest : {manifest_filename} with #{num_files} files')

        manifest = [
            metadata
            for metadata in tqdm(
                pool.imap(metadata_fn, dataset, __get_mp_chunksize(len(dataset), num_processes)),
                total=num_files,
                desc=f'Preparing {manifest_filename}',
            )
        ]

        with open(os.path.join(dst_folder, manifest_filename), 'w') as fout:
            for metadata in manifest:
                fout.write(metadata + '\n')

        logging.info(f'Finished construction of manifest. Path: {os.path.join(dst_folder, manifest_filename)}')

    pool.close()

    if skip_duration:
        logging.info(
            f'\n<<NOTE>> Duration computation was skipped for demonstration purposes on Colaboratory.\n'
            f'In order to replicate paper results and properly perform data augmentation, \n'
            f'please recompute the manifest file without the `--skip_duration` flag !\n'
        )


def main():
    parser = argparse.ArgumentParser(description='Google Speech Commands Data download and preprocessing')
    parser.add_argument('--data_root', required=True, help='Root directory for storing data')
    parser.add_argument(
        '--data_version',
        required=True,
        default=1,
        type=int,
        choices=[1, 2],
        help='Version of the speech commands dataset to download',
    )
    parser.add_argument(
        '--class_split', default='all', choices=['all', 'sub'], help='Whether to consider all classes or only a subset'
    )
    parser.add_argument('--num_processes', default=1, type=int, help='Number of processes')
    parser.add_argument('--rebalance', action='store_true', help='Rebalance the number of samples in each class')
    parser.add_argument('--skip_duration', action='store_true', help='Skip computing duration of audio files')
    parser.add_argument('--log', action='store_true', help='Generate logs')
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    data_root = args.data_root
    data_set = f'google_speech_recognition_v{args.data_version}'
    data_folder = os.path.join(data_root, data_set)

    logging.info(f'Working on: {data_set}')

    URL = URL_v1 if args.data_version == 1 else URL_v2

    # Download and extract
    if not os.path.exists(data_folder):
        file_path = os.path.join(data_root, data_set + '.tar.bz2')
        logging.info(f'Getting {data_set}')
        __maybe_download_file(file_path, URL)
        logging.info(f'Extracting {data_set}')
        __extract_all_files(file_path, data_folder)

    logging.info(f'Processing {data_set}')
    __process_data(
        data_folder,
        data_folder,
        num_processes=args.num_processes,
        rebalance=args.rebalance,
        class_split=args.class_split,
        skip_duration=args.skip_duration,
    )
    logging.info('Done!')


if __name__ == '__main__':
    main()
