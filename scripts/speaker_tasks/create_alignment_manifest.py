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
import os
import shutil
from pathlib import Path

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_ctm, write_manifest
from nemo.utils import logging


def get_unaligned_files(unaligned_path):
    """
    Get files without alignments in order to filter them out (as they cannot be used for data simulation).
    In the unaligned file, each line contains the file name and the reason for the unalignment, if necessary to specify.

    Example: unaligned.txt

    <utterance_id> <comment>
    1272-128104-0000 (no such file)
    2289-152257-0025 (no such file)
    2289-152257-0026 (mapping failed)
    ...

    Args:
        unaligned_path (str): Path to the file containing unaligned examples

    Returns:
        skip_files (list): Unaligned file names to skip
    """
    skip_files = []
    with open(unaligned_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            unaligned_file = line.split()[0]
            skip_files.append(unaligned_file)
    return skip_files


def create_new_ctm_entry(session_name, speaker_id, wordlist, alignments, output_precision=3):
    """
    Create new CTM entry (to write to output ctm file)

    Args:
        session_name (str): Current session name.
        speaker_id (int): LibriSpeech speaker ID for the current entry.
        wordlist (list): List of words
        alignments (list): List of alignments
        output_precision (int): Precision for CTM outputs
    Returns:
        arr (list): List of ctm entries
    """
    arr = []
    for i in range(len(wordlist)):
        word = wordlist[i]
        if word != "":
            # note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
            align1 = float(round(alignments[i - 1], output_precision))
            align2 = float(round(alignments[i] - alignments[i - 1], output_precision,))
            text = f"{session_name} {speaker_id} {align1} {align2} {word} 0\n"
            arr.append((align1, text))
    return arr


def load_librispeech_alignment(alignment_filepath: str) -> dict:
    """
    Load alignment data for librispeech
    
    Args:
        alignment_filepath (str): Path to the file containing alignments
    Returns:
        alignments (dict[tuple]): A dictionary containing file index and alignments
    """
    alignments = {}
    with open(alignment_filepath, "r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            file_id, words, timestamps = line.split()
            alignments[file_id] = (words, timestamps)
    return alignments


def create_librispeech_ctm_alignments(
    input_manifest_filepath, base_alignment_path, ctm_output_directory, libri_dataset_split
):
    """
    Create new CTM alignments using input LibriSpeech word alignments. 

    Args:
        input_manifest_filepath (str): Path to the input LibriSpeech manifest file
        base_alignment_path (str): Path to the base directory containing the LibriSpeech word alignments
        ctm_source_dir (str): Directory to write the CTM files to
        libri_dataset_split (str): Which split of the LibriSpeech dataset is being used
    """
    manifest = read_manifest(input_manifest_filepath)
    unaligned_path = os.path.join(base_alignment_path, "unaligned.txt")

    if os.path.exists(unaligned_path):
        unaligned_file_ids = set(get_unaligned_files(unaligned_path))
    else:
        unaligned_file_ids = set()

    libri_dataset_split = libri_dataset_split.replace("_", "-")

    # delete output directory if it exists or throw warning
    if os.path.isdir(ctm_output_directory):
        logging.info(f"Removing existing output directory: {ctm_output_directory}")
        shutil.rmtree(ctm_output_directory)
    if not os.path.exists(ctm_output_directory):
        logging.info(f"Creating output directory: {ctm_output_directory}")
        os.mkdir(ctm_output_directory)

    if len(manifest) == 0:
        raise Exception(f"Input manifest is empty: {input_manifest_filepath}")

    for entry in manifest:
        audio_file = entry['audio_filepath']
        file_id = Path(audio_file).stem

        if file_id in unaligned_file_ids:
            continue

        speaker_id = file_id.split('-')[0]
        book_id = file_id.split('-')[1]
        book_dir = os.path.join(base_alignment_path, "LibriSpeech", libri_dataset_split, speaker_id, book_id)
        alignment_filepath = os.path.join(book_dir, f"{speaker_id}-{book_id}.alignment.txt")

        alignment_data = load_librispeech_alignment(alignment_filepath)
        if file_id not in alignment_data:
            logging.warning(f"Cannot find alignment data for {audio_file} in {alignment_filepath}")
            continue

        words, end_times = alignment_data[file_id]
        words = words.replace('\"', '').lower().split(',')
        end_times = [float(e) for e in end_times.replace('\"', '').split(',')]

        ctm_list = create_new_ctm_entry(file_id, speaker_id, words, end_times)
        write_ctm(os.path.join(ctm_output_directory, file_id + '.ctm'), ctm_list)


def create_manifest_with_alignments(
    input_manifest_filepath,
    ctm_source_dir,
    output_manifest_filepath,
    data_format_style,
    silence_dur_threshold=0.1,
    output_precision=3,
):
    """
    Create new manifest file with word alignments using CTM files

    Args:
        input_manifest_filepath (str): Path to the input manifest file
        ctm_source_dir (str): Directory to read the CTM files from
        output_manifest_filepath (str): Path to the output manifest file containing word alignments
        precision (int): How many decimal places to keep in the manifest file
    """
    manifest = read_manifest(input_manifest_filepath)

    target_manifest = []
    src_i = 0
    tgt_i = 0
    while src_i < len(manifest):
        f = manifest[src_i]
        fn = f['audio_filepath'].split('/')[-1]
        filename = fn.split('.')[0]  # assuming that there is only one period in the input filenames
        if "voxceleb" in data_format_style:
            fn_split = f['audio_filepath'].split('/')
            filename = fn_split[-3] + '-' + fn_split[-2] + '-' + fn_split[-1].split('.')[0]
            ctm_filepath = os.path.join(ctm_source_dir, filename + '.ctm')
        else:
            ctm_filepath = os.path.join(ctm_source_dir, filename + '.ctm')

        if not os.path.isfile(ctm_filepath):
            logging.info(f"Skipping {filename}.wav as there is no corresponding CTM file")
            src_i += 1
            continue

        with open(ctm_filepath, 'r') as ctm_file:
            lines = ctm_file.readlines()

        # One-word samples should be filtered out.
        if len(lines) <= 1:
            src_i += 1
            continue

        words = []
        end_times = []
        i = 0
        prev_end = 0
        for i in range(len(lines)):
            ctm = lines[i].split(' ')
            speaker_id = ctm[1]
            start = float(ctm[2])
            end = float(ctm[2]) + float(ctm[3])
            start = round(start, output_precision)
            end = round(end, output_precision)
            interval = start - prev_end

            if (i == 0 and interval > 0) or (i > 0 and interval > silence_dur_threshold):
                words.append("")
                end_times.append(start)
            elif i > 0:
                end_times[-1] = start

            words.append(ctm[4])
            end_times.append(end)

            i += 1
            prev_end = end

        # append last end
        if f['duration'] > prev_end:
            words.append("")
            end_times.append(f['duration'])

        # build target manifest entry
        target_manifest.append({})
        target_manifest[tgt_i]['audio_filepath'] = f['audio_filepath']
        target_manifest[tgt_i]['duration'] = f['duration']
        target_manifest[tgt_i]['text'] = f['text']
        target_manifest[tgt_i]['words'] = words
        target_manifest[tgt_i]['alignments'] = end_times
        target_manifest[tgt_i]['speaker_id'] = speaker_id

        src_i += 1
        tgt_i += 1

    logging.info(f"Writing output manifest file to {output_manifest_filepath}")
    write_manifest(output_manifest_filepath, target_manifest)


def main():
    """
    Create a combined manifest file including word alignments and speaker IDs
    """
    input_manifest_filepath = args.input_manifest_filepath
    base_alignment_path = args.base_alignment_path
    output_manifest_filepath = args.output_manifest_filepath
    ctm_output_directory = args.ctm_output_directory
    libri_dataset_split = args.libri_dataset_split
    use_ctm_alignment_source = args.use_ctm_alignment_source
    output_precision = args.output_precision

    # Case 1: args.base_alignment_path is containing the ctm files
    if use_ctm_alignment_source:
        ctm_source_dir = args.base_alignment_path
    # Case 2: args.base_alignment_path is containing *.lab style alignments for the dataset
    else:
        create_librispeech_ctm_alignments(
            input_manifest_filepath, base_alignment_path, ctm_output_directory, libri_dataset_split
        )
        ctm_source_dir = ctm_output_directory

    create_manifest_with_alignments(
        input_manifest_filepath,
        ctm_source_dir,
        output_manifest_filepath,
        data_format_style=args.data_format_style,
        silence_dur_threshold=args.silence_dur_threshold,
        output_precision=output_precision,
    )


if __name__ == "__main__":
    """
    This script creates a manifest file to be used for generating synthetic
    multispeaker audio sessions. The script takes in the default manifest file
    for a LibriSpeech dataset and corresponding word alignments and produces
    a combined manifest file that contains word alignments and speaker IDs
    per example. It can also be used to produce a manifest file for a different
    dataset if alignments are passed in CTM files.

    The alignments are obtained from: https://github.com/CorentinJ/librispeech-alignments

    Args:
        input_manifest_filepath (str): Path to input manifest file
        base_alignment_path (str): Path to the base directory for the LibriSpeech alignment dataset 
                                   (specifically to the LibriSpeech-Alignments directory containing 
                                   both the LibriSpeech folder as well as the unaligned.txt file) 
                                   or to a directory containing the requisite CTM files
        output_manifest_filepath (str): Path to output manifest file
        ctm_output_directory (str): Path to output CTM directory (only used for LibriSpeech)
        libri_dataset_split (str): Which dataset split to create a combined manifest file for
        use_ctm_alignment_source (bool): If true, base_alignment_path points to a directory containing ctm files
    """
    parser = argparse.ArgumentParser(description="LibriSpeech Alignment Manifest Creator")
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--base_alignment_path", help="path to alignments (LibriSpeech)", type=str, required=False)
    parser.add_argument("--output_manifest_filepath", help="path to output manifest file", type=str, required=True)
    parser.add_argument(
        "--ctm_output_directory",
        help="path to output ctm directory for LibriSpeech (or to input CTM directory)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--libri_dataset_split",
        help="which test/dev/training set to create a manifest for (only used for LibriSpeech)",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "--use_ctm_alignment_source",
        help="if true, base_alignment_path points to a directory containing ctm files",
        action='store_true',
        required=False,
    )
    parser.add_argument(
        "--data_format_style",
        help="Use specific format for speaker IDs and utterance IDs. e.g. 'voxceleb', 'librispeech', 'swbd'",
        default="",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_precision", help="precision for output alignments", type=int, required=False, default=3
    )
    parser.add_argument(
        "--silence_dur_threshold", help="threshold for inserting silence", type=float, required=False, default=0.1
    )
    args = parser.parse_args()

    main()
