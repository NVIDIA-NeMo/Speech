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

import csv
import json
from argparse import ArgumentParser
from glob import glob
from typing import List, Tuple

from tqdm import tqdm


"""
Converts WikiHomograph data to .json manifest format for HeteronymClassificationModel training.
WikiHomograph dataset could be found here:
    https://github.com/google-research-datasets/WikipediaHomographData

"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_folder', help="Path to data folder with .tsv files", type=str, required=True)
    parser.add_argument("--output", help="Path to output .json file to store the data", type=str, required=True)
    return parser.parse_args()


def read_wikihomograph_file(file: str) -> Tuple[List[str], List[List[int]], List[str], List[str]]:
    """
    Reads .tsv file from WikiHomograph dataset,
    e.g. https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/eval/live.tsv

    Args:
        file: path to .tsv file
    Returns:
        sentences: Text.
        start_end_indices: Start and end indices of the homograph in the sentence.
        heteronyms: Target heteronyms for each sentence
        word_ids: Word_ids corresponding to each heteronym, i.e. label.
    """
    excluded_sentences = 0
    sentences = []
    start_end_indices = []
    heteronyms = []
    word_ids = []
    with open(file, "r", encoding="utf-8") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for i, line in enumerate(tsv_file):
            if i == 0:
                continue
            heteronym, wordid, sentence, start, end = line
            start, end = int(start), int(end)
            sentence, start, end = correct_wikihomograph_data(sentence, start, end)

            heteronym_span = sentence[start:end]
            if heteronym_span.lower() != heteronym:
                if sentence.lower().count(heteronym) == 1:
                    start = sentence.lower().index(heteronym)
                    end = start + len(heteronym)
                    heteronym_span = sentence[start:end].lower()
                    assert heteronym == heteronym_span.lower()
                else:
                    excluded_sentences += 1
                    raise ValueError(f"heteronym {heteronym} != heteronym_span {heteronym_span} in {sentence}")

            heteronyms.append(heteronym)
            start_end_indices.append([start, end])
            sentences.append(sentence)
            word_ids.append(wordid)

    return sentences, start_end_indices, heteronyms, word_ids


def correct_wikihomograph_data(sentence: str, start: int = None, end: int = None):
    """
    Correct indices for WikiHomograph data

    Args:
        sentence: sentence
        start: start index of homograph
        end: end index of homograph

    """
    corrections = {
        "It is traditionally composed of 85–99% tin, mixed with copper, antimony, bismuth, and sometimes lead, although the use of lead is less common today.": [
            96,
            100,
        ],
        "B₁₀₅ can be conceptually divided into a B₄₈ fragment and B₂₈-B-B₂₈ (B₅₇) fragment.": [44, 52],
        "Pierrefonds Airport on Réunion recorded just 18 mm (0.71 in) of rainfall from November to January, a record minimum.": [
            101,
            107,
        ],
        "Consort Chen Farong (陳法容) was an imperial consort during the Chinese dynasty Liu Song.": [42, 49],
        "Unlike TiO₂, which features six-coordinate Ti in all phases, monoclinic zirconia consists of seven-coordinate zirconium centres.": [
            32,
            42,
        ],
        "Its area is 16 km², its approximate length is 10 km, and its approximate width is 3 km.": [24, 35],
        "The conjugate momentum to X has the expressionwhere the pᵢ are the momentum functions conjugate to the coordinates.": [
            86,
            95,
        ],
        "Furthermore 17β-HSD1 levels positively correlate with E2 and negatively correlate with DHT levels in breast cancer cells.": [
            39,
            48,
        ],
        "Electric car buyers get a €4,000 (US$4,520) discount while buyers of plug-in hybrid vehicles get a discount of €3,000 (US$3,390).": [
            99,
            107,
        ],
    }

    if sentence in corrections:
        start, end = corrections[sentence]

    sentence = sentence.replace("2014Coordinate", "2014 Coordinate")  # for normalized data for G2P OOV models
    sentence = sentence.replace("AAA", "triple A")  # for normalized data for G2P OOV models

    return sentence, start, end


def convert_wikihomograph_data_to_manifest(data_folder: str, output_manifest: str):
    """
    Convert WikiHomograph data to .json manifest

    Args:
        data_folder: data_folder that contains .tsv files
        output_manifest: path to output file
    """
    with open(output_manifest, "w") as f_out:
        for file in tqdm(glob(f"{data_folder}/*.tsv")):
            sentences, start_end_indices, heteronyms, word_ids = read_wikihomograph_file(file)
            for i, sent in enumerate(sentences):
                start, end = start_end_indices[i]
                heteronym_span = sent[start:end]
                entry = {
                    "text_graphemes": sent,
                    "start_end": [start, end],
                    "heteronym_span": heteronym_span,
                    "word_id": word_ids[i],
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Data saved at {output_manifest}")


if __name__ == '__main__':
    args = parse_args()
    convert_wikihomograph_data_to_manifest(args.data_folder, args.output)
