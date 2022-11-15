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


from argparse import ArgumentParser
from typing import List

import pynini
from pynini import Far

from nemo.utils import logging


"""
This files takes 1. Far file containing a fst graph created by TN or ITN 2. entire string.
Optionally: 3. start position of substring 4. end (exclusive) position of substring
and returns input output mapping of all words in the string bounded by whitespace. 
If start and end position specified returns
1. mapped output string 2. start and end indices of mapped substring

Usage: 

python alignment.py --fst=<fst file> --text=\"2615 Forest Av, 1 Aug 2016\" --rule=\"tokenize_and_classify\" --start=22 --end=26

Output:
inp string: |2615 Forest Av, 1 Aug 2016|
out string: |twenty six fifteen Forest Avenue , the first of august twenty sixteen|
inp indices: [22:26]
out indices: [55:69]
in: |2016| out: |twenty sixteen|


python alignment.py --fst=<fst file> --text=\"2615 Forest Av, 1 Aug 2016\" --rule=\"tokenize_and_classify\"

Output:
inp string: |2615 Forest Av, 1 Aug 2016|
out string: |twenty six fifteen Forest Avenue , the first of august twenty sixteen|
inp indices: [0:4] out indices: [0:18]
in: |2615| out: |twenty six fifteen|
inp indices: [5:11] out indices: [19:25]
in: |Forest| out: |Forest|
inp indices: [12:15] out indices: [26:34]
in: |Av,| out: |Avenue ,|
inp indices: [16:17] out indices: [39:44]
in: |1| out: |first|
inp indices: [18:21] out indices: [48:54]
in: |Aug| out: |august|
inp indices: [22:26] out indices: [55:69]
in: |2016| out: |twenty sixteen|


Disclaimer: The heuristic algorithm relies on monotonous alignment and can fail in certain situations,
e.g. when word pieces are reordered by the fst:


python alignment.py --fst=<fst file> --text=\"$1\" --rule=\"tokenize_and_classify\" --start=0 --end=1
inp string: |$1|
out string: |one dollar|
inp indices: [0:1] out indices: [0:3]
in: |$| out: |one|
"""


def parse_args():
    args = ArgumentParser("map substring to output with FST")
    args.add_argument("--fst", help="FAR file containing FST", type=str, required=True)
    args.add_argument(
        "--rule",
        help="rule name in FAR file containing FST",
        type=str,
        default='tokenize_and_classify',
        required=False,
    )
    args.add_argument(
        "--text",
        help="input string",
        type=str,
        default="2615 Forest Av, 90601 CA, Santa Clara. 10kg, 12/16/2018, $123.25. 1 Aug 2016.",
    )
    args.add_argument("--start", help="start index of substring to be mapped", type=int, required=False)
    args.add_argument("--end", help="end index of substring to be mapped", type=int, required=False)
    return args.parse_args()


EPS = "<eps>"
WHITE_SPACE = "\u23B5"


def get_word_segments(text: str) -> List[List[int]]:
    """
    Returns word segments from given text based on white space in form of list of index spans.
    """
    spans = []
    cur_span = [0]
    for idx, ch in enumerate(text):
        if len(cur_span) == 0 and ch != " ":
            cur_span.append(idx)
        elif ch == " ":
            cur_span.append(idx)
            assert len(cur_span) == 2
            spans.append(cur_span)
            cur_span = []
        elif idx == len(text) - 1:
            idx += 1
            cur_span.append(idx)
            assert len(cur_span) == 2
            spans.append(cur_span)
    return spans


def create_symbol_table() -> pynini.SymbolTable:
    """
    Creates and returns Pynini SymbolTable used to label alignment with ascii instead of integers
    """
    table = pynini.SymbolTable()
    for num in range(34, 200):  # ascii alphanum + letter range
        table.add_symbol(chr(num), num)
    table.add_symbol(EPS, 0)
    table.add_symbol(WHITE_SPACE, 32)
    return table


def get_string_alignment(fst: pynini.Fst, input_text: str, symbol_table: pynini.SymbolTable):
    """
    create alignment of input text based on shortest path in FST. Symbols used for alignment are from symbol_table

    Returns:
        output: list of tuples, each mapping input character to output
    """
    lattice = pynini.shortestpath(input_text @ fst)
    paths = lattice.paths(input_token_type=symbol_table, output_token_type=symbol_table)

    ilabels = paths.ilabels()
    olabels = paths.olabels()
    logging.debug(paths.istring())
    logging.debug(paths.ostring())
    output = list(zip([symbol_table.find(x) for x in ilabels], [symbol_table.find(x) for x in olabels]))
    paths.next()
    assert paths.done()
    output_str = "".join(map(remove, [x[1] for x in output]))
    return output, output_str


def _get_aligned_index(alignment: List[tuple], index: int):
    """
    Given index in contracted input string computes corresponding index in alignment (which has EPS)
    """
    aligned_index = 0
    idx = 0

    while idx < index:
        if alignment[aligned_index][0] != EPS:
            idx += 1
        aligned_index += 1
    while alignment[aligned_index][0] == EPS:
        aligned_index += 1
    return aligned_index


def _get_original_index(alignment, aligned_index):
    """
    Given index in aligned output, returns corresponding index in contracted output string
    """

    og_index = 0
    idx = 0
    while idx < aligned_index:
        if alignment[idx][1] != EPS:
            og_index += 1
        idx += 1
    return og_index


remove = lambda x: "" if x == EPS else " " if x == WHITE_SPACE else x


def indexed_map_to_output(alignment: List[tuple], start: int, end: int):
    """
    Given input start and end index of contracted substring return corresponding output start and end index

    Args:
        alignment: alignment generated by FST with shortestpath, is longer than original string since including eps transitions
        start: inclusive start position in input string
        end: exclusive end position in input string

    Returns:
        output_og_start_index: inclusive start position in output string
        output_og_end_index: exclusive end position in output string
    """
    # get aligned start and end of input substring
    aligned_start = _get_aligned_index(alignment, start)
    aligned_end = _get_aligned_index(alignment, end - 1)  # inclusive

    logging.debug(f"0: |{list(map(remove, [x[0] for x in alignment[aligned_start:aligned_end+1]]))}|")

    # extend aligned_start to left
    while (
        aligned_start - 1 > 0
        and alignment[aligned_start - 1][0] == EPS
        and (alignment[aligned_start - 1][1].isalpha() or alignment[aligned_start - 1][1] == EPS)
    ):
        aligned_start -= 1

    while (
        aligned_end + 1 < len(alignment)
        and alignment[aligned_end + 1][0] == EPS
        and (alignment[aligned_end + 1][1].isalpha() or alignment[aligned_end + 1][1] == EPS)
    ):
        aligned_end += 1

    while (aligned_end + 1) < len(alignment) and (
        alignment[aligned_end + 1][1].isalpha() or alignment[aligned_end + 1][1] == EPS
    ):
        aligned_end += 1

    output_og_start_index = _get_original_index(alignment=alignment, aligned_index=aligned_start)
    output_og_end_index = _get_original_index(alignment=alignment, aligned_index=aligned_end + 1)
    return output_og_start_index, output_og_end_index


if __name__ == '__main__':
    logging.setLevel(logging.INFO)
    args = parse_args()
    fst = Far(args.fst, mode='r')
    try:
        fst = fst[args.rule]
    except:
        raise ValueError(f"{args.rule} not found. Please specify valid --rule argument.")
    input_text = args.text

    table = create_symbol_table()
    alignment, output_text = get_string_alignment(fst=fst, input_text=input_text, symbol_table=table)
    print(f"inp string: |{args.text}|")
    print(f"out string: |{output_text}|")

    if args.start is None:
        indices = get_word_segments(input_text)
    else:
        indices = [(args.start, args.end)]
    for x in indices:
        start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        print(f"inp indices: [{x[0]}:{x[1]}] out indices: [{start}:{end}]")
        print(f"in: |{input_text[x[0]:x[1]]}| out: |{output_text[start:end]}|")
