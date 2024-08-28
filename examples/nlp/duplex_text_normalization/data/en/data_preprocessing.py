# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script can be used to clean the splits of English Google Text Normalization dataset
for better training performance. Without these processing steps we noticed that the model would have a hard time to learn certain input cases, and instead starts to either make unrecoverable errors
or hallucinate. For example, the model struggles to learn numbers with five or more digits due to limited examples in the training data, so we simplified the task for the model by letting it verbalize those cases
digit by digit. This makes the model more rebust to errors.
The operations include:
    - numbers that are longer than `max_integer_length` will be verbalized digit by digit, e.g. the mapping "10001" -> "ten thousand and one" in the data
will be changed to "10001" -> "one zero zero zero one"
    - denominators of fractions that are longer than `max_denominator_length` will be verbalized digit by digit
    - sentences with non-English characters will be removed
    - some class formats converted to standardized format, e.g. for `Fraction` "½" become "1/2"   
    - urls that have a spoken form of "*_letter" e.g. "dot h_letter  _letter t_letter  _letter m_letter  _letter l_letter" are converted to "dot h t m l"
    - for class types "PLAIN", "LETTERS", "ELECTRONIC", "VERBATIM", "PUNCT" the spoken form is changed to "<self>" which means this class should be left unchanged


USAGE Example:
1. Download the Google TN dataset from https://www.kaggle.com/google-nlu/text-normalization
2. Unzip the English subset (e.g., by running `tar zxvf  en_with_types.tgz`). Then there will a folder named `en_with_types`.
3. Run the data_split.py scripts to obtain the data splits
4. Run this script on the different splits
# python data_preprocessing.py       \
        --input_path=data_split/train           \
        --output_dir=train_processed \
        --max_integer_length=4  \
        --max_denominator_length=3 

In this example, the cleaned files will be saved in train_processed/.

After this script, you can use upsample.py to create a more class balanced training dataset for better performance.
"""


import os
from argparse import ArgumentParser
from functools import cache

import regex as re
from tqdm import tqdm

from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from nemo.collections.nlp.data.text_normalization.constants import EN_GREEK_TO_SPOKEN
from nemo.collections.nlp.data.text_normalization.utils import (
    add_space_around_dash,
    convert_fraction,
    convert_superscript,
)
from nemo.utils import logging


@cache
def inflect_engine():
    import inflect

    return inflect.engine()


# these are all words that can appear in a verbalized number, this list will be used later as a filter to detect numbers in verbalizations
number_verbalizations = list(range(0, 20)) + list(range(20, 100, 10))
number_verbalizations = (
    [
        inflect_engine().number_to_words(x, zero="zero").replace("-", " ").replace(",", "")
        for x in number_verbalizations
    ]
    + ["hundred", "thousand", "million", "billion", "trillion"]
    + ["point"]
)
digit = "0123456789"
processor = MosesProcessor(lang_id="en")


def process_url(o):
    """
    The function is used to process the spoken form of every URL in an example.
    E.g., "dot h_letter  _letter t_letter  _letter m_letter  _letter l_letter" ->
          "dot h t m l"
    Args:
        o: The expected outputs for the spoken form
    Return:
        o: The outputs for the spoken form with preprocessed URLs.
    """

    def flatten(l):
        """flatten a list of lists"""
        return [item for sublist in l for item in sublist]

    if o != '<self>' and '_letter' in o:
        o_tokens = o.split(' ')
        all_spans, cur_span = [], []
        for j in range(len(o_tokens)):
            if len(o_tokens[j]) == 0:
                continue
            if o_tokens[j] == '_letter':
                all_spans.append(cur_span)
                all_spans.append([' '])
                cur_span = []
            else:
                o_tokens[j] = o_tokens[j].replace('_letter', '')
                cur_span.append(o_tokens[j])
        if len(cur_span) > 0:
            all_spans.append(cur_span)
        o_tokens = flatten(all_spans)

        o = ''
        for o_token in o_tokens:
            if len(o_token) > 1:
                o += ' ' + o_token + ' '
            else:
                o += o_token
        o = o.strip()
        o_tokens = processor.tokenize(o).split()
        o = ' '.join(o_tokens)

    return o


def convert2digits(digits: str):
    """
    Verbalizes integer digit by digit, e.g. "12,000.12" -> "one two zero zero zero point one two"
    It can also take in a string that has an integer as prefix and outputs only the verbalized part of that, e.g. "12 kg" -> "one two"
    and outputs a warning

    Args:
        digits: integer in string format
    Return:
        res: number verbalization of the integer prefix of the input
    """
    engine = inflect_engine()
    res = []
    for i, x in enumerate(digits):
        if x in digit:
            res.append(engine.number_to_words(str(x), zero="zero").replace("-", " ").replace(",", ""))
        elif x == ".":
            res.append("point")
        elif x in [" ", ","]:
            continue
        else:
            # logging.warning(f"remove {digits[:i]} from {digits[i:]}")
            break
    res = " ".join(res)
    return res, i


def convert(example):
    engine = inflect_engine()
    cls, written, spoken = example

    written = convert_fraction(written)
    written = re.sub("é", "e", written)
    written = convert_superscript(written)

    if cls == "TIME":
        written = re.sub("([0-9]): ([0-9])", "\\1:\\2", written)
    if cls == "MEASURE":
        written = re.sub("([0-9])\s?''", '\\1"', written)

    spoken = process_url(spoken)

    if cls in ["TELEPHONE", "DIGIT", "MEASURE", "DECIMAL", "MONEY", "ADDRESS"]:
        spoken = re.sub(" o ", " zero ", spoken)
        spoken = re.sub(" o ", " zero ", spoken)
        spoken = re.sub("^o ", "zero ", spoken)
        spoken = re.sub(" o$", " zero", spoken)
        spoken = re.sub("^sil ", "", spoken)
        spoken = re.sub(" sil ", " ", spoken)
        spoken = re.sub(" sil ", " ", spoken)
        spoken = re.sub(" sil$", "", spoken)

    if cls != "ELECTRONIC":
        written = add_space_around_dash(written)

    example[1] = written
    example[2] = spoken

    l = args.max_integer_length - 2

    # if written form does not fulfill this format return
    if not re.search("[0-9]{%s}[,\s]?[0-9]{3}" % l, written):
        if cls != "FRACTION":
            return
        idx = written.index("/")
        denominator = written[idx + 1 :].strip()
        if not re.search(r"[0-9]{%s}" % (args.max_denominator_length + 1), denominator):
            return

    # convert spoken forms for different classes
    if cls == "CARDINAL":
        if written[0] == "-":
            digits = "minus " + convert2digits(written[1:])[0]
        else:
            digits = convert2digits(written)[0]
        spoken = digits
    elif cls == "ADDRESS":
        idx = re.search("[0-9]", written).start()
        number = convert2digits(written[idx:].strip())[0]
        s_words = spoken.split()
        for i, x in enumerate(s_words):
            if x in number_verbalizations:
                break
        spoken = " ".join(s_words[:i]) + " " + number
    elif cls == "DECIMAL":
        res = []
        for i, x in enumerate(written):
            if i == 0 and x == "-":
                res.append("minus")
            elif x in digit:
                res.append(engine.number_to_words(str(x), zero="zero").replace("-", " ").replace(",", ""))
            elif x == ".":
                res.append("point")
        spoken = " ".join(res)
        m = re.search("([a-z]+)", written)
        if m:
            spoken += " " + m.group(1)
    elif cls == "FRACTION":
        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        idx = written.index("/")
        numerator = written[:idx].strip()
        denominator = written[idx + 1 :].strip()
        if len(numerator) > args.max_integer_length:
            numerator = convert2digits(numerator)[0]
        else:
            numerator = engine.number_to_words(str(numerator), zero="zero").replace("-", " ").replace(",", "")
        if len(denominator) > args.max_denominator_length:
            denominator = convert2digits(denominator)[0]
        else:
            denominator = engine.number_to_words(str(denominator), zero="zero").replace("-", " ").replace(",", "")
        spoken = numerator + " slash " + denominator
        if res:
            spoken = "minus " + spoken
    elif cls == "MEASURE":
        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        idx = re.search("(?s:.*)([0-9]\s?[a-zA-Zµμ\/%Ω'])", written).end()
        number, unit_idx = convert2digits(written[:idx].strip())
        s_words = spoken.split()
        for i, x in enumerate(s_words):
            if x not in number_verbalizations:
                break

        spoken = number + " " + " ".join(s_words[i:])
        if res:
            spoken = "minus " + spoken
    elif cls == "MONEY":
        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        idx = re.search("[0-9]", written).start()
        m = re.search("\.", written[idx:])
        idx_end = len(written)
        if m:
            idx_end = m.start() + idx
        number, unit_idx = convert2digits(written[idx:idx_end].strip())
        s_words = spoken.split()
        for i, x in enumerate(s_words):
            if x not in number_verbalizations:
                break
        spoken = number + " " + " ".join(s_words[i:])
        if res:
            spoken = "minus " + spoken
    elif cls == "ORDINAL":
        res = []
        if written[0] == "-":
            res.append("minus")
            written = written[1:]
        if "th" in written.lower():
            idx = written.lower().index("th")
        elif "rd" in written.lower():
            idx = written.lower().index("rd")
        elif "nd" in written.lower():
            idx = written.lower().index("nd")
        elif "st" in written.lower():
            idx = written.lower().index("st")
        if re.search(r"[¿¡ºª]", written) is None:
            spoken = convert2digits(written[:idx].strip())[0] + " " + written[idx:].lower()
        if res:
            spoken = "minus " + spoken
    example[2] = spoken


def ignore(example):
    """
    This function makes sure specific class types like 'PLAIN', 'ELECTRONIC' etc. are left unchanged.

    Args:
        example: data example
    """
    cls, _, _ = example
    if cls in ["PLAIN", "LETTERS", "ELECTRONIC", "VERBATIM", "PUNCT"]:
        example[2] = "<self>"
    if example[1] == 'I' and re.search("(first|one)", example[2]):
        example[2] = "<self>"


def process_file(fp):
    """Reading the raw data from a file of NeMo format and preprocesses it. Write is out to the output directory.
    For more info about the data format, refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.

    Args:
        fp: file path
    """
    file_name = fp.split("/")[-1]
    output_path = f"{args.output_dir}/{file_name}"
    logging.info(f"-----input_file--------\n{fp}")
    logging.info(f"-----output_file--------\n{output_path}")

    insts, w_words, s_words, classes = [], [], [], []
    delete_sentence = False
    with open(fp, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            es = [e.strip() for e in line.strip().split('\t')]
            if es[0] == '<eos>':
                if not delete_sentence:
                    inst = (classes, w_words, s_words)
                    insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []
                delete_sentence = False
            else:
                # convert data sample
                convert(es)
                # decide if this data sample's spoken form should be same as written form
                ignore(es)

                characters_ignore = "¿¡ºª" + "".join(EN_GREEK_TO_SPOKEN.keys())
                # delete sentence with greek symbols, etc.
                if re.search(rf"[{characters_ignore}]", es[1]) is not None:
                    delete_sentence = True
                # delete characters from chinese, japanese, korean
                if re.search(r'[\u4e00-\u9fff]+', es[1]) is not None:
                    delete_sentence = True

                if es[0] == 'MONEY' and re.search("\s?DM$", es[1]):
                    delete_sentence = True

                if es[0] == 'MEASURE' and re.search("\s?Da$", es[1]):
                    delete_sentence = True

                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])

        inst = (classes, w_words, s_words)
        insts.append(inst)

    output_f = open(output_path, 'w+', encoding='utf-8')
    for _, inst in enumerate(insts):
        cur_classes, cur_tokens, cur_outputs = inst
        for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
            output_f.write(f'{c}\t{t}\t{o}\n')

        output_f.write(f'<eos>\t<eos>\n')


def main():
    if not os.path.exists(args.input_path):
        raise ValueError(f"Input path {args.input_path} does not exist")
    if os.path.exists(args.output_dir):
        logging.info(
            f"Output directory {args.output_dir} exists already. Existing files could be potentially overwritten."
        )
    else:
        logging.info(f"Creating output directory {args.output_dir}.")
        os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input_path):
        input_paths = sorted([os.path.join(args.input_path, f) for f in os.listdir(args.input_path)])
    else:
        input_paths = [args.input_path]

    for input_file in input_paths:
        process_file(input_file)


if __name__ == "__main__":

    parser = ArgumentParser(description="Text Normalization Data Preprocessing for English")
    parser.add_argument("--output_dir", required=True, type=str, help='Path to output directory.')
    parser.add_argument("--input_path", required=True, type=str, help='Path to input file or input directory.')
    parser.add_argument(
        "--max_integer_length",
        default=4,
        type=int,
        help='Maximum number of digits for integers that are allowed. Beyond this, the integers are verbalized digit by digit.',
    )
    parser.add_argument(
        "--max_denominator_length",
        default=3,
        type=int,
        help='Maximum number of digits for denominators that are allowed. Beyond this, the denominator is verbalized digit by digit.',
    )
    args = parser.parse_args()

    main()
