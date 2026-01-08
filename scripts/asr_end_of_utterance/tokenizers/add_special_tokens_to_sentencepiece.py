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

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import logging
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import sentencepiece as spm

from nemo.collections.asr.data.audio_to_eou_label_lhotse import EOB_STRING, EOU_STRING
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

try:
    import sentencepiece_model_pb2 as spt
except (ImportError, ModuleNotFoundError):
    raise Exception("Ensure that sentencepiece_model_pb2.py has been generated from the protoc compiler")


SPECIAL_TOKENS = [EOU_STRING, EOB_STRING]

"""Utility to add special tokens to existing sentencepiece models.

Generate sentencepiece_model_pb2.py in the directory of this script before running
To generate run `protoc --python_out=<path_to_NeMo>/scripts/asr_end_of_utterance/tokenizers sentencepiece_model.proto`
inside the src folder in sentencepiece repo
Refer: https://github.com/google/sentencepiece/issues/121

Usage:
python add_special_tokens_to_sentencepiece.py \
    --input_file your_model.nemo \
    --output_dir /path/to/new/tokenizer_dir/
"""


parser = ArgumentParser(description="Add special tokens to sentencepiece model")
parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    required=True,
    help="Path to nemo model file, or sentencepiece model file",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
    help="Path to output directory for new tokenizer",
)
parser.add_argument(
    "--tokens",
    type=str,
    nargs='+',
    help="Special tokens to add to tokenizer",
    default=SPECIAL_TOKENS,
)
parser.add_argument(
    "--extract_only",
    action="store_true",
    help="Extract tokenizer without adding special tokens",
)


def extract_nemo_tokenizer(nemo_filepath, output_dir):
    SaveRestoreConnector._unpack_nemo_file(path2file=nemo_filepath, out_folder=output_dir)
    tokenizer = None
    for file in Path(output_dir).glob("**/*"):
        if file.is_file() and file.name.endswith("tokenizer.model"):
            tokenizer = file
            break
    if tokenizer is None:
        raise ValueError(f"Tokenizer not found in {output_dir}: {os.listdir(output_dir)}")
    return str(tokenizer.absolute())


def edit_spt_model(input_file, output_dir, tokens, is_userdefined, extract_only=False):
    if extract_only:
        logging.info("Extracting tokenizer only, no special tokens will be added.")

    output_dir = Path(output_dir)

    if output_dir.exists():
        logging.warning(f"Output directory {output_dir} already exists. Overwriting it.")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = str(output_dir / "tokenizer.model")

    token_type = 3
    if is_userdefined:
        token_type = 4

    model = spt.ModelProto()
    model.ParseFromString(open(input_file, 'rb').read())

    if not extract_only:
        for token in tokens:
            piece = model.SentencePiece(piece=token, score=0.0, type=token_type)
            if piece in model.pieces:
                logging.error(f"Special Token '{token}' already exists in the input model!")
                sys.exit(1)
            model.pieces.append(piece)

    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(model.SerializeToString())

    if not extract_only:
        try:
            for token in tokens:
                id = sp.piece_to_id(token)
                logging.info(f"Created token '{token}' at ID {id}")
            logging.info(f"New tokenizer vocab size: {sp.get_piece_size()}")
        except:
            logging.error(
                "Could not appropriately configure new tokenizer. Verify if the special tokens already exist."
            )
            sys.exit(1)

    with open(output_file, 'wb') as outf:
        outf.write(model.SerializeToString())
    logging.info(f"Created new tokenizer at: {output_file}")

    # Write the vocab to file
    vocab_file = str(output_dir / "tokenizer.vocab")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            score = sp.get_score(i)  # Optional: only available if using newer SentencePiece versions
            f.write(f"{piece}\t{score}\n")  # Format follows the original vocab format
    logging.info(f"Created new tokenizer vocab at: {vocab_file}")

    special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    special_tokens.extend(tokens)
    vocab_txt_file = str(output_dir / "vocab.txt")
    with open(vocab_txt_file, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            if piece in special_tokens:
                # skip special tokens
                continue
            token = piece[1:] if piece.startswith("▁") else f"##{piece}"
            if len(token) == 0:
                tokens = piece[0]
            f.write(f"{token}\n")  # Format follows the original vocab format
    logging.info(f"Created new tokenizer vocab at: {vocab_txt_file}")


def inject_special_tokens(input_file, output_dir, tokens, is_userdefined=True, extract_only=False):
    """
    NOTE: is_userdefined should be set to True in order for ASR model to work
    with the new special tokens properly.
    """
    if not os.path.exists(input_file):
        raise ValueError(f"Input file {input_file} does not exist")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Check if input file is a Nemo file
        if input_file.endswith(".nemo"):
            input_file = extract_nemo_tokenizer(input_file, temp_dir)
            logging.info(f"Extracted tokenizer from Nemo file: {input_file}")
        else:
            input_file = os.path.abspath(input_file)
            logging.info(f"Using input file: {input_file}")

        edit_spt_model(input_file, output_dir, tokens, is_userdefined, extract_only=extract_only)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parser.parse_args()
    inject_special_tokens(args.input_file, args.output_dir, args.tokens, extract_only=args.extract_only)
