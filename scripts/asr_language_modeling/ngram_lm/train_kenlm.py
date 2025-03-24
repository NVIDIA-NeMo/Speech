# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#


# This script would train an N-gram language model with KenLM library (https://github.com/kpu/kenlm) which can be used
# with the beam search decoders on top of the ASR models. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# After the N-gram model is trained, and stored in the binary format, you may use
# 'scripts/ngram_lm/eval_beamsearch_ngram.py' to evaluate it on an ASR model.
#
# You need to install the KenLM library and also the beam search decoders to use this feature. Please refer
# to 'scripts/ngram_lm/install_beamsearch_decoders.sh' on how to install them.
#
# USAGE: python train_kenlm.py nemo_model_file=<path to the .nemo file of the model> \
#                              train_paths=<list of paths to the training text or JSON manifest file> \
#                              kenlm_bin_path=<path to the bin folder of KenLM library> \
#                              kenlm_model_file=<path to store the binary KenLM model> \
#                              ngram_length=<order of N-gram model> \
#
# After training is done, the binary LM model is stored at the path specified by '--kenlm_model_file'.
# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import List

from omegaconf import MISSING

from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel, kenlm_utils
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
NeMo's beam search decoders only support char-level encodings. In order to make it work with BPE-level encodings, we
use a trick to encode the sub-word tokens of the training data as unicode characters and train a char-level KenLM. 
"""


@dataclass
class TrainKenlmConfig:
    """
    Train an N-gram language model with KenLM to be used with beam search decoder of ASR models.
    """

    train_paths: List[str] = (
        MISSING  # List of training files or folders. Files can be a plain text file or ".json" manifest or ".json.gz". Example: [/path/to/manifest/file,/path/to/folder]
    )

    nemo_model_file: str = MISSING  # The path to '.nemo' file of the ASR model, or name of a pretrained NeMo model
    kenlm_model_file: str = MISSING  # The path to store the KenLM binary model file
    ngram_length: int = MISSING  # The order of N-gram LM
    kenlm_bin_path: str = MISSING  # The path to the bin folder of KenLM.

    preserve_arpa: bool = False  # Whether to preserve the intermediate ARPA file.
    ngram_prune: List[int] = field(
        default_factory=lambda: [0]
    )  # List of digits to prune Ngram. Example: [0,0,1]. See Pruning section on the https://kheafield.com/code/kenlm/estimation
    cache_path: str = ""  # Cache path to save tokenized files.
    verbose: int = 1  # Verbose level, default is 1.
    save_nemo: bool = False  # Save .nemo checkpoint to use with NGramGPULanguageModel
    normalize_unk_nemo: bool = True  # Normalize the UNK token in the NGramGPULanguageModel model


@hydra_runner(config_path=None, config_name='TrainKenlmConfig', schema=TrainKenlmConfig)
def main(args: TrainKenlmConfig):
    train_paths = kenlm_utils.get_train_list(args.train_paths)

    if isinstance(args.ngram_prune, str):
        args.ngram_prune = [args.ngram_prune]

    tokenizer, encoding_level, is_aggregate_tokenizer, full_vocab_size = kenlm_utils.setup_tokenizer(
        args.nemo_model_file
    )

    if encoding_level == "subword":
        discount_arg = "--discount_fallback"  # --discount_fallback is needed for training KenLM for BPE-based models
    else:
        discount_arg = ""

    arpa_file = f"{args.kenlm_model_file}.tmp.arpa"
    """ LMPLZ ARGUMENT SETUP """
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, 'lmplz'),
        "-o",
        str(args.ngram_length),
        "--arpa",
        arpa_file,
        discount_arg,
        "--prune",
    ] + [str(n) for n in args.ngram_prune]

    if args.cache_path:
        if not os.path.exists(args.cache_path):
            os.makedirs(args.cache_path, exist_ok=True)

        """ DATASET SETUP """
        encoded_train_files = []
        for file_num, train_file in enumerate(train_paths):
            logging.info(f"Encoding the train file '{train_file}' number {file_num+1} out of {len(train_paths)} ...")

            cached_files = glob(os.path.join(args.cache_path, os.path.split(train_file)[1]) + "*")
            encoded_train_file = os.path.join(args.cache_path, os.path.split(train_file)[1] + f"_{file_num}.tmp.txt")
            if (
                cached_files and cached_files[0] != encoded_train_file
            ):  # cached_files exists but has another file name: f"_{file_num}.tmp.txt"
                os.rename(cached_files[0], encoded_train_file)
                logging.info("Rename", cached_files[0], "to", encoded_train_file)

            encoded_train_files.append(encoded_train_file)

        kenlm_utils.iter_files(
            source_path=train_paths,
            dest_path=encoded_train_files,
            tokenizer=tokenizer,
            encoding_level=encoding_level,
            is_aggregate_tokenizer=is_aggregate_tokenizer,
            verbose=args.verbose,
        )

        first_process_args = ["cat"] + encoded_train_files
        first_process = subprocess.Popen(first_process_args, stdout=subprocess.PIPE, stderr=sys.stderr)

        logging.info(f"Running lmplz command \n\n{' '.join(kenlm_args)}\n\n")
        kenlm_p = subprocess.run(
            kenlm_args,
            stdin=first_process.stdout,
            capture_output=False,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        first_process.wait()

    else:
        logging.info(f"Running lmplz command \n\n{' '.join(kenlm_args)}\n\n")
        kenlm_p = subprocess.Popen(kenlm_args, stdout=sys.stdout, stdin=subprocess.PIPE, stderr=sys.stderr)
        kenlm_utils.iter_files(
            source_path=train_paths,
            dest_path=kenlm_p.stdin,
            tokenizer=tokenizer,
            encoding_level=encoding_level,
            is_aggregate_tokenizer=is_aggregate_tokenizer,
            verbose=args.verbose,
        )

        kenlm_p.communicate()

    if kenlm_p.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")

    """ BINARY BUILD """

    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        arpa_file,
        args.kenlm_model_file,
    ]
    logging.info(f"Running binary_build command \n\n{' '.join(kenlm_args)}\n\n")
    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")

    if args.save_nemo:
        if full_vocab_size is None:
            raise NotImplementedError("Unknown vocab size, cannot convert the model")
        nemo_model = NGramGPULanguageModel.from_arpa(
            lm_path=arpa_file, vocab_size=full_vocab_size, normalize_unk=args.normalize_unk_nemo
        )
        nemo_model.save_to(f"{args.kenlm_model_file}.nemo")

    if not args.preserve_arpa:
        os.remove(arpa_file)
        logging.info(f"Deleted the arpa file '{arpa_file}'.")


if __name__ == '__main__':
    main()
