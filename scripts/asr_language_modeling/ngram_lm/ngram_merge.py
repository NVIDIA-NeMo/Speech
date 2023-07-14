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

"""
This script would interpolate two arpa N-gram language models (LMs), 
culculate perplexity of resulted LM, and make binary KenLM from it.

Minimun usage example to interpolate two N-gram language models with weights:
alpha * ngram_a + beta * ngram_b = 2 * ngram_a + 1 * ngram_b

python3 ngram_merge.py  --kenlm_bin_path /workspace/nemo/decoders/kenlm/build/bin \
                    --arpa_a /path/ngram_a.kenlm.tmp.arpa \
                    --alpha 2 \
                    --arpa_b /path/ngram_b.kenlm.tmp.arpa \
                    --beta 1 \
                    --out_path /path/out


Merge two N-gram language models and calculate its perplexity with test_file.
python3 ngram_merge.py  --kenlm_bin_path /workspace/nemo/decoders/kenlm/build/bin \
                    --ngram_bin_path /workspace/nemo/decoders/ngram-1.3.14/src/bin \
                    --arpa_a /path/ngram_a.kenlm.tmp.arpa \
                    --alpha 0.5 \
                    --arpa_b /path/ngram_b.kenlm.tmp.arpa \
                    --beta 0.5 \
                    --out_path /path/out \
                    --nemo_model_file /path/to/model_tokenizer.nemo \
                    --test_file /path/to/test_manifest.json \
                    --force
"""

import argparse
import os
import subprocess
import sys
from typing import Tuple

import kenlm_utils
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.modules.rnnt import RNNTDecoder
from nemo.collections.asr.parts.submodules.ctc_beam_decoding import DEFAULT_TOKEN_OFFSET
from nemo.utils import logging


class NgramMerge:
    def __init__(self, ngram_bin_path):
        self.ngram_bin_path = ngram_bin_path

    def ngrammerge(self, arpa_a: str, alpha: float, arpa_b: str, beta: float, arpa_c: str, force: bool) -> str:
        """
        Merge two ARPA n-gram language models using the ngrammerge command-line tool and output the result in ARPA format.
        
        Args:
            arpa_a (str): Path to the first input ARPA file.
            alpha (float): Interpolation weight for the first model.
            arpa_b (str): Path to the second input ARPA file.
            beta (float): Interpolation weight for the second model.
            arpa_c (str): Path to the output ARPA file.
            force (bool): Whether to overwrite existing output files.
        
        Returns:
            str: Path to the output ARPA file in mod format.
        """
        mod_a = arpa_a + ".mod"
        mod_b = arpa_b + ".mod"
        mod_c = arpa_c + ".mod"
        if os.path.isfile(mod_c) and not force:
            logging.info("File " + mod_c + " exists. Skipping.")
        else:
            sh_args = [
                os.path.join(self.ngram_bin_path, "ngrammerge"),
                "--alpha=" + str(alpha),
                "--beta=" + str(beta),
                "--normalize",
                # "--use_smoothing",
                mod_a,
                mod_b,
                mod_c,
            ]
            logging.info(
                "\n"
                + str(subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,))
                + "\n",
            )
        return mod_c

    def arpa2mod(self, arpa_path: str, force: bool):
        """
        This function reads an ARPA n-gram model and converts it to a binary format. The binary model is saved to the same directory as the ARPA model with a ".mod" extension. If the binary model file already exists and force argument is False, then the function skips conversion and returns a message. Otherwise, it executes the command to create a binary model using the subprocess.run method.

        Parameters:
            arpa_path (string): The file path to the ARPA n-gram model.
            force (bool): If True, the function will convert the ARPA model to binary even if the binary file already exists. If False and the binary file exists, the function will skip the conversion.
        Returns:
            If the binary model file already exists and force argument is False, returns a message indicating that the file exists and the conversion is skipped.
            Otherwise, returns a subprocess.CompletedProcess object, which contains information about the executed command. The subprocess's output and error streams are redirected to stdout and stderr, respectively.
        """
        mod_path = arpa_path + ".mod"
        if os.path.isfile(mod_path) and not force:
            return "File " + mod_path + " exists. Skipping."
        else:
            sh_args = [
                os.path.join(self.ngram_bin_path, "ngramread"),
                "--ARPA",
                arpa_path,
                mod_path,
            ]
            return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)

    def merge(
        self, arpa_a: str, alpha: float, arpa_b: str, beta: float, out_path: str, force: bool
    ) -> Tuple[str, str]:
        """
        Merges two ARPA language models using the ngrammerge tool.

        Args:
            arpa_a (str): Path to the first ARPA language model file.
            alpha (float): Interpolation weight for the first model.
            arpa_b (str): Path to the second ARPA language model file.
            beta (float): Interpolation weight for the second model.
            out_path (str): Path to the output directory for the merged ARPA model.
            force (bool): Whether to force overwrite of existing files.

        Returns:
            Tuple[str, str]: A tuple containing the path to the merged binary language model file and the path to the 
            merged ARPA language model file.
        """
        logging.info("\n" + str(self.arpa2mod(arpa_a, force)) + "\n")

        logging.info("\n" + str(self.arpa2mod(arpa_b, force)) + "\n")
        arpa_c = os.path.join(out_path, f"{os.path.split(arpa_a)[1]}-{alpha}-{os.path.split(arpa_b)[1]}-{beta}.arpa",)
        mod_c = self.ngrammerge(arpa_a, alpha, arpa_b, beta, arpa_c, force)
        return mod_c, arpa_c

    def perplexity(self, ngram_mod: str, test_far: str) -> str:
        """
        Calculates perplexity of a given ngram model on a test file.

        Args:
            ngram_mod (str): The path to the ngram model file.
            test_far (str): The path to the test file.

        Returns:
            str: A string representation of the perplexity calculated.

        Raises:
            AssertionError: If the subprocess to calculate perplexity returns a non-zero exit code.

        Example:
            >>> perplexity("/path/to/ngram_model", "/path/to/test_file")
            'Perplexity: 123.45'
        """
        sh_args = [
            os.path.join(self.ngram_bin_path, "ngramperplexity"),
            "--v=1",
            ngram_mod,
            test_far,
        ]
        ps = subprocess.Popen(sh_args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = ps.communicate()
        exit_code = ps.wait()
        command = " ".join(sh_args)
        assert (
            exit_code == 0
        ), f"Exit_code must be 0.\n bash command: {command} \n stdout: {stdout} \n stderr: {stderr}"
        perplexity_out = "\n".join(stdout.split("\n")[-6:-1])
        return perplexity_out

    def make_arpa(self, ngram_mod: str, ngram_arpa: str, force: bool):
        """
        Converts an ngram model in binary format to ARPA format.

        Args:
        - ngram_mod (str): The path to the ngram model in binary format.
        - ngram_arpa (str): The desired path for the ARPA format output file.
        - force (bool): If True, the ARPA format file will be generated even if it already exists.

        Returns:
        - Tuple[bytes, bytes]

        Raises:
        - AssertionError: If the shell command execution returns a non-zero exit code.
        - FileNotFoundError: If the binary ngram model file does not exist.
        """
        if os.path.isfile(ngram_arpa) and not force:
            logging.info("File " + ngram_arpa + " exists. Skipping.")
            return None
        else:
            sh_args = [
                os.path.join(self.ngram_bin_path, "ngramprint"),
                "--ARPA",
                ngram_mod,
                ngram_arpa,
            ]
            return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)

    def test_perplexity(self, mod_c: str, symbols: str, test_txt: str, nemo_model_file: str, tmp_path: str) -> str:
        """
        Tests the perplexity of a given ngram model on a test file.

        Args:
            mod_c (str): The path to the ngram model file.
            symbols (str): The path to the symbol table file.
            test_txt (str): The path to the test text file.
            nemo_model_file (str): The path to the NeMo model file.
            tmp_path (str): The path to the temporary directory where the test far file will be created.
            force (bool): If True, overwrites any existing far file.

        Returns:
            str: A string representation of the perplexity calculated.

        Example:
            >>> test_perplexity("/path/to/ngram_model", "/path/to/symbol_table", "/path/to/test_file", "/path/to/tokenizer_model", "/path/to/tmp_dir", True)
            'Perplexity: 123.45'
        """

        test_far = farcompile(symbols, test_txt, tmp_path, nemo_model_file)
        res_p = self.perplexity(mod_c, test_far)
        return res_p


def farcompile(symbols: str, text_file: str, tmp_path: str, nemo_model_file: str) -> str:
    """
    Compiles a text file into a FAR file using the given symbol table or tokenizer.

    Args:
        symbols (str): The path to the symbol table file.
        text_file (str): The path to the text file to compile.
        tmp_path (str): The path to the temporary directory where the test far file will be created.
        nemo_model_file (str): The path to the NeMo model file (.nemo).
        force (bool): If True, overwrites any existing FAR file.

    Returns:
        test_far (str): The path to the resulting FAR file.

    Example:
        >>> farcompile("/path/to/symbol_table", "/path/to/text_file", "/path/to/far_file", "/path/to/tokenizer_model", "/path/to/nemo_model", True)
    """
    test_far = os.path.join(tmp_path, os.path.split(text_file)[1] + ".far")

    sh_args = [
        "farcompilestrings",
        "--generate_keys=10",
        "--fst_type=compact",
        "--symbols=" + symbols,
        "--keep_symbols",
        ">",
        test_far,
    ]

    tokenizer, encoding_level, is_aggregate_tokenizer = kenlm_utils.setup_tokenizer(nemo_model_file)

    ps = subprocess.Popen(" ".join(sh_args), shell=True, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr,)

    kenlm_utils.iter_files(
        source_path=[text_file],
        dest_path=ps.stdin,
        tokenizer=tokenizer,
        encoding_level=encoding_level,
        is_aggregate_tokenizer=is_aggregate_tokenizer,
        verbose=1,
    )
    stdout, stderr = ps.communicate()

    exit_code = ps.returncode

    command = " ".join(sh_args)
    assert exit_code == 0, f"Exit_code must be 0.\n bash command: {command} \n stdout: {stdout} \n stderr: {stderr}"
    return test_far


def make_kenlm(kenlm_bin_path: str, ngram_arpa: str, force: bool):
    """
    Builds a language model from an ARPA format file using the KenLM toolkit.

    Args:
    - kenlm_bin_path (str): The path to the KenLM toolkit binary.
    - ngram_arpa (str): The path to the ARPA format file.
    - force (bool): If True, the KenLM language model will be generated even if it already exists.

    Raises:
    - AssertionError: If the shell command execution returns a non-zero exit code.
    - FileNotFoundError: If the KenLM binary or ARPA format file does not exist.
    """
    ngram_kenlm = ngram_arpa + ".kenlm"
    if os.path.isfile(ngram_kenlm) and not force:
        logging.info("File " + ngram_kenlm + " exists. Skipping.")
        return None
    else:
        sh_args = [os.path.join(kenlm_bin_path, "build_binary"), "trie", "-i", ngram_arpa, ngram_kenlm]
        return subprocess.run(sh_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr,)


def make_symbol_list(nemo_model_file, symbols, force):
    """
    Function: make_symbol_list

    Create a symbol table for the input tokenizer model file.

    Args:
        nemo_model_file (str): Path to the NeMo model file.
        symbols (str): Path to the file where symbol list will be saved.
        force (bool): Flag to force creation of symbol list even if it already exists.
    
    Returns:
        None

    Raises:
        None
    """
    if os.path.isfile(symbols) and not force:
        logging.info("File " + symbols + " exists. Skipping.")
    else:
        if nemo_model_file.endswith('.nemo'):
            asr_model = nemo_asr.models.ASRModel.restore_from(nemo_model_file, map_location=torch.device('cpu'))
        else:
            logging.warning(
                "nemo_model_file does not end with .nemo, therefore trying to load a pretrained model with this name."
            )
            asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_file, map_location=torch.device('cpu'))

        if isinstance(asr_model.decoder, RNNTDecoder):
            vocab_size = asr_model.decoder.blank_idx
        else:
            vocab_size = len(asr_model.decoder.vocabulary)

        vocab = [chr(idx + DEFAULT_TOKEN_OFFSET) for idx in range(vocab_size)]
        with open(symbols, "w", encoding="utf-8") as f:
            for i, v in enumerate(vocab):
                f.write(v + " " + str(i) + "\n")


def main(
    kenlm_bin_path: str,
    ngram_bin_path: str,
    arpa_a: str,
    alpha: float,
    arpa_b: str,
    beta: float,
    out_path: str,
    test_file: str,
    symbols: str,
    nemo_model_file: str,
    force: bool,
) -> None:
    """
    Entry point function for merging ARPA format language models, testing perplexity, creating symbol list, 
    and making ARPA and Kenlm models.

    Args:
    - kenlm_bin_path (str): The path to the Kenlm binary.
    - arpa_a (str): The path to the first ARPA format language model.
    - alpha (float): The weight given to the first language model during merging.
    - arpa_b (str): The path to the second ARPA format language model.
    - beta (float): The weight given to the second language model during merging.
    - out_path (str): The path where the output files will be saved.
    - test_file (str): The path to the file on which perplexity needs to be calculated.
    - symbols (str): The path to the file where symbol list for the tokenizer model will be saved.
    - nemo_model_file (str): The path to the NeMo model file.
    - force (bool): If True, overwrite existing files, otherwise skip the operations.

    Returns:
    - None
    """
    nm = NgramMerge(ngram_bin_path)
    mod_c, arpa_c = nm.merge(arpa_a, alpha, arpa_b, beta, out_path, force)

    if test_file and nemo_model_file:
        if not symbols:
            symbols = os.path.join(out_path, os.path.split(nemo_model_file)[1] + ".syms")
            make_symbol_list(nemo_model_file, symbols, force)
        for test_f in test_file.split(","):
            test_p = nm.test_perplexity(mod_c, symbols, test_f, nemo_model_file, out_path)
            logging.info("Perplexity summary " + test_f + " : " + test_p)

    logging.info("Making ARPA and Kenlm model " + arpa_c)
    out = nm.make_arpa(mod_c, arpa_c, force)
    if out:
        logging.info("\n" + str(out) + "\n")

    out = make_kenlm(kenlm_bin_path, arpa_c, force)
    if out:
        logging.info("\n" + str(out) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate ARPA N-gram language models and make KenLM binary model to be used with beam search decoder of ASR models."
    )
    parser.add_argument(
        "--kenlm_bin_path", required=True, type=str, help="The path to the bin folder of KenLM library.",
    )  # Use /workspace/nemo/decoders/kenlm/build/bin if installed it with scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh
    parser.add_argument(
        "--ngram_bin_path", required=True, type=str, help="The path to the bin folder of OpenGrm Ngram library.",
    )  # Use /workspace/nemo/decoders/ngram-1.3.14/src/bin if installed it with scripts/installers/install_opengrm.sh
    parser.add_argument("--arpa_a", required=True, type=str, help="Path to the arpa_a")
    parser.add_argument("--alpha", required=True, type=float, help="Weight of arpa_a")
    parser.add_argument("--arpa_b", required=True, type=str, help="Path to the arpa_b")
    parser.add_argument("--beta", required=True, type=float, help="Weight of arpa_b")
    parser.add_argument(
        "--out_path", required=True, type=str, help="Path to write tmp and resulted files.",
    )
    parser.add_argument(
        "--test_file",
        required=False,
        type=str,
        default=None,
        help="Path to test file to count perplexity if provided.",
    )
    parser.add_argument(
        "--symbols",
        required=False,
        type=str,
        default=None,
        help="Path to symbols (.syms) file . Could be calculated if it is not provided. Use as: --symbols /path/to/earnest.syms",
    )
    parser.add_argument(
        "--nemo_model_file",
        required=False,
        type=str,
        default=None,
        help="The path to '.nemo' file of the ASR model, or name of a pretrained NeMo model",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Whether to recompile and rewrite all files")
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(_parse_args()))
