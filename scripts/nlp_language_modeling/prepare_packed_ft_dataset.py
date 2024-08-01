# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.sequence_packing_utils import create_hist, create_packing_strategy, fill_packing_strategy

if TYPE_CHECKING:
    from omegaconf import DictConfig

""" 
Script to prepare packed dataset from a SFT/PEFT dataset in the jsonl format.
Two main steps are run in this script:
1. The online processing code in GPTSFTDataset is run (including prompt template manipulation, 
sequence length truncation, tokenization, etc) and the result is an array of tokenized sequences, 
represented by indices). 
2. The sequences are grouped by length, and a packing algorithm is run. (https://en.wikipedia.org/wiki/Bin_packing_problem#Offline_algorithms)
Currently, two variants of "first fit" are supported.
"first_fit_decreasing" sorts the sequences in decreasing order before applying first-fit. 
It generates a more optimal packing, but it tends to keep all short sequences together, which may affect convergence.
"first_fit_shuffle" runs first-fit in a random order. Packing is less optimal but it keeps the dataset order random.
The recommendation is to run "first_fit_shuffle" and check the packed sequence lengths in the printout. 
If they are similar to the target length (i.e. packing is efficient), then use shuffle. Otherwise try first_fit_decreasing.

Example usage:

python scripts/nlp_language_modeling/prepare_packed_ft_dataset.py \
   model.data.train_ds.file_names=[/path/to/training.jsonl] \
   model.data.train_ds.max_seq_length=2048 \
   +tokenizer_path=<see note 1 below> \
   +output_dir=/path/to/output_folder \
   +pack_sizes=[2048,4096,8192]
   
Note: 
  - Tokenizer path supports SentencePiece tokenizer and HF tokenizer. 
    For SentencePiece tokenizer, specify the file /path/to/tokenizer.model 
    For HF tokenizer, specify a folder /path/to/hf_folder which contains tokenizer.json, tokenizer_config.json
    and special_tokens_map.json

  - If your model or dataset requires non-default configs for conventional SFT/PEFT training in NeMo, you will
    need to pass in the same configs to ``model.data.train_ds`` as you would for training with unpacked dataset.

  - ``model.data.train_ds.max_seq_length`` is the length to truncate each sequence before packing multiple sequences
    to the size of packed sequence (``pack_size``). ``max_seq_length`` should be set to the same value as unpacked data,
    and can be determined by examining the distribution of sequence lengths in the dataset.

  - ``pack_sizes`` is a list of packed sequence lengths. In this example, there will be three output files, one for
    each pack size. The output files are named ``<output_folder>/packed_{pack_size}_seed{seed}.npy``.
    This argument is a list because you will likely want to experiment with a few ``pack_sizes`` to find out which length
    can fill the GPU memory without exceeding it. Adjusting ``pack_size`` is analogous to adjusting the micro batch size in
    the unpacked case.
"""


def tokenize_dataset(cfg: 'DictConfig'):
    """
    Tokenizes a dataset using the same configuration file as finetuninng with GPTSFTDataset.

    This function reads a dataset and tokenizes it using SentencePiece tokenizer based on the provided configuration.

    Args:
      cfg: A Hydra configuration object containing parameters for tokenization.

    Returns:
      A NumPy array containing the tokenized sequences from the dataset.
    """

    logging.info("Tokenizing dataset...")
    # using the same template as SFT/PEFT script. This may be overkill but guarantees the preprocess settings
    # are identical to normal SFT training
    data_cfg = cfg.model.data.train_ds
    if os.path.isdir(cfg.tokenizer_path):
        # pass in a Hugging Face folder which contains tokenizer.json
        tokenizer = get_nmt_tokenizer(library="huggingface", model_name=cfg.tokenizer_path, use_fast=True)
    else:
        tokenizer = get_nmt_tokenizer(library="sentencepiece", tokenizer_model=cfg.tokenizer_path)

    dataset = GPTSFTDataset(
        file_path=data_cfg.file_names[0],
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        min_seq_length=data_cfg.min_seq_length,
        pad_seq_length_to_mult=16,  # adds padding in collate_fn so this value is irrelevant here
        add_bos=data_cfg.get('add_bos', False),
        add_eos=data_cfg.get('add_eos', True),
        add_sep=data_cfg.get('add_sep', False),
        sep_id=cfg.get('sep_id', 49704),
        max_num_samples=None,
        seed=data_cfg.get('seed', 1234),
        label_key=data_cfg.get('label_key', 'answer'),
        answer_only_loss=cfg.get('answer_only_loss', True),
        truncation_field=data_cfg.get('truncation_field', 'text'),
        pad_to_max_length=data_cfg.get('pad_to_max_length', False),
        index_mapping_dir=data_cfg.get('index_mapping_dir', None),
        prompt_template=data_cfg.get('prompt_template', None),
        virtual_tokens=0,
        tokens_to_generate=data_cfg.get('tokens_to_generate', 0),
        memmap_workers=data_cfg.get('memmap_workers', None),
        hf_dataset=data_cfg.get('hf_dataset', False),
        truncation_method=data_cfg.get('truncation_method', 'right'),
        special_tokens=data_cfg.get('chat_prompt_tokens', None),
        is_test=True,
    )

    return np.array([dataset[i] for i in range(len(dataset))])


@dataclass
class PackingArgs:
    output_dir: str = "output"
    pack_sizes: Tuple[int] = (2048,)
    packing_algorithm: str = "first_fit_shuffle"
    seed: int = 0

    def from_config(self, cfg: 'DictConfig'):
        for required_arg in ('output_dir', 'pack_sizes'):
            assert cfg.get(required_arg, None), f"Please specify +{required_arg}=..."
        self.output_dir = cfg.output_dir
        self.pack_sizes = cfg.pack_sizes
        self.packing_algorithm = cfg.get("packing_algorithm", "first_fit_shuffle")
        self.seed = cfg.get("seed", 0)
        return self


@hydra_runner(
    config_path="../../examples/nlp/language_modeling/tuning/conf", config_name="megatron_gpt_finetuning_config"
)
def main(cfg: 'DictConfig') -> None:
    args = PackingArgs().from_config(cfg)
    dataset = tokenize_dataset(cfg)
    sequences, histogram = create_hist(dataset, cfg.model.data.train_ds.max_seq_length)
    for pack_size in args.pack_sizes:
        assignments = create_packing_strategy(histogram, pack_size, args.packing_algorithm)
        output_data = fill_packing_strategy(assignments, sequences, pack_size)

        # save output data
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'packed_{pack_size}_seed{args.seed}.npy')
        np.save(output_path, output_data)
        logging.info(f"Done, output written to {output_path}")

    logging.info(
        f"""
✅ Packed datasets with pack sizes {args.pack_sizes} are prepared successfully. 
To train with packed sequences, you need to make changes to the SFT/PEFT config file. See NeMo Documentation 
for more details: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/throughput_optimizations.html#sequence-packing-for-sft-peft>
"""
    )


if __name__ == '__main__':
    main()
