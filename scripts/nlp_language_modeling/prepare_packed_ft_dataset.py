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

import collections
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

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
   model.restore_from_path=<path/to/nemo_model> \
   +output_dir=<output_folder> 
   +pack_sizes=[2048,4096,8192]
   
Note: 
- pack_sizes can take in a list 
- model.data.train_ds.max_seq_length is the length to truncate long sequences before packing, and is different from the packing sizes
- currenlty, we require a full nemo model file for simplicity and readability of code, but in theory only a tokenizer file is needed.
  This part can be improved in a future iteration of the script.
"""

PACKING_ALGOS = ['first_fit_decreasing', 'first_fit_shuffle']


def find_first_bin_that_fits(bins, s, bin_size):
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seqlens, pack_size):
    res = []
    for s in seqlens:
        first_bin = find_first_bin_that_fits(res, s, pack_size)
        if first_bin == -1:  # open a new bin
            res.append([s])
        else:
            res[first_bin].append(s)
    return res


def first_fit_decreasing(seqlens, pack_size):
    sorted_seqlens = sorted(seqlens, reverse=True)
    return first_fit(sorted_seqlens, pack_size)


def first_fit_shuffle(seqlens, pack_size):
    shuffled_seqlens = seqlens[:]
    np.random.shuffle(shuffled_seqlens)
    return first_fit(shuffled_seqlens, pack_size)


def create_assignment(output_path, assignments, ifile_handles):
    n_samples_in_this_shard = len(assignments)
    input_ids, loss_mask, seq_start_id = {}, {}, {}

    for oindex, assignment in tqdm(enumerate(assignments), total=n_samples_in_this_shard):
        _input_ids, _loss_mask, _seq_start_id = [], [], [0]

        for seq_length in assignment:
            _input_ids.extend(ifile_handles[seq_length][0].pop())
            _loss_mask.extend(ifile_handles[seq_length][1].pop())
            _seq_start_id.append(len(_input_ids))

        input_ids[oindex] = _input_ids
        loss_mask[oindex] = _loss_mask
        seq_start_id[oindex] = _seq_start_id[:-1]

    output_data = []
    for i in range(len(input_ids)):
        item_dict = {'input_ids': input_ids[i], 'loss_mask': loss_mask[i], 'seq_start_id': seq_start_id[i]}
        output_data.append(item_dict)

    assert all(not seq[0] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    assert all(not seq[1] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    np.save(output_path, output_data)
    logging.info(f"Done, output written to {output_path}")


def tokenize_dataset(cfg):
    logging.info("Tokenizing dataset...")
    # using the same template as SFT/PEFT script. This may be overkill but guarantees the preprocess settings
    # are identical to normal SFT training
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
    model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)

    # we set is_train=False to turn off samples mapping and get the actual length of train dataset
    train_ds = model._build_dataset(cfg.model.data.train_ds, is_train=False)[0]
    return np.array([train_ds[i] for i in range(len(train_ds))])


def create_hist(dataset, truncate_seq_len):
    logging.info("Creating histogram from tokenized dataset...")

    sequences = collections.defaultdict(list)
    counts = [0] * truncate_seq_len

    for item_dict in dataset:
        seq_len = len(item_dict['input_ids']) - 1
        sequences[seq_len].append(item_dict)
        counts[seq_len] += 1

    logging.info("Histogram of sequence lengths")
    logging.info(counts)

    histogram = []
    for seq_len in range(truncate_seq_len):
        histogram.append(len(sequences[seq_len]))

    return sequences, histogram


def run_packing(sequences, histogram, output_dir, pack_size, packing_algorithm, seed=0):
    logging.info(f"Packing sequences to length {pack_size}...")

    all_seq_lens = []
    for i, count in enumerate(histogram):
        all_seq_lens.extend([i] * count)

    packing_fn = globals()[packing_algorithm]
    assignments = packing_fn(all_seq_lens, pack_size)
    packed_seq_lens = [sum(x) for x in assignments]
    packing_factor = len(all_seq_lens) / len(packed_seq_lens)

    logging.info("Packed sequence lengths:")
    logging.info(packed_seq_lens)
    logging.info(
        f">>>>> For pack size {pack_size}, average number of sequences per pack is n = {packing_factor} <<<<<"
    )

    ifile_handles = {}
    for seq_len in tqdm(range(pack_size + 1)):
        per_seq_data = sequences[seq_len]
        if len(per_seq_data) > 0:
            input_ids = np.array([x['input_ids'] for x in per_seq_data])
            loss_mask = np.array(
                [[idx >= x['answer_start_idx'] for idx in range(len(x['input_ids']))] for x in per_seq_data]
            )
            perm = np.random.permutation(len(input_ids))
            ifile_handles[seq_len] = (input_ids[perm].tolist(), loss_mask[perm].tolist())
        else:
            ifile_handles[seq_len] = [], []

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'packed_{pack_size}_seed{seed}.npy')
    create_assignment(output_path, assignments, ifile_handles)


@dataclass
class PackingArgs:
    output_dir: str = "output"
    pack_sizes: Tuple[int] = (2048,)
    packing_algorithm: str = "first_fit_shuffle"
    seed: int = 0

    def from_config(self, cfg):
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
def main(cfg) -> None:
    args = PackingArgs().from_config(cfg)
    dataset = tokenize_dataset(cfg)
    sequences, histogram = create_hist(dataset, cfg.model.data.train_ds.max_seq_length)
    for pack_size in args.pack_sizes:
        run_packing(sequences, histogram, args.output_dir, pack_size, args.packing_algorithm, args.seed)
    logging.info(
        f"""
✅ Packed datasets with pack sizes {args.pack_sizes} are prepared successfully.
To train with packed sequences, you need to change three things in the SFT/PEFT config file
1. Turn on the packed_sequence flag 
   > +model.data.train_ds.packed_sequence=True
2. Use the new dataset file instead of the original jsonl file
   > model.data.train_ds.file_names=/path/to/packed_dataset.npy
3. Adjust the batch sizes. 
   Micro batch size has to be set to 1 as a nominal constraint. This is because batches are now concatenated 
   in the preprocessing step. You can increase the pack_size to achieve the same purpose of increasing micro batch size.
   Global batch size has to be reduced by the average number of sequences per pack `n`, 
   where n = total number of sequences / total number of packs. This ensures that each gradient iteration 
   sees (on average) the same number of sequences so that the recipe is maintained.
   Please scroll up to see the value of n for each of your pack sizes.
   > model.micro_batch_size=1
   > model.global_batch_size=<previous GBS divided by n>
"""
    )


if __name__ == '__main__':
    main()
