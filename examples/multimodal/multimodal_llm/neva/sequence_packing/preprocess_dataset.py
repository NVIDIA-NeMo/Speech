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

"""
Example Usage:
--------------
This script preprocesses a dataset for the NeMo Multimodal Learning framework. It requires specifying paths for data, images, and the tokenizer model, among other parameters.

Command:
python examples/multimodal/multimodal_llm/neva/sequence_packing/preprocess_dataset.py \
 --data_path=/path/to/LLaVA-Instruct-150K/llava_v1_5_mix665k_filtered.json \
 --image_folder=/path/to/LLaVA-Instruct-150K/images \
 --tokenizer_path=/path/to/checkpoints/tokenizer_add_special.model \
 --output_dir=/path/to/LLaVA-Instruct-150K/packed_seq_4096_336_v1 \
 --max_seq_length=12288 \
 --packing_algorithm=first_fit_shuffle \
 --hf_vision_encoder=openai/clip-vit-large-patch14-336 \
 --conv_template=v1 \
 --image_aspect_ratio=pad \
 --seed=42

Parameters:
-----------
--data_path: Path to the dataset file in JSON format.
--image_folder: Directory containing the images referenced in the dataset.
--tokenizer_path: Path to the tokenizer model.
--output_dir: Directory where the processed dataset will be stored.
--max_seq_length: The maximum sequence length of the model.
--packing_algorithm: Algorithm used for packing sequences. Defaults to 'first_fit_shuffle'.
--hf_vision_encoder: The Hugging Face vision encoder to use. Default is 'openai/clip-vit-large-patch14-336'.
--conv_template: Template for data conversion. Default is 'plain', with 'v1' as an alternative.
--image_aspect_ratio: The aspect ratio for processing images. Defaults to 'square', 'pad' for padding to maintain aspect ratio.
--seed: Seed for random operations in 'first_fit_shuffle'.
--hparams_file: Optional path to a YAML file containing additional hyperparameters.
"""

import collections
import os
import random
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder, get_bin_path, get_idx_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.multimodal.data.neva.neva_dataset import make_supervised_data_module
from nemo.collections.multimodal.parts.utils import create_image_processor
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

PACKING_ALGOS = ['first_fit_decreasing', 'first_fit_shuffle', 'shuffle_and_pack']


def first_fit(seq_lens, max_seq_length):
    """
    Assigns sequences to bins using the First Fit algorithm, by integrating the search
    and assignment within the same function. It moves bins that can no longer fit the minimum sequence length
    to a completed bins list, avoiding direct modification of the bins list during iteration.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    min_seq_len = min(seq_lens)  # Find the minimum sequence length
    completed_bins = []  # Initialize the completed bins list
    bins = []  # Initialize the bins list to store active bins

    for s in tqdm(seq_lens):  # Iterate through each sequence length
        found_bin = False
        for i, abin in enumerate(bins[:]):  # Iterate over a shallow copy of bins
            if sum(abin) + min_seq_len > max_seq_length:
                completed_bins.append(abin)  # Add to completed bins
                bins[i] = 'TO_REMOVE'  # Mark this bin for removal
                continue
            if sum(abin) + s <= max_seq_length:  # Check if the bin can fit the sequence
                bins[i].append(s)  # If so, add the sequence to this bin
                found_bin = True
                break

        if not found_bin:  # If no existing bin can fit the sequence
            bins.append([s])  # Open a new bin for this sequence

        # Clean up bins marked 'TO_REMOVE'
        bins = [bin for bin in bins if bin != 'TO_REMOVE']

    # Combine completed bins with any remaining active bins
    all_bins = completed_bins + bins
    return all_bins


def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parallel_first_fit(seq_lens, max_seq_length, chunk_size, num_workers):
    """
    Assigns sequences to bins in parallel using the First Fit algorithm.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.
    - chunk_size: Size of chunks to divide seq_lens into for parallel processing.
    - num_workers: Number of worker threads to use in the ThreadPoolExecutor.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    # Split the sequence lengths into chunks
    chunks = list(chunkify(seq_lens, chunk_size))

    # Function to process each chunk
    def process_chunk(chunk):
        return first_fit(chunk, max_seq_length)

    bins = []  # This will hold the final bins
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit each chunk to the executor
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        # As each future completes, combine its bins with the final bins
        for future in as_completed(futures):
            bins.extend(future.result())

    return bins


def first_fit_decreasing(seq_lens, max_seq_length):
    """
    Assigns sequences to bins using the First Fit Decreasing algorithm.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    sorted_seq_lens = sorted(seq_lens, reverse=True)
    return first_fit(sorted_seq_lens, max_seq_length)


def first_fit_shuffle(seq_lens, max_seq_length):
    """
    Assigns sequences to bins using a shuffled version of the First Fit algorithm.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    shuffled_seq_lens = seq_lens[:]
    np.random.shuffle(shuffled_seq_lens)
    return parallel_first_fit(shuffled_seq_lens, max_seq_length, 20000, 32)


def shuffle_and_pack(seq_lens, max_seq_length):
    """
    Assigns sequences to bins with shuffling, trying to maximize the packing efficiency.
    After shuffling the sequences, they will be added to one bin in order. Once the bin cannot
    take more sequences, we will move on to the next bin.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    shuffled_seq_lens = np.array(seq_lens)
    np.random.shuffle(shuffled_seq_lens)
    bins = [[]]
    cur_bin_total = 0
    for s in tqdm(shuffled_seq_lens):
        if cur_bin_total + s <= max_seq_length:
            bins[-1].append(s)
            cur_bin_total += s
        else:
            bins.append([s])
            cur_bin_total = s
    return bins


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument("--max_seq_length", default=4096, type=int)
    parser.add_argument('--packing_algorithm', default='first_fit_shuffle', choices=PACKING_ALGOS, type=str)
    parser.add_argument("--hf_vision_encoder", default='openai/clip-vit-large-patch14-336', type=str)
    parser.add_argument("--conv_template", default='plain', type=str)
    parser.add_argument("--image_aspect_ratio", default='square', type=str)
    parser.add_argument('--seed', default=0, type=int, help="Seed for shuffling, used with first_fit_shuffle.")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), '../conf/llava_config.yaml'),
        required=False,
        help="Path to the hparams file.",
    )
    return parser.parse_args()


def pack_sequence(args, seq_lens):
    """
    Packs sequences according to the specified algorithm in args.

    Parameters:
    - args: Command line arguments.
    - seq_lens: List of sequence lengths.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)

    packing_fn = globals()[args.packing_algorithm]
    bins = packing_fn(seq_lens, args.max_seq_length)
    return bins


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = get_args()
    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model.mm_cfg.vision_encoder.from_pretrained = args.hf_vision_encoder
    nemo_config.model.data.data_path = args.data_path
    nemo_config.model.data.image_folder = args.image_folder
    nemo_config.model.data.conv_template = args.conv_template
    nemo_config.model.data.image_aspect_ratio = args.image_aspect_ratio

    tokenizer = get_nmt_tokenizer(
        library="sentencepiece",
        tokenizer_model=args.tokenizer_path,
    )
    image_processor = create_image_processor(nemo_config.model.mm_cfg)
    train_ds = make_supervised_data_module(
        tokenizer=tokenizer, image_processor=image_processor, model_cfg=nemo_config.model
    )["train_dataset"]
    train_dl = DataLoader(train_ds, num_workers=32, collate_fn=None, shuffle=False)
    # Example shape: {'tokens': torch.Size([1, 344]), 'labels': torch.Size([1, 344]), 'image': torch.Size([1, 1, 3, 224, 224])}

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    prefix_path = f"{output_dir}/packed_seq_dataset"
    # Original Datasets to Sequence Lengths Files
    builders = {}
    for item_dict in tqdm(train_dl, desc="Building indexed datasets"):
        item_dict = {k: v[0] for k, v in item_dict.items()}
        seq_len = len(item_dict['tokens'])
        if seq_len in builders:
            builder = builders[seq_len]
        else:
            builder_path = get_bin_path(f"{prefix_path}/seqlen_{seq_len}")
            logging.info(f"Creating builder for sequence length {seq_len} at {builder_path}")
            builder = IndexedDatasetBuilder(builder_path, dtype=np.float32, multimodal=True)
            builders[seq_len] = builder
        builder.add_item(item_dict['tokens'])
        builder.add_item(item_dict['labels'])
        builder.add_item(item_dict['image'], 1)
        builder.end_document()
        del item_dict

    for seq_len, builder in builders.items():
        idx_path = get_idx_path(f"{prefix_path}/seqlen_{seq_len}")
        logging.info(f"Finalizing builder for sequence length {seq_len} at {idx_path}")
        builder.finalize(idx_path)

    # Packing Sequences into Bins
    files = os.listdir(f"{output_dir}/packed_seq_dataset")
    pattern = rf"seqlen_(\d+).bin"
    seq_len_list = []
    for file in files:
        match = re.match(pattern, file)
        if match:
            seq_len = int(match.group(1))
            seq_len_list.append(seq_len)

    aggregated_seq_lens = []
    doc_pop_order = {}
    indexed_datasets = {}
    for seq_len in seq_len_list:
        dataset_path = f"{prefix_path}/seqlen_{seq_len}"
        dataset = IndexedDataset(dataset_path, multimodal=True)
        aggregated_seq_lens.extend([seq_len] * (len(dataset.document_indices) - 1))
        doc_pop_order[seq_len] = list(np.random.permutation(len(dataset.document_indices) - 1))
        indexed_datasets[seq_len] = dataset

    logging.info("Getting bins")
    bins = pack_sequence(args, aggregated_seq_lens)
    logging.info("Finished getting bins")

    num_bins = len(bins)
    avg_bins_len = sum([len(x) for x in bins]) / num_bins
    avg_bins_sum = sum([sum(x) for x in bins]) / num_bins
    logging.info(f"Number of bins: {num_bins}, Average bin length: {avg_bins_len}, Average bin sum: {avg_bins_sum}")

    # Reading Sequence Lengths and Packing into New Files
    final_builder_path = get_bin_path(f"{prefix_path}")
    logging.info(f"Creating final builder at {final_builder_path}")
    final_builder = IndexedDatasetBuilder(final_builder_path, dtype=np.float32, multimodal=True)

    for assignment in tqdm(bins, desc="Building final dataset"):
        packed_items = collections.defaultdict(list)
        packed_items["seq_indices"] = [0]
        for seq_len in assignment:
            doc_index = doc_pop_order[seq_len].pop()
            doc_start = indexed_datasets[seq_len].document_indices[doc_index]
            doc_end = indexed_datasets[seq_len].document_indices[doc_index + 1]
            item_dict = {
                "tokens": torch.tensor((indexed_datasets[seq_len][doc_start:doc_end][0])[0]),
                "labels": torch.tensor((indexed_datasets[seq_len][doc_start:doc_end][0])[1]),
                "image": torch.tensor((indexed_datasets[seq_len][doc_start:doc_end][0])[2]),
            }
            for key in ["tokens", "labels", "image"]:
                packed_items[key].append(item_dict[key])
            packed_items["seq_indices"].append(packed_items["seq_indices"][-1] + seq_len)

        for key in ["seq_indices", "tokens", "labels", "image"]:
            final_builder.add_item(
                torch.tensor(packed_items[key]) if key == "seq_indices" else torch.cat(packed_items[key], dim=0),
                1 if key == "image" else 0,
            )
        final_builder.end_document()

    idx_path = get_idx_path(f"{prefix_path}")
    logging.info(f"Finalizing final builder at {idx_path}")
    final_builder.finalize(idx_path)
    logging.info(f"Number of bins: {num_bins}, Average bin length: {avg_bins_len}, Average bin sum: {avg_bins_sum}")


if __name__ == "__main__":
    main()
