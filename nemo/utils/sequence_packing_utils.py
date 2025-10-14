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

import collections
import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from nemo.utils import logging

PACKING_ALGOS = ["first_fit_decreasing", "first_fit_shuffle"]


def find_first_bin_that_fits(bins: List[List[int]], s: int, bin_size: int) -> int:
    """
    Finds the first bin in a list of bins that has enough space to fit a sequence of size 's'.

    Args:
      bins: A list of lists, where each inner list represents a bin and contains the current elements in that bin.
      s: The size of the sequence to be placed in a bin.
      bin_size: The maximum capacity of each bin.

    Returns:
      The index of the first bin that can fit the sequence 's', or -1 if no such bin exists.
    """
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, where each inner list represents a bin and contains the indices
        of the sequences assigned to that bin.
    """
    res = []
    for s in seqlens:
        first_bin = find_first_bin_that_fits(res, s, pack_size)
        if first_bin == -1:  # open a new bin
            res.append([s])
        else:
            res[first_bin].append(s)
    return res


def first_fit_decreasing(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit Decreasing algorithm.

    This is a variation of the First-Fit algorithm where the sequences are sorted by decreasing length before packing.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    sorted_seqlens = sorted(seqlens, reverse=True)
    return first_fit(sorted_seqlens, pack_size)


def first_fit_shuffle(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit with Shuffling algorithm.

    This variation shuffles the order of the sequences before applying the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    shuffled_seqlens = seqlens[:]
    np.random.shuffle(shuffled_seqlens)
    return first_fit(shuffled_seqlens, pack_size)


def first_fit_shuffle_with_heap(
    seqlens: list[int], pack_size: int, shuffle: bool = True, seed: int | None = 234
) -> list[list[int]]:
    """A custom packing routine.
    Packs sequences of varying lengths into bins using a First-Fit-like algorithm.
    
    This routine is similar in logic to First-Fit: for every new sequence, look for an 
    existing bin that can fit it, otherwise open a new bin.
    While the original First-Fit version uses a greedy function called
    `find_first_bin_that_fits`, here we greedily look for an accomodating bin using a
    continuously updated heap. For large datasets, this makes it 100x-1000x faster.

    In this routine, seqlens can be shuffled before packing, which is necessary to
    preserve the packing efficiency (i.e. the average number of sequences per pack).

    It is recommended to use shuffle=True (default) to increase the packing efficiency.

    Args:
        seqlens: A list of integers, representing the lengths of the sequences to be packed.
        pack_size: The maximum capacity of each bin.
        shuffle: Whether to shuffle the sequence lengths before packing.
        seed: Random seed for shuffling.

    Returns:
        A list of lists, similar to the output of the 'first_fit' function.
    """

    if not seqlens:
        return []

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(seqlens)

    s = seqlens[0]
    res = [(s, [s])]
    for s in tqdm(seqlens[1:], desc="Creating packing strategy"):
        # Check the first bin: it is the one with the smallest total sequence length.
        # If it is possible to add the sequence to it without exceeding the pack size,
        # then add the sequence to the bin. Otherwise, open a new bin.
        first_bin_sum = res[0][0]
        if first_bin_sum + s <= pack_size:
            first_bin_sum, first_bin = heapq.heappop(res)
            first_bin.append(s)
            first_bin_sum += s
            heapq.heappush(res, (first_bin_sum, first_bin))
        else:
            heapq.heappush(res, (s, [s]))
    return [bin for _, bin in res]


def next_multiple_of(n, m):
    """Return the next multiple of m greater than or equal to n."""
    return (n + m - 1) // m * m


def create_hist(dataset: np.array, truncate_seq_len: int, divisibility_factor: Optional[int] = 16):
    """
    Creates a histogram of sequence lengths from a tokenized dataset.

    This function analyzes the tokenized dataset and creates a histogram showing the distribution of sequence lengths.

    Args:
      dataset: A NumPy array containing the tokenized sequences. Each element is a dictionary that contains at minimum
               the key `input_ids`.
      truncate_seq_len: The maximum sequence length to consider in the histogram.

    Returns:
      sequences: A dictionary where keys are sequence lengths and values are lists
                 of corresponding sequences from the dataset.
      histogram: A list representing the histogram data (number of sequences for each length).
    """
    logging.info("Creating histogram from tokenized dataset...")

    if divisibility_factor is not None and truncate_seq_len % divisibility_factor:
        raise ValueError(
            f"{truncate_seq_len=} must be a multiple of {divisibility_factor=}"
        )

    sequences = collections.defaultdict(list)
    counts = [0] * (truncate_seq_len + 1)

    for item_dict in dataset:
        # The data processing pipeline downstream is expected to be the following:
        # - REMOVE THE LAST TOKEN -> the -1 here to account for the fact that
        #   transformer input and labels have one less token than the full sequence:
        #   input is missing the last token and label is missing the first token
        #   (this way the tokens are aligned for next token prediction).
        # - (POSSIBLY) PAD TO THE NEXT MULTIPLE OF `divisibility_factor` -> we
        #   virtually pad the sequence length to the next multiple of this value (the
        #   sequence is not modified, it is only assigned to a different length bin). If
        #   the sequence is not padded downstream, nothing is impacted except the data
        #   packing is slightly less optimal, since we may pack less sequences together.
        # - PACKING -> concatenate the resulting sequences into a single packed one.
        seq_len = len(item_dict["input_ids"]) - 1
        if divisibility_factor is not None:
            seq_len = next_multiple_of(seq_len, divisibility_factor)
        sequences[seq_len].append(item_dict)
        counts[seq_len] += 1

    logging.debug("Histogram of sequence lengths")
    logging.debug(counts)

    histogram = []
    for seq_len in range(truncate_seq_len + 1):
        histogram.append(len(sequences[seq_len]))

    return sequences, histogram


def create_packing_strategy(
    histogram: List[int], pack_size: int, packing_algorithm: str = "first_fit"
) -> Tuple[List[List[int]], dict]:
    """
    Packs sequences into bins using the specified packing algorithm.

    This function takes the histogram of sequence lengths, desired pack size, and a string representing the packing
    algorithm to use. It then calls the corresponding function (e.g., 'first_fit_decreasing') and performs the
    packing process using only sequence lengths as input (without the actual sequences).

    Args:
          histogram: A list representing the histogram data (number of sequences for each length).
          pack_size: The maximum capacity of each bin.
          packing_algorithm: One of the supported packing algorithms from ['first_fit_decreasing', 'first_fit_shuffle']

    Returns:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin.
          pack_metadata: A dict that records packing metadata, for instance the max number of samples per bin.
    """

    logging.info(f"Packing sequences to length {pack_size}...")

    all_seq_lens = []
    for i, count in enumerate(histogram):
        all_seq_lens.extend([i] * count)

    packing_fn = globals()[packing_algorithm]
    assignments: List[List[int]] = packing_fn(all_seq_lens, pack_size)
    packed_seq_lens = [sum(x) for x in assignments]
    packing_factor = len(all_seq_lens) / len(packed_seq_lens)

    max_seqlen = max(all_seq_lens)
    max_samples_per_bin = max([len(b) for b in assignments])
    min_packed_seqlen = min(packed_seq_lens)
    packing_metadata = {
        "dataset_max_seqlen": max_seqlen,
        "max_samples_per_bin": max_samples_per_bin,
        "packing_factor": round(packing_factor, 2),
        "packing_efficiency": round(sum(packed_seq_lens) / len(packed_seq_lens) / pack_size * 100, 2),
        "pack_size": pack_size,
        'min_packed_seqlen': min_packed_seqlen,
    }
    logging.debug("Packed sequence lengths:")
    logging.debug(packed_seq_lens)
    logging.info(f"Packing is {sum(packed_seq_lens) / len(packed_seq_lens) / pack_size * 100:.2f}% efficient")
    logging.info(
        f">>>>> For pack size {pack_size}, average number of sequences per pack is n = {packing_factor:.3f} <<<<<"
    )
    return assignments, packing_metadata


def fill_packing_strategy(
    assignments: List[List[int]],
    sequences: Dict[int, List[Dict]],
    pack_size: int,
    pad_id: int,
) -> List[Dict]:
    """
    Fills the packing strategy with actual sequence data based on assignments and sequence information.

    This function takes the assignments generated by the packing algorithm (containing sequence length indices),
    the original sequences data, and the pack size. It iterates through the assignments, retrieves the corresponding
    sequences from the sequences dictionary, and constructs the final output data structure with input IDs, loss masks
    (if available), and starting indices for each sequence in a packed sequence.

    Args:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin (output of 'create_packing_strategy').
          sequences: A dictionary where keys are sequence lengths and values are lists of corresponding sequences
                      from the dataset (output of 'create_hist').
          pack_size: The maximum capacity of each bin.
          pad_id: The tokenizer's padding token.

    Returns:
          output_data: A list of dictionaries, where each dictionary represents a packed sequence with its input IDs,
                        loss mask (if available), and starting indices.
    """
    ifile_handles = dict()
    for seq_len in tqdm(range(pack_size + 1)):
        per_seq_data = sequences[seq_len]
        if len(per_seq_data) > 0:
            perm = np.random.permutation(len(per_seq_data))
            input_ids = [per_seq_data[idx]["input_ids"] for idx in perm]
            try:
                loss_mask = [per_seq_data[idx]["loss_mask"] for idx in perm]
            except KeyError:
                try:
                    loss_mask = [
                            [
                                # (x['answer_start_idx'] - 1) because we want to train on the output
                                # after the last context token
                                idx >= (x["answer_start_idx"] - 1)
                                for idx in range(len(x["input_ids"]))
                            ]
                            for x in per_seq_data
                        ]
                    loss_mask = [loss_mask[idx] for idx in perm]
                except KeyError as err:
                    err_msg = "Key errors loss_mask and answer_start_idx missing in example - "
                    err_msg += f"{err} {per_seq_data[0]}"
                    logging.error(err_msg)
                    raise ValueError(err_msg)

            ifile_handles[seq_len] = (input_ids, loss_mask)

    input_ids = [[0] * len(assignment) for assignment in assignments]
    loss_mask = [[0] * len(assignment) for assignment in assignments]
    seq_start_id = [[0] * (len(assignment) + 1) for assignment in assignments]
    for oindex, assignment in tqdm(
        enumerate(assignments),
        total=len(assignments),
        desc="Creating packed sequences",
    ):
        seq_start_id[oindex][0] = 0
        for j, seq_length in enumerate(assignment):
            input_ids[oindex][j] = ifile_handles[seq_length][0].pop()
            loss_mask[oindex][j] = ifile_handles[seq_length][1].pop()
            seq_start_id[oindex][j + 1] = (
                len(input_ids[oindex][j]) + seq_start_id[oindex][j]
            )

    output_data = []
    for i in range(len(input_ids)):
        item_dict = {
            "input_ids": np.concatenate([np.array(x) for x in input_ids[i]]).reshape(-1),
            "loss_mask": np.concatenate([np.array(x) for x in loss_mask[i]]).reshape(-1),
            "seq_start_id": seq_start_id[i],
        }
        output_data.append(item_dict)

    assert all(not seq[0] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    assert all(not seq[1] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    return output_data


def pad_thd_sequences_for_cp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    padding_token_id: int = 0,
    padding_label_id: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences to be divisible by the divisibility factor.
    Literally a copy-paste of the same function from transformer_engine, see
    https://github.com/NVIDIA/TransformerEngine/blob/dfacd9f76bcabcdd53cb30a17679ad6032cf54f4/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py

    Args:
        input_ids: Tensor of shape (1, N) or (N,) containing concatenated sequences
        labels: Tensor of shape (1, N) or (N,) containing labels for each token
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        padding_token_id: Token ID to use for padding (default: 0)
        padding_label_id: Label ID to use for padding (default: -100)

    Returns:
        Tuple of:
        - input_ids_padded: Padded input_ids tensor
        - labels_padded: Padded labels tensor
        - cu_seqlens_padded: Cumulative sequence lengths accounting for padding
    """
    # Flatten input_ids and labels if needed
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)

    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence (make length a multiple of divisibility_factor)
    padding_amounts = [
        ((l.item() + divisibility_factor - 1) // divisibility_factor)
        * divisibility_factor
        - l.item()
        for l in seqlens
    ]

    # Extract sequences and labels for each batch item
    batch_sequences = [
        input_ids[start.item() : end.item()]
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ]
    batch_labels = [
        labels[start.item() : end.item()]
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ]

    # Pad sequences and labels to required length
    input_ids_padded = torch.cat(
        [
            (
                torch.cat([seq, torch.full((pad,), padding_token_id, dtype=seq.dtype)])
                if pad > 0
                else seq
            )
            for seq, pad in zip(batch_sequences, padding_amounts)
        ]
    )
    labels_padded = torch.cat(
        [
            (
                torch.cat([seq, torch.full((pad,), padding_label_id, dtype=seq.dtype)])
                if pad > 0
                else seq
            )
            for seq, pad in zip(batch_labels, padding_amounts)
        ]
    )

    # Compute cumulative padded sequence lengths, starting from 0
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype)
    cu_seqlens_padded = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype), padded_lengths]), dim=0
    )

    return input_ids_padded, labels_padded, cu_seqlens_padded


def generate_positional_ids_for_cp(
    cu_seqlens: torch.Tensor,
    divisibility_factor: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Generate positional IDs for sequences padded to be divisible by divisibility_factor.
    Literally a copy-paste of the same function from transformer_engine, see
    https://github.com/NVIDIA/TransformerEngine/blob/dfacd9f76bcabcdd53cb30a17679ad6032cf54f4/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py

    Args:
        cu_seqlens: Tensor of shape (M,) containing cumulative sequence lengths
        divisibility_factor: Each sequence length must be divisible by this factor
        dtype: Data type for the generated positional IDs (default: torch.long)

    Returns:
        Generated positional_ids tensor where each sequence starts from 0 and continues through padding
    """
    # Compute the sequence lengths from cu_seqlens
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    # List: amount of padding needed for each sequence
    padding_amounts = [
        ((l.item() + divisibility_factor - 1) // divisibility_factor)
        * divisibility_factor
        - l.item()
        for l in seqlens
    ]

    # Generate positional IDs for each padded sequence (each starts from 0)
    padded_lengths = seqlens + torch.tensor(padding_amounts, dtype=seqlens.dtype)
    positional_ids = torch.cat(
        [torch.arange(0, int(length), dtype=dtype) for length in padded_lengths]
    )

    return positional_ids
