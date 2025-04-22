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

from typing import Dict, List, Union

import torch
from datasets import Dataset
from torch.nn import functional as F
from tqdm import tqdm

from nemo.utils import logging

CROSS_ENTROPY_IGNORE_IDX = -100
PACK_TYPE = Dict[str, Union[torch.Tensor, List[int]]]


# based on https://github.com/pytorch/torchtune/blob/v0.6.1/torchtune/datasets/_packed.py#L17
class HFDatasetPackedSequenceHelper:
    """
    Args:
    dataset: Actual dataset (can be 'train', 'val' or 'test')
    split (str): Whether the dataset is 'train', 'val' or 'test'
    """

    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
        self.padding_idx = 0  # Padding value to pack a sequence to self.packed_sequence_size
        self.contains_loss_mask = False

    def pack(self, packed_sequence_size, split_across_pack, max_packs):
        """Iterate through the dataset. Use a buffer to hold samples until packed_sequence_size,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset.

        Args:
        packed_sequence_size (int): Number of tokens in a pack
        split_across_pack (bool): If the last sample in a pack does not fit in ``packed_sequence_size``,
        split the sample into the next pack, or move it entirely to the beginning of the next pack.
        max_packs (int): Maximum number of packs.

        Returns the Packed dataset which is an object of Dataset class.

        """
        self.packed_sequence_size = packed_sequence_size
        self.split_across_pack = split_across_pack
        self.max_packs = max_packs
        # Only show progress bar on rank 0
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )

        # Pack dataset
        self.packs: List[PACK_TYPE] = []
        if "loss_mask" in self.dataset[0]:
            self.contains_loss_mask = True
        # Buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "tokens": [],
            "labels": [],
            "position_ids": [],
            "seq_lens": [],
        }
        if self.contains_loss_mask:
            current_pack["loss_mask"] = []
        self.previous_sample_boundary: int = 0
        if rank == 0:
            pbar = tqdm(total=len(self.dataset), desc=f"Packing {self.split} dataset", dynamic_ncols=True)
        for sample in self.dataset:
            tokens, labels = sample["input_ids"], sample["labels"]
            if self.contains_loss_mask:
                loss_mask = sample["loss_mask"]
            # If the dataset outputs samples that are larger than the specified
            # packed_sequence_size and we're unable to split it, user needs to modify
            # one of the two parameters
            seq_len = len(tokens)
            if seq_len > self.packed_sequence_size and not split_across_pack:
                raise ValueError(
                    f"Dataset sample is too long ({seq_len} > {self.packed_sequence_size}). "
                    "Please set `split_across_pack=True` or increase `packed_sequence_size`."
                )
            # Update the current pack
            # "position_ids" is the pos ids, "seq_lens" is the len of each seq within the pack
            current_pack["tokens"] += tokens
            current_pack["labels"] += labels
            current_pack["position_ids"] += [x % self.packed_sequence_size for x in range(seq_len)]
            current_pack["seq_lens"] += [seq_len]
            if self.contains_loss_mask:
                current_pack["loss_mask"] += loss_mask

            # If the current pack is over the packed_sequence_size, add it to self.packs and
            # retain any truncated or bumped samples for next pack
            while len(current_pack["tokens"]) > self.packed_sequence_size and not self._should_stop_packing():
                current_pack = self._split_and_add_pack(current_pack)

            if rank == 0:
                pbar.update()

            # Keep track of previous sample boundary
            self.previous_sample_boundary = len(current_pack["tokens"])

            if self._should_stop_packing():
                break

        # Handle the last pack if there's leftover and we haven't filled up the max packs
        if len(current_pack["tokens"]) > 0 and (self.max_packs is None or len(self.packs) < self.max_packs):
            # No need to handle splitting at this point so we can just add the current pack
            self._add_pack(current_pack)

        # After packing all samples, convert self.packs to a Dataset object
        packed_dataset = Dataset.from_dict({key: [pack[key] for pack in self.packs] for key in self.packs[0].keys()})
        logging.info(f">>>>> Total number of packs created: {len(self.packs)} <<<<<")
        return packed_dataset

    def _should_stop_packing(self) -> bool:
        """If max packs is set, stop packing when we reach that number."""

        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
        """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
        returns the start of the next pack."""

        if self.split_across_pack:
            boundary = self.packed_sequence_size
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.packed_sequence_size``
            leftover_seq_len = self.packed_sequence_size - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "tokens": current_pack["tokens"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "position_ids": current_pack["position_ids"][:boundary],
            "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
        }
        if self.contains_loss_mask:
            pack["loss_mask"] = current_pack["loss_mask"][:boundary]

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["tokens"][boundary:]) if self.split_across_pack else current_pack["seq_lens"][-1]
        )

        output_dict = {
            "tokens": current_pack["tokens"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "position_ids": current_pack["position_ids"][boundary:],
            "seq_lens": [next_seq_len],
        }
        if self.contains_loss_mask:
            output_dict["loss_mask"] = current_pack["loss_mask"][boundary:]
        return output_dict

    def _add_pack(self, pack: PACK_TYPE) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
        tensor_pack = {
            "tokens": torch.tensor(pack["tokens"], dtype=torch.long),
            "labels": torch.tensor(pack["labels"], dtype=torch.long),
            "position_ids": torch.tensor(pack["position_ids"], dtype=torch.long),
            "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
        }
        if self.contains_loss_mask:
            tensor_pack["loss_mask"] = torch.tensor(pack["loss_mask"], dtype=torch.long)
        return tensor_pack

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.packed_sequence_size``."""
        # Pad tokens
        num_padding_tokens = self.packed_sequence_size - len(pack["tokens"])
        padded_tokens = F.pad(
            pack["tokens"],
            (0, num_padding_tokens),
            value=padding_idx,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.packed_sequence_size - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Pad loss_mask
        if self.contains_loss_mask:
            padded_loss_mask = F.pad(
                pack["loss_mask"],
                (0, self.packed_sequence_size - len(pack["loss_mask"])),
                value=0,
            )

        # Add padding tokens as a last seq len to ensure sum is packed_sequence_size
        padded_seq_lens = (
            torch.cat([pack["seq_lens"], torch.tensor([num_padding_tokens])])
            if num_padding_tokens > 0
            else pack["seq_lens"]
        )

        # Pad position_ids continuing the sequence from last value
        # in position_ids
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.packed_sequence_size = 6
        num_range = torch.arange(
            pack["position_ids"][-1] + 1,
            pack["position_ids"][-1] + self.packed_sequence_size - len(pack["position_ids"]) + 1,
        )
        # Clamp to packed_sequence_size - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.packed_sequence_size - 1)
        padded_position_ids = torch.cat([pack["position_ids"], clamped_num_range])

        padded_pack = {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "position_ids": padded_position_ids,
            "seq_lens": padded_seq_lens,
        }
        if self.contains_loss_mask:
            padded_pack["loss_mask"] = padded_loss_mask
        return padded_pack


def create_block_causal_mask(seq_lens: List[torch.Tensor]) -> torch.Tensor:
    """
    Given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::
        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.
    Returns:
        Tensor: Block causal mask of shape (batch_size, packed_sequence_size, packed_sequence_size).
    """
    batch_block_attn_masks = []
    batch_size = len(seq_lens)
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                )
            )
            for i, seq_len in enumerate(seq_lens[sample_idx])
        ]

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
    # Transformers expects the attn_mask to be 4d [bs, 1, packed_sequence_size, packed_sequence_size], hence adding
    # singleton (size 1) dimension at position 1.
    return torch.stack(batch_block_attn_masks).unsqueeze(1)


def packed_block_causal_mask(seq_lens: List[torch.Tensor]):
    """
    Create a block causal document mask for a batch of packed sequences. A standard 2D block causal mask is created
    and returned.
    Args:
        seq_lens (List[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.
    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """
    return create_block_causal_mask(seq_lens=seq_lens)
