# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

from typing import ClassVar, Dict, Optional

import torch
from megatron.core.datasets.gpt_dataset import GPTDataset
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import make_upper_case


class Evo2Dataset(GPTDataset):
    """Dataset for training Evo2."""

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0
    TO_UPPER_TOKENS: bool = True  # If set, do an in-place transform to make all tokens capital letters
    RESET_PAD_EOD_MASK: bool = True  # If set, unset the mask for [pad] and [eod] tokens (matches Evo2 paper).
    # Valid DNA tokens: A, C, G, T, U, W, S, M, K, R, Y, B, D, H, V, N, -,  (both uppercase and lowercase and
    #   degenerate bases and RNA)
    MAX_TAG_LEN = 2048

    VALID_DNA_AND_DEGENERATE: ClassVar[set[int]] = {
        45,
        45,
        65,
        66,
        67,
        68,
        71,
        72,
        75,
        77,
        78,
        82,
        83,
        84,
        85,
        86,
        87,
        89,
        97,
        98,
        99,
        100,
        103,
        104,
        107,
        109,
        110,
        114,
        115,
        116,
        117,
        118,
        119,
        121,
    }
    DNA_TOKENS: list[int] = [65, 67, 71, 84, 97, 99, 103, 116]

    def _get_gpt_batch(self, idx: Optional[int]) -> dict[str, torch.Tensor]:
        return super().__getitem__(idx)

    def _modify_gpt_batch(self, databatch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss_mask = databatch.get("loss_mask", None)
        if self.RESET_PAD_EOD_MASK and loss_mask is not None:
            # Reset the mask for 'pad', '[eod]', '[pad token]', which will lower the loss, but matches Evo2 pub.
            loss_mask = torch.ones_like(loss_mask)
        labels = databatch.get("labels", None)
        if labels is None or loss_mask is None:
            # No next-token labels or loss to mask.
            return databatch

        # Mask special label tags in loss.
        control_mask = torch.isin(labels, torch.tensor(self.CONTROL_TAGS, device=labels.device))
        # Mask degenerate (and U) DNA tokens
        not_dna_mask = ~torch.isin(labels, torch.tensor(self.DNA_TOKENS, device=labels.device))
        loss_mask[control_mask | not_dna_mask] = 0
        phylotag_mask = self.mask_phylogenetic_tags(
            labels,
            self.TAG_BOUNDS,
            self.TAG_CHARS,
            self.config.tokenizer.eod if self.config.tokenizer is not None else self.DEFAULT_EOD,
            self.MAX_TAG_LEN,
        )
        databatch["loss_mask"] = loss_mask * phylotag_mask
        if self.TO_UPPER_TOKENS:
            # When making tokens uppercase, make sure this is done after the mask_phylogenetic_tags function which
            #  relies in part on the original case of the tag tokens.
            databatch["tokens"], _ = make_upper_case(databatch["tokens"])
        return databatch

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Get data at the specified index."""
        # 1. Call the default gpt dataset object
        databatch: dict = self._get_gpt_batch(idx)
        # 2. Modify loss tokens and upper-case as configured.
        return self._modify_gpt_batch(databatch)

    @staticmethod
    def mask_phylogenetic_tags(
        tokenized_sequence: torch.Tensor,
        terminal_tag_char: int,  # e.g. ASCII for '|'
        other_tag_chars: set[int],  # e.g. {95, 59, 32} for '_', ';', space
        eod_token_id: int,  # e.g. 0
        max_tag_len: int,
    ) -> torch.Tensor:
        """
        Creates a binary mask for sequences containing phylogenetic tags and DNA.
        The rules are as follows (applied per contiguous sub‐sequence between EOD tokens):

          - Any token equal to the terminal_tag_char (the pipe, '|') is masked.
          - For the region *before* the first pipe (the “prefix”):
              * If the first token is in taxonomy_prefixes (d, p, c, o, f, g, s),
                or if the prefix is exactly one lowercase letter,
                or if any token in the prefix is one of other_tag_chars,
                or if not every token is a valid DNA base,
                then mask the entire prefix.
          - For the region between pipes:
              * If any token is in other_tag_chars or not all tokens are valid DNA, mask that region.
          - For the region *after* the last pipe (the “suffix”):
              * If the first token is the letter 'd' (ASCII 100) or if the region contains
                any other tag characters or any EOD tokens or non‐DNA, mask the suffix.

        Finally, any token equal to eod_token_id is forced to remain unmasked.
        (EOD tokens “break” a sequence so that tags never span across them.)

        Args:
            tokenized_sequence (torch.Tensor): shape (seq_len,) or (batch_size, seq_len)
              containing ASCII values.
            terminal_tag_char (int): ASCII value for the pipe character.
            other_tag_chars (set[int]): Set of ASCII values that appear only in tags.
            eod_token_id (int): The token ID for EOD.

        Notes:
        - The tag token is constructed as follows: So note that one way to know you have a tag is if you look
         at the first token after the pipe and it is a 'd' character. Make sure implementation handles this.
            ```
            return (
                "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
                    lineage.domain if random.random() >= dropout else None,
                    lineage.phylum if random.random() >= dropout else None,
                    lineage.clazz if random.random() >= dropout else None,
                    lineage.order if random.random() >= dropout else None,
                    lineage.family if random.random() >= dropout else None,
                    lineage.genus if random.random() >= dropout else None,
                    lineage.species if random.random() >= dropout else None,
                )
                if lineage is not None
                else None
            )
            ```
        Returns:
            torch.Tensor: A mask of the same shape as input where 1 = keep (DNA) and 0 = mask (tag).
        """
        device = tokenized_sequence.device
        dtype = tokenized_sequence.dtype
        # Handle empty or single-token sequences.
        if tokenized_sequence.numel() == 0:
            return torch.ones(0, device=device, dtype=torch.int)
        if tokenized_sequence.numel() == 1:
            mask = torch.ones(1, device=device, dtype=torch.int)
            token = tokenized_sequence.item()
            if token == terminal_tag_char or token in other_tag_chars:
                mask[0] = 0
            return mask

        # Ensure input is 2D (batch, seq_len)
        batched = tokenized_sequence.ndim == 2
        if not batched:
            tokenized_sequence = tokenized_sequence.unsqueeze(0)
        batch_size, seq_len = tokenized_sequence.shape

        valid_dna_or_control_tensor = torch.tensor(
            list(Evo2Dataset.VALID_DNA_AND_DEGENERATE | set(Evo2Dataset.CONTROL_TAGS)), device=device, dtype=dtype
        )

        # Initialize output mask to all ones.
        out_mask = torch.ones_like(tokenized_sequence, dtype=torch.int)

        # Helper: Check if all tokens in a region are valid DNA.
        def region_all_valid_or_control(region: torch.Tensor) -> bool:
            if region.numel() == 0:
                return True
            # Using torch's all() over the token values.
            return bool(torch.all(torch.isin(region, valid_dna_or_control_tensor)).cpu().item())

        # Process one EOD-free segment using the O1 logic.
        def process_segment(seg_seq: torch.Tensor) -> torch.Tensor:
            seg_len = seg_seq.size(0)
            seg_mask = torch.ones(seg_len, device=device, dtype=torch.int)
            # Identify positions of terminal tag (pipe)
            pipe_pos = (seg_seq == terminal_tag_char).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(pipe_pos) == 0:
                # If no pipe exists and any token is a known tag char or not valid DNA,
                # mask the entire segment, unless it is longer than the maximum tag len.
                if len(seg_seq) < max_tag_len and not region_all_valid_or_control(seg_seq):
                    seg_mask.zero_()
                return seg_mask

            # Always mask the pipe positions.
            seg_mask[pipe_pos] = 0

            # Does tag start before the first pipe? This determines the starting state of our state machine.
            first_pipe = pipe_pos[0]
            if first_pipe >= 0 and first_pipe < seg_len - 1:
                # fastest check is to look at the first token after the pipe, if it is a '*_' then the
                # tag starts _after_ the pipe, otherwise it starts before.
                if len(seg_seq) > first_pipe + 2:
                    first_tok = seg_seq[first_pipe + 1].item()
                    next_tok = seg_seq[first_pipe + 2].item()
                    if next_tok == 95 or first_tok == 59:  # ord('_') = 95, ord(';') = 59
                        # the syntax is [tag char]_ or a missing segment would start with ;
                        is_tag = False
                    else:
                        # tag starts before the pipe.
                        is_tag = True
                elif len(seg_seq) > first_pipe + 1:
                    next_tok = seg_seq[first_pipe + 1].item()
                    # Much weaker signal since these are sometimes degenerate bases, but at least the length
                    #  check later will prevent overmasking.
                    if next_tok in {68, 100, 82, 114, 59}:  # D,d,R,r for domain or realm (viruses) or missing ;
                        is_tag = False
                    else:
                        is_tag = True
                elif first_pipe >= max_tag_len or region_all_valid_or_control(seg_seq[:first_pipe]):
                    is_tag = False
                else:
                    is_tag = True
            else:
                # The sequence ends with a pipe, so just check everything before the pipe and return the seg mask
                assert first_pipe == seg_len - 1
                # The sequence ends with a pipe, so just check everything before the pipe.
                if first_pipe >= max_tag_len or region_all_valid_or_control(seg_seq[:first_pipe]):
                    return seg_mask  # Pipe pos has already been masked
                else:
                    seg_mask[:first_pipe] = 0
                    return seg_mask
            start = 0
            for end in pipe_pos:
                seg_len = end - start
                if is_tag and seg_len < max_tag_len:  # Prefer faulty tag mask logic over masking entire seqs
                    seg_mask[start:end] = 0
                elif is_tag and seg_len >= max_tag_len:
                    # We were wrong about this segment being a tag, so change the current state to what
                    #  it should have been
                    is_tag = False
                else:
                    pass
                # Done with this segment, now advance the state machine for the next segment
                is_tag = not is_tag  # Flip the state machine for the next state
                start = end + 1  # position after the pipe
            # Process the last segment after the last pipe.
            seg_len = len(seg_mask) - start
            if is_tag and seg_len < max_tag_len:  # Prefer faulty tag mask logic over overmasking
                seg_mask[start:] = 0
            return seg_mask

        # Process each row by splitting on EOD tokens.
        for b in range(batch_size):
            row = tokenized_sequence[b]
            # Get indices of EOD tokens.
            eod_positions = (row == eod_token_id).nonzero(as_tuple=True)[0].cpu().tolist()
            start_idx = 0
            for pos in eod_positions:
                if pos > start_idx:
                    seg = row[start_idx:pos]
                    seg_mask = process_segment(seg)
                    out_mask[b, start_idx:pos] = seg_mask
                # Leave the EOD token itself unmasked.
                start_idx = pos + 1
            # Process any remaining tokens after the last EOD.
            if start_idx < seq_len:
                seg = row[start_idx:]
                seg_mask = process_segment(seg)
                out_mask[b, start_idx:] = seg_mask

        # Just to make sure we do not allow any non-DNA tokens to be unmasked, even if something went wrong with our
        #  mask logic.
        out_mask[~torch.isin(tokenized_sequence, valid_dna_or_control_tensor)] = 0
        # Finally, force every EOD token to be unmasked. User decides outside of this function if they want EOD mask.
        out_mask[tokenized_sequence == eod_token_id] = 1

        if not batched:
            out_mask = out_mask.squeeze(0)
        return out_mask


class Evo2DatasetPadEodLossMask(Evo2Dataset):
    """Dataset for training Evo2 with pad and eod loss mask (more standard approach than the Evo2 paper)."""

    TO_UPPER_TOKENS: bool = True
    RESET_PAD_EOD_MASK: bool = False
