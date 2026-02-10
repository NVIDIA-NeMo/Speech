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
import torch

from nemo.collections.common.tokenizers import AutoTokenizer


def tokens_to_str(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    tokenizer: AutoTokenizer,
    pad_id: int,
    eval_text_turn_taking: bool = False,
    show_eot_timestamps: bool = False,
) -> list[str]:
    """
    Convert token IDs to text strings, filtering out special tokens.

    Args:
        tokens: Token IDs tensor (B, T)
        lengths: Length of each sequence (B,)
        tokenizer: Tokenizer for decoding
        pad_id: Pad token ID to filter out
        eval_text_turn_taking: If True, insert timestamps at bos/eos positions
        show_eot_timestamps: If True, also insert timestamps at end-of-text (first pad after BOS)

    Returns:
        List of decoded text strings
    """
    ans = []

    # Helper function to filter special tokens from token IDs
    # This filtering is applied regardless of eval_text_turn_taking mode
    def filter_special_tokens(token_ids):
        # Filter out pad
        token_ids = token_ids[token_ids != pad_id]
        # Filter out agent bos/eos
        token_ids = token_ids[token_ids != tokenizer.bos]
        token_ids = token_ids[token_ids != tokenizer.eos]
        return token_ids

    for hyp_ids, hyp_len in zip(tokens.cpu(), lengths.cpu()):
        if eval_text_turn_taking:
            # Insert timestamps to the text
            agent_bos_positions = (hyp_ids == tokenizer.bos).nonzero(as_tuple=True)[0].tolist()
            agent_eos_positions = (hyp_ids == tokenizer.eos).nonzero(as_tuple=True)[0].tolist()

            # Detect end-of-text (EOT) positions: find first pad after each BOS
            agent_eot_positions = []
            if show_eot_timestamps:
                for bos_pos in agent_bos_positions:
                    # Find the corresponding EOS position for this BOS
                    matching_eos = [eos for eos in agent_eos_positions if eos > bos_pos]
                    end_search_pos = matching_eos[0] if matching_eos else len(hyp_ids)

                    # Search for first pad token after BOS
                    for pos in range(bos_pos + 1, end_search_pos):
                        if hyp_ids[pos] == pad_id:
                            agent_eot_positions.append(pos)
                            break

            # Combine and sort all positions with their types
            all_positions = []
            for pos in agent_bos_positions:
                all_positions.append((pos, 'bos'))
            for pos in agent_eos_positions:
                all_positions.append((pos, 'eos'))
            for pos in agent_eot_positions:
                all_positions.append((pos, 'eot'))

            # Sort by position
            all_positions.sort(key=lambda x: x[0])

            start_idx = 0
            out_str = []
            for pos, pos_type in all_positions:
                text_ids = hyp_ids[start_idx:pos]
                # Filter out special tokens before converting to text
                text_ids = filter_special_tokens(text_ids)
                start_idx = pos
                timestamp = round(float(pos) * 0.08, 3)
                out_str.append(tokenizer.ids_to_text(text_ids))
                if pos_type == 'bos':
                    out_str.append(f"<|{timestamp}|>")
                elif pos_type == 'eos':
                    out_str.append(f"<${timestamp}$>")
                else:  # eot
                    out_str.append(f"<{timestamp}>")
            # Filter the remaining tokens after the last position
            remaining_ids = filter_special_tokens(hyp_ids[start_idx:])
            out_str.append(tokenizer.ids_to_text(remaining_ids))
            ans.append(" ".join(out_str))
        else:
            # For non-turn-taking mode: filter out ALL special tokens, return only pure text
            hyp_ids = hyp_ids[:hyp_len]
            hyp_ids = filter_special_tokens(hyp_ids)
            ans.append(tokenizer.ids_to_text(hyp_ids))
    return ans
