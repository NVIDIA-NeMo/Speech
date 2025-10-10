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


import random
import timeit


import pytest
import torch
from megatron.core.datasets.utils import Split

from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset, Evo2DatasetPadEodLossMask

"""
The tag token is constructed as follows: So note that one way to know you have a tag is if you look at the first
token after the pipe and it is a 'd' character. Make sure tests are consistent with this simplification.
    @staticmethod
    def _construct_taxonomy_token(
        lineage: Evo2TaxonomyLineage, dropout: float = 0.0, seed: Optional[int] = None
    ) -> Optional[str]:
        '''Construct a special Taxonomy token for natural language prompting of DNA generation models.

        Args:
            lineage (Evo2TaxonomyLineage): The taxonomy lineage information.
            dropout (float): The probability of dropping out segments of the lineage. Defaults to 0.0.
            seed (Optional[int]): The seed for the random number generator. Defaults to None.

        Returns:
            Optional[str]: The constructed taxonomy token or None if lineage is None.
        '''
        # If dropout > 0, randomly drop out segments of the lineage for training on incomplete lineages.
        with Evo2Preprocessor.preprocessing_context_manager(seed if seed is not None else None):
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
"""

MAX_TAG_LEN = 2048


@pytest.fixture
def tag_tokens():
    """Standard tokens for phylogenetic tag tests, defined in Evo2_DataseT:

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0
    """
    return {
        "terminal": 124,  # |
        "other_chars": {95, 59, 32},  # _, ;, space
        "eod": 0,  # end of document token
    }


def test_mask_phylogenetic_tags_with_eod(tag_tokens):
    """
    Tests a sequence where an EOD splits two partial tags.

    Example sequence (ASCII):
      65       124   100    0     124   65
      'A'      '|'   'd'   EOD   '|'   'A'

    - Segment 1: "A|d" => keep 'A' (DNA), mask '|' and 'd'
    - EOD => masked
    - Segment 2: "|A" => mask '|', keep 'A' (DNA)

    Expected masking: [1, 0, 0, 1, 0, 1]
    """
    sequence = torch.tensor([65, 124, 100, 0, 124, 65])  # "A|d" + EOD + "|A"

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # '|'
        other_tag_chars=tag_tokens["other_chars"],  # { '_',';',' ' }
        eod_token_id=tag_tokens["eod"],  # 0
        max_tag_len=MAX_TAG_LEN,
    )

    expected_mask = torch.tensor([1, 0, 0, 1, 0, 1])
    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_middle(tag_tokens):
    """Tests masking a phylogenetic tag that appears in the middle of a DNA sequence.

    The sequence contains:
    1. Normal DNA (ATG)
    2. A phylo tag (|d_|)
    3. More DNA (TCGA)

    Expected behavior: The DNA should be unmasked (1s) while everything between
    and including the pipe characters should be masked (0s), as it's a valid phylo tag.
    """
    # use a viral style phylo tag in upper case this time.
    sequence_chrs = "ATG|R__tag|TCGA"
    expected_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # |
        other_tag_chars=tag_tokens["other_chars"],  # _, ;, space
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_middle_new_format(tag_tokens):
    """Tests masking a phylogenetic tag that appears in the middle of a DNA sequence.

    The sequence contains:
    1. Normal DNA (ATG)
    2. A novel format phylo tag (|X_|)
    3. More DNA (TCGA)

    Expected behavior: The DNA should be unmasked (1s) while everything between
    and including the pipe characters should be masked (0s), as it's a valid phylo tag.
    """
    # use a viral style phylo tag in upper case this time.
    sequence_chrs = "ATG|X__tag|TCGA"
    expected_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # |
        other_tag_chars=tag_tokens["other_chars"],  # _, ;, space
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_middle_missing_top_level_annotation(tag_tokens):
    """Tests masking a phylogenetic tag that appears in the middle of a DNA sequence.

    The sequence contains:
    1. Normal DNA (ATG)
    2. A novel format phylo tag (|;;;|)
    3. More DNA (TCGA)

    Expected behavior: The DNA should be unmasked (1s) while everything between
    and including the pipe characters should be masked (0s), as it's a valid phylo tag.
    """
    # use a viral style phylo tag in upper case this time.
    sequence_chrs = "ATG|;;;;;|TCGA"
    expected_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # |
        other_tag_chars=tag_tokens["other_chars"],  # _, ;, space
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_start(tag_tokens):
    """Tests handling a sequence that starts with a partial phylogenetic tag.

    The sequence starts with characters that would be inside a phylo tag,
    followed by a closing pipe and DNA. Since we want to prevent the model from
    learning non-DNA outputs, we mask all potential tag characters even without
    complete tag delimiters.

    Sequence: "tag;_|ATG" (starting mid-tag)
    Expected: All tag characters and delimiters masked, only DNA unmasked
    """
    sequence = torch.tensor(
        [
            116,
            97,
            103,
            59,
            95,  # tag;_
            124,  # |
            65,
            84,
            71,  # ATG
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    expected_mask = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,  # partial tag start masked
            0,  # closing pipe masked
            1,
            1,
            1,  # DNA unmasked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_end(tag_tokens):
    """Tests handling a sequence that ends with a partial phylogenetic tag.

    The sequence contains DNA followed by an opening pipe and tag characters,
    but no closing pipe. Per requirements, we aggressively mask any potential
    tag characters to ensure the model only learns DNA bases {A,C,G,T}.

    Sequence: "ATG|info_" (ending mid-tag)
    Expected: DNA unmasked, all tag-related characters masked
    """
    sequence_chrs = "ATG|r_"
    expected_mask = torch.tensor([1, 1, 1, 0, 0, 0])
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_end2(tag_tokens):
    """Tests handling a sequence that ends with a partial phylogenetic tag.

    The sequence contains DNA followed by an opening pipe and tag characters,
    but no closing pipe. Per requirements, we aggressively mask any potential
    tag characters to ensure the model only learns DNA bases {A,C,G,T}.

    Sequence: "ATG|info_" (ending mid-tag)
    Expected: DNA unmasked, all tag-related characters masked
    """
    sequence_chrs = "ATG|D"
    expected_mask = torch.tensor([1, 1, 1, 0, 0])
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_end3(tag_tokens):
    """Tests handling a sequence that ends with a partial phylogenetic tag.

    The sequence contains DNA followed by an opening pipe and tag characters,
    but no closing pipe. Per requirements, we aggressively mask any potential
    tag characters to ensure the model only learns DNA bases {A,C,G,T}.

    Sequence: "ATG|info_" (ending mid-tag)
    Expected: DNA unmasked, all tag-related characters masked
    """
    sequence_chrs = "ATG|;"
    expected_mask = torch.tensor([1, 1, 1, 0, 0])
    sequence = torch.tensor([ord(c) for c in sequence_chrs])

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    assert torch.equal(mask, expected_mask)


def test_standalone_tag(tag_tokens):
    """Tests masking of a single complete tag with no surrounding sequence.

    Tests that a standalone tag (|tag_|) is fully masked since it contains
    non-DNA characters. This ensures the model only learns to output
    {A,C,G,T} tokens.

    Sequence: |d__tag|
    Expected: All tokens masked (all zeros)
    """
    sequence_chrs = "|d__tag|"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])  # |d__tag|
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    expected = torch.tensor([0] * len(sequence_chrs))  # All masked
    assert torch.equal(mask, expected)


def test_sequence_starting_with_tag(tag_tokens):
    """Tests sequence that begins with a complete tag followed by DNA.

    Verifies that when a sequence starts with a complete tag followed by
    DNA bases, the tag portion is masked while the DNA portion remains
    unmasked.

    Sequence: |D__|ATG
    Expected: Tag masked (zeros), DNA unmasked (ones)
    """
    sequence_chrs = "|D__|ATG"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])
    expected = torch.tensor([0] * len('|D__|') + [1, 1, 1])  # Tag masked, DNA unmasked
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    assert torch.equal(mask, expected)


def test_sequence_ending_with_tag(tag_tokens):
    """Tests sequence that ends with a complete tag.

    Verifies that when a sequence ends with a complete tag, the DNA portion
    remains unmasked while the entire tag portion is masked.

    Sequence: ATG|tag_|
    Expected: DNA unmasked (ones), tag masked (zeros)
    """
    sequence_chrs = "ATG|D__|"
    expected = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])  # DNA unmasked, tag masked
    sequence = torch.tensor([ord(c) for c in sequence_chrs])
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    assert torch.equal(mask, expected)


def test_mask_multiple_tags(tag_tokens):
    """Tests handling multiple phylogenetic tags in sequence, demonstrating state transitions.

    This tests how the masking switches states between phylo and non-phylo regions:
    1. Starts in non-phylo state with DNA
    2. Switches to phylo state at first pipe (with tag chars)
    3. Switches back to non-phylo at closing pipe
    4. Pattern repeats for second tag

    Sequence: "ATG|tag_1|CG|tag_2|AT"
    Expected: Only DNA sequences should remain unmasked
    """
    sequence_chrs = "ATG|d__|CG|r__|AT"
    sequence = torch.tensor([ord(c) for c in sequence_chrs])
    expected_mask = torch.tensor(
        [1] * 3 + [0] * len('|d__|') + [1] * 2 + [0] * len('|r__|') + [1] * 2
    )  # DNA unmasked, tag masked

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    assert torch.equal(mask, expected_mask)


def test_mask_dna_after_pipe(tag_tokens):
    """Tests the scenario where we have a pipe followed by DNA sequence.

    This tests the edge case of a pipe character appearing at the start of a sequence.
    Even if DNA follows, we mask the pipe character to prevent the model from
    learning to output non-DNA tokens.

    Sequence: "|ATG" (pipe followed by DNA)
    Expected: Pipe masked, DNA unmasked
    """
    sequence = torch.tensor(
        [
            124,  # |
            65,
            84,
            71,  # ATG
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    expected_mask = torch.tensor([0, 1, 1, 1])  # Pipe masked, DNA unmasked
    assert torch.equal(mask, expected_mask)


def test_ambiguous_dna_char_followed_by_tag_start(tag_tokens):
    """Tests handling of an ambiguous DNA character followed by a tag start.

    When we see a character that could be either DNA or the end of a truncated tag
    followed by a pipe, we should mask both for safety since we can't disambiguate
    whether the character was part of a tag.

    Sequence: "t|AAAT" (t could be DNA or end of tag)
    Expected: First t and pipe masked (0), AAAT unmasked (1)
    """
    sequence = torch.tensor(
        [
            116,  # t (ambiguous - could be DNA or end of tag)
            124,  # |
            65,  # A
            65,  # A
            65,  # A
            84,  # T
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    expected_mask = torch.tensor([0, 0, 1, 1, 1, 1])  # Ambiguous t and pipe masked, DNA unmasked
    assert torch.equal(mask, expected_mask)


def test_dna_followed_by_unambiguous_tag_start(tag_tokens):
    """Tests handling of DNA sequence followed by clear tag start.

    When we see DNA followed by |d, it's unambiguous - the d clearly indicates
    the start of a phylogenetic tag (d__), so we can safely unmask the DNA and
    mask the tag portion.

    Sequence: "AAAT|d" (AAAT is DNA, |d starts tag)
    Expected: AAAT unmasked (1), |d masked (0)
    """
    sequence = torch.tensor(
        [
            65,  # A
            65,  # A
            65,  # A
            84,  # T
            124,  # |
            100,  # d (clearly starts d__ tag)
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    expected_mask = torch.tensor([1, 1, 1, 1, 0, 0])  # DNA unmasked, tag start masked
    assert torch.equal(mask, expected_mask)


def test_double_partial_tags_with_dna_middle(tag_tokens):
    """Tests a sequence that has partial tags at both ends with DNA in the middle.

    Tests the specific case where a sequence slice cuts through phylogenetic tags
    on both ends, with valid DNA sequence in the middle. The behavior we want is:
    1. The partial tag at the start should be masked
    2. The DNA in the middle should be unmasked
    3. The partial tag at the end should be masked

    Sequence: "cacata|acagataaaataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)
    """
    sequence = torch.tensor(
        [
            99,
            97,
            99,
            97,
            116,
            97,  # cacata
            124,  # |
            97,
            99,
            97,
            103,
            97,
            116,
            97,
            97,
            97,
            97,
            116,
            97,  # acagataaaata
            84,
            65,
            67,
            65,
            71,
            71,
            71,
            65,
            65,
            84,
            65,  # TACAGGGAATA
            124,  # |
            100,
            95,
            95,  # d__
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )

    expected_mask = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,  # partial start tag masked
            0,  # pipe masked
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # middle DNA unmasked
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # middle DNA unmasked
            0,  # pipe masked
            0,
            0,
            0,  # partial end tag masked
        ]
    )

    assert torch.equal(mask, expected_mask)


def test_packed_partial_tag_subsequence_predna(tag_tokens):
    """
    Sequence: "GAATA[EOD]cacata|acagataaaataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "GAATA0cacata|acagataaaataTACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("GAATA0") * [1] + [0] * len("cacata|") + len("acagataaaataTACAGGGAATA") * [1] + [0] * len("|d__"),
        dtype=torch.int32,
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_pretag(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaataTACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1] + [0] + [1] * len("0acagataaaataTACAGGGAATA") + len("|d__") * [0], dtype=torch.int32
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_predna_middletag(tag_tokens):
    """
    Sequence: "GAATA[EOD]cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "GAATA0cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("GAATA0") * [1]
        + len("cacata|") * [0]
        + [1] * len("acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_pretag_middletag(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1]
        + [0]  # masked pipe.
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_pretag_middletag_bs2(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1]
        + [0]
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )
    expected_mask = torch.stack([expected_mask, expected_mask])
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_pretag_middletag_bs3(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__somet"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1]
        + [0]
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__somet") * [0],
        dtype=torch.int32,
    )

    sequence_alpha2 = "GAATA0cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence2 = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha2], dtype=torch.int32)
    expected_mask2 = torch.tensor(
        len("GAATA0") * [1]
        + len("cacata|") * [0]
        + [1] * len("acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )

    expected_mask = torch.stack([expected_mask, expected_mask, expected_mask2])

    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence, sequence2]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_partial_tag_subsequence_pretag_middletag_bs3_cuda(tag_tokens):
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__somet"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1]
        + [0]
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__somet") * [0],
        dtype=torch.int32,
    )

    sequence_alpha2 = "GAATA0cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence2 = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha2], dtype=torch.int32)
    expected_mask2 = torch.tensor(
        len("GAATA0") * [1]
        + len("cacata|") * [0]
        + [1] * len("acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )

    expected_mask = torch.stack([expected_mask, expected_mask, expected_mask2])

    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence, sequence2]).cuda(),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask.cpu(), expected_mask)


def test_multiple_packed_tags(tag_tokens):
    """
    Tests a sequence with multiple packed tags.
    """
    sequence_alpha = "|d__tag;|0|d__tag;|0|d__somet"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("|d__tag;|") * [0] + len("0") * [1] + len("|d__tag;|") * [0] + len("0") * [1] + len("|d__somet") * [0],
        dtype=torch.int32,
    )
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_multiple_eods(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "ACGT0tacg0"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len(sequence_alpha) * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_multiple_eods_prefix_no_suffix(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "0ACGT0tacg0aa"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len(sequence_alpha) * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_no_eods_with_batch(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "ACATAGATTT"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len(sequence_alpha) * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, torch.stack([expected_mask, expected_mask]))


def test_no_eods_one_tag_with_batch_bs2(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "ACAT|d__tag;|AGATTT"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len("ACAT") * [1] + len("|d__tag;|") * [0] + len("AGATTT") * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, torch.stack([expected_mask, expected_mask]))


def test_packed_partial_tag_subsequence_predna_with_control_and_degenerate_base(tag_tokens):
    """
    Sequence: "GAATA[EOD]cacata|acagataaa@ataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "GAWTA0cacata|acagaraaaa@taTACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("GAWTA0") * [1] + [0] * len("cacata|") + len("acagataaaa@taTACAGGGAATA") * [1] + [0] * len("|d__"),
        dtype=torch.int32,
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_predna_with_control2(tag_tokens):
    """
    Sequence: "GAATA[EOD]cacata|acagataaa@ataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "GA#ATA0cacata|acagataaaa@taTACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("GA#ATA0") * [1] + [0] * len("cacata|") + len("acagataaaa@taTACAGGGAATA") * [1] + [0] * len("|d__"),
        dtype=torch.int32,
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
        max_tag_len=MAX_TAG_LEN,
    )
    torch.testing.assert_close(mask, expected_mask)


def _construct_taxonomy_token(dropout: float = 0.0) -> str:
    """Construct a special Taxonomy token for natural language prompting of DNA generation models.

    Args:
        dropout (float): The probability of dropping out segments of the lineage. Defaults to 0.0.

    Returns:
        Optional[str]: The constructed taxonomy token or None if lineage is None.
    """
    # If dropout > 0, randomly drop out segments of the lineage for training on incomplete lineages.
    return "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
        "somedomain" if random.random() >= dropout else None,
        "somephylum" if random.random() >= dropout else None,
        "someclass" if random.random() >= dropout else None,
        "someorder" if random.random() >= dropout else None,
        "somefamily" if random.random() >= dropout else None,
        "lineage.genus" if random.random() >= dropout else None,
        "lineage.speciescactaca" if random.random() >= dropout else None,
    )


def mask_phylogenetic_tags_old(tokenized_sequence, terminal_tag_char, other_tag_chars, eod_token_id):
    """
    Optimized version to create a phylonetic tag mask for batched tokenized sequences with correct handling of partial tags.
    Args:
    - tokenized_sequence (torch.Tensor): A batched tensor of shape (batch_size, seq_length).
    - terminal_tag_char (int): The token ID representing the start and end of a phylogenetic tag ('|').
    - other_tag_chars (set of int): A set of token IDs that are uniquely part of the tag ('_', ';', etc.).
    - eod_token_id (int): The token ID representing the end-of-document (EOD).
    Returns:
    - mask_vector (torch.Tensor): A batched mask of the same shape as tokenized_sequence where
      1 represents non-tag tokens and 0 represents tokens within the masked region.
    """
    device = tokenized_sequence.device
    batch_size, seq_len = tokenized_sequence.shape
    mask_vector = torch.ones_like(tokenized_sequence, dtype=torch.int, device=device)

    # To address when unbalanced tags are present
    terms = torch.tensor([0, seq_len - 1], device=device)
    other_tags = torch.tensor(list(other_tag_chars), device=device)
    for batch_idx in range(batch_size):
        tag_term_locs = torch.where(tokenized_sequence[batch_idx] == terminal_tag_char)[0]
        tag_end_locs = torch.where(tokenized_sequence[batch_idx] == eod_token_id)[0]

        merged_tags = torch.cat((terms, tag_term_locs, tag_end_locs)).sort()[0]
        merged_tags = merged_tags.unique()

        start = 0  # First and last locations are always added
        for end in merged_tags[1:]:
            if torch.isin(tokenized_sequence[batch_idx][start:end], other_tags).sum() > 0:
                # end token is not part of the tag
                if eod_token_id == tokenized_sequence[batch_idx][end]:
                    end = end - 1
                if eod_token_id == tokenized_sequence[batch_idx][start]:
                    start = start + 1

                mask_vector[batch_idx][start : (end + 1)] = 0
            start = end
    return mask_vector


def benchmark_phylo_tag_masking(num_iterations: int = 1000) -> tuple[float, float]:
    """Benchmark the performance of phylogenetic tag masking functions.

    Args
        num_iterations: Number of iterations to run for timing
    """
    tax_token = _construct_taxonomy_token(dropout=0.0)
    sequence_alpha = (
        tax_token[2:]
        + "".join(random.choice("ACGTacgt") for _ in range(5000))
        + tax_token[:-25]
        + "0"
        + tax_token[36:]
        + "".join(random.choice("ACGTacgt") for _ in range(5000))
    )
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32, device="cpu")

    # Time the new implementation
    new_time1 = timeit.timeit(
        lambda: Evo2Dataset.mask_phylogenetic_tags(
            sequence.unsqueeze(0), 124, {95, 59, 32}, 0, max_tag_len=MAX_TAG_LEN
        ),
        number=num_iterations,
    )

    # Time the old implementation
    old_time1 = timeit.timeit(
        lambda: mask_phylogenetic_tags_old(sequence.unsqueeze(0), 124, {95, 59, 32}, 0),
        number=num_iterations,
    )

    # Time the new implementation
    new_time2 = timeit.timeit(
        lambda: Evo2Dataset.mask_phylogenetic_tags(
            sequence.unsqueeze(0), 124, {95, 59, 32}, 0, max_tag_len=MAX_TAG_LEN
        ),
        number=num_iterations,
    )

    # Time the old implementation
    old_time2 = timeit.timeit(
        lambda: mask_phylogenetic_tags_old(sequence.unsqueeze(0), 124, {95, 59, 32}, 0),
        number=num_iterations,
    )
    new_time = (new_time1 + new_time2) / 2
    old_time = (old_time1 + old_time2) / 2
    return old_time, new_time


if __name__ == "__main__":
    num_iterations = 2000
    old_time, new_time = benchmark_phylo_tag_masking(num_iterations=num_iterations)
    print(f"Old implementation average time: {old_time/num_iterations:.6f} seconds")
    print(f"New implementation average time: {new_time/num_iterations:.6f} seconds")
    print(f"Speed improvement: {(old_time/new_time - 1)*100:.2f}%")


def test_evo2_dataset_getitem(monkeypatch):
    """Test Evo2Dataset.__getitem__ method."""
    import numpy as np
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

    tokenizer = get_nmt_tokenizer("byte-level")
    eod_token_id = tokenizer.eod
    # labels are all case, tokens are converted to upper case.
    input_string = f"a  @  t  |  d  _  _  t  {eod_token_id}  #  a  t".replace(" ", "")
    starting_loss_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
    expected_loss_mask = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
    input_tokens = [
        ord(t) if t != str(eod_token_id) else eod_token_id for t in input_string
    ]  # starts out both lower/upper
    input_labels = [ord(t) if t != str(eod_token_id) else eod_token_id for t in input_string]

    class MockIndexedDataset:
        def __init__(self):
            self.sequence_lengths = np.ones(100, dtype=np.int32) * 10
            self.path_prefix = "/mock/path"

        def get(self, idx, offset=0, length=None):
            return np.ones(10, dtype=np.int64)

    class MockConfig:
        def __init__(self):
            # GPTDatasetConfig specific
            self.reset_position_ids = False
            self.reset_attention_mask = False
            self.eod_mask_loss = False
            self.create_attention_mask = True
            self.drop_last_partial_validation_sequence = True
            self.add_extra_token_to_sequence = True
            self.s3_cache_path = None

            # BlendedMegatronDatasetConfig
            self.random_seed = 42
            self.sequence_length = len(input_tokens)
            self.blend = None
            self.blend_per_split = None
            self.split = "1,1,1"
            self.split_matrix = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)]
            self.num_dataset_builder_threads = 1
            self.path_to_cache = None
            self.mmap_bin_files = True
            self.mock = True
            self.tokenizer = tokenizer

    mock_indexed_dataset = MockIndexedDataset()

    # Now when Evo2Dataset is instantiated, it will inherit from MockGPTDataset
    # Create a real instance with minimal arguments
    dataset = Evo2Dataset(
        indexed_dataset=mock_indexed_dataset,
        dataset_path="/mock/path",
        indexed_indices=np.arange(5, dtype=np.int32),
        num_samples=5,
        index_split=Split.train,
        config=MockConfig(),
    )
    dataset.RESET_PAD_EOD_MASK = False
    dataset.TO_UPPER_TOKENS = True
    parent_batch = {
        "loss_mask": starting_loss_mask,  # Will be modified by Evo2Dataset
        "labels": torch.tensor(input_labels),  # A@T|d_T#AT
        "tokens": torch.tensor(input_tokens),  # a@t|d_t#at
        "attention_mask": torch.ones(len(input_tokens), len(input_tokens)),  # Add attention mask
        "position_ids": torch.arange(len(input_tokens)),  # Add position ids
    }
    # monkey patch the _get_gpt_batch method in this dataset so that we use our parent_batch as the starting point.
    dataset._get_gpt_batch = lambda x: parent_batch

    result = dataset[0]

    torch.testing.assert_close(result["loss_mask"], expected_loss_mask.to(torch.int32))
