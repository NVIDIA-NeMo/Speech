# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes.module import NeuralModule
from nemo.utils.enum import PrettyStrEnum


class LocalTransformerType(PrettyStrEnum):
    """
    Enum for the type of local transformer to use in the MagpieTTS model.
    These strings are the values allowed in the YAML config file.
    """

    NO_LT = "none"
    AR = "autoregressive"
    MASKGIT = "maskgit"


class EOSDetectionMethod(PrettyStrEnum):
    """
    Enum for the EOS detection method to use in the MagpieTTS model.
    These strings are the values allowed in the YAML config file.
    """

    ARGMAX_ANY = "argmax_any"
    ARGMAX_OR_MULTINOMIAL_ANY = "argmax_or_multinomial_any"
    ARGMAX_ALL = "argmax_all"
    ARGMAX_OR_MULTINOMIAL_ALL = "argmax_or_multinomial_all"
    ARGMAX_ZERO_CB = "argmax_zero_cb"
    ARGMAX_OR_MULTINOMIAL_ZERO_CB = "argmax_or_multinomial_zero_cb"

    @staticmethod
    def detection_type(detection_method: EOSDetectionMethod):
        if detection_method in [EOSDetectionMethod.ARGMAX_ANY, EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY]:
            return "any"
        elif detection_method in [EOSDetectionMethod.ARGMAX_ALL, EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ALL]:
            return "all"
        elif detection_method in [EOSDetectionMethod.ARGMAX_ZERO_CB, EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ZERO_CB]:
            return "zero_cb"
        else:
            raise ValueError(f"Invalid EOS detection method: {detection_method}")

    @staticmethod
    def sampling_type(detection_method: EOSDetectionMethod):
        if detection_method in [
            EOSDetectionMethod.ARGMAX_ANY,
            EOSDetectionMethod.ARGMAX_ALL,
            EOSDetectionMethod.ARGMAX_ZERO_CB,
        ]:
            return "argmax"
        elif detection_method in [
            EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY,
            EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ALL,
            EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ZERO_CB,
        ]:
            return "argmax_or_multinomial"
        else:
            raise ValueError(f"Invalid EOS detection method: {detection_method}")


class SpecialAudioToken(Enum):
    """
    Enum for the special tokens to use in the MagpieTTS model.
    The special tokens are appended at the end of the codebook after the actual audio codec tokens.
    The actual embedding table index is the value below plus the number of codec tokens - do not use the Enum directly.
    """

    AUDIO_BOS = 0
    AUDIO_EOS = 1
    AUDIO_CONTEXT_BOS = 2
    AUDIO_CONTEXT_EOS = 3
    MASK_TOKEN = 4
    # Reserve these values so that if we need to add more special tokens in the future the codebook size will remain the same
    RESERVED_1 = 5
    RESERVED_2 = 6
    RESERVED_3 = 7

    @staticmethod
    def get_index(token: SpecialAudioToken, base_codebook_size: int):
        """
        Returns the index of the special token in the embedding table.
        """
        return base_codebook_size + token.value

    @staticmethod
    def get_forbidden_tokens(base_codebook_size: int, forbid_audio_eos: bool = False) -> list[int]:
        """
        Returns a list of token indices that should not be sampled or returned to user.
        Args:
            base_codebook_size (int): The size of the codec codebook (which is the first part of the embedding table).
            forbid_audio_eos (bool): Whether AUDIO_EOS should be forbidden. Default: False (i.e. allowed).
        """
        all_special_tokens = list(SpecialAudioToken)
        if not forbid_audio_eos:
            all_special_tokens.remove(SpecialAudioToken.AUDIO_EOS)
        return [SpecialAudioToken.get_index(token, base_codebook_size) for token in all_special_tokens]


def cosine_schedule(x: torch.Tensor):
    """
    Maps input values from [0, 1] to [1, 0] using the first quadrant of the cosine function.
    Used for MaskGit mask scheduling.
    """
    return torch.cos(x * (torch.pi / 2))


def build_vocabs(subword_vocab: dict, subword_padding_idx: int, special_vocab: dict = None) -> tuple[dict, dict]:
    """
    Builds the character vocabulary and the mapping from subword ids to character ids.
    Args:
        subword_vocab (dict): A dictionary of subword vocab items. Eg.
            tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
            subword_vocab = tokenizer.vocab
        subword_padding_idx (int): The padding index for the subword vocabulary.
        special_vocab (dict): items of special token dictionary (usually BOS, EOS)
            eg. special_vocab = {'<BOS>': 0, '<EOS>': 1}
    Returns:
        subword_id_to_char_ids: A dictionary mapping subword ids to character ids.
        char_vocab: A dictionary mapping character ids to their corresponding characters.
    """
    org_char_vocab = {subword: subword_id for subword, subword_id in subword_vocab.items() if len(subword) == 1}

    # Add special tokens directly to char vocab
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in org_char_vocab:
                raise ValueError(f"Special token {special_token} already exists in the character vocabulary.")
            org_char_vocab[special_token] = special_token_id

    sorted_char_vocab = dict(sorted(org_char_vocab.items(), key=lambda x: x[1]))
    char_vocab = {k: i for i, (k, _) in enumerate(sorted_char_vocab.items())}
    assert sorted(char_vocab.values()) == list(range(len(char_vocab)))
    subword_id_to_char_ids = {
        subword_id: tuple(char_vocab[char] for char in subword) for subword, subword_id in subword_vocab.items()
    }

    # Creating mapping from subword ids of special tokens to their char ids
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in subword_id_to_char_ids:
                raise ValueError(f"Special token {special_token} already exists in the subword id Vocabulary.")
            subword_id_to_char_ids[special_token_id] = (char_vocab[special_token],)

    assert max(subword_id_to_char_ids) == len(subword_id_to_char_ids) - 1

    # Always add padding token to the end of the vocab (this is the convention used in the original code)
    subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)

    return subword_id_to_char_ids, char_vocab


class CharAwareSubwordEncoder(NeuralModule):
    """
    Char-aware subword encoder for the MagpieTTS model.
    This module takes subword ids as input, maps them to character ids, and then applies a transformer encoder to the character embeddings.
    The output is a tensor of shape (batch_size, max_subword_length, d_embed).
    """

    def __init__(self, d_embed: int, llm_tokenizer_vocab: dict, subword_padding_idx: int, special_vocab: dict = None):
        """
        Args:
            d_embed (int): The dimension of the embedding.
            llm_tokenizer_vocab (dict): A dictionary of subword vocab items. Eg.
                tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
                llm_tokenizer_vocab = tokenizer.vocab
            subword_padding_idx (int): The padding index for the subword vocabulary.
            special_vocab (dict): items of special token dictionary (usually BOS, EOS)
                eg. special_vocab = {'<BOS>': 30001, '<EOS>': 30002}
        """
        super().__init__()
        self.subword_id_to_char_ids, self.char_vocab = build_vocabs(
            llm_tokenizer_vocab, subword_padding_idx, special_vocab
        )
        self.embed_tokens = torch.nn.Embedding(self.vocab_size + 1, d_embed, padding_idx=self.vocab_size)
        self.encoder = transformer_2501.Transformer(
            n_layers=1,
            d_model=d_embed,
            d_ffn=d_embed * 4,
            sa_n_heads=8,
            kernel_size=1,
            max_length_causal_mask=256,
            use_learnable_pos_emb=True,
        )

    @property
    def vocab_size(self):
        return len(self.char_vocab)

    def prepare_inputs(self, subword_ids: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        device = subword_ids.device

        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        char_id_list = [list(self.subword_id_to_char_ids[x]) for x in subword_id_list]

        char_lengths = torch.tensor([len(x) for x in char_id_list], dtype=torch.long, device=device)
        batch_size = char_lengths.size(0)

        char_ids = torch.full((batch_size, int(char_lengths.max().item())), self.vocab_size, dtype=torch.long)
        for i in range(batch_size):
            char_ids[i, : char_lengths[i]] = torch.tensor(char_id_list[i])
        char_ids = char_ids.to(device=device)
        return char_ids, char_lengths

    def forward(self, subword_ids: Tensor, subword_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            subword_ids (Tensor): A tensor of shape (batch_size, max_subword_length) containing the subword ids.
            subword_mask (Tensor | None): A tensor of shape (batch_size, max_subword_length) containing the mask for the subword ids.
                If None, a mask of ones will be used.
        Returns:
            Tensor: A tensor of shape (batch_size, max_subword_length, d_embed) containing the subword embeddings.
        """
        device = subword_ids.device
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids).bool()
        else:
            subword_mask = subword_mask.bool()

        if subword_mask.ndim == 3:
            subword_mask = subword_mask.squeeze(-1)

        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)
        char_mask = get_mask_from_lengths(char_lengths)
        char_emb = self.embed_tokens(char_ids)
        # char emb has the shape  [B*T, N, channels], where N is the max number of chars tokens decoded from bpe tokens
        x = self.encoder(x=char_emb, x_mask=char_mask)['output']

        # Get average embedding over the chars
        mean_emb = ((x / char_mask.unsqueeze(-1).sum(1, keepdim=True)) * char_mask.unsqueeze(-1)).sum(1)
        subword_emb = torch.zeros((subword_mask.size(0), subword_mask.size(1), mean_emb.size(-1)), device=device)
        subword_emb[subword_mask.unsqueeze(-1).expand(-1, -1, mean_emb.size(-1))] = mean_emb.view(-1)

        return subword_emb
