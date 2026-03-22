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
from typing import Dict, List, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.utils.data import get_worker_info

from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes.module import NeuralModule
from nemo.utils import logging
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


def worker_init_fn(worker_id):
    """Per-worker init for DataLoader workers.

    Sets up tokenizers for the dataset (text and optionally phoneme)
    when using multiprocessing.
    """
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers

    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    tokenizer = setup_tokenizers(dataset.tokenizer_config, mode=dataset.dataset_type)
    dataset.text_tokenizer = tokenizer
    if hasattr(dataset, 'phoneme_tokenizer_config'):
        dataset.phoneme_tokenizer = instantiate(dataset.phoneme_tokenizer_config)


def add_eos_token(codes, codes_len, eos_id, num_eos_tokens=1):
    """Appends EOS tokens at the end of each sequence in the batch.

    Args:
        codes: (B, C, T')
        codes_len: (B,)
        eos_id: Token id to use as EOS.
        num_eos_tokens: Number of EOS tokens to append.
    """
    codes = torch.nn.functional.pad(input=codes, pad=(0, num_eos_tokens), value=0)
    codes_len = codes_len + num_eos_tokens
    for idx in range(codes.size(0)):
        codes[idx, :, codes_len[idx] - 1] = eos_id
    return codes, codes_len


def add_special_tokens(codes, codes_len, bos_id, eos_id, num_bos_tokens=1, num_eos_tokens=1):
    """Prepends BOS and appends EOS tokens to each sequence.

    Args:
        codes: (B, C, T')
    """
    codes = torch.nn.functional.pad(input=codes, pad=(num_bos_tokens, 0), value=bos_id)
    codes_len = codes_len + num_bos_tokens
    codes, codes_len = add_eos_token(codes=codes, codes_len=codes_len, eos_id=eos_id, num_eos_tokens=num_eos_tokens)
    return codes, codes_len


def remove_bos_token(codes, codes_len, num_tokens=1):
    codes = codes[:, :, num_tokens:]
    codes_len = codes_len - num_tokens
    return codes, codes_len


def remove_embedded_bos_token(embedded, embedded_len):
    embedded = embedded[:, 1:, :]
    embedded_len = embedded_len - 1
    return embedded, embedded_len


def remove_eos_token(codes, codes_len):
    codes_len = codes_len - 1
    codes = codes[:, :, :-1]
    mask = get_mask_from_lengths(lengths=codes_len)
    codes = codes * mask.unsqueeze(1)
    return codes, codes_len


def remove_embedded_eos_token(embedded, embedded_len):
    """Remove the last token from embedded sequences.

    Args:
        embedded: (B, T', D)
    """
    embedded_len = embedded_len - 1
    embedded = embedded[:, :-1, :]
    mask = get_mask_from_lengths(lengths=embedded_len)
    embedded = embedded * mask.unsqueeze(2)
    return embedded, embedded_len


def remove_special_tokens(codes, codes_len, num_bos_tokens=1):
    codes, codes_len = remove_bos_token(codes=codes, codes_len=codes_len, num_tokens=num_bos_tokens)
    codes, codes_len = remove_eos_token(codes=codes, codes_len=codes_len)
    return codes, codes_len


def pad_audio_codes(audio_codes: torch.Tensor, frame_stacking_factor: int) -> torch.Tensor:
    """Pads the time dimension of audio codes to a multiple of *frame_stacking_factor*.

    Args:
        audio_codes: (B, C, T)
        frame_stacking_factor: Factor to pad to.
    Returns:
        (B, C, T_padded)
    """
    T = audio_codes.size(2)
    T_padded = int(np.ceil(T / frame_stacking_factor) * frame_stacking_factor)
    num_pad = T_padded - T
    audio_codes = torch.nn.functional.pad(input=audio_codes, pad=(0, num_pad))
    return audio_codes


def clear_forbidden_logits(logits: torch.Tensor, codebook_size: int, forbid_audio_eos: bool = False) -> torch.Tensor:
    """Sets logits of forbidden tokens to ``-inf`` so they will never be sampled.

    Specifically, we forbid sampling of all special tokens except AUDIO_EOS
    which is allowed by default.

    Args:
        logits: (B, C, num_audio_tokens_per_codebook) or compatible shape.
        codebook_size: Base codebook size (excluding special tokens).
        forbid_audio_eos: If True, also forbid AUDIO_EOS tokens from being sampled.
    """
    logits[
        :,
        :,
        SpecialAudioToken.get_forbidden_tokens(codebook_size, forbid_audio_eos=forbid_audio_eos),
    ] = float('-inf')
    return logits


class CodecHelper:
    """Thin wrapper around a codec model and optional token converter.

    Instantiate once per model and use ``audio_to_codes`` / ``codes_to_audio``
    without having to pass the codec objects every time.
    """

    def __init__(self, codec_model, codec_converter=None):
        self.codec_model = codec_model
        self.codec_converter = codec_converter

    def audio_to_codes(self, audio, audio_len, sample_rate=None):
        """Encode audio waveforms into codec codes."""
        self.codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=audio.device.type, dtype=torch.float32):
            codes, codes_len = self.codec_model.encode(audio=audio, audio_len=audio_len, sample_rate=sample_rate)
            return codes, codes_len

    def codes_to_audio(self, codes, codes_len):
        """Decode codec codes back into audio waveforms.

        ``codes`` must already be unstacked to the shape the codec expects.
        """
        self.codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=codes.device.type, dtype=torch.float32):
            if self.codec_converter is not None:
                codes = self.codec_converter.convert_new_to_original(audio_tokens=codes, audio_lens=codes_len)
            audio, audio_len = self.codec_model.decode(tokens=codes, tokens_len=codes_len)
            return audio, audio_len, codes


class LocalTransformerHelper:
    """Orchestrates local-transformer forward passes and sampling.

    This is a plain Python class (not ``nn.Module``) that holds *references*
    to nn.Module sub-modules owned by the parent model.  Keeping it non-Module
    preserves checkpoint key compatibility.

    Args:
        local_transformer: The local transformer module.
        audio_embeddings: List/ModuleList of per-codebook embedding layers.
        audio_in_projection: Linear projection applied after per-codebook embedding.
        local_transformer_in_projection: Projection into the local transformer input space.
        local_transformer_audio_out_projection: Projection applied to local transformer output
            before the per-codebook output heads.
        local_transformer_out_projections: List/ModuleList of per-codebook output heads.
        num_audio_codebooks: Number of audio codebooks (C).
        frame_stacking_factor: Frame stacking factor (S).
        audio_eos_id: Token id for audio EOS.
        mask_token_id: Token id used for MaskGit masking.
        codebook_size: Base codebook size (excluding special tokens).
    """

    def __init__(
        self,
        local_transformer,
        audio_embeddings,
        audio_in_projection,
        local_transformer_in_projection,
        local_transformer_audio_out_projection,
        local_transformer_out_projections,
        num_audio_codebooks: int,
        frame_stacking_factor: int,
        audio_eos_id: int,
        mask_token_id: int,
        codebook_size: int,
    ):
        self.local_transformer = local_transformer
        self.audio_embeddings = audio_embeddings
        self.audio_in_projection = audio_in_projection
        self.local_transformer_in_projection = local_transformer_in_projection
        self.local_transformer_audio_out_projection = local_transformer_audio_out_projection
        self.local_transformer_out_projections = local_transformer_out_projections
        self.num_audio_codebooks = num_audio_codebooks
        self.frame_stacking_factor = frame_stacking_factor
        self.audio_eos_id = audio_eos_id
        self.mask_token_id = mask_token_id
        self.codebook_size = codebook_size

    def create_random_mask(self, codes):
        """Creates a mask where True indicates positions that should be replaced with MASK_TOKEN."""
        B, C, T = codes.shape
        rand_values = torch.rand(B, T, device=codes.device)
        frac_masked = cosine_schedule(rand_values)
        n_masked = torch.ceil(frac_masked * C).long()
        random_permutations = torch.argsort(torch.rand(B, C, T, device=codes.device), dim=1)
        mask_indices = torch.arange(C, device=codes.device).view(1, C, 1)
        mask = mask_indices < n_masked.view(B, 1, T)
        mask = torch.gather(mask, 1, random_permutations)
        return mask

    def apply_random_mask(self, codes):
        """Randomly replaces some codes with MASK_TOKEN following the cosine schedule."""
        mask = self.create_random_mask(codes)
        codes_with_mask = torch.where(mask, self.mask_token_id, codes)
        return codes_with_mask, mask

    def compute_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """Predicts the logits for all codebooks using the local transformer.

        Used in both autoregressive (AR) and MaskGit (MG) modes during
        training and validation (not inference/sampling).

        The sequence layout is slightly different between AR and MG modes, as shown below
        (using an 8-codebook setup as an example)::

            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            | AR target  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |   none  |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            | MG target  |  none   |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            |   Input    | Magpie  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
            |            | Latent  | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            | Seq. Index |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+

        Args:
            dec_out: (B, T', E)
            audio_codes_target: (B, C, T')
            targets_offset_by_one: if False, target for index 0 is codebook 0 (AR);
                if True, target for index 1 is codebook 0 (MaskGit).
        """
        C = self.num_audio_codebooks
        dec_out_all = dec_out.reshape(-1, dec_out.size(-1))  # (B*T', E)
        local_transformer_input = [dec_out_all]
        audio_codes_target = pad_audio_codes(audio_codes_target, self.frame_stacking_factor).long()
        for fs_index in range(self.frame_stacking_factor):
            for codebook_num in range(C):
                codes = audio_codes_target[:, codebook_num, fs_index :: self.frame_stacking_factor]
                codes = codes.reshape(-1)
                codebook_embedding = self.audio_embeddings[codebook_num + fs_index * C](codes)
                codebook_embedding = self.audio_in_projection(codebook_embedding)
                local_transformer_input.append(codebook_embedding)

        local_transformer_input = torch.stack(local_transformer_input, dim=1)
        local_transformer_input = self.local_transformer_in_projection(local_transformer_input)
        _mask = torch.ones(
            local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
        )
        local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']
        if not targets_offset_by_one:
            local_transformer_output = local_transformer_output[:, :-1, :]
        else:
            local_transformer_output = local_transformer_output[:, 1:, :]

        local_transformer_output = self.local_transformer_audio_out_projection(local_transformer_output)

        all_code_logits = []
        for fs_index in range(self.frame_stacking_factor):
            for codebook_num in range(audio_codes_target.size(1)):
                codebook_logits = self.local_transformer_out_projections[codebook_num + fs_index * C](
                    local_transformer_output[:, codebook_num + fs_index * C, :]
                )
                all_code_logits.append(codebook_logits)
        all_code_logits = torch.cat(all_code_logits, dim=1)

        all_code_logits = all_code_logits.view(
            audio_codes_target.size(0), audio_codes_target.size(2) // self.frame_stacking_factor, -1
        )

        return all_code_logits

    def sample_autoregressive(
        self,
        dec_output: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        use_kv_cache: bool = True,
        forbid_audio_eos: bool = False,
        sanitize_logits: bool = False,
    ) -> torch.Tensor:
        """Sample audio codes autoregressively across codebooks using the local transformer.

        Args:
            dec_output: Decoder output tensor (B, E).
            temperature: Sampling temperature. When <= 0, uses argmax.
            topk: Number of top-probability tokens to consider.
            unfinished_items: Batch indices that have not completed generation (EOS forbidden).
            finished_items: Batch indices that are completed (EOS forced).
            use_cfg: Whether to use classifier-free guidance (doubled batch).
            cfg_scale: Scale factor for CFG.
            use_kv_cache: Whether to use key-value caching in the local transformer.
            forbid_audio_eos: Whether to globally forbid audio EOS.
            sanitize_logits: Whether to clamp/clean logits before sampling.

        Returns:
            Sampled audio codes (B, num_codebooks, frame_stacking_factor).
        """
        self.local_transformer.reset_cache(use_cache=use_kv_cache)
        dec_output = dec_output.unsqueeze(1)  # (B, 1, E)
        local_transformer_input = self.local_transformer_in_projection(dec_output)
        all_preds = []
        for codebook_num in range(self.num_audio_codebooks * self.frame_stacking_factor):
            _mask = torch.ones(
                local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
            )
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']

            lt_out_for_proj = self.local_transformer_audio_out_projection(local_transformer_output[:, -1, :])
            codebook_logits = self.local_transformer_out_projections[codebook_num](lt_out_for_proj)

            if use_cfg:
                actual_batch_size = codebook_logits.size(0) // 2
                conditional_logits = codebook_logits[:actual_batch_size]
                unconditional_logits = codebook_logits[actual_batch_size:]
                cfg_logits = cfg_scale * conditional_logits + (1.0 - cfg_scale) * unconditional_logits
                codebook_logits[:actual_batch_size] = cfg_logits

            if sanitize_logits:
                codebook_logits = torch.nan_to_num(codebook_logits, nan=0.0, posinf=100.0, neginf=-100.0)
                codebook_logits = codebook_logits.clamp(min=-100.0, max=100.0)

            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0

            codebook_logits = clear_forbidden_logits(
                codebook_logits.unsqueeze(1), self.codebook_size, forbid_audio_eos=forbid_audio_eos
            ).squeeze(1)

            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            if temperature <= 0.0:
                codebook_preds = codebook_logits_rescored.argmax(dim=-1, keepdim=True)
            else:
                codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)
                codebook_preds = torch.multinomial(codebook_probs, 1)

            if use_cfg:
                codebook_preds[actual_batch_size:] = codebook_preds[:actual_batch_size]
            all_preds.append(codebook_preds)

            next_local_transformer_input = self.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(1)
            next_local_transformer_input = self.audio_in_projection(next_local_transformer_input)
            next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input)
            local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

        all_preds = torch.cat(all_preds, dim=1)  # (B, num_codebooks * frame_stacking_factor)
        all_preds = all_preds.reshape(-1, self.frame_stacking_factor, self.num_audio_codebooks).permute(0, 2, 1)
        if use_cfg:
            all_preds = all_preds[:actual_batch_size]

        return all_preds

    def sample_maskgit(
        self,
        dec_output: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        n_steps: int = 3,
        noise_scale: float = 0.0,
        fixed_schedule: Optional[List[int]] = None,
        dynamic_cfg_scale: bool = False,
        sampling_type: Optional[str] = None,
        forbid_audio_eos: bool = False,
    ) -> torch.Tensor:
        """Sample audio codes using MaskGit-like iterative prediction with the local transformer.

        Args:
            dec_output: Decoder output tensor (B, E).
            temperature: Sampling temperature.
            topk: Number of top-probability tokens to consider.
            unfinished_items: Batch indices that have not completed generation.
            finished_items: Batch indices that are completed.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: Scale factor for CFG.
            n_steps: Number of iterative refinement steps.
            noise_scale: Scale factor for noise added to confidence scores.
            fixed_schedule: Fixed schedule for number of tokens to unmask per step.
            dynamic_cfg_scale: Whether to dynamically adjust CFG scale.
            sampling_type: Sampling strategy.
            forbid_audio_eos: Whether to globally forbid audio EOS.

        Returns:
            Sampled audio codes (B, num_codebooks, frame_stacking_factor).
        """
        device = dec_output.device
        self.local_transformer.reset_cache(use_cache=False)
        dec_output = dec_output.unsqueeze(1)
        local_transformer_input_init = self.local_transformer_in_projection(dec_output)
        codebook_seq_len = self.num_audio_codebooks * self.frame_stacking_factor
        B = dec_output.size(0)

        min_confidence = 0
        max_confidence = 5
        confidences = min_confidence * torch.ones(B, codebook_seq_len, device=device)
        codes = self.mask_token_id * torch.ones((B, codebook_seq_len), device=device, dtype=torch.long)
        sampled_codes = codes.clone()
        if fixed_schedule is not None:
            n_steps = len(fixed_schedule)
        for step in range(n_steps):
            progress = step / n_steps
            frac_masked = cosine_schedule(torch.tensor(progress))
            if sampling_type == "causal" or sampling_type == "purity_causal":
                frac_masked = torch.ones_like(frac_masked) * (1.0 - progress)
            if fixed_schedule is None:
                n_masked = torch.ceil(codebook_seq_len * frac_masked).long()
            else:
                n_masked = codebook_seq_len - fixed_schedule[step]
            n_unmasked = codebook_seq_len - n_masked

            if sampling_type == "causal" or sampling_type == "purity_causal":
                n_frames_to_allow = int(np.floor(progress * self.frame_stacking_factor + 1))
                confidences[:, n_frames_to_allow * self.num_audio_codebooks :] = min_confidence - 1

            _, topk_indices = torch.topk(confidences, k=n_unmasked, dim=1)
            if use_cfg:
                actual_batch_size = topk_indices.size(0) // 2
                assert (
                    topk_indices[actual_batch_size:] == topk_indices[:actual_batch_size]
                ).all(), "Topk indices are not the same for conditional and unconditional codes"

            unmasked_codes = torch.gather(sampled_codes, dim=1, index=topk_indices)
            codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)

            local_transformer_input = local_transformer_input_init
            for codebook_num in range(codebook_seq_len):
                next_local_transformer_input = self.audio_embeddings[codebook_num](codes[:, codebook_num]).unsqueeze(1)
                next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input)
                local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

            _mask = torch.ones(B, codebook_seq_len + 1, device=device)
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']

            logits = []
            for codebook_num in range(codebook_seq_len):
                codebook_logits = self.local_transformer_out_projections[codebook_num](
                    local_transformer_output[:, codebook_num + 1, :]
                )
                logits.append(codebook_logits)
            logits = torch.stack(logits, dim=1)

            if use_cfg:
                actual_batch_size = logits.size(0) // 2
                conditional_logits = logits[:actual_batch_size]
                unconditional_logits = logits[actual_batch_size:]
                if not dynamic_cfg_scale:
                    current_cfg_scale = cfg_scale
                else:
                    progress = step / (n_steps - 1)
                    interp = progress
                    current_cfg_scale = (cfg_scale - 1) * interp + 1.0
                cfg_logits = current_cfg_scale * conditional_logits + (1.0 - current_cfg_scale) * unconditional_logits
                logits[:actual_batch_size] = cfg_logits

            logits = clear_forbidden_logits(logits, self.codebook_size, forbid_audio_eos=forbid_audio_eos)

            for item_idx in unfinished_items:
                logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                logits[item_idx, :, :] = float('-inf')
                logits[item_idx, :, self.audio_eos_id] = 0.0

            logits_topk = torch.topk(logits, topk, dim=-1)[0]
            indices_to_remove = logits < logits_topk[:, :, -1].unsqueeze(-1)
            logits_rescored = logits.clone()
            logits_rescored[indices_to_remove] = float('-inf')
            probs = torch.softmax(logits_rescored / temperature, dim=-1)
            sampled_codes = torch.multinomial(probs.view(B * codebook_seq_len, -1), 1).view(B, codebook_seq_len)
            if use_cfg:
                sampled_codes[actual_batch_size:] = sampled_codes[:actual_batch_size]
                probs[actual_batch_size:] = probs[:actual_batch_size]
            if sampling_type != "purity_causal" and sampling_type != "purity_default":
                confidences = torch.gather(probs, dim=2, index=sampled_codes.unsqueeze(-1)).squeeze(-1)
            else:
                confidences = probs.max(dim=2)[0]
            sampled_codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)
            if noise_scale > 0.0:
                noise = (torch.rand_like(confidences) - 0.5) * noise_scale * (1 - (step + 2) / n_steps)
                confidences += noise
                confidences[actual_batch_size:] = confidences[:actual_batch_size]
            confidence_eps = 0.1
            assert (
                confidences.max() + confidence_eps < max_confidence
            ), f"Predicted confidence is approaching max_confidence: {confidences.max()}"
            confidences.scatter_(
                index=topk_indices, dim=1, src=max_confidence * torch.ones_like(topk_indices, dtype=torch.float)
            )
        codes = sampled_codes
        assert not (
            codes == self.mask_token_id
        ).any(), "Codes contain mask tokens after completion of MaskGit sampling"

        codes = codes.reshape(B, self.frame_stacking_factor, self.num_audio_codebooks).permute(0, 2, 1)

        if use_cfg:
            codes = codes[:actual_batch_size]
        return codes
