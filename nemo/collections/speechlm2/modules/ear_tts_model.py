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

import json
import math
import os
from dataclasses import dataclass, fields
from typing import Any

import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoModelForTextEncoding, Cache
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init
from nemo.utils import logging

# ==============================================================================
# MLP module and Norm
# ==============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MLPLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        y = self.pre_norm(x)
        y = self.mlp(y)
        y = self.post_norm(y)
        x = x + y
        return x


# ==============================================================================
# Triton-accelerated and Fallback Functions
# ==============================================================================

TRITON_IMPORTED = False
try:
    import triton
    import triton.language as tl

    TRITON_IMPORTED = True
except ImportError:
    TRITON_IMPORTED = False

USE_TRITON = TRITON_IMPORTED and torch.cuda.is_available()

if USE_TRITON:
    logging.info("Triton available & CUDA detected. Using Triton kernel for batch_matmul.")

    @triton.jit
    def batch_matmul_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        result_ptr,
        b,
        d_in,
        d_out,
        n,
        BLOCK_SIZE_DIN: tl.constexpr,
        BLOCK_SIZE_DOUT: tl.constexpr,
    ):
        batch_id = tl.program_id(axis=0)
        dout_block_id = tl.program_id(axis=1)

        if batch_id >= b:
            return

        idx = tl.load(y_ptr + batch_id)

        x_offset = x_ptr + batch_id * d_in
        w_offset = w_ptr + idx * d_out * d_in

        dout_offsets = dout_block_id * BLOCK_SIZE_DOUT + tl.arange(0, BLOCK_SIZE_DOUT)
        dout_mask = dout_offsets < d_out

        result_block = tl.zeros([BLOCK_SIZE_DOUT], dtype=tl.float32)

        for din_start in range(0, d_in, BLOCK_SIZE_DIN):
            din_offsets = din_start + tl.arange(0, BLOCK_SIZE_DIN)
            din_mask = din_offsets < d_in

            x_i = tl.load(x_offset + din_offsets, mask=din_mask, other=0.0)

            w_i_block = tl.load(
                w_offset + dout_offsets[:, None] * d_in + din_offsets[None, :],
                mask=(dout_mask[:, None] & din_mask[None, :]),
                other=0.0,
            )

            result_block += tl.sum(w_i_block * x_i[None, :], axis=1)

        result_offset = result_ptr + batch_id * d_out + dout_offsets
        tl.store(result_offset, result_block, mask=dout_mask)

    def batch_matmul_triton(x, w, y, BLOCK_SIZE_DIN: int = 16, BLOCK_SIZE_DOUT: int = 64):
        assert x.is_contiguous() and w.is_contiguous() and y.is_contiguous()

        b, d_in = x.shape
        n, d_out, _ = w.shape
        result = torch.empty(b, d_out, device=x.device, dtype=torch.float32)

        batch_matmul_kernel[lambda meta: (b, triton.cdiv(d_out, meta["BLOCK_SIZE_DOUT"]))](
            x.float(),
            w.float(),
            y,
            result,
            b,
            d_in,
            d_out,
            n,
            BLOCK_SIZE_DIN=BLOCK_SIZE_DIN,
            BLOCK_SIZE_DOUT=BLOCK_SIZE_DOUT,
        )

        return result.to(dtype=x.dtype)

    batch_matmul = batch_matmul_triton

else:
    logging.info("Using PyTorch fallback (Triton unavailable or no CUDA).")

    # Fallback to PyTorch implementation if Triton is not available
    def batch_matmul_pytorch(x: Tensor, w: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        """
        Performs a batched matrix multiplication using PyTorch's native functions.

        This function serves as a fallback when Triton is not available. It achieves
        the same result by gathering the appropriate weight matrices and using `torch.bmm`.

        Args:
            x (Tensor): The input tensor of shape `[batch_size, d_in]`.
            w (Tensor): The weight tensor of shape `[num_weights, d_out, d_in]`.
            y (Tensor): The index tensor of shape `[batch_size]`.

        Returns:
            Tensor: The result of the multiplication, shape `[batch_size, d_out]`.
        """
        # w[y] gathers the weight matrices for each item in the batch.
        # x.unsqueeze(2) reshapes x to [batch_size, d_in, 1] for bmm.
        # The result is squeezed to remove the trailing dimension of size 1.
        return torch.bmm(w[y], x.unsqueeze(2)).squeeze(2)

    batch_matmul = batch_matmul_pytorch


# ==============================================================================
# Core Mathematical and Masking Functions
# ==============================================================================


def gumbel_like(tensor: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Generates a tensor of Gumbel noise with the same shape as the input tensor.

    This is used for the Gumbel-Max trick, a technique to sample from a categorical
    distribution in a differentiable way (using a straight-through estimator).

    Args:
        tensor (torch.Tensor): The input tensor to match the shape of.
        eps (float): A small epsilon value for numerical stability.

    Returns:
        torch.Tensor: A tensor containing Gumbel noise.
    """
    # Sample from a uniform distribution
    u = torch.rand_like(tensor)
    # Apply the inverse CDF of the Gumbel distribution
    return -torch.log(-torch.log(u + eps) + eps)


def sequence_mask(lengths: Tensor, max_length: Tensor | int | None = None) -> Tensor:
    """
    Creates a boolean mask from a 1D tensor of sequence lengths.

    This function is useful for masking out padding in sequences. Given a tensor
    of lengths, it produces a 2D boolean tensor where `mask[i, j]` is `True` if
    `j < lengths[i]` and `False` otherwise.

    Args:
        lengths (Long Tensor): A 1D tensor of integer lengths. Shape: `[batch_size]`.
        max_length (Long Tensor | int | None, optional): The maximum length of the mask. If None,
                                           it is inferred from the maximum value
                                           in `lengths`. Defaults to None.

    Returns:
        Tensor: The boolean mask. Shape: `[batch_size, max_length]`.
    """
    if max_length is None:
        max_length = lengths.max()

    # Create a range tensor from 0 to max_length - 1
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)  # type: ignore[arg-type]

    # Compare each length with the range tensor to create the mask via broadcasting
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def get_masking_rate(rate: Tensor, exponent: float = 3.0) -> Tensor:
    """
    Converts a desired token keep rate to a masking rate using a power function.

    This function is part of a scheduling strategy for masking, where the effective
    masking rate changes non-linearly with the desired keep rate. This function is
    its own inverse.

    Args:
        rate (Tensor): The desired rate of tokens to keep (0 to 1).
        exponent (float, optional): The exponent for the transformation. Defaults to 3.0.

    Returns:
        Tensor: The corresponding masking rate.
    """
    return (1 - rate.pow(exponent)).pow(1 / exponent)


# Alias the function for clarity in the inverse context
get_rate = get_masking_rate


def get_mask(
    code_mask: Tensor,
    num_masking: Tensor,
    unmasking: bool = False,
    validate: bool = False,
) -> Tensor:
    """
    Adjusts a boolean mask by masking or unmasking tokens from the end.

    This function operates on a `code_mask` where `True` values represent valid
    tokens and are assumed to be contiguous at the start of the sequence. It
    calculates a new mask by decreasing (masking) or increasing (unmasking)
    the number of `True` values.

    Args:
        code_mask (Tensor): The input boolean mask. Shape: `[..., depth]`.
        num_masking (Tensor): The number of tokens to mask or unmask.
                              Shape matching `code_mask`'s batch dimensions.
        unmasking (bool, optional): If `True`, increases the number of valid
                                  tokens (unmasking). Defaults to `False`.
        validate (bool, optional): If `True`, asserts that the input `code_mask`
                                 is contiguous. This adds a slight overhead and
                                 is mainly for debugging. Defaults to `False`.

    Returns:
        Tensor: A new boolean mask with the adjusted length of valid tokens.
                Shape is identical to `code_mask`.
    """
    depth = code_mask.size(-1)
    num_valid = code_mask.sum(dim=-1, dtype=torch.long)

    if validate:
        # Reconstruct the expected contiguous mask and assert equality.
        expected_mask = sequence_mask(num_valid.view(-1), depth).view_as(code_mask)
        assert torch.equal(
            code_mask, expected_mask
        ), "Input `code_mask` must have contiguous `True` values at the beginning."

    # Calculate the target number of valid tokens.
    if not unmasking:
        # Masking: reduce the number of valid tokens, ensuring it's not negative.
        num_to_keep = (num_valid - num_masking).clamp_min(0)
    else:
        # Unmasking: increase the number of valid tokens, capped by total depth.
        num_to_keep = (num_valid + num_masking).clamp_max(depth)

    # Generate the new mask using the final number of tokens to keep.
    return sequence_mask(num_to_keep.view(-1), depth).view_as(code_mask)


# ==============================================================================
# Model and Vocabulary Utilities
# ==============================================================================


@dataclass
class RVQEARTTSOutput:
    """
    Output type for the RVQEARTTSModel, providing a structured way to return model outputs.
    This class allows accessing outputs by attribute, key, or index.
    """

    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    c_loss: Tensor | None = None
    k_loss: Tensor | None = None

    hidden_states: Tensor | None = None
    past_key_values: Tensor | None = None

    codes: Tensor | None = None
    lm_logits: Tensor | None = None
    eos_flag: Tensor | None = None

    def __getitem__(self, item: str | int):
        """Allows for accessing attributes by key or index."""
        if isinstance(item, str):
            return getattr(self, item)
        else:
            # Access fields in the order they are defined in the dataclass
            return getattr(self, fields(self)[item].name)


def find_and_delete_module(parent_module: nn.Module, target_module: nn.Module, parent_name: str) -> str | None:
    """
    Recursively searches for a specific module instance and deletes it from its parent.

    This is useful for dynamically modifying a model's architecture, such as replacing
    an existing embedding layer with a custom one.

    Args:
        parent_module (nn.Module): The module to search within.
        target_module (nn.Module): The exact module instance to find and delete.
        parent_name (str): The initial name of the parent module for constructing the path.

    Returns:
        str | None: The full dotted name of the deleted attribute if found, otherwise None.
    """
    # Iterate over all direct children of the parent module
    for name, module in parent_module.named_children():
        # Use the 'is' operator to check for object identity, not just value equality
        if module is target_module:
            # If found, delete the attribute from the parent and return its name
            delattr(parent_module, name)
            return f"{parent_name}.{name}"

        # If not found, recurse into the child module
        found_path = find_and_delete_module(module, target_module, parent_name=f"{parent_name}.{name}")
        if found_path:
            return found_path
    return None


def build_vocabs(
    tokenizer: AutoTokenizer, vocab_dir: str | None = None
) -> tuple[dict[int, tuple[int, ...]], dict[str, int], int]:
    """
    Builds or loads a character-level vocabulary derived from a subword tokenizer.

    This function creates a mapping from each subword in a pretrained tokenizer to a
    sequence of character IDs. It follows a modern practice of using a directory
    to save and load vocabulary files, making the process more robust and extensible.

    The primary source of truth is the `char_vocab.json` file. If it exists, it's
    loaded. Otherwise, it's created from the pretrained tokenizer and saved.

    Args:
        tokenizer (AutoTokenizer): The pretrained Hugging Face tokenizer class.
        vocab_dir (str | None, optional): The directory to save or load the character
                                          vocabulary from. Defaults to None.

    Returns:
        tuple[dict[int, tuple[int, ...]], dict[str, int], int]: A tuple containing:
            - A mapping from subword IDs to tuples of character IDs.
            - The character-to-ID vocabulary dictionary.
            - The ID for the subword padding token.
    """

    def _build_char_vocab() -> dict[str, int]:
        # Find all single-character tokens in the original tokenizer's vocabulary
        single_chars = {
            subword: subword_id for subword, subword_id in tokenizer.tokenizer.vocab.items() if len(subword) == 1
        }
        # Create a new, dense character vocabulary sorted by the original token ID
        sorted_chars = sorted(single_chars.keys(), key=lambda k: single_chars[k])
        char_vocab = {char: i for i, char in enumerate(sorted_chars)}
        return char_vocab

    # 1. Load or build the character vocabulary
    if vocab_dir:
        from filelock import FileLock

        char_vocab_file = os.path.join(vocab_dir, "char_vocab.json")
        os.makedirs(vocab_dir, exist_ok=True)
        with FileLock(char_vocab_file + ".lock", timeout=60):
            if not os.path.exists(char_vocab_file):
                char_vocab = _build_char_vocab()

                logging.info(f"Saving character vocabulary to {char_vocab_file}")
                with open(char_vocab_file, "w", encoding="utf-8") as f:
                    json.dump(char_vocab, f, ensure_ascii=False, indent=2)

        # All processes can now safely load the file.
        logging.info(f"Loading character vocabulary from {char_vocab_file}")
        with open(char_vocab_file, encoding="utf-8") as f:
            char_vocab = json.load(f)
    else:
        # No cache directory provided, build in memory.
        logging.info("Building character vocabulary from tokenizer.")
        char_vocab = _build_char_vocab()

    # 2. Reconstruct the subword-to-character mapping on the fly
    subword_id_to_char_ids = {
        subword_id: tuple(char_vocab[char] for char in subword if char in char_vocab)
        for subword, subword_id in tokenizer.tokenizer.vocab.items()
    }
    # Filter out subwords that contain characters not in our character vocabulary
    subword_id_to_char_ids = {k: v for k, v in subword_id_to_char_ids.items() if v}

    # 3. Define a padding index for subwords
    subword_padding_idx = len(tokenizer.vocab)
    # The padding subword maps to a new character padding ID
    subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)
    return subword_id_to_char_ids, char_vocab, subword_padding_idx


@torch.compile
def depthsum_encoding_step(
    embs: Tensor,
    r: Tensor,
    code: Tensor,
    depth_str: int = 0,
    k: int = 72,
) -> Tensor:
    for i in range(depth_str, depth_str + k):
        idx_sel = (
            embs[i].pow(2).sum(-1)  # [g?, v]
            - 2
            * (r.unsqueeze(-2) @ embs[i].transpose(-1, -2)).squeeze(-2)  # [b, ?, g?, h] , [g?, h, v] -> [b, ?, g?, v]
        ).argmin(-1)

        emb_i = F.embedding(idx_sel, embs[i])
        r = r - emb_i

        code[..., i] = idx_sel

    return code


class MoGHead(nn.Module):
    """
    A Mixture of Gaussians (MoG) prediction head.

    This module takes a hidden state and predicts the parameters for a mixture of
    Gaussian distributions. It's suitable for modeling continuous, multi-modal data.

    Args:
        hidden_size (int): The dimensionality of the input hidden state.
        intermediate_size (int): The dimensionality of the MLP layers.
        out_size (int): The dimensionality of the output vectors (the mean of each Gaussian).
        num_layers (int): The number of MLP layers in the stack.
        num_predictions (int): The number of Gaussian components in the mixture.
        low_rank (int | None): The dimensionality used for compressing the hidden states.
        min_log_std (float): The minimum value for the logarithm of the standard deviation.
        eps (float): A small epsilon value for the RMSNorm layers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        out_size: int,
        num_layers: int,
        num_predictions: int,
        low_rank: int | None = 64,
        min_log_std: float = -4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_size = out_size
        self.low_rank = low_rank
        self.num_predictions = num_predictions
        self.min_log_std = min_log_std

        self.mlp_stack = nn.Sequential(
            *[MLPLayer(hidden_size, intermediate_size, eps=eps) for _ in range(num_layers)],
            RMSNorm(hidden_size, eps=eps),
        )

        if low_rank is None:
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)  # Predicts mixture weights
            self.proj_mus = nn.Linear(hidden_size, num_predictions * out_size, bias=False)  # Predicts means
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)  # Predicts log standard deviations
        else:
            assert low_rank < out_size
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)  # Predicts mixture weights
            self.proj_mus = nn.Linear(hidden_size, num_predictions * low_rank, bias=False)  # Predicts means
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)  # Predicts log standard deviations
            self.proj_else = nn.Linear(hidden_size, out_size, bias=False)
            self.low_mat = nn.Parameter(torch.randn(num_predictions, out_size, low_rank) * (low_rank**-0.5))

    def infer(self, x: Tensor, guidance_scale: float = 0.0, top_p_or_k: float | int = 1.0) -> tuple[Tensor, Tensor]:
        """
        Performs inference by sampling from the predicted mixture distribution.

        Args:
            x (Tensor): The input hidden state.
            guidance_scale (float): The weight for classifier-free guidance.
            top_p_or_k (float | int): The value for top-p (nucleus) or top-k sampling of the mixture components.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the mean of the chosen component,
                                   and the log standard deviations.
        """
        b, t, _ = x.size()
        n, d = self.num_predictions, self.low_rank or self.out_size
        x = self.mlp_stack(x)
        if guidance_scale > 0:
            b //= 2
            x_cond, x_uncond = x.chunk(2, dim=0)
            x = x_cond + guidance_scale * (x_cond - x_uncond)

        logits = self.proj_logits(x)

        # Apply top-p or top-k filtering to the mixture logits
        if top_p_or_k is not None:

            logits = (
                TopPLogitsWarper(top_p_or_k)(
                    None,
                    logits.view(-1, n),
                ).view_as(logits)
                if isinstance(top_p_or_k, float)
                else TopKLogitsWarper(top_p_or_k)(
                    None,
                    logits.view(-1, n),
                ).view_as(logits)
            )

        # Sample a mixture component using the Gumbel-Max trick
        mixture_indices = (F.log_softmax(logits, dim=-1) + gumbel_like(logits)).argmax(-1)

        # Select the mean corresponding to the sampled component
        mu = batch_matmul(
            x.view(b * t, -1),
            self.proj_mus.weight.detach().view(n, d, -1),
            mixture_indices.view(b * t),
        ).view(b, t, d)
        if self.proj_mus.bias is not None:
            mu += self.proj_mus.bias.detach().view(n, d)[mixture_indices]

        if self.low_rank:
            assert math.log2(d).is_integer() and math.log2(self.out_size).is_integer()
            mu = batch_matmul(
                mu.view(b * t, -1),
                self.low_mat.detach().view(n, self.out_size, -1),
                mixture_indices.view(b * t),
                BLOCK_SIZE_DIN=d,
                BLOCK_SIZE_DOUT=self.out_size,
            ).view(b, t, self.out_size)

            mu_res = self.proj_else(x)
        else:
            mu_res = torch.zeros((b, t, d), device=x.device)

        logs = self.proj_logs(x).clamp_min(self.min_log_std)
        return mu * torch.exp(logs) + mu_res, logs

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Performs a forward pass for training.

        Args:
            x (Tensor): The input hidden state.

        Returns:
            tuple[Tensor, Tensor, Tensor]: A tuple containing the mixture logits,
                                           the means for all components, and the
                                           log standard deviations.
        """
        b, t, _ = x.size()
        d = self.low_rank or self.out_size
        x = self.mlp_stack(x)
        logits = self.proj_logits(x)
        mus = self.proj_mus(x).view(b, t, self.num_predictions, d)
        logs = self.proj_logs(x).clamp_min(self.min_log_std)

        if self.low_rank:
            mu_res = self.proj_else(x)
        else:
            mu_res = torch.zeros((b, t, d), device=x.device)
        return logits, mus, mu_res, logs

    def dist(self, mus: Tensor, mu: Tensor) -> Tensor:
        """
        mus: [b, t, n, d]
        mu: [b, t, d]

        return: [b, t, n]
        """
        if self.low_rank is None:
            return (mus - mu.unsqueeze(-2)).pow(2).sum(-1)
        else:
            low_mat_sq = self.low_mat.transpose(-1, -2) @ self.low_mat
            x, y = mus, mu
            wx_sq = (
                x
                * torch.einsum(
                    "btni,nij->btnj",
                    x,
                    low_mat_sq.to(x),
                )
            ).sum(
                -1
            )  # [b, t, n]
            y_sq = y.pow(2).sum(-1, keepdim=True)  # [b, t, 1]
            xwy = (x * torch.einsum("bti,nij->btnj", y, self.low_mat.to(y))).sum(
                -1
            )  # [b, t, n, d_l], [n, d_i, d_l], [b, t, d_i] -> [b, t, n]

            dist = wx_sq + y_sq - 2 * xwy
            return torch.abs(dist)


class NeMoSubwordFlagEmbedding(nn.Module):
    """
    Adds a tiny embedding table for continuation tokens
    (subwords that do NOT start with Ġ or the word-boundary marker).
    Compatible with NeMo AutoTokenizer.
    """

    def __init__(self, tokenizer: AutoTokenizer, d_model: int):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.d_model = d_model

        # Precompute continuation flags
        tokens = [self.tokenizer.ids_to_tokens(i) for i in range(self.vocab_size)]
        self.register_buffer(
            'is_continuation',
            torch.tensor(
                [1 if not (tok.startswith("Ġ") or tok.startswith("▁")) else 0 for tok in tokens], dtype=torch.long
            ),
        )

        # Tiny embedding table: 0 = word-start, 1 = continuation
        init_std = self.d_model**-0.5
        self.cont_emb = nn.Embedding(2, self.d_model)
        nn.init.normal_(self.cont_emb.weight, mean=0.0, std=init_std)

        # Force word-start embedding to zero so only continuation tokens get shifted
        self.cont_emb.weight.data[0].zero_()

    def forward(self, subword_embeds: torch.Tensor, token_ids: torch.LongTensor):
        # Continuation flags
        cont_flags = self.is_continuation[token_ids]

        # Add continuation embedding
        cont_emb = self.cont_emb(cont_flags)
        return subword_embeds + cont_emb


class SubwordFlagEmbedding(nn.Module):
    """
    Adds a small continuation embedding for subwords (tokens without word-boundary marker).
    Automatically adds a custom padding token at index vocab_size.
    Ignores special tokens (starting with '<') when computing continuation flags.
    """

    def __init__(self, tokenizer: AutoTokenizer, d_model: int):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.d_model = d_model

        # Custom pad token at vocab_size
        self.pad_id = self.vocab_size
        # register pad_id as a tensor buffer to avoid device issues
        self.register_buffer("pad_tensor", torch.tensor(self.pad_id, dtype=torch.long))

        # Precompute continuation flags
        tokens = [self.tokenizer.ids_to_tokens(i) for i in range(self.vocab_size)]
        cont_flags = [
            1 if not (tok.startswith("Ġ") or tok.startswith("▁") or tok.startswith("<")) else 0 for tok in tokens
        ]
        cont_flags.append(0)  # for the custom pad token
        self.register_buffer("is_continuation", torch.tensor(cont_flags, dtype=torch.long))

        # Continuation embedding
        init_std = self.d_model**-0.5
        self.cont_emb = nn.Embedding(2, self.d_model)
        nn.init.normal_(self.cont_emb.weight, mean=0.0, std=init_std)
        self.cont_emb.weight.data[0].zero_()

    def forward(self, subword_embeds: torch.Tensor, token_ids: torch.LongTensor):
        # Replace OOV token IDs with pad_id safely
        token_ids_clamped = torch.where(token_ids >= self.vocab_size, self.pad_tensor, token_ids)
        # Continuation flags
        cont_flags = self.is_continuation[token_ids_clamped]
        # Add continuation embedding
        cont_emb = self.cont_emb(cont_flags)
        return subword_embeds + cont_emb


class BOSEOSEmbedding(nn.Module):
    """
    Adds independent embeddings for BOS and EOS tokens using a single embedding table.
    Index 0 = regular token (ignored), 1 = BOS, 2 = EOS.
    Compatible with Hugging Face tokenizers that may or may not have BOS/EOS.
    """

    def __init__(self, tokenizer: AutoTokenizer, d_model: int):
        super().__init__()

        self.tokenizer = tokenizer
        # vocab size that includes special tokens
        vocab_dict = self.tokenizer.tokenizer.get_vocab()
        self.vocab_size = max(vocab_dict.values())
        self.d_model = d_model

        # Custom pad token for OOVs
        self.pad_id = self.vocab_size
        self.register_buffer("pad_tensor", torch.tensor(self.pad_id, dtype=torch.long))

        # Identify BOS and EOS tokens (may be None)
        tokens = [self.tokenizer.ids_to_tokens(i) for i in range(self.vocab_size)]

        special_flags = []
        for tok in tokens:
            if self.tokenizer.bos_token is not None and tok == self.tokenizer.bos_token:
                special_flags.append(1)
            elif self.tokenizer.eos_token is not None and tok == self.tokenizer.eos_token:
                special_flags.append(2)
            else:
                special_flags.append(0)
        special_flags.append(0)  # for custom pad token
        self.register_buffer("special_flags", torch.tensor(special_flags, dtype=torch.long))
        # Embedding table: 0 = regular, 1 = BOS, 2 = EOS
        init_std = self.d_model**-0.5
        self.special_emb = nn.Embedding(3, d_model)
        nn.init.normal_(self.special_emb.weight, mean=0.0, std=init_std)
        self.special_emb.weight.data[0].zero_()  # regular tokens ignored

    def forward(self, token_embeds: torch.Tensor, token_ids: torch.LongTensor):
        """
        token_embeds: (B, T, d_model)
        token_ids:    (B, T)
        """
        # Clamp OOVs to custom pad token
        safe_ids = torch.where(token_ids >= self.vocab_size, self.pad_tensor, token_ids)

        # Lookup flags (0=regular, 1=BOS, 2=EOS)
        flags = self.special_flags[safe_ids]
        return token_embeds + self.special_emb(flags)


class SubwordEmbedding(nn.Module):
    """
    Produces subword embeddings from a Hugging Face tokenizer vocabulary.
    No special handling for OOVs or padding — assumes token_ids are valid.
    """

    def __init__(self, tokenizer: AutoTokenizer, d_model: int):
        super().__init__()
        self.tokenizer = tokenizer

        # Get vocab size from tokenizer
        vocab_dict = self.tokenizer.tokenizer.get_vocab()
        self.vocab_size = max(vocab_dict.values()) + 1  # +1 for safety
        self.d_model = d_model

        # Subword embedding table
        init_std = d_model**-0.5
        self.subword_emb = nn.Embedding(self.vocab_size, d_model)
        nn.init.normal_(self.subword_emb.weight, mean=0.0, std=init_std)

    def forward(self, token_ids: torch.LongTensor, subword_mask: torch.tensor = None):
        """
        token_ids: (B, T)
        subword_mask: (B, T)
        Returns:
            subword_embeds: (B, T, d_model)
        """
        return self.subword_emb(token_ids)


class CharAwareSubwordEncoder(nn.Module):
    """
    An encoder that creates subword embeddings from character-level embeddings.

    This module replaces a standard subword embedding layer. It breaks down each
    subword into its constituent characters, embeds the characters, and then
    aggregates these character embeddings (e.g., via mean pooling) to form the
    final subword representation. This allows the model to handle rare or out-of-vocabulary
    subwords more gracefully.

    Args:
        out_size (int): The dimensionality of the output embedding vectors.
        tokenizer (AutoTokenizer): The Hugging Face tokenizer class.
        vocab_dir (str | None): Directory to save/load the character vocabulary.
        backbone_type (str | None): The type of backbone model from Hugging Face (e.g., "t5gemma").
        backbone_model_class (str | None): The class name of the backbone model if not using AutoModel.
        backbone_config_class (str | None): The class name of the backbone config.
        backbone_config (DictConfig | None): A configuration for the backbone model.
    """

    def __init__(
        self,
        out_size: int,
        tokenizer: AutoTokenizer,
        vocab_dir: str | None = None,
        backbone_type: str | None = "t5gemma",
        backbone_model_class: str | None = None,
        backbone_config_class: str | None = None,
        backbone_config: DictConfig | None = None,
        use_subword_flag_emb: bool = True,
        use_bos_eos_emb: bool = True,
        use_cumulative_word_emb: bool = False,
    ):
        super().__init__()

        # 1. Build or load the character vocabulary
        self.subword_id_to_char_ids, self.char_vocab, self.subword_padding_idx = build_vocabs(
            tokenizer,
            vocab_dir,
        )

        self.char_padding_idx = len(self.char_vocab)
        self.use_subword_flag_emb = use_subword_flag_emb
        self.use_bos_eos_emb = use_bos_eos_emb

        # 2. Initialize the backbone model
        if backbone_type:
            config = AutoConfig.for_model(
                backbone_type, **(OmegaConf.to_container(backbone_config, resolve=True) if backbone_config else {})
            )
            self.backbone = AutoModelForTextEncoding.from_config(config)
        else:
            assert backbone_model_class and backbone_config_class
            config_class = getattr(transformers, backbone_config_class)
            model_class = getattr(transformers, backbone_model_class)
            config = config_class(**(OmegaConf.to_container(backbone_config, resolve=True) if backbone_config else {}))
            self.backbone = model_class(config)

        self.hidden_size = self.backbone.get_input_embeddings().weight.size(-1)

        # 3. Delete the original subword embedding layer and replace it with our character embedding layer
        find_and_delete_module(self.backbone, self.backbone.get_input_embeddings(), "backbone")
        self.embed_tokens = nn.Embedding(len(self.char_vocab) + 1, self.hidden_size, padding_idx=self.char_padding_idx)
        self.proj_embedding = nn.Linear(self.hidden_size, out_size, bias=False)

        if self.use_subword_flag_emb:
            self.subword_flag_emb = SubwordFlagEmbedding(tokenizer, self.hidden_size)

        if self.use_bos_eos_emb:
            self.bos_eos_emb = BOSEOSEmbedding(tokenizer, self.hidden_size)

    def prepare_inputs(self, subword_ids: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Converts a batch of subword IDs into a padded batch of character IDs.

        Args:
            subword_ids (Tensor): A tensor of subword IDs. Shape: `[batch, seq_len]`.
            padding_mask (Tensor): A boolean mask indicating valid (non-padding) subwords.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - Padded character IDs. Shape: `[num_valid_subwords, max_char_len]`.
                - Lengths of each character sequence. Shape: `[num_valid_subwords]`.
        """
        device = subword_ids.device
        # Select only the valid subword IDs
        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        # Map each subword ID to its sequence of character IDs
        char_id_list = [list(self.subword_id_to_char_ids.get(x, ())) for x in subword_id_list]

        char_lengths = torch.tensor([len(x) for x in char_id_list], dtype=torch.long, device=device)
        batch_size = char_lengths.size(0)
        max_len = int(char_lengths.max().item()) if batch_size > 0 else 0

        # Create a padded tensor for the character IDs
        char_ids = torch.full((batch_size, max_len), self.char_padding_idx, dtype=torch.long, device=device)
        for i, char_seq in enumerate(char_id_list):
            char_ids[i, : len(char_seq)] = torch.tensor(char_seq, dtype=torch.long, device=device)

        return char_ids, char_lengths

    def forward(self, subword_ids: Tensor, subword_mask: Tensor | None = None) -> Tensor:
        """
        Performs the forward pass to get character-aware subword embeddings.

        Args:
            subword_ids (Tensor): A tensor of subword IDs. Shape: `[batch, seq_len]`.
            subword_mask (Tensor | None): A boolean mask for padding. Defaults to None.

        Returns:
            Tensor: The final subword embeddings. Shape: `[batch, seq_len, hidden_size]`.
        """
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids, dtype=torch.bool)

        # 1. Convert subword IDs to character IDs
        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)

        # char_mask = sequence_mask(char_lengths).float()
        char_mask = sequence_mask(char_lengths)

        # 2. Get character embeddings and pass them through the backbone
        char_embeds = self.embed_tokens(char_ids)

        # The backbone model should be able to accept `inputs_embeds`
        char_hidden_states = self.backbone(inputs_embeds=char_embeds, attention_mask=char_mask).last_hidden_state

        # 3. Aggregate character embeddings to form subword embeddings (mean pooling)
        # We mask the padding characters before summing to get a correct mean.
        masked_sum = (char_hidden_states * char_mask.unsqueeze(-1)).sum(dim=1)
        # Avoid division by zero for empty sequences
        mean_emb = masked_sum / (char_lengths.unsqueeze(-1).clamp(min=1))

        # 4. Scatter the aggregated embeddings back to the original subword sequence shape
        out_emb = self.proj_embedding(mean_emb)
        subword_embeds = torch.zeros(
            subword_ids.shape + (out_emb.size(-1),), device=subword_ids.device, dtype=out_emb.dtype
        )
        subword_embeds[subword_mask] = out_emb

        if self.use_subword_flag_emb:
            subword_embeds = self.subword_flag_emb(subword_embeds, subword_ids)

        if self.use_bos_eos_emb:
            subword_embeds = self.bos_eos_emb(subword_embeds, subword_ids)

        return subword_embeds


class GatedProjectedSumRMSNorm(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim, final_norm=True, num_codebooks=31, init_residual_scale=0.5):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        nn.init.normal_(self.audio_proj.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.audio_proj.bias)
        nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.text_proj.bias)

        # FP32 gate params
        self.gate = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(init_residual_scale, dtype=torch.float32))

        self.final_norm = RMSNorm(hidden_dim) if final_norm else nn.Identity()

    def forward(self, audio_emb, text_emb):
        audio_emb = audio_emb / self.num_codebooks

        # projections run in model dtype (BF16)
        audio_h = self.audio_proj(audio_emb)
        text_h = self.text_proj(text_emb)

        dtype = audio_h.dtype

        with fp32_precision():
            gate = torch.sigmoid(self.gate)  # FP32
            res = torch.sigmoid(self.residual_scale)  # FP32

        h = gate.to(dtype) * audio_h + (1 - gate).to(dtype) * text_h
        h = res.to(dtype) * h
        h = self.final_norm(h.float()).to(dtype)

        return h


class RVQEARTTSModel(nn.Module):
    """
    Main RVQEARTTS model for training and inference.

    This model integrates a character-aware text encoder with a transformer backbone
    and a Mixture-of-Gaussians (MoG) prediction head.

    The architecture is based on the Streaming TTS model proposed in
    "Audio Flamingo 3" (https://arxiv.org/abs/2507.08128), with several improvements:

        1. Gated fusion of text and audio representations
           (`GatedProjectedSumRMSNorm`).

        2. Subword-aware embeddings for improved pronunciation of multi-token words
           (`SubwordFlagEmbedding`).

        3. Custom BOS and EOS embeddings for duplex interaction support,
           enabling interruption-aware generation (`BOSEOSEmbedding`).

    Args:
        config (DictConfig | dict[str, Any]): The configuration object for the model.
    """

    config_class = DictConfig
    rvq_embs: Tensor

    def __init__(self, config: DictConfig | dict[str, Any], tokenizer: AutoTokenizer = None):
        super().__init__()
        self.config = config

        # Backbone module
        if self.config.get("pretrained_text_name", None):
            # Load pretrained backbone from huggingface
            from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf

            llm = load_pretrained_hf(self.config.pretrained_text_name, pretrained_weights=True).train()
            self.backbone = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        else:
            if self.config.get("backbone_type", None) is None:
                assert (
                    self.config.get("backbone_model_class", None) is not None
                    and self.config.get("backbone_config_class", None) is not None
                )
                backbone_config = getattr(transformers, self.config.backbone_config_class)(
                    **(
                        OmegaConf.to_container(self.config.backbone_config, resolve=True)
                        if self.config.backbone_config
                        else {}
                    ),
                )
                self.backbone = getattr(transformers, self.config.backbone_model_class)(backbone_config)
            else:
                backbone_config = AutoConfig.for_model(
                    self.config.backbone_type,
                    **(
                        OmegaConf.to_container(self.config.backbone_config, resolve=True)
                        if self.config.backbone_config
                        else {}
                    ),
                )
                self.backbone = AutoModel.from_config(backbone_config)

        self.hidden_size = self.backbone.get_input_embeddings().weight.size(-1)
        find_and_delete_module(self.backbone, self.backbone.get_input_embeddings(), "backbone")

        # Embedding and projection layers
        self.bos_emb = nn.Parameter(torch.randn(self.hidden_size))
        self.null_emb = nn.Parameter(torch.randn(self.hidden_size))
        if self.config.random_target_masking:
            self.embed_target_mask = nn.Embedding(self.config.num_quantizers, self.hidden_size)

        self.embed_code = nn.Linear(self.config.latent_size, self.hidden_size, bias=False)

        self.embed_context = (
            nn.Linear(self.config.context_hidden_size, self.hidden_size, bias=False)
            if self.config.context_hidden_size
            else None
        )

        self.embed_subword = (
            CharAwareSubwordEncoder(
                tokenizer=tokenizer,
                out_size=self.hidden_size,
                use_subword_flag_emb=self.config.use_subword_flag_emb,
                use_bos_eos_emb=self.config.use_bos_eos_emb,
                **self.config.cas_config,
            )
            if self.config.cas_config
            else None
        )

        if self.config.use_gated_fusion_for_text_audio:
            self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
                self.hidden_size, self.hidden_size, self.hidden_size, self.config.num_quantizers
            )

        # Prediction Heads
        if not self.config.disable_eos_prediction:
            self.lm_head = nn.Linear(self.hidden_size, 2, bias=False)

        self.mog_head = MoGHead(
            hidden_size=self.hidden_size,
            out_size=self.config.latent_size,
            **self.config.mog_head_config,
        )

    def set_rvq_embs(self, rvq_embs: Tensor):
        self.register_buffer("rvq_embs", rvq_embs.detach().clone())

    def depthsum_embedding(self, code: Tensor) -> Tensor:
        """
        code: [b, t, d]
        rvq_embs: [d, v, h]

        ret: [b, t, h]
        """
        b, t, d = code.size()
        _, v, h = self.rvq_embs.size()
        device = code.device

        ret = torch.zeros((b, t, h), device=device)
        embs = F.pad(self.rvq_embs, [0, 0, 0, 1])
        for i in range(d):
            emb = embs[i]
            ret = ret + F.embedding(code[..., i], emb)
        return ret

    def prepare_training_inputs(self, code: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Prepares masked and dropped-out versions of the code for training."""
        b, t, d = code.size()
        device = code.device

        src_rate = torch.rand((b, t), device=device) * self.config.max_training_rate
        src_masking_rate = get_masking_rate(src_rate, self.config.exponent)
        src_num_masking = torch.ceil(src_masking_rate * self.config.num_quantizers).long()

        src_code_mask = torch.ones((b, t, d), dtype=torch.bool, device=device)
        src_code_mask = get_mask(src_code_mask, src_num_masking)
        src_masked_code = code * src_code_mask + (torch.zeros_like(code) + self.config.codebook_size) * (
            ~src_code_mask
        )

        if self.config.random_target_masking:
            tgt_rate = src_rate + (1.0 - src_rate) * torch.rand((b, t), device=device)
            tgt_masking_rate = get_masking_rate(tgt_rate, self.config.exponent)
            tgt_num_masking = torch.floor(tgt_masking_rate * self.config.num_quantizers).long()

            tgt_code_mask = torch.ones((b, t, d), dtype=torch.bool, device=device)
            tgt_code_mask = get_mask(tgt_code_mask, tgt_num_masking)
            tgt_masked_code = code * tgt_code_mask + (torch.zeros_like(code) + self.config.codebook_size) * (
                ~tgt_code_mask
            )
        else:
            tgt_code_mask = torch.ones((b, t, d), dtype=torch.bool, device=device)
            tgt_masked_code = code

        dropout_mask = torch.where(
            torch.rand((b, t, 1), device=device) < self.config.quantizer_dropout,
            (torch.randint(0, self.config.num_quantizers + 1, (b, t, 1), device=device)),
            self.config.num_quantizers,
        ) > torch.arange(d, dtype=torch.long, device=device)
        dropped_code = code * dropout_mask + (torch.zeros_like(code) + self.config.codebook_size) * (~dropout_mask)

        return src_masked_code, src_code_mask, tgt_masked_code, tgt_code_mask, dropped_code

    def _prepare_conditioning(
        self,
        context_hidden_state: Tensor | None,
        subword_ids: Tensor | None,
        subword_mask: Tensor | None,
        uncond_dec_flag: Tensor,
        asr_speech_tokens_emb: Tensor | None,
    ) -> Tensor:
        """Computes the final conditioning tensor by combining all sources."""
        cond = torch.zeros((1, 1, self.hidden_size), device=uncond_dec_flag.device)

        if self.embed_context is not None and context_hidden_state is not None:
            cond = cond + self.embed_context(context_hidden_state)

        if self.embed_subword is not None and subword_ids is not None:
            # Infer subword mask from context if not provided
            if subword_mask is None and context_hidden_state is not None:
                subword_mask = torch.any(context_hidden_state != 0, dim=-1)
            # at least one value should be true, otherwise we can completly skip it to avoid errors
            if subword_mask is not None and subword_mask.any():
                cond = cond + self.embed_subword(subword_ids, subword_mask)

        if asr_speech_tokens_emb is not None:
            cond = cond + asr_speech_tokens_emb

        # Replace with null embedding for unconditional generation
        cond = torch.where(uncond_dec_flag, self.null_emb, cond)
        return cond

    def _compute_losses(
        self,
        code: Tensor,
        lm_logits: Tensor,
        mog_logits: Tensor,
        mog_mus: Tensor,
        mog_mu_res: Tensor,
        mog_logs: Tensor,
        src_code_mask: Tensor,
        tgt_code_mask: Tensor,
        audio_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Helper to compute all losses for the training step."""
        with torch.autocast(code.device.type, enabled=False):
            # 1. LM Loss (predicting discrete tokens)
            if not self.config.disable_eos_prediction:
                eos_mask = (~audio_mask) & F.pad(audio_mask[:, :-1], [1, 0])
                lm_mask = eos_mask | audio_mask
                lm_target = torch.where(eos_mask, 1, 0)
                lm_loss = (
                    F.cross_entropy(lm_logits.transpose(1, 2), lm_target, reduction="none") * lm_mask
                ).sum() / lm_mask.sum().clamp_min(1)
            else:
                lm_loss = 0.0

            # 2. Continuous & KL Losses (for the MoG head)
            target_mask = (~src_code_mask & tgt_code_mask) & audio_mask.unsqueeze(-1)
            reduced_target_mask = target_mask.any(dim=-1)

            cont_code_target = self.depthsum_embedding(
                code * target_mask + (torch.zeros_like(code) + self.config.codebook_size) * (~target_mask)
            )
            mog_logits = mog_logits.float()
            mog_mus = mog_mus.float()
            mog_mu_res = mog_mu_res.float()
            mog_logs = mog_logs.float()

            # Log probability of the true code under each Gaussian component
            logp_code = (-0.5 * math.log(2 * math.pi) - mog_logs) * self.config.latent_size - 0.5 * self.mog_head.dist(
                mog_mus, (cont_code_target - mog_mu_res) * torch.exp(-mog_logs)
            )

            # Compute posterior q(k|c)
            q_kc = (
                torch.softmax(
                    logp_code,
                    -1,
                )
                * (1 - self.config.label_smoothing)
                + self.config.label_smoothing / self.mog_head.num_predictions
            ).detach()
            log_q_kc = torch.log(q_kc + 1e-8).detach()

            #  Continuous Loss (negative log-likelihood)
            c_loss = (-(q_kc * logp_code).sum(-1) * reduced_target_mask).sum() / target_mask.sum().clamp_min(1)

            # KL Divergence Loss
            k_loss = (
                (q_kc * (log_q_kc - F.log_softmax(mog_logits, -1))).sum(-1) * reduced_target_mask
            ).sum() / target_mask.sum().clamp_min(1)

        return lm_loss, c_loss, k_loss

    def forward(
        self,
        code: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        context_hidden_state: Tensor | None = None,
        subword_ids: Tensor | None = None,
        subword_mask: Tensor | None = None,
        audio_mask: Tensor | None = None,
        non_prompt_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        training: bool | None = None,
        guidance_enabled: bool = False,
        generation_config: dict[str, Any] | None = None,
        teacher_forcing_inference: bool = False,
        ignore_eos_flag_stop: bool = False,
        asr_speech_tokens_emb: Tensor | None = None,
    ) -> RVQEARTTSOutput:
        """
        Performs a forward pass handling training, generation, or single-step inference.

        Args:
            code (Tensor): Input audio codes. For training, this is the ground truth.
                           For generation, this is the previously generated code token.
            attention_mask (Tensor | None): Attention mask for the backbone transformer.
            position_ids (Tensor | None): Position ids for the backbone transformer.
            context_hidden_state (Tensor | None): Conditioning from a language model.
            subword_ids (Tensor | None): Subword token IDs for conditioning.
            subword_mask (Tensor | None): Mask for subword IDs.
            audio_mask (Tensor | None): Mask for valid audio positions (for training and inference initialization).
            past_key_values (Cache | None): Cache for past key-values for fast decoding.
            use_cache (bool): If True, returns the updated `past_key_values`.
            training (bool | None): Explicitly set training mode. If `None`, uses `self.training`.
            guidance_enabled (bool): If True, duplicates inputs internally to run both
                                     conditional and unconditional passes
            generation_config (dict[str, Any] | None): If provided, triggers an iterative code generation.

        Returns:
            RVQEARTTSOutput: A dataclass containing losses (for training) or generated outputs
                          and the cache (for inference).
        """

        # Determine operating mode.
        if training is None:
            training = self.training

        if audio_mask is not None:
            if training:
                (src_masked_code, src_code_mask, tgt_masked_code, tgt_code_mask, dropped_code) = (
                    self.prepare_training_inputs(code)
                )
                uncond_dec_flag = torch.rand(code.size(0), 1, 1, device=code.device) < self.config.p_uncond
            else:
                dropped_code = code
                uncond_dec_flag = torch.zeros(code.size(0), 1, 1, device=code.device, dtype=torch.bool)

            code_embeds = (
                self.embed_code(self.depthsum_embedding(F.pad(dropped_code[:, :-1], [0, 0, 1, 0])))
                + (audio_mask & (~F.pad(audio_mask[:, :-1], [1, 0]))).unsqueeze(-1) * self.bos_emb
            )

        else:  # Inference
            code_embeds = self.embed_code(self.depthsum_embedding(code))
            uncond_dec_flag = torch.zeros(code.size(0), 1, 1, device=code.device, dtype=torch.bool)

        if guidance_enabled:
            assert not training, "Classifier-free guidance can only be used when `training` is False."
            code_embeds = torch.cat([code_embeds] * 2, 0)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask] * 2, 0)
            if position_ids is not None:
                position_ids = torch.cat([position_ids] * 2, 0)
            if context_hidden_state is not None:
                context_hidden_state = torch.cat([context_hidden_state] * 2, 0)
            if subword_ids is not None:
                subword_ids = torch.cat([subword_ids] * 2, 0)
                if subword_mask is not None:
                    subword_mask = torch.cat([subword_mask] * 2, 0)
            if asr_speech_tokens_emb is not None:
                asr_speech_tokens_emb = torch.cat([asr_speech_tokens_emb] * 2, 0)

            uncond_dec_flag = torch.cat([uncond_dec_flag, torch.ones_like(uncond_dec_flag)], 0)

        # Prepare conditioning
        cond = self._prepare_conditioning(
            context_hidden_state,
            subword_ids,
            subword_mask,
            uncond_dec_flag,
            asr_speech_tokens_emb=asr_speech_tokens_emb,
        )

        if self.config.use_gated_fusion_for_text_audio:
            inputs_embeds = self.gated_fusion_audio_text(code_embeds, cond)
        else:
            inputs_embeds = code_embeds + cond

        # Main backbone pass
        backbone_outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = backbone_outputs.last_hidden_state

        if audio_mask is not None and training:
            # --- Training-specific loss computation ---
            if not self.config.disable_eos_prediction:
                lm_logits = self.lm_head(hidden_states)
            else:
                lm_logits = None

            mog_input_embeds = self.embed_code(self.depthsum_embedding(src_masked_code))
            if self.config.random_target_masking:
                mog_input_embeds = mog_input_embeds + self.embed_target_mask((tgt_code_mask.sum(-1) - 1).clamp_min(0))
            mog_input_embeds = mog_input_embeds + hidden_states
            mog_logits, mog_mus, mog_mu_res, mog_logs = self.mog_head(mog_input_embeds)

            lm_loss, c_loss, k_loss = self._compute_losses(
                code, lm_logits, mog_logits, mog_mus, mog_mu_res, mog_logs, src_code_mask, tgt_code_mask, audio_mask
            )
            total_loss = lm_loss + c_loss + k_loss

            return RVQEARTTSOutput(
                loss=total_loss, lm_loss=lm_loss, c_loss=c_loss, k_loss=k_loss, hidden_states=hidden_states
            )
        else:  # Inference
            if not generation_config:
                return RVQEARTTSOutput(
                    hidden_states=hidden_states,
                    past_key_values=backbone_outputs.past_key_values,
                )
            else:
                if teacher_forcing_inference:
                    generated_codes, lm_logits, eos_flag = self.generate_teacher_forcing(
                        hidden_states, generation_config
                    )
                else:
                    generated_codes, lm_logits, eos_flag = self.generate_step(
                        hidden_states, ignore_eos_flag_stop=ignore_eos_flag_stop, **generation_config
                    )
                return RVQEARTTSOutput(
                    past_key_values=backbone_outputs.past_key_values,
                    codes=generated_codes,
                    lm_logits=lm_logits,
                    eos_flag=eos_flag,
                    hidden_states=hidden_states,
                )

    @torch.no_grad()
    def generate_teacher_forcing(self, hidden_states: Tensor, generation_config: dict):
        """
        Teacher-forcing wrapper for generate_step, processing all frames in parallel
        using a per-frame loop internally.

        Args:
            hidden_states: [B, T, H] hidden states
            generation_config: kwargs for self.generate_step()

        Returns:
            generated_codes: [B, T, ...] generated codes per frame
            lm_logits: [B, T, vocab_size] language model logits
            eos_flag: [B, T] boolean tensor indicating EOS
        """
        B, T, H = hidden_states.shape

        # Preallocate caches
        generated_codes_cache = []
        lm_logits_cache = []
        eos_flag_cache = []

        # Iterate over time steps (frames)
        for t in range(T):
            # extract one frame (as the original generate_step expects)
            frame_hidden = hidden_states[:, t, :]  # [B, H]

            # call original generate_step
            generated_codes, lm_logits, eos_flag = self.generate_step(
                frame_hidden.unsqueeze(1), **generation_config  # keep batch dim + frame dim
            )
            if generated_codes is not None:
                # store in cache
                generated_codes_cache.append(generated_codes)
                lm_logits_cache.append(lm_logits)
                eos_flag_cache.append(eos_flag)

        # Stack results along time dimension
        generated_codes = torch.stack(generated_codes_cache, dim=1)  # [B, T, ...]
        if not self.config.disable_eos_prediction:
            lm_logits = torch.stack(lm_logits_cache, dim=1)  # [B, T, vocab_size]
            eos_flag = torch.stack(eos_flag_cache, dim=1)  # [B, T]
        else:
            lm_logits = None
            eos_flag = None

        return generated_codes, lm_logits, eos_flag

    @torch.no_grad()
    def generate_step(
        self,
        hidden_states: Tensor,
        num_iter: int,
        guidance_scale: list[float] | float | None = None,
        top_p_or_k: list[float | int] | float | int | None = None,
        noise_scale: list[float] | float | None = None,
        exponent: float | None = None,
        eos_threshold: float | None = None,
        ignore_eos_flag_stop: bool = False,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """
        Performs the iterative unmasking process for a single generation step.

        This function takes the hidden state from the backbone transformer and generates
        codes through an iterative unmasking process.

        Args:
            hidden_states (Tensor): The hidden states from the backbone. If using CFG,
                                    this should be the combined [uncond, cond] tensor.
            num_iter (int): The number of unmasking iterations.
            guidance_scale (list[float] | float | None): The scale for Classifier-Free Guidance.
            top_p_or_k (ist[float | int] | float | int | None): The value for top-p or top-k sampling.
            noise_scale (list[float] | float | None): The scale of noise to add during MoG sampling.
            exponent (float | None): The exponent for the masking schedule.
            eos_threshold (float | None): The threshold for EOS prediction.

        Returns:
            tuple[Tensor | None, Tensor, Tensor]: A tuple containing:
                - the generated codes.
                - The logits from `lm_head`.
                - The EOS flag.
        """
        # 1. Preparation
        if guidance_scale is not None:
            if not isinstance(guidance_scale, list):
                guidance_scale = [guidance_scale] * (1 + num_iter)  # includes one step for `lm_head`
            assert len(guidance_scale) == 1 + num_iter
        if top_p_or_k is not None:
            if not isinstance(top_p_or_k, list):
                top_p_or_k = [top_p_or_k] * (1 + num_iter)  # includes one step for `lm_head`
            assert len(top_p_or_k) == 1 + num_iter
        if noise_scale is not None:
            if not isinstance(noise_scale, list):
                noise_scale = [noise_scale] * num_iter
            assert len(noise_scale) == num_iter
        if exponent is None:
            exponent = self.config.exponent

        if guidance_scale is not None:
            # The effective batch size is halved
            hidden_states, uncond_hidden_states = hidden_states.chunk(2, dim=0)
        else:
            uncond_hidden_states = hidden_states[:0, :0, :0]

        b, t, _ = hidden_states.size()
        d = self.config.num_quantizers
        device = hidden_states.device

        # 2. Predict the discrete part of the code
        if not self.config.disable_eos_prediction:
            if guidance_scale is not None:
                lm_logits = self.lm_head(hidden_states + guidance_scale[0] * (hidden_states - uncond_hidden_states))
            else:
                lm_logits = self.lm_head(hidden_states)
            if top_p_or_k is not None:
                lm_logits = (
                    TopPLogitsWarper(top_p_or_k[0])(
                        None,
                        lm_logits.view(-1, lm_logits.size(-1)),
                    ).view_as(lm_logits)
                    if isinstance(top_p_or_k[0], float)
                    else TopKLogitsWarper(top_p_or_k[0])(
                        None,
                        lm_logits.view(-1, lm_logits.size(-1)),
                    ).view_as(lm_logits)
                )
            lm_logits = F.log_softmax(lm_logits, -1)
            if eos_threshold is not None:
                eos_flag = lm_logits[..., -1] > eos_threshold
            else:
                eos_flag = lm_logits.argmax(-1) == 1

            if torch.all(eos_flag) and not ignore_eos_flag_stop:
                return None, lm_logits, eos_flag
        else:
            lm_logits = None
            eos_flag = None

        # Initialize the full code tensor
        code = torch.zeros((b, t, d), dtype=torch.long, device=device) + self.config.codebook_size

        # 3. Set up the iterative denoising schedule for the continuous part
        rates = torch.linspace(0.0, 1.0, num_iter + 1, device=device)[:-1].unsqueeze(-1)
        masking_rates = get_masking_rate(rates, exponent=exponent)
        num_maskings = torch.ceil(masking_rates * self.config.num_quantizers).long()

        ks = num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])

        # 4. Iteratively unmask the continuous part of the code
        cnt = 0
        for i, k in enumerate(ks):
            if torch.all(k == 0):
                continue

            # Prepare input for the MoG head
            guidance_scale_i = guidance_scale[i] if guidance_scale is not None else 0.0
            top_p_or_k_i = top_p_or_k[i] if top_p_or_k is not None else 1.0
            noise_scale_i = noise_scale[i] if noise_scale is not None else 1.0

            mog_input_embeds = self.embed_code(self.depthsum_embedding(code))
            if self.config.random_target_masking:
                mog_input_embeds += self.embed_target_mask(cnt + k - 1)
            if guidance_scale_i > 0.0:
                mog_input_embeds = torch.cat(
                    [mog_input_embeds + hidden_states, mog_input_embeds + uncond_hidden_states], 0
                )
            else:
                mog_input_embeds += hidden_states

            mog_mu, mog_logs = self.mog_head.infer(
                mog_input_embeds,
                guidance_scale=guidance_scale_i,
                top_p_or_k=top_p_or_k_i,
            )
            z = mog_mu + torch.exp(mog_logs) * torch.randn_like(mog_mu) * noise_scale_i
            code = depthsum_encoding_step(self.rvq_embs, z, code, cnt, k[0].item())
            cnt += k[0].item()
        return code, lm_logits, eos_flag

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info("Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
