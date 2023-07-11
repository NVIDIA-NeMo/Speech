# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from einops import rearrange
from torch import nn

from nemo.collections.tts.modules.submodules import ConditionalInput, ConvNorm
from nemo.collections.tts.parts.utils.helpers import binarize_attention_parallel


class AlignmentEncoder(torch.nn.Module):
    """
    Module for alignment text and mel spectrogram.

    Args:
        n_mel_channels: Dimension of mel spectrogram.
        n_text_channels: Dimension of text embeddings.
        n_att_channels: Dimension of model
        temperature: Temperature to scale distance by.
            Suggested to be 0.0005 when using dist_type "l2" and 15.0 when using "cosine".
        condition_types: List of types for nemo.collections.tts.modules.submodules.ConditionalInput.
        dist_type: Distance type to use for similarity measurement. Supports "l2" and "cosine" distance.
    """

    def __init__(
        self,
        n_mel_channels=80,
        n_text_channels=512,
        n_att_channels=80,
        temperature=0.0005,
        condition_types=[],
        dist_type="l2",
    ):
        super().__init__()
        self.temperature = temperature
        self.cond_input = ConditionalInput(n_text_channels, n_text_channels, condition_types)
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )
        if dist_type == "l2":
            self.dist_fn = self.get_euclidean_dist
        elif dist_type == "cosine":
            self.dist_fn = self.get_cosine_dist
        else:
            raise ValueError(f"Unknown distance type '{dist_type}'")

    @staticmethod
    def _apply_mask(inputs, mask, mask_value):
        if mask is None:
            return

        mask = rearrange(mask, "B T2 1 -> B 1 1 T2")
        inputs.data.masked_fill_(mask, mask_value)

    def get_dist(self, keys, queries, mask=None):
        """Calculation of distance matrix.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries and also can be used
                for ignoring unnecessary elements from keys in the resulting distance matrix (True = mask element, False = leave unchanged).
        Output:
            dist (torch.tensor): B x T1 x T2 tensor.
        """
        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        dist = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)

        self._apply_mask(dist, mask, float("inf"))

        return dist

    @staticmethod
    def get_euclidean_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        # B x C x T1 x T2
        distance = (queries_enc - keys_enc) ** 2
        # B x 1 x T1 x T2
        l2_dist = distance.sum(axis=1, keepdim=True)
        return l2_dist

    @staticmethod
    def get_cosine_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        cosine_dist = -torch.nn.functional.cosine_similarity(queries_enc, keys_enc, dim=1)
        cosine_dist = rearrange(cosine_dist, "B T1 T2 -> B 1 T1 T2")
        return cosine_dist

    @staticmethod
    def get_durations(attn_soft, text_len, spect_len):
        """Calculation of durations.

        Args:
            attn_soft (torch.tensor): B x 1 x T1 x T2 tensor.
            text_len (torch.tensor): B tensor, lengths of text.
            spect_len (torch.tensor): B tensor, lengths of mel spectrogram.
        """
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        durations = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), spect_len))
        return durations

    @staticmethod
    def get_mean_dist_by_durations(dist, durations, mask=None):
        """Select elements from the distance matrix for the given durations and mask and return mean distance.

        Args:
            dist (torch.tensor): B x T1 x T2 tensor.
            durations (torch.tensor): B x T2 tensor. Dim T2 should sum to T1.
            mask (torch.tensor): B x T2 x 1 binary mask for variable length entries and also can be used
                for ignoring unnecessary elements in dist by T2 dim (True = mask element, False = leave unchanged).
        Output:
            mean_dist (torch.tensor): B x 1 tensor.
        """
        batch_size, t1_size, t2_size = dist.size()
        assert torch.all(torch.eq(durations.sum(dim=1), t1_size))

        AlignmentEncoder._apply_mask(dist, mask, 0)

        # TODO(oktai15): make it more efficient
        mean_dist_by_durations = []
        for dist_idx in range(batch_size):
            mean_dist_by_durations.append(
                torch.mean(
                    dist[
                        dist_idx,
                        torch.arange(t1_size),
                        torch.repeat_interleave(torch.arange(t2_size), repeats=durations[dist_idx]),
                    ]
                )
            )

        return torch.tensor(mean_dist_by_durations, dtype=dist.dtype, device=dist.device)

    @staticmethod
    def get_mean_distance_for_word(l2_dists, durs, start_token, num_tokens):
        """Calculates the mean distance between text and audio embeddings given a range of text tokens.

        Args:
            l2_dists (torch.tensor): L2 distance matrix from Aligner inference. T1 x T2 tensor.
            durs (torch.tensor): List of durations corresponding to each text token. T2 tensor. Should sum to T1.
            start_token (int): Index of the starting token for the word of interest.
            num_tokens (int): Length (in tokens) of the word of interest.
        Output:
            mean_dist_for_word (float): Mean embedding distance between the word indicated and its predicted audio frames.
        """
        # Need to calculate which audio frame we start on by summing all durations up to the start token's duration
        start_frame = torch.sum(durs[:start_token]).data

        total_frames = 0
        dist_sum = 0

        # Loop through each text token
        for token_ind in range(start_token, start_token + num_tokens):
            # Loop through each frame for the given text token
            for frame_ind in range(start_frame, start_frame + durs[token_ind]):
                # Recall that the L2 distance matrix is shape [spec_len, text_len]
                dist_sum += l2_dists[frame_ind, token_ind]

            # Update total frames so far & the starting frame for the next token
            total_frames += durs[token_ind]
            start_frame += durs[token_ind]

        return dist_sum / total_frames

    def forward(self, queries, keys, mask=None, attn_prior=None, conditioning=None):
        """Forward pass of the aligner encoder.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries (True = mask element, False = leave unchanged).
            attn_prior (torch.tensor): prior for attention matrix.
            conditioning (torch.tensor): B x 1 x C2 conditioning embedding
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        keys = self.cond_input(keys.transpose(1, 2), conditioning).transpose(1, 2)
        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        distance = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)
        attn = -self.temperature * distance

        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)

        attn_logprob = attn.clone()

        self._apply_mask(attn, mask, -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob
