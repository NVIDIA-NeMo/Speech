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

import torch
import torch.nn.functional as F
from torch import nn

from nemo.core import NeuralModule
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import LabelsType, NeuralType, SpectrogramType


class RandomProjectionVectorQuantizer(NeuralModule, Exportable):
    DIST_FN_LIST = ["l2", "cosine"]

    def __init__(
        self,
        feat_in: int,
        code_dim: int,
        num_classes: int,
        num_books: int,
        dist_fn: str = "cosine",
        time_ahead: bool = False,
        freeze: bool = True,
        squeeze_single: bool = False,
        combine_time_steps: int = 1,
    ):
        """Vector quantization using random projection proposed in BEST-RQ paper:
        'Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition'

         Args:
            feat_in: input feature dimension
            code_dim: dimension of the codebook features
            num_classes: number of classes
            num_books: number of codebooks
            dist_fn: distance function to use, one of "l2" or "cosine"
            time_ahead: if Ture, the input is of shape (B, T, D), otherwise (B, D, T)
            freeze: whether to freeze the projection matrix
            squeeze_single: if True, squeeze codebook dimension if num_books is 1
        """
        super().__init__()

        if dist_fn not in self.DIST_FN_LIST:
            raise ValueError(f"Unknown distance function {dist_fn}, must be one of {self.DIST_FN_LIST}")

        self.feat_in = feat_in
        self.code_dim = code_dim
        self.num_classes = num_classes
        self.num_books = num_books
        self.dist_fn = dist_fn
        self.time_ahead = time_ahead
        self.squeeze_single = squeeze_single
        self.combine_time_steps = combine_time_steps

        # (B, T, D) -> (B, T, num_books, code_dim)
        self.proj = nn.Linear(self.feat_in * combine_time_steps, self.num_books * self.code_dim, bias=False)
        torch.nn.init.xavier_normal_(self.proj.weight)

        # (num_books, num_classes, hid_dim)
        codebooks = torch.randn(self.num_books, self.num_classes, self.code_dim).double()
        torch.nn.init.normal_(codebooks, mean=0, std=1)
        codebooks = F.normalize(codebooks, dim=-1)
        self.codebooks = nn.Parameter(codebooks)
        if freeze:
            self.freeze()

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        if self.time_ahead:
            return {"input_signal": NeuralType(('B', 'T', 'D'), SpectrogramType())}
        return {"input_signal": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        if self.time_ahead:
            if self.num_books == 1 and self.squeeze_single:
                return {
                    "xq": NeuralType(('B', 'T', 'D'), SpectrogramType()),
                    "xid": NeuralType(('B', 'T'), LabelsType()),
                }
            return {
                "xq": NeuralType(('B', 'T', 'D', 'H'), SpectrogramType()),
                "xid": NeuralType(('B', 'T', 'H'), LabelsType()),
            }
        if self.num_books == 1 and self.squeeze_single:
            return {
                "xq": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "xid": NeuralType(('B', 'T'), LabelsType()),
            }
        return {
            "xq": NeuralType(('B', 'D', 'T', 'H'), SpectrogramType()),
            "xid": NeuralType(('B', 'T', 'H'), LabelsType()),
        }

    @typecheck()
    def forward(self, input_signal):
        """
        Args:
            input_signal: input features of shape (B, T, D) or (B, D, T)
        Returns:
            xq: quantized features of shape (B, T, D, N) or (B, D, T, N)
            xid: quantized tokens of shape (B, T, N)
        """
        if not self.time_ahead:
            # (B, D, T) -> (B, T, D)
            input_signal = input_signal.transpose(1, 2)

        B, T, _ = input_signal.size()

        if self.combine_time_steps > 1:
            input_signal = input_signal.contiguous().reshape(B, T // self.combine_time_steps, -1)
            T = T // self.combine_time_steps

        # (B, T, D) -> (B, T, num_books*code_dim)
        x = self.proj(input_signal)

        # normalize each feature vector
        # (B, T, num_books*code_dim) -> (B, T, num_books, code_dim)
        x = F.normalize(x.view(B, T, self.num_books, self.code_dim), dim=-1)

        # get tokens (xid) of shape (B, T, num_books)
        if self.dist_fn == "cosine":
            # (B, T, num_books, code_dim) -> (B, T, num_books, num_classes)
            xid = torch.einsum('btdh,dch->btdc', x, self.codebooks)
            # (B, T, num_books, num_classes) -> (B, T, num_books)
            xid = xid.max(dim=-1)[1]
        elif self.dist_fn == "l2":
            # (B, T, num_books, code_dim) -> (B, T, num_books, code_dim, num_classes)
            xid = x.unsqueeze(-1) - self.codebooks.transpose(1, 2).unsqueeze(0).unsqueeze(0)
            xid = xid.norm(dim=-2).argmin(dim=-1)
        else:
            raise ValueError(f"Unknown distance function {self.dist_fn}, must be one of {self.DIST_FN_LIST}")

        # xid2: (B, T, num_books) -> (B, T, num_books)
        xid2 = xid + self.num_classes * torch.arange(self.num_books, device=xid.device).unsqueeze(0).unsqueeze(0)
        # xid2: (B, T, num_books) -> (B*num_books, T)
        xid2 = xid2.transpose(1, 2).contiguous().view(-1, T)

        # get quantized vector (xq) of shape (B, T, code_dim, num_books)
        # codebook: (num_books, num_classes, code_dim) -> (num_books*num_classes, code_dim)
        xq = F.embedding(xid2.view(-1), self.codebooks.view(-1, self.code_dim)).view(
            B, T, self.code_dim, self.num_books
        )

        if not self.time_ahead:
            # (B, T, D) -> (B, D, T)
            xq = xq.transpose(1, 2)

        if self.num_books == 1 and self.squeeze_single:
            xq = xq.squeeze(-1)
            xid = xid.squeeze(-1)

        return xq, xid
