# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import enum
from typing import Dict, Optional

import torch
from torch import nn

from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, init_method_normal
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import ChannelType, NeuralType

try:
    from apex.transformer import parallel_state, tensor_parallel

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


__all__ = ["PromptEncoder", "PromptEncoderType"]


class PromptEncoderType(enum.Enum):
    TPMLP = "tpmlp"  # mlp model that support tensor parallel, better work together with a large language model
    MLP = "mlp"
    LSTM = "lstm"


class TPMLP(NeuralModule, Exportable):
    """
    The Tensor Parallel MLP prompt encoder network that is used to generate the virtual 
    token embeddings for p-tuning. It only have two layers.
    """

    def __init__(
        self, total_virtual_tokens: int, hidden_size: int, output_size: int, init_std: float,
    ):
        """
        Initializes the Tensor Model parallel MLP PromptEncoderMLP module.
        Args:
            total_virtual_tokens: the total number of vitural tokens
            hidden_size: hidden dimension
            output_size:  the output dimension
            init_std: the MLP init std value 
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_virtual_tokens = total_virtual_tokens
        self.activation = "gelu"

        sequence_parallel = False
        gradient_accumulation_fusion = False
        no_async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )
        self.first = tensor_parallel.ColumnParallelLinear(
            self.output_size,
            self.hidden_size,
            gather_output=False,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            use_cpu_initialization=False,
            bias=True,
            sequence_parallel_enabled=sequence_parallel,
            no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        self.second = tensor_parallel.RowParallelLinear(
            self.hidden_size,
            self.output_size,
            input_is_parallel=True,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            use_cpu_initialization=False,
            bias=True,
            sequence_parallel_enabled=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )

    def forward(self, input_embeds) -> torch.Tensor:
        intermediate_parallel, bias_parallel = self.first(input_embeds)
        intermediate_parallel = fused_bias_gelu(intermediate_parallel, bias_parallel)
        output_embeds, bias_parallel = self.second(intermediate_parallel)
        output_embeds = output_embeds + bias_parallel
        return output_embeds


class PromptEncoder(NeuralModule, Exportable):
    """
    The prompt encoder network that is used to generate the virtual 
    token embeddings for p-tuning.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "taskname_embeddings": NeuralType(("B", "T", "C"), ChannelType(), optional=False),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(("B", "T", "C"), ChannelType())}

    def __init__(
        self,
        encoder_type: enum,
        total_virtual_tokens: int,
        token_dim: int,
        hidden_size,
        lstm_dropout: float,
        num_layers: int,
        init_std: float,
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            total_virtual_tokens: the total number of vitural tokens
            hidden_size: hidden dimension
            lstm_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
            init_std: used for TPMLP encoder type to initialize the mlp weights
        """
        super().__init__()
        self.token_dim = token_dim
        self.input_size = token_dim
        self.output_size = token_dim
        self.hidden_size = hidden_size
        self.lstm_hidden_size = self.hidden_size // 2
        self.total_virtual_tokens = total_virtual_tokens
        self.encoder_type = encoder_type
        self.activation = "gelu"
        self.init_std = init_std

        # Set fixed indicies for forward pass
        self.register_buffer("indices", torch.LongTensor(list(range(self.total_virtual_tokens))))

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)

        if self.encoder_type == PromptEncoderType.LSTM:
            # LSTM
            self.lstm_head = torch.nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                bidirectional=True,
                batch_first=True,
            )

            self.mlp_head = nn.Sequential(
                nn.Linear(self.lstm_hidden_size * 2, self.lstm_hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.lstm_hidden_size * 2, self.output_size),
            )

        elif self.encoder_type == PromptEncoderType.MLP:
            if num_layers <= 1:
                raise ValueError(
                    "The MLP prompt encoder must have at least 2 layers, and exactly 2 layers is recommended."
                )

            layers = [nn.Linear(self.input_size, self.hidden_size), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()])

            layers.append(nn.Linear(self.hidden_size, self.output_size))
            self.mlp_head = nn.Sequential(*layers)

        elif self.encoder_type == PromptEncoderType.TPMLP:
            self.tpmlp = TPMLP(self.total_virtual_tokens, self.hidden_size, self.output_size, self.init_std,)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    @typecheck()
    def forward(self, taskname_embeddings) -> torch.Tensor:
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        batch_size, task_seq_length, _ = taskname_embeddings.shape
        input_embeds = input_embeds.expand(batch_size, self.total_virtual_tokens, self.token_dim).clone()
        length = min(task_seq_length, self.total_virtual_tokens)

        # Replace general input with task specific embeddings to specify the correct task
        input_embeds[:, 0:length, :] = taskname_embeddings[:, 0:length, :]

        if self.encoder_type == PromptEncoderType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.TPMLP:
            output_embeds = self.tpmlp(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds
