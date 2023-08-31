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

import copy
import enum
from typing import Dict, Optional

import torch
import torch.nn.init as init
from torch import nn

from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, init_method_normal
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck

try:
    from megatron.core import ModelParallelConfig, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


__all__ = ["PromptEncoder", "PromptEncoderType"]


class PromptEncoderType(enum.Enum):
    TPMLP = "tpmlp"  # mlp model that support tensor parallel, better work together with a large language model
    MLP = "mlp"
    LSTM = "lstm"
    EMBEDDING = "embedding"


class PromptEmbedding(NeuralModule, Exportable):
    """Prompt embeddings

    Arugments:
        init_from_prompt_text: Whether to intialize prompt embeddings
                               from from certain lm embeddings
                               corresponding to a prompt string
        hidden_size: hidden size should match lm embedding size
        total_virtual_tokens: length of prompt initalized from torch init method
    """

    def __init__(
        self, hidden_size, total_virtual_tokens,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.total_virtual_tokens = total_virtual_tokens

        # Randomly init token and position embeddings
        self.prompt_embeddings = torch.nn.Embedding(self.total_virtual_tokens, self.hidden_size)
        self.prompt_embeddings.weight.data.fill_(0.0)
        self.prompt_embeddings.weight.requires_grad = False

        # Set fixed indicies for forward pass
        self.register_buffer("indices", torch.LongTensor(list(range(self.total_virtual_tokens))), persistent=False)

    def clear_prompt_embedding_weights(self,):
        """
        Method sets the prompt embedding weights to 0.0
        """
        self.prompt_embeddings.weight.fill_(0.0)

    def set_prompt_embedding_weights(self, weight: torch.Tensor):
        """
        Method sets the prompt embedding weights with a new weight w
        """
        self.prompt_embeddings.weight.data = weight.type_as(self.prompt_embeddings.weight.data)

    def forward(self,):
        """ 
        Does forward pass
        """
        return self.prompt_embeddings(self.indices)


class InferenceTable(NeuralModule, Exportable):
    """ 
    A wrapper class that holds the output representations of the PromptEncoder Model. 
    At inference time we do not need to forward pass through the full PromptEncoder and can just use this class.
    """

    def __init__(self, taskname, hidden_size, total_virtual_tokens, is_inference_ready=False):
        super().__init__()
        self.taskname = taskname
        self.hidden_size = hidden_size
        self.total_virtual_tokens = total_virtual_tokens
        self.prompt_table = torch.nn.ModuleDict()
        self.prompt_table[self.taskname] = PromptEmbedding(self.hidden_size, self.total_virtual_tokens)
        self.prompt_table[self.taskname].clear_prompt_embedding_weights()
        self.is_inference_ready = is_inference_ready
        for p in self.prompt_table.parameters():
            p.requires_grad = False

    def set_prompt_table(self, prompt_representation: torch.Tensor):
        """
        Method sets the prompt embedding inside self.prompt_table[taskname] with new weights
        """
        self.prompt_table[self.taskname].set_prompt_embedding_weights(prompt_representation)
        self.is_inference_ready = True

    def get_prompt_table(self,):
        """ 
        Returns the prompt representation cached in the prompt table
        """
        return self.prompt_table[self.taskname].forward()

    def clear_prompt_table(self,):
        """
        Method "clears" the prompt embedding inside self.prompt_table[taskname] by setting it to zero.
        """
        self.prompt_table[self.taskname].clear_prompt_embedding_weights()
        self.is_inference_ready = False


class TPMLP(NeuralModule, Exportable):
    """
    The Tensor Parallel MLP prompt encoder network that is used to generate the virtual 
    token embeddings for p-tuning. It only have two layers.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        total_virtual_tokens: int,
        hidden_size: int,
        output_size: int,
        init_std: float,
    ):
        """
        Initializes the Tensor Model parallel MLP PromptEncoderMLP module.
        Args:
            config: the model parallel config used my megatron core
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

        config = copy.deepcopy(config)
        config.sequence_parallel = False
        config.gradient_accumulation_fusion = False

        self.first = tensor_parallel.ColumnParallelLinear(
            self.output_size,
            self.hidden_size,
            config=config,
            gather_output=False,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            bias=True,
        )
        self.second = tensor_parallel.RowParallelLinear(
            self.hidden_size,
            self.output_size,
            config=config,
            input_is_parallel=True,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            bias=True,
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

    def __init__(
        self,
        config: ModelParallelConfig,
        encoder_type: enum,
        total_virtual_tokens: int,
        token_dim: int,
        hidden_size,
        lstm_dropout: float,
        num_layers: int,
        init_std: float,
        taskname: str = "taskname",
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            config: the model parallel config used my megatron core
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
        self.total_virtual_tokens = total_virtual_tokens
        self.encoder_type = encoder_type
        self.activation = "gelu"
        self.init_std = init_std
        self.taskname = taskname

        # Set fixed indicies for forward pass
        self.register_buffer("indices", torch.LongTensor(list(range(self.total_virtual_tokens))))

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        self.inference_table = InferenceTable(taskname, self.token_dim, self.total_virtual_tokens)

        if self.encoder_type == PromptEncoderType.EMBEDDING:
            init.xavier_normal_(self.embedding.weight)
        elif self.encoder_type == PromptEncoderType.LSTM:
            # LSTM
            self.lstm_head = torch.nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                dropout=lstm_dropout,
                bidirectional=True,
                batch_first=True,
            )

            self.mlp_head = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, self.output_size),
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
            self.tpmlp = TPMLP(config, self.total_virtual_tokens, self.hidden_size, self.output_size, self.init_std,)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    def set_inference_table(self, prompt_representation: torch.Tensor):
        """
        This method caches the output representation from the Encoder and saves it inside `self.inference_table`.
        """
        prompt_representation = prompt_representation.detach().clone()
        self.inference_table.set_prompt_table(prompt_representation)

    def clear_inference_table(self,):
        self.inference_table.clear_prompt_table()

    def get_inference_table(self,):
        return self.inference_table.get_prompt_table()

    def state_dict(self, desination=None, prefix=None, keep_vars=False):
        _state_dict = {}
        _state_dict[
            'prompt_table'
        ] = (
            self.inference_table.state_dict()
        )  # (@adithyare) this key is for backward compatibility with downstream users of the "inference ready" model.
        _state_dict['embeddings'] = self.embedding.state_dict()
        if self.encoder_type == PromptEncoderType.EMBEDDING:
            pass
        elif self.encoder_type == PromptEncoderType.LSTM:
            _state_dict['mlp_head'] = self.mlp_head.state_dict()
            _state_dict['lstm_head'] = self.lstm_head.state_dict()
        elif self.encoder_type == PromptEncoderType.MLP:
            _state_dict['mlp_head'] = self.mlp_head.state_dict()
        elif self.encoder_type == PromptEncoderType.TPMLP:
            _state_dict['tpmlp'] = self.tpmlp.state_dict()
        else:
            raise ValueError("Prompt encoder type not recognized. Pl.")
        return _state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.inference_table.load_state_dict(state_dict['prompt_table'])
        self.embedding.load_state_dict(state_dict['embeddings'])
        if self.encoder_type == PromptEncoderType.EMBEDDING:
            pass
        elif self.encoder_type == PromptEncoderType.LSTM:
            self.mlp_head.load_state_dict(state_dict['mlp_head'])
            self.lstm_head.state_dict(state_dict['lstm_head'])
        elif self.encoder_type == PromptEncoderType.MLP:
            self.mlp_head.load_state_dict(state_dict['mlp_head'])
        elif self.encoder_type == PromptEncoderType.TPMLP:
            self.tpmlp.load_state_dict(state_dict['tpmlp'])
        else:
            raise ValueError("Prompt encoder type not recognized. Pl.")
        return

    def _forward(self,):
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        if self.encoder_type == PromptEncoderType.EMBEDDING:
            output_embeds = input_embeds
        elif self.encoder_type == PromptEncoderType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        elif self.encoder_type == PromptEncoderType.TPMLP:
            output_embeds = self.tpmlp(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Pl.")
        return output_embeds

    @typecheck()
    def forward(self, batch_size: int, use_cached_reps: bool) -> torch.Tensor:
        """ 
        Forward pass through the encoder with caching of prompt representations
        """
        if use_cached_reps:
            output_embeds = self.get_inference_table().unsqueeze(0)
        else:
            if self.training:
                if self.inference_table.is_inference_ready:
                    self.clear_inference_table()
                output_embeds = self._forward()
            else:
                if not self.inference_table.is_inference_ready:
                    output_embeds = self._forward()
                    self.set_inference_table(output_embeds.squeeze(0))
                output_embeds = self.get_inference_table().unsqueeze(0)

        output_embeds = output_embeds.expand(batch_size, self.total_virtual_tokens, self.token_dim)
        return output_embeds
