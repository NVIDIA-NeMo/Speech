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
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from nemo.core.classes import Exportable, NeuralModule

try:
    from apex.transformer import tensor_parallel

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['VirtualPromptSource', 'VirtualPromptStyle', 'VirtualPromptPlaceholderToken']


class VirtualPromptStyle(enum.Enum):
    P_TUNING = 'p-tuning'
    NO_PROMPT = 'no-prompts'


class VirtualPromptSource(enum.Enum):
    PROMPT_ENCODER = 'prompt_encoder'
    NO_PROMPT = 'no-prompts'


class VirtualPromptPlaceholderToken(enum.Enum):
    BASE = '<prompt_'
    END = '>'
