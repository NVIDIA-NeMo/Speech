# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from nemo.collections.tts.losses.audio_codec_loss import MaskedMSELoss
from nemo.collections.tts.parts.utils.distributed import broadcast_tensors
from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import EncodedRepresentation, Index, LengthsType, LossType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental
