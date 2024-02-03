# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.modules.transformer.bridge_encoders import BridgeEncoder
from nemo.collections.asr.modules.transformer.perceiver_encoders import PerceiverEncoder
from nemo.collections.asr.modules.transformer.transformer_bottleneck import (
    NeMoTransformerBottleneckConfig,
    NeMoTransformerBottleneckDecoderConfig,
    NeMoTransformerBottleneckEncoderConfig,
    TransformerBottleneckEncoderNM,
)
from nemo.collections.asr.modules.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.asr.modules.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithLanguageModel,
    EnsembleBeamSearchSequenceGenerator,
    GreedySequenceGenerator,
    TopKSequenceGenerator,
)
from nemo.collections.asr.modules.transformer.transformer_modules import AttentionBridge, TransformerEmbedding
from nemo.collections.asr.modules.transformer.transformer_utils import get_nemo_transformer
