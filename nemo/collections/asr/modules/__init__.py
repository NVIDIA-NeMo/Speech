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

from nemo.collections.asr.modules.asr_transformer_encoder import ASRTransformerEncoder
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor,
    CropOrPadSpectrogramAugmentation,
    MaskedPatchAugmentation,
    SpectrogramAugmentation,
)
from nemo.collections.asr.modules.beam_search_decoder import BeamSearchDecoderWithLM
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder, ConformerEncoderAdapter
from nemo.collections.asr.modules.conv_asr import (
    ConvASRDecoder,
    ConvASRDecoderClassification,
    ConvASRDecoderReconstruction,
    ConvASREncoder,
    ConvASREncoderAdapter,
    ECAPAEncoder,
    ParallelConvASREncoder,
    SpeakerDecoder,
)
from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint
from nemo.collections.asr.modules.lstm_decoder import LSTMDecoder
from nemo.collections.asr.modules.msdd_diarizer import MSDD_module
from nemo.collections.asr.modules.rnn_encoder import RNNEncoder
from nemo.collections.asr.modules.rnnt import (
    RNNTDecoder,
    RNNTDecoderJointSSL,
    RNNTJoint,
    SampledRNNTJoint,
    StatelessTransducerDecoder,
)
from nemo.collections.asr.modules.squeezeformer_encoder import SqueezeformerEncoder, SqueezeformerEncoderAdapter
from nemo.collections.asr.modules.ssl_modules import (
    ConformerMultiLayerFeatureExtractor,
    ConformerMultiLayerFeaturePreprocessor,
    ConvFeatureMaksingWrapper,
    MultiSoftmaxDecoder,
    RandomBlockMasking,
    RandomProjectionVectorQuantizer,
)
