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

from nemo.collections.asr.modules.audio_preprocessing import (  # noqa: F401
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor,
    CropOrPadSpectrogramAugmentation,
    MaskedPatchAugmentation,
    SpectrogramAugmentation,
)
from nemo.collections.asr.modules.beam_search_decoder import BeamSearchDecoderWithLM  # noqa: F401
from nemo.collections.asr.modules.conformer_encoder import (  # noqa: F401
    ConformerEncoder,
    ConformerEncoderAdapter,
    ConformerMultiLayerFeatureExtractor,
)
from nemo.collections.asr.modules.conv_asr import (  # noqa: F401
    ConvASRDecoder,
    ConvASRDecoderClassification,
    ConvASRDecoderReconstruction,
    ConvASREncoder,
    ConvASREncoderAdapter,
    ECAPAEncoder,
    ParallelConvASREncoder,
    SpeakerDecoder,
)
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint  # noqa: F401
from nemo.collections.asr.modules.lstm_decoder import LSTMDecoder  # noqa: F401
from nemo.collections.asr.modules.rnn_encoder import RNNEncoder  # noqa: F401
from nemo.collections.asr.modules.rnnt import (  # noqa: F401
    RNNTDecoder,
    RNNTDecoderJointSSL,
    RNNTJoint,
    SampledRNNTJoint,
    StatelessTransducerDecoder,
)
from nemo.collections.asr.modules.ssl_modules import (
    ConformerMultiLayerFeaturePreprocessor,
    ConvFeatureMaksingWrapper,
    MultiSoftmaxDecoder,
    RandomBlockMasking,
    RandomProjectionVectorQuantizer,
)

__all__ = [
    'AudioToMelSpectrogramPreprocessor',
    'AudioToMFCCPreprocessor',
    'CropOrPadSpectrogramAugmentation',
    'MaskedPatchAugmentation',
    'SpectrogramAugmentation',
    'BeamSearchDecoderWithLM',
    'ConformerEncoder',
    'ConformerEncoderAdapter',
    'ConformerMultiLayerFeatureExtractor',
    'ConvASRDecoder',
    'ConvASRDecoderClassification',
    'ConvASRDecoderReconstruction',
    'ConvASREncoder',
    'ConvASREncoderAdapter',
    'ECAPAEncoder',
    'ParallelConvASREncoder',
    'SpeakerDecoder',
    'HATJoint',
    'LSTMDecoder',
    'RNNTDecoder',
    'RNNTDecoderJointSSL',
    'RNNTJoint',
    'SampledRNNTJoint',
    'StatelessTransducerDecoder',
    'ConformerMultiLayerFeaturePreprocessor',
    'ConvFeatureMaksingWrapper',
    'MultiSoftmaxDecoder',
    'RandomBlockMasking',
    'RandomProjectionVectorQuantizer',
]
