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

from nemo.collections.asr.models.configs.asr_models_config import (  # noqa: F401
    ASRDatasetConfig,
    CacheAwareStreamingConfig,
    EncDecCTCConfig,
    EncDecCTCModelConfig,
)
from nemo.collections.asr.models.configs.classification_models_config import (  # noqa: F401
    EncDecClassificationConfig,
    EncDecClassificationDatasetConfig,
    EncDecClassificationModelConfig,
)
from nemo.collections.asr.models.configs.matchboxnet_config import (  # noqa: F401
    EncDecClassificationModelConfigBuilder,
    MatchboxNetModelConfig,
    MatchboxNetVADModelConfig,
)
from nemo.collections.asr.modules.audio_preprocessing import (  # noqa: F401
    AudioToMelSpectrogramPreprocessorConfig,
    AudioToMFCCPreprocessorConfig,
    CropOrPadSpectrogramAugmentationConfig,
    SpectrogramAugmentationConfig,
)
from nemo.collections.asr.modules.conv_asr import (  # noqa: F401
    ConvASRDecoderClassificationConfig,
    ConvASRDecoderConfig,
    ConvASREncoderConfig,
    JasperEncoderConfig,
)
