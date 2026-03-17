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

from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel  # noqa: F401
from nemo.collections.asr.models.asr_model import ASRModel  # noqa: F401
from nemo.collections.asr.models.classification_models import (  # noqa: F401
    ClassificationInferConfig,
    EncDecClassificationModel,
    EncDecFrameClassificationModel,
)
from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer  # noqa: F401
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE  # noqa: F401
from nemo.collections.asr.models.ctc_models import EncDecCTCModel  # noqa: F401
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel  # noqa: F401
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (  # noqa: F401
    EncDecHybridRNNTCTCBPEModelWithPrompt,
)
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel  # noqa: F401
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel  # noqa: F401
from nemo.collections.asr.models.multitalker_asr_models import EncDecMultiTalkerRNNTBPEModel  # noqa: F401
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel  # noqa: F401
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel  # noqa: F401
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel  # noqa: F401
from nemo.collections.asr.models.ssl_models import (  # noqa: F401
    EncDecDenoiseMaskedTokenPredModel,
    EncDecMaskedTokenPredModel,
    SpeechEncDecSelfSupervisedModel,
)
from nemo.collections.asr.models.transformer_bpe_models import EncDecTransfModelBPE  # noqa: F401

__all__ = [
    'ASRModel',
    'ClassificationInferConfig',
    'ClusteringDiarizer',
    'EncDecCTCModel',
    'EncDecCTCModelBPE',
    'EncDecClassificationModel',
    'EncDecDenoiseMaskedTokenPredModel',
    'EncDecFrameClassificationModel',
    'EncDecHybridRNNTCTCBPEModel',
    'EncDecHybridRNNTCTCBPEModelWithPrompt',
    'EncDecHybridRNNTCTCModel',
    'EncDecMaskedTokenPredModel',
    'EncDecMultiTaskModel',
    'EncDecMultiTalkerRNNTBPEModel',
    'EncDecRNNTBPEModel',
    'EncDecRNNTModel',
    'EncDecSpeakerLabelModel',
    'EncDecTransfModelBPE',
    'SortformerEncLabelModel',
    'SpeechEncDecSelfSupervisedModel',
]
