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

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.audio_to_audio_model import AudioToAudioModel
from nemo.collections.asr.models.classification_models import EncDecClassificationModel, EncDecFrameClassificationModel
from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.enhancement_models import EncMaskDecAudioToAudioModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.models.k2_sequence_models import (
    EncDecK2RnntSeqModel,
    EncDecK2RnntSeqModelBPE,
    EncDecK2SeqModel,
    EncDecK2SeqModelBPE,
)
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel, NeuralDiarizer
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.models.slu_models import SLUIntentSlotBPEModel
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.models.transformer_bpe_models import EncDecTransfModelBPE
