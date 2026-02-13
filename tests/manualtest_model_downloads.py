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

"""
WARNING: Running this test will download ALL pre-trained NeMo models.
This is bandwidth and disk space consuming.
"""

import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
import nemo.collections.tts as nemo_tts


def testclass_downloads(cls, refresh_cache, model_names=None):
    for model_info in cls.list_available_models():
        model = cls.from_pretrained(model_name=model_info.pretrained_model_name, refresh_cache=refresh_cache)
        assert isinstance(model, cls)
    if model_names is not None:
        assert set(model_names) == set([m.pretrained_model_name for m in cls.list_available_models()])


for refresh_cache in [True, False]:
    # Test ASR collection
    testclass_downloads(
        nemo_asr.models.EncDecCTCModel,
        refresh_cache,
        [
            'QuartzNet15x5Base-En',
            'QuartzNet15x5Base-Zh',
            'QuartzNet5x5LS-En',
            'QuartzNet15x5NR-En',
            'Jasper10x5Dr-En',
        ],
    )
    testclass_downloads(nemo_asr.models.EncDecCTCModelBPE, refresh_cache, ['ContextNet-192-WPE-1024-8x-Stride'])
    testclass_downloads(
        nemo_asr.models.EncDecClassificationModel,
        refresh_cache,
        [
            'MatchboxNet-3x1x64-v1',
            'MatchboxNet-3x2x64-v1',
            'MatchboxNet-3x1x64-v2',
            'MatchboxNet-3x1x64-v2',
            'MatchboxNet-3x1x64-v2-subset-task',
            'MatchboxNet-3x2x64-v2-subset-task',
            'MatchboxNet-VAD-3x2',
        ],
    )
    testclass_downloads(
        nemo_asr.models.EncDecSpeakerLabelModel,
        refresh_cache,
        [
            'speakerrecognition_speakernet',
            'speakerverification_speakernet',
            'speakerdiarization_speakernet',
            'ecapa_tdnn',
        ],
    )

    # Test NLP collection
    testclass_downloads(nemo_nlp.models.TokenClassificationModel, refresh_cache, ['NERModel'])
    testclass_downloads(
        nemo_nlp.models.PunctuationCapitalizationModel,
        refresh_cache,
        ['Punctuation_Capitalization_with_BERT', 'Punctuation_Capitalization_with_DistilBERT'],
    )
    testclass_downloads(
        nemo_nlp.models.QAModel,
        refresh_cache,
        [
            'BERTBaseUncasedSQuADv1.1',
            'BERTBaseUncasedSQuADv2.0',
            'BERTLargeUncasedSQuADv1.1',
            'BERTLargeUncasedSQuADv2.0',
        ],
    )
    # testclass_downloads(nemo_nlp.models.IntentSlotClassificationModel, refresh_cache, ['Joint_Intent_Slot_Assistant'])

    # Test TTS collection
    testclass_downloads(nemo_tts.models.SqueezeWaveModel, refresh_cache, ['SqueezeWave-22050Hz'])
    testclass_downloads(nemo_tts.models.GlowTTSModel, refresh_cache, ['GlowTTS-22050Hz'])


print("############ THAT'S ALL FOLKS! ############")
