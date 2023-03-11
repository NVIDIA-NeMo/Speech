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

from nemo.collections.nlp.data.data_utils import *
from nemo.collections.nlp.data.entity_linking.entity_linking_dataset import EntityLinkingDataset
from nemo.collections.nlp.data.information_retrieval.information_retrieval_dataset import (
    BertInformationRetrievalDataset,
)
from nemo.collections.nlp.data.language_modeling.l2r_lm_dataset import (
    L2RLanguageModelingDataset,
    TarredL2RLanguageModelingDataset,
)
from nemo.collections.nlp.data.language_modeling.lm_bert_dataset import (
    BertPretrainingDataset,
    BertPretrainingPreprocessedDataloader,
)
from nemo.collections.nlp.data.language_modeling.sentence_dataset import SentenceDataset, TarredSentenceDataset
from nemo.collections.nlp.data.machine_translation.machine_translation_dataset import (
    TarredTranslationDataset,
    TranslationDataset,
)
from nemo.collections.nlp.data.question_answering_squad.qa_dataset import SquadDataset
from nemo.collections.nlp.data.text2sparql.text2sparql_dataset import Text2SparqlDataset
from nemo.collections.nlp.data.text_normalization.decoder_dataset import TextNormalizationDecoderDataset
from nemo.collections.nlp.data.text_normalization.tagger_dataset import TextNormalizationTaggerDataset
from nemo.collections.nlp.data.text_normalization.test_dataset import TextNormalizationTestDataset
from nemo.collections.nlp.data.token_classification.token_classification_dataset import (
    BertTokenClassificationDataset,
    BertTokenClassificationInferDataset,
)
from nemo.collections.nlp.data.zero_shot_intent_recognition.zero_shot_intent_dataset import (
    ZeroShotIntentDataset,
    ZeroShotIntentInferenceDataset,
)
