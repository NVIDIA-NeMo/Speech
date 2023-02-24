# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.collections.nlp.modules.common.huggingface import (
    AlbertEncoder,
    BertEncoder,
    CamembertEncoder,
    DistilBertEncoder,
    RobertaEncoder,
)
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model, get_pretrained_lm_models_list
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder, PromptEncoderType
from nemo.collections.nlp.modules.common.prompt_table import (
    VirtualPromptPlaceholderToken,
    VirtualPromptSource,
    VirtualPromptStyle,
)
from nemo.collections.nlp.modules.common.sequence_classifier import SequenceClassifier
from nemo.collections.nlp.modules.common.sequence_regression import SequenceRegression
from nemo.collections.nlp.modules.common.sequence_token_classifier import SequenceTokenClassifier
from nemo.collections.nlp.modules.common.token_classifier import BertPretrainingTokenClassifier, TokenClassifier
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer, get_tokenizer_list
