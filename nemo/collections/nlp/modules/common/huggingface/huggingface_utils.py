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

import os
from typing import List, Optional

from transformers import (
    AlbertConfig,
    AutoModel,
    BertConfig,
    CamembertConfig,
    DistilBertConfig,
    GPT2Config,
    RobertaConfig,
)

from nemo.collections.nlp.modules.common.huggingface.albert import AlbertEncoder
from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.collections.nlp.modules.common.huggingface.camembert import CamembertEncoder
from nemo.collections.nlp.modules.common.huggingface.distilbert import DistilBertEncoder
from nemo.collections.nlp.modules.common.huggingface.gpt2 import GPT2Encoder
from nemo.collections.nlp.modules.common.huggingface.roberta import RobertaEncoder
from nemo.utils import logging

__all__ = ["get_huggingface_lm_model", "get_huggingface_pretrained_lm_models_list", "VOCAB_FILE_NAME"]

# Manually specify the model archive lists since these are now removed in HF
# https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/deprecated/_archive_maps.py
ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert/albert-base-v1",
    "albert/albert-large-v1",
    "albert/albert-xlarge-v1",
    "albert/albert-xxlarge-v1",
    "albert/albert-base-v2",
    "albert/albert-large-v2",
    "albert/albert-xlarge-v2",
    "albert/albert-xxlarge-v2",
]

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "google-bert/bert-base-cased",
    "google-bert/bert-large-cased",
    "google-bert/bert-base-multilingual-uncased",
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-base-chinese",
    "google-bert/bert-base-german-cased",
    "google-bert/bert-large-uncased-whole-word-masking",
    "google-bert/bert-large-cased-whole-word-masking",
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
    "google-bert/bert-base-cased-finetuned-mrpc",
    "google-bert/bert-base-german-dbmdz-cased",
    "google-bert/bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
]
CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "almanach/camembert-base",
    "Musixmatch/umberto-commoncrawl-cased-v1",
    "Musixmatch/umberto-wikipedia-uncased-v1",
]

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
]
GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "distilbert/distilgpt2",
]
ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "FacebookAI/roberta-large-mnli",
    "distilbert/distilroberta-base",
    "openai-community/roberta-base-openai-detector",
    "openai-community/roberta-large-openai-detector",
]


HUGGINGFACE_MODELS = {
    "BertModel": {
        "default": "bert-base-uncased",
        "class": BertEncoder,
        "config": BertConfig,
        "pretrained_model_list": BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "DistilBertModel": {
        "default": "distilbert-base-uncased",
        "class": DistilBertEncoder,
        "config": DistilBertConfig,
        "pretrained_model_list": DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "CamembertModel": {
        "default": "camembert-base-uncased",
        "class": CamembertEncoder,
        "config": CamembertConfig,
        "pretrained_model_list": CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "RobertaModel": {
        "default": "roberta-base",
        "class": RobertaEncoder,
        "config": RobertaConfig,
        "pretrained_model_list": ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "AlbertModel": {
        "default": "albert-base-v2",
        "class": AlbertEncoder,
        "config": AlbertConfig,
        "pretrained_model_list": ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
    "GPT2Model": {
        "default": "gpt2",
        "class": GPT2Encoder,
        "config": GPT2Config,
        "pretrained_model_list": GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
    },
}

VOCAB_FILE_NAME = {
    'AlbertTokenizer': "spiece.model",
    'RobertaTokenizer': "vocab.json",
    'BertTokenizer': "vocab.txt",
    'DistilBertTokenizer': "vocab.txt",
    'CamembertTokenizer': "sentencepiece.bpe.model",
    'GPT2Tokenizer': "vocab.json",
    'T5Tokenizer': "spiece.model",
    "BartTokenizer": "vocab.json",
}


def get_huggingface_lm_model(
    pretrained_model_name: str,
    config_dict: Optional[dict] = None,
    config_file: Optional[str] = None,
):
    """
    Returns lm model instantiated with Huggingface

    Args:
        pretrained_mode_name: specify this to instantiate pretrained model from Huggingface,
            e.g. bert-base-cased. For entire list, see get_huggingface_pretrained_lm_models_list().
        config_dict: model configuration dictionary used to instantiate Huggingface model from scratch
        config_file: path to model configuration file used to instantiate Huggingface model from scratch

    Returns:
        BertModule
    """

    try:
        automodel = AutoModel.from_pretrained(pretrained_model_name)
    except Exception as e:
        raise ValueError(f"{pretrained_model_name} is not supported by HuggingFace. {e}")

    model_type = type(automodel).__name__

    if model_type in HUGGINGFACE_MODELS:
        model_class = HUGGINGFACE_MODELS[model_type]["class"]
        if config_file:
            if not os.path.exists(config_file):
                logging.warning(
                    f"Config file was not found at {config_file}. Will attempt to use config_dict or pretrained_model_name."
                )
            else:
                config_class = HUGGINGFACE_MODELS[model_type]["config"]
                return model_class(config_class.from_json_file(config_file))
        if config_dict:
            config_class = HUGGINGFACE_MODELS[model_type]["config"]
            return model_class(config=config_class(**config_dict))
        else:
            return model_class.from_pretrained(pretrained_model_name)
    else:
        raise ValueError(f"Use HuggingFace API directly in NeMo for {pretrained_model_name}")


def get_huggingface_pretrained_lm_models_list(
    include_external: bool = False,
) -> List[str]:
    """
    Returns the list of pretrained HuggingFace language models

    Args:
        include_external if true includes all HuggingFace model names, not only those supported language models in NeMo.

    Returns the list of HuggingFace models
    """

    huggingface_models = []
    for model in HUGGINGFACE_MODELS:
        model_names = HUGGINGFACE_MODELS[model]["pretrained_model_list"]
        huggingface_models.extend(model_names)
    return huggingface_models
