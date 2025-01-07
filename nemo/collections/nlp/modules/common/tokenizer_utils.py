# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import os.path
from dataclasses import MISSING, dataclass
from typing import Dict, List, Optional

from nemo.utils import logging

from .huggingface.huggingface_utils import get_huggingface_pretrained_lm_models_list

__all__ = ['get_tokenizer', 'get_tokenizer_list']


megatron_tokenizer_model_map = {
    'BertWordPieceLowerCase': 'megatron-bert-345m-uncased',
    'BertWordPieceCase': 'megatron-bert-345m-cased',
    'GPT2BPETokenizer': 'megatron-gpt-345m',
}


def get_tokenizer_list() -> List[str]:
    """
    Returns all all supported tokenizer names
    """
    s = set(get_huggingface_pretrained_lm_models_list(include_external=False))
    s.update(set(get_huggingface_pretrained_lm_models_list(include_external=True)))
    return ["sentencepiece", "char", "word"] + list(s)


@dataclass
class TokenizerConfig:
    library: str = MISSING
    tokenizer_model: Optional[str] = None
    vocab_size: Optional[int] = None
    vocab_file: Optional[str] = None
    special_tokens: Optional[Dict[str, str]] = None
    bpe_dropout: Optional[float] = 0.0
    coverage: Optional[float] = 0.999
    training_sample_size: Optional[int] = None
    r2l: Optional[bool] = False
    sentencepiece_legacy: Optional[bool] = False


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_model: Optional[str] = None,
    vocab_file: Optional[str] = None,
    merges_file: Optional[str] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    use_fast: Optional[bool] = False,
    bpe_dropout: Optional[float] = 0.0,
    chat_template: Optional[Dict] = None,
):
    """
    Args:
        tokenizer_name: sentencepiece or pretrained model from the hugging face list,
            for example: bert-base-cased
            To see the list of all HuggingFace pretrained models, use:
            nemo_nlp.modules.common.get_huggingface_pretrained_lm_models_list()
        tokenizer_model: tokenizer model file of sentencepiece
        special_tokens: dict of special tokens.
            For additional special tokens besides standard special tokens (bos, eos, pad, etc.), such as sentinel tokens for T5 (<extra_id_0>, <extra_id_1>, etc.), use key 'additional_special_tokens'
        vocab_file: path to vocab file
        use_fast: (only for HuggingFace AutoTokenizer) set to True to use fast HuggingFace tokenizer
        bpe_dropout: (experimental) BPE dropout tries to corrupt the standard segmentation
            procedure of BPE to help
            model better learn word compositionality and become robust to segmentation errors.
            It has emperically been shown to improve inference time BLEU scores.
    """

    if special_tokens is None:
        special_tokens_dict = {}
    else:
        special_tokens_dict = special_tokens

    if 'megatron' in tokenizer_name:
        try:
            from nemo.collections.nlp.modules.common.megatron.megatron_utils import (
                get_megatron_merges_file,
                get_megatron_tokenizer,
                get_megatron_vocab_file,
            )
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        if vocab_file is None:
            vocab_file = get_megatron_vocab_file(tokenizer_name)
            merges_file = get_megatron_merges_file(tokenizer_name)
        tokenizer_name = get_megatron_tokenizer(tokenizer_name)

    if tokenizer_name == 'sentencepiece':
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

        logging.info("tokenizer_model: " + str(tokenizer_model))
        return SentencePieceTokenizer(
            model_path=tokenizer_model,
            special_tokens=special_tokens,
            legacy=True,
            chat_template=chat_template,
        )
    elif tokenizer_name == 'tiktoken':
        from nemo.collections.common.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(vocab_file=vocab_file, special_tokens=special_tokens['additional_special_tokens'])
    elif tokenizer_name == 'word':
        from nemo.collections.common.tokenizers.word_tokenizer import WordTokenizer

        return WordTokenizer(vocab_file=vocab_file, **special_tokens_dict)
    elif tokenizer_name == 'char':
        from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer

        return CharTokenizer(vocab_file=vocab_file, **special_tokens_dict)
    elif tokenizer_name == 'regex':
        from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer

        return RegExTokenizer().load_tokenizer(regex_file=tokenizer_model, vocab_file=vocab_file)

    logging.info(
        f"Getting HuggingFace AutoTokenizer with pretrained_model_name: {tokenizer_name}, vocab_file: {vocab_file}, merges_files: {merges_file}, "
        f"special_tokens_dict: {special_tokens_dict}, and use_fast: {use_fast}"
    )
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

    return AutoTokenizer(
        pretrained_model_name=tokenizer_name,
        vocab_file=vocab_file,
        merges_file=merges_file,
        **special_tokens_dict,
        use_fast=use_fast,
    )


def get_nmt_tokenizer(
    library: str = 'sentencepiece',
    model_name: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    vocab_file: Optional[str] = None,
    merges_file: Optional[str] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    use_fast: Optional[bool] = False,
    bpe_dropout: Optional[float] = 0.0,
    r2l: Optional[bool] = False,
    legacy: Optional[bool] = False,
    delimiter: Optional[str] = None,
    trust_remote_code: Optional[bool] = False,
    chat_template: Optional[Dict] = None,
):
    """
    Args:
        model_name: if using a pretrained model from NeMo, HuggingFace, or Megatron
        tokenizer_model: tokenizer model file of sentencepiece
        special_tokens: dict of special tokens
        vocab_file: path to vocab file
        use_fast: (only for HuggingFace AutoTokenizer) set to True to use fast HuggingFace tokenizer
        bpe_dropout: (experimental) BPE dropout tries to corrupt the standard segmentation procedure
            of BPE to help model better learn word compositionality and become robust to segmentation errors.
            It has empirically been shown to improve inference time BLEU scores.
        r2l: Whether to return subword IDs from right to left
    """
    import omegaconf
    from omegaconf import OmegaConf

    if isinstance(special_tokens, omegaconf.listconfig.ListConfig):
        special_tokens = OmegaConf.to_container(special_tokens)
    if special_tokens is None:
        special_tokens_dict = {}
    else:
        special_tokens_dict = special_tokens

    if (library != 'byte-level') and (
        model_name is None and (tokenizer_model is None or not os.path.isfile(tokenizer_model))
    ):
        raise ValueError("No Tokenizer path provided or file does not exist!")

    if library == 'huggingface':
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        logging.info(f'Getting HuggingFace AutoTokenizer with pretrained_model_name: {model_name}')
        return AutoTokenizer(
            pretrained_model_name=model_name,
            vocab_file=vocab_file,
            merges_file=merges_file,
            **special_tokens_dict,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )
    elif library == 'sentencepiece':
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

        logging.info(f'Getting SentencePiece with model: {tokenizer_model}')

        return SentencePieceTokenizer(
            model_path=tokenizer_model,
            special_tokens=special_tokens,
            legacy=legacy,
            chat_template=chat_template,
        )
    elif library == 'byte-level':
        from nemo.collections.common.tokenizers.bytelevel_tokenizers import ByteLevelTokenizer

        logging.info(f'Using byte-level tokenization')
        return ByteLevelTokenizer(special_tokens_dict)
    elif library == 'regex':
        from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer

        logging.info(f'Using regex tokenization')
        return RegExTokenizer().load_tokenizer(regex_file=tokenizer_model, vocab_file=vocab_file)
    elif library == 'megatron':

        if model_name == 'GPTSentencePieceTokenizer':
            from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

            logging.info("tokenizer_model: ")
            logging.info(tokenizer_model)
            return SentencePieceTokenizer(model_path=tokenizer_model, legacy=legacy)

        if model_name in megatron_tokenizer_model_map:
            model_name = megatron_tokenizer_model_map[model_name]
        logging.info(
            f'Getting Megatron tokenizer for pretrained model name: {model_name}, custom vocab file: {vocab_file}, and merges file: {merges_file}'
        )
        return get_tokenizer(
            tokenizer_name=model_name,
            vocab_file=vocab_file,
            merges_file=merges_file,
            special_tokens=special_tokens_dict,
            chat_template=chat_template,
        )
    elif library == 'tabular':
        from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer

        return TabularTokenizer(vocab_file, delimiter=delimiter)
    elif library == 'tiktoken':
        from nemo.collections.common.tokenizers.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(vocab_file=vocab_file)
    else:
        raise NotImplementedError(
            'Currently we only support "huggingface", "sentencepiece", "megatron", and "byte-level" tokenizer'
            'libraries.'
        )
