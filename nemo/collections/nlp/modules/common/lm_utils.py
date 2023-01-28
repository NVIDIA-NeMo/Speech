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
import os
from typing import List, Optional, Union

from attr import asdict
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.nlp.modules.common.bert_module import BertModule
from nemo.collections.nlp.modules.common.decoder_module import DecoderModule
from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import (
    get_huggingface_lm_model,
    get_huggingface_pretrained_lm_models_list,
)
from nemo.collections.nlp.modules.common.megatron.megatron_utils import get_megatron_pretrained_bert_models
from nemo.collections.nlp.modules.common.transformer.transformer import NeMoTransformerConfig
from nemo.collections.nlp.modules.common.transformer.transformer_utils import (
    get_huggingface_transformer,
    get_nemo_transformer,
)
from nemo.utils import AppState, logging

__all__ = ['get_pretrained_lm_models_list', 'get_lm_model', 'pad_batch']


def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_pretrained_lm_models_list(include_external: bool = False) -> List[str]:
    """
    Returns the list of supported pretrained model names

    Args:
        include_external if true includes all HuggingFace model names, not only those supported language models in NeMo.

    """
    return get_huggingface_pretrained_lm_models_list(include_external=include_external)


def get_lm_model(
    config_dict: Optional[dict] = None,
    config_file: Optional[str] = None,
    vocab_file: Optional[str] = None,
    trainer: Optional[Trainer] = None,
    cfg: DictConfig = None,
) -> BertModule:
    """
    Helper function to instantiate a language model encoder, either from scratch or a pretrained model.
    If only pretrained_model_name are passed, a pretrained model is returned.
    If a configuration is passed, whether as a file or dictionary, the model is initialized with random weights.

    Args:
        config_dict: path to the model configuration dictionary
        config_file: path to the model configuration file
        vocab_file: path to vocab_file to be used with Megatron-LM
        trainer: an instance of a PyTorch Lightning trainer
        cfg: a model configuration
    Returns:
        Pretrained BertModule
    """

    # check valid model type
    if cfg.language_model.get('pretrained_model_name'):
        if (
            not cfg.language_model.pretrained_model_name
            or cfg.language_model.pretrained_model_name not in get_pretrained_lm_models_list(include_external=False)
        ):
            logging.warning(
                f'{cfg.language_model.pretrained_model_name} is not in get_pretrained_lm_models_list(include_external=False), '
                f'will be using AutoModel from HuggingFace.'
            )

    # warning when user passes both configuration dict and file
    if config_dict and config_file:
        logging.warning(
            f"Both config_dict and config_file were found, defaulting to use config_file: {config_file} will be used."
        )

    pretrain_model_name = ''
    if cfg.get('language_model') and cfg.language_model.get('pretrained_model_name', ''):
        pretrain_model_name = cfg.language_model.get('pretrained_model_name', '')
    all_pretrained_megatron_bert_models = get_megatron_pretrained_bert_models()
    if (
        cfg.tokenizer is not None
        and cfg.tokenizer.get("tokenizer_name", "") is not None
        and "megatron" in cfg.tokenizer.get("tokenizer_name", "")
    ) or pretrain_model_name in all_pretrained_megatron_bert_models:
        import torch

        from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel

        class Identity(torch.nn.Module):
            def __init__(self):
                super(Identity, self).__init__()

            def forward(self, x, *args):
                return x

        if cfg.language_model.get("lm_checkpoint"):
            model = MegatronBertModel.restore_from(restore_path=cfg.language_model.lm_checkpoint, trainer=trainer)
        else:
            model = MegatronBertModel.from_pretrained(cfg.language_model.get('pretrained_model_name'), trainer=trainer)
        # remove the headers that are only revelant for pretraining
        model.model.lm_head = Identity()
        model.model.binary_head = Identity()
        model.model.language_model.pooler = Identity()

    else:
        model = get_huggingface_lm_model(
            config_dict=config_dict,
            config_file=config_file,
            pretrained_model_name=cfg.language_model.pretrained_model_name,
        )

        if cfg.language_model.get("lm_checkpoint"):
            app_state = AppState()
            if not app_state.is_model_being_restored and not os.path.exists(cfg.language_model.lm_checkpoint):
                raise ValueError(f'{cfg.language_model.lm_checkpoint} not found')
            model.restore_weights(restore_path=cfg.language_model.lm_checkpoint)

    return model


# @dataclass
# class TransformerConfig:
#     library: str = 'nemo'
#     model_name: Optional[str] = None
#     pretrained: bool = False
#     config_dict: Optional[dict] = None
#     checkpoint_file: Optional[str] = None
#     encoder: bool = True


def get_transformer(
    library: str = 'nemo',
    model_name: Optional[str] = None,
    pretrained: bool = False,
    config_dict: Optional[dict] = None,
    checkpoint_file: Optional[str] = None,
    encoder: bool = True,
    pre_ln_final_layer_norm: bool = True,
    padding_idx: int = 0,
) -> Union[EncoderModule, DecoderModule]:
    """Gets Transformer based model to be used as an Encoder or Decoder in NeMo NLP.
       First choose the library to get the transformer from. This can be huggingface,
       megatron, or nemo. Use the model_name arg to get a named model architecture
       and use the pretrained arg to get the named model architecture with pretrained weights.

       If model_name is None, then we can pass in a custom configuration via the config_dict.
       For example, to instantiate a HuggingFace BERT model with custom configuration we would do:
       encoder = get_transformer(library='huggingface',
                                 config_dict={
                                     '_target_': 'transformers.BertConfig',
                                     'hidden_size': 1536
                                 }) 


    Args:
        library (str, optional): Can be 'nemo', 'huggingface', or 'megatron'. Defaults to 'nemo'.
        model_name (Optional[str], optional): Named model architecture from the chosen library. Defaults to None.
        pretrained (bool, optional): Use True to get pretrained weights. 
                                     False will use the same architecture but with randomly initialized weights.
                                     Defaults to False.
        config_dict (Optional[dict], optional): Use for custom configuration of transformer. Defaults to None.
        checkpoint_file (Optional[str], optional): Provide weights for the transformer from a local checkpoint. Defaults to None.
        encoder (bool, optional): True returns an EncoderModule, False returns a DecoderModule. Defaults to True.

    Returns:
        Union[EncoderModule, DecoderModule]: Ensures that Encoder/Decoder will work in EncDecNLPModel
    """

    model = None

    if library == 'nemo':
        if isinstance(config_dict, NeMoTransformerConfig):
            config_dict = asdict(config_dict)
        model = get_nemo_transformer(
            model_name=model_name,
            pretrained=pretrained,
            config_dict=config_dict,
            encoder=encoder,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
            padding_idx=padding_idx,
        )

        if checkpoint_file is not None:
            if os.path.isfile(checkpoint_file):
                raise ValueError(f'Loading transformer weights from checkpoint file has not been implemented yet.')

    elif library == 'huggingface':
        model = get_huggingface_transformer(
            model_name=model_name, pretrained=pretrained, config_dict=config_dict, encoder=encoder
        )

    elif library == 'megatron':
        raise ValueError(
            f'megatron-lm bert support has been deprecated in NeMo 1.5+. Please use NeMo 1.4 for support.'
        )
        # TODO: enable megatron bert in nemo
        # model = get_megatron_transformer(
        #     model_name=model_name,
        #     pretrained=pretrained,
        #     config_dict=config_dict,
        #     encoder=encoder,
        #     checkpoint_file=checkpoint_file,
        # )

    else:
        raise ValueError("Libary must be 'nemo', 'huggingface' or 'megatron'")

    return model
