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


from typing import Optional, Union

from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.modules.common.huggingface.huggingface_decoder import HuggingFaceDecoderModule
from nemo.collections.nlp.modules.common.huggingface.huggingface_encoder import HuggingFaceEncoderModule
from nemo.collections.nlp.modules.common.transformer.transformer import TransformerDecoderNM, TransformerEncoderNM
from nemo.collections.nlp.modules.common.transformer.transformer_bottleneck import TransformerBottleneckEncoderNM


def get_nemo_transformer(
    model_name: Optional[str] = None,
    pretrained: bool = False,
    config_dict: Optional[Union[dict, DictConfig]] = None,
    encoder: bool = True,
    pre_ln_final_layer_norm: bool = True,
    padding_idx: int = 0,
) -> Union[TransformerEncoderNM, TransformerDecoderNM]:
    """Returns NeMo transformer.
    The following configurations are mandatory:
        vocab_size: int
        hidden_size: int
        num_layers: int
        inner_size: int
    and must be specified if using config_dict.

    Args:
        model_name (Optional[str]): model name to download from NGC
        pretrained: (bool): False will instantiate the named model architecture with random weights.
        config_dict (Optional[dict], optional): model configuration parameters. Defaults to None.
        config_file (Optional[str], optional): path to json file containing model configuration. Defaults to None.
        checkpoint_file (Optional[str], optional): load weights from path to local checkpoint. Defaults to None.
        encoder (bool, optional): True will use EncoderTransformerNM, False will use DecoderTransformerNM. Defaults to True.
    """
    if model_name is not None:
        raise ValueError(f'NeMo transformers cannot be loaded from NGC yet. model_name should be None')

    if pretrained:
        raise ValueError(f'NeMo transformers cannot be loaded from NGC yet. pretrained should be False')

    cfg = None

    if not pretrained:
        assert (
            config_dict.get('vocab_size') is not None
            and config_dict.get('hidden_size') is not None
            and config_dict.get('num_layers') is not None
            and config_dict.get('inner_size') is not None
        ), f'Using config_dict: {config_dict}. vocab_size, hidden_size, num_layers, and inner_size must are mandatory arguments'

        cfg = config_dict

    if encoder:
        # if arch exists in cfg we return TransformerBottleneckEncoderNM
        arch = cfg.get('arch', '')
        if not arch:
            model = TransformerEncoderNM(
                vocab_size=cfg.get('vocab_size'),
                hidden_size=cfg.get('hidden_size'),
                num_layers=cfg.get('num_layers'),
                inner_size=cfg.get('inner_size'),
                max_sequence_length=cfg.get('max_sequence_length', 512),
                embedding_dropout=cfg.get('embedding_dropout', 0.0),
                learn_positional_encodings=cfg.get('learn_positional_encodings', False),
                num_attention_heads=cfg.get('num_attention_heads'),
                ffn_dropout=cfg.get('ffn_dropout', 0.0),
                attn_score_dropout=cfg.get('attn_score_dropout', 0.0),
                attn_layer_dropout=cfg.get('attn_layer_dropout', 0.0),
                hidden_act=cfg.get('hidden_act', 'relu'),
                mask_future=cfg.get('mask_future', True),
                pre_ln=cfg.get('pre_ln', False),
                pre_ln_final_layer_norm=pre_ln_final_layer_norm,
                num_token_types=cfg.get('num_token_types', 2),
                padding_idx=padding_idx,
            )
        elif arch in TransformerBottleneckEncoderNM._SUPPORTED_ARCH:
            model = TransformerBottleneckEncoderNM(
                vocab_size=cfg.get('vocab_size'),
                hidden_size=cfg.get('hidden_size'),
                num_layers=cfg.get('num_layers'),
                inner_size=cfg.get('inner_size'),
                max_sequence_length=cfg.get('max_sequence_length', 512),
                embedding_dropout=cfg.get('embedding_dropout', 0.0),
                learn_positional_encodings=cfg.get('learn_positional_encodings', False),
                num_attention_heads=cfg.get('num_attention_heads'),
                ffn_dropout=cfg.get('ffn_dropout', 0.0),
                attn_score_dropout=cfg.get('attn_score_dropout', 0.0),
                attn_layer_dropout=cfg.get('attn_layer_dropout', 0.0),
                hidden_act=cfg.get('hidden_act', 'relu'),
                mask_future=cfg.get('mask_future', False),
                pre_ln=cfg.get('pre_ln', False),
                pre_ln_final_layer_norm=pre_ln_final_layer_norm,
                num_token_types=cfg.get('num_token_types', 2),
                arch=cfg.get('arch', 'full'),
                hidden_steps=cfg.get('hidden_steps', -1),
                hidden_blocks=cfg.get('hidden_blocks', 1),
                hidden_init_method=cfg.get('hidden_init_method', 'default'),
                return_mask=cfg.get('return_mask', True),
                padding_idx=padding_idx,
            )
        else:
            raise ValueError(f"Unknown arch = {arch}")
    else:
        model = TransformerDecoderNM(
            vocab_size=cfg.get('vocab_size'),
            hidden_size=cfg.get('hidden_size'),
            num_layers=cfg.get('num_layers'),
            inner_size=cfg.get('inner_size'),
            max_sequence_length=cfg.get('max_sequence_length', 512),
            embedding_dropout=cfg.get('embedding_dropout', 0.0),
            learn_positional_encodings=cfg.get('learn_positional_encodings', False),
            num_attention_heads=cfg.get('num_attention_heads'),
            ffn_dropout=cfg.get('ffn_dropout', 0.0),
            attn_score_dropout=cfg.get('attn_score_dropout', 0.0),
            attn_layer_dropout=cfg.get('attn_layer_dropout', 0.0),
            hidden_act=cfg.get('hidden_act', 'relu'),
            pre_ln=cfg.get('pre_ln', False),
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
            num_token_types=cfg.get('num_token_types', 2),
            padding_idx=padding_idx,
        )

    return model


def get_huggingface_transformer(
    model_name: Optional[str] = None,
    pretrained: bool = False,
    config_dict: Optional[Union[dict, DictConfig]] = None,
    encoder: bool = True,
) -> Union[HuggingFaceEncoderModule, HuggingFaceDecoderModule]:

    if encoder:
        model = HuggingFaceEncoderModule(model_name, pretrained, config_dict)
    else:
        model = HuggingFaceDecoderModule(model_name, pretrained, config_dict)

    return model


def get_megatron_transformer(
    model_name: Optional[str] = None,
    pretrained: bool = True,
    config_dict: Optional[Union[dict, DictConfig]] = None,
    encoder: bool = True,
    checkpoint_file: str = None,
) -> None:

    raise ValueError(
        "megatron-lm bert encoders are deprecated in NeMo 1.5.0. Please use NeMo 1.4.0 until megatron bert support is added again."
    )

    # vocab_file = config_dict.pop('vocab_file', None)
    # if encoder:
    #     model = MegatronEncoderModule(
    #         model_name=model_name,
    #         pretrained=pretrained,
    #         config_dict=config_dict,
    #         checkpoint_file=checkpoint_file,
    #         vocab_file=vocab_file,
    #     )
    # else:
    #     raise ValueError('Megatron decoders are not currently supported.')

    # return model
