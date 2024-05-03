import os
from argparse import ArgumentParser

from omegaconf.omegaconf import OmegaConf
from transformers import AutoConfig, RecurrentGemmaModel

from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_griffin_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--input_path", type=str, default=None, required=True)
    parser.add_argument(
        "--precision", type=str, default="32", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    args = parser.parse_args()
    return args


def convert(args):

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision

    logging.info(f"Loading checkpoint from NeMo: `{args.input_path}`")

    trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()

    nemo_model = MegatronGriffinModel.restore_from(args.input_path, trainer=trainer)
    hf_config = AutoConfig.from_pretrained("google/recurrentgemma-2b")

    # NeMo doesn't support LM Head for Griffin yet, so RecurrentGemmaModel is used instead of AutoModelForCausalLM
    hf_model = RecurrentGemmaModel._from_config(hf_config)

    new_state_dict = {}

    new_state_dict['embed_tokens.weight'] = nemo_model.state_dict()['model.embedding.word_embeddings.weight']
    new_state_dict['final_norm.weight'] = nemo_model.state_dict()['model.decoder.final_layernorm.weight']

    for l in range(nemo_config.model.num_layers):
        print(f"Converting Layer {l}")
        print("********************")

        (
            new_state_dict[f'layers.{l}.mlp_block.gate_proj.weight'],
            new_state_dict[f'layers.{l}.mlp_block.up_proj.weight'],
        ) = nemo_model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.weight'].chunk(2)
        (
            new_state_dict[f'layers.{l}.mlp_block.gate_proj.bias'],
            new_state_dict[f'layers.{l}.mlp_block.up_proj.bias'],
        ) = nemo_model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.bias'].chunk(2)
        new_state_dict[f'layers.{l}.mlp_block.down_proj.weight'] = nemo_model.state_dict()[
            f'model.decoder.layers.{l}.mlp.linear_fc2.weight'
        ]
        new_state_dict[f'layers.{l}.mlp_block.down_proj.bias'] = nemo_model.state_dict()[
            f'model.decoder.layers.{l}.mlp.linear_fc2.bias'
        ]

        new_state_dict[f'layers.{l}.channel_pre_norm.weight'] = nemo_model.state_dict()[
            f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
        ]

        if l % 3 == 2:

            new_state_dict[f'layers.{l}.temporal_block.o_proj.weight'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
            ]
            new_state_dict[f'layers.{l}.temporal_block.o_proj.bias'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.self_attention.linear_proj.bias'
            ]
            new_state_dict[f'layers.{l}.temporal_pre_norm.weight'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
            ]
            (
                new_state_dict[f'layers.{l}.temporal_block.q_proj.weight'],
                new_state_dict[f'layers.{l}.temporal_block.k_proj.weight'],
                new_state_dict[f'layers.{l}.temporal_block.v_proj.weight'],
            ) = nemo_model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'].split(
                [2560, 256, 256]
            )

        else:

            new_state_dict[f'layers.{l}.temporal_pre_norm.weight'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.linear_in.layer_norm_weight'
            ]
            (
                new_state_dict[f'layers.{l}.temporal_block.linear_x.weight'],
                new_state_dict[f'layers.{l}.temporal_block.linear_y.weight'],
            ) = nemo_model.state_dict()[f'model.decoder.layers.{l}.recurrent_layer.linear_in.weight'].chunk(2)
            (
                new_state_dict[f'layers.{l}.temporal_block.linear_x.bias'],
                new_state_dict[f'layers.{l}.temporal_block.linear_y.bias'],
            ) = nemo_model.state_dict()[f'model.decoder.layers.{l}.recurrent_layer.linear_in.bias'].chunk(2)

            new_state_dict[f'layers.{l}.temporal_block.linear_out.weight'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.linear_out.weight'
            ]
            new_state_dict[f'layers.{l}.temporal_block.linear_out.bias'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.linear_out.bias'
            ]

            new_state_dict[f'layers.{l}.temporal_block.conv_1d.weight'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.conv_1d.conv_1d.weight'
            ]
            new_state_dict[f'layers.{l}.temporal_block.conv_1d.bias'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.conv_1d.conv_1d.bias'
            ]

            new_state_dict[f'layers.{l}.temporal_block.rg_lru.recurrent_param'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.rg_lru.a_param'
            ]
            new_state_dict[f'layers.{l}.temporal_block.rg_lru.input_gate_weight'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.rg_lru.input_gate.w'
            ]
            new_state_dict[f'layers.{l}.temporal_block.rg_lru.input_gate_bias'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.rg_lru.input_gate.b'
            ]
            new_state_dict[f'layers.{l}.temporal_block.rg_lru.recurrent_gate_weight'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.rg_lru.a_gate.w'
            ]
            new_state_dict[f'layers.{l}.temporal_block.rg_lru.recurrent_gate_bias'] = nemo_model.state_dict()[
                f'model.decoder.layers.{l}.recurrent_layer.rg_lru.a_gate.b'
            ]

    hf_model.load_state_dict(new_state_dict, strict=True)
    dtype = torch_dtype_from_precision(args.precision)
    hf_model = hf_model.to(dtype=dtype)

    hf_model.save_pretrained(args.output_path)
    logging.info(f'Full HF model model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
