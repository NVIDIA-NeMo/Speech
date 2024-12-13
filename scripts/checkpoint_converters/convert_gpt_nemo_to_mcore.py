# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import AppState, logging

r"""
Script to convert a legacy (non-mcore path) nemo checkpoint into mcore-path checkpoint for GPT models.

Please use a container later than 23.10 or the current github main branch

*Important* Before running this script, please first
1) convert your legacy checkpoint to TP1 PP1 format:
    python examples/nlp/language_modeling/megatron_change_num_partitions.py \
    <follow the readme in that script> \
    --target_tensor_model_parallel_size=1 \
    --target_pipeline_model_parallel_size=1
2) extract your nemo file to a folder with
    tar -xvf filename.nemo

Then, run this conversion script:
python convert_gpt_nemo_to_mcore.py \
 --input_name_or_path <path to extracted, TP1 PP1 legacy checkpoint folder> \
 --output_path <path to output nemo file>
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to extracted, TP1 PP1 NeMo GPT checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to output mcore weights file (ends in .nemo).",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
        "but this option makes the conversion script significantly slower.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run conversion again and overwrite output file when the output file already exists",
    )
    parser.add_argument(
        "--ignore_if_missing",
        default="rotary_pos_emb.inv_freq",
        help="comma-separated list of state_dict keys that are known to be missing in mcore and can be safely ignored",
    )
    args = parser.parse_args()
    return args


def get_mcore_model_from_nemo_file(nemo_restore_from_path, cpu_only=False):
    model_cfg = MegatronGPTModel.restore_from(nemo_restore_from_path, return_config=True)
    model_cfg.tokenizer.vocab_file = None
    model_cfg.tokenizer.merge_file = None
    model_cfg.mcore_gpt = True
    model_cfg.use_cpu_initialization = cpu_only

    # The key mappings use TE spec, hence set the TE flag to True
    model_cfg.transformer_engine = True

    logging.info("*** initializing mcore model with the following config")
    logging.info(OmegaConf.to_yaml(model_cfg))
    trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())

    app_state = AppState()
    if os.path.isdir(nemo_restore_from_path):
        app_state.nemo_file_folder = nemo_restore_from_path
    else:
        logging.warning(
            "⚠️ `nemo_file_folder` is NOT set because checkpoint is not pre-extracted. Subsequent operations may fail."
        )
    mcore_model = MegatronGPTModel(model_cfg, trainer=trainer)
    return mcore_model


def print_mcore_parameter_names(restore_from_path):
    mcore_model = get_mcore_model_from_nemo_file(restore_from_path)

    print("*********")
    print('\n'.join(sorted([k + '###' + str(v.shape) for k, v in mcore_model.named_parameters()])))
    print("*********")


def build_key_mapping(nemo_cfg):
    num_layers = nemo_cfg.num_layers
    has_bias = nemo_cfg.get("bias", True)
    has_layernorm_bias = (
        nemo_cfg.get("normalization", "layernorm") != "rmsnorm"
    )  # llama model uses rmsnorm which does not have bias
    model_str = 'model.module' if nemo_cfg.get('megatron_amp_O2', False) else 'model'

    # For GPT there is a 1:1 mapping of keys
    mcore_to_nemo_mapping = {
        f"{model_str}.embedding.word_embeddings.weight": "model.language_model.embedding.word_embeddings.weight",
        f"{model_str}.decoder.final_layernorm.weight": "model.language_model.encoder.final_layernorm.weight",
    }
    if has_layernorm_bias:
        mcore_to_nemo_mapping[f"{model_str}.decoder.final_layernorm.bias"] = (
            "model.language_model.encoder.final_layernorm.bias"
        )

    if not nemo_cfg.get("share_embeddings_and_output_weights", True):
        mcore_to_nemo_mapping[f"{model_str}.output_layer.weight"] = "model.language_model.output_layer.weight"

    if nemo_cfg.get("position_embedding_type", 'learned_absolute') == 'rope':
        mcore_to_nemo_mapping[f"{model_str}.rotary_pos_emb.inv_freq"] = "model.language_model.rotary_pos_emb.inv_freq"
    else:
        mcore_to_nemo_mapping[f"{model_str}.embedding.position_embeddings.weight"] = (
            "model.language_model.embedding.position_embeddings.weight"
        )

    nemo_prefix = "model.language_model.encoder.layers"
    mcore_prefix = f"{model_str}.decoder.layers"
    for i in range(num_layers):
        for wb in ('weight', 'bias') if has_bias else ('weight',):
            mcore_to_nemo_mapping.update(
                {
                    f"{mcore_prefix}.{i}.mlp.linear_fc2.{wb}": f"{nemo_prefix}.{i}.mlp.dense_4h_to_h.{wb}",
                    f"{mcore_prefix}.{i}.mlp.linear_fc1.{wb}": f"{nemo_prefix}.{i}.mlp.dense_h_to_4h.{wb}",
                    f"{mcore_prefix}.{i}.self_attention.linear_proj.{wb}": f"{nemo_prefix}.{i}.self_attention.dense.{wb}",
                    f"{mcore_prefix}.{i}.self_attention.linear_qkv.{wb}": f"{nemo_prefix}.{i}.self_attention.query_key_value.{wb}",
                }
            )
        # layernorm layers always have bias, but llama model uses rmsnorm which does not have bias
        for wb in ('weight', 'bias') if has_layernorm_bias else ('weight',):
            mcore_to_nemo_mapping.update(
                {
                    f"{mcore_prefix}.{i}.self_attention.linear_qkv.layer_norm_{wb}": f"{nemo_prefix}.{i}.input_layernorm.{wb}",
                    f"{mcore_prefix}.{i}.mlp.linear_fc1.layer_norm_{wb}": f"{nemo_prefix}.{i}.post_attention_layernorm.{wb}",
                }
            )

    return mcore_to_nemo_mapping


def load_model(model, state_dict, ignore_if_missing=tuple()):
    # try:
    for name, module in model.named_parameters():
        if name in state_dict:
            module.data = state_dict.pop(name)
        else:
            raise RuntimeError(f"Unexpected key: {name} not in state_dict but in model.")

    for name, buffer in model.named_buffers():
        if name in state_dict:
            buffer.data = state_dict.pop(name)

    # Some previous buffers are known to be removed in new mcore models => it is ok to ignore them.
    for key in list(state_dict):
        if any(key.endswith(suffix) for suffix in ignore_if_missing):
            state_dict.pop(key)

    if state_dict:
        raise RuntimeError(f"Additional keys: {state_dict.keys()} in state_dict but not in model.")

    return model


def restore_model(nemo_file, cpu_only=False):
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    map_location = torch.device('cpu') if cpu_only else None
    model_config = MegatronGPTModel.restore_from(
        nemo_file, trainer=dummy_trainer, return_config=True, map_location=map_location
    )
    model_config.use_cpu_initialization = cpu_only

    if model_config.get('sequence_parallel', None):
        model_config.sequence_parallel = False

    # To copy weights in the original precision, we have to turn on O2.
    orig_megatron_amp_O2_value = model_config.megatron_amp_O2
    if "target" in model_config and model_config.target.endswith("MegatronGPTSFTModel"):
        logging.warning(
            "⚠️ Model target is `MegatronGPTSFTModel` which may not work with this conversion script. "
            "This is a known issue. For now, please modify the config yaml file to use `MegatronGPTModel`."
        )

    if model_config.get("precision", None) in ['bf16', 'bf16-mixed']:
        model_config.megatron_amp_O2 = True

    model = MegatronGPTModel.restore_from(
        nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )

    # restore O2 to the original value so mcore model has the same config
    model.cfg.megatron_amp_O2 = orig_megatron_amp_O2_value
    return model


def convert(input_nemo_file, output_nemo_file, skip_if_output_exists=True, cpu_only=False, ignore_if_missing=tuple()):
    if skip_if_output_exists and os.path.exists(output_nemo_file):
        logging.info(f"Output file already exists ({output_nemo_file}), skipping conversion...")
        logging.info("If you want to overwrite the output file, please run with --overwrite flag")
        return
    nemo_model = restore_model(input_nemo_file, cpu_only=cpu_only)

    nemo_tokenizer_model = nemo_model.cfg.tokenizer.model
    nemo_state_dict = nemo_model.state_dict()
    mcore_state_dict = OrderedDict()
    for mcore_param, nemo_param in build_key_mapping(nemo_model.cfg).items():
        if mcore_param.endswith("linear_fc1.weight"):
            # in llama models, need to concat dense_h_to_4h.weight and dense_h_to_4h_2.weight for the corresponding linear_fc1.weight
            second_param = nemo_param.replace("dense_h_to_4h.weight", "dense_h_to_4h_2.weight")
            if second_param in nemo_state_dict:
                mcore_state_dict[mcore_param] = torch.cat(
                    [nemo_state_dict[nemo_param], nemo_state_dict[second_param]], dim=0
                )
            else:
                mcore_state_dict[mcore_param] = nemo_state_dict[nemo_param]
        else:
            mcore_state_dict[mcore_param] = nemo_state_dict[nemo_param]

    mcore_model = get_mcore_model_from_nemo_file(input_nemo_file, cpu_only=cpu_only)
    mcore_model = load_model(mcore_model, mcore_state_dict, ignore_if_missing=ignore_if_missing)

    if nemo_model.cfg.tokenizer.model is not None:
        logging.info("registering artifact: tokenizer.model = " + nemo_tokenizer_model)
        mcore_model.register_artifact("tokenizer.model", nemo_tokenizer_model)

    mcore_model.cfg.use_cpu_initialization = False
    mcore_model.save_to(output_nemo_file)
    logging.info(f"✅ Done. Model saved to {output_nemo_file}")
    del mcore_model
    del nemo_model


def run_sanity_checks(nemo_file, mcore_file, cpu_only=False, ignore_if_missing=tuple()):

    nemo_model = restore_model(nemo_file, cpu_only=cpu_only).eval()
    mcore_model = restore_model(mcore_file, cpu_only=cpu_only).eval()

    logging.debug("*** Mcore model restored config")
    logging.debug(OmegaConf.to_yaml(mcore_model.cfg))

    nemo_summary = nemo_model.summarize()
    mcore_summary = mcore_model.summarize()

    logging.info("Sanity checks:")

    # check num weights match
    assert nemo_summary.total_parameters == mcore_summary.total_parameters, "❌ total parameters do not match"
    assert nemo_summary.model_size == mcore_summary.model_size, "❌ model sizes do not match"
    logging.info("✅ Number of weights match")

    # check weights match
    mcore_state_dict = mcore_model.state_dict()
    nemo_state_dict = nemo_model.state_dict()
    with open_dict(nemo_model.cfg):
        nemo_model.cfg.megatron_amp_O2 = False  # we want build_key_mapping in the next line to not use O2 prefix
    for mcore_param, nemo_param in build_key_mapping(nemo_model.cfg).items():
        try:
            mcore_weight = mcore_state_dict.pop(mcore_param)
            nemo_weight = nemo_state_dict.pop(nemo_param)
            if mcore_param.endswith("linear_fc1.weight"):
                # linear_fc1.weight should map to concat(dense_h_to_4h.weight, dense_h_to_4h_2.weight)
                # but build_key_mapping only maps it to dense_h_to_4h.weight, so we handle the concat here.
                second_param = nemo_param.replace("dense_h_to_4h.weight", "dense_h_to_4h_2.weight")
                if second_param in nemo_state_dict:
                    nemo_weight = torch.cat([nemo_weight, nemo_state_dict.pop(second_param)])
            assert torch.allclose(mcore_weight, nemo_weight), f"❌ parameter {mcore_param} does not match"
        except KeyError:
            buffers = [k for k, v in mcore_model.named_buffers()]
            assert (
                mcore_param in buffers
                or mcore_param.replace('model.', 'model.module.', 1) in buffers
                or any(mcore_param.endswith(suffix) for suffix in ignore_if_missing)
            ), f"❌ parameter {mcore_param} is not found in the state dict or named_buffers()"
            nemo_state_dict.pop(nemo_param)

    logging.info("✅ Weights match")

    # check for unexpected weights in state dict
    assert (
        len([k for k in nemo_state_dict if not k.endswith('_extra_state')]) == 0
    ), f"❌ unexpected items in nemo_state_dict: {nemo_state_dict}"
    assert (
        len([k for k in mcore_state_dict if not k.endswith('_extra_state')]) == 0
    ), f"❌ unexpected items in mcore_state_dict: {mcore_state_dict}"
    logging.info("✅ No unexpected weights in state dicts")


if __name__ == '__main__':
    args = get_args()

    input_nemo_file = args.input_name_or_path
    output_nemo_file = args.output_path
    cpu_only = args.cpu_only
    overwrite = args.overwrite
    ignore_if_missing = {key.strip() for key in args.ignore_if_missing.split(",")}

    os.makedirs(os.path.dirname(output_nemo_file), exist_ok=True)
    try:
        convert(
            input_nemo_file,
            output_nemo_file,
            skip_if_output_exists=not overwrite,
            cpu_only=cpu_only,
            ignore_if_missing=ignore_if_missing,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error("Could not convert due to torch.cuda.OutOfMemoryError.")
        logging.error("Please run the script with --cpu-only flag")
        exit(1)
    torch.cuda.empty_cache()
    try:
        run_sanity_checks(input_nemo_file, output_nemo_file, cpu_only=cpu_only, ignore_if_missing=ignore_if_missing)
    except torch.cuda.OutOfMemoryError:
        logging.info(
            "✅ Conversion was successful, but could not run sanity check due to torch.cuda.OutOfMemoryError."
        )
        logging.info("Please run the script with the same command again to run sanity check.")
