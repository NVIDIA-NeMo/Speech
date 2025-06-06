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

r"""
Script to convert NeMo 1.0 checkpoints to NeMo 2.0 format.
Available model listed in MODEL_CONFIG_MAPPING
Example usage:

a. Convert a .nemo checkpoint
    python /opt/NeMo/scripts/checkpoint_converters/convert_nemo1_to_nemo2.py \
        --input_path=Meta-Llama-3-8B.nemo \
        --output_path=your_output_dir \
        --model_id=meta-llama/Meta-Llama-3-8B

b. Convert a model weight directory.
   The checkpoint should be similar to `model_weights` subdir after extracting the .nemo file.
   Please also provide tokenizer_library and tokenizer_path when loading from weight directory.
    python /opt/NeMo/scripts/checkpoint_converters/convert_nemo1_to_nemo2.py \
        --input_path=nemotron3-8b-extracted/model_weights \
        --tokenizer_path=path_to_your_tokenizer_model.model \
        --tokenizer_library=sentencepiece \
        --output_path=your_output_dir \
        --model_id=nvidia/nemotron-3-8b-base-4k

"""

import os
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import torch
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject
from omegaconf import OmegaConf
from transformers import AutoTokenizer as HFAutoTokenizer

from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.utils import logging
from nemo.utils.model_utils import load_config

MODEL_CONFIG_MAPPING = {
    "meta-llama/Llama-2-7b-hf": (llm.LlamaModel, llm.Llama2Config7B),
    "meta-llama/Llama-2-13b-hf": (llm.LlamaModel, llm.Llama2Config13B),
    "meta-llama/Llama-2-70b-hf": (llm.LlamaModel, llm.Llama2Config70B),
    "meta-llama/Meta-Llama-3-8B": (llm.LlamaModel, llm.Llama3Config8B),
    "meta-llama/Meta-Llama-3-70B": (llm.LlamaModel, llm.Llama3Config70B),
    "mistralai/Mixtral-8x7B-v0.1": (llm.MixtralModel, llm.MixtralConfig8x7B),
    "mistralai/Mixtral-8x22B-v0.1": (llm.MixtralModel, llm.MixtralConfig8x22B),
    "mistralai/Mistral-7B-v0.1": (llm.MistralModel, llm.MistralConfig7B),
    "nvidia/nemotron-3-8b-base-4k": (llm.NemotronModel, llm.Nemotron3Config8B),
    "nemotron4-22b": (llm.NemotronModel, llm.Nemotron3Config22B),
    "nemotron4-15b": (llm.NemotronModel, llm.Nemotron4Config15B),
    "nemotron4-340b": (llm.NemotronModel, llm.Nemotron4Config340B),
    "nemotronh4b": (llm.MambaModel, llm.NemotronHConfig4B),
    "nemotronh8b": (llm.MambaModel, llm.NemotronHConfig8B),
    "nemotronh47b": (llm.MambaModel, llm.NemotronHConfig47B),
    "nemotronh56b": (llm.MambaModel, llm.NemotronHConfig56B),
}


def get_args():
    """
    Parse the command line arguments.
    """
    parser = ArgumentParser(
        description="""Script to convert NeMo 1.0 checkpoints to NeMo 2.0 format.
                    This script may download from Hugging Face, make sure you have
                    access to gate repo and have logged into Hugging Face (e.g. huggingface-cli login)"""
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="""Path to NeMo 1.0 checkpoints. Could be .nemo file, or `model_weights` directory a
        fter untar the .nemo. Please also provide tokenizer_library and tokenizer_path if you pass
        in `model_weights` directory.""",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, required=True, help="Path to output NeMo 2.0 directory."
    )
    parser.add_argument(
        "--model_id", type=str, default=None, required=True, help="Hugging Face or nemotron model id for the model"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        required=False,
        help="""Path to tokenizer. If not provided, will 1. try instantiate from nemo1 config
        2. pull AutoTokenizer from Hugging Face according to model_id if 1 fails""",
    )
    parser.add_argument(
        "--tokenizer_library",
        type=str,
        default=None,
        required=False,
        help="Tokenizer library, e.g. `sentencepiece`, `megatron`. Defaults to `sentencepiece`",
    )
    parser.add_argument(
        "--tokenizer_vocab_file",
        type=str,
        default=None,
        required=False,
        help="Tokenizer vocab file. Defaults to None",
    )
    parser.add_argument(
        "--tokenizer_model_name",
        type=str,
        default="TiktokenTokenizer",
        required=False,
        help="Tokenizer model name, e.g. TiktokenTokenizer. Defaults to TiktokenTokenizer",
    )
    args = parser.parse_args()
    return args


def load_fp8_config(model_path: str) -> Dict[str, Any]:
    """
    Loads fp8 configuration of the NeMo 1.0 model.

    Args:
        model_path (str): Path to NeMo 1.0 checkpoint.

    Returns:
        (dict): NeMo 1.0 model fp8 settings.
    """
    fp8_params = ['fp8', 'fp8_amax_history_len', 'fp8_interval', 'fp8_margin', 'fp8_amax_compute_algo']
    config = load_config(model_path)
    fp8_config = {key: config[key] for key in fp8_params if key in config}
    return fp8_config


def get_nemo2_model(model_id, tokenizer, input_path) -> llm.GPTModel:
    """
    Get NeMo 2.0 model class from model_id and tokenizer. Use bf16 for NeMo 1.0 ckpts.

    Returns:
        llm.GPTModel: NeMo 2.0 model instance
    """
    if os.path.isdir(model_id):
        from nemo.lightning import io

        model = io.load_context(Path(model_id), subpath="model")
        model.config.bf16 = True
        model.config.params_dtype = torch.bfloat16
        return model

    if model_id not in MODEL_CONFIG_MAPPING:
        valid_ids = "\n- ".join([""] + list(MODEL_CONFIG_MAPPING.keys()))
        raise ValueError(f"Unsupported model_id: {model_id}. Please provide a valid model_id from {valid_ids}")
    model_cls, config_cls = MODEL_CONFIG_MAPPING[model_id]

    fp8_config = load_fp8_config(input_path)
    # nemo1 ckpts are bf16
    return model_cls(config_cls(bf16=True, params_dtype=torch.bfloat16, **fp8_config), tokenizer=tokenizer)


def get_tokenizer(input_path: Path, tokenizer_tmp_dir: Path) -> AutoTokenizer:
    """
    Get tokenizer from input .nemo file, or args.tokenizer_path, or Hugging Face.
    Only SentencePiece and Hugging Face tokenizers are supported.

    Returns:
        AutoTokenizer: tokenizer instance
    """
    if args.tokenizer_vocab_file:
        return get_nmt_tokenizer(
            library=args.tokenizer_library,
            model_name=args.tokenizer_model_name,
            vocab_file=args.tokenizer_vocab_file,
            use_fast=True,
        )
    if not input_path.is_dir():  # if .nemo tar
        with tempfile.TemporaryDirectory() as tmp_dir:  # we want to clean up this tmp dir
            NLPSaveRestoreConnector._unpack_nemo_file(input_path, tmp_dir)
            cfg = OmegaConf.load(f"{tmp_dir}/model_config.yaml")
            tokenizer_lib = cfg.tokenizer.library
            tokenizer_model = cfg.tokenizer.get("model") and cfg.tokenizer.get("model").split("nemo:", 1)[-1]
            if tokenizer_model:
                shutil.copy(f"{tmp_dir}/{tokenizer_model}", f"{tokenizer_tmp_dir}/{tokenizer_model}")
            elif cfg.tokenizer.library == "huggingface":
                HFAutoTokenizer.from_pretrained(cfg.tokenizer.type).save_pretrained(tokenizer_tmp_dir)
            tokenizer_model = f"{tokenizer_tmp_dir}/{tokenizer_model}" if tokenizer_model else None
    else:
        if (
            args.tokenizer_path or args.tokenizer_vocab_file
        ):  # not .nemo file, only weight dir need to specify tokenizer lib and path
            tokenizer_lib = args.tokenizer_library or "sentencepiece"
            if args.tokenizer_library is None:
                logging.warning(
                    "You specified tokenizer_path but did not provide tokenizer_library using default sentencepiece"
                )
            tokenizer_model = args.tokenizer_path
        else:  # no .nemo config, no tokenizer path specified, grab from HF, reload
            tokenizer_lib = "huggingface"
            HFAutoTokenizer.from_pretrained(args.model_id).save_pretrained(tokenizer_tmp_dir)

    if tokenizer_lib == "huggingface":
        return AutoTokenizer(tokenizer_tmp_dir)
    else:  # not directly use huggingface tokenizer in get_nmt_tokenizer since it will pull from HF and no reload
        return get_nmt_tokenizer(library=tokenizer_lib, tokenizer_model=tokenizer_model)


def main() -> None:
    """
    Main function to convert NeMo 1.0 checkpoint to NeMo 2.0 format.
    """
    tokenizer_tmp_dir = Path("/tmp/nemo_tokenizer")
    tokenizer_tmp_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(Path(args.input_path), tokenizer_tmp_dir)
    model = get_nemo2_model(args.model_id, tokenizer=tokenizer, input_path=args.input_path)
    model.optim = None

    trainer = Trainer(
        devices=1,
        accelerator="cpu",
        strategy=MegatronStrategy(ddp="pytorch", setup_optimizers=False, plugins=bf16_mixed()),
    )

    trainer.strategy.connect(model)
    trainer.strategy.setup_environment()
    if not model.state_dict():
        with _strategy_lib.megatron_cpu_init_context(model.config):
            model.configure_model()

    trainer.strategy.setup(trainer)

    logging.info(f"loading checkpoint {args.input_path}")

    sharded_state_dict = {"state_dict": trainer.strategy.megatron_parallel.sharded_state_dict()}

    for key in list(sharded_state_dict['state_dict'].keys()):
        new_key = key.replace('module', 'model', 1)
        sharded_state_dict['state_dict'][new_key] = sharded_state_dict['state_dict'].pop(key)
        sharded_state_dict['state_dict'][new_key].key = sharded_state_dict['state_dict'][new_key].key.replace(
            'module', 'model', 1
        )

    def skip_fp8_load(x):
        if isinstance(x, ShardedObject) and 'core_attention' in x.key and '_extra_state' in x.key:
            x = LocalNonpersistentObject(x.data)  # use the FP8 state from initialization, not from ckpt
        return x

    dict_list_map_inplace(skip_fp8_load, sharded_state_dict)
    if not Path(args.input_path).is_dir():
        with tempfile.TemporaryDirectory() as tmp_dir:
            NLPSaveRestoreConnector._unpack_nemo_file(args.input_path, tmp_dir)
            model_weight_dir = f"{tmp_dir}/model_weights"
            model_ckpt = trainer.strategy.checkpoint_io.load_checkpoint(model_weight_dir, sharded_state_dict, None)
    else:
        model_ckpt = trainer.strategy.checkpoint_io.load_checkpoint(args.input_path, sharded_state_dict, None)

    logging.info(f"Saving checkpoint to {args.output_path}")
    model_ckpt['state_dict'] = {k.replace('model', 'module', 1): v for k, v in model_ckpt['state_dict'].items()}
    trainer.model.module.load_state_dict(model_ckpt['state_dict'])
    trainer.save_checkpoint(ckpt_to_weights_subdir(args.output_path, is_saving=False))
    if getattr(trainer.strategy, "async_save", False):
        trainer.strategy.checkpoint_io.maybe_finalize_save_checkpoint(blocking=True)

    # Corresponding to Connector: on_import_ckpt
    if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, '__io__'):
        trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
    TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(args.output_path), yaml_attrs=["model"])

    # remove tmp dir
    if os.path.isdir(tokenizer_tmp_dir):
        shutil.rmtree(tokenizer_tmp_dir)

    logging.info(f"NeMo 2.0 checkpoint saved at {args.output_path}")


if __name__ == '__main__':
    args = get_args()
    main()
