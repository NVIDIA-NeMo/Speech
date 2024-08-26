# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import tempfile

import torch
from megatron.core import dist_checkpointing
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.multimodal.speech_llm.modules.perception_modules import AudioPerceptionModule
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import load_state_dict_helper
from nemo.utils import logging
from nemo.utils.model_utils import inject_model_parallel_rank


def get_config_and_state_dict_from_nemo(filepath, map_location, output_dir, sharded_state_dict=None):
    cwd = os.getcwd()
    save_restore_connector = NLPSaveRestoreConnector()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if os.path.isfile(filepath):
                save_restore_connector._unpack_nemo_file(path2file=filepath, out_folder=tmpdir)
            else:
                tmpdir = filepath

            os.chdir(tmpdir)
            config_yaml = "model_config.yaml"
            model_weights_ckpt = "model_weights.ckpt"

            # find file in tmpdir that endswith "tokenizer.model"
            tokenizer = None
            for file in os.listdir(tmpdir):
                if file.endswith("tokenizer.model"):
                    tokenizer = file
                    break
            if tokenizer is None:
                raise ValueError(f"Tokenizer not found in {tmpdir}")
            tokenizer_path = os.path.join(tmpdir, tokenizer)
            # copy tokenizer_path to current directory
            os.system(f"cp {tokenizer_path} {output_dir}")
            tokenizer_path = os.path.join(output_dir, tokenizer)

            # load conf
            with open(config_yaml) as f:
                conf = OmegaConf.load(f)

            os.chdir(cwd)
            model_weights = os.path.join(tmpdir, model_weights_ckpt)
            model_weights = inject_model_parallel_rank(model_weights)
            state_dict = save_restore_connector._load_state_dict_from_disk(model_weights, map_location=map_location)

            # distributed checkpointing
            if state_dict is None and sharded_state_dict is not None:
                checkpoint = dict(state_dict=sharded_state_dict)
                tmp_model_weights_ckpt = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
                tmp_model_weights_dir = os.path.splitext(tmp_model_weights_ckpt)[0]
                assert os.path.isdir(tmp_model_weights_dir), f'Expected {tmp_model_weights_dir} to be a directory.'
                checkpoint = dist_checkpointing.load(
                    sharded_state_dict=checkpoint,
                    checkpoint_dir=tmp_model_weights_dir,
                )
                state_dict = checkpoint["state_dict"]

            conf.tokenizer.model = tokenizer_path
            return conf, state_dict
        finally:
            os.chdir(cwd)


def get_llm_model_state_dict(state_dict, lora_model_state_dict):
    llm_model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            if key not in lora_model_state_dict and value != None:
                llm_model_state_dict[key] = value
    return llm_model_state_dict


def get_lora_state_dict(state_dict):
    lora_model_state_dict = {}
    for key, value in state_dict.items():
        if "adapter_layer.lora" in key and value != None:
            lora_model_state_dict[key] = value
    return lora_model_state_dict


def get_perception_state_dict(state_dict):
    perception_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("perception."):
            key = key.replace("perception.", "", 1)
            perception_state_dict[key] = value
    return perception_state_dict


def save_llm_model(state_dict, nemo_config, output_path):
    if nemo_config.get('megatron_amp_O2', False):
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key.replace('model.', 'model.module.', 1)] = state_dict['state_dict'].pop(key)

    trainer = Trainer(accelerator='cpu', strategy=NLPDDPStrategy())
    model = load_state_dict_helper(MegatronGPTModel, nemo_config, trainer, state_dict)
    model._save_restore_connector = NLPSaveRestoreConnector()
    model.cfg.use_cpu_initialization = False

    model.save_to(output_path)
    logging.info(f'llm model saved to: {output_path}')


def save_nemo_weights(state_dict, output_dir, config, save_nemo_model=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = os.path.join(output_dir, "model_weights.ckpt")
    torch.save(state_dict, weight_file)
    # convert config to yaml
    config_file = os.path.join(output_dir, "model_config.yaml")
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(config))

    if save_nemo_model:
        # create nemo file
        nemo_model_name = f"{output_dir}.nemo"
        nemo_path = os.path.join(output_dir, nemo_model_name)
        # tar model_config.yaml and model_weights.ckpt
        os.system(f"tar -C {output_dir} -cvf {nemo_path} model_config.yaml model_weights.ckpt")
        # remove model_config.yaml and model_weights.ckpt
        os.system(f"rm {config_file} {weight_file}")
        # remove the empty directory
        os.system(f"rmdir {output_dir}")


def separate_speechllm_model(model_file_path, output_dir, map_location="cuda:0"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.abspath(output_dir)

    logging.info(f"Separating {model_file_path} into perception, lora, and llm model")
    filepath = model_file_path
    conf, state_dict = get_config_and_state_dict_from_nemo(filepath, map_location, output_dir)

    base_model_name = os.path.basename(filepath).split(".")[0]

    perception_state_dict = get_perception_state_dict(state_dict)
    perception_model_dir = None
    if perception_state_dict:
        perception_model_dir = f"{base_model_name}_perception"
        perception_model_dir = os.path.join(output_dir, perception_model_dir)
        save_nemo_weights(perception_state_dict, perception_model_dir, conf.perception, save_nemo_model=False)

        # verify if the exported perception model is correct
        perception = AudioPerceptionModule(cfg=conf.perception)
        perception.load_state_dict(perception_state_dict)
        perception.eval()
        print(perception)
        print(perception(input_signal=torch.randn(1, 1000), input_signal_length=torch.tensor([1000])))
    # absolute path of perception model
    logging.info(f"Perception model saved to:  {perception_model_dir}")

    lora_model_weights = get_lora_state_dict(state_dict)
    lora_model_dir = None
    if lora_model_weights:
        lora_model_dir = f"{base_model_name}_lora"
        lora_model_dir = os.path.join(output_dir, lora_model_dir)
        save_nemo_weights(lora_model_weights, lora_model_dir, conf)
        logging.info(f"Lora model saved to: {lora_model_dir}.nemo")
    # hard code the target model for now
    llm_model_weights = get_llm_model_state_dict(state_dict, lora_model_weights)
    if llm_model_weights:
        llm_model = f"{base_model_name}_llm.nemo"
        llm_model = os.path.join(output_dir, llm_model)
        conf.target = "nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel"
        save_llm_model(llm_model_weights, conf, llm_model)
        logging.info(f"LLM model saved to: {llm_model}")


# filepath = "/ws/speechllm_fc_llama2_7b.nemo"
# output_dir = "/ws/speechllm_fc_llama2_7b_separated"
# perception_model_dir, lora_model, llm_model = separate_speechllm_model(filepath, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Separate speechllm model')
    parser.add_argument('--model_file_path', type=str, help='Path to the speechllm model')
    parser.add_argument('--output_dir', type=str, help='Output directory to save the separated models')
    args = parser.parse_args()
    separate_speechllm_model(args.model_file_path, args.output_dir)
