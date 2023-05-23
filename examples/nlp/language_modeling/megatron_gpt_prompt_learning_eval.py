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

import os

import torch
import torch.multiprocessing as mp
from megatron.core import parallel_state
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner

mp.set_start_method("spawn", force=True)

"""
This is the script to run GPT text generation.
    a. run greedy inference from a p-tuned/prompt-tuned model's nemo file:
        python megatron_gpt_prompt_learning_eval.py \
            virtual_prompt_model_file=PATH_TO_NEMO_PROMPT_LEARNING_MODEL_FILE \
            gpt_model_file=PATH_TO_FROZEN_GPT_MODEL_FILE \
            inference.greedy=True \
            inference.add_BOS=False \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=1 \
            pred_file_path=PATH_WHERE_PRED_TEXT_FILE_WILL_BE_SAVED \
            data_paths=[path/to/dataset1.jsonl, path/to/dataset2.jsonl]

        virtual_prompt_model_file should be a path to a .nemo file saved after p-tuning/prompt tuning and model file
        is still the path to the gpt model's .nemo file.         

        data_paths should be a list of .json or .jsonl files containing json objects similar to the ones 
        used during prompt learning. They should have keys that match the fields specified in the prompt template.
        Fields can be dropped from the prompt dict and their corresponding section of the prompt template will 
        be automatically removed. 

        For example, say the prompt template during p-tuning/prompt-tuning looked like:

        '<|VIRTUAL_PROMPT_0|> Context: {context} Question: {question} Answer: {answer}'

        but you don't want to include the answer field during inference. Just don't 
        include the answer field in the prompt dict like below:

        {"taskname": "squad", "context": "some paragraph", "question": "question related to paragraph"}
        {"taskname": "squad", "context": "another paragraph", "question": "a different question related to paragraph"}

        And the dataset class will automatically format your input to have the form:

        [
            '<|VIRTUAL_PROMPT_0|> Context: some paragraph Question: question related to paragraph Answer:',
            '<|VIRTUAL_PROMPT_0|> Context: another paragraph Question: a different question related to paragraph Answer:'
        ]

        Similarly for other senarios, just add virtual_prompt_model_file=PATH_TO_NEMO_PROMPT_LEARNING_MODEL_FILE if you're using a 
        p-tuned/prompt-tuned model. 
"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_prompt_learning_inference")
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is needed for the inference")

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file
        model_config = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    # Update frozen GPT model path if it is given in case it has changed
    prompt_learning_cfg = MegatronGPTPromptLearningModel.restore_from(
        cfg.virtual_prompt_model_file, trainer=trainer, return_config=True,
    )
    if cfg.get("gpt_model_file"):
        with open_dict(prompt_learning_cfg):
            prompt_learning_cfg.language_model_path = cfg.gpt_model_file
            prompt_learning_cfg.sequence_parallel = False
            prompt_learning_cfg.activations_checkpoint_method = None
            prompt_learning_cfg.activations_checkpoint_granularity = None
            prompt_learning_cfg.activations_checkpoint_num_layers = None

    # Load prompt tuned model, virtual_prompt_model_file must be provided in config
    # Now load prompt learning model with frozen gpt model base

    model = MegatronGPTPromptLearningModel.restore_from(
        restore_path=cfg.virtual_prompt_model_file, trainer=trainer, override_config_path=prompt_learning_cfg,
    )
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.frozen_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def placeholder():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(placeholder, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
    }

    max_seq_length = model.frozen_model.cfg.encoder_seq_length - length_params["max_length"]
    max_seq_length = min(max_seq_length, cfg.get("max_seq_length", 8192))

    _, dataloader = model.build_virtual_prompt_dataset(
        data=cfg.data_paths,
        batch_size=cfg.inference.get('batch_size', 1),
        max_seq_length=max_seq_length,
        min_seq_length=model.cfg.data.get('min_seq_length', 1),
        add_bos=sampling_params["add_BOS"],
        add_eos=False,
        for_train=False,
        tokens_to_generate=length_params["max_length"],
        drop_last=False,
        shuffle=False,
        num_workers=cfg.get("num_workers", 1),
    )

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, dataloader)

    print("***************************")
    with open(cfg.pred_file_path, "w", encoding="utf-8") as pred_file:
        for i in range(len(response)):
            for sent in response[i]["sentences"]:
                sent = sent.strip()
                sent = sent.replace("\n", " ")
                pred_file.write(sent + "\n")
    print(f"Inference Complete, prediction file saved at {cfg.pred_file_path}")
    print("***************************")


if __name__ == '__main__':
    main()
