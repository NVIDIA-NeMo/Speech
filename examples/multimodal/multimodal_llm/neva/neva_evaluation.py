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

import json
import os

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import Dataset

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank


try:
    import ammo.torch.quantization as atq

    HAVE_AMMO = True

except (ImportError, ModuleNotFoundError):

    HAVE_AMMO = False


"""
This is the script to run GPT text generation.

Usage:
    Assume the model has TP=1, PP=1 in the following use cases.
    a. run greedy inference from a nemo file:
        python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[prompt1,prompt2]

    b. run greedy inference from a PTL checkpoint file:
        python neva_evaluation.py \
            checkpoint_dir=PATH_TO_CHECKPOINT_FILE \
            checkpoint_name=CHECKPOINT_FILE_NAME \
            hparams_file=HPARAMS_FILE \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[prompt1,prompt2]

    c. run top_p inference from a nemo file:
        python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            inference.greedy=False \
            inference.top_k=0 \
            inference.top_p=0.9 \
            inference.repetition_penalty=1.2 \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[prompt1,prompt2]

    d. If you don't need to generate tokens and need model to compute logprobs:
         python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            inference.compute_logprob=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[text to get logprob]

    e. Launch the inference server
         python neva_evaluation.py \
            neva_model_file=PATH_TO_MODEL \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            server=True
        
        To send a request to the server, here is one example code:
        ```python
        import json
        import requests

        batch_size = 8
        port_num = 5555
        headers = {"Content-Type": "application/json"}


        def request_data(data):
            resp = requests.put('http://localhost:{}/generate'.format(port_num),
                                data=json.dumps(data),
                                headers=headers)
            sentences = resp.json()['sentences']
            return sentences


        data = {
            "sentences": [""] * batch_size,
            "images" : [] * batch_size,
            "tokens_to_generate": 300,
            "temperature": 1.0,
            "add_BOS": True,
            "top_k": 0,
            "top_p": 0.9,
            "greedy": False,
            "all_probs": False,
            "repetition_penalty": 1.2,
            "min_tokens_to_generate": 2,
        }

        sentences = request_data(data)
        ```
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


@hydra_runner(config_path="conf", config_name="neva_inference")
def main(cfg) -> None:

    plugins = []
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())
    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=plugins, strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        model_config = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file, trainer=trainer, return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.neva_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.neva_model_file):
            save_restore_connector.model_extracted_dir = cfg.neva_model_file

        neva_cfg = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(neva_cfg, True)
        with open_dict(neva_cfg):
            neva_cfg.sequence_parallel = False
            neva_cfg.activations_checkpoint_granularity = None
            neva_cfg.activations_checkpoint_method = None
            neva_cfg.precision = trainer.precision
            neva_cfg.mm_cfg.llm.from_pretrained = cfg.get('llm_model_file', None)
        #    neva_cfg.mm_cfg.vision_encoder.from_pretrained = None

        model = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file,
            trainer=trainer,
            override_config_path=neva_cfg,
            save_restore_connector=save_restore_connector,
        )
        if neva_cfg.get('peft') is not None:
            peft_cfg_cls = PEFT_CONFIG_MAP[neva_cfg.peft.peft_scheme]
            if peft_cfg_cls is not None:
                model.load_adapters(cfg.neva_model_file, peft_cfg_cls(neva_cfg))

    elif cfg.checkpoint_dir:
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            )
        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        # TODO: This wont work properly (We need to set model.llm.from_pretrained model.vision.from_pretrained to nul)
        model = MegatronNevaModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    try:
        model.model.module.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

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
        "end_strings": cfg.inference.end_strings,
    }

    with open(cfg.prompt_file, 'r') as f:
        lines = f.readlines()

    final_prompts = []
    for line in lines:
        prompt_dict = json.loads(line)
        final_prompts.append(prompt_dict)

    responses = model.generate(
        input_prompts=final_prompts, length_params=length_params, sampling_params=sampling_params, inference_config=cfg
    )

    # =================== Start Quantization ====================
    if HAVE_AMMO and cfg.quantization.enable == True:
        print(f"Using quantization algorithm: {cfg.quantization.algorithm}")
        if cfg.quantization.algorithm == "int8_sq":
            atq_config = atq.INT8_SMOOTHQUANT_CFG
        elif cfg.quantization.algorithm == "fp8":
            atq_config = atq.FP8_DEFAULT_CFG
        elif cfg.quantization.algorithm == "awq":
            atq_config = atq.INT4_AWQ_CFG
        else:
            raise ValueError(f"Unsupported quantization algorithm: {cfg.quantization.algorithm}")

        def forward_loop():
            model.generate(
                input_prompts=final_prompts,
                length_params=length_params,
                sampling_params=sampling_params,
                inference_config=cfg,
            )

        atq.quantize(model, atq_config, forward_loop)

        responses = model.generate(
            input_prompts=final_prompts,
            length_params=length_params,
            sampling_params=sampling_params,
            inference_config=cfg,
        )
    # ============== Quantization End =========================

    results = []
    for response, prompt in zip(responses, final_prompts):
        prompt['full_text'] = response["clean_text"]
        prompt['text'] = response["clean_response"]
        prompt['model_id'] = cfg.neva_model_file
        prompt['answer_id'] = 0
        prompt['metadata'] = {}
        results.append(prompt)

    with open(cfg.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    """ 
    # Second method of running text generation, call trainer.predict
    ds = RequestDataSet(final_prompts)
    request_dl = DataLoader(dataset=ds, batch_size=1)
    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print("***************************")
    """


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
