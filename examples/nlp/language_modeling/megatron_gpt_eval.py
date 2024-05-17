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

import asyncio
import datetime
import os
import threading
from functools import partial

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

"""
This is the script to run GPT text generation.

Usage:
    Assume the model has TP=1, PP=1 in the following use cases.
    a. run greedy inference from a nemo file:
        python megatron_gpt_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[prompt1,prompt2]

    b. run greedy inference from a PTL checkpoint file:
        python megatron_gpt_eval.py \
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
        python megatron_gpt_eval.py \
            gpt_model_file=PATH_TO_MODEL \
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
         python megatron_gpt_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            inference.compute_logprob=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompts=[text to get logprob]

    e. Launch the inference server
         python megatron_gpt_eval.py \
            gpt_model_file=PATH_TO_MODEL \
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

    def __len__(
        self,
    ):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def remove_padded_prompts(response, nb_paddings):
    result = {}
    for k, v in response.items():
        if v != None and (type(v) is list or type(v) is torch.Tensor):
            v = v[:-nb_paddings]
        result[k] = v
    return result


@hydra_runner(config_path="conf", config_name="megatron_gpt_inference")
def main(cfg) -> None:

    callbacks = []
    # enable_progress_bar is True by default. If cfg.trainer.enable_progress_bar=False, CustomProgressBar is not appended to callbacks
    if 'enable_progress_bar' not in cfg.trainer or cfg.trainer.enable_progress_bar:
        callbacks.append(CustomProgressBar())
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=callbacks,
    )

    if cfg.gpt_model_file is not None:
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

            # with dist checkpointing we don't need to set this
            if not model_config.get('mcore_gpt', False):
                with open_dict(cfg):
                    cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
                    cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
                    cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size
        * cfg.pipeline_model_parallel_size
        * max(1, cfg.get('expert_model_parallel_size', 1))
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            pretrained_cfg["use_flash_attention"] = cfg.inference.get("use_flash_attention", False)
            pretrained_cfg["apply_rope_fusion"] = False
            if pretrained_cfg.get('mcore_gpt', False):
                # with dist checkpointing we can use the model parallel config specified by the user
                pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
                pretrained_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
                pretrained_cfg.expert_model_parallel_size = cfg.get('expert_model_parallel_size', 1)
                pretrained_cfg.micro_batch_size = 1
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
            elif trainer.precision in ['bf16', 'bf16-mixed'] and cfg.get('megatron_amp_O2', False):
                pretrained_cfg.megatron_amp_O2 = True
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f'cuda:{trainer.local_rank}',  # map_location is needed for converted models
        )
    elif cfg.checkpoint_dir:
        app_state = AppState()
        if (
            cfg.tensor_model_parallel_size > 1
            or cfg.pipeline_model_parallel_size > 1
            or cfg.get('expert_model_parallel_size', 1) > 1
        ):
            app_state.model_parallel_size = (
                cfg.tensor_model_parallel_size
                * cfg.pipeline_model_parallel_size
                * cfg.get('expert_model_parallel_size', 1)
            )
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            app_state.expert_model_parallel_size = cfg.get('expert_model_parallel_size', 1)
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.expert_model_parallel_rank,
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
                expert_model_parallel_size_=cfg.get('expert_model_parallel_size', 1),
            )
        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
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

    fp8_enabled = hasattr(model.cfg, "fp8") and (model.cfg.fp8 == True)
    if fp8_enabled:
        nb_paddings = 0
        while len(cfg.prompts) % 8 != 0:
            cfg.prompts.append("")
            nb_paddings += 1

    # First method of running text generation, call model.generate method
    response = model.generate(
        inputs=OmegaConf.to_container(cfg.prompts), length_params=length_params, sampling_params=sampling_params
    )

    if fp8_enabled:
        response = remove_padded_prompts(response, nb_paddings)
    print("***************************")
    print(response)
    print("***************************")

    # Second method of running text generation, call trainer.predict [recommended]
    bs = 8 if fp8_enabled else 2
    ds = RequestDataSet(OmegaConf.to_container(cfg.prompts))
    request_dl = DataLoader(dataset=ds, batch_size=bs)
    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)

    if fp8_enabled:
        response[-1] = remove_padded_prompts(response[-1], nb_paddings)
    print("***************************")
    print(response)
    print("***************************")

    # Third method of running text generation, use inference server
    if cfg.server:
        from nemo.collections.nlp.modules.common.megatron_web_server import get_chatbot_demo, get_demo

        if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
            if cfg.web_server:
                if cfg.chat:
                    defaults = {
                        'user': cfg.chatbot_config.user,
                        'assistant': cfg.chatbot_config.assistant,
                        'system': cfg.chatbot_config.system,
                    }
                    web_ui = partial(
                        get_chatbot_demo,
                        defaults=defaults,
                        value=cfg.chatbot_config.value,
                        attributes=cfg.chatbot_config.attributes,
                    )
                else:
                    web_ui = get_demo
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=web_ui,
                    daemon=True,
                    args=(cfg.share, cfg.username, cfg.password, cfg.port, cfg.web_port, loop),
                )
                thread.start()
            server = MegatronServer(model.cuda())
            server.run("0.0.0.0", port=cfg.port)

        while True:
            choice = torch.cuda.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            if choice[0].item() == 0:
                generate(model.cuda())


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
