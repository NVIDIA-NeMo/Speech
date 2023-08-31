#!/usr/bin/env
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

"""
Merge lora weights into a base GPT LM. Only PP=1 supported so far.
"""


import os
import tempfile
from typing import Any, Dict

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import MegatronGPTLoRAModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def load_lora(lora_nemo, tp):
    lora_state_dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        NLPSaveRestoreConnector._unpack_nemo_file(lora_nemo, tmpdir)
        # assert os.path.isdir(lora_extracted_dir), "requires the untar'ed the lora .nemo file"
        for i in range(tp):
            if tp == 1:
                ckpt_file = f"{tmpdir}/model_weights.ckpt"
            else:
                ckpt_file = f"{tmpdir}/mp_rank_0{i}/model_weights.ckpt"

            l = torch.load(ckpt_file, map_location=torch.device('cpu'))
            lora_state_dict[i] = l
        return lora_state_dict


def merge(
    base_model_state_dict: Dict[str, Any],
    lora_state_dict: Dict[int, Any],
    tp: int,
    num_layers: int,
    curr_rank: int,
    mcore: bool,
):
    """ 
    Iterate through all the self_attention.query_key_value projection feedforward weights in all the layers.
    Collect the corresponding lora weights for each layer and across tp ranks.
    Computes the "full rank" weight from the two low-rank weights and add it to the self_attention.query_key_value weight.
    Args:
        base_model_state_dict: A state_dict for the base model for the current rank.
        lora_state_dict: A complete set of weights for the lora model across all tp ranks. They key for this dict is an int tp rank.
        tp: the tensor_model_parallel_size for the base_model (and the lora model)
        num_layers: the number of layers in the base_model to iterate over.
        curr_rank: current tp rank of the base model which is being merged with Lora.
        mcore: whether the model uses megatron core.
    """

    for nl in range(num_layers):
        if mcore:
            key_self_attn_kqv = f'model.decoder.layers.{nl}.self_attention.linear_qkv.weight'
            key_lora_in = f'model.decoder.layers.{nl}.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight'
            key_lora_out = f'model.decoder.layers.{nl}.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight'
        else:
            key_self_attn_kqv = f'model.language_model.encoder.layers.{nl}.self_attention.query_key_value.weight'
            key_lora_in = f'model.language_model.encoder.layers.{nl}.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight'
            key_lora_out = f'model.language_model.encoder.layers.{nl}.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight'
        wt_lora_in = torch.cat([lora_state_dict[_tp][key_lora_in] for _tp in range(tp)], dim=0)
        wt_lora_out = lora_state_dict[curr_rank][key_lora_out]
        wt_self_attn = base_model_state_dict[key_self_attn_kqv]
        wt_lora = wt_lora_out @ wt_lora_in
        base_model_state_dict[key_self_attn_kqv] = wt_self_attn + wt_lora.type_as(wt_self_attn)
    return base_model_state_dict


@hydra_runner(config_path="conf", config_name="merge_lora_weights")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        model_config = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file, trainer=trainer, return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
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
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            map_location=torch.device("cpu"),
            save_restore_connector=save_restore_connector,
        )
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
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    lora_model_cfg = MegatronGPTLoRAModel.restore_from(
        restore_path=cfg.lora_model_path, trainer=trainer, return_config=True, mcore=model.mcore_gpt,
    )

    # load the lora weights on cpu for all ranks of the lora model
    lora_weights = load_lora(cfg.lora_model_path, model.cfg.tensor_model_parallel_size)

    # merge the lora weights with the base model, for this current rank.
    merged_weights = merge(
        model.state_dict(),
        lora_weights,
        tp=model.cfg.tensor_model_parallel_size,
        num_layers=model.cfg.num_layers,
        curr_rank=model.global_rank,
    )

    # load the merged_weights back into the base model, for this current rank.
    model.load_state_dict(merged_weights)

    # Going to go through the motions of inference to force PTL to run subprocess for loading all base model's ranks.
    input = "Context: In 2004, philosopher and psychologist Michel ter Hark (Groningen, The Netherlands) published a book, called Popper, Otto Selz and the rise of evolutionary epistemology, in which he claimed that Popper took some of his ideas from his tutor, the German psychologist Otto Selz. Selz never published his ideas, partly because of the rise of Nazism, which forced him to quit his work in 1933, and the prohibition of referring to Selz' work. Popper, the historian of ideas and his scholarship, is criticised in some academic quarters for his rejection of Plato, Hegel and Marx. Question: Who claimed Otto Selz deserved credit for ideas published by Popper? Answer:"
    ds = RequestDataSet([input])
    request_dl = DataLoader(dataset=ds, batch_size=1)
    config = {'greedy': True, 'compute_logprob': False, 'tokens_to_generate': 5, 'add_BOS': False}
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)
    print(response)

    with open_dict(model.cfg):
        model.cfg.restore_from_path = cfg.merged_model_path
        model.cfg.data = lora_model_cfg.data
        model.cfg.target = f"{MegatronGPTSFTModel.__module__}.{MegatronGPTSFTModel.__name__}"

    model.save_to(cfg.merged_model_path)
    logging.info(f"saved merged model to {cfg.merged_model_path}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
