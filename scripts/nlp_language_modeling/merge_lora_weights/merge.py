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

Example usage:
python scripts/nlp_language_modeling/merge_lora_weights/merge.py \
    trainer.accelerator=gpu \   (use 'cpu' if model cannot fit in memory)
    tensor_model_parallel_size=<TP of lora checkpoint> \
    pipeline_model_parallel_size=1 \
    gpt_model_file=<path to base model nemo file or extracted folder> \
    lora_model_path=<path to megatron_gpt_peft_lora_tuning.nemo> \
    merged_model_path=<output nemo file>

TP of lora checkpoint can be found by visually examining the output of
`tar -tvf /path/to/lora.nemo`
"""


import os
import tempfile
from typing import Any, Dict

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
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
        config_file = f"{tmpdir}/model_config.yaml"
        lora_config = OmegaConf.load(config_file)
        return lora_state_dict, lora_config


def fix_for_O2(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model.module." not in k:
            new_state_dict[k.replace('model.', 'model.module.')] = v
    return new_state_dict


def merge(
    base_model_state_dict: Dict[str, Any], lora_state_dict: Dict[int, Any], tp: int, num_layers: int, mcore: bool,
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
    mcore_layer_to_lora = {}
    mcore_layer_to_lora["attention_qkv"] = {
        "base_model_layer": "self_attention.linear_qkv.weight",
        "lora_in": "self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight",
        "lora_out": "self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight",
    }
    mcore_layer_to_lora["attention_dense"] = {
        "base_model_layer": "self_attention.linear_proj.weight",
        "lora_in": "self_attention.adapter_layer.lora_dense_attention_adapter.linear_in.weight",
        "lora_out": "self_attention.adapter_layer.lora_dense_attention_adapter.linear_out.weight",
    }
    mcore_layer_to_lora["mlp_fc1"] = {
        "base_model_layer": "mlp.linear_fc1.weight",
        "lora_in": "mlp.adapter_layer.lora_hto4h_adapter.linear_in.weight",
        "lora_out": "mlp.adapter_layer.lora_hto4h_adapter.linear_out.weight",
    }
    mcore_layer_to_lora["mlp_fc2"] = {
        "base_model_layer": "mlp.linear_fc2.weight",
        "lora_in": "mlp.adapter_layer.lora_4htoh_adapter.linear_in.weight",
        "lora_out": "mlp.adapter_layer.lora_4htoh_adapter.linear_out.weight",
    }

    if mcore:
        for nl in range(num_layers):
            for key in mcore_layer_to_lora.keys():
                key_base = f'model.decoder.layers.{nl}.{mcore_layer_to_lora[key]["base_model_layer"]}'
                key_lora_in = f'model.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_in"]}'
                key_lora_out = f'model.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_out"]}'
                if key_lora_in in lora_state_dict[0] and key_lora_out in lora_state_dict[0]:
                    if key in ["attention_qkv", 'mlp_fc1']:
                        wt_lora_in = torch.cat([lora_state_dict[_tp][key_lora_in] for _tp in range(tp)], dim=0).float()
                    else:
                        wt_lora_in = torch.cat([lora_state_dict[_tp][key_lora_in] for _tp in range(tp)], dim=1).float()

                    wt_lora_out = torch.cat([lora_state_dict[_tp][key_lora_out] for _tp in range(tp)], dim=0).float()
                    wt_base = base_model_state_dict[key_base]
                    wt_lora = wt_lora_out @ wt_lora_in
                    base_model_state_dict[key_base] = (wt_base.float() + wt_lora.to(wt_base.device)).type_as(wt_base)
                    print(f'merging for weight {key_base}')
    else:
        logging.warning("Non-mcore model only supports merging lora weights for attention_qkv layers")
        for nl in range(num_layers):
            key_self_attn_kqv = f'model.language_model.encoder.layers.{nl}.self_attention.query_key_value.weight'
            key_lora_in = f'model.language_model.encoder.layers.{nl}.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight'
            key_lora_out = f'model.language_model.encoder.layers.{nl}.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight'

            wt_lora_in = torch.cat([lora_state_dict[_tp][key_lora_in] for _tp in range(tp)], dim=0).float()
            wt_lora_out = torch.cat([lora_state_dict[_tp][key_lora_out] for _tp in range(tp)], dim=0).float()
            wt_self_attn = base_model_state_dict[key_self_attn_kqv]
            wt_lora = wt_lora_out @ wt_lora_in
            base_model_state_dict[key_self_attn_kqv] = (
                wt_self_attn.float() + wt_lora.to(wt_self_attn.device)
            ).type_as(wt_self_attn)
            print("merging for weight", key_self_attn_kqv)

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
            pretrained_cfg.use_cpu_initialization = cfg.trainer.accelerator == 'cpu'
            pretrained_cfg["apply_rope_fusion"] = False
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            map_location=torch.device("cpu") if cfg.trainer.accelerator == 'cpu' else None,
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

    # load the lora weights on cpu for all ranks of the lora model
    lora_weights, lora_model_cfg = load_lora(cfg.lora_model_path, cfg.tensor_model_parallel_size)

    # merge the lora weights with the base model, for this current rank.
    merged_weights = merge(
        model.state_dict(),
        lora_weights,
        tp=cfg.tensor_model_parallel_size,
        num_layers=model.cfg.num_layers,
        mcore=model.mcore_gpt,
    )

    # load the merged_weights back into the base model, for this current rank.
    if model.cfg.megatron_amp_O2:
        merged_weights = fix_for_O2(merged_weights)
    model.cfg.use_cpu_initialization = (
        False  # set it back to False otherwise the merged model won't be loaded properly for futher tuning
    )
    model.load_state_dict(merged_weights)

    if cfg.trainer.accelerator != 'cpu' and model.global_rank == 0:
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
    else:
        logging.info("Skipping inference validation of merged model since device is 'cpu'.")

    model.save_to(cfg.merged_model_path)
    logging.info(f"saved merged model to {cfg.merged_model_path}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
