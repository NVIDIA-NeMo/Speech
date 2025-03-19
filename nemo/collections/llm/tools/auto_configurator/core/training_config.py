# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import List, Tuple

from nemo.collections.llm.tools.auto_configurator.core import utils


GPT_BASED_MODELS = [
    "gpt3",
    "bert",
    "llama",
    "qwen",
    "mixtral",
    "mistral",
    "gemma",
    "nemotron",
    "starcoder",
]


def generate_grid_search_configs(
    base_cfg: dict,
    train_cfg: dict,
) -> Tuple[dict, dict]:
    """Generates the grid of all possible configurations for the given model,
        and stores each different configuration in a yaml file.

    Args:
        base_cfg (dict): base configuration of the model to be trained.
        train_cfg (dict): train configuration of the model to be trained.

    Returns:
        dict: base config.
        dict: generated configs.
    """

    model_name = train_cfg.model_type
    model_size_in_b = train_cfg.model_size_in_b
    path_to_logs = train_cfg.path_to_logs

    # 2 * num_layers is needed because of encoder/decoder architecture.
    multiplier = 1 if model_name in GPT_BASED_MODELS else 2

    seq_length = base_cfg.model.config.seq_length
    num_layers = base_cfg.model.config.num_layers

    act_method = None

    params = _calculate_tp_pp_mbs_grid(
        model_size_in_b=model_size_in_b,
        num_layers=num_layers,
        model_name=model_name,
        seq_length=seq_length,
        train_cfg=train_cfg,
    )

    max_steps = train_cfg.max_steps_per_run
    num_nodes = train_cfg.num_nodes

    valid_tp_pp_list = []
    for tp in params.tp:
        for pp in params.pp:
            for cp in params.cp:
                for ep in params.ep:
                    for mbs in params.mbs:
                        num_gpus = base_cfg.trainer.num_nodes * base_cfg.trainer.devices
                        base_cfg.data.global_batch_size = params.gbs
                        att_heads = base_cfg.model.config.num_attention_heads
                        num_layers = base_cfg.model.config.num_layers
                        model_parallelism = (tp * pp * cp * ep) if (cp and ep) else (tp * pp)
                        mod_gbs = params.gbs % (mbs * num_gpus / model_parallelism)
                        mod_att_heads = att_heads % tp
                        mod_layers = (multiplier * num_layers) % pp
                        mod_cp = cp if cp else 1
                        mod_ep = ep if ep else 1
                        if (
                            mod_gbs == 0
                            and mod_att_heads == 0
                            and mod_layers == 0
                            and (tp, pp, cp, ep) not in valid_tp_pp_list
                            and (mod_cp // mod_ep == mod_cp or mod_ep // mod_cp == mod_ep)
                            and params.min_model_parallel <= model_parallelism <= params.max_model_parallel
                        ):
                            valid_tp_pp_list.append((tp, pp, cp, ep))

    # Generate grid search configs.
    configs = {}
    for tp, pp, cp, ep in valid_tp_pp_list:
        (
            virtual_pipelines,
            act_ckpt_layers,
            num_micro_batches_partial_act_ckpt,
            act_ckpt_layers_per_pipeline,
        ) = _set_activations_checkpoint_params(
            tp,
            pp,
            cp,
            ep,
            num_layers,
            act_method,
            multiplier,
            model_size_in_b,
            model_name,
        )
        for mbs in params.mbs:
            kwargs = {
                "base_cfg": base_cfg,
                "act": None,
                "num_mbs_act": None,
                "act_per_pipe": None,
                "tp": tp,
                "pp": pp,
                "cp": cp,
                "ep": ep,
                "virtual_pipelines": virtual_pipelines,
                "mbs": mbs,
                "max_steps": max_steps,
                "num_nodes": num_nodes,
                "path_to_logs": path_to_logs,
                "model_name": model_name,
                "model_size": model_size_in_b,
            }

            act_layers = None
            if act_ckpt_layers[0] is not None:
                if act_layers is not None and act_layers != "auto":
                    act_ckpt_layers = act_layers
                for act in act_ckpt_layers:
                    for num_mbs_act in num_micro_batches_partial_act_ckpt:
                        for act_per_pipe in act_ckpt_layers_per_pipeline:
                            kwargs["act"] = act
                            kwargs["num_mbs_act"] = num_mbs_act
                            kwargs["act_per_pipe"] = act_per_pipe
                            new_cfg = utils.modify_cfg(**kwargs)
                            if new_cfg:  # Save candidate cfg.
                                configs[new_cfg["name"]] = new_cfg
            else:
                new_cfg = utils.modify_cfg(**kwargs)
                if new_cfg:  # Save candidate cfg.
                    config_name = new_cfg["name"]
                    configs[config_name] = new_cfg

    print(f"\nAll candidate configurations created correctly. Total number of configs: {len(configs)}.\n")
    return base_cfg, configs


def _set_activations_checkpoint_params(
    tp, pp, cp, ep, num_layers, act_method, multiplier, model_size_in_b, model_name
):
    act_multiple = 4 // pp
    if act_method == "block":
        if 1.0 <= model_size_in_b < 11.3:
            act_multiple = 8 // pp
        elif 11.3 <= model_size_in_b < 26.0:
            act_multiple = 16 // pp
        elif 26.0 <= model_size_in_b < 60.0:
            act_multiple = 16 // pp
        elif 60.0 <= model_size_in_b:
            act_multiple = 32 // pp
    act_multiple = max(act_multiple, 1)

    virtual_pipelines = None
    # Num micro batches with partial act ckpt
    min_micro_b = 0  # 0 will not be used, minimum will be set to 1 later in the code.
    max_micro_b = pp
    interval_micro_b = 1
    # Act ckpt layers per pipeline
    min_layers_per_pipe = 0
    max_layers_per_pipe = num_layers
    interval_layers_per_pipe = act_multiple
    if model_name in GPT_BASED_MODELS and pp > 2:  # Interleaved pipeline scheduling.
        virtual_pipelines = num_layers // pp  # TODO: verify that this is the best value.
        act_multiple = 1
        max_micro_b = pp * (virtual_pipelines - 1) + (pp - 1) * 2 + 1
        interval_micro_b = virtual_pipelines * 8
        max_layers_per_pipe = multiplier * num_layers // pp // virtual_pipelines + 1

    (
        act_ckpt_layers,
        num_micro_batches_partial_act_ckpt,
        act_ckpt_layers_per_pipeline,
    ) = ([None], [None], [None])
    if act_method == "block":
        # Act ckpt num layers
        if virtual_pipelines is None:
            act_ckpt_layers = range(0, multiplier * num_layers // pp + 1, act_multiple)
        else:
            act_ckpt_layers = range(0, multiplier * num_layers // pp // virtual_pipelines + 1, act_multiple)

        if pp > 1 and model_name in GPT_BASED_MODELS:
            # Num micro batches with partial act ckpt
            num_micro_batches_partial_act_ckpt = list(range(min_micro_b, max_micro_b + 1, interval_micro_b))
            if num_micro_batches_partial_act_ckpt[0] == 0:
                num_micro_batches_partial_act_ckpt[0] = 1

            # Act ckpt layers per pipeline
            act_ckpt_layers_per_pipeline = range(
                min_layers_per_pipe, max_layers_per_pipe + 1, interval_layers_per_pipe
            )

    return (
        virtual_pipelines,
        act_ckpt_layers,
        num_micro_batches_partial_act_ckpt,
        act_ckpt_layers_per_pipeline,
    )


@dataclass
class GPT3GridSearch:
    """Selects grid search space for TP, PP, CP, EP, MBS parameters for GPT-3 and 80GB GPUs.

    Args:
        model_size_in_b (float): number of parameters in the model.
        valid_pp (List[int]): list of valid Pipeline Parallelism (PP) values for this config.
        seq_length (int): sequence length to use for training.
        gpu_memory_gb (int): size of GPU memory in GB.
    """

    model_size_in_b: int
    valid_pp: List[int]
    seq_length: int
    gpu_memory_gb: int

    tp = [1, 2, 4, 8]
    pp = [1]
    cp = [1]
    ep = [1]
    mbs = [1, 2, 4, 8]

    gbs: int = 1024
    min_model_parallel: int = 1
    max_model_parallel: int = 8

    def init_params(self):
        """Initialize model parallelism parameters"""

        model_size_in_b = self.model_size_in_b
        gpu_memory_gb = self.gpu_memory_gb
        seq_length = self.seq_length

        if gpu_memory_gb == 80:
            if seq_length == 2048:
                if model_size_in_b <= 1.0:
                    self.tp = [1, 2]
                    self.gbs = 256
                elif model_size_in_b <= 4.0:
                    self.tp = [1, 2, 4]
                    self.gbs = 1024
                elif model_size_in_b <= 8.0:
                    self.tp = [1, 2, 4]
                    self.gbs = 2048
                elif model_size_in_b <= 13.0:
                    self.tp = [1, 2, 4, 8]
                    self.gbs = 2048
                elif model_size_in_b <= 23.0:
                    self.tp = [1, 2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 4]
                    self.mbs = [1, 2, 4]
                    self.min_model_parallel = 4
                    self.max_model_parallel = 8
                    self.gbs = 2048
                elif model_size_in_b <= 45.0:
                    self.tp = [2, 4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 4]
                    self.mbs = [1, 2, 4]
                    self.min_model_parallel = 8
                    self.max_model_parallel = 32
                    self.gbs = 2048
                elif model_size_in_b <= 95:
                    self.tp = [2, 4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 8]
                    self.mbs = [1, 2, 4, 8]
                    self.min_model_parallel = 8
                    self.max_model_parallel = 64
                    self.gbs = 2048
                elif model_size_in_b <= 130.0:
                    self.tp = [2, 4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 16]
                    self.mbs = [1, 2, 4, 8]
                    self.min_model_parallel = 16
                    self.max_model_parallel = 128
                    self.gbs = 2048
                elif model_size_in_b <= 195.0:
                    self.tp = [8]
                    self.pp = [x for x in self.valid_pp if 4 <= x <= 16]
                    self.mbs = [1, 2, 4]
                    self.min_model_parallel = 32
                    self.max_model_parallel = 256
                    self.gbs = 2048
                elif model_size_in_b <= 395.0:
                    self.tp = [8]
                    self.pp = [x for x in self.valid_pp if 8 <= x <= 32]
                    self.mbs = [1, 2, 4]
                    self.min_model_parallel = 64
                    self.max_model_parallel = 512
                    self.gbs = 2048
                elif model_size_in_b <= 790.0:
                    self.tp = [8]
                    self.pp = [x for x in self.valid_pp if 8 <= x <= 100]
                    self.mbs = [1, 2, 4]
                    self.min_model_parallel = 128
                    self.max_model_parallel = 1024
                    self.gbs = 2048
                elif model_size_in_b <= 1100.0:
                    self.tp = [8]
                    self.pp = [x for x in self.valid_pp if 16 <= x <= 130]
                    self.mbs = [1, 2, 4]
                    self.min_model_parallel = 256
                    self.max_model_parallel = 2048
                    self.gbs = 2048
            elif seq_length == 4096:
                if model_size_in_b <= 1.0:
                    self.tp = [1, 2, 4]
                    self.mbs = [1, 2, 4, 8]
                    self.gbs = 128
                elif model_size_in_b <= 4.0:
                    self.tp = [1, 2, 4]
                    self.mbs = [1, 2, 4, 8]
                    self.gbs = 512
                elif model_size_in_b <= 8.0:
                    self.tp = [1, 2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1, 2, 4]
                    self.gbs = 1024
                elif model_size_in_b <= 13.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1, 2, 4]
                    self.gbs = 1024
                elif model_size_in_b <= 23.0:
                    self.tp = [4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1, 2]
                    self.min_model_parallel = 4
                    self.max_model_parallel = 16
                    self.gbs = 1024
                elif model_size_in_b <= 45.0:
                    self.tp = [4, 8]
                    self.pp = [x for x in self.valid_pp if 2 <= x <= 4]
                    self.mbs = [1, 2]
                    self.min_model_parallel = 8
                    self.max_model_parallel = 32
                    self.gbs = 1024
                elif model_size_in_b <= 95:
                    self.tp = [4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 8]
                    self.mbs = [1, 2]
                    self.min_model_parallel = 8
                    self.max_model_parallel = 64
                    self.gbs = 1024
            elif seq_length == 8192:
                if model_size_in_b <= 1.0:
                    self.tp = [1, 2]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1, 2, 4]
                    self.gbs = 64
                elif model_size_in_b <= 4.0:
                    self.tp = [1, 2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1, 2, 4]
                    self.gbs = 128
                elif model_size_in_b <= 8.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1, 2]
                    self.gbs = 256
                elif model_size_in_b <= 13.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1, 2]
                    self.gbs = 256
                elif model_size_in_b <= 23.0:
                    self.tp = [4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 4]
                    self.mbs = [1]
                    self.min_model_parallel = 8
                    self.max_model_parallel = 32
                    self.gbs = 256
                elif model_size_in_b <= 45.0:
                    self.tp = [8]
                    self.pp = [x for x in self.valid_pp if 4 <= x <= 8]
                    self.mbs = [1]
                    self.min_model_parallel = 32
                    self.max_model_parallel = 64
                    self.gbs = 256
            elif seq_length == 16384:
                if model_size_in_b <= 1.0:
                    self.tp = [2, 4]
                    self.mbs = [1, 2]
                    self.gbs = 32
                elif model_size_in_b <= 4.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1]
                    self.gbs = 64
                elif model_size_in_b <= 8.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1]
                    self.gbs = 128
                elif model_size_in_b <= 13.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1]
                    self.gbs = 128
                elif model_size_in_b <= 23.0:
                    self.tp = [4, 8]
                    self.pp = [x for x in self.valid_pp if 2 <= x <= 4]
                    self.mbs = [1]
                    self.min_model_parallel = 8
                    self.max_model_parallel = 32
                    self.gbs = 128
            elif seq_length == 32768:
                if model_size_in_b <= 1.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1]
                    self.gbs = 16
                elif model_size_in_b <= 4.0:
                    self.tp = [2, 4]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.mbs = [1]
                    self.gbs = 32
                elif model_size_in_b <= 8.0:
                    self.tp = [4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.min_model_parallel = 4
                    self.max_model_parallel = 16
                    self.mbs = [1]
                    self.gbs = 64
                elif model_size_in_b <= 13.0:
                    self.tp = [4, 8]
                    self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                    self.min_model_parallel = 4
                    self.max_model_parallel = 16
                    self.mbs = [1]
                    self.gbs = 64
                elif model_size_in_b <= 23.0:
                    self.tp = [8]
                    self.pp = [x for x in self.valid_pp if 2 <= x <= 4]
                    self.mbs = [1]
                    self.min_model_parallel = 16
                    self.max_model_parallel = 32
                    self.gbs = 64
        elif gpu_memory_gb == 40:
            if model_size_in_b <= 1.0:
                self.tp = [1, 2, 4]
                self.mbs = [1, 2, 4, 8]
                self.gbs = 256
            elif model_size_in_b <= 4.0:
                self.tp = [1, 2, 4, 8]
                self.mbs = [1, 2, 4, 8]
                self.gbs = 1024
            elif model_size_in_b <= 8.0:
                self.tp = [2, 4, 8]
                self.pp = [1, 2]
                self.mbs = [1, 2, 4]
                self.min_model_parallel = 2
                self.gbs = 2048
            elif model_size_in_b <= 13.0:
                self.tp = [4, 8]
                self.pp = [1, 2, 4]
                self.mbs = [1, 2, 4]
                self.min_model_parallel = 4
                self.max_model_parallel = 32
                self.gbs = 2048
            elif model_size_in_b <= 23.0:
                self.tp = [2, 4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 8]
                self.min_model_parallel = 8
                self.max_model_parallel = 64
                self.gbs = 2048
            elif model_size_in_b <= 45.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 12]
                self.mbs = [1, 2, 4]
                self.min_model_parallel = 16
                self.max_model_parallel = 128
                self.gbs = 2048
            elif model_size_in_b <= 95:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 16]
                self.mbs = [1, 2, 4]
                self.min_model_parallel = 16
                self.max_model_parallel = 256
                self.gbs = 2048
            elif model_size_in_b <= 130.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 2 <= x <= 26]
                self.mbs = [1, 2]
                self.min_model_parallel = 32
                self.max_model_parallel = 512
                self.gbs = 2048
            elif model_size_in_b <= 195.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 2 <= x <= 32]
                self.mbs = [1, 2]
                self.min_model_parallel = 64
                self.max_model_parallel = 1024
                self.gbs = 2048
            elif model_size_in_b <= 395.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 4 <= x <= 64]
                self.mbs = [1, 2]
                self.min_model_parallel = 128
                self.max_model_parallel = 2048
                self.gbs = 2048
            elif model_size_in_b <= 790.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 8 <= x <= 128]
                self.mbs = [1, 2]
                self.min_model_parallel = 256
                self.max_model_parallel = 4096
                self.gbs = 2048
            elif model_size_in_b <= 1100.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 8 <= x <= 192]
                self.mbs = [1, 2]
                self.min_model_parallel = 512
                self.max_model_parallel = 8192
                self.gbs = 2048


@dataclass
class T5GridSearch:
    """Selects grid search space for TP, PP, MBS parameters for T5/mT5 and 80GB GPUs.

    Args:
        model_size_in_b (float): number of parameters in the model.
        valid_pp (List[int]): list of valid Pipeline Parallelism (PP) values for this config.
        seq_length (int): sequence length to use for training.
        gpu_memory_gb (int): size of GPU memory in GB.
    """

    model_size_in_b: int
    seq_length: int
    gpu_memory_gb: int
    valid_pp: List[int]

    tp = [1, 2, 4, 8]
    pp = [1]
    cp = [None]
    ep = [None]
    mbs = [1, 2, 4, 6, 8, 12, 16]

    gbs: int = 1920
    min_model_parallel: int = 1
    max_model_parallel: int = 8

    def init_params(self):
        """Initialize model parallelism parameters"""

        model_size_in_b = self.model_size_in_b
        gpu_memory_gb = self.gpu_memory_gb

        if gpu_memory_gb == 80:
            if model_size_in_b <= 1.0:
                self.tp = [1, 2]
                self.mbs = [16, 32, 64, 128]
                self.gbs = 2048
            elif model_size_in_b <= 4.0:
                self.tp = [1, 2, 4]
                self.mbs = [4, 6, 8, 12, 16, 24, 32, 48]
                self.gbs = 1920
            elif model_size_in_b <= 8.0:
                self.tp = [2, 4, 8]
                self.mbs = [4, 6, 8, 12, 16, 24, 32]
                self.gbs = 1920
            elif model_size_in_b <= 14.5:
                self.tp = [4, 8]
                self.mbs = [2, 4, 6, 8, 12, 16, 24]
                self.gbs = 1920
            elif model_size_in_b <= 25.9:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                self.mbs = [1, 2, 4, 6, 8]
                self.min_model_parallel = 4
                self.max_model_parallel = 16
                self.gbs = 1920
            elif model_size_in_b <= 43.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 4]
                self.mbs = [1, 2, 4, 6, 8]
                self.min_model_parallel = 8
                self.max_model_parallel = 32
                self.gbs = 1920
            elif model_size_in_b <= 85.5:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 2 <= x <= 8]
                self.mbs = [1, 2, 4, 6, 8]
                self.min_model_parallel = 16
                self.max_model_parallel = 64
                self.gbs = 1920
            elif model_size_in_b <= 165.5:
                self.tp = [8]
                self.pp = [x for x in self.valid_pp if 4 <= x <= 16]
                self.mbs = [1, 2, 4, 6]
                self.min_model_parallel = 32
                self.max_model_parallel = 128
                self.gbs = 1920
            elif model_size_in_b <= 250:
                self.tp = [8]
                self.pp = [x for x in self.valid_pp if 4 <= x <= 32]
                self.mbs = [1, 2, 4, 6, 8]
                self.min_model_parallel = 64
                self.max_model_parallel = 256
                self.gbs = 1920
        elif gpu_memory_gb == 40:
            if model_size_in_b <= 1.0:
                self.tp = [1, 2]
                self.mbs = [16, 32, 64, 128]
                self.gbs = 2048
            elif model_size_in_b <= 4.0:
                self.tp = [1, 2, 4]
                self.mbs = [4, 8, 12, 16, 24, 32, 48]
                self.gbs = 1920
            elif model_size_in_b <= 8.0:
                self.tp = [2, 4, 8]
                self.mbs = [4, 6, 8, 12, 16, 24]
                self.gbs = 1920
            elif model_size_in_b <= 14.5:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 2]
                self.mbs = [2, 4, 6, 8, 12, 16]
                self.min_model_parallel = 4
                self.max_model_parallel = 16
                self.gbs = 1920
            elif model_size_in_b <= 25.9:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 8]
                self.mbs = [1, 2, 4, 6, 8]
                self.min_model_parallel = 8
                self.max_model_parallel = 32
                self.gbs = 1920
            elif model_size_in_b <= 43.0:
                self.tp = [4, 8]
                self.pp = [x for x in self.valid_pp if 1 <= x <= 8]
                self.mbs = [1, 2, 4, 6, 8]
                self.min_model_parallel = 16
                self.max_model_parallel = 32
                self.gbs = 1920
            elif model_size_in_b <= 85.5:
                self.tp = [8]
                self.pp = [x for x in self.valid_pp if 2 <= x <= 8]
                self.mbs = [1, 2, 4, 6, 8]
                self.min_model_parallel = 32
                self.max_model_parallel = 64
                self.gbs = 1920
            elif model_size_in_b <= 165.5:
                self.tp = [8]
                self.pp = [x for x in self.valid_pp if 4 <= x <= 32]
                self.mbs = [1, 2, 4]
                self.min_model_parallel = 64
                self.max_model_parallel = 128
                self.gbs = 1920
            elif model_size_in_b <= 250:
                self.tp = [8]
                self.pp = [x for x in self.valid_pp if 8 <= x <= 64]
                self.mbs = [1, 2, 4]
                self.min_model_parallel = 128
                self.max_model_parallel = 256
                self.gbs = 1920


@dataclass
class BertGridSearch:
    """Selects grid search space for TP, PP, MBS parameters for BERT and 80GB GPUs.

    Args:
        model_size_in_b (float): number of parameters in the model.
        valid_pp (List[int]): list of valid Pipeline Parallelism (PP) values for this config.
        seq_length (int): sequence length to use for training.
        gpu_memory_gb (int): size of GPU memory in GB.
    """

    model_size_in_b: int
    seq_length: int
    gpu_memory_gb: int
    valid_pp: List[int]

    tp = [1, 2, 4, 8]
    pp = [1]
    cp = [None]
    ep = [None]
    mbs = [1, 2, 4, 6, 8, 12, 16]

    gbs: int = 1920
    min_model_parallel: int = 1
    max_model_parallel: int = 8

    def init_params(self):
        """Initialize model parallelism parameters"""

        model_size_in_b = self.model_size_in_b
        gpu_memory_gb = self.gpu_memory_gb

        if gpu_memory_gb == 80:
            if model_size_in_b <= 1.0:
                self.tp = [1, 2]
                self.gbs = 256
            elif model_size_in_b <= 4.0:
                self.tp = [1, 2, 4]
                self.gbs = 1024
            elif model_size_in_b <= 8.0:
                self.tp = [2, 4, 8]
                self.min_model_parallel = 2
                self.gbs = 2048
            elif model_size_in_b <= 13.0:
                self.tp = [2, 4, 8]
                self.mbs = [1, 2, 3, 4, 6]
                self.min_model_parallel = 2
                self.gbs = 2048
            elif model_size_in_b <= 25.0:
                self.tp = [4, 8]
                self.mbs = [1, 2, 3, 4]
                self.min_model_parallel = 4
                self.gbs = 2048
            elif model_size_in_b <= 46.5:
                self.tp = [4, 8]
                self.pp = [1, 2, 4]
                self.mbs = [1, 2, 3, 4]
                self.min_model_parallel = 4
                self.max_model_parallel = 16
                self.gbs = 2048
            elif model_size_in_b <= 87.5:
                self.tp = [4, 8]
                self.pp = [2, 4, 6, 8]
                self.mbs = [1, 2, 3, 4]
                self.min_model_parallel = 8
                self.max_model_parallel = 32
                self.gbs = 2048
            elif model_size_in_b <= 165.5:
                self.tp = [4, 8]
                self.pp = [4, 6, 8, 16]
                self.mbs = [2, 4, 6, 8]
                self.min_model_parallel = 16
                self.max_model_parallel = 128
                self.gbs = 2048
            elif model_size_in_b <= 250.5:
                self.tp = [8]
                self.pp = [4, 8, 16, 32]
                self.mbs = [1, 2, 3, 4]
                self.min_model_parallel = 32
                self.max_model_parallel = 256
                self.gbs = 2048
            else:
                raise ValueError("No BERT model larger than 250B parameters is supported.")
        elif gpu_memory_gb == 40:
            if model_size_in_b <= 1.0:
                self.tp = [1, 2, 4]
                self.gbs = 256
            elif model_size_in_b <= 4.0:
                self.tp = [1, 2, 4, 8]
                self.gbs = 1024
            elif model_size_in_b <= 8.0:
                self.tp = [2, 4, 8]
                self.mbs = [1, 2, 4]
                self.gbs = 2048
            elif model_size_in_b <= 13.0:
                self.tp = [2, 4, 8]
                self.mbs = [1, 2, 4]
                self.gbs = 2048
            elif model_size_in_b <= 25.0:
                self.tp = [2, 4, 8]
                self.pp = [1, 2]
                self.mbs = [1, 2, 4]
                self.min_model_parallel = 2
                self.max_model_parallel = 16
                self.gbs = 2048
            elif model_size_in_b <= 46.5:
                self.tp = [4, 8]
                self.pp = [1, 2, 4, 8]
                self.mbs = [1, 2, 3]
                self.min_model_parallel = 8
                self.max_model_parallel = 32
                self.gbs = 2048
            elif model_size_in_b <= 87.5:
                self.tp = [4, 8]
                self.pp = [2, 4, 6, 8]
                self.mbs = [1, 2, 3]
                self.min_model_parallel = 16
                self.max_model_parallel = 64
                self.gbs = 2048
            elif model_size_in_b <= 165.5:
                self.tp = [8]
                self.pp = [4, 6, 8, 16]
                self.mbs = [1, 2]
                self.min_model_parallel = 32
                self.max_model_parallel = 256
                self.gbs = 2048
            elif model_size_in_b <= 250.5:
                self.tp = [8]
                self.pp = [8, 16, 32]
                self.mbs = [1, 2]
                self.min_model_parallel = 64
                self.max_model_parallel = 512
                self.gbs = 2048
            else:
                raise ValueError("No BERT model larger than 250B parameters is supported.")


def _calculate_tp_pp_mbs_grid(
    model_size_in_b: float,
    num_layers: int,
    model_name: str,
    seq_length: int,
    train_cfg: dict,
) -> Tuple[int, int, int]:
    """Selects grid search space for TP, PP, MBS parameters for any model,
        and calls the necessary heuristics function accordingly.

    Args:
        model_size_in_b (float): number of parameters in the model.
        num_layers (int): number of layers in the model config.
        model_name (str): name of the model to be used, such as gpt3, t5, mt5...
        seq_length (int): sequence length to use for training.
        train_cfg (dict): config of the model that will be launched.

    Returns:
        dataclass object with model parallelism parameters.

    Raises:
        NotImplementedError: if the model_name is not one of the supported models.
    """

    tp_sizes = train_cfg.tensor_parallel_sizes
    pp_sizes = train_cfg.pipeline_parallel_sizes
    cp_sizes = train_cfg.context_parallel_sizes
    ep_sizes = train_cfg.expert_parallel_sizes
    min_model_parallel_size = train_cfg.min_model_parallel_size
    max_model_parallel_size = train_cfg.max_model_parallel_size
    mbs_sizes = train_cfg.micro_batch_sizes
    gbs_size = train_cfg.global_batch_size
    gpu_memory_gb = train_cfg.gpu_memory_gb
    multiplier = 1 if model_name in GPT_BASED_MODELS else 2
    init_pp = [] if model_name in GPT_BASED_MODELS else [1]
    valid_pp = init_pp + [
        multiplier * x for x in range(1, num_layers + 1) if num_layers % x == 0
    ]  # Only divisors of num_layers are possible.

    kwargs = {
        "model_size_in_b": model_size_in_b,
        "valid_pp": valid_pp,
        "seq_length": seq_length,
        "gpu_memory_gb": gpu_memory_gb,
    }

    if model_name in GPT_BASED_MODELS:
        search_class = GPT3GridSearch
    elif model_name in ["t5", "mt5"]:
        search_class = T5GridSearch
    elif model_name == "bert":
        search_class = BertGridSearch
    else:
        raise NotImplementedError("Model name not implemented.")

    params = search_class(**kwargs)
    params.init_params()

    # Override the tp, pp, mbs search if indicated in the config params.
    if tp_sizes is not None and tp_sizes != "auto":
        params.tp = tp_sizes
    if pp_sizes is not None and pp_sizes != "auto":
        params.pp = pp_sizes
    if cp_sizes is not None and cp_sizes != "auto":
        params.cp = cp_sizes
    if ep_sizes is not None and ep_sizes != "auto":
        params.ep = ep_sizes
    if mbs_sizes is not None and mbs_sizes != "auto":
        params.mbs = mbs_sizes
    if gbs_size is not None and gbs_size != "auto":
        params.gbs = gbs_size
    if min_model_parallel_size is not None and min_model_parallel_size != "auto":
        params.min_model_parallel = min_model_parallel_size
    if max_model_parallel_size is not None and max_model_parallel_size != "auto":
        params.max_model_parallel = max_model_parallel_size

    return params
