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


GPT_BASED_MODELS = [
    "gpt3",
    "llama",
    "qwen",
    "mixtral",
    "mistral",
    "gemma",
    "nemotron",
    "starcoder",
]


@dataclass
class ModelSizeParams:
    """Calculates the parameters that affect model_size: hidden size, attention heads,
        KV channels, and FFN size. It also calculates the learning rate.

    Args:
        model_size_in_b (float): number of parameters in the desired model config, in billions.
        vocab_size (int): size of the vocabulary to use for training.
        seq_length (int): sequence length to be used during training.
        model_name (str): name of the model to be trained, i.e. gpt3, t5, mt5...

    Raises:
        ValueError: if the model size is larger than the max supported model size.
        NotImplementedError: if the model name is not supported.
    """

    model_size_in_b: float
    vocab_size: int
    seq_length: int
    model_name: str

    # Model size params
    layers: int = None
    hs: int = None
    att_h: int = None
    ffn: int = None
    kv: int = None
    lr: float = None

    def init_params(self):
        "Initialize model architecure."
        model_name = self.model_name
        model_size_in_b = self.model_size_in_b
        if model_name in GPT_BASED_MODELS:
            if model_size_in_b < 0.25:
                self.hs, self.att_h, self.lr = 768, 12, 6e-4
            elif model_size_in_b < 0.5:
                self.hs, self.att_h, self.lr = 1024, 16, 3e-4
            elif model_size_in_b < 1:
                self.hs, self.att_h, self.lr = 1536, 16, 2.5e-4
            elif model_size_in_b < 2:
                self.hs, self.att_h, self.lr = 2048, 16, 2e-4
            elif model_size_in_b < 3:
                self.hs, self.att_h, self.lr = 2560, 32, 1.6e-4
            elif model_size_in_b < 4.5:
                self.hs, self.att_h, self.lr = 3072, 32, 1.4e-4
            elif model_size_in_b < 8:
                self.hs, self.att_h, self.lr = 4096, 32, 1.2e-4
            elif model_size_in_b < 15:
                self.hs, self.att_h, self.lr = 5120, 40, 1e-4
            elif model_size_in_b < 25:
                self.hs, self.att_h, self.lr = 6144, 48, 1e-4
            elif model_size_in_b < 52:
                self.hs, self.att_h, self.lr = 8192, 64, 0.8e-4
            elif model_size_in_b < 105:
                self.hs, self.att_h, self.lr = 10240, 80, 0.7e-4
            elif model_size_in_b < 205:
                self.hs, self.att_h, self.lr = 12288, 96, 0.6e-4
            elif model_size_in_b < 405:
                self.hs, self.att_h, self.lr = 20480, 128, 0.5e-4
            elif model_size_in_b < 805:
                self.hs, self.att_h, self.lr = 20480, 128, 0.4e-4
            elif model_size_in_b < 1105:
                self.hs, self.att_h, self.lr = 25600, 160, 0.3e-4
            else:
                raise ValueError("Model_size for GPT-3 must be smaller than 1.1T parameters.")
        elif model_name == "t5":
            self.kv, self.lr = 64, 1e-4
            if model_size_in_b < 0.1:
                self.hs, self.att_h, self.ffn = 512, 6, 1024
            elif model_size_in_b < 0.4:
                self.hs, self.att_h, self.ffn = 768, 12, 2048
            elif model_size_in_b < 1:
                self.hs, self.att_h, self.ffn = 1024, 16, 2816
            elif model_size_in_b < 5:
                self.hs, self.att_h, self.ffn = 2048, 32, 5120
            elif model_size_in_b < 15:
                self.hs, self.att_h, self.ffn = 4096, 64, 10240
            elif model_size_in_b < 25.9:
                self.hs, self.att_h, self.ffn = 5120, 80, 10880
            elif model_size_in_b < 43.0:
                self.hs, self.att_h, self.ffn = 6144, 96, 10880
            elif model_size_in_b <= 85.5:
                self.hs, self.att_h, self.ffn = 6144, 96, 16384
            elif model_size_in_b <= 165.5:
                self.hs, self.att_h, self.ffn = 7680, 96, 20480
            elif model_size_in_b <= 250:
                self.hs, self.att_h, self.ffn = 12288, 96, 32768
            else:
                raise ValueError("Model_size for T5 must be smaller than 250B parameters.")
        elif model_name == "mt5":
            self.kv, self.lr = 64, 1e-4
            if model_size_in_b < 0.25:
                self.hs, self.att_h, self.ffn = 512, 6, 1024
            elif model_size_in_b < 0.5:
                self.hs, self.att_h, self.ffn = 768, 12, 2048
            elif model_size_in_b < 1.2:
                self.hs, self.att_h, self.ffn = 1024, 16, 2816
            elif model_size_in_b < 5:
                self.hs, self.att_h, self.ffn = 2048, 32, 5120
            elif model_size_in_b < 15:
                self.hs, self.att_h, self.ffn = 4096, 64, 10240
            elif model_size_in_b < 25.9:
                self.hs, self.att_h, self.ffn = 5120, 80, 10880
            elif model_size_in_b < 43.0:
                self.hs, self.att_h, self.ffn = 6144, 96, 10880
            elif model_size_in_b <= 85.5:
                self.hs, self.att_h, self.ffn = 6144, 96, 16384
            elif model_size_in_b <= 165.5:
                self.hs, self.att_h, self.ffn = 7680, 96, 20480
            elif model_size_in_b <= 250:
                self.hs, self.att_h, self.ffn = 12288, 96, 32768
            else:
                raise ValueError("Model_size for mT5 must be smaller than 250B parameters.")
        elif model_name == "bert":
            self.lr = 1e-4
            if model_size_in_b < 0.25:
                self.hs, self.att_h, self.lr = 768, 12, 2e-4
            elif model_size_in_b < 0.5:
                self.hs, self.att_h, self.lr = 1024, 16, 2e-4
            elif model_size_in_b < 1:
                self.hs, self.att_h = 1536, 16
            elif model_size_in_b < 2:
                self.hs, self.att_h = 2048, 16
            elif model_size_in_b < 3:
                self.hs, self.att_h = 2560, 32
            elif model_size_in_b < 4.5:
                self.hs, self.att_h = 2560, 32
            elif model_size_in_b < 8:
                self.hs, self.att_h = 4096, 32
            elif model_size_in_b < 15:
                self.hs, self.att_h = 5120, 40
            elif model_size_in_b <= 25:
                self.hs, self.att_h = 6144, 48
            elif model_size_in_b <= 46.5:
                self.hs, self.att_h = 7680, 48
            elif model_size_in_b <= 87.5:
                self.hs, self.att_h = 9216, 96
            elif model_size_in_b <= 165.5:
                self.hs, self.att_h = 9216, 96
            elif model_size_in_b <= 250.5:
                self.hs, self.att_h = 12288, 96
            else:
                raise ValueError("Model_size for BERT must be smaller than 250B parameters.")
            self.ffn = 4 * self.hs
        else:
            raise NotImplementedError("Model name is not valid.")

        # Try powers of 2
        margin = 0.01
        for attempt in range(0, 10):
            for layers in (2**p for p in range(1, 10)):
                out_size = _calculate_model_size(
                    vocab_size=self.vocab_size,
                    seq_length=self.seq_length,
                    hidden_size=self.hs,
                    num_layers=layers,
                    ffn_size=self.ffn,
                    kv_channels=self.kv,
                    att_heads=self.att_h,
                    model_name=self.model_name,
                )
                if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin) and not self.layers:
                    self.layers = layers
            margin += 0.01  # Double margin of acceptable model sizes.

        # Try multiples of 16
        margin = 0.01
        for attempt in range(0, 6):
            for layers in range(16, 201, 16):
                out_size = _calculate_model_size(
                    vocab_size=self.vocab_size,
                    seq_length=self.seq_length,
                    hidden_size=self.hs,
                    num_layers=layers,
                    ffn_size=self.ffn,
                    kv_channels=self.kv,
                    att_heads=self.att_h,
                    model_name=self.model_name,
                )
                if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin) and not self.layers:
                    self.layers = layers
            margin += 0.01  # Double margin of acceptable model sizes.

        # Try multiples of 2
        margin = 0.01
        for attempt in range(0, 6):
            for layers in range(2, 201, 2):
                out_size = _calculate_model_size(
                    vocab_size=self.vocab_size,
                    seq_length=self.seq_length,
                    hidden_size=self.hs,
                    num_layers=layers,
                    ffn_size=self.ffn,
                    kv_channels=self.kv,
                    att_heads=self.att_h,
                    model_name=self.model_name,
                )
                if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin) and not self.layers:
                    self.layers = layers
            margin += 0.01  # Double margin of acceptable model sizes.

        # Try multiples of 5
        margin = 0.01
        for attempt in range(0, 6):
            for layers in range(5, 201, 5):
                out_size = _calculate_model_size(
                    vocab_size=self.vocab_size,
                    seq_length=self.seq_length,
                    hidden_size=self.hs,
                    num_layers=layers,
                    ffn_size=self.ffn,
                    kv_channels=self.kv,
                    att_heads=self.att_h,
                    model_name=self.model_name,
                )
                if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin) and not self.layers:
                    self.layers = layers
            margin += 0.01  # Double margin of acceptable model sizes.

        # Try any valid number
        margin = 0.01
        for attempt in range(0, 10):
            for layers in range(1, 200):
                out_size = _calculate_model_size(
                    vocab_size=self.vocab_size,
                    seq_length=self.seq_length,
                    hidden_size=self.hs,
                    num_layers=layers,
                    ffn_size=self.ffn,
                    kv_channels=self.kv,
                    att_heads=self.att_h,
                    model_name=self.model_name,
                )
                if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin) and not self.layers:
                    self.layers = layers
            margin += 0.01  # Double margin of acceptable model sizes.

        if not self.layers:
            raise Exception("Number of layers not found, config is not possible.")


def _calculate_model_size(
    vocab_size: int = None,
    seq_length: int = None,
    hidden_size: int = None,
    num_layers: int = None,
    ffn_size: int = None,
    kv_channels: int = None,
    att_heads: int = None,
    model_name: str = "gpt3",
):
    """Calculates the model size (number of parameters in billions), given the model parameters and name.

    Args:
        vocab_size (int): vocabulary size to be used during training.
        seq_length (int): input sequence length to be used during training.
        hidden_size (int): size of the hidden layers of the model.
        num_layers (int): number of layers in the model.
        ffn_size (int): FFN size of the model.
        kv_channels (int): number of KV channels in the transformer layers.
        att_heads (int): number of attention heads in the transformer layers.
        model_name (str): name of the model, i.e gpt3, t5, mt5...

    Returns:
        float: size of the model in billions of parameters.

    Raises:
        NotImplementedError: if the model name is not valid.
    """

    if model_name in GPT_BASED_MODELS:
        model_size = (
            12
            * num_layers
            * hidden_size**2
            * (1 + (13 / (12 * hidden_size)) + ((vocab_size + seq_length) / (12 * num_layers * hidden_size)))
            / 1e9
        )
    elif model_name in ["t5", "mt5"]:
        # 2 L F + 3 L P + H (2 + 4 L F + L (21 + 12 P) + 1 S + 1 V)
        proj_size = att_heads * kv_channels
        model_size = (
            2 * num_layers * 1.5 * ffn_size
            + 3 * num_layers * proj_size
            + hidden_size
            * (2 + 4 * num_layers * 1.5 * ffn_size + num_layers * (21 + 12 * proj_size) + seq_length + vocab_size)
        ) / 1e9
    elif model_name == "bert":
        model_size = (
            num_layers * (ffn_size + hidden_size * (4 * hidden_size + 3 * att_heads + 2 * ffn_size + 6))
            + hidden_size * (vocab_size + seq_length + hidden_size + 5)
        ) / 1e9

    else:
        raise NotImplementedError("Model name is not valid.")

    return model_size


def generic_base_config(config) -> dict:
    """Generates a base config dictionary from a base config python file.

    Args:
        config (AutoConfigurator): config object for the Auto Configurator tool.

    Returns:
        BaseConfig: base configuration for the model.
        AutoConfigurator: config object for the Auto Configurator tool.
    """

    from nemo.collections.llm.tools.auto_configurator.core.base_config import calculate_model_size

    model_size_in_b = calculate_model_size(
        config.gpu_count,
        config.max_training_days,
        config.model_size_in_b,
        config.tflops_per_gpu,
        config.num_tokens_in_b,
        config.model_type,
    )
    base_cfg = config.recipe

    if config.calculate_model_size:
        params = ModelSizeParams(
            model_size_in_b,
            config.vocab_size,
            config.seq_length,
            config.model_type,
        )
        params.init_params()

        base_cfg.model.config.num_layers = params.layers
        base_cfg.model.config.hidden_size = params.hs
        base_cfg.model.config.num_attention_heads = params.att_h
        base_cfg.model.config.kv_channels = params.kv
        if not params.ffn:
            base_cfg.model.config.ffn_hidden_size = params.hs * 4
        else:
            base_cfg.model.config.ffn_hidden_size = params.ffn

    config.model_size_in_b = model_size_in_b

    return base_cfg, config


def modify_cfg(
    base_cfg: dict,
    act: int,
    num_mbs_act: int,
    act_per_pipe: int,
    tp: int,
    pp: int,
    cp: int,
    ep: int,
    virtual_pipelines: int,
    mbs: int,
    max_steps: int,
    num_nodes: int,
    model_name: str,
    path_to_logs: str,
    model_size: float,
) -> dict:
    """Modify the base configuration for the model with the new parameters that are specific to the current model,
        which the Auto Configurator tool heuristics selected.

    Args:
        base_cfg (dict): base configuration for the current model, which will be modified in this function.
        act (int): number of activation checkpointing layers to use for the model.
        num_mbs_act (int): sets the number of micro-batches where only a partial number of Transformer layers
            get checkpointed and recomputed within a window of micro-batches.
        act_per_pipe (int): sets the number of Transformer layers to skip checkpointing at later pipeline stages.
        tp (int): Tensor Parallelism (TP) value to be set for the model.
        pp (int): Pipeline Parallelism (PP) value to be set for the model.
        cp (int): Context Parallelism (CP) value to be set for the model.
        ep (int): Expert Parallelism (EP) value to be set for the model.
        virtual_pipelines (int): Virtual Pipelines value to be set for the model.
        mbs (int): Micro Batch Size (MBS) value to be set for the model.
        max_steps (int): maximum number of steps to run this model for.
        num_nodes (int): number of nodes to use for the training run.
        model_name (str): name of the model, i.e. gpt3, t5, mt5...
        path_to_logs (str): path to directory where logs will be saved.
        model_size (float): model size.

    Returns:
        dict: dictionary containing the updated model configuration parameters.
    """

    att_heads = base_cfg.model.config.num_attention_heads
    num_layers = base_cfg.model.config.num_layers

    # gbs = mbs * num_gpus * accumulate_grad_batches / (tp * pp)
    num_gpus = base_cfg.trainer.num_nodes * base_cfg.trainer.devices
    gbs = base_cfg.data.global_batch_size
    seq_len = base_cfg.model.config.seq_length

    new_cfg = {}  # dict(run=base_cfg.run)
    if act is not None:
        new_cfg["activations_checkpoint_num_layers"] = act

    if num_mbs_act is not None and model_name in GPT_BASED_MODELS:
        new_cfg["num_micro_batches_with_partial_activation_checkpoints"] = num_mbs_act

    if act_per_pipe is not None and model_name in GPT_BASED_MODELS:
        new_cfg["activations_checkpoint_layers_per_pipeline"] = act_per_pipe

    if virtual_pipelines is not None and model_name in GPT_BASED_MODELS:
        new_cfg["virtual_pipeline_model_parallel_size"] = virtual_pipelines

    new_cfg["tensor_model_parallel_size"] = tp
    new_cfg["pipeline_model_parallel_size"] = pp
    new_cfg["micro_batch_size"] = mbs
    new_cfg["global_batch_size"] = gbs
    new_cfg["max_steps"] = max_steps
    new_cfg["path_to_logs"] = path_to_logs

    if cp is not None:
        new_cfg["context_parallel_size"] = cp

    if ep is not None:
        new_cfg["expert_model_parallel_size"] = ep

    mod_gbs = gbs % (mbs * num_gpus / (tp * pp))
    mod_att_heads = att_heads % tp
    mod_layers = num_layers % pp
    if mod_gbs == 0 and mod_att_heads == 0 and mod_layers == 0:
        # Valid config
        new_cfg["name"] = (
            f"{model_name}_{str(model_size)}b_{num_nodes}nodes_"
            f"tp_{tp}_pp_{pp}_cp_{cp}_ep_{ep}_mbs_{mbs}_vp_{virtual_pipelines}"
        )
        print(
            f"Valid config: SeqLen={seq_len}, GBS={gbs}, MBS={mbs}, TP={tp}, PP={pp}, CP={cp}, EP={ep}, "
            f"VP={virtual_pipelines}. Adding to directory."
        )
        return new_cfg
    return None
