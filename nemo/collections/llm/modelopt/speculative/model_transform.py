# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
import torch.nn as nn

from nemo.collections.llm import GPTModel
from nemo.utils import logging
from nemo.utils.model_utils import unwrap_model

ALGORITHMS = {
    "eagle3": mtsp.EAGLE3_DEFAULT_CFG,
    # more TBD
}


def apply_speculative_decoding(model: nn.Module, algorithm: str = "eagle3") -> nn.Module:
    """Transform a model to enable Speculative Decoding using Model Optimizer.

    Args:
        model: The model to transform.
        algorithm: The algorithm to use for Speculative Decoding.
            (See https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/speculative/config.py)

    Returns:
        The transformed model.
    """
    assert algorithm in ALGORITHMS, f"Invalid algorithm: {algorithm}. Choices: {ALGORITHMS.keys()}"
    mode_cfg = ALGORITHMS[algorithm]
    mode, cfg = mode_cfg["algorithm"], mode_cfg["config"]

    assert isinstance(model, GPTModel), "Speculative Decoding currently only supported for GPT models."
    unwrapped_model = unwrap_model(model)

    # Check if the model has already been transformed with speculative decoding
    if _has_same_speculative_decoding_state(unwrapped_model, mode):
        logging.info("Model has already been transformed with Speculative Decoding. Skipping transformation.")
        return model

    # Verify model is compatible with speculative decoding
    assert hasattr(unwrapped_model, "config"), "Model must have a config attached."
    if unwrapped_model.config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Speculative decoding is incompatible with virtual pipeline parallelism.")

    # Adjust decoder head architecture
    if "eagle_architecture_config" in cfg:
        # These ones are necessary
        cfg["eagle_architecture_config"]["hidden_size"] = unwrapped_model.config.hidden_size
        cfg["eagle_architecture_config"]["vocab_size"] = unwrapped_model.vocab_size
        cfg["eagle_architecture_config"]["draft_vocab_size"] = unwrapped_model.vocab_size
        # These ones are optional but we copy base model's to scale memory usage reasonably
        cfg["eagle_architecture_config"]["intermediate_size"] = unwrapped_model.config.ffn_hidden_size
        cfg["eagle_architecture_config"]["num_attention_heads"] = unwrapped_model.config.num_attention_heads
        cfg["eagle_architecture_config"]["num_key_value_heads"] = unwrapped_model.config.num_query_groups

    # Convert
    logging.info(f"Converting to Speculative Decoding model with mode: '{mode}' and config:\n{cfg}")
    mtsp.convert(unwrapped_model, [(mode, cfg)])  # assumes in-place

    return model


def _has_same_speculative_decoding_state(model: nn.Module, mode: str) -> bool:
    """Check if the model has the same Speculative Decoding state as the incoming algorithm mode."""
    from modelopt.torch.opt.mode import _ModeRegistryCls

    mode_registry = _ModeRegistryCls.get_registry_by_name("speculative")
    modelopt_state = mto.modelopt_state(model)
    for _mode, _ in modelopt_state["modelopt_state_dict"]:
        if _mode in mode_registry:
            if _mode != mode:
                raise ValueError(
                    "Model has already been transformed with Speculative Decoding, but"
                    f" with a different mode: {_mode}. Please use the same mode."
                )
            return True
    return False
