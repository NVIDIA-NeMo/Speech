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

import re
import torch
from modelopt.torch.quantization.nn import QuantLinear, QuantLinearConvBase

from nemo.collections.multimodal.modules.stable_diffusion.attention import LinearWrapper
from .plugin_calib import PercentileCalibrator


def filter_func(name):
    pattern = re.compile(r".*(emb_layers.1|time_embed|input_blocks.0.0|^out\.1$|skip_connection|label_emb).*")
    return pattern.match(name) is not None


class _QuantNeMoLinearWrapper(QuantLinearConvBase):
    default_quant_desc_weight = QuantLinear.default_quant_desc_weight


AXES_NAME = {
    "nemo": {
        "x": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        "timesteps": {0: "steps"},
        "y": {0: "batch_size"},
        "context": {0: "batch_size", 1: "sequence_length"},
    }
}


def generate_dummy_inputs(sd_version, device, latent_dim=32, adm_in_channels=1280):
    dummy_input = {}
    if sd_version == "nemo":
        dummy_input["x"] = torch.ones(2, 4, latent_dim, latent_dim).to(
            device
        )  ## latent dim should be sampling resolution // 8
        dummy_input["y"] = torch.ones(2, adm_in_channels).to(
            device
        )  ## adm_in_channels is 1280 when conditioner in consisted of only tokenizers, otherwise each additional embedding adds 512 in this dimention
        dummy_input["timesteps"] = torch.ones(2).to(device)
        dummy_input["context"] = torch.ones(2, 80, 2048).to(device)
    else:
        raise NotImplementedError(f"Unsupported sd_version: {sd_version}")

    return dummy_input


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    with open(calib_data_path, "r") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def get_int8_config(model, quant_level=3, alpha=0.8, percentile=1.0, num_inference_steps=20, global_min=False):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear, LinearWrapper)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "global_min": global_min,
                    },
                ),
            }
    return quant_config


def quantize_lvl(unet, quant_level=2.5):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current ModelOpt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear, LinearWrapper)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
