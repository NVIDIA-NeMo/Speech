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

import os
from pathlib import Path

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.quantization.nn import QuantModuleRegistry
from torch.onnx import export as onnx_export

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import MegatronDiffusionEngine
from nemo.collections.multimodal.modules.stable_diffusion.attention import LinearWrapper
from nemo.collections.multimodal.modules.stable_diffusion.quantization_utils.utils import (
    AXES_NAME,
    _QuantNeMoLinearWrapper,
    generate_dummy_inputs,
    get_int8_config,
    load_calib_prompts,
    quantize_lvl,
)
from nemo.collections.multimodal.parts.stable_diffusion.sdxl_pipeline import SamplingPipeline
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner
from nemo.utils.trt_utils import build_engine


def do_calibrate(base, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        base.text_to_image(
            params=kwargs['param_dict'],
            prompt=prompts,
            negative_prompt="",
            samples=kwargs['num_samples'],
            return_latents=False,
        )


def get_input_profile_unet(
    batch_size, static_batch=False, min_batch_size=1, max_batch_size=8, latent_dim=32, adm_in_channels=1280
):
    assert batch_size >= min_batch_size and batch_size <= max_batch_size
    if static_batch:
        min_batch_size = batch_size if static_batch else min_batch_size
        max_batch_size = batch_size if static_batch else max_batch_size
    input_profile = {}
    dummy_input = generate_dummy_inputs(
        sd_version="nemo", device='cuda', latent_dim=latent_dim, adm_in_channels=adm_in_channels
    )
    for key, value in dummy_input.items():
        input_profile[key] = [
            (min_batch_size, *(value.shape[1:])),
            (batch_size, *(value.shape[1:])),
            (max_batch_size, *(value.shape[1:])),
        ]
    return input_profile


@hydra_runner(config_path='conf', config_name='sd_xl_quantize')
def main(cfg):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None
        model_cfg.first_stage_config._target_ = 'nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKLInferenceWrapper'
        model_cfg.fsdp = False

    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronDiffusionEngine, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )

    model = megatron_diffusion_model.model
    model.cuda()
    base = SamplingPipeline(model, use_fp16=cfg.use_fp16, is_legacy=cfg.model.is_legacy)

    QuantModuleRegistry.register({LinearWrapper: "nemo_linear_wrapper"})(_QuantNeMoLinearWrapper)

    if cfg.run_quantization:
        # Start quantization with ModelOpt

        cali_prompts = load_calib_prompts(
            cfg.quantize.batch_size,
            "/opt/NeMo/nemo/collections/multimodal/modules/stable_diffusion/quantization_utils/calib_prompts.txt",
        )
        extra_step = 0  # Following sdxl example

        if cfg.quantize.format == "int8":
            quant_config = get_int8_config(
                base.model.model.diffusion_model,
                cfg.quantize.quant_level,
                cfg.quantize.alpha,
                cfg.quantize.percentile,
                cfg.quantize.n_steps + extra_step,
                global_min=False,
            )
        else:
            raise NotImplementedError

        calib_size = cfg.quantize.calib_size // cfg.quantize.batch_size

        def forward_loop():
            do_calibrate(
                base=base,
                calibration_prompts=cali_prompts,
                calib_size=calib_size,
                n_steps=cfg.quantize.n_steps,
                param_dict=cfg.sampling.base,
                num_samples=cfg.infer.num_samples,
            )

        mtq.quantize(base.model.model.diffusion_model, quant_config, forward_loop)
        mto.save(base.model.model.diffusion_model, cfg.quantize.quantized_ckpt)

    if cfg.run_onnx_export:
        os.makedirs(cfg.onnx_export.onnx_dir, exist_ok=True)
        output = Path(f"{cfg.onnx_export.onnx_dir}/unet.onnx")
        # Export quantized model to ONNX
        if not cfg.run_quantization:
            mto.restore(base.model.model.diffusion_model, cfg.onnx_export.quantized_ckpt)
        quantize_lvl(base.model.model.diffusion_model, cfg.quantize.quant_level)

        # QDQ needs to be in FP32
        base.model.model.diffusion_model.to(torch.float32).to("cpu")
        base.device = 'cpu'

        input_names = [
            "x",
            "timesteps",
            "context",
            "y",
        ]
        output_names = ["out"]
        sd_version = "nemo"

        dynamic_axes = AXES_NAME[sd_version]
        latent_dim = int(cfg.sampling.base.width // 8)
        adm_in_channels = base.model.model.diffusion_model.adm_in_channels
        dummy_inputs = generate_dummy_inputs(
            sd_version, base.device, latent_dim=latent_dim, adm_in_channels=adm_in_channels
        )
        do_constant_folding = True
        opset_version = 17

        onnx_export(
            base.model.model.diffusion_model,
            (dummy_inputs,),
            f=output.as_posix(),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
        )

    if cfg.run_trt_export:
        torch.cuda.empty_cache()
        batch_size = cfg.infer.get('num_samples', 1)
        min_batch_size = cfg.trt_export.min_batch_size
        max_batch_size = cfg.trt_export.max_batch_size
        static_batch = cfg.trt_export.static_batch
        fp16 = cfg.trainer.precision in ['16', '16-mixed', 16]
        build_engine(
            f"{cfg.onnx_export.onnx_dir}/unet.onnx",
            f"{cfg.trt_export.trt_engine}",
            fp16=fp16,
            input_profile=get_input_profile_unet(
                batch_size,
                static_batch=static_batch,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                latent_dim=cfg.sampling.base.height // 8,
                adm_in_channels=base.model.model.diffusion_model.adm_in_channels,
            ),
            timing_cache=None,
            int8=cfg.trt_export.int8,
            builder_optimization_level=cfg.trt_export.builder_optimization_level,
        )


if __name__ == "__main__":
    main()
