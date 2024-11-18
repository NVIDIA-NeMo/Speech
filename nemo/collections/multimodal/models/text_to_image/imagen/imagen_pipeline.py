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
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch
from lightning.pytorch import Trainer
from omegaconf.omegaconf import OmegaConf
from torch.cuda.amp import autocast

from nemo.collections.multimodal.models.text_to_image.imagen.imagen import Imagen, MegatronImagen
from nemo.collections.multimodal.parts.utils import numpy_to_pil, setup_trainer_and_models_for_inference


@dataclass
class ImagenCustomizedModelConfig:
    base_ckpt: Optional[str] = None
    base_cfg: Optional[str] = None
    sr256_ckpt: Optional[str] = None
    sr256_cfg: Optional[str] = None
    sr1024_ckpt: Optional[str] = None
    sr1024_cfg: Optional[str] = None


@dataclass
class ImagenSamplingConfig:
    step: Optional[int] = None
    cfg: Optional[float] = 1


@dataclass
class ImagenPipelineConfig:
    model_name: Optional[str] = None
    run_ema_model: Optional[bool] = True
    customized_model: Optional[ImagenCustomizedModelConfig] = None
    num_images_per_promt: Optional[int] = 8
    texts: Optional[List[str]] = field(default_factory=lambda: [])
    output_path: Optional[str] = 'output/imagen_inference'
    record_time: Optional[bool] = False
    encoder_path: Optional[str] = None
    target_resolution: Optional[int] = 256
    inference_precision: Optional[str] = '32'
    thresholding_method: Optional[str] = 'dynamic'
    samplings: Optional[List[ImagenSamplingConfig]] = field(default_factory=lambda: list())
    part: Optional[int] = 0


class ImagenPipeline(Callable):
    def __init__(self, models: List[Imagen], text_encoder, cfg, device):
        self.models = [model.to(device) for model in models]
        self.text_encoder = text_encoder.to(device)
        self.cfg = cfg
        self.device = device

    def _load_model(model_ckpt: str, model_cfg: str, eval_mode: bool = True, trainer: Trainer = None):
        assert model_ckpt is not None, 'model ckpt cannot be None'
        if model_ckpt.endswith('.nemo'):
            model_cfg = MegatronImagen.restore_from(restore_path=model_ckpt, trainer=trainer, return_config=True)
            model_cfg.unet.flash_attention = False
            model_cfg.micro_batch_size = 1
            model_cfg.global_batch_size = 1
            model = MegatronImagen.restore_from(
                restore_path=model_ckpt,
                override_config_path=model_cfg,
                trainer=trainer,
            )
        elif model_ckpt.endswith('.ckpt'):
            model_cfg = OmegaConf.load(model_cfg)
            model_cfg.model.unet.flash_attention = False
            model_cfg.model.micro_batch_size = 1
            model_cfg.model.global_batch_size = 1
            model = MegatronImagen(cfg=model_cfg.model, trainer=trainer)
            checkpoint = torch.load(model_ckpt, map_location=lambda storage, loc: storage)

            # Change weight keys if training using TorchInductor
            state_dict = checkpoint['state_dict']
            del_keys = []
            for k, v in state_dict.items():
                if '._orig_mod' in k:
                    del_keys.append(k)
            if len(del_keys) != 0:
                print('ckpt was saved with TorchInductor. Renaming weights..')
            for k in del_keys:
                new_k = k.replace("._orig_mod", "")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]
            model.load_state_dict(state_dict, strict=True)
        else:
            raise Exception('Invalid ckpt type. Should be either .nemo or .ckpt with cfg')

        model = model.model  # We do not need Megatron Instance for inference
        model.model.set_inference_mode(True)  # Used for adding the least noise for EDM inference for SR model.
        if eval_mode:
            model.unet.cuda().eval()
        return model

    @staticmethod
    def _load_customized_model(cfg: ImagenPipelineConfig, trainer=None, megatron_loading=False, megatron_cfg=None):
        if megatron_loading:
            assert megatron_cfg

            def model_cfg_modifier(model_cfg):
                model_cfg.inductor = False
                model_cfg.unet.flash_attention = False
                model_cfg.micro_batch_size = megatron_cfg.fid.ncaptions_per_batch
                model_cfg.global_batch_size = model_cfg.micro_batch_size * megatron_cfg.fid.ntasks_per_node

            trainer, megatron_models = setup_trainer_and_models_for_inference(
                MegatronImagen, cfg=megatron_cfg, model_cfg_modifier=model_cfg_modifier
            )
            models = [mm.model for mm in megatron_models]
            for model in models:
                model.cuda().eval()
                model.model.set_inference_mode(True)
            return models
        customized_models = cfg.customized_model
        models = []
        print('Load base model.')
        model = ImagenPipeline._load_model(
            model_ckpt=customized_models.base_ckpt,
            model_cfg=customized_models.base_cfg,
            trainer=trainer,
        )
        models.append(model)

        if cfg.target_resolution >= 256:
            print('Load SR256 model.')
            model = ImagenPipeline._load_model(
                model_ckpt=customized_models.sr256_ckpt, model_cfg=customized_models.sr256_cfg, trainer=trainer
            )
            models.append(model)

        if cfg.target_resolution >= 1024:
            print('Load SR1024 model.')
            model = ImagenPipeline._load_model(
                model_ckpt=customized_models.sr1024_ckpt, model_cfg=customized_models.sr1024_cfg, trainer=trainer
            )
            models.append(model)
        return models

    @classmethod
    def from_pretrained(
        cls, cfg: ImagenPipelineConfig, trainer=None, device='cuda', megatron_loading=False, megatron_cfg=None
    ):
        target_resolution = cfg.target_resolution
        assert target_resolution in [64, 256, 1024]

        # Set encoder_path which will be used when inst the model
        if cfg.encoder_path is not None:
            os.environ['ENCODER_PATH'] = cfg.encoder_path

        assert cfg.model_name is None, 'No predefined model for now'
        assert cfg.customized_model is not None, 'Need to provide customized models for inference'
        models = ImagenPipeline._load_customized_model(cfg, trainer, megatron_loading, megatron_cfg)
        assert len(models) >= 1, 'Need to load at least one model'
        if cfg.inference_precision == '16':
            print('Running Inference in FP16.')
            print('Converting all difussion models to FP16..')
            for model in models:
                model.half()

        print('Loading text encoder')
        text_encoder = models[0].get_text_encoder(encoder_path=cfg.encoder_path)
        if cfg.inference_precision == '16':
            print('Converting text encoders to FP16..')
            text_encoder.half()
        return ImagenPipeline(models=models, text_encoder=text_encoder, cfg=cfg, device=device)

    @torch.no_grad()
    def get_text_encodings(self, input_text, repeat=1):
        # Repeat the inputs so that we generate multiple samples per query
        if isinstance(input_text, str):
            inp_text_batch = [input_text]
        else:
            inp_text_batch = input_text
        # Encode the text embeddings using text encoder.
        text_encodings, text_mask = self.text_encoder.encode(inp_text_batch, device=self.device)
        if repeat != 1:
            assert len(inp_text_batch) == 1, 'Repeat should only be applied if we feed single text to encoder.'
            text_encodings = text_encodings.repeat(repeat, 1, 1)
            text_mask = text_mask.repeat(repeat, 1)
        return text_encodings, text_mask

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]] = None,
        inference_steps: Union[int, List[int]] = None,
        classifier_free_guidance: Union[float, List[float]] = None,
        num_images_per_promt: Optional[int] = 0,
        thresholding_method: bool = None,
        output_type: Optional[str] = 'pil',
        seed: Union[int, List[int]] = 2000,
        single_batch_mode: bool = False,
        output_res: Optional[int] = None,
        low_res_input: Optional[torch.Tensor] = None,
    ):
        if prompts is None:
            prompts = OmegaConf.to_object(self.cfg.texts)
        if num_images_per_promt == 0:
            num_images_per_promt = self.cfg.num_images_per_promt
        if thresholding_method is None:
            thresholding_method = self.cfg.thresholding_method
        device = self.device
        inference_precision = self.cfg.inference_precision
        assert inference_precision in ['16', '32', 'AMP'], "Inference Precision should be one of ['16', '32', 'AMP']"
        print(f'Running inference in {inference_precision} mode.')
        amp_enabled = inference_precision == 'AMP'

        # Based on output_res and low_res_input, determine which models to run
        if output_res is not None or low_res_input is not None:
            models = []
            if output_res is not None:
                for model in self.models:
                    models.append(model)
                    if model.image_size == output_res:
                        break
            else:
                models = self.models
            if low_res_input is not None:
                print(f'Low-res input shape: {low_res_input.shape}')
                low_res_dim = low_res_input.shape[-1]
                num_images_per_promt = low_res_input.shape[0]
                for idx, model in enumerate(models):
                    if model.image_size == low_res_dim:
                        models = models[idx + 1 :]
                        break
            print(f'Running inference on {len(models)} models.')
        else:
            models = self.models

        if classifier_free_guidance is None:
            cfgs = [each.cfg for each in self.cfg.samplings]
            cfgs = cfgs[: len(models)]
        else:
            cfgs = classifier_free_guidance
            if isinstance(cfgs, int) or isinstance(cfgs, float):
                cfgs = [cfgs] * len(models)

        if inference_steps is None:
            steps = [each.step for each in self.cfg.samplings]
            steps = steps[: len(models)]
        else:
            steps = inference_steps
            if isinstance(steps, int):
                steps = [steps] * len(models)

        assert len(steps) == len(cfgs) == len(models)

        output = []
        all_res_output = [[] for _ in range(len(models))]
        if single_batch_mode:
            num_images_per_promt = len(prompts)

        throughputs = {'text-encoding': []}
        for idx in range(len(models)):
            throughputs[f'stage-{idx+1}'] = []
        for prompt in prompts:
            if single_batch_mode:
                text_input = prompts
            else:
                text_input = prompt.strip('\n')
            print('Input caption: {}'.format(text_input))
            tic = time.perf_counter()
            text_encodings, text_mask = self.get_text_encodings(
                text_input, repeat=num_images_per_promt if not single_batch_mode else 1
            )
            throughputs['text-encoding'].append(time.perf_counter() - tic)

            # Set seed
            noise_maps = []
            if isinstance(seed, int):
                # Single seed for the batch
                torch.random.manual_seed(seed)
                # Generate noise maps
                for model in models:
                    noise_map = torch.randn(
                        (num_images_per_promt, 3, model.unet.image_size, model.unet.image_size), device=device
                    )
                    noise_map = noise_map.half() if inference_precision == '16' else noise_map
                    noise_maps.append(noise_map)
            elif isinstance(seed, list):
                assert len(seed) == num_images_per_promt
                for model in models:
                    noise_map_batch = []
                    for single_seed in seed:
                        torch.random.manual_seed(single_seed)
                        noise_map_single = torch.randn(
                            (1, 3, model.unet.image_size, model.unet.image_size), device=device
                        )
                        noise_map_batch.append(noise_map_single)
                    noise_map_batch = torch.cat(noise_map_batch, dim=0)
                    noise_map_batch = noise_map_batch.half() if inference_precision == '16' else noise_map_batch
                    noise_maps.append(noise_map_batch)
            else:
                raise RuntimeError('Seed type incorrect.')

            x_low_res = low_res_input
            all_res = []
            for idx, (model, noise_map, cfg, step) in enumerate(zip(models, noise_maps, cfgs, steps)):
                tic = time.perf_counter()
                with autocast(enabled=amp_enabled):
                    generated_images = model.sample_image(
                        noise_map=noise_map,
                        text_encoding=text_encodings,
                        text_mask=text_mask,
                        x_low_res=x_low_res,
                        cond_scale=cfg,
                        sampling_steps=step,
                        thresholding_method=thresholding_method,
                    )
                x_low_res = generated_images
                all_res.append(generated_images)
                throughputs[f'stage-{idx+1}'].append(time.perf_counter() - tic)
            # recenter from [-1, 1] to [0, 1]
            assert generated_images is not None
            generated_images = ((generated_images + 1) / 2).clamp_(0, 1)
            all_res = [((each + 1) / 2).clamp_(0, 1) for each in all_res]
            output.append(generated_images)
            for idx, each in enumerate(all_res):
                all_res_output[idx].append(each)
            if single_batch_mode:
                break

        if output_type == 'torch':
            return torch.cat(output, dim=0), [torch.cat(each, dim=0) for each in all_res_output]
        output_new = []
        for x_samples_image in output:
            # Convert to numpy
            x_samples_image = x_samples_image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == 'pil':
                x_samples_image = numpy_to_pil(x_samples_image)
            output_new.append(x_samples_image)

        all_res_output_new = [[] for each in range(len(models))]
        for idx, res_output in enumerate(all_res_output):
            for x_samples_image in res_output:
                # Convert to numpy
                x_samples_image = x_samples_image.cpu().permute(0, 2, 3, 1).numpy()
                if output_type == 'pil':
                    x_samples_image = numpy_to_pil(x_samples_image)
            all_res_output_new[idx].append(x_samples_image)

        for item in throughputs:
            throughputs[item] = sum(throughputs[item]) / len(throughputs[item])

        return output_new, all_res_output_new, throughputs
