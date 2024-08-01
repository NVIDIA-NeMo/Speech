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

from typing import Any

import einops
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch._inductor import config as inductor_config

from nemo.collections.multimodal.data.controlnet.controlnet_dataset import build_train_valid_datasets
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import LatentDiffusion
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers.ddim import DDIMSampler
from nemo.collections.multimodal.modules.stable_diffusion.attention import SpatialTransformer
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    UNetModel,
)
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
)
from nemo.collections.multimodal.parts.stable_diffusion.utils import exists, log_txt_as_img
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.utils import logging

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from torchvision.utils import make_grid

    TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCHVISION_AVAILABLE = False


class ControlledUnetModel(UNetModel):
    '''
    Modified Unet class that combines the output of controlling copy and frozen copy during forward pass.
    '''

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        '''
        :param x: latents of diffusion process
        :param timesteps: diffusion step
        :param context: text embedding guiding the denoising process
        :param control: output from controlling copy of each corresponding layer
        :param only_mid_control: whether to add the output of controlling copy from middle block only
        '''
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(emb.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlLDM(LatentDiffusion):
    def __init__(self, cfg, model_parallel_config):
        super().__init__(cfg=cfg, model_parallel_config=model_parallel_config)
        self.control_model = ControlLDM.from_config_dict(cfg.control_stage_config)
        self.control_key = cfg.control_key
        self.only_mid_control = cfg.only_mid_control
        self.control_scales = [1.0] * 13
        self.sd_locked = cfg.sd_locked
        self.channels_last = cfg.channels_last

        if cfg.get("inductor", False):
            # TorchInductor with CUDA graph can lead to OOM
            inductor_config.triton.cudagraphs = cfg.get("inductor_cudagraphs", False)
            torch._dynamo.config.dynamic_shapes = False
            torch._dynamo.config.automatic_dynamic_shapes = False
            self.control_model = torch.compile(self.control_model)

        if self.channels_last:
            self.control_model = self.control_model.to(memory_format=torch.channels_last)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(torch.cuda.current_device())
        if self.channels_last:
            control = control.permute(0, 3, 1, 2).to(non_blocking=True)
        else:
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=c, c_concat=control)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # cond_txt = torch.cat(cond['c_crossattn'], 1) ## Has removed this first dim in the get_input function, same for below hint input
        cond_txt = cond['c_crossattn']

        if cond['c_concat'] is None:
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control
            )
        else:
            control = self.control_model(x=x_noisy, hint=cond['c_concat'], timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control
            )
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        batch = next(batch)
        batch['images'] = batch['images'].to(torch.cuda.current_device())
        batch['hint'] = batch['hint'].to(torch.cuda.current_device())
        N = batch['images'].shape[0]
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][:N], c["c_crossattn"][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            assert TORCHVISION_AVAILABLE, "Torchvision imports failed but they are required."
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": c_cat, "c_crossattn": c},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": uc_cat, "c_crossattn": uc_cross}
            samples_cfg, _ = self.sample_log(
                cond={"c_concat": c_cat, "c_crossattn": c},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def parameters(self):
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        return params

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,  ###TODO MMY these are new
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        use_flash_attention=False,
        from_pretrained_unet=None,
        from_NeMo=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks)))
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                use_flash_attention=use_flash_attention,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                    use_flash_attention=use_flash_attention,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        if from_pretrained_unet is not None:
            self.load_from_unet(from_pretrained_unet=from_pretrained_unet, from_NeMo=from_NeMo)

    def load_from_unet(self, from_pretrained_unet, from_NeMo=True):
        if not from_NeMo:
            print('loading from other source of unet is experimental! Carefully check if keys are loaded correctly.')
        else:
            print("Loading unet blocks from sd")

            state_dict = torch.load(from_pretrained_unet, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            model_state_dict = self.state_dict()
            model_state_keys = model_state_dict.keys()

            re_state_dict = {}
            for key_, value_ in state_dict.items():
                # check if key is a raw parameter
                if key_ in model_state_keys:
                    re_state_dict[key_] = value_
                    continue
                # prune from model prefix
                if key_.startswith('model.model.diffusion_model'):
                    re_state_dict[key_.replace('model.model.diffusion_model.', '')] = value_
                if key_.startswith('model.diffusion_model'):
                    re_state_dict[key_.replace('model.diffusion_model.', '')] = value_
                if key_.startswith('model.model._orig_mod.diffusion_model'):
                    re_state_dict[key_.replace('model.model._orig_mod.diffusion_model.', '')] = value_
                if key_.startswith('model._orig_mod.diffusion_model'):
                    re_state_dict[key_.replace('model._orig_mod.diffusion_model.', '')] = value_

            expected_keys = list(model_state_dict.keys())
            loaded_keys = list(re_state_dict.keys())
            missing_keys = list(set(expected_keys) - set(loaded_keys))
            unexpected_keys = list(set(loaded_keys) - set(expected_keys))

            if (
                'input_blocks.1.0.in_layers.2.weight' in loaded_keys
                and 'input_blocks.1.0.in_layers.1.weight' in expected_keys
            ):
                # GroupNormOpt fuses activation function to one layer, thus the indexing of weights are shifted for following
                for key_ in missing_keys:
                    if key_.startswith('input_blocks') or key_.startswith('middle_block.'):
                        s = key_.split('.')
                        idx = int(s[-2])
                        new_key_ = ".".join(s[:-2] + [str(int(idx + 1))] + [s[-1]])
                        re_state_dict[key_] = re_state_dict[new_key_]

                loaded_keys = list(re_state_dict.keys())
                missing_keys = list(set(expected_keys) - set(loaded_keys))
                unexpected_keys = list(set(loaded_keys) - set(expected_keys))

            self.load_state_dict(re_state_dict, strict=False)

            if len(missing_keys) > 42:
                print(
                    'warning: only input hint blocks and zero conv layers are randomly initialized. This message indicates some unet blocks are not loaded correctly.'
                )
                print(f'There is {len(missing_keys)} total missing keys')
                print("Missing:", missing_keys)
                print("Unexpected:", unexpected_keys)
            else:
                print("sd blocks loaded successfully")

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class MegatronControlNet(MegatronBaseModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        super().__init__(cfg, trainer=trainer)

        self._validate_trainer()

        # megatron_amp_O2 is not yet supported in diffusion models
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        self.model = self.model_provider_func()

        self.conditioning_keys = []

        if self.trainer.precision in ['bf16', 'bf16-mixed']:
            self.autocast_dtype = torch.bfloat16
        elif self.trainer.precision in [32, '32', '32-true']:
            self.autocast_dtype = torch.float
        elif self.trainer.precision in [16, '16', '16-mixed']:
            self.autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in ["32-true", "16-mixed", "bf16-mixed"]')

    def get_module_list(self):
        if isinstance(self.model, list):
            return [model.module if isinstance(model, Float16Module) else model for model in self.model]
        elif isinstance(self.model, Float16Module):
            return [self.model.module]
        else:
            return [self.model]

    def model_provider_func(self, pre_process=True, post_process=True):
        """Model depends on pipeline paralellism."""
        model = ControlLDM(cfg=self.cfg, model_parallel_config=self.model_parallel_config)
        return model

    def forward(self, x, c, *args, **kwargs):
        output_tensor = self.model(x, c, *args, **kwargs)
        return output_tensor

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert self.cfg.scale_factor == 1.0, 'rather not use custom rescaling and std-rescaling simultaneously'
            batch[self.cfg.first_stage_key] = batch[self.cfg.first_stage_key].cuda(non_blocking=True)
            self.model.on_train_batch_start(batch, batch_idx)

    def fwd_bwd_step(self, dataloader_iter, forward_only):
        tensor_shape = None  # Placeholder

        # handle asynchronous grad reduction
        no_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(
                self._optimizer.no_sync,
                greedy_grad_copy=self.megatron_amp_O2,
            )

        # pipeline schedules will get these from self.model.config
        for module in self.get_module_list():
            module.config.no_sync_func = no_sync_func

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # losses_reduced_per_micro_batch is a list of dictionaries
        # [{"loss": 0.1}, {"loss": 0.2}, ...] which are from gradient accumulation steps
        # only the last stages of the pipeline return losses
        loss_dict = {}
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                for key in losses_reduced_per_micro_batch[0]:
                    loss_tensors_list = [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                    loss_tensor = torch.stack(loss_tensors_list)
                    loss_dict[key] = loss_tensor.mean()
                loss_mean = loss_dict["train/loss"]
            else:
                raise NotImplementedError("Losses of micro batches sizes must be uniform!")
        else:
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0, device=torch.cuda.current_device())

        return loss_mean, loss_dict

    def training_step(self, dataloader_iter):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        Batch should be a list of microbatches and those microbatches should on CPU.
        Microbatches are then moved to GPU during the pipeline.
        The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        loss_mean, loss_dict = self.fwd_bwd_step(dataloader_iter, False)

        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # gradients are reduced internally in distributed optimizer
            pass
        elif self.megatron_amp_O2:
            # # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            # if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
            #     # main grads are stored in the MainParamsOptimizer wrapper
            #     self._optimizer.allreduce_main_grads()
            self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.precision == [16, '16', '16-mixed']:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, rank_zero_only=True, batch_size=1)
        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step + 1, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        return loss_mean

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward in the fwd/bwd functions from apex.
        No need to call it here.
        """
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        pass

    def _append_sequence_parallel_module_grads(self, module, grads):
        """Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            if sequence_parallel_param and param.requires_grad:
                if self.megatron_amp_O2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def get_forward_output_and_loss_func(self):
        def process_batch(batch):
            """Prepares the global batch for apex fwd/bwd functions.
            Global batch is a list of micro batches.
            """
            # noise_map, condition
            batch[self.cfg.first_stage_key] = batch[self.cfg.first_stage_key].cuda(non_blocking=True)
            if isinstance(batch[self.cfg.cond_stage_key], torch.Tensor):
                # in the case of precached text embeddings, cond_stage is also a tensor
                batch[self.cfg.cond_stage_key] = batch[self.cfg.cond_stage_key].cuda(non_blocking=True)

            # SD has more dedicated structure for encoding, so we enable autocasting here as well
            with torch.cuda.amp.autocast(
                self.autocast_dtype in (torch.half, torch.bfloat16),
                dtype=self.autocast_dtype,
            ):
                x, c = self.model.get_input(batch, self.cfg.first_stage_key)

            if not isinstance(c, dict):
                return [x, c]

            if len(self.conditioning_keys) == 0:
                self.conditioning_keys = list(c.keys())
            c_list = [c[key] for key in self.conditioning_keys]
            return [x, *c_list]

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            batch = process_batch(batch)
            batch = [x.cuda(non_blocking=True) for x in batch]
            if len(self.conditioning_keys) == 0:
                x, c = batch
            else:
                x = batch[0]
                c = {}
                for idx, key in enumerate(self.conditioning_keys):
                    c[key] = batch[1 + idx]
            loss, loss_dict = model(x, c)

            def dummy(output_tensor):
                return loss, loss_dict

            # output_tensor, and a function to convert output_tensor to loss + loss_dict
            return loss, dummy

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            raise NotImplementedError

        return fwd_output_only_func

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        tensor_shape = None  # Placeholder
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            tensor_shape=None,  # required by pipeline parallelism
            dtype=self.autocast_dtype,
            sequence_parallel=self.cfg.get('sequence_parallel', False),
            enable_autocast=True,
        )
        # only the last stages of the pipeline return losses
        val_loss_dict = {}
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            for key in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced[key] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.stack(loss_tensors_list)
                val_loss_dict[key] = loss_tensor.mean()

        self.log_dict(val_loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def setup(self, stage=None):
        """PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        self.model.rng.manual_seed(self.cfg.seed + 100 * parallel_state.get_data_parallel_rank())

        # log number of parameters
        if isinstance(self.model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in self.model]
            )
        else:
            num_parameters_on_device = sum([p.nelement() for p in self.model.parameters()])

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda(non_blocking=True)

        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        # allowing restored models to optionally setup datasets
        self.build_train_valid_test_datasets()

        # Batch size need to be provided for webdatset
        self._num_micro_batches = get_num_microbatches()
        self._micro_batch_size = self.cfg.micro_batch_size

        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for Stable Diffusion...')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        if self.cfg.first_stage_key.endswith("encoded"):
            self._train_ds, self._validation_ds = build_train_valid_precached_datasets(
                model_cfg=self.cfg,
                consumed_samples=self.compute_consumed_samples(0),
            )
        else:
            self._train_ds, self._validation_ds = build_train_valid_datasets(
                model_cfg=self.cfg, consumed_samples=self.compute_consumed_samples(0)
            )
        self._test_ds = None

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building datasets for LatentDiffusion.')
        return self._train_ds, self._validation_ds, self._test_ds

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = torch.utils.data.DataLoader(
                self._train_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = torch.utils.data.DataLoader(
                self._validation_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = torch.utils.data.DataLoader(
                self._test_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        When using pipeline parallelism, we need the global batch to remain on the CPU,
        since the memory overhead will be too high when using a large number of microbatches.
        Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """Certain trainer configurations can break training.
        Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    @classmethod
    def list_available_models(cls):
        return None

    def log_images(self, *args, **kwargs):
        return self.model.log_images(*args, **kwargs)

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()
