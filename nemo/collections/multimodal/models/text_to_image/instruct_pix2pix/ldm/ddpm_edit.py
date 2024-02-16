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
"""
https://github.com/timothybrooks/instruct-pix2pix/blob/2afcb7e45bd350765f21a58a0c135871e9dc5a78/stable_diffusion/ldm/models/diffusion/ddpm_edit.py
"""

import torch
from einops import rearrange

from nemo.collections.multimodal.data.instruct_pix2pix.edit_dataset import EditDataset
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    LatentDiffusion,
    MegatronLatentDiffusion,
)
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class LatentDiffusionEdit(LatentDiffusion):
    def init_from_ckpt(
        self, path, ignore_keys=list(), only_model=False, load_vae=True, load_unet=True, load_encoder=True,
    ):
        pl_sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(pl_sd.keys()):
            pl_sd = pl_sd["state_dict"]
        sd = {}

        first_key = list(pl_sd.keys())[0]
        # State keys of model trained with TorchDynamo changed from
        # "model.xxx" to "model._orig_mod.xxx"
        for k, v in pl_sd.items():
            new_k = k.replace("._orig_mod", "")
            # compatibility for stable diffusion old checkpoint
            # remove megatron wrapper prefix
            if first_key == "model.betas":
                new_k = new_k.lstrip("model.")
            sd[new_k] = v
        keys = list(sd.keys())

        # Our model adds additional channels to the first layer to condition on an input image.
        # For the first layer, copy existing channel weights and initialize new channel weights to zero.
        input_keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
        ]

        self_sd = self.state_dict()
        for input_key in input_keys:
            if input_key not in sd or input_key not in self_sd:
                continue

            input_weight = self_sd[input_key]
            if input_weight.size() != sd[input_key].size():
                print(f"Manual init: {input_key}")
                input_weight.zero_()
                input_weight[:, :4, :, :].copy_(sd[input_key])
                ignore_keys.append(input_key)

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        )
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
        uncond=0.05,
    ):
        x = batch[k]
        if bs is not None:
            x = x[:bs]

        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        cond_key = cond_key or self.cond_stage_key
        xc = batch[cond_key]
        if bs is not None:
            xc["c_crossattn"] = xc["c_crossattn"][:bs]
            xc["c_concat"] = xc["c_concat"][:bs]
        cond = {}

        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        random = torch.rand(x.size(0), device=x.device)
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        cond["c_crossattn"] = torch.where(
            prompt_mask, null_prompt, self.get_learned_conditioning(xc["c_crossattn"]).detach()
        )
        cond["c_concat"] = input_mask * self.encode_first_stage((xc["c_concat"].to(x.device))).mode().detach()

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out


class MegatronLatentDiffusionEdit(MegatronLatentDiffusion):
    def model_provider_func(self, pre_process=True, post_process=True):
        """Model depends on pipeline paralellism."""
        model = LatentDiffusionEdit(cfg=self.cfg, model_parallel_config=self.model_parallel_config)
        return model

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
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

        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def build_train_valid_test_datasets(self):
        # TODO (yuya): set up splits ratio and other params
        if self.cfg.data.data_path is not None:
            self._train_ds = EditDataset(path=self.cfg.data.data_path, split="train", flip_prob=0.5)
            self._validation_ds = EditDataset(path=self.cfg.data.data_path, split="val")
            self._test_ds = EditDataset(path=self.cfg.data.data_path, split="test")

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            drop_last = True
            if not self.cfg.get('validation_drop_last', True):
                logging.info(f'Drop last in validation dataset is set to False')
                drop_last = False
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples, drop_last)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            drop_last = True
            if not self.cfg.get('validation_drop_last', True):
                logging.info(f'Drop last in validation dataset is set to False')
                drop_last = False
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples, drop_last)

    def build_pretraining_data_loader(self, dataset, consumed_samples, drop_last=True):
        """Build dataloader given an input dataset."""

        if dataset is None:
            return None
        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self._cfg.data, 'dataloader_type') and self._cfg.data.dataloader_type is not None:
            # TODO (yuya): fix this
            if self._cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self._cfg.micro_batch_size,
                    global_batch_size=self._cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                )
            elif self._cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self._cfg.micro_batch_size,
                    global_batch_size=self._cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                )
            else:
                raise Exception(f'{self._cfg.dataloader_type} dataloader type is not supported.')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self._cfg.data.num_workers, pin_memory=True,
        )
