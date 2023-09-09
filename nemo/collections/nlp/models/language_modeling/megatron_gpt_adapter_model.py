# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# This code has been adapted from the following private repo: https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/tree/prompt-learning/prefix_tuning_v2
# Adapted by: @adithyare


import itertools
import os

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging, model_utils


class MegatronGPTBaseAdapterModel(MegatronGPTPromptLearningModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.get('language_model_path')):
            save_restore_connector.model_extracted_dir = cfg.get('language_model_path')
        self.frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        self.adapter_name_keys = []

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        taskname_ids,
        labels=None,
        inference=True,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                encoder_input=None,
                attention_mask=attention_mask,
                labels=labels,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    encoder_input=None,
                    attention_mask=attention_mask,
                    labels=labels,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                )

        return output

    def setup(self, stage=None):
        if stage == 'predict':
            self.frozen_model.freeze()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        self.setup_training_data()
        self.setup_validation_data()
        logging.info(f'setup completed:\n{self.frozen_model.summarize()}')

    def on_train_end(self):
        # Save the best nemo model
        self.save_to(save_path=self.cfg.nemo_path)

    def get_forward_output_only_func(self):
        """
        Used for generate method only for now.
        """

        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            (
                tokens,
                attention_mask,
                position_ids,
                task_ids,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch

            tokens = tokens.cuda()
            attention_mask = attention_mask.cuda()
            position_ids = position_ids.cuda()
            task_ids = task_ids.cuda()
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            output_tensor = model(tokens, position_ids, attention_mask, task_ids, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        for name, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups. 
        Makes two optimizer param groups, one for the frozen model params
        and one for the prompt-table/prompt-encoder params. The learning 
        rate for the frozen model's params will always be zero effectively
        freezing the model's params but still allowing for the needed gradients
        to be passed around in pipeline parallel models. The prompt-encoder 
        and/or prompt table will use the learning rate set by the user. 
        """
        self.frozen_model.freeze()  # Freeze the entire model
        opt_params = []
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += [p for p in module.parameters()]

        self._optimizer_param_groups = [{'params': opt_params}]
        logging.info(f'Optimizer groups set:\n{self.frozen_model.summarize()}')

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) for x in batch]
            input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids = batch
            output_tensor = model(input_ids, position_ids, attention_mask, taskname_ids, labels, inference=False)

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def training_step(self, dataloader_iter, batch_idx):
        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()
        batch = next(dataloader_iter)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.torch_dtype == torch.float16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)

        # Need to make sure the frozen model param learning rate stays 0.0
        # so forceing lr to be 0.0 for gpt layers before param update
        return loss_mean


class MegatronGPTAdapterLearningModel(MegatronGPTBaseAdapterModel):
    """
    MegatronGPTAdapterLearningModel is a model that combines a base model (GPTModel) with a adapters.
    This class only supports the canonical Adapter training described in Houlsby et al. (https://arxiv.org/pdf/1902.00751.pdf)

    Two adapter's are inserted into each Transformer layer in the base GPT Model.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded 
    and infused into the same GPT Model for inference. 
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        assert cfg.adapter_tuning.get('adapter_dim', 0) > 0, "adapter_dim has not been set."
        assert (
            cfg.adapter_tuning.adapter_dim % cfg.tensor_model_parallel_size == 0
        ), "The adapter dim should be divisible by tensor_model_parallel_size."
        assert cfg.adapter_tuning.type in [
            'linear_adapter',
            'parallel_adapter',
        ], "Adapter type should be 'linear_adapter' or 'parallel_adapter'"

        self.adapter_name_keys = [AdapterName.PRE_ATTN_ADAPTER, AdapterName.POST_ATTN_ADAPTER]
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )

        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')

        if cfg.adapter_tuning.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=self.frozen_model_cfg.hidden_size,
                out_features=self.frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.adapter_dim,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                norm_type=cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=cfg.adapter_tuning.get('column_init_method', 'xavier'),
                row_init_method=cfg.adapter_tuning.get('row_init_method', 'zero'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )
        else:
            adapter_cfg = LinearAdapterConfig(
                in_features=self.frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.adapter_dim,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )

        self.frozen_model.freeze()
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=adapter_key, cfg=adapter_cfg,
                        )

        logging.info(f'After adding adapters:\n{self.frozen_model.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass


class MegatronGPTInfusedAdapterModel(MegatronGPTBaseAdapterModel):
    """
    MegatronGPTInfusedAdapterModel is a model that combines a base model (GPTModel) with a "Infused Adapter that can Inhibiting and Amplify Inner Activations", known as IA3.
    This class supports the addition of IA3 into a transformer based LM as described in Liu et al. (https://arxiv.org/pdf/2205.05638.pdf)

    Three adapter's are inserted into each Transformer layer in the base GPT Model. Each adapter is basically a vector that simply scales the key, value or ffn hidden representations.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded 
    and infused into the same GPT Model for inference. 
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.adapter_name_keys = [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED, AdapterName.MLP_INFUSED]
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )

        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')

        self.frozen_model.freeze()
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    if adapter_key == AdapterName.MLP_INFUSED:
                        cfg = MLPInfusedAdapterConfig(
                            in_features=self.frozen_model_cfg.ffn_hidden_size
                            // self.frozen_model_cfg.tensor_model_parallel_size
                        )
                    elif adapter_key in [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED]:
                        cfg = InfusedAdapterConfig(
                            in_features=self.frozen_model_cfg.hidden_size
                            // self.frozen_model_cfg.tensor_model_parallel_size
                        )
                    else:
                        raise ValueError(f"Adapter Key {adapter_key} is unknown.")
                    if model_utils.import_class_by_path(cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(name=adapter_key, cfg=cfg)

        logging.info(f'After adding adapters:\n{self.frozen_model.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass
