# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common import (
    BIGLSTMPromptEncoder,
    PromptEncoder,
    PromptEncoderMLP,
    PromptEncoderType,
    PromptTable,
    VirtualPromptPlaceholderToken,
    VirtualPromptSource,
    VirtualPromptStyle,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam, TextGeneration
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronGPTPromptLearningModel']


class MegatronGPTPromptLearningModel(MegatronBaseModel, TextGeneration):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron GPT model. 

    Prompt Tuning initalizes virtual prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained GPT model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in 
    initalization are specified by the user in the config file. The model can 
    be prompt-tuned for multiple tasks at once. virtual prompts are stored in a 
    prompt table and can be added or deleted without disrupting virtual prompts 
    for other tasks. 

    P-tuning initializes an LSTM encoder model that generates virtual prompt
    embeddings for every task. Each task shares the same encoder. After ptuning
    is compelete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a 
    new virtual prompt via p-tuning, they do not need to retrain on all previous 
    tasks. This gives p-tuning the same task flexiblity as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        self.cfg = cfg
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.get('language_model_path')):
            save_restore_connector.model_extracted_dir = cfg.get('language_model_path')
        frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        # Need to overwrite some params in frozen model's config before restoring
        with open_dict(frozen_model_cfg):
            frozen_model_cfg.megatron_amp_O2 = False
            frozen_model_cfg.optim.name = "fused_adam"
            frozen_model_cfg.micro_batch_size = self.cfg.micro_batch_size
            frozen_model_cfg.global_batch_size = self.cfg.global_batch_size
            frozen_model_cfg.precision = trainer.precision
            frozen_model_cfg.sequence_parallel = self.cfg.get("sequence_parallel", False)
            frozen_model_cfg.activations_checkpoint_granularity = self.cfg.get(
                "activations_checkpoint_granularity", None
            )
            frozen_model_cfg.activations_checkpoint_num_layers = self.cfg.get(
                "activations_checkpoint_num_layers", None
            )
            frozen_model_cfg.activations_checkpoint_method = self.cfg.get("activations_checkpoint_method", None)

        if self.trainer.precision == 32:
            self.autocast_dtype = torch.float
        elif self.trainer.precision == 16:
            self.autocast_dtype = torch.half
        elif self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        if cfg.get('language_model_path', None):
            self.frozen_model = MegatronGPTModel.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                override_config_path=frozen_model_cfg,
            ).to(dtype=self.autocast_dtype)

        # TODO: Enable amp_o2 training
        self.megatron_amp_o2 = False
        self.pipeline_parallel = self.cfg.get('pipeline_model_parallel_size', 1) > 1
        self.tokenizer = self.frozen_model.tokenizer
        self.hidden_size = self.frozen_model.cfg.hidden_size
        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))
        self.virtual_prompt_style = VirtualPromptStyle(cfg.virtual_prompt_style)

        if self.pipeline_parallel:
            assert (
                self.cfg.optim.sched.get("min_lr", 0.0) == 0.0
            ), "Minimum lr must be 0.0 when pipeline parallel size is > 1"

        # Load templates for assigning virtual prompt token positions
        self.load_task_templates(self.cfg.task_templates)

        if self.frozen_model.model.pre_process and self.virtual_prompt_style in [
            VirtualPromptStyle.P_TUNING,
            VirtualPromptStyle.PROMPT_TUNING,
            VirtualPromptStyle.INFERENCE,
        ]:

            self.word_embeddings = self.frozen_model.model.language_model.embedding.word_embeddings

            # Prompt table stores all task embeddings, p-tuning virtual prompts get added to the table after training
            self.prompt_table = PromptTable(
                existing_tasks=self.existing_tasks,
                task_templates=self.task_templates,
                task_id_num_to_name=self.task_id_num_to_name,
                hidden_size=self.hidden_size,
            )

        self.padded_vocab_size = self.frozen_model.padded_vocab_size
        self._prompt_table_key = VirtualPromptSource.PROMPT_TABLE.value
        self._prompt_encoder_key = VirtualPromptSource.PROMPT_ENCODER.value

        # Prepare pseudo token ids for virtual/virtual prompt tokens
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0] if self.pseudo_token_ids else None
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        # Prompt tuning stores virtual prompts in the prompt table and tunes their weight directly
        if self.virtual_prompt_style in [VirtualPromptStyle.PROMPT_TUNING, VirtualPromptStyle.INFERENCE]:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_TABLE

        # P-Tuning uses an LSTM Encoder to produce virtual token embeddings
        elif self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_ENCODER
        elif self.virtual_prompt_style == VirtualPromptStyle.NO_PROMPT:
            self.virtual_prompt_source = VirtualPromptSource.NO_PROMPT
        else:
            raise ValueError(
                f"\nvirtual prompt style '{cfg.virtual_prompt_style}' not recognized, please use one of 'prompt-tuning' or 'p-tuning'"
            )

        self._reduced_loss_buffer = []
        self._inference_config = None

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = True
        self.lowest_val_loss = None

    def load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns  
        it into a table where each task's prompt template and 
        the number of virtual tokens to insert in a given part of 
        the prompt template are specified. 
        """
        self.task_templates = {}
        self.task_id_num_to_name = {}
        self.max_virtual_tokens = 0

        task_id_num = 0
        for task in task_templates:
            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
                "answer_only_loss": task.get("answer_only_loss", False),
                "answer_field": task.get("answer_field", None),
                "truncate_field": task.truncate_field,
                "total_virtual_tokens": task.total_virtual_tokens,
                "virtual_token_splits": task.virtual_token_splits,
                "task_id_num": task_id_num,
            }

            self.max_virtual_tokens = max(self.max_virtual_tokens, task.total_virtual_tokens)
            self.task_id_num_to_name[task_id_num] = task.taskname
            task_id_num += 1

        # Check that all new tasks have the same total num virtual tokens
        # Num virtual tokens for new tasks don't need to match num used for previously tuned tasks
        if self.new_tasks:
            new_task_name = self.new_tasks[0]
            self.total_new_task_virtual_tokens = self.task_templates[new_task_name]["total_virtual_tokens"]

            assert all(
                self.task_templates[taskname]["total_virtual_tokens"] == self.total_new_task_virtual_tokens
                for taskname in self.new_tasks
            ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."

    def init_new_prompts(self):
        """
        Initialize new virtual prompts to be tuned using prompt tuning 
        """
        for idx, taskname in enumerate(self.new_tasks):
            init_method = self.cfg.prompt_tuning.new_prompt_init_methods[idx].lower()
            total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]

            if init_method == "text":
                init_text = self.cfg.prompt_tuning.new_prompt_init_text[idx]
                init_text_ids = self.tokenizer.text_to_ids(init_text)
                self.prompt_table.init_prompt_from_text(
                    taskname, init_text_ids, self.word_embeddings, total_virtual_tokens
                )

            elif init_method == 'random':
                self.prompt_table.init_prompt_from_random(taskname, total_virtual_tokens)

            else:
                raise AttributeError(
                    f'\nvirtual prompt init method {init_method} is not recognized\
                                        please use one of text or random'
                )

    def init_prompt_encoder(self):
        """
        Init the prompt encoder needed for p-tuning on a new task
        """
        # Total virtual tokens should be the same across all new tasks, so just need one
        new_task = self.new_tasks[0]
        total_virtual_tokens = self.task_templates[new_task]["total_virtual_tokens"]

        encoder_type = PromptEncoderType(self.cfg.p_tuning.get("encoder_type", "tpmlp").lower())
        if encoder_type == PromptEncoderType.TPMLP:
            self.prompt_encoder = PromptEncoderMLP(
                total_virtual_tokens=total_virtual_tokens,
                hidden_size=self.cfg.p_tuning.get("encoder_hidden", 2048),
                output_size=self.hidden_size,
                init_std=self.cfg.p_tuning.get("init_std", 0.023),
            )
        elif encoder_type == PromptEncoderType.BIGLSTM:
            self.prompt_encoder = BIGLSTMPromptEncoder(
                total_virtual_tokens=total_virtual_tokens,
                hidden_size=self.cfg.p_tuning.encoder_hidden,
                output_size=self.hidden_size,
                lstm_dropout=self.cfg.p_tuning.dropout,
                num_layers=self.cfg.p_tuning.num_layers,
            )
        elif encoder_type == PromptEncoderType.LSTM or encoder_type == PromptEncoderType.MLP:
            self.prompt_encoder = PromptEncoder(
                encoder_type=encoder_type,
                total_virtual_tokens=total_virtual_tokens,
                token_dim=self.hidden_size,
                hidden_size=self.cfg.p_tuning.get("encoder_hidden", self.hidden_size // 2),
                lstm_dropout=self.cfg.p_tuning.get("dropout", 0.0),
                num_layers=self.cfg.p_tuning.get("num_layers", 2),
            )
        else:
            raise ValueError('not supported')
        # cast the model weights to bf16 only for 'bf16' precision
        # For fp16, cannot cast the model weights as AMP will complain
        if self.trainer.precision == 'bf16':
            self.prompt_encoder = self.prompt_encoder.to(dtype=self.autocast_dtype)

    def add_ptuned_prompts_to_prompt_table(self):
        """
        Adds all newly p-tuned virtual prompts to the prompt table 
        for inference. p-tuned virtual prompts WILL NOT be further
        tuned once added to the prompt table. After p-tuned prompts
        are added to the prompt table, the prompt encoder weights
        are removed from the model to avoid needing to load unnecessary
        weights during inference or future p-tuning/prompt-tuning.
        """
        # Save p-tuned prompts to prompt table
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING and self.frozen_model.model.pre_process:
            for taskname in self.new_tasks:
                device = next(self.word_embeddings.parameters()).device
                tokenized_taskname = torch.tensor(self.tokenizer.text_to_ids(taskname)).to(device)
                taskname_embeddings = self.word_embeddings(tokenized_taskname).unsqueeze(0)
                virtual_prompt_embeddings = self.prompt_encoder(taskname_embeddings=taskname_embeddings).squeeze(0)
                total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
                self.prompt_table.add_prompt_from_p_tuning_encoder(
                    taskname, virtual_prompt_embeddings, total_virtual_tokens
                )

    def freeze_existing_virtual_prompt_params(self):
        """Freeze params of existing virtual prompts that should not be tuned further
        """
        # Only want new prompt tags to be tunable, leave existing prompt tags alone
        for taskname in self.prompt_table.prompt_table.keys():
            if taskname in set(self.new_tasks):
                for params in self.prompt_table.prompt_table[taskname].parameters():
                    params.requires_grad = True
            else:
                for params in self.prompt_table.prompt_table[taskname].parameters():
                    params.requires_grad = False

        # Make sure word embeddings are frozen
        for params in self.word_embeddings.parameters():
            params.requires_grad = False

    def get_model_tasks(self):
        """
        For user to inspect which tasks the model has been
        p-tuned/prompt-tuned to preform.
        """
        tasks = {}
        for taskname in self.prompt_table.prompt_table.keys():
            tasks[taskname] = self.task_templates[taskname].copy()

        return tasks

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Custom state dict that only contains prompt table and prompt encoder parameters. 
        No frozen model parameters are stored in the state dict. Prompt encoder parameters 
        are only in state dict for intermediate checkpoints saved during training. Final
        nemo checkpoints at the end of training will contain prompt table parameters only. 
        """
        state_dict_ = {}
        if self.frozen_model.model.pre_process:
            state_dict_[self._prompt_table_key] = self.prompt_table.state_dict()

            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                state_dict_[self._prompt_encoder_key] = self.prompt_encoder.state_dict()

        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method. 
        """
        if self.frozen_model.model.pre_process:
            if self._prompt_table_key in state_dict:
                state_dict_ = state_dict[self._prompt_table_key]
            else:
                # Handle loading state dict before weight saving change for backward compatibility.
                state_dict_ = OrderedDict()
                for key in state_dict.keys():
                    if key.startswith(self._prompt_table_key):
                        key_substring = ".".join(key.split(".")[1:])
                        state_dict_[key_substring] = state_dict[key]

            self.prompt_table.load_state_dict(state_dict_, strict)

            if (
                self._prompt_encoder_key in state_dict
                and self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER
            ):
                state_dict_ = state_dict[self._prompt_encoder_key]
                if not hasattr(self, "prompt_encoder"):
                    self.init_prompt_encoder()
                self.prompt_encoder.load_state_dict(state_dict_, strict)

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
        # Freeze frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        virtual_prompt_params = {'params': []}

        if self.frozen_model.model.pre_process:
            virtual_prompt_params['params'].extend([param for param in self.prompt_table.parameters()])

            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                virtual_prompt_params['params'].extend([param for param in self.prompt_encoder.parameters()])
        self._optimizer_param_groups = (virtual_prompt_params,)

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
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        GPT style models. Bypasses the vocab token preprocessing done
        in the MegatronGPT class.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        if self.frozen_model.model.pre_process:
            if inference:
                input_embeds = self.embed_input_inference(input_ids, taskname_ids)
            else:
                input_embeds = self.embed_input_train(input_ids, taskname_ids)
            position_embeddings = self.frozen_model.model.language_model.embedding.position_embeddings(position_ids)
            encoder_input = input_embeds + position_embeddings
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            if self.cfg.get("sequence_parallel", False):
                encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        else:
            encoder_input = None

        # Call forward on GPT model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.model(
                input_ids=None,
                position_ids=None,
                encoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.model(
                    input_ids=None,
                    position_ids=None,
                    encoder_input=encoder_input,
                    attention_mask=attention_mask,
                    labels=labels,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                )

        return output

    def embed_input_train(self, input_ids: Tensor, taskname_ids: Tensor):
        """
        Replaces the virtual tokens in the input_ids with embeddings 
        calculated from either the 'prompt_table' or 'prompt_encoder'. 
        The virtual token placeholders have token_ids listed in
        `self.pseudo_token_ids`.

        params:
            input_ids: the input token ids
            taskname_ids: the NLP task tag token ids
        returns:
            the token embedding for the LM model.
        """
        # Replace virtual token ids with padding for forward pass through vocab embeddings
        discrete_token_ids = input_ids.clone()
        discrete_token_ids[(input_ids >= self.pseudo_token_ids_start)] = self.pad_token_id
        discrete_token_embeds = self.word_embeddings(discrete_token_ids).clone()

        # Find the indicies where virtual tokens should be inserted
        virtual_token_locations = input_ids >= self.pseudo_token_ids_start

        # If there are no virtual tokens, just return discrete token embeds
        if not virtual_token_locations.any():
            return discrete_token_embeds

        # Get virtual token embeddings from the prompt table or prompt encoder
        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_TABLE:
            virtual_token_embeds = [self.prompt_table(task_id_num) for task_id_num in taskname_ids]
            virtual_token_embeds = torch.stack(virtual_token_embeds)

        elif self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            taskname_embeddings = self.word_embeddings(taskname_ids)
            virtual_token_embeds = self.prompt_encoder(taskname_embeddings=taskname_embeddings)

        # Create index template specifying where virtual token embeddings should be placed
        batch_size, _, embedding_size = discrete_token_embeds.shape
        virtual_token_index = virtual_token_locations.nonzero().reshape((batch_size, -1, 2))[:, :, 1][:, :, None]
        virtual_token_index = virtual_token_index.expand(
            batch_size, self.total_new_task_virtual_tokens, embedding_size
        )

        # Make sure discrete_token_embeds and virtual_token_embeds share the same dtype
        discrete_token_embeds = discrete_token_embeds.type(virtual_token_embeds.dtype)

        # Insert virtual token embeddings where they belong amoung the discrete token embeddings
        discrete_token_embeds.scatter_(1, virtual_token_index, virtual_token_embeds)
        input_embeds = discrete_token_embeds

        return input_embeds

    def embed_input_inference(self, input_ids: Tensor, taskname_ids: Tensor):
        """
        Replaces the virtual tokens in the input_ids with embeddings the 
        'prompt_table' only. The virtual token placeholders each have their
        own unique token_id within `self.pseudo_token_ids` to facilitate 
        placing the virtual tokens in their correct locations at each 
        decoding time step. 

        params:
            input_ids: the input token ids
            taskname_ids: the NLP task tag token ids
        returns:
            the token embedding for the LM model.
        """
        batch_size, seq_length = input_ids.shape

        # Replace virtual token ids with padding for forward pass through vocab embeddings
        discrete_token_ids = input_ids.clone()
        discrete_token_ids[(input_ids >= self.pseudo_token_ids_start)] = self.pad_token_id
        discrete_token_embeds = self.frozen_model.model.language_model.embedding.word_embeddings(
            discrete_token_ids
        ).clone()

        # Find the indicies where virtual tokens should be inserted
        virtual_token_locations = input_ids >= self.pseudo_token_ids_start
        virtual_token_locations = virtual_token_locations.unsqueeze(-1)
        virtual_token_locations = virtual_token_locations.expand(batch_size, seq_length, self.hidden_size)

        # If there are no virtual tokens, just return discrete token embeds
        if not virtual_token_locations.any():
            return discrete_token_embeds

        # Convert virtual token vocab_id to virtual token embedding idx
        virtual_token_ids = input_ids.clone()
        virtual_token_ids = torch.sub(virtual_token_ids, self.pseudo_token_ids_start)
        virtual_token_ids = torch.clamp(virtual_token_ids, min=0)

        # Only get needed virtual token embeddings from the prompt table according to virtual token ids
        virtual_token_embeds = [self.prompt_table(taskname_ids[i], virtual_token_ids[i]) for i in range(batch_size)]
        virtual_token_embeds = torch.stack(virtual_token_embeds)

        # Make sure discrete_token_embeds and virtual_token_embeds share the same dtype
        discrete_token_embeds = discrete_token_embeds.type(virtual_token_embeds.dtype)

        # Put virtual and discrete token embs in their correct locations for final output
        input_embeds = torch.where(virtual_token_locations, virtual_token_embeds, discrete_token_embeds)
        return input_embeds

    def fwd_bwd_step(self, batch, batch_idx, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # Get seq length of batch
        _, seq_length = batch[0].shape
        tensor_shape = [seq_length, self.cfg.micro_batch_size, self.hidden_size]

        if self.pipeline_parallel:
            losses_reduced_per_micro_batch = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
                sequence_parallel_enabled=self.cfg.get("sequence_parallel", False),
                sync_batch_comm=self.frozen_model.cfg.get('sync_batch_comm', False),
            )
        else:
            losses_reduced_per_micro_batch = forward_backward_no_pipelining(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
                sync_batch_comm=self.frozen_model.cfg.get('sync_batch_comm', False),
            )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def training_step(self, batch, batch_idx):
        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()
        loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        return loss_mean

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def validation_step(self, batch, batch_idx):
        loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=True)
        if loss_mean.item == 0.0:
            loss_mean = []

        return loss_mean

    def validation_epoch_end(self, outputs):
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        logging.info(f'val_loss: {averaged_loss}')

        # Save inference ready .nemo checkpoint version
        if self.cfg.get("save_nemo_on_validation_end", False):
            if self.lowest_val_loss is None or averaged_loss < self.lowest_val_loss:
                self.save_checkpoint_as_nemo_file()
                self.lowest_val_loss = averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def update_config_for_inference_and_save(self):
        self.virtual_prompt_style = VirtualPromptStyle.INFERENCE
        self.virtual_prompt_source = VirtualPromptSource.PROMPT_TABLE

        # Move new tags to existing tag list for loading during inference later
        with open_dict(self.cfg):
            self.cfg.existing_tasks = self.existing_tasks + self.new_tasks
            self.cfg.new_tasks = []
            self.cfg.virtual_prompt_style = VirtualPromptStyle.INFERENCE.value

        # Save the best nemo model
        self.save_to(save_path=self.cfg.nemo_path)
        logging.info(f"An inference ready model was saved to {self.cfg.nemo_path}")

    def save_checkpoint_as_nemo_file(self):
        self.add_ptuned_prompts_to_prompt_table()

        # Save current config and state dict params in temp values
        current_virtual_prompt_style = self.virtual_prompt_style
        current_virtual_prompt_source = self.virtual_prompt_source
        current_existing_tasks = self.cfg.existing_tasks
        current_new_tasks = self.cfg.new_tasks

        # Temporarily overwrite params to save an inference ready .nemo checkpoint
        self.update_config_for_inference_and_save()

        # Set values back to their training state to continue training
        self.virtual_prompt_style = current_virtual_prompt_style
        self.virtual_prompt_source = current_virtual_prompt_source

        # Revert prompt table back to previous state
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING and self.frozen_model.model.pre_process:
            for taskname in current_new_tasks:
                if taskname in self.prompt_table.prompt_table:
                    del self.prompt_table.prompt_table[taskname]

        with open_dict(self.cfg):
            self.cfg.existing_tasks = current_existing_tasks
            self.cfg.new_tasks = current_new_tasks
            self.cfg.virtual_prompt_style = current_virtual_prompt_style.value

    def on_train_end(self):
        # Save p-tuned prompts to prompt table for inference or future task training
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING and self.frozen_model.model.pre_process:
            self.add_ptuned_prompts_to_prompt_table()
            logging.info(f"All p-tuned prompts where moved to the prompt table.")

            # Remove prompt encoder from model
            self.prompt_encoder = None
            logging.info(f"Prompt encoder deleted")

        self.update_config_for_inference_and_save()

    def setup(self, stage=None):
        if (
            stage == 'predict' or self.virtual_prompt_style == VirtualPromptStyle.INFERENCE
        ) and self.frozen_model.model.pre_process:
            self.freeze_existing_virtual_prompt_params()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        if self.frozen_model.model.pre_process:
            if self.virtual_prompt_style == VirtualPromptStyle.PROMPT_TUNING:
                self.init_new_prompts()
            elif self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
                self.init_prompt_encoder()

            self.freeze_existing_virtual_prompt_params()

        self.setup_training_data()
        self.setup_validation_data()

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('train_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def setup_validation_data(self, validation_data_config=None):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.validation_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('validation_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.test_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('test_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def build_virtual_prompt_dataset(
        self,
        data,
        batch_size=None,
        max_seq_length=2048,
        min_seq_length=1,
        add_bos=False,
        add_eos=False,
        for_train=True,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        tokens_to_generate=None,
        get_dataset_only=False,
        cache_data_path=None,
        load_cache=False,
    ):
        dataset = GPTPromptLearningDataset(
            data=data,
            tokenizer=self.tokenizer,
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
            tokens_to_generate=tokens_to_generate,
            cache_data_path=cache_data_path,
            load_cache=load_cache,
        )

        if get_dataset_only:
            return dataset

        # Make distributed dataloader
        rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=data_parallel_size, rank=rank, shuffle=shuffle
        )

        assert batch_size % data_parallel_size == 0, "Global batch size must be evenly divisible by data parallel size"

        if for_train:
            if self.cfg.get("sequence_parallel", False):
                collate_fn = partial(
                    dataset.collate_fn, tp_workers=parallel_state.get_tensor_model_parallel_world_size()
                )
            else:
                collate_fn = partial(dataset.collate_fn, tp_workers=0)
        else:
            collate_fn = dataset.inference_collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=batch_size // data_parallel_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataset, dataloader

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""

        self.frozen_model.model.set_input_tensor(input_tensor)

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids = batch
            output_tensor = model(input_ids, position_ids, attention_mask, taskname_ids, labels, inference=False)

            if isinstance(output_tensor, tuple):
                output_tensor, _ = output_tensor

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        """
        Used for generate method only for now.
        """

        def fwd_output_only_func(batch, model):
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

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ):

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()
            sampling_params["add_BOS"] = self.cfg.data.get("add_bos", False)

        if length_params is None:
            length_params = get_default_length_params()

        max_input_length = self.frozen_model.cfg.encoder_seq_length - length_params["max_length"]

        # input dicts are either dataset paths or already loaded example dicts
        if "taskname" not in inputs[0].keys():
            data = [path["data_path"] for path in inputs]
        else:
            data = inputs

        dataset = self.build_virtual_prompt_dataset(
            data=data,
            max_seq_length=max_input_length,
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=sampling_params["add_BOS"],
            add_eos=False,
            for_train=False,
            tokens_to_generate=length_params["max_length"],
            get_dataset_only=True,
        )

        full_dataset = [dataset[i] for i in range(len(dataset))]
        task_ids, processed_inputs = dataset.inference_collate_fn(full_dataset)
        self.frozen_model.model.parallel_output = False

        # Call same generate code as in MegatronGPT
        return megatron_gpt_generate(
            self.cuda(), processed_inputs, self.tokenizer, length_params, sampling_params, task_ids=task_ids
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            length_params: LengthParam = {
                "max_length": inference_config["tokens_to_generate"],
                "min_length": inference_config["min_tokens_to_generate"],
            }

            sampling_params: SamplingParam = {
                "use_greedy": inference_config["greedy"],
                "temperature": inference_config["temperature"],
                "top_k": inference_config["top_k"],
                "top_p": inference_config["top_p"],
                "repetition_penalty": inference_config["repetition_penalty"],
                "add_BOS": inference_config["add_BOS"],
                "all_probs": inference_config["all_probs"],
                "compute_logprob": inference_config["compute_logprob"],
            }

            task_ids, processed_inputs = batch
            self.frozen_model.model.parallel_output = False

            # Call same generate code as in MegatronGPT
            return megatron_gpt_generate(
                self.cuda(), processed_inputs, self.tokenizer, length_params, sampling_params, task_ids=task_ids
            )

    @classmethod
    def list_available_models(cls):
        pass


def get_pseudo_tokens(num_virtual_tokens):
    """
    Takes in an integer and returns a list of strings where each string
    is a numbered virtual token placeholder. If 
    num_virtual_tokens = 3, then this function returns:

    ["<prompt_0>", "<prompt_1>", "<prompt_2>"]

    Args:
        num_virtual_tokens: (int) Number of virtual token strings you want to make

    returns a list of string. 

    """
    pseudo_tokens = [
        VirtualPromptPlaceholderToken.BASE.value + str(i) + VirtualPromptPlaceholderToken.END.value
        for i in range(num_virtual_tokens)
    ]

    return pseudo_tokens
