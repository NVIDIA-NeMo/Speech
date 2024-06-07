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

import itertools
import re
from collections import OrderedDict
from typing import Any, Optional

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.nlp.metrics.prompt_learning_metrics import AccuracyScore, BLEUScore, ROUGEScores
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common import (
    PromptEncoder,
    PromptEncoderType,
    VirtualPromptPlaceholderToken,
    VirtualPromptSource,
    VirtualPromptStyle,
)
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.collections.nlp.modules.common.transformer.text_generation import TextGeneration
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import ModelParallelConfig, parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


__all__ = ['MegatronBasePromptLearningModel']


class MegatronBasePromptLearningModel(MegatronBaseModel, TextGeneration):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron model.

    Prompt Tuning initalizes virtual prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained model's vocabulary
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
        self.init_model(cfg, trainer)

    def init_model(self, cfg: DictConfig, trainer: Trainer):

        self.config: ModelParallelConfig = self.model_parallel_config

        self.load_frozen_model(cfg, trainer)
        self.prompt_encoder = None
        self.tokenizer = self.frozen_model.tokenizer

        if hasattr(self.frozen_model.cfg, "encoder") and hasattr(self.frozen_model.cfg, "decoder"):
            self.hidden_size = (
                self.frozen_model.cfg.encoder.hidden_size
            )  # Encoder and decoder need to have the same hidden size and we check for this in the frozen enc-dec model.
            self.config.hidden_size = self.hidden_size
        else:
            self.hidden_size = self.frozen_model.cfg.hidden_size
            self.config.hidden_size = self.hidden_size

        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))
        self.virtual_prompt_style = VirtualPromptStyle(cfg.virtual_prompt_style)

        # Load templates for assigning virtual prompt token positions
        self.load_task_templates(self.cfg.task_templates)

        if self.first_stage_of_pipeline() and self.virtual_prompt_style in [
            VirtualPromptStyle.P_TUNING,
        ]:

            # TODO: Handle this when moving GPT prompt learning to the base class.
            self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings

        # P-Tuning uses an LSTM Encoder to produce virtual token embeddings
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_ENCODER
        elif self.virtual_prompt_style == VirtualPromptStyle.NO_PROMPT:
            self.virtual_prompt_source = VirtualPromptSource.NO_PROMPT
        else:
            raise ValueError(f"\nvirtual prompt style '{cfg.virtual_prompt_style}'")

        self._reduced_loss_buffer = []
        self._inference_config = None

        # Prepare pseudo token ids for virtual/virtual prompt tokens
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        if isinstance(self.tokenizer, SentencePieceTokenizer):
            self.tokenizer.add_special_tokens(self.pseudo_tokens)
        else:
            self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0] if self.pseudo_token_ids else None
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id
        self.decoder_seq_length = cfg.get('decoder_seq_length', 40)

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = True
        self.lowest_val_loss = None
        self.prompt_encoder = None

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        # define validation metric
        if self.cfg.get('report_validation_metric', False):
            validation_metric = self.cfg.get('validation_metric', 'accuracy')
            if validation_metric == 'accuracy':
                self.validation_metric = AccuracyScore()
            elif validation_metric == 'bleu':
                self.validation_metric = BLEUScore()
            elif validation_metric == 'rouge':
                self.validation_metric = ROUGEScores()

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

    def init_prompt_encoder(self):
        """
        Init the prompt encoder needed for p-tuning on a new task
        """
        # Total virtual tokens should be the same across all new tasks, so just need one
        new_task = self.new_tasks[0]
        total_virtual_tokens = self.task_templates[new_task]["total_virtual_tokens"]

        encoder_type = PromptEncoderType(self.cfg.p_tuning.get("encoder_type", "tpmlp").lower())
        self.prompt_encoder = PromptEncoder(
            config=self.model_parallel_config,
            encoder_type=encoder_type,
            total_virtual_tokens=total_virtual_tokens,
            token_dim=self.hidden_size,
            hidden_size=self.cfg.p_tuning.get("encoder_hidden", self.hidden_size // 2),
            lstm_dropout=self.cfg.p_tuning.get("dropout", 0.0),
            num_layers=self.cfg.p_tuning.get("num_layers", 2),
            init_std=self.cfg.p_tuning.get("init_std", 0.023),
            taskname=new_task,
        )

    def freeze_existing_word_embeddings(self):
        """Freeze params of existing virtual prompts that should not be tuned further"""
        # Make sure word embeddings are frozen
        for params in self.word_embeddings.parameters():
            params.requires_grad = False

    def state_dict(self):
        """
        Custom state dict that only contains prompt table and prompt encoder parameters.
        No frozen model parameters are stored in the state dict. Prompt encoder parameters
        are only in state dict for intermediate checkpoints saved during training. Final
        nemo checkpoints at the end of training will contain prompt table parameters only.
        """
        state_dict_ = {}

        if self.first_stage_of_pipeline():
            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                state_dict_ = self.prompt_encoder.state_dict()
            else:
                raise ValueError("invalid virtual prompt source")

        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method.
        """
        if self.first_stage_of_pipeline():
            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                if self.prompt_encoder is None:
                    self.init_prompt_encoder()
                self.prompt_encoder.load_state_dict(state_dict, strict)
            else:
                raise ValueError("invalid virtual prompt source")

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups.
        Only want virtual prompt params to be passed to the optimizer.
        """
        ## Freeze frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        virtual_prompt_params = {'params': []}

        if self.first_stage_of_pipeline():
            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                virtual_prompt_params['params'].extend([param for param in self.prompt_encoder.parameters()])
            else:
                raise ValueError("Optimizer only supports Prompt Encoder.")

        self._optimizer_param_groups = (virtual_prompt_params,)

    def embed_input(self, input_ids: Tensor, taskname_ids: Tensor, use_cached_reps: bool):
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

        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            # taskname_embeddings = self.word_embeddings(taskname_ids)
            batch_size, _ = taskname_ids.size()
            virtual_token_embeds = self.prompt_encoder(batch_size=batch_size, use_cached_reps=use_cached_reps)
        else:
            raise ValueError("invalid VirtualPromptSource.")

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

    def on_train_end(self):
        # Save p-tuned prompts to prompt table for inference or future task training
        self.save_to(save_path=self.cfg.nemo_path)

    def setup(self, stage=None):
        if stage == 'predict' and self.first_stage_of_pipeline():
            self.freeze_existing_word_embeddings()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        if self.first_stage_of_pipeline():
            if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
                if self.prompt_encoder is None:
                    self.init_prompt_encoder()
            self.freeze_existing_word_embeddings()

        self.setup_training_data()
        self.setup_validation_data()

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_validation_data(self, validation_data_config=None):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.validation_ds,
                batch_size=self.cfg.get("validation_global_batch_size", self.cfg.global_batch_size),
                for_train=True,
                drop_last=self.cfg.get("validation_drop_last", True),
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.test_ds,
                batch_size=self.cfg.get("validation_global_batch_size", self.cfg.global_batch_size),
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def _reconfigure_and_process_inference_batch(self, global_batch_size_per_gpu, gbs):
        # This should happen only on the last batch of the dataset.
        if global_batch_size_per_gpu != gbs // parallel_state.get_data_parallel_world_size():
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            app_state = AppState()
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_size_per_gpu,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

    def _reconfigure_batch_sizes(self, gbs: int, mbs: int):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def set_input_tensor(self, input_tensor):
        pass

    def first_stage_of_pipeline(self):
        pass

    @classmethod
    def list_available_models(cls):
        pass

    def load_frozen_model(self, cfg, trainer):
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
