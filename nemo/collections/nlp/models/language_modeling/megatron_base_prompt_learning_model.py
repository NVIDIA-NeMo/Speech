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

import re
from collections import OrderedDict

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common import (
    PromptEncoder,
    PromptEncoderType,
    PromptTable,
    VirtualPromptPlaceholderToken,
    VirtualPromptSource,
    VirtualPromptStyle,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import TextGeneration
from nemo.utils import logging

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

        self.cfg = cfg

        self.load_frozen_model(cfg, trainer)

        self.tokenizer = self.frozen_model.tokenizer

        if hasattr(self.frozen_model.cfg, "encoder") and hasattr(self.frozen_model.cfg, "decoder"):
            self.hidden_size = (
                self.frozen_model.cfg.encoder.hidden_size
            )  # Encoder and decoder need to have the same hidden size and we check for this in the frozen enc-dec model.
        else:
            self.hidden_size = self.frozen_model.cfg.hidden_size

        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))
        self.virtual_prompt_style = VirtualPromptStyle(cfg.virtual_prompt_style)

        # Load templates for assigning virtual prompt token positions
        self.load_task_templates(self.cfg.task_templates)

        if self.first_stage_of_pipeline() and self.virtual_prompt_style in [
            VirtualPromptStyle.P_TUNING,
            VirtualPromptStyle.PROMPT_TUNING,
            VirtualPromptStyle.INFERENCE,
        ]:

            # TODO: Handle this when moving GPT prompt learning to the base class.
            self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings

            # Prompt table stores all task embeddings, p-tuning virtual prompts get added to the table after training
            self.prompt_table = PromptTable(
                existing_tasks=self.existing_tasks,
                task_templates=self.task_templates,
                task_id_num_to_name=self.task_id_num_to_name,
                hidden_size=self.hidden_size,
            )

        self._prompt_table_key = VirtualPromptSource.PROMPT_TABLE.value
        self._prompt_encoder_key = VirtualPromptSource.PROMPT_ENCODER.value

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

        if self.trainer.precision == 32:
            self.autocast_dtype = torch.float
        elif self.trainer.precision == 16:
            self.autocast_dtype = torch.half
        elif self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')
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

        self.prompt_encoder = PromptEncoder(
            encoder_type=PromptEncoderType(self.cfg.p_tuning.get("encoder_type", "mlp").lower()),
            total_virtual_tokens=total_virtual_tokens,
            token_dim=self.hidden_size,
            hidden_size=self.cfg.p_tuning.get("encoder_hidden", self.hidden_size // 2),
            lstm_dropout=self.cfg.p_tuning.get("dropout", 0.0),
            num_layers=self.cfg.p_tuning.get("num_layers", 2),
        )

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
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING and self.first_stage_of_pipeline():
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

    def state_dict(self):
        """
        Custom state dict that only contains prompt table and prompt encoder parameters. 
        No frozen model parameters are stored in the state dict. Prompt encoder parameters 
        are only in state dict for intermediate checkpoints saved during training. Final
        nemo checkpoints at the end of training will contain prompt table parameters only. 
        """
        state_dict_ = {}

        if self.first_stage_of_pipeline():
            state_dict_[self._prompt_table_key] = self.prompt_table.state_dict()

            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                state_dict_[self._prompt_encoder_key] = self.prompt_encoder.state_dict()

        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method. 
        """
        if self.first_stage_of_pipeline():
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
                self.prompt_encoder.load_state_dict(state_dict_, strict)

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
        discrete_token_embeds = self.word_embeddings(discrete_token_ids).clone()

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
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING and self.first_stage_of_pipeline():
            for taskname in current_new_tasks:
                if taskname in self.prompt_table.prompt_table:
                    del self.prompt_table.prompt_table[taskname]

        with open_dict(self.cfg):
            self.cfg.existing_tasks = current_existing_tasks
            self.cfg.new_tasks = current_new_tasks
            self.cfg.virtual_prompt_style = current_virtual_prompt_style.value

    def on_train_end(self):
        # Save p-tuned prompts to prompt table for inference or future task training
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
            self.add_ptuned_prompts_to_prompt_table()
            logging.info(f"All p-tuned prompts where moved to the prompt table.")

            # Remove prompt encoder from model
            self.prompt_encoder = None
            logging.info(f"Prompt encoder deleted")

        self.update_config_for_inference_and_save()

    def setup(self, stage=None):
        if (
            stage == 'predict' or self.virtual_prompt_style == VirtualPromptStyle.INFERENCE
        ) and self.first_stage_of_pipeline():
            self.freeze_existing_virtual_prompt_params()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        if self.first_stage_of_pipeline():
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
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.test_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
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
