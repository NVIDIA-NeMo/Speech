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
import os
from functools import partial
from typing import Any, List, Optional, Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.metrics.prompt_learning_metrics import AccuracyScore, BLEUScore, ROUGEScores
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronBasePromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common import VirtualPromptPlaceholderToken, VirtualPromptSource, VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging
from nemo.utils.decorators import deprecated_warning

try:
    from megatron.core import InferenceParams, ModelParallelConfig, parallel_state, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import get_micro_batch_size, get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_micro_batch_size, get_num_microbatches


__all__ = ['MegatronGPTPromptLearningModel']


class MegatronGPTPromptLearningModel(MegatronBasePromptLearningModel):
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
        # deprecation warning
        deprecated_warning("MegatronGPTPromptLearningModel")

        super().__init__(cfg, trainer)

        self.inference_params = None

        # init_model is called by parent class already.
        # self.init_model(cfg, trainer)

    def init_model(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        self.config: ModelParallelConfig = self.model_parallel_config
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.get('language_model_path')):
            save_restore_connector.model_extracted_dir = cfg.get('language_model_path')
        frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        # set hidden size in the model parallel config for pipeline parallel schedules
        setattr(self.config, 'hidden_size', frozen_model_cfg.hidden_size)

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

        if cfg.get('language_model_path', None):
            self.frozen_model = MegatronGPTModel.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                override_config_path=frozen_model_cfg,
            ).to(dtype=self.autocast_dtype)

        self.megatron_amp_O2 = self.cfg.get('megatron_amp_O2', False)
        self.pipeline_parallel = self.cfg.get('pipeline_model_parallel_size', 1) > 1
        self.tokenizer = self.frozen_model.tokenizer
        self.hidden_size = self.frozen_model.cfg.hidden_size
        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))
        with open_dict(self.cfg):
            self.cfg.existing_tasks = (
                self.existing_tasks + self.new_tasks
            )  # TODO: for backward compatibility (@adithyare) in general these tasks lists should be deprecated

        self.virtual_prompt_style = VirtualPromptStyle(cfg.virtual_prompt_style)
        self.model_type = ModelType.encoder_or_decoder

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        if self.pipeline_parallel:
            assert (
                self.cfg.optim.sched.get("min_lr", 0.0) == 0.0
            ), "Minimum lr must be 0.0 when pipeline parallel size is > 1"

        # Load templates for assigning virtual prompt token positions
        self.load_task_templates(self.cfg.task_templates)

        if self.first_stage_of_pipeline() and self.virtual_prompt_style in [
            VirtualPromptStyle.P_TUNING,
        ]:
            if self.frozen_model.mcore_gpt:
                self.word_embeddings = self.frozen_model.model.embedding.word_embeddings
            else:
                self.word_embeddings = self.frozen_model.model.language_model.embedding.word_embeddings

        self.padded_vocab_size = self.frozen_model.padded_vocab_size

        # Prepare pseudo token ids for virtual/virtual prompt tokens
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        if isinstance(self.tokenizer, SentencePieceTokenizer):
            if not self.tokenizer.legacy:
                if self.tokenizer.pad_id != -1:
                    self.tokenizer.pad_token = self.tokenizer.ids_to_tokens([self.tokenizer.pad_id])[0]
                else:
                    self.tokenizer.pad_token = self.tokenizer.ids_to_tokens([self.tokenizer.eos_id])[0]
                self.tokenizer.bos_token = self.tokenizer.ids_to_tokens([self.tokenizer.bos_id])[0]
                self.tokenizer.eos_token = self.tokenizer.ids_to_tokens([self.tokenizer.eos_id])[0]
                self.tokenizer.legacy = True
            self.tokenizer.add_special_tokens(self.pseudo_tokens)
        else:
            self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0] if self.pseudo_token_ids else None
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        # P-Tuning uses an LSTM Encoder to produce virtual token embeddings
        if self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_ENCODER
        elif self.virtual_prompt_style == VirtualPromptStyle.NO_PROMPT:
            self.virtual_prompt_source = VirtualPromptSource.NO_PROMPT
        else:
            raise ValueError(f"\nvirtual prompt style '{cfg.virtual_prompt_style}.'")

        self._reduced_loss_buffer = []
        self._inference_config = None

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = True
        self.lowest_val_loss = None
        self.prompt_encoder = None

        # default inference related params -> for evaluation metrics
        if hasattr(self.cfg, 'inference') and self.cfg.get("report_validation_metric", False):
            self.length_params: LengthParam = {
                "max_length": self.cfg.inference.get('tokens_to_generate', 30),
                "min_length": self.cfg.inference.get('min_tokens_to_generate', 0),
            }

            self.sampling_params: SamplingParam = {
                "use_greedy": self.cfg.inference.get('greedy', False),
                "temperature": self.cfg.inference.get('temperature', 1.0),
                "top_k": self.cfg.inference.get('tok_k', 0),
                "top_p": self.cfg.inference.get('top_p', 0.9),
                "repetition_penalty": self.cfg.inference.get('repetition_penalty', 1.2),
                "add_BOS": True,
                "all_probs": False,
                "compute_logprob": False,
                "end_strings": self.cfg.inference.get('end_strings', ["<|endoftext|>"]),
            }
        elif self.cfg.get("report_validation_metric", False) and not hasattr(self.cfg, 'inference'):
            raise ValueError("Must provide inference parameters for reporting validation metric!")

        # define validation metric
        if self.cfg.get('report_validation_metric', False):
            validation_metric = self.cfg.get('validation_metric', 'accuracy')
            if validation_metric == 'accuracy':
                self.validation_metric = AccuracyScore()
            elif validation_metric == 'bleu':
                self.validation_metric = BLEUScore()
            elif validation_metric == 'rouge':
                self.validation_metric = ROUGEScores()

    def first_stage_of_pipeline(self):
        return self.frozen_model.model.pre_process

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
        inference_params=None,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        GPT style models. Bypasses the vocab token preprocessing done
        in the MegatronGPT class.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        if self.first_stage_of_pipeline():
            input_embeds = self.embed_input(input_ids, taskname_ids, use_cached_reps=inference)
            if self.frozen_model.mcore_gpt and hasattr(self.frozen_model.model.embedding, "position_embeddings"):
                position_embeddings = self.frozen_model.model.embedding.position_embeddings(position_ids)
                encoder_input = input_embeds + position_embeddings
            elif not self.frozen_model.mcore_gpt and hasattr(
                self.frozen_model.model.language_model.embedding, "position_embeddings"
            ):
                position_embeddings = self.frozen_model.model.language_model.embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            if self.cfg.get("sequence_parallel", False):
                encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        else:
            encoder_input = None

        # Call forward on GPT model with preprocessed embeddings
        if self.frozen_model.mcore_gpt:
            output = self.frozen_model.model(
                input_ids=None,
                position_ids=None,
                decoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                inference_params=inference_params,
            )
        else:
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

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        """
        Dataloader produces a global batch which is turned into an iterator of microbatches.
        The iterator of microbatches is then piped through the pipeline using Core's fwd/bwd functions.
        """
        # Get seq length of batch
        batch, _, _ = next(dataloader_iter)
        _, seq_length = batch[0].shape
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=[self],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
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

    def training_step(self, dataloader_iter):
        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()
        batch, batch_idx, _ = next(dataloader_iter)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.torch_dtype == torch.float16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        return loss_mean

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
        No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def validation_step(self, dataloader_iter):
        mode = 'test' if self.trainer.testing else 'val'
        batch, batch_idx, _ = next(dataloader_iter)
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(batch[0].size(0), gbs)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=True)
        if loss_mean.item == 0.0:
            loss_mean = []

        if self.cfg.get('report_validation_metric', False):
            preds_text, labels_text = [], []
            input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids = batch
            input_lenghts = torch.argmax(loss_mask, 1, keepdim=True)

            res = megatron_gpt_generate(
                self.cuda(),
                (
                    torch.cat(
                        (
                            input_ids,
                            torch.zeros(
                                input_ids.shape[0], self.length_params['max_length'], dtype=input_ids.dtype
                            ).to(self.device),
                        ),
                        axis=1,
                    ),
                    input_lenghts.squeeze(1),
                ),
                self.tokenizer,
                self.length_params,
                self.sampling_params,
                task_ids=taskname_ids,
            )

            for pred, label in zip(res['token_ids'], labels):
                # ids_to_text ignores special tokens by default
                pred = self.tokenizer.ids_to_text(pred)
                label = self.tokenizer.ids_to_text(label)
                preds_text.append(pred)
                labels_text.append(label)
            if mode == 'val':
                self.validation_step_outputs.append(
                    {
                        'loss': loss_mean,
                        'preds': preds_text,
                        'labels': labels_text,
                    }
                )
            else:
                self.test_step_outputs.append(
                    {
                        'loss': loss_mean,
                        'preds': preds_text,
                        'labels': labels_text,
                    }
                )
            return {
                'loss': loss_mean,
                'preds': preds_text,
                'labels': labels_text,
            }

        (
            self.validation_step_outputs.append({'loss': loss_mean})
            if mode == 'val'
            else self.test_step_outputs.append({'loss': loss_mean})
        )
        return {'loss': loss_mean}

    def on_train_epoch_start(self) -> None:
        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        mbs = self.cfg.get('validation_micro_batch_size', self.cfg.micro_batch_size)
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            averaged_loss = torch.stack([i['loss'] for i in self.validation_step_outputs]).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        logging.info(f'val_loss: {averaged_loss}')

        if self.cfg.get("report_validation_metric", False):
            gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]

            all_preds = list(itertools.chain(*[item['preds'] for item in self.validation_step_outputs]))
            all_labels = list(itertools.chain(*[item['labels'] for item in self.validation_step_outputs]))

            assert len(all_preds) == len(all_labels)

            # Gather inputs, preds, labels from all workers
            torch.distributed.all_gather_object(
                gather_results,
                [(pred, label) for (pred, label) in zip(all_preds, all_labels)],
                group=parallel_state.get_data_parallel_group(),
            )

            # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
            if parallel_state.get_data_parallel_rank() == 0:

                gather_results_dedup = list(set(itertools.chain(*gather_results)))

                val_metric_dict = self.validation_metric.get_score(
                    [i[1] for i in gather_results_dedup],
                    [i[0] for i in gather_results_dedup],
                )

                for metric, val in val_metric_dict.items():
                    logging.info(f'Validation {metric}: {val}')
                val_metric = list(val_metric_dict.items())[0][1]
                metric_name = list(val_metric_dict.items())[0][0]
            else:
                val_metric = torch.tensor(0.0).cuda()
                metric_name = ''

            self.log(f'val_{metric_name}', val_metric, prog_bar=True, rank_zero_only=True, batch_size=1)

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, dataloader_iter):
        return self.validation_step(dataloader_iter)

    def on_test_epoch_end(self):
        averaged_loss = average_losses_across_data_parallel_group(self.test_step_outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        self.test_step_outputs.clear()  # free memory

    def setup(self, stage=None):
        super().setup(stage)

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_gpt', False):
            self.frozen_model.setup_transformer_engine_tp_groups()

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            max_seq_length = self.frozen_model.cfg.encoder_seq_length
            if "max_seq_length" in self.cfg.data and self.cfg.data.max_seq_length:
                max_seq_length = min(self.cfg.data.max_seq_length, max_seq_length)
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=max_seq_length,
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
            max_seq_length = self.frozen_model.cfg.encoder_seq_length
            if "max_seq_length" in self.cfg.data and self.cfg.data.max_seq_length:
                max_seq_length = min(self.cfg.data.max_seq_length, max_seq_length)
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.validation_ds,
                batch_size=self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size),
                max_seq_length=max_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=self.cfg.get('validation_drop_last', True),
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
                batch_size=self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size),
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
        batch_size,
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
            dataset, num_replicas=data_parallel_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
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
            persistent_workers=(
                True if num_workers > 0 else False
            ),  # (@adithyare and @eharper) We need this to make spawn=True to work.
        )

        return dataset, dataloader

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""

        self.frozen_model.model.set_input_tensor(input_tensor)

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
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

        def fwd_output_only_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
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

            if self.frozen_model.mcore_gpt:
                # if first step, then clear KV cache, otherwise reuse inference_paarms
                if set_inference_key_value_memory[0].item():
                    self.inference_params = InferenceParams(
                        max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                    )
                extra_arg['inference_params'] = self.inference_params
            else:
                extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            output_tensor = model(tokens, position_ids, attention_mask, task_ids, **extra_arg)

            # Advance inference sequence offset.
            if self.inference_params:
                self.inference_params.sequence_len_offset += output_tensor.size(1)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        batch_size: Optional[int] = 1,
    ):

        # check whether the DDP is initialized
        if not parallel_state.is_initialized():

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
            batch_size=batch_size,
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
                "compute_attention_mask": inference_config.get("compute_attention_mask", True),
                "end_strings": inference_config.get('end_strings', ["<|endoftext|>"]),
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
