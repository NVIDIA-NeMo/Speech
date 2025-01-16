# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import queue
import types
import warnings
from dataclasses import fields
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from lightning.pytorch.accelerators import CPUAccelerator
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)

# from nemo.collections.nlp.data.language_modeling.megatron.retro_dummy_dataset import build_train_valid_test_datasets as dummy_build_train_valid_test_datasets  # turn on when running with dummy data
from nemo.collections.nlp.data.language_modeling.megatron.retro_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_ltor_masks_and_position_ids,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from nemo.collections.nlp.parts import utils_funcs
from nemo.collections.nlp.parts.utils_funcs import activation_to_func, get_last_rank
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging
from nemo.utils.import_utils import safe_import, safe_import_from

try:
    from megatron.core import InferenceParams, parallel_state
    from megatron.core.models.retro import RetroModel as MCoreRetroModel
    from megatron.core.models.retro.config import RetroConfig
    from megatron.core.models.retro.decoder_spec import get_retro_decoder_block_spec
    from megatron.core.models.retro.utils import get_config_path as get_retro_config_path
    from megatron.core.models.retro.utils import get_gpt_data_dir as get_retro_data_dir
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal, scaled_init_method_normal

    # TODO @tmoon: Use once available in Megatron-LM
    # from megatron.core.pipeline_parallel.schedules import DataIteratorList

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    TransformerConfig = ApexGuardDefaults
    RetroConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

transformer_engine, HAVE_TE = safe_import("transformer_engine")
te_module, HAVE_TE_MODULE = safe_import_from("transformer_engine.pytorch", "module")
HAVE_TE = HAVE_TE and HAVE_TE_MODULE


class MegatronRetroModel(MegatronGPTModel):
    """
    Megatron Retro pretraining
    """

    def load_retro_config(self, cfg: DictConfig):
        assert cfg.retro.get('retro_project_dir') is not None, "`--retro-project-dir` must be set to use Retro."

        # Retro config path.
        retro_config_path = get_retro_config_path(cfg.retro.get('retro_project_dir'))
        assert os.path.exists(retro_config_path), "retro project dir missing config.json."

        # Load retro config.
        with open(retro_config_path) as f:

            # Parse config.
            retro_preprocess_config = types.SimpleNamespace(**json.load(f))

            # Retro data path is relative to data path (via hard or soft links).
            data_dir = get_retro_data_dir(cfg.retro.get('retro_project_dir'))
            data_path = list(retro_preprocess_config.retro_gpt_data_path)
            if len(data_path) % 2 == 0:
                for i in range(len(data_path) - 1, -1, -2):
                    data_path[i] = os.path.join(data_dir, data_path[i])
            else:
                assert len(data_path) == 1
                data_path[0] = os.path.join(data_dir, data_path[0])

            # Update args.
            cfg.global_batch_size = retro_preprocess_config.retro_gpt_global_batch_size
            cfg.seed = retro_preprocess_config.retro_gpt_seed
            cfg.data.data_prefix = data_path
            cfg.encoder_seq_length = retro_preprocess_config.retro_gpt_seq_length
            cfg.data.seq_length = retro_preprocess_config.retro_gpt_seq_length
            cfg.max_position_embeddings = retro_preprocess_config.retro_gpt_seq_length
            # cfg.data.splits_string = retro_preprocess_config.retro_gpt_split      # remove because lastest RETRO data-object have separate RETRO training split and RETRO preprocessing split
            cfg.tokenizer.model = (
                cfg.retro.get('retro_project_dir') + '/' + retro_preprocess_config.retro_gpt_tokenizer_model
            )
            cfg.tokenizer.type = retro_preprocess_config.retro_gpt_tokenizer_type
            cfg.tokenizer.vocab_file = retro_preprocess_config.retro_gpt_vocab_file
            cfg.tokenizer.merge_file = retro_preprocess_config.retro_gpt_merge_file
            with open_dict(cfg):
                cfg.retro_train_samples_with_neighbors = retro_preprocess_config.retro_gpt_train_samples
                cfg.retro_valid_samples_with_neighbors = retro_preprocess_config.retro_gpt_valid_samples
            cfg.data.retro_data.retro_block_size = retro_preprocess_config.retro_block_size
            cfg.data.retro_data.retro_chunk_length = retro_preprocess_config.retro_gpt_chunk_length
            cfg.data.retro_data.retro_split_preprocessing = retro_preprocess_config.retro_gpt_split
            cfg.data.retro_data.retro_neighbor_dirs = retro_preprocess_config.retro_neighbor_dirs

        return cfg

    def __init__(self, cfg: DictConfig, trainer: Trainer):

        # override pre-processing arguments with retro pre-processing arguments
        cfg = self.load_retro_config(cfg)

        super().__init__(cfg, trainer=trainer)

        logging.info(
            "\n\n************** Experiment configuration (after overriding with RETRO's workdir values) ***********"
        )
        logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

        return

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if self.mcore_gpt:
            self.retro_model_config = self.build_retro_config()
            model = MCoreRetroModel(
                config=self.retro_model_config,
                transformer_layer_spec=get_retro_decoder_block_spec(
                    self.retro_model_config, use_transformer_engine=True
                ),
                vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
                max_sequence_length=self.cfg.data.get('seq_length', 512),
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
                rotary_percent=self.cfg.get('rotary_percentage', 1.0),
                seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
            )

            return model
        else:
            assert self.mcore_gpt == True, "Currently only support mcore Retro."

    def forward(
        self, tokens, text_position_ids, attention_mask, labels, context_input_ids, context_position_ids, context_mask
    ):
        output_tensor = self.model(
            tokens,
            text_position_ids,
            attention_mask,
            context_input_ids=context_input_ids,
            context_position_ids=context_position_ids,
            context_mask=context_mask,
            labels=labels,
        )
        return output_tensor

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None, **extra) -> Any:
        # batch = {'prompts': List, 'neighbors': List[List]}

        inference_config = self.get_inference_config()

        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                inference_config['inputs'] = batch['prompts']
                inference_config['neighbors'] = batch['neighbors']
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                inference_config['retro_inference'] = inference_config['retro_inference']
                response = generate(self, **inference_config)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
                return compute_prob_response
            else:
                inference_config['inputs'] = batch['prompts']
                inference_config['neighbors'] = batch['neighbors']
                inference_config['retro_inference'] = inference_config['retro_inference']
                return generate(self, **inference_config)

    def get_batch(self, data_iterator):
        """Generate a batch."""

        # Broadcast data.
        if data_iterator is not None:
            # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
            data = next(data_iterator)
            if isinstance(data, tuple):
                data = data[0]
        else:
            data = None

        batch = {
            'tokens': data["tokens"],
            'labels': data["labels"],
            'loss_mask': data["loss_mask"],
            'attention_mask': data["attention_mask"],
            'position_ids': data["position_ids"],
            'context_input_ids': data["context_input_ids"],
            'context_attention_mask': data["context_attention_mask"],
            'context_position_ids': data["context_position_ids"],
        }

        return batch

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = self.get_batch(dataloader_iter)

            # Transfer needed data to GPU
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(
                        ('tokens', 'position_ids', 'context_input_ids', 'context_position_ids', 'context_mask')
                    )
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion:
                required_keys.remove('attention_mask')
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # reshape context_input_ids and context_position_ids for RETRO from [bs, l*k, r] => [bs*l*k, r]
            context_input_ids = batch['context_input_ids']
            context_position_ids = batch['context_position_ids']
            context_input_ids = context_input_ids.view(-1, context_input_ids.shape[-1]).long()
            context_position_ids = context_position_ids.view(-1, context_position_ids.shape[-1]).long()
            batch['context_input_ids'] = context_input_ids
            batch['context_position_ids'] = context_position_ids

            # slice batch along sequence dimension for context parallelism
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            # Model forward pass
            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': batch['attention_mask'],
                'context_input_ids': batch['context_input_ids'],
                'context_position_ids': batch['context_position_ids'],
                'context_mask': None,  # batch neighbor_attention_mask will be set to None following Lawrence's implementation
                'labels': batch['labels'],
                'loss_mask': batch['loss_mask'],
            }

            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop('loss_mask')
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop('loss_mask')
            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], batch['num_valid_tokens_in_ub'], output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['loss_mask'].sum()
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            if len(batch) == 6:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids, context_input_ids, context_mask, context_position_ids = batch
                attention_mask = attention_mask[0:1]
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    context_input_ids,
                    context_mask,
                    context_position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                # Transfer needed data to GPU
                tokens = tokens.cuda()
                position_ids = position_ids.cuda()
                context_input_ids = context_input_ids.cuda()
                context_position_ids = context_position_ids.cuda()
                context_mask = None
                if self.mcore_gpt:
                    # No caching key, value because currently it's not supported for mcore RETRO in NeMo
                    pass

                else:
                    extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                    extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                context_input_ids=context_input_ids,
                context_position_ids=context_position_ids,
                context_mask=None,  # batch neighbor_attention_mask will be set to None following Lawrence's implementation
                **extra_arg,
            )

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def build_retro_config(self) -> RetroConfig:
        """This method build RetroConfig from the already built TransformerConfig
        by adding Retro relevant variables. This method runs after running build_transformer_config() method.
        """
        retro_config = self.transformer_config

        # retro model args
        retro_config.retro_project_dir = self.cfg.retro.get('retro_project_dir')
        retro_config.retro_block_size = self.cfg.data.retro_data.get('retro_block_size')
        retro_config.retro_chunk_length = self.cfg.data.retro_data.get('retro_chunk_length')
        retro_config.retro_encoder_num_layers = self.cfg.retro.get('retro_encoder_num_layers', 2)
        retro_config.retro_encoder_hidden_dropout = self.cfg.retro.get('retro_encoder_hidden_dropout', 0.1)
        retro_config.retro_encoder_attention_dropout = self.cfg.retro.get('retro_encoder_attention_dropout', 0.1)
        retro_config.retro_num_neighbors = self.cfg.retro.get('retro_num_neighbors', 2)
        retro_config.retro_num_retrieved_chunks = self.cfg.retro.get('retro_num_retrieved_chunks', 2)
        retro_config.retro_verify_neighbor_count = self.cfg.retro.get('retro_verify_neighbor_count', True)
        retro_config.retro_retrieved_length = retro_config.retro_num_retrieved_chunks * retro_config.retro_chunk_length
        retro_config.retro_split_preprocessing = self.cfg.data.retro_data.get('retro_split_preprocessing')
        retro_config.retro_neighbor_dirs = self.cfg.data.retro_data.get('retro_neighbor_dirs')
        logging.info("retro_config: ")
        logging.info(retro_config)

        # Validate Transformer Engine version.
        from importlib.metadata import version

        import packaging

        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("1.3"):
            if HAVE_MEGATRON_CORE:
                retro_config.attention_backend = AttnBackend.unfused
            try:
                os.environ["NVTE_FLASH_ATTN"] = "0"
                os.environ["NVTE_FUSED_ATTN"] = "0"
                assert os.getenv("NVTE_FLASH_ATTN") == "0"
                assert os.getenv("NVTE_FUSED_ATTN") == "0"
            except Exception as e:
                raise Exception(
                    "When using Transformer Engine >= 1.3, environment vars NVTE_FLASH_ATTN and NVTE_FUSED_ATTN most both be defined and set to '0'. Currently, NVTE_FLASH_ATTN == %s, NVTE_FUSED_ATTN == %s."
                    % (
                        os.getenv("NVTE_FLASH_ATTN", "[unset]"),
                        os.getenv("NVTE_FUSED_ATTN", "[unset]"),
                    )
                )

        return retro_config

    def build_train_valid_test_datasets(self):
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting in between a step
        # self._reconfigure_val_batches()
        logging.info('Building mcore RETRO datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.cfg.global_batch_size
        # max_train_steps = self.trainer.max_steps
        # eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches # check this carefully, we want to match mcore dataset value, should this computed, or overriden?
        # test_iters = self.trainer.limit_test_batches

        # getting train_valid_test_num_samples from values in RETRO's workdir
        train_valid_test_num_samples = [  # compute the number of training/validating samples from workdir/query/train_*; dividing number of chunks for (2048/64)
            self.cfg.retro_train_samples_with_neighbors,
            self.cfg.retro_valid_samples_with_neighbors,
            0,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[1] = (
                1  # This is to make sure we only have one epoch on every validation iteration
            )

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            retro_config=self.retro_model_config,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.cfg.data.seq_length,
            tokenizer=self.tokenizer,
        )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building mcore RETRO datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(
        self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    global_batch_size=self.cfg.global_batch_size,
                    rampup_batch_size=self.cfg.get('rampup_batch_size', None),
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def fwd_bwd_step(self, dataloader_iter, forward_only):

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam and not self.use_mcore_dist_optim:
            no_sync_func = partial(
                self._optimizer.no_sync,
                greedy_grad_copy=self.megatron_amp_O2,
            )
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        # pipeline schedules will get these from self.model.config
        for module in self.get_model_module_list():
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func
            module.config.param_sync_func = param_sync_func

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: add num_micro_batches_with_partial_activation_checkpoints when ready
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=self._make_data_iterator_list(dataloader_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=self.cfg.encoder_seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                loss_sum_tensors_list = [
                    loss_sum['loss_sum_and_ub_size']
                    for loss_sum in losses_reduced_per_micro_batch
                    if loss_sum['loss_sum_and_ub_size'][1] > 0
                ]
                loss_sum = (
                    torch.vstack(loss_sum_tensors_list).sum(axis=0)
                    if len(loss_sum_tensors_list) > 0
                    else torch.tensor([0.0, 0.0]).cuda()
                )
                return loss_sum
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def validation_step(self, dataloader_iter, dataloader_idx=0):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        mode = 'test' if self.trainer.testing else 'val'
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.eval()
        else:
            self.model.eval()

        if self.cfg.get('fp8', False):
            first_val_step = self.prev_step_training and not self.training
            self.prev_step_training = self.training
        else:
            first_val_step = None

        with torch.no_grad():
            loss = self.fwd_bwd_step(dataloader_iter, True)

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.train()
        else:
            self.model.train()

        if mode == 'val':
            # Append with the correct dataloader_idx in case of multiple dataloaders
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                self.validation_step_outputs[dataloader_idx].append(loss)
            else:
                self.validation_step_outputs.append(loss)
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(loss)
            else:
                self.test_step_outputs.append(loss)

        return loss
