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
import itertools
import json
from functools import partial
from typing import Any, Optional

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loops.fetchers import _DataFetcherWrapper
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics import MetricStringToTorchMetric
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset, GPTSFTPackedDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.modules.common.text_generation_utils import generate, get_computeprob_response
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import (
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
        reconfigure_num_microbatches_calculator,
    )

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator as reconfigure_num_microbatches_calculator,
    )
    from apex.transformer.pipeline_parallel.utils import (
        get_current_global_batch_size,
        get_micro_batch_size,
        get_num_microbatches,
    )


__all__ = ['MegatronGPTSFTModel']


class MegatronGPTSFTModel(NLPAdapterModelMixin, MegatronGPTModel):
    """
    Megatron GPT Supervised Fine-Tuning
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.sep_id = cfg.get('sep_id', 49704)
        if hasattr(self.cfg.data, "validation_ds"):
            self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
            self.val_metric = torch.nn.ModuleList(self.val_metric) if self.val_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.validation_ds, "metric"):
                self.val_metric_label_key = self.cfg.data.validation_ds.metric.get('label_key', 'labels')

        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric) if self.test_metric is not None else None
            # Used other keys from metadata to calulate metrics
            if hasattr(self.cfg.data.test_ds, "metric"):
                self.test_metric_label_key = self.cfg.data.test_ds.metric.get('label_key', 'labels')

        # Set the profile start and end steps in the unit of global batach
        if hasattr(self, '_nsys_profile_enabled'):
            self._nsys_profile_start_step = self.cfg.nsys_profile.get('start_step', 0)
            self._nsys_profile_end_step = self.cfg.nsys_profile.get('end_step', 0)
        if hasattr(self, '_memory_profile_enabled'):
            self._memory_profile_start_step = self.cfg.memory_profile.get('start_step', 0)
            self._memory_profile_end_step = self.cfg.memory_profile.get('end_step', 0)

        self.virtual_tokens = 0
        self.init_global_step = 0
        self.enforce_divisible_batch = True  # used for gradient accumulation

    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metric"):
            metric = MetricStringToTorchMetric["exact_string_match"]
        else:
            if not hasattr(data_cfg.metric, "name"):
                raise ValueError("Metric name is not provided in the metric config.")
            if data_cfg.metric.name == "loss":
                return None, "loss"
            if data_cfg.metric.name not in MetricStringToTorchMetric:
                raise KeyError(
                    f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                )
            if data_cfg.metric.name in self._metrics_require_string2category_map:
                if data_cfg.metric.average is None:
                    raise ValueError(
                        f"{data_cfg.metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                    )
            if (
                data_cfg.metric.get('labels_are_strings', False)
                and data_cfg.metric.name in self._metrics_require_string2category_map
            ):
                if data_cfg.metric.num_classes is None:
                    raise ValueError(
                        "Number of classes is not provided in the metric section within the data config. "
                        f"Please provide the number of classes in the data config to use the {data_cfg.metric.name} metric."
                    )
                if data_cfg.metric.get('class_labels', None) is None or not isinstance(
                    data_cfg.metric.get('class_labels', None), ListConfig
                ):
                    raise ValueError(
                        "Class labels are not provided properly in the metric section witnin the data config. "
                        f"Please provide the class labels as a list of strings in the data config to use the {data_cfg.metric.name} metric."
                    )
                if len(data_cfg.metric.get('class_labels', None)) != data_cfg.metric.num_classes:
                    raise ValueError(
                        f"Number of class labels {len(data_cfg.metric.get('class_labels', None))} does not match `num_classes` : {data_cfg.metric.num_classes}"
                    )

            metric_name = data_cfg.metric.name
            metric = MetricStringToTorchMetric[metric_name]

            if isinstance(data_cfg.file_names, ListConfig):
                if 'rouge' not in data_cfg.metric.name:
                    metric = [
                        metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
                        for _ in range(len(data_cfg.file_names))
                    ]
                else:
                    metric = [metric() for _ in range(len(data_cfg.file_names))]
            else:
                if 'rouge' not in data_cfg.metric.name:
                    metric = [metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)]
                else:
                    metric = [metric()]

        return metric, metric_name

    @property
    def _metrics_require_string2category_map(self):
        return set(["f1", "accuracy", "average_precision"])

    def maybe_setup_test(self):
        if hasattr(self.cfg.data, 'test_ds') and self.cfg.data.test_ds.get('file_names', None) is not None:
            self._test_dl = self.setup_eval_dataloader(self._test_ds, self.cfg.data.test_ds)
        return

    def setup(self, stage=None):
        # NOTE: super().__init__ will try and setup train/val/test datasets, but we sidestep this using a if self._train_ds is not None condition
        # We then set things up for real only once setup() of this class is called.
        resume_checkpoint_path = self.trainer.ckpt_path
        self.setup_complete = True
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        if stage == 'predict':
            return

        # If the user wants to manually override train and validation dataloaders before calling `.fit()`
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets(stage=stage)
        if hasattr(self, '_train_ds'):
            self.setup_training_dataloader()
        if hasattr(self, '_validation_ds'):
            self._validation_dl = self.setup_eval_dataloader(self._validation_ds, self.cfg.data.validation_ds)
        self.maybe_setup_test()

        # when using pipeline model parallel the final stage need to initialize word embeddings
        self.initialize_last_rank_embeddings()

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_gpt', False):
            self.setup_transformer_engine_tp_groups()
            self.setup_transformer_engine_cp_groups()
        self.setup_complete = True

    def _build_dataset(self, data_cfg, is_train=True):
        packed_sequence = data_cfg.get("packed_sequence", False)
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"SFT train/validation datasets must be provided as a list of individual JSONL files.")

        if is_train:
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
            # that is of the format [weight1,file_name1,weight2,file_name2,...]
            if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                        f"Found: {data_cfg.concat_sampling_probabilities}"
                    )
                )

            if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be of the same size as file_names.",
                        f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
                    )
                )

            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * data_cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        # Check dataset max_seq_legnth and max_position_embeddings size
        if (
            self.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
            and data_cfg.max_seq_length > self.cfg.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.max_position_embeddings

        # TE requires that the first input dim is divisible by 8 and the second by 16 for fp8
        # When using sequence parallel, sequence will further be split by TP size
        # When using context parallel, sequence is split by CP size as well
        pad_seq_length_to_mult = (
            8 * self.cfg.get('tensor_model_parallel_size', 1) if self.cfg.get('sequence_parallel', False) else 16
        )
        pad_seq_length_to_mult *= self.cfg.get('context_parallel_size', 1)

        dataset_kwargs = {}
        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            if self.cfg.data.get("chat", False):
                dataset_cls = GPTSFTChatDataset
            elif packed_sequence:
                dataset_cls = GPTSFTPackedDataset
                dataset_kwargs = {'return_cu_seqlen': data_cfg.get("packed_sequence_return_cu_seqlen", True)}
                assert data_cfg.micro_batch_size == 1, "Micro batch size must be 1 if using packed sequence"
            else:
                dataset_cls = GPTSFTDataset

            # TODO(akoumparouli): MCore assumes/requires equal length input sequences.
            if not data_cfg.get('pad_to_max_length', False) and self.cfg.get('expert_model_parallel_size', 1) > 1:
                raise ValueError('Expert parallelism requires pad_to_max_length')

            dataset = dataset_cls(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                pad_seq_length_to_mult=pad_seq_length_to_mult,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                label_key=data_cfg.get('label_key', 'answer'),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'text'),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                prompt_template=data_cfg.get('prompt_template', None),
                ceil_to_power_2=data_cfg.get('ceil_to_power_2', False),
                get_attention_mask_from_fusion=data_cfg.get('get_attention_mask_from_fusion', False),
                global_sample_mapping=data_cfg.get('global_sample_mapping', False),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
                hf_dataset=data_cfg.get(
                    'hf_dataset', False
                ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
                truncation_method=data_cfg.get(
                    'truncation_method', 'right'
                ),  # used to choose truncation method. Options: ['random', 'left', 'right']
                special_tokens=self.cfg.data.get(
                    'chat_prompt_tokens', None
                ),  # special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
                is_test=not is_train,
                **dataset_kwargs,
            )
            datasets.append(dataset)
        if is_train:
            if packed_sequence:
                num_train_samples_after_blend = sum(len(dataset) for dataset in datasets)
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets

    def _determine_log_key(self, data_config, dataloader_idx, metric_name, mode):
        # Function that determines whether to log based on the user provided name of the dataset or the dataloader index.
        base_key = f"{mode}_{metric_name}_" if metric_name is not None else f"{mode}_"
        # If the user provided names for each validation/test dataset, use those.
        if hasattr(data_config, "names") and data_config.names is not None:
            # With only a single validation/test dataset, the name is not a list.
            if not isinstance(data_config.names, ListConfig):
                name = data_config.names
            else:
                name = data_config.names[dataloader_idx]
            return base_key + name
        else:
            return base_key + f"dataloader{dataloader_idx}"

    def fwd_bwd_step(self, dataloader_iter, forward_only, first_val_step=None):
        # Return only batch if batch, batch_idx, dataloder_idx are extracted as a tuple in the previous func
        # call like validation_step otherwise return tuple (in which case dataloader_iter is still a PTL _DataFetcherWrapper object)
        if isinstance(dataloader_iter, _DataFetcherWrapper):
            batch, _, _ = next(dataloader_iter)
        else:
            batch = next(dataloader_iter)

        log_token_counts = self.cfg.get('log_token_counts', False)
        if log_token_counts:
            token_count_avg = sum(batch['token_count']) / len(batch['token_count'])

        # Pass only torch.Tensor to prevent errors when process get_iterator_k_split()
        batch = {k: v for k, v in batch.items() if isinstance(v, (torch.Tensor, list))}
        _, seq_length = batch['tokens'].shape
        data_iter = get_iterator_k_split(batch, get_num_microbatches(), self.enforce_divisible_batch)

        if log_token_counts:
            self.log('seq_length_padded', seq_length, prog_bar=True, batch_size=1)
            self.log('tokens_avg', token_count_avg, prog_bar=True, sync_dist=True, batch_size=1)

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

        for module in self.get_model_module_list():
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func
            module.config.param_sync_func = param_sync_func

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(tuning=True, validation_step=forward_only),
            data_iterator=self._make_data_iterator_list(data_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
            first_val_step=first_val_step,
        )

        non_loss_tensors = {}
        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            for item in losses_reduced_per_micro_batch:
                for k, v in item.items():
                    if k != 'avg':
                        av = non_loss_tensors.get(k, [])
                        av.append(v)
                        non_loss_tensors[k] = av
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

        # if forward_only:
        # return loss_mean
        if non_loss_tensors:  # TODO: need a nicer way to do this via inheritance (@adithyare)
            return loss_mean, non_loss_tensors
        else:
            return loss_mean

    def validation_step(self, dataloader_iter):
        return self.inference_step(dataloader_iter, 'validation')

    def test_step(self, dataloader_iter):
        return self.inference_step(dataloader_iter, 'test')

    def inference_step(self, dataloader_iter, mode):
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds
        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        outputs = self.inference_step_validation_call(batch, batch_idx, data_cfg, dataloader_idx)

        if mode == 'validation':
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[dataloader_idx][-1] = outputs
            else:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[-1] = outputs
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx][-1] = outputs
            else:
                self.test_step_outputs[-1] = outputs
        return outputs

    def inference_step_validation_call(self, batch, batch_idx, data_cfg, dataloader_idx=0):
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        # Pass dataloader_idx, as it's needed in val_step of GPTModel to append the loss correctly to self.val/test_step_outputs
        # in case of multi dataloaders
        loss = super().validation_step(itertools.chain([batch]), dataloader_idx)

        if data_cfg.get("write_predictions_to_file", False) or data_cfg.metric.name != 'loss':
            # We need _inference_config to get generation params
            # add_BOS and tokens_to_generate are set in dataset
            if self.get_inference_config() is None:
                self.set_inference_config(inference_config={})
            self._inference_config['add_BOS'] = data_cfg.add_bos
            self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

            output = self.predict_step(batch, batch_idx, dataloader_idx)
            if output:
                inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['contexts']]
                labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['answers']]
                preds_text = [
                    self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
                    for t, l in zip(output['token_ids'], batch['context_lengths'])
                ]
            else:
                inputs_text, labels_text, preds_text = [], [], []
        else:
            inputs_text, labels_text, preds_text = [], [], []

        outputs = {
            'loss': loss,
            'preds': preds_text,  # [str]
            'labels': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
        }
        return outputs

    def gather_and_maybe_write_predictions(self, output, data_cfg, mode, averaged_metric, dataloader_idx=0):
        # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
        gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            gathered_outputs,
            [
                {'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'], 'metadata': x['metadata']}
                for x in output
            ],
            group=parallel_state.get_data_parallel_group(),
        )

        # Remove duplicate examples due to distributed sampler.
        deduplicated_outputs = {
            'preds': [],
            'labels': [],
            'inputs': [],
            'metadata': [],
        }
        total_size = 0
        for rank in range(0, parallel_state.get_data_parallel_world_size()):
            for batch in gathered_outputs[rank]:
                for pred, label, input, metadata in zip(
                    batch['preds'], batch['labels'], batch['inputs'], batch['metadata']
                ):
                    total_size += 1
                    if not metadata.get("__AUTOGENERATED__", False):
                        deduplicated_outputs['preds'].append(pred)
                        deduplicated_outputs['labels'].append(label)
                        deduplicated_outputs['inputs'].append(input)
                        deduplicated_outputs['metadata'].append(metadata)
                    else:
                        logging.info(f"skipping autogenerated example example {input} prediction {pred} label {label}")

        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        metric_label_key = self.val_metric_label_key if mode == 'validation' else self.test_metric_label_key
        if metric_name != 'loss':
            metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
            metric_fn = self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
            if metric_label_key in deduplicated_outputs['metadata'][0]:
                labels = [m[metric_label_key] for m in deduplicated_outputs['metadata']]
            else:
                labels = deduplicated_outputs['labels']

            for pred, label in zip(deduplicated_outputs['preds'], labels):
                _ = metric_fn(pred, label)

            metric_result = metric_fn.compute()

            if metric_name == 'rouge':
                for k, v in metric_result.items():
                    if 'fmeasure' in k:
                        self.log(metric_log_key + f'_{k}', v.item(), sync_dist=True)
                        logging.info(f"{mode} {metric_name} {k}: {v.item()}")
                metric_result = metric_result['rouge1_fmeasure']
            else:
                self.log(metric_log_key, metric_result.item(), sync_dist=True)
                logging.info(f"{mode} {metric_name}: {metric_result.item()}")

            metric_fn.reset()
            averaged_metric.append(metric_result)

        # Write predictions to file
        if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
            logging.info(
                f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
            )

            # Check if the user provided a prefix path to the file(s) they want to write.
            if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                raise ValueError(
                    f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                )
            filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
            self.write_predictions_to_file(
                deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}"
            )

        return deduplicated_outputs, total_size

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # TODO: this method should be modularized. It is too long and does too many things. (@adithyare)
        # Parent class will handle logging of the loss.
        if not outputs or not outputs[0]:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            # loss = super().on_validation_epoch_end([x['loss'] for x in output])
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self.cfg.data.get('validation_drop_last', True):
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)
            self.gather_and_maybe_write_predictions(output, data_cfg, mode, averaged_metric, dataloader_idx)

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) >= 1 else None

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("validation_loss", averaged_loss, batch_size=1)
            if averaged_metric is not None:
                self.log(f"validation_{self.val_metric_name}", averaged_metric)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        self._restore_activation_checkpointing_args()
        if hasattr(self, "_train_ds"):
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return averaged_loss, averaged_metric

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        # need to overwrite some configuration, make it immutable
        inference_config = inference_config.copy()
        global_batch_size_per_gpu = batch['tokens'].size(0)
        num_micro_batches_before_decode = get_num_microbatches()

        compute_logprob = inference_config.get('compute_logprob', False)
        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config["add_BOS"] = False
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            response = get_computeprob_response(self.tokenizer, response, batch)
        else:
            # for megatron_gpt_eval.py
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            else:
                # peft_eval.py
                inference_config['inputs'] = (batch['contexts'].cuda(), batch['context_lengths'].cuda())
            response = generate(self, **inference_config)

        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_size_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        return response

    def write_predictions_to_file(self, outputs, output_file_path_prefix):
        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m in zip(outputs['inputs'], outputs['preds'], outputs['labels'], outputs['metadata']):
                json_string = {'input': i, 'pred': p, 'label': l}
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    def cast_for_metric(self, pred, label, metric_name, class_labels=None, labels_are_strings=False):
        if metric_name == 'exact_string_match' or 'rouge' in metric_name:
            return pred, label
        pred = pred.replace(' ', '')
        label = label.replace(' ', '')

        # Correlation metrics require casting to float.
        if metric_name in ['pearson_corr_coef', 'spearman_corr_coef']:
            # Text-to-text model predictions may not always be valid floating point numbers.
            try:
                pred = float(pred)
            except ValueError:
                pred = 0.0

            try:
                label = float(label)
            except ValueError:
                raise ValueError(f'Could not convert {label} to float.')

            pred = torch.FloatTensor([pred]).to(self.device)
            label = torch.FloatTensor([label]).to(self.device)

        # Other metrics require casting to integers.
        elif metric_name in self._metrics_require_string2category_map and not labels_are_strings:
            # Text-to-text model predictions may not always be valid integers.
            try:
                pred = int(pred)
            except ValueError:
                pred = 0

            try:
                label = int(label)
            except ValueError:
                raise ValueError(f'Could not convert {label} to int.')

            pred = torch.LongTensor([pred]).to(self.device)
            label = torch.LongTensor([label]).to(self.device)

        # If labels are strings, we need to convert them to indices for some metrics.
        elif metric_name in self._metrics_require_string2category_map and labels_are_strings:
            # Cast string labels to integers before computing the metric.
            if pred not in class_labels:
                pred = 0  # If the prediction is not in the class labels, use the first class label.
            else:
                pred = class_labels.index(pred)
            if label not in class_labels:
                raise ValueError(f"Ground truth labe; {label} is not in the class labels list : {class_labels}")
            label = class_labels.index(label)
            pred = torch.LongTensor([pred]).to(self.device)
            label = torch.LongTensor([label]).to(self.device)
        else:
            raise ValueError(f'Metric {metric_name} not supported.')

        return pred, label

    # Override the parent batch reconfiguring logic.
    def _reconfigure_and_process_inference_batch(self, batch, data_cfg):
        global_batch_size_per_gpu = batch['tokens'].size(0)
        # This should happen only on the last batch of the dataset.
        if (
            global_batch_size_per_gpu
            != get_current_global_batch_size() // parallel_state.get_data_parallel_world_size()
        ):
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            if (
                global_batch_size_per_gpu
                != data_cfg.global_batch_size // parallel_state.get_data_parallel_world_size()
            ):
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                    micro_batch_size=global_batch_size_per_gpu,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            # NOTE: need to explicitly handle resetting for multi-validation
            else:
                app_state = AppState()
                reconfigure_num_microbatches_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=data_cfg.global_batch_size,
                    micro_batch_size=data_cfg.micro_batch_size,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )

    def maybe_build_test(self):
        if hasattr(self.cfg.data, 'test_ds') and self.cfg.data.test_ds.get('file_names', None) is not None:
            logging.info('Building GPT SFT test datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._test_ds = self._build_dataset(self.cfg.data.test_ds, is_train=False)
            logging.info(f'Length of test dataset: {len(self._test_ds[0])}')
        return

    def build_train_valid_test_datasets(self, stage):
        if stage != 'test':
            logging.info('Building GPT SFT validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(self.cfg.data.validation_ds, is_train=False)
            if self._validation_ds:
                logging.info(f'Length of val dataset: {len(self._validation_ds[0])}')

        if stage != 'validate':
            self.maybe_build_test()

        if stage == 'validate' or stage == 'test':
            return
        logging.info('Building GPT SFT traing datasets.')
        self._train_ds = self._build_dataset(self.cfg.data.train_ds)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')

    def build_data_loader(self, dataset, data_cfg, consumed_samples=0):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=data_cfg.micro_batch_size,
            global_batch_size=data_cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=data_cfg.drop_last,
            pad_samples_to_global_batch_size=not data_cfg.drop_last,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=True if data_cfg.num_workers > 0 else False,
        )

    def setup_training_dataloader(self):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_data_loader(
                dataset=self._train_ds,
                data_cfg=self.cfg.data.train_ds,
                consumed_samples=consumed_samples,
            )

    def setup_eval_dataloader(self, datasets, data_cfg):
        dataloaders = []
        for dataset in datasets:
            eval_dl = self.build_data_loader(
                dataset=dataset,
                data_cfg=data_cfg,
                consumed_samples=0,
            )
            dataloaders.append(eval_dl)
        return dataloaders

    def on_validation_epoch_start(self):
        self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        self._reset_activation_checkpointing_args()
        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_predict_epoch_start(self):
        return self.on_test_epoch_start()

    def on_test_epoch_end(self):
        _ = self.inference_epoch_end(self.test_step_outputs, 'test', self.cfg.data.test_ds)
        # Commenting as on_test_epoch_end was a no-op in PTL 1.9
        # return super().on_test_epoch_end()

    def on_validation_epoch_end(self):
        _ = self.inference_epoch_end(self.validation_step_outputs, 'validation', self.cfg.data.validation_ds)
        # Commenting as on_validation_epoch_end was a no-op in PTL 1.9
        # return super().on_validation_epoch_end()

    def on_train_epoch_start(self) -> None:
        # Same logic as validation epoch end, but this may be need if there is no validation sanity check to trigger on_validation_epoch_end()
        self.on_validation_epoch_end()
        return super().on_train_epoch_start()
