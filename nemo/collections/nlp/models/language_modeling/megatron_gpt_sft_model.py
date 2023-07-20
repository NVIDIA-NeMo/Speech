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

import json
from functools import partial
from typing import Any, Optional

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics import MetricStringToTorchMetric
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split
from nemo.collections.nlp.modules.common.text_generation_utils import (
    LengthParam,
    SamplingParam,
    generate,
    get_computeprob_response,
    megatron_gpt_generate,
)
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = ['MegatronGPTSFTModel']


class MegatronGPTSFTModel(MegatronGPTModel):
    """
    Megatron GPT Supervised Fine-Tuning
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg, trainer=trainer)
        self.sep_id = cfg.get('sep_id', 49704)
        self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
        self.val_metric = torch.nn.ModuleList(self.val_metric) if self.val_metric is not None else None
        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric) if self.test_metric is not None else None

        if self.cfg.get('megatron_amp_O2', False):
            base_module = self.model.module
        else:
            base_module = self.model

        self.original_checkpointing_granularity = base_module.language_model.encoder.activations_checkpoint_granularity
        self.original_checkpointing_num_layers = base_module.language_model.encoder.activations_checkpoint_num_layers
        self.original_checkpointing_method = base_module.language_model.encoder.activations_checkpoint_method
        self.virtual_tokens = 0

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

    def setup(self, stage=None):
        # NOTE: super().__init__ will try and setup train/val/test datasets, but we sidestep this using a if self._train_ds is not None condition
        # We then set things up for real only once setup() of this class is called.
        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
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
        if hasattr(self.cfg.data, 'test_ds'):
            self._test_dl = self.setup_eval_dataloader(self._test_ds, self.cfg.data.test_ds)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    module.sync_initial_word_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                self.model.sync_initial_word_embeddings()

        if self.cfg.get('transformer_engine', False):
            self.setup_transformer_engine_tp_groups()

    def _build_dataset(self, data_cfg, is_train=True):
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

        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            if self.cfg.data.get("chat", False):
                dataset_cls = GPTSFTChatDataset
            else:
                dataset_cls = GPTSFTDataset
            dataset = dataset_cls(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                context_key=data_cfg.get('context_key', 'text'),
                label_key=data_cfg.get('label_key', 'answer'),
                separate_prompt_and_response_with_newline=data_cfg.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'context'),
                pad_to_max_length=False,
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
            )
            datasets.append(dataset)

        if is_train:
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

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        batch = next(dataloader_iter)
        _, seq_length = batch['tokens'].shape
        tensor_shape = [seq_length, get_micro_batch_size(), self.cfg.hidden_size]
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(self._optimizer.no_sync, greedy_grad_copy=self.megatron_amp_o2,)
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler.scale if self.cfg.precision == 16 else None,
            sequence_parallel=self.cfg.get('sequence_parallel', False),
            enable_autocast=self.enable_autocast,
            no_sync_func=no_sync_func,
            grad_sync_func=grad_sync_func,
            param_sync_func=param_sync_func,
            overlap_p2p_comm=self.cfg.get('overlap_p2p_comm', False),
            batch_p2p_comm=self.cfg.get('batch_p2p_comm', True),
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

    def validation_step(self, dataloader_iter, batch_idx, dataloader_idx=0):
        return self.inference_step(dataloader_iter, batch_idx, 'validation', dataloader_idx)

    def validation_epoch_end(self, outputs):
        _ = self.inference_epoch_end(outputs, 'validation', self.cfg.data.validation_ds)

    def test_step(self, dataloader_iter, batch_idx, dataloader_idx=0):
        return self.inference_step(dataloader_iter, batch_idx, 'test', dataloader_idx)

    def test_epoch_end(self, outputs):
        _ = self.inference_epoch_end(outputs, 'test', self.cfg.data.test_ds)

    def inference_step(self, dataloader_iter, batch_idx, mode, dataloader_idx=0):
        # Call parent validation step to get the loss.
        loss = super().validation_step(dataloader_iter, batch_idx)
        return {
            'loss': loss,
            'preds': None,
            'labels': None,
            'inputs': None,
        }
        # TODO (sandeepsub): Figure out the subsequent decode bits.
        length_params: LengthParam = {
            "min_length": 0,
            "max_length": batch['tokens'].size(1) - batch['context_lengths'].max(),
        }
        sampling_params: SamplingParam = {
            "use_greedy": True,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.94,
            "repetition_penalty": 1.2,
            "add_BOS": False,
            "all_probs": False,
            "compute_logprob": False,
            "end_strings": ["<|endoftext|>"],
        }
        result = megatron_gpt_generate(
            model=self,
            inputs=(
                batch['tokens'].cuda(),
                (batch['context_lengths'] - 1).cuda(),
            ),  # NOTE: We do -1 here to remove the space between context and response.
            tokenizer=self.tokenizer,
            sampling_params=sampling_params,
            length_params=length_params,
            check_sequence_parallel_and_checkpointing=False,  # We need to skip these checks since we'll manually enbale and disable checkpointing between training and validation.
        )

        preds_text = []
        labels_text = []
        input_text = []
        for idx, item in enumerate(result['token_ids']):
            pred = self.tokenizer.ids_to_text(item[batch['context_lengths'][idx] - 1 :])
            input = self.tokenizer.ids_to_text(item[: batch['context_lengths'][idx] - 1])
            label = self.tokenizer.ids_to_text(batch['tokens'][idx][batch['context_lengths'][idx] :].tolist())
            preds_text.append(pred.strip())
            labels_text.append(label.strip())
            input_text.append(input.strip())

        metric = self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
        assert len(preds_text) == len(labels_text) == len(input_text)
        for _, (pred, label) in enumerate(zip(preds_text, labels_text)):
            # To compute metrics like pearson or spearman correlation, we need to cast the predicted string and labels to floats.
            pred, label = self.cast_for_metric(
                pred=pred.strip(),
                label=label.strip(),
                metric_name=self.val_metric_name if mode == 'validation' else self.test_metric_name,
                class_labels=self.cfg.data.validation_ds.metric.get('class_labels', None)
                if mode == 'validation'
                else self.cfg.data.test_ds.metric.get('class_labels', None),
                labels_are_strings=self.cfg.data.validation_ds.metric.get('labels_are_strings', False)
                if mode == 'validation'
                else self.cfg.data.test_ds.metric.get('labels_are_strings', False),
            )
            _ = metric(pred, label)

        return {
            'loss': loss,
            'preds': preds_text,
            'labels': labels_text,
            'inputs': input_text,
        }

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            loss = super().validation_epoch_end([x['loss'] for x in output])
            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss)
            averaged_loss.append(loss)

            # Skip the rest of this loop if the user wants to monitor the loss only.
            if self.val_metric is None:
                continue
            # Determine the key used to log the eval metric based on the user provided name of the dataset or the dataloader index.
            metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
            metric_object = (
                self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
            )
            metric = metric_object.compute()
            # Handle logging of GLUE/XNLI separately here. XNLI has a separate metric per language.
            if isinstance(metric, dict):
                if metric_name == 'rouge':
                    metric = metric['rougeL_fmeasure']
                else:
                    metric = metric['acc']
            torch.distributed.all_reduce(
                metric, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_data_parallel_group()
            )
            metric = metric / parallel_state.get_data_parallel_world_size()
            self.log(metric_log_key, metric)
            logging.info(f"{mode} {metric_name}: {metric}")

            metric_object.reset()

            averaged_metric.append(metric)

            # Write predictions, labels, and inputs to a file for each validation/test dataset.
            if data_cfg.get("write_predictions_to_file", False):

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )

                # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
                gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
                torch.distributed.all_gather_object(
                    gathered_outputs,
                    [{'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'],} for x in output],
                    group=parallel_state.get_data_parallel_group(),
                )

                # Figure out what the suffix of the file should be.
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)

                # Keep a set of ground truths and inputs to write deduplicated predictions. Distributed Sampler may duplicate examples.
                gt_inp_set = set()
                deduplicated_outputs = {
                    'preds': [],
                    'labels': [],
                    'inputs': [],
                }

                # PTL models have a self.global_rank attribute and we want to write to disk only on global rank 0.
                if self.global_rank == 0:
                    for rank in range(0, parallel_state.get_data_parallel_world_size()):
                        for batch in gathered_outputs[rank]:
                            for pred, label, input in zip(batch['preds'], batch['labels'], batch['inputs']):
                                gt_inp_set.add(input + label)
                                deduplicated_outputs['preds'].append(pred)
                                deduplicated_outputs['labels'].append(label)
                                deduplicated_outputs['inputs'].append(input)
                    self.write_predictions_to_file(
                        deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}"
                    )
                torch.distributed.barrier()

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) > 1 else None

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("validation_loss", averaged_loss)
            if averaged_metric is not None:
                self.log(f"validation_{self.val_metric_name}", averaged_metric)
        elif mode == 'test':
            self.log("test_loss", averaged_loss)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric)

        return averaged_loss, averaged_metric

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        # need to overwrite some configuration, make it immutable
        inference_config = inference_config.copy()
        compute_logprob = inference_config['compute_logprob']
        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config["add_BOS"] = False
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
            return compute_prob_response
        else:
            # for megatron_gpt_eval.py
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            else:
                # peft_eval.py
                inference_config['inputs'] = (batch['contexts'].cuda(), batch['context_lengths'].cuda())
            return generate(self, **inference_config)

    def write_predictions_to_file(self, outputs, output_file_path_prefix):
        with open(output_file_path_prefix + "_inputs_preds_labels.jsonl", "w") as f_json:
            assert len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels'])
            for i, p, l in zip(outputs['inputs'], outputs['preds'], outputs['labels']):
                f_json.write(json.dumps({'input': i, 'pred': p, 'label': l}) + '\n')

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
    def _reconfigure_and_process_inference_batch(self, batch):
        global_batch_per_gpu = batch['tokens'].size(0)
        # This should happen only on the last batch of the validation/test dataset with drop_last=False.
        if global_batch_per_gpu != self.cfg.data.validation_ds.global_batch_size:
            app_state = AppState()
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_per_gpu,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        return batch

    def build_train_valid_test_datasets(self, stage):
        if stage != 'test':
            logging.info('Building GPT SFT validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(self.cfg.data.validation_ds, is_train=False)
            logging.info(f'Length of val dataset: {len(self._validation_ds[0])}')

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                logging.info('Building GPT SFT test datasets.')
                # Wrap this in a list since the general finetuning parent class supports multi-validation.
                self._test_ds = self._build_dataset(self.cfg.data.test_ds, is_train=False)
                logging.info(f'Length of test dataset: {len(self._test_ds[0])}')

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
            drop_last=True,
            pad_samples_to_global_batch_size=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )

    def setup_training_dataloader(self):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_data_loader(
                dataset=self._train_ds, data_cfg=self.cfg.data.train_ds, consumed_samples=consumed_samples,
            )

    def setup_eval_dataloader(self, datasets, data_cfg):
        dataloaders = []
        for dataset in datasets:
            eval_dl = self.build_data_loader(dataset=dataset, data_cfg=data_cfg, consumed_samples=0,)
            dataloaders.append(eval_dl)
        return dataloaders

    def _reset_activation_checkpointing_args(self):
        if self.cfg.get('megatron_amp_O2', False):
            base_module = self.model.module
        else:
            base_module = self.model

        base_module.language_model.encoder.activations_checkpoint_granularity = None
        base_module.language_model.encoder.activations_checkpoint_method = None
        base_module.language_model.encoder.activations_checkpoint_num_layers = None

    def _restore_activation_checkpointing_args(self):
        if self.cfg.get('megatron_amp_O2', False):
            base_module = self.model.module
        else:
            base_module = self.model
        base_module.language_model.encoder.activations_checkpoint_granularity = self.original_checkpointing_granularity
        base_module.language_model.encoder.activations_checkpoint_method = self.original_checkpointing_method
        base_module.language_model.encoder.activations_checkpoint_num_layers = self.original_checkpointing_num_layers

    def on_validation_epoch_start(self):
        self._reset_activation_checkpointing_args()
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        app_state = AppState()
        self._reset_activation_checkpointing_args()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        self.on_inference_epoch_end(self.cfg.data.test_ds)
        return super().on_test_epoch_end()

    def on_validation_epoch_end(self):
        self.on_inference_epoch_end(self.cfg.data.validation_ds)
        return super().on_validation_epoch_end()

    def on_inference_epoch_end(self, ds):
        app_state = AppState()
        self._restore_activation_checkpointing_args()
        if hasattr(self, "_train_ds"):
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=ds.global_batch_size,
                micro_batch_size=ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

    def on_train_epoch_start(self) -> None:
        # Same logic as validation epoch end, but this may be need if there is no validation sanity check to trigger validation_epoch_end()
        self.on_validation_epoch_end()
        return super().on_train_epoch_start()
