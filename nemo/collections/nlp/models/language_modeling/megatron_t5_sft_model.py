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
import json
from typing import Dict, List

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.data import ConcatMapDataset
from nemo.collections.common.metrics import MetricStringToTorchMetric
from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.common.sequence_to_sequence_dataset import SequenceToSequenceDataset
from nemo.collections.nlp.data.language_modeling.megatron.t5_sft_dataset import T5SFTDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model, T5Sentinel
from nemo.collections.nlp.modules.common.megatron.utils import get_iterator_k_split

from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_current_global_batch_size,
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

__all__ = ['MegatronT5SFTModel']


class MegatronT5SFTModel(NLPAdapterModelMixin, MegatronT5Model):
    """ T5 Finetuning model in the same format as MegatronGPTSFTModel """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg, trainer=trainer)
        self.val_metric = self.test_metric = None
        if hasattr(self.cfg.data, "validation_ds"):
            self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
            self.val_metric = torch.nn.ModuleList(self.val_metric) if self.val_metric is not None else None
        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric) if self.test_metric is not None else None

    def setup_metric(self, data_cfg):
        # XNLI is a special case.
        metric_name = "exact_string_match"
        if hasattr(self.cfg, "eval_languages"):
            metric = [ExactStringPerCategoryMatchMetric(self.cfg.eval_languages)]
        else:
            if not hasattr(data_cfg, "metric"):
                metric_class = MetricStringToTorchMetric["exact_string_match"]
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
                metric_class = MetricStringToTorchMetric[metric_name]

            # GLUE will not have a "src_file_name" attribute and will always have only a single metric.
            if hasattr(data_cfg, "src_file_name") or hasattr(data_cfg, "file_names"):
                if (
                    hasattr(data_cfg, "src_file_name")
                    and isinstance(data_cfg.src_file_name, ListConfig)
                    and metric_name != 'rouge'
                ):
                    metric = [
                        metric_class(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
                        for _ in range(len(data_cfg.src_file_name))
                    ]
                elif (
                    hasattr(data_cfg, "file_names")
                    and isinstance(data_cfg.file_names, ListConfig)
                    and metric_name != 'rouge'
                ):
                    metric = [
                        metric_class(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
                        for _ in range(len(data_cfg.file_names))
                    ]
                elif hasattr(data_cfg, "src_file_name") and isinstance(data_cfg.src_file_name, ListConfig):
                    metric = [metric_class() for _ in range(len(data_cfg.src_file_name))]
                elif hasattr(data_cfg, "file_names") and isinstance(data_cfg.file_names, ListConfig):
                    metric = [metric_class() for _ in range(len(data_cfg.file_names))]
                else:
                    metric = [metric_class(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)]
            else:
                metric = [metric_class()]  # GLUE does need to specify average or num_classes.

        return metric, metric_name

    @property
    def _metrics_require_string2category_map(self):
        return set(["f1", "accuracy", "average_precision"])

    @property
    def model(self):
        return self.enc_dec_model

    def setup(self, stage=None):
        # This is just to keep the parent class happy since we override its setup() method.
        self.init_consumed_samples = 0
        self.init_global_step = 0
        if stage == 'predict':
            return

        # NOTE: PTL uses the same stage string "test" for both testing and validation.
        self.build_train_valid_test_datasets(stage=stage)
        if hasattr(self, '_validation_ds'):
            self.setup_validation_data()
        if hasattr(self, '_test_ds'):
            self.setup_test_data()
        if hasattr(self, '_train_ds'):
            self.setup_training_data()
        self.setup_complete = True

    def on_validation_epoch_start(self):
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
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_train_epoch_start(self) -> None:
        # Same logic as validation epoch end, but this may be need if there is no validation sanity check to trigger validation_epoch_end()
        # Commenting as on_validation_epoch_end was a no-op in PTL 1.9
        # self.on_validation_epoch_end()
        return super().on_train_epoch_start()

    def cast_for_metric(self, pred, label, metric_name, class_labels=None, labels_are_strings=False):
        if metric_name == 'exact_string_match' or 'rouge':
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
                raise ValueError(f"Ground truth label {label} is not in the class labels list : {class_labels}")
            label = class_labels.index(label)
            pred = torch.LongTensor([pred]).to(self.device)
            label = torch.LongTensor([label]).to(self.device)
        else:
            raise ValueError(f'Metric {metric_name} not supported.')

        return pred, label

    def _reconfigure_and_process_inference_batch(self, batch, ds_config):
        global_batch_size_per_gpu = batch['text_enc'].size(0)
        # This should happen only on the last batch of the dataset.
        if (
            global_batch_size_per_gpu
            != get_current_global_batch_size() // parallel_state.get_data_parallel_world_size()
        ):
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            if (
                global_batch_size_per_gpu
                != ds_config.global_batch_size // parallel_state.get_data_parallel_world_size()
            ):
                app_state = AppState()
                _reconfigure_microbatch_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                    micro_batch_size=global_batch_size_per_gpu,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            # NOTE: need to explicitly handle resetting for multi-validation
            else:
                app_state = AppState()
                _reconfigure_microbatch_calculator(
                    rank=app_state.global_rank,
                    rampup_batch_size=None,
                    global_batch_size=ds_config.global_batch_size,
                    micro_batch_size=ds_config.micro_batch_size,
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        batch = next(dataloader_iter)
        if isinstance(batch, dict):
            # convert to list if not already converted.
            batch = self._process_batch(batch)

        # Get seq length of batch
        encoder_seq_length = batch[0].size(1)
        decoder_seq_length = batch[1].size(1)

        tensor_shape = [encoder_seq_length, get_micro_batch_size(), self.cfg.encoder.hidden_size]
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        return self._execute_fwd_bwd_function(
            data_iterator=data_iter,
            forward_only=forward_only,
            tensor_shape=tensor_shape,
            decoder_seq_length=decoder_seq_length,
        )

    def inference_step(self, dataloader_iter, batch_idx: int, mode: str, dataloader_idx=0):
        # Check if iterator is exhausted
        dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        if done:
            return
        # Regular finetuning datasets will return a list of dicts for each microbatch.
        # But T0 datasets will return a single dict for the global batch.
        batch = next(dataloader_iter)
        batch_has_lang_information = isinstance(batch, list) and len(batch[0]) == 7
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds

        self._reconfigure_and_process_inference_batch(batch, data_cfg)

        # NOTE: There could be extra keys in the processed_batch dictionary such as "langs" for XNLI,
        # this will be ignored.
        loss = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=True)

        predicted_token_ids, _ = self.decode(
            tokens_enc=batch['text_enc'],
            enc_mask=batch['enc_mask'],
            num_tokens_to_generate=30,
            bos_id=self.tokenizer.pad_id if data_cfg.get('replace_bos_with_pad', False) else self.tokenizer.bos_id,
        )

        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        labels_text = MegatronT5SFTModel.ids_to_text(batch['labels'], self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(batch['text_enc'], self.tokenizer)

        if not batch_has_lang_information:
            categories = [None] * len(preds_text)
        else:
            categories = batch['lang']

        if self.val_metric is not None or self.test_metric is not None:
            metric = self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
            assert len(categories) == len(preds_text) == len(labels_text)
            for _, (pred, label, category) in enumerate(zip(preds_text, labels_text, categories)):
                # To compute metrics like pearson or spearman correlation, we need to cast the predicted string and labels to floats.
                pred, label = self.cast_for_metric(
                    pred=pred,
                    label=label,
                    metric_name=self.val_metric_name if mode == 'validation' else self.test_metric_name,
                    class_labels=data_cfg.metric.get('class_labels', None),
                    labels_are_strings=data_cfg.metric.get('labels_are_strings', False),
                )
                if batch_has_lang_information:
                    _ = metric(pred, label, category)
                else:
                    _ = metric(pred, label)

        outputs = {
            'preds': preds_text,
            'labels': labels_text,
            'categories': categories,
            'inputs': input_text,
        }

        if isinstance(loss, dict):
            outputs.update(loss)
        else:
            outputs['loss'] = loss
        if mode == 'validation':
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                self.validation_step_outputs[dataloader_idx].append(outputs)
            else:
                self.validation_step_outputs.append(outputs)
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(outputs)
            else:
                self.test_step_outputs.append(outputs)
        return outputs

    @classmethod
    def ids_to_text(cls, batch_ids, tokenizer):
        batch_ids = batch_ids.cpu().numpy().tolist()
        texts = []
        for ids in batch_ids:
            if tokenizer.eos_id in ids:
                idx = ids.index(tokenizer.eos_id)
                ids = ids[:idx]

            if (
                len(tokenizer.text_to_ids(T5Sentinel.END.value)) == 1
                and tokenizer.text_to_ids(T5Sentinel.END.value)[0] in ids
            ):
                idx = ids.index(tokenizer.text_to_ids(T5Sentinel.END.value)[0])
                ids = ids[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(tokenizer, 'special_token_to_id'):
                ids = [id for id in ids if id not in tokenizer.special_token_to_id.values()]
            text = tokenizer.ids_to_text(ids)
            texts.append(text)

        return texts

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

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            # Expand on_validation_epoch_end from parent class MegatronLMEncoderDecoderModel as it doesnt take arg outputs
            # loss = super().validation_epoch_end([x['loss'] for x in output])
            loss_vals = [x['loss'] for x in output]
            # NOTE: we need to make sure outputs is not empty (this is a workaround for a bug in pytorch lightning (?))
            if len(loss_vals) == 0:
                logging.warning("validation_epoch_end: outputs is empty")
                return
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                loss = torch.stack(loss_vals).mean()
            else:
                loss = torch.tensor(0.0).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())
            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1)
            self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
            averaged_loss.append(loss)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            if metric_name != 'loss':
                # Determine the key used to log the eval metric based on the user provided name of the dataset or the dataloader index.
                metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
                self.log(loss_log_key, loss, batch_size=1)
                metric_object = (
                    self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
                )
                metric = metric_object.compute()
                if metric_name == 'rouge':
                    metric = metric['rouge1_fmeasure']
                # Handle logging of GLUE/XNLI separately here. XNLI has a separate metric per language.
                if isinstance(metric, dict):
                    # GLUE case:
                    if len(metric) == 1 and 'acc' in metric:
                        metric = metric['acc']
                        self.log(metric_log_key, metric, batch_size=1)
                        logging.info(f"{mode} {metric_name}: {metric}")
                    # XNLI case where the metric dictionary contains the language and the computed metric as values.
                    else:
                        for k, v in metric.items():
                            if k != 'acc' and 'total' not in k:
                                self.log(metric_log_key + f'_{k}', v, batch_size=1)
                                logging.info(f"{mode} {metric_name} lang {k} : {v}")
                        if metric_name != 'rouge':
                            metric = metric['acc']
                else:
                    self.log(metric_log_key, metric, batch_size=1)
                    logging.info(f"{metric_log_key}: {metric}")
                metric_object.reset()
                averaged_metric.append(metric)

            # Write predictions, labels, and inputs to a file for each validation/test dataset.
            if data_cfg.get("write_predictions_to_file", False):

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )

                # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDPDDP ranks.
                gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
                torch.distributed.all_gather_object(
                    gathered_outputs,
                    [
                        {
                            'preds': x['preds'],
                            'labels': x['labels'],
                            'categories': x['categories'],
                            'inputs': x['inputs'],
                        }
                        for x in output
                    ],
                    group=parallel_state.get_data_parallel_group(),
                )

                # Figure out what the suffix of the file should be.
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)

                # Keep a set of ground truths and inputs to write deduplicated predictions. Distributed Sampler may duplicate examples.
                gt_inp_set = set()
                deduplicated_outputs = {
                    'preds': [],
                    'labels': [],
                    'categories': [],
                    'inputs': [],
                }

                # PTL models have a self.global_rank attribute and we want to write to disk only on global rank 0.
                if self.global_rank == 0:
                    for rank in range(0, parallel_state.get_data_parallel_world_size()):
                        for batch in gathered_outputs[rank]:
                            for pred, label, input, category in zip(
                                batch['preds'], batch['labels'], batch['inputs'], batch['categories']
                            ):
                                gt_inp_set.add(input + label)
                                deduplicated_outputs['preds'].append(pred)
                                deduplicated_outputs['labels'].append(label)
                                deduplicated_outputs['categories'].append(category)
                                deduplicated_outputs['inputs'].append(input)
                    self.write_predictions_to_file(
                        deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}"
                    )
                torch.distributed.barrier()
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
                self.log(f"validation_{self.val_metric_name}", averaged_metric, batch_size=1)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, batch_size=1)

        app_state = AppState()
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
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return averaged_loss, averaged_metric

    def write_predictions_to_file(self, outputs, output_file_path_prefix):
        with open(output_file_path_prefix + "_inputs_preds_labels.jsonl", "w") as f_json:
            assert len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels'])
            for i, p, l in zip(outputs['inputs'], outputs['preds'], outputs['labels']):
                f_json.write(json.dumps({'input': i, 'pred': p, 'label': l}) + '\n')

    def validation_step(self, dataloader_iter, batch_idx, dataloader_idx=0):
        return self.inference_step(dataloader_iter, batch_idx, 'validation', dataloader_idx)

    def on_validation_epoch_end(self):
        _ = self.inference_epoch_end(self.validation_step_outputs, 'validation', self.cfg.data.validation_ds)
        # Commenting as on_validation_epoch_end was a no-op in PTL 1.9
        # return super().on_validation_epoch_end()

    def test_step(self, dataloader_iter, batch_idx, dataloader_idx=0):
        return self.inference_step(dataloader_iter, batch_idx, 'test', dataloader_idx)

    def on_test_epoch_end(self):
        _ = self.inference_epoch_end(self.test_step_outputs, 'test', self.cfg.data.test_ds)
        # Commenting as on_test_epoch_end was a no-op in PTL 1.9
        # return super().on_test_epoch_end()

    def build_data_loader(
        self, dataset, global_batch_size, shuffle, num_workers, pin_memory, drop_last,
    ):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        if isinstance(dataset, ConcatMapDataset):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn
        # Data loader. Note that batch size is the per GPU batch size.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=global_batch_size // parallel_state.get_data_parallel_world_size(),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def setup_training_data(self):
        if not self.cfg.data.train_ds.drop_last:
            raise AttributeError(
                "`drop_last` is required for the training dataset to ensure each batch is the same micro-batch size."
                "To set this, set the variable `data.train_ds.drop_last=True` in the config."
            )
        self._train_dl = self.build_data_loader(
            self._train_ds,
            global_batch_size=self.cfg.data.train_ds.global_batch_size,
            shuffle=self.cfg.data.train_ds.shuffle,
            num_workers=self.cfg.data.train_ds.num_workers,
            pin_memory=self.cfg.data.train_ds.pin_memory,
            drop_last=self.cfg.data.train_ds.drop_last,
        )

    def setup_eval_data(self, datasets, data_cfg):
        dataloaders = []
        for dataset in datasets:
            eval_dl = self.build_data_loader(
                dataset,
                global_batch_size=self.cfg.data.test_ds.global_batch_size
                if hasattr(self.cfg.data, "test_ds")
                else self.cfg.data.validation_ds.global_batch_size,
                shuffle=data_cfg.shuffle,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
                drop_last=data_cfg.drop_last,
            )
            dataloaders.append(eval_dl)
        return dataloaders

    def setup_validation_data(self):
        self._validation_dl = self.setup_eval_data(self._validation_ds, self.cfg.data.validation_ds)

    def setup_test_data(self):
        self._test_dl = self.setup_eval_data(self._test_ds, self.cfg.data.test_ds)

    def _build_train_dataset(self, data_cfg):
        """Build the training dataset."""
        if (
            data_cfg.drop_last is False
            and data_cfg.global_batch_size > data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size()
        ):
            raise ValueError(
                f"Cannot use drop_last=False in your training data with gradient accumulation found grad acc of {data_cfg.global_batch_size // (data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size())} with global_batch_size {data_cfg.global_batch_size}, micro_batch_size {data_cfg.micro_batch_size}, data parallel size {parallel_state.get_data_parallel_world_size()}"
            )
        datasets = []
        if hasattr(data_cfg, "src_file_name") and hasattr(data_cfg, "tgt_file_name"):
            # Determine if we are using a single dataset or a list of datasets.
            is_src_list_config = isinstance(data_cfg.src_file_name, ListConfig)
            is_tgt_list_config = isinstance(data_cfg.tgt_file_name, ListConfig)

            if (is_src_list_config and not is_tgt_list_config) or (is_tgt_list_config and not is_src_list_config):
                raise ValueError("src_list and tgt_list must both be either a ListConfig or a string. ")
            if is_src_list_config:
                if len(data_cfg.src_file_name) != len(data_cfg.tgt_file_name):
                    raise ValueError("src_file_name and tgt_file_name must have the same number of elements. ")
            else:
                data_cfg.src_file_name = [data_cfg.src_file_name]
                data_cfg.tgt_file_name = [data_cfg.tgt_file_name]

            for src, tgt in zip(data_cfg.src_file_name, data_cfg.tgt_file_name):
                dataset = SequenceToSequenceDataset(
                    src_file_name=src,
                    tgt_file_name=tgt,
                    src_tokenizer=self.tokenizer,
                    tgt_tokenizer=self.tokenizer,
                    max_src_seq_length=data_cfg.max_src_seq_length,
                    max_tgt_seq_length=data_cfg.max_tgt_seq_length,
                    add_bos_to_input=data_cfg.get('add_bos_to_input', True),
                    add_eos_to_input=data_cfg.get('add_eos_to_input', True),
                    replace_bos_with_pad=data_cfg.get('replace_bos_with_pad', False),
                )
                datasets.append(dataset)
        elif hasattr(data_cfg, "file_names"):
            for file_path in data_cfg.file_names:
                dataset = T5SFTDataset(
                    file_path=file_path,
                    src_tokenizer=self.tokenizer,
                    tgt_tokenizer=self.tokenizer,
                    max_src_seq_length=data_cfg.max_seq_length,
                    max_tgt_seq_length=data_cfg.max_seq_length,
                    add_bos_to_input=data_cfg.get('add_bos', True),
                    add_eos_to_input=data_cfg.get(
                        'add_eos', True
                    ),  # review: need domain knowledge to undertand if these args are ok
                    index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                    memmap_workers=data_cfg.get('memmap_workers', None),
                    hf_dataset=data_cfg.get('hf_dataset', False),
                )
                datasets.append(dataset)
        else:
            raise ValueError("You must specify either (src_file_name and tgt_file_name) or file_names in data config")

        if len(datasets) > 1:
            dataset = ConcatMapDataset(
                datasets=datasets,
                sampling_technique=data_cfg.get('concat_sampling_technique', 'temperature'),
                sampling_temperature=data_cfg.get('concat_sampling_temperature', 5),
                sampling_probabilities=data_cfg.get(
                    'concat_sampling_probabilities', [1 / len(datasets)] * len(datasets)
                ),
            )
            return dataset
        else:
            return datasets[0]

    def _build_eval_dataset(self, data_cfg):
        """Build the evaluation dataset."""
        if data_cfg.global_batch_size > data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size():
            raise ValueError(
                f'You are trying to use "implicit gradient accumulation" of {data_cfg.global_batch_size // (data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size())} in your validation/test datasets. This is not supported. Please set global_batch_size equal to micro_batch_size * data_parallel_world_size.'
            )
        datasets = []
        if hasattr(data_cfg, "src_file_name") and hasattr(data_cfg, "tgt_file_name"):
            # Determine if we are using a single dataset or a list of datasets.
            is_src_list_config = isinstance(data_cfg.src_file_name, ListConfig)
            is_tgt_list_config = isinstance(data_cfg.tgt_file_name, ListConfig)
            is_names_list_config = False
            if hasattr(data_cfg, "names"):
                if isinstance(data_cfg.names, ListConfig):
                    is_names_list_config = True

            if (is_src_list_config and not is_tgt_list_config) or (is_tgt_list_config and not is_src_list_config):
                raise ValueError("src_list and tgt_list must both be either a ListConfig or a string. ")
            if is_src_list_config:
                if len(data_cfg.src_file_name) != len(data_cfg.tgt_file_name):
                    raise ValueError("src_file_name and tgt_file_name must have the same number of elements. ")
                if is_names_list_config and len(data_cfg.names) != len(data_cfg.src_file_name):
                    raise ValueError(
                        "If you are providing names for each src/tgt file, they must have the same number of elements."
                    )
            else:
                data_cfg.src_file_name = [data_cfg.src_file_name]
                data_cfg.tgt_file_name = [data_cfg.tgt_file_name]

            for src, tgt in zip(data_cfg.src_file_name, data_cfg.tgt_file_name):
                dataset = SequenceToSequenceDataset(
                    src_file_name=src,
                    tgt_file_name=tgt,
                    src_tokenizer=self.tokenizer,
                    tgt_tokenizer=self.tokenizer,
                    max_src_seq_length=data_cfg.max_src_seq_length,
                    max_tgt_seq_length=data_cfg.max_tgt_seq_length,
                    add_bos_to_input=data_cfg.get('add_bos_to_input', True),
                    add_eos_to_input=data_cfg.get('add_eos_to_input', True),
                    replace_bos_with_pad=data_cfg.get('replace_bos_with_pad', False),
                )
                datasets.append(dataset)
        elif hasattr(data_cfg, "file_names"):
            for file_path in data_cfg.file_names:
                dataset = T5SFTDataset(
                    file_path=file_path,
                    src_tokenizer=self.tokenizer,
                    tgt_tokenizer=self.tokenizer,
                    max_src_seq_length=data_cfg.max_seq_length,
                    max_tgt_seq_length=data_cfg.max_seq_length,
                    add_bos_to_input=data_cfg.get('add_bos', True),
                    add_eos_to_input=data_cfg.get('add_eos', True),
                    index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                    memmap_workers=data_cfg.get('memmap_workers', None),
                    hf_dataset=data_cfg.get('hf_dataset', False),
                )
                datasets.append(dataset)
        else:
            raise ValueError("You must specify either (src_file_name and tgt_file_name) or file_names in data config")
        return datasets

    def build_train_valid_test_datasets(self, stage):
        logging.info('Building datasets ...')
        if stage != 'test':
            self._validation_ds = self._build_eval_dataset(self.cfg.data.validation_ds)

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                self._test_ds = self._build_eval_dataset(self.cfg.data.test_ds)

        if stage == 'validate' or stage == 'test':
            return
        self._train_ds = self._build_train_dataset(self.cfg.data.train_ds)
        logging.info(f'Finished building datasets ...')
