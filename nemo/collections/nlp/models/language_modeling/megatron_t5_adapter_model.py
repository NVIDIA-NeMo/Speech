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
from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    LoraKQVAdapterConfig,
    LoraKVAdapterConfig,
    LoraQAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging, model_utils

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


class MegatronT5BaseAdapterModel(MegatronT5PromptLearningModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.adapter_name_keys = []

    def forward(
        self, input_ids, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels=None, inference=False,
    ):
        # Call forward on T5 model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.enc_dec_model(
                enc_input_ids=input_ids,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=None,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.enc_dec_model(
                    enc_input_ids=input_ids,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=None,
                )

        return output, None

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

    def compute_accuracy(self, enc_input, enc_mask, encoder_input, labels):
        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=enc_input,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
        )

        processed_inputs, processed_preds, processed_labels = [], [], []
        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        enc_inputs = enc_input.cpu().numpy().tolist()

        for i, (enc_input, pred, label) in enumerate(zip(enc_inputs, preds, labels)):
            if self.tokenizer.eos_id in pred:
                idx = pred.index(self.tokenizer.eos_id)
                pred = pred[:idx]

            additional_special_tokens_ids = []
            if hasattr(self.tokenizer.tokenizer, "additional_special_tokens_ids"):
                additional_special_tokens_ids = self.tokenizer.tokenizer.additional_special_tokens_ids

            pred = [id for id in pred if id not in additional_special_tokens_ids]
            label = [id for id in label if id not in additional_special_tokens_ids]
            enc_input = [id for id in enc_input if id not in additional_special_tokens_ids]

            pred = self.tokenizer.ids_to_text(pred)
            label = self.tokenizer.ids_to_text(label)
            enc_input = self.tokenizer.ids_to_text(enc_input)

            processed_preds.append(pred)
            processed_labels.append(label)
            processed_inputs.append(enc_input)

        return {
            'predicted_token_ids': processed_preds,
            'labels': processed_labels,
            'enc_inputs': processed_inputs,
        }

    def validation_step(self, dataloader_iter, batch_idx, inference=False):
        # Check if iterator is exhausted
        dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        if done:
            return
        batch = next(dataloader_iter)
        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        mode = self.training
        self.eval()
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(enc_input.size(0), gbs)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=True)

        if self.cfg.get('report_validation_metric', False):
            metrics = self.compute_accuracy(enc_input, enc_mask, labels)
            metrics['loss'] = loss_mean
        else:
            metrics = {'loss': loss_mean}

        self.validation_step_outputs.append(metrics)
        self.train(mode=mode)
        return metrics

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(enc_input.size(0), gbs)
        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=enc_input,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=None,
        )

        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(enc_input, self.tokenizer)

        if labels is not None:
            labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        else:
            labels_text = [None] * len(preds_text)

        return {
            'input_text': input_text,
            'preds_text': preds_text,
            'labels_text': labels_text,
        }

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

    def on_validation_epoch_end(self):
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                averaged_loss = torch.stack([i['loss'] for i in self.validation_step_outputs]).mean()
            else:
                averaged_loss = torch.tensor(0.0).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(averaged_loss, get_last_rank())

            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
            logging.info(f'Validation loss: {averaged_loss}')

        else:
            averaged_loss = torch.stack([item['loss'] for item in self.validation_step_outputs]).mean()
            logging.info(f'Validation loss: {averaged_loss}')
            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)

        if self.cfg.get('report_validation_accuracy', False):
            gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            all_preds = list(itertools.chain(*[item['predicted_token_ids'] for item in self.validation_step_outputs]))
            all_labels = list(itertools.chain(*[item['labels'] for item in self.validation_step_outputs]))
            all_inputs = list(itertools.chain(*[item['enc_inputs'] for item in self.validation_step_outputs]))

            assert len(all_preds) == len(all_labels)
            assert len(all_preds) == len(all_inputs)

            # Gather inputs, preds, labels from all workers
            torch.distributed.all_gather_object(
                gather_results,
                [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
                group=parallel_state.get_data_parallel_group(),
            )

            # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
            if parallel_state.get_data_parallel_rank() == 0:

                gather_results_dedup = list(set(itertools.chain(*gather_results)))

                correct = 0
                for (input, pred, label) in gather_results_dedup:
                    if pred == label:
                        correct += 1

                val_acc = correct / len(gather_results_dedup)
                val_acc = torch.tensor(val_acc).cuda()

                logging.info(f'Validation accuracy: {val_acc}')
            else:
                val_acc = torch.tensor(0.0).cuda()

            self.log('val_acc', val_acc, prog_bar=True, rank_zero_only=True, batch_size=1)

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        self.validation_step_outputs.clear()  # free memory


class MegatronT5AdapterLearningModel(MegatronT5BaseAdapterModel):
    """
    TODO  (@adithyare)
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
        frozen_model_cfg = MegatronT5Model.restore_from(
            cfg.get('language_model_path'), trainer=trainer, return_config=True
        )
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )

        self.frozen_model.freeze()
        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder

        if encoder:
            encoder_cfg = self._get_component_cfg('encoder', frozen_model_cfg, cfg)
            self._add_adapters_to_component(encoder, encoder_cfg, self.adapter_name_keys)
            logging.info(f'Adding encoder adapters:\n{self.frozen_model.summarize()}')

        if decoder:
            decoder_cfg = self._get_component_cfg('decoder', frozen_model_cfg, cfg)
            self._add_adapters_to_component(decoder, decoder_cfg, self.adapter_name_keys)
            logging.info(f'Adding decoder adapters:\n{self.frozen_model.summarize()}')

    def _add_adapters_to_component(self, component, component_cfg, adapter_name_keys):
        for _, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in adapter_name_keys:
                    adapter_cfg = self._get_adapter_cfg(component_cfg)
                    if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(name=adapter_key, cfg=adapter_cfg)

    def _get_component_cfg(self, component_name, frozen_model_cfg, cfg):
        if component_name in frozen_model_cfg:
            component_cfg = frozen_model_cfg.get(component_name)
            with open_dict(component_cfg):
                component_cfg.tensor_model_parallel_size = frozen_model_cfg.tensor_model_parallel_size
                component_cfg.adapter_tuning = cfg.adapter_tuning
        else:
            component_cfg = frozen_model_cfg
            with open_dict(component_cfg):
                component_cfg.adapter_tuning = cfg.adapter_tuning
        return component_cfg

    def _get_adapter_cfg(self, component_cfg):
        if component_cfg.adapter_tuning.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=component_cfg.hidden_size,
                out_features=component_cfg.hidden_size,
                dim=component_cfg.adapter_tuning.adapter_dim,
                norm_position=component_cfg.adapter_tuning.get('norm_position', 'pre'),
                norm_type=component_cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=component_cfg.adapter_tuning.get('column_init_method', 'xavier'),
                row_init_method=component_cfg.adapter_tuning.get('row_init_method', 'zero'),
                dropout=component_cfg.adapter_tuning.adapter_dropout,
            )
        else:
            adapter_cfg = LinearAdapterConfig(
                in_features=component_cfg.hidden_size,
                dim=component_cfg.adapter_tuning.adapter_dim,
                norm_position=component_cfg.adapter_tuning.get('norm_position', 'pre'),
                dropout=component_cfg.adapter_tuning.adapter_dropout,
            )
        return adapter_cfg

    @classmethod
    def list_available_models(cls):
        pass


class MegatronT5LoraModel(MegatronT5BaseAdapterModel):
    """
    TODO  (@adithyare)
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        # assert cfg.lora_tuning.get('adapter_dim', 0) > 0, "adapter_dim has not been set."
        # assert (
        #     cfg.lora_tuning.adapter_dim % cfg.tensor_model_parallel_size == 0
        # ), "The adapter dim should be divisible by tensor_model_parallel_size."

        encoder_adapter_name_keys = [AdapterName.LORA_KQV_ADAPTER]
        decoder_adapter_name_keys = [
            AdapterName.LORA_KQV_ADAPTER,
            AdapterName.LORA_KV_ADAPTER,
            AdapterName.LORA_Q_ADAPTER,
        ]

        # add adapter keys to the list -> to update state dict
        self.adapter_name_keys = encoder_adapter_name_keys + decoder_adapter_name_keys

        frozen_model_cfg = MegatronT5Model.restore_from(
            cfg.get('language_model_path'), trainer=trainer, return_config=True
        )
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )

        self.frozen_model.freeze()
        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder

        if encoder:
            encoder_cfg = self._get_component_cfg('encoder', frozen_model_cfg, cfg)
            self._add_adapters_to_component(encoder, encoder_cfg, encoder_adapter_name_keys)
            logging.info(f'Adding encoder adapters:\n{self.frozen_model.summarize()}')

        if decoder:
            decoder_cfg = self._get_component_cfg('decoder', frozen_model_cfg, cfg)
            self._add_adapters_to_component(decoder, decoder_cfg, decoder_adapter_name_keys)
            logging.info(f'Adding decoder adapters:\n{self.frozen_model.summarize()}')

    def _add_adapters_to_component(self, component, component_cfg, adapter_name_keys):
        for _, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in adapter_name_keys:
                    adapter_cfg = self._get_adapter_cfg(component_cfg, adapter_key)
                    if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(name=adapter_key, cfg=adapter_cfg)
                        print(f"in adding {adapter_key}")

    def _get_component_cfg(self, component_name, frozen_model_cfg, cfg):
        if component_name in frozen_model_cfg:
            component_cfg = frozen_model_cfg.get(component_name)
            with open_dict(component_cfg):
                component_cfg.tensor_model_parallel_size = frozen_model_cfg.tensor_model_parallel_size
                component_cfg.lora_tuning = cfg.lora_tuning
        else:
            component_cfg = frozen_model_cfg
            with open_dict(component_cfg):
                component_cfg.lora_tuning = cfg.lora_tuning
        return component_cfg

    def _get_adapter_cfg(self, component_cfg, adapter_key):
        if component_cfg.kv_channels is None:
            assert (
                component_cfg.hidden_size % component_cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = component_cfg.hidden_size // component_cfg.num_attention_heads
        else:
            kv_channels = component_cfg.kv_channels
        projection_size = kv_channels * component_cfg.num_attention_heads

        if adapter_key == AdapterName.LORA_KQV_ADAPTER:
            adapter_cfg = LoraKQVAdapterConfig(
                in_features=component_cfg.hidden_size,
                out_features=3 * projection_size,
                dim=component_cfg.lora_tuning.kqv_adapter_dim,
                norm_position="none",
                norm_type="none",
                activation="identity",
                column_init_method=component_cfg.lora_tuning.get("column_init_method", "normal"),
                row_init_method=component_cfg.lora_tuning.get("row_init_method", "zero"),
                gather_output=False,
                dropout=0.0,
            )
        elif adapter_key == AdapterName.LORA_KV_ADAPTER:
            adapter_cfg = LoraKVAdapterConfig(
                in_features=component_cfg.hidden_size,
                out_features=2 * projection_size,
                dim=component_cfg.lora_tuning.kv_adapter_dim,
                norm_position="none",
                norm_type="none",
                activation="identity",
                column_init_method=component_cfg.lora_tuning.get("column_init_method", "normal"),
                row_init_method=component_cfg.lora_tuning.get("row_init_method", "zero"),
                gather_output=False,
                dropout=0.0,
            )
        elif adapter_key == AdapterName.LORA_Q_ADAPTER:
            adapter_cfg = LoraQAdapterConfig(
                in_features=component_cfg.hidden_size,
                out_features=1 * projection_size,
                dim=component_cfg.lora_tuning.q_adapter_dim,
                norm_position="none",
                norm_type="none",
                activation="identity",
                column_init_method=component_cfg.lora_tuning.get("column_init_method", "normal"),
                row_init_method=component_cfg.lora_tuning.get("row_init_method", "zero"),
                gather_output=False,
                dropout=0.0,
            )
        else:
            raise RuntimeError("Unexpected adapter key name..")

        return adapter_cfg

    @classmethod
    def list_available_models(cls):
        pass


class MegatronT5InfusedAdapterModel(MegatronT5BaseAdapterModel):
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
        frozen_model_cfg = MegatronT5Model.restore_from(
            cfg.get('language_model_path'), trainer=trainer, return_config=True
        )
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )

        self.adapter_name_keys = [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED, AdapterName.MLP_INFUSED]
        self.frozen_model.freeze()
        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder

        if encoder:
            encoder_cfg = self._get_component_cfg('encoder', frozen_model_cfg)
            self._add_adapters_to_component(encoder, encoder_cfg, self.adapter_name_keys)
            logging.info(f'After adding encoder adapters:\n{self.frozen_model.summarize()}')

        if decoder:
            decoder_cfg = self._get_component_cfg('decoder', frozen_model_cfg)
            self._add_adapters_to_component(decoder, decoder_cfg, self.adapter_name_keys)
            logging.info(f'After adding all adapters:\n{self.frozen_model.summarize()}')

    def _add_adapters_to_component(self, component, component_cfg, adapter_name_keys):
        for _, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in adapter_name_keys:
                    adapter_cfg = self._get_adapter_cfg(component_cfg, adapter_key)
                    if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(name=adapter_key, cfg=adapter_cfg)

    def _get_component_cfg(self, component_name, frozen_model_cfg):
        if component_name in frozen_model_cfg:
            component_cfg = frozen_model_cfg.get(component_name)
            with open_dict(component_cfg):
                component_cfg.tensor_model_parallel_size = frozen_model_cfg.tensor_model_parallel_size
        else:
            component_cfg = frozen_model_cfg
        return component_cfg

    def _get_adapter_cfg(self, component_cfg, adapter_key):
        if adapter_key == AdapterName.MLP_INFUSED:
            cfg = MLPInfusedAdapterConfig(
                in_features=component_cfg.ffn_hidden_size // component_cfg.tensor_model_parallel_size
            )
        elif adapter_key in [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED]:
            if component_cfg.get('kv_channels', None):
                cfg = InfusedAdapterConfig(
                    in_features=component_cfg.kv_channels
                    * component_cfg.num_attention_heads
                    // component_cfg.tensor_model_parallel_size
                )
            else:
                cfg = InfusedAdapterConfig(
                    in_features=component_cfg.hidden_size // component_cfg.tensor_model_parallel_size
                )
        else:
            raise ValueError(f"Adapter Key {adapter_key} is unknown.")

        return cfg

    def _component_state_dict(self, component_name, component, adapter_name_keys):
        state_dict_ = {}
        for name, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([component_name, name, adapter_key])
                        state_dict_[state_adapter_key] = adapter_module.state_dict()
                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def _load_component_state_dict(
        self, component_name, component, adapter_name_keys, state_dict, strict: bool = True
    ):
        for name, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([component_name, name, adapter_key])
                        adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder
        encoder_state_dict = self._component_state_dict('encoder', encoder, self.adapter_name_keys) if encoder else {}
        decoder_state_dict = self._component_state_dict('decoder', decoder, self.adapter_name_keys) if decoder else {}
        state_dict_ = {
            **encoder_state_dict,
            **decoder_state_dict,
        }  # merge the two state dicts (does not check for collisions in keys)
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder
        if encoder:
            self._load_component_state_dict('encoder', encoder, self.adapter_name_keys, state_dict, strict)
        if decoder:
            self._load_component_state_dict('decoder', decoder, self.adapter_name_keys, state_dict, strict)

    @classmethod
    def list_available_models(cls):
        pass
