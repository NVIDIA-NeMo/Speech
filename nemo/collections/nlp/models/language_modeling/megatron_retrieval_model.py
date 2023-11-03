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

import math
from typing import Any, List, Optional, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.retro_dataset import (
    build_mock_train_valid_test_datasets,
    build_train_valid_test_datasets,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.mup.init import normal_
from nemo.collections.nlp.modules.common.megatron.mup.shape import set_base_shapes
from nemo.collections.nlp.modules.common.megatron.retrieval_token_level_encoder_decoder import (
    MegatronRetrievalTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.text_generation_strategy import model_inference_strategy_dispatcher
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.utils import AppState, logging

try:
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = ["MegatronRetrievalModel"]


class MegatronRetrievalModel(MegatronBaseModel, TextGeneration):
    """
    Megatron Retrieval enhanced language model
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # TODO does not support PP yet
        self.model = self.model_provider_func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True)

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        if self.megatron_amp_O2:

            if not self.with_distributed_adam:
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                self.model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            self.model = Float16Module(
                config=self.model_parallel_config, module=self.model, precision=self.cfg.precision
            )

        # self.setup_optimizer_param_groups()
        self.model.model_type = ModelType.encoder_and_decoder

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        if hasattr(self.cfg, "shape_file"):
            set_base_shapes(self, self.register_artifact("shape_file", self.cfg.shape_file), rescale_params=False)

            # here manually initialize all the named parameters with the muTranfer normal initializer
            for name, tensor in self.named_parameters():
                if name.endswith('.dense_4h_to_h.weight') or name.endswith('.dense.weight'):
                    # initialize all the output dense matrix weight
                    # match the megatron lm model
                    std = self.cfg.init_method_std / math.sqrt(2.0 * 12.0)
                    normal_(tensor, 0, std)
                elif name.endswith('layernorm.weight'):
                    # initialize all the layer norm weight
                    if tensor.std() != 0 and tensor.mean() != 1:
                        raise ValueError(f'need to check {name} init')
                    normal_(tensor, 1, 0)
                elif name.endswith('.weight'):
                    # initialize all the other dense matrix weight
                    normal_(tensor, 0, self.cfg.init_method_std)
                else:
                    if tensor.std() != 0 and tensor.mean() != 0:
                        raise ValueError(f'need to check {name} init')

            # here manually overwrite the norm factor
            # note, has to turn off the model.apply_query_key_layer_scaling
            assert not self.cfg.apply_query_key_layer_scaling
            for name, layer in self.named_modules():
                if (
                    name.endswith('.self_attention')
                    or name.endswith('.inter_attention')
                    or name.endswith('.cross_attention')
                    or name.endswith('.core_attention')
                ):
                    if hasattr(layer, 'norm_factor') and hasattr(layer, 'hidden_size_per_attention_head'):
                        layer.norm_factor = (
                            layer.hidden_size_per_attention_head / 8.0
                        )  # divide 8 to make it consist with ADLR setting
                else:
                    if hasattr(layer, 'norm_factor') or hasattr(layer, 'hidden_size_per_attention_head'):
                        logging.error(
                            f'module {name} has norm factor but its name is not ending with attention, need to double check'
                        )

    def _build_tokenizer(self):
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merge_file", self._cfg.tokenizer.merge_file),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            legacy=False,
        )

        # add pad special token
        if not hasattr(self.tokenizer, "pad_id"):
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        elif hasattr(self.tokenizer, "pad_id") and (self.tokenizer.pad_id is None or self.tokenizer.pad_id < 0):
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

    def model_provider_func(self, pre_process, post_process, add_encoder, add_decoder):
        # TODO: create get_encoder_decoder_model()here for different losses (e..g, nll, vae, mim)

        model = MegatronRetrievalTokenLevelEncoderDecoderModule(
            config=self.model_parallel_config,
            vocab_size=self.padded_vocab_size,
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_attention_heads=self.cfg.num_attention_heads,
            ffn_hidden_size=self.cfg.ffn_hidden_size,
            apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=self.cfg.get('init_method_std', 0.02),
            fp16_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
            attention_dropout=self.cfg.get('attention_dropout', 0.1),
            precision=self.cfg.get('precision', 16),
            fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
            layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
            persist_layer_norm=self.cfg.get('persist_layer_norm', False),
            bias_gelu_fusion=self.cfg.get('bias_gelu_fusion', True),
            bias_dropout_add_fusion=self.cfg.get('bias_dropout_add_fusion', True),
            masked_softmax_fusion=self.cfg.get('masked_softmax_fusion', True),
            onnx_safe=self.cfg.get('onnx_safe', False),
            activation=self.cfg.get('activation', 'gelu'),
            bias=self.cfg.get('bias', True),
            normalization=self.cfg.get('normalization', 'layernorm'),
            headscale=self.cfg.get('headscale', False),
            transformer_block_type=self.cfg.get('transformer_block_type', 'pre_ln'),
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            chunk_size=self.cfg.get('chunk_size', 64),  # the chunk size used to retrive
            enc_num_layers=self.cfg.get('enc_num_layers', 4),  # total number of encoder layers
            dec_num_layers=self.cfg.get('dec_num_layers', 6),  # total number of decoder layers
            enc_cross_attention=self.cfg.get('enc_cross_attention', [3]),  # layer numbers for cross attention
            dec_cross_attention=self.cfg.get(
                'dec_cross_attention', [3, 5]
            ),  # layer numbers for chunked cross attention
            add_position_embedding=self.cfg.get(
                'add_position_embedding', False
            ),  # whether use the absolute postion encoding
            tokenizer=self.tokenizer,
            activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
            megatron_lm_compatible=self.cfg.get('megatron_lm_compatible', False),
            version=self.cfg.get('version', 1),
        )
        return model

    def forward(
        self,
        input_ids,
        input_attn_mask,
        retrieved_ids,
        retrieved_attn_mask,
        token_type_ids=None,
        labels=None,
        input_emb=None,
        position_ids=None,
    ):
        output_tensor = self.model(
            input_ids=input_ids,
            input_attn_mask=input_attn_mask,
            retrieved_ids=retrieved_ids,
            retrieved_attn_mask=retrieved_attn_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            input_emb=input_emb,
            position_ids=position_ids,
        )
        return output_tensor

    def training_step(self, batch, batch_idx):
        input_tokens_id = batch['tokens']
        input_attn_mask = batch['tokens_mask']
        loss_mask = batch['loss_mask']
        retrieved_ids = batch['retrieved_ids']
        retrieved_attn_mask = batch['retrieved_emb_mask']
        labels = batch['labels']
        if self.cfg.get('add_position_embedding', False):
            input_position_ids = build_position_ids(input_tokens_id)
        else:
            input_position_ids = None
        loss = self(
            input_tokens_id,
            input_attn_mask,
            retrieved_ids,
            retrieved_attn_mask,
            labels=labels,
            position_ids=input_position_ids,
        )
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(loss.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        reduced_loss = average_losses_across_data_parallel_group([lm_loss])
        self._reduced_loss_buffer.append(reduced_loss[0])

        if self.torch_dtype == torch.float16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        if self.with_distributed_adam:
            # gradients are reduced internally in distributed optimizer
            pass
        elif self.megatron_amp_O2:
            # while async grad allreduce is enabled, bprop will keep moving forward without waiting for
            # the finish of async grad AR works. Hence, to guarantee the correctness of grads reduction,
            # we cannot start weight update until all async grad AR works are done.
            if self.cfg.get('pipeline_model_parallel_size', 1) == 1:
                torch.cuda.synchronize()
            # when using pipeline parallelism grads must be reduced after the pipeline (not asynchronously)
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # no pipeline, so use the default pytorch lightning way of doing all_reduce
            # self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)
            pass

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True, batch_size=1)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr, batch_size=1)
            self.log('global_step', self.trainer.global_step, prog_bar=True, batch_size=1)
            self.log(
                'consumed_samples', self._compute_consumed_samples_after_training_step(), prog_bar=True, batch_size=1,
            )
            self._reduced_loss_buffer = []
        return lm_loss

    def validation_step(self, batch, batch_idx):
        prefix = "test" if self.trainer.testing else "val"
        input_tokens_id = batch['tokens']
        input_attn_mask = batch['tokens_mask']
        loss_mask = batch['loss_mask']
        retrieved_ids = batch['retrieved_ids']
        retrieved_attn_mask = batch['retrieved_emb_mask']
        labels = batch['labels']
        if self.cfg.get('add_position_embedding', False):
            input_position_ids = build_position_ids(input_tokens_id)
        else:
            input_position_ids = None
        loss = self(
            input_tokens_id,
            input_attn_mask,
            retrieved_ids,
            retrieved_attn_mask,
            labels=labels,
            position_ids=input_position_ids,
        )
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(loss.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        reduced_loss = average_losses_across_data_parallel_group([lm_loss])
        if prefix == 'val':
            self.validation_step_outputs.append(reduced_loss)
        else:
            self.test_step_outputs.apped(reduced_loss)
        return reduced_loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return None
        averaged_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True, batch_size=1)
        # formula to compute the perplexity
        # https://towardsdatascience.com/the-relationship-between-perplexity-and-entropy-in-nlp-f81888775ccc
        self.log('perplexity', torch.exp(averaged_loss), prog_bar=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory
        return averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        averaged_loss = torch.stack(self.test_step_outputs).mean()
        self.log('test_loss', averaged_loss, prog_bar=True, batch_size=1)
        logging.info(f'test_loss: {averaged_loss} ')
        self.log('perplexity', torch.exp(averaged_loss), prog_bar=True, batch_size=1)
        self.test_step_outputs.clear()  # free memory
        return averaged_loss

    def build_train_valid_test_datasets(self):
        logging.info('Building RETRO datasets.')
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        if self.cfg.data.get('mock', False):
            self._train_ds, self._validation_ds, self._test_ds = build_mock_train_valid_test_datasets(
                cfg=self.cfg,
                trainer=self.trainer,
                splits_string=self.cfg.data.splits_string,
                tokenizer=self.tokenizer,
                mock_data_size=self.cfg.data.get('mock_data_size', 10000),
            )
        else:
            self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
                cfg=self.cfg,
                trainer=self.trainer,
                data_prefix=self.cfg.data.data_prefix,
                data_impl=self.cfg.data.data_impl,
                splits_string=self.cfg.data.splits_string,
                train_valid_test_num_samples=train_valid_test_num_samples,
                seq_length=self.cfg.data.seq_length,
                seed=self.cfg.seed,
                skip_warmup=self.cfg.data.get('skip_warmup', True),
                tokenizer=self.tokenizer,
                retrieval_prefix=self.cfg.data.retrieval_prefix,
                knn_map_path=self.cfg.data.knn_index,
            )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building RETRO datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

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
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        """A PTL method to setup the training, validation and test datasets."""
        if stage == 'predict':
            return
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets()
        self.setup_training_data(self._cfg.data)
        self.setup_validation_data(self._cfg.data)
        self.setup_test_data(self._cfg.data)

    def set_inference_config(self, inference_config, retrieval_config):
        self._inference_config = inference_config
        self.inference_strategy = model_inference_strategy_dispatcher(self, **retrieval_config)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self._inference_config
        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                inference_config['inputs'] = batch
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                response = generate(self, **inference_config, strategy=self.inference_strategy)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
                return compute_prob_response
            else:
                inference_config['inputs'] = batch
                return generate(self, **inference_config, strategy=self.inference_strategy)

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        **args,
    ) -> OutputType:

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

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        return megatron_gpt_generate(self.cuda(), inputs, self.tokenizer, length_params, sampling_params, **args)

    def get_forward_output_only_func(self):
        """
        Used for generate method only.
        """

        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            (
                tokens,
                attention_mask,
                retrieved,
                retrieved_mask,
                set_inference_key_value_memory,
                inference_max_sequence_len,
                neighbors,
                position_ids,
            ) = batch

            if len(retrieved.shape) == 1:
                retrieved = None
                retrieved_mask = None
            else:
                retrieved = retrieved.cuda()
                retrieved_mask = retrieved_mask.cuda()

            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            extra_arg['neighbors'] = neighbors[0].item()
            extra_arg['position_ids'] = position_ids

            output_tensor = model(tokens, attention_mask, retrieved, retrieved_mask, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization([self.model])

    def list_available_models(self):
        pass
