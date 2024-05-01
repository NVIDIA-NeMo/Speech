# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging

try:
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False
import torch
from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.information_retrieval.bert_embedding_dataset import BertEmbeddingDataset
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.models.information_retrieval.bert_embedding_model import (
    MCoreBertEmbeddingModel,
    NeMoBertEmbeddingModel,
)
from nemo.collections.nlp.models.language_modeling.megatron.bert.bert_spec import (
    bert_layer_with_transformer_engine_spec_postln,
)
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


class MegatronBertEmbeddingModel(MegatronBertModel):
    """
    Megatron Bert pretraining.
    Model returns [batch, seq, hidden] shape
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):

        super().__init__(cfg, trainer=trainer)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
        softmax_temp = cfg.get('softmax_temp', 0.05)
        self.scale = 1.0 / softmax_temp

    def model_provider_func(self, pre_process, post_process):
        cfg = self.cfg
        num_tokentypes = 2 if cfg.bert_binary_head else 0
        transformer_block_type = cfg.get('transformer_block_type', 'post_ln')
        if self.mcore_bert:
            if transformer_block_type == 'pre_ln':
                layer_spec = bert_layer_with_transformer_engine_spec
            else:
                layer_spec = bert_layer_with_transformer_engine_spec_postln
            model = MCoreBertEmbeddingModel(
                config=self.transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=self.padded_vocab_size,
                max_sequence_length=cfg.max_position_embeddings,
                num_tokentypes=num_tokentypes,
                add_binary_head=cfg.bert_binary_head,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                transformer_block_type=transformer_block_type,
                add_pooler=self.cfg.get('add_pooler', True),
            )

        else:
            model = NeMoBertEmbeddingModel(
                config=self.model_parallel_config,
                vocab_size=self.padded_vocab_size,
                hidden_size=cfg.hidden_size,
                max_position_embeddings=cfg.max_position_embeddings,
                num_layers=cfg.num_layers,
                num_attention_heads=cfg.num_attention_heads,
                apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
                kv_channels=cfg.get('kv_channels', None),
                ffn_hidden_size=cfg.ffn_hidden_size,
                num_tokentypes=num_tokentypes,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=cfg.get('init_method_std', 0.02),
                fp16_lm_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
                hidden_dropout=cfg.get('hidden_dropout', 0.1),
                precision=cfg.get('precision', 16),
                fp32_residual_connection=cfg.get('fp32_residual_connection', False),
                activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
                activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
                activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
                activations_checkpoint_layers_per_pipeline=self.cfg.get(
                    'activations_checkpoint_layers_per_pipeline', None
                ),
                layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
                masked_softmax_fusion=cfg.get('masked_softmax_fusion', True),
                normalization=cfg.get('normalization', 'layernorm'),
                transformer_block_type=transformer_block_type,
                bias_gelu_fusion=cfg.get('bias_gelu_fusion', True),
                bias_dropout_add_fusion=cfg.get("bias_dropout_add_fusion", True),
                onnx_safe=cfg.get('onnx_safe', False),
                add_binary_head=cfg.bert_binary_head,
                megatron_legacy=cfg.get('megatron_legacy', False),
                position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
                add_pooler=cfg.get('add_pooler', True),
                add_lm_head=cfg.get('add_lm_head', False),
            )

        return model

    def build_train_valid_test_datasets(self):

        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None

        self._train_ds = BertEmbeddingDataset(
            self.cfg.data.data_train,
            tokenizer=self.tokenizer,
            add_bos=True,
            num_hard_negatives=self.cfg.data.get("hard_negatives_to_train", 4),
            max_seq_length=self.cfg.encoder_seq_length,
        )
        if self.cfg.data.data_validation:
            self._validation_ds = BertEmbeddingDataset(
                self.cfg.data.data_validation,
                tokenizer=self.tokenizer,
                add_bos=True,
                num_hard_negatives=self.cfg.data.get("hard_negatives_to_train", 4),
                max_seq_length=self.cfg.encoder_seq_length,
            )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building SBert datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def setup(self, stage=None):
        """
        PTL hook that is executed after DDP spawns.
        We setup datasets here as megatron datasets require DDP to instantiate.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.

        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert()

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if stage == 'predict':
            return
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            if self.cfg.data.dataloader_type == "LDDL":
                self.build_LDDL_data(self.cfg.data)
                torch.distributed.barrier()
            else:
                self.build_train_valid_test_datasets()
                self.setup_training_data(self.cfg.data)
                self.setup_validation_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    sync_embeddings = (
                        module.setup_embeddings_and_output_layer
                        if self.mcore_bert
                        else module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                sync_embeddings = (
                    self.model.setup_embeddings_and_output_layer
                    if self.mcore_bert
                    else self.model.sync_initial_word_embeddings
                )
                sync_embeddings()

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_bert', False):
            self.setup_transformer_engine_tp_groups()

    @classmethod
    def merge_cfg_with(cls, path, cfg):
        """
        Merge a given configuration dictionary `cfg` with the configuration dictionary
        obtained from restoring a MegatronBertModel at the specified `path`.

        Args:
            path (str): The path to the Bert model checkpoint to be restored.
            cfg (DictConfig): The configuration dictionary to merge.

        Returns:
            DictConfig: The merged configuration dictionary.

        Examples:
            >>> path = "/path/to/model/checkpoint"
            >>> cfg = DictConfig({"model": {"key": "value"}, "trainer": {"precision": 16}})
            >>> merged_cfg = merge_cfg_with(path, cfg)

        Notes:
            - The function resolves variables within the `cfg` dictionary using `OmegaConf.resolve`.
            - Keys in `cfg.model` will override the corresponding keys in the output dictionary.
            - If "train_ds" exists in `cfg.model.data`, it updates `micro_batch_size` and `global_batch_size`.
            - If `cfg.trainer` contains a "precision" key, it updates `output.precision`.

        """

        base_cfg = cls.restore_from(path, return_config=True)

        OmegaConf.resolve(cfg)
        with open_dict(base_cfg):
            for key, val in cfg.model.items():
                base_cfg[key] = val
            if "train_ds" in cfg.model.data:
                base_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
                base_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
            if cfg.get("trainer", None) and cfg.trainer.get("precision"):
                base_cfg.precision = cfg.trainer.precision

        return base_cfg

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
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

        # Torch dataloader.

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
            collate_fn=dataset.collate_fn,
        )
        return dataloader

    def setup_training_data(self, cfg):
        if self._train_ds:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if self._validation_ds:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def training_step(self, dataloader_iter):

        self._optimizer.zero_grad()

        if self.with_distributed_adam:
            # hack to enable overlapping param sync and forward compute
            # note: the distributed optimizer monkey-patches each
            # parameter's __getattribute__ function so that it can
            # launch parameter all-gathers the first time the
            # parameter is accessed after the optimizer step. However,
            # PyTorch directly passes embedding parameters into a C++,
            # bypassing this process. A quick-and-dirty hack is to
            # manually interact with the parameter.
            modules = self.model if isinstance(self.model, list) else [self.model]
            for module in modules:
                if isinstance(module, (Float16Module, MCoreFloat16Module)):
                    module = module.module
                if not self.mcore_bert:
                    module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        if self.cfg.data.dataloader_type == "LDDL":
            # this is of type bert dataset
            seq_length = dataloader_iter.iterator.loaders.get_seqlen()
        else:
            seq_length = self.cfg.encoder_seq_length

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=self._make_data_iterator_list(dataloader_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=False,
            seq_length=seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        if losses_reduced_per_micro_batch:
            loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.vstack(loss_tensors_list)
            loss_mean = loss_tensor.mean(axis=0)
        else:
            if self.cfg.bert_binary_head == True:
                loss_mean = torch.tensor([0.0, 0.0, 0.0]).cuda()
            else:
                loss_mean = torch.tensor([0.0, 0.0]).cuda()

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_O2:
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
                # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.torch_dtype == torch.float16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean[0], prog_bar=True, batch_size=1)
        if len(loss_mean) > 2:
            self.log('reduced_lm_train_loss', loss_mean[1], prog_bar=True, batch_size=1)
            self.log('reduced_sop_train_loss', loss_mean[2], prog_bar=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, batch_size=1)
        self.log('global_step', self.trainer.global_step, prog_bar=True, batch_size=1)
        self.log(
            'consumed_samples', self._compute_consumed_samples_after_training_step(), prog_bar=True, batch_size=1,
        )
        return loss_mean[0]

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            batches = next(dataloader_iter)[0]
            batches = {k: v.cuda(non_blocking=True) for k, v in batches.items()}

            if self.mcore_bert:

                batches["tokentype_ids"] = batches.pop("token_type_ids")
                output_tensor = model(**batches)
            else:
                output_tensor = self.forward(**batches).permute(1, 0)

            def loss_func(output_tensor):

                loss_dict = self.loss_func(output_tensor)

                if 'sop loss' in loss_dict:
                    lm_loss = loss_dict['lm loss']
                    sop_loss = loss_dict['sop loss']
                    loss = lm_loss + sop_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss, sop_loss])
                else:
                    lm_loss = loss_dict['lm loss']
                    loss = lm_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss])

                return loss, {'loss': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def loss_func(self, output_tensor):

        chunks = output_tensor.chunk(self.cfg.micro_batch_size)
        queries = torch.stack([item[0] for item in chunks])  # shape (bs, embedding_dim)
        positives = torch.stack([item[1] for item in chunks])  # shape (bs, embedding_dim)

        pos_inbatch_negs_scores = torch.mm(
            queries, positives.transpose(0, 1)
        )  # shape (bs, bs); each positive is negative for other queries.

        hard_negs = [
            torch.stack([item[i + 2] for item in chunks])
            for i in range(self.cfg.data.get("hard_negatives_to_train", 4))
        ]  # List of length "num_negatives", each tensor of shape (bs, embedding_dim)

        hard_negs_scores = (
            torch.multiply(queries.unsqueeze(0).repeat(len(hard_negs), 1, 1), torch.stack(hard_negs),).sum(axis=-1).T
        )  # shape = (bs, num_negatives); Hard negatives are not shared between queries.

        scores = torch.cat([pos_inbatch_negs_scores, hard_negs_scores], axis=1)

        scores *= self.scale

        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Indices of the (query, positive) pairs

        return {'lm loss': self.cross_entropy_loss(scores, labels)}
