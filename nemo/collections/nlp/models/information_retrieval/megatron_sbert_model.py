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

import json
import logging
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor, nn

from nemo.collections.nlp.data.information_retrieval.bert_embedding_dataset import BertEmbeddingDataset
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import BertModel, bert_extended_attention_mask
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    build_position_ids,
)
from nemo.utils import logging

try:
    from megatron.core import ModelParallelConfig, parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


##########################
# Below class is copied from SentenceTransformer library: https://github.com/UKPLab/sentence-transformers/blob/08a57b4a19ddaf7cccda51cd0c2c8af7bbc339a3/sentence_transformers/models/Normalize.py
##########################


class Normalize(nn.Module):
    """
    This layer normalizes embeddings to unit length
    """

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, features: Dict[str, Tensor]):
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features


##########################
# Below class is copied from SentenceTransformer library: https://github.com/UKPLab/sentence-transformers/blob/08a57b4a19ddaf7cccda51cd0c2c8af7bbc339a3/sentence_transformers/models/Pooling.py
##########################


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but divide by sqrt(input_length).
    :param pooling_mode_weightedmean_tokens: Perform (position) weighted mean pooling, see https://arxiv.org/abs/2202.08904
    :param pooling_mode_lasttoken: Perform last token pooling, see https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
    ):
        super(Pooling, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
        ]

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ["mean", "max", "cls", "weightedmean", "lasttoken"]
            pooling_mode_cls_token = pooling_mode == "cls"
            pooling_mode_max_tokens = pooling_mode == "max"
            pooling_mode_mean_tokens = pooling_mode == "mean"
            pooling_mode_weightedmean_tokens = pooling_mode == "weightedmean"
            pooling_mode_lasttoken = pooling_mode == "lasttoken"

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken

        pooling_mode_multiplier = sum(
            [
                pooling_mode_cls_token,
                pooling_mode_max_tokens,
                pooling_mode_mean_tokens,
                pooling_mode_mean_sqrt_len_tokens,
                pooling_mode_weightedmean_tokens,
                pooling_mode_lasttoken,
            ]
        )
        self.pooling_output_dimension = pooling_mode_multiplier * word_embedding_dimension

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
                .to(token_embeddings.device)
            )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
            # Any sequence where min == 1, we use the entire sequence length since argmin = 0
            values, indices = torch.min(attention_mask, 1, keepdim=False)
            gather_indices = torch.where(values == 0, indices, seq_len) - 1  # Shape [bs]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sentence_embedding": output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}


class SBertModel(BertModel):
    """
    Bert Language model.
    Model returns [seq, batch, hidden] shape
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_lm_cross_entropy=False,
        hidden_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        layernorm_epsilon=1e-5,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        masked_softmax_fusion=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        add_binary_head=True,
        skip_head=False,
        megatron_legacy=False,
        sequence_parallel=False,
        position_embedding_type='learned_absolute',
    ):
        super().__init__(
            config,
            vocab_size,
            hidden_size,
            max_position_embeddings,
            num_layers,
            num_attention_heads,
            ffn_hidden_size,
            apply_query_key_layer_scaling,
            kv_channels,
            num_tokentypes,
            parallel_output,
            pre_process,
            post_process,
            init_method_std,
            fp16_lm_cross_entropy,
            hidden_dropout,
            precision,
            fp32_residual_connection,
            activations_checkpoint_granularity,
            activations_checkpoint_method,
            activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline,
            layernorm_epsilon,
            normalization,
            transformer_block_type,
            masked_softmax_fusion,
            bias_gelu_fusion,
            bias_dropout_add_fusion,
            openai_gelu,
            onnx_safe,
            add_binary_head,
            skip_head,
            megatron_legacy,
            sequence_parallel,
            position_embedding_type,
        )

        self.pooling_add_on = Pooling(
            word_embedding_dimension=1024,
            pooling_mode_cls_token=False,
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_sqrt_len_tokens=False,
        )

        self.normalize_add_on = Normalize()

    def forward(
        self,
        bert_model_input,
        attention_mask,
        token_type_ids=None,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
    ):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            input_ids = bert_model_input
            position_ids = build_position_ids(input_ids)
        else:
            position_ids = None
            input_ids = None

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            token_type_ids=token_type_ids,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.post_process and self.add_binary_head:

            lm_output, _ = lm_output

        add_on_inputs = {"token_embeddings": lm_output[0].permute(1, 0, 2), "attention_mask": attention_mask}
        lm_output = self.pooling_add_on(add_on_inputs)
        lm_output = self.normalize_add_on(lm_output)

        return lm_output['sentence_embedding']


class MegatronSBertModel(MegatronBertModel):
    """
    Megatron Bert pretraining.
    Model returns [batch, seq, hidden] shape
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):

        super().__init__(cfg, trainer=trainer)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
        softmax_temp = cfg.get('softmax_temp', 0.05)
        self.scale = 1.0 / softmax_temp
        train_file_path = self.cfg.data.data_prefix
        with open(train_file_path) as f:
            train_data = json.load(f)

        random_seed = 42
        set_seed(random_seed)
        random.shuffle(train_data)

        self.train_data = train_data

    def model_provider_func(self, pre_process, post_process):
        cfg = self.cfg
        num_tokentypes = 2 if cfg.bert_binary_head else 0

        if self.mcore_bert:
            raise ValueError("mcore not supported for SBERT")

        else:
            model = SBertModel(
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
                transformer_block_type=cfg.get('transformer_block_type', 'pre_ln'),
                bias_gelu_fusion=cfg.get('bias_gelu_fusion', True),
                bias_dropout_add_fusion=cfg.get("bias_dropout_add_fusion", True),
                onnx_safe=cfg.get('onnx_safe', False),
                add_binary_head=cfg.bert_binary_head,
                skip_head=cfg.get('skip_head', False),
                megatron_legacy=cfg.get('megatron_legacy', False),
                position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
            )

        return model

    def build_train_valid_test_datasets(self):

        train_file_path = self.cfg.data.data_prefix

        train_data = self.train_data

        query_prefix = "query:"
        passage_prefix = "passage:"
        evaluation_sample_size = self.cfg.data.get("evaluation_sample_size", 100)
        hard_negatives_to_train = self.cfg.data.get("hard_negatives_to_train", 4)
        evaluation_steps = self.cfg.data.get("evaluation_steps", 100)

        # TODO @ataghibakhsh: Handle valid and test datasets better

        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None

        if train_file_path:  # we don't support calculating validation loss for multiple train files
            valid_data = None
            if evaluation_sample_size:
                if evaluation_steps == 0:
                    raise ValueError(
                        "The --evaluation_steps should be greater than 0 " "when --evaluation_sample_size is set"
                    )

                if evaluation_sample_size >= len(train_data):
                    raise ValueError("The --evaluation_sample_size cannot be greater " "than train set size.")

                valid_data = train_data[-evaluation_sample_size:]
                train_data = train_data[:-evaluation_sample_size]

            if evaluation_sample_size:
                self._validation_ds = BertEmbeddingDataset(
                    valid_data,
                    num_hard_negs=hard_negatives_to_train,
                    query_prefix=query_prefix,
                    passage_prefix=passage_prefix,
                )

        self._train_ds = BertEmbeddingDataset(
            train_data, num_hard_negs=hard_negatives_to_train, query_prefix=query_prefix, passage_prefix=passage_prefix
        )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building Bert datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert(
            self.model
        )

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
                # self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    sync_embeddings = (
                        module.initialize_last_stage_with_word_embeddings
                        if self.mcore_bert
                        else module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                sync_embeddings = (
                    self.model.initialize_last_stage_with_word_embeddings
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
        )

        dataloader.collate_fn = self.batching_collate

        return dataloader

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):

        max_seq_length = self.cfg.encoder_seq_length
        do_lower_case = self.cfg.tokenizer.get("do_lower_case", False)
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer.tokenizer(
                *to_tokenize, padding=True, truncation="longest_first", return_tensors="pt", max_length=max_seq_length,
            )
        )
        return output

    def batching_collate(self, batch):
        """
            Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
            Here, batch is a list of InputExample instances: [InputExample(...), ...]

            :param batch:
                a batch from a SmartBatchingDataset
            :return:
                a batch of tensors for the model
            """

        sentence_features = [self.tokenize(sentence) for sentence in zip(*batch)]

        return sentence_features

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            batches = next(dataloader_iter)

            (
                tokens_batch,
                types_batch,
                sentence_order_batch,
                loss_mask_batch,
                lm_labels_batch,
                padding_mask_batch,
            ) = ([], [], [], [], [], [])
            for batch in batches:
                tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = (
                    batch['input_ids'].cuda(non_blocking=True),
                    batch['token_type_ids'].cuda(non_blocking=True),
                    None,
                    None,
                    None,
                    batch['attention_mask'].cuda(non_blocking=True),
                )
                tokens_batch.append(tokens)
                types_batch.append(types)
                sentence_order_batch.append(sentence_order)
                loss_mask_batch.append(loss_mask)
                lm_labels_batch.append(lm_labels)
                padding_mask_batch.append(padding_mask)

            if not self.cfg.bert_binary_head:
                types = None

            forward_args = [
                {"input_ids": tokens, "token_type_ids": types, "attention_mask": padding_mask}
                for tokens, padding_mask, types in zip(tokens_batch, padding_mask_batch, types_batch)
            ]

            if self.mcore_bert:
                raise Exception("mcore not supported at the moment. It will be added in the near future")
            else:
                output_tensor = [self.forward(**forward_arg).permute(1, 0) for forward_arg in forward_args]

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
        queries = output_tensor[0]  # shape (bs, embedding_dim)
        positives = output_tensor[1]  # shape (bs, embedding_dim)

        pos_inbatch_negs_scores = torch.mm(
            queries, positives.transpose(0, 1)
        )  # shape (bs, bs); each positive is negative for other queries.

        hard_negs = output_tensor[2:]  # List of length "num_negatives", each tensor of shape (bs, embedding_dim)

        hard_negs_scores = (
            torch.multiply(queries.unsqueeze(0).repeat(len(hard_negs), 1, 1), torch.stack(hard_negs),).sum(axis=-1).T
        )  # shape = (bs, num_negatives); Hard negatives are not shared between queries.

        scores = torch.cat([pos_inbatch_negs_scores, hard_negs_scores], axis=1)

        scores *= self.scale

        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Indices of the (query, positive) pairs

        return {'lm loss': self.cross_entropy_loss(scores, labels)}
