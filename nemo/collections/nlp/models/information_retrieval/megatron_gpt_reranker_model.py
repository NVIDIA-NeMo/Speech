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

import itertools
import os

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.information_retrieval.gpt_embedding_dataset import GPTRerankerDataset
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.models.information_retrieval.megatron_gpt_embedding_model import (
    MegatronGPTEmbeddingModel,
    _gather_global_inbatch_representations,
)
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def listify(tensor):
    l_tensor = []
    for t in tensor:
        for rid in range(t.shape[0]):
            r = t[rid, :].unsqueeze(0).cpu()
            l_tensor.append(r)
    return l_tensor


class MegatronGPTRerankerModel(MegatronGPTEmbeddingModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.reward_model_loss = cfg.get("reward_model_loss", False)
        super().__init__(cfg, trainer=trainer)

    def model_provider_func(self, pre_process, post_process):
        # (@adithyare) We need post_process to be False to get hidden states in the loss_func
        return super().model_provider_func(pre_process, post_process=False)

    def maybe_setup_test(self):
        if hasattr(self.cfg.data, 'test_ds') and self.cfg.data.test_ds.get('file_names', None) is not None:
            self._test_dl = self.setup_eval_dataloader(self._test_ds, self.cfg.data.test_ds)
        return

    def maybe_build_test(self):
        if hasattr(self.cfg.data, 'test_ds') and self.cfg.data.test_ds.get('file_names', None) is not None:
            logging.info('Building GPT Reranker test datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._test_ds = self._build_dataset(self.cfg.data.test_ds, is_train=False)

    def _build_dataset(self, data_cfg, is_train=True):
        packed_sequence = data_cfg.get("packed_sequence", False)

        # Determine if we are using a single dataset or a list of datasets.
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
        pad_seq_length_to_mult = (
            8 * self.cfg.get('tensor_model_parallel_size', 1) if self.cfg.get('sequence_parallel', False) else 16
        )
        pad_seq_length_to_mult *= self.cfg.get('context_parallel_size', 1)

        datasets = []
        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            dataset = GPTRerankerDataset(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                virtual_tokens=self.virtual_tokens,
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
                truncation_method=data_cfg.get(
                    'truncation_method', 'right'
                ),  # used to choose truncation method. Options: ['random', 'left', 'right']
                special_tokens=self.cfg.data.get(
                    'chat_prompt_tokens', None
                ),  # special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
                data_type="train" if is_train else "validation",
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

    def training_step_fwd_bwd_step_call(self, dataloader_iter, forward_only):
        loss_mean, non_loss_tensors = self.fwd_bwd_step(dataloader_iter, forward_only)
        logit_diff = non_loss_tensors['logit_diff'][0].item()
        self.log("logit_diff", logit_diff, prog_bar=True, rank_zero_only=True, batch_size=1)
        return loss_mean

    def inference_step_validation_call(self, batch, batch_idx, data_cfg, dataloader_idx=0):
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        loss, non_loss_tensors = self.local_validation_step(itertools.chain([dataloader_idx], [batch]))
        outputs = {
            'loss': loss,
            'metadata': metadata,  # [dict]
            'query_pos_doc_logit': non_loss_tensors['query_pos_doc_logit'],  # [batch_size, hidden_size]
        }
        return outputs

    def inference_loss_func(self, loss_mask, num_valid_tokens_in_ub, eos_tensors):
        query_pos_doc_hs = eos_tensors
        _blank = torch.zeros(1, device=query_pos_doc_hs.device, dtype=query_pos_doc_hs.dtype)[0]
        return {
            "loss": _blank,
            "query_pos_doc_logit": query_pos_doc_hs,
            "query_neg_doc_logit": _blank,
            "logit_diff": _blank,
        }

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        idx = torch.arange(output_tensor.shape[1], device=output_tensor.device)
        eos_tensors = output_tensor[loss_mask, idx, :]  # (bs x 1)
        if self.global_inbatch_negatives and self.trainer.training:
            eos_tensors = _gather_global_inbatch_representations(eos_tensors)
        if not self.trainer.training:
            return self.inference_loss_func(loss_mask, num_valid_tokens_in_ub, eos_tensors)
        bs = eos_tensors.shape[0] // 2
        query_pos_doc_hs = eos_tensors[::2, :]  # every second tensor from idx 0 is a query w pos_doc (bs x 1)
        query_neg_doc_hs = eos_tensors[1::2, :]  # every second tensor from idx 1 is a query w negative doc (bs x 1)

        if self.reward_model_loss:
            loss = -torch.nn.functional.logsigmoid(query_pos_doc_hs - query_neg_doc_hs).mean()
        else:
            cs = torch.cat([query_pos_doc_hs, query_neg_doc_hs], dim=1)  # (bs x 2)
            cs = cs / self.temperature
            labels = torch.zeros(bs, device=cs.device).long()
            loss = torch.nn.functional.cross_entropy(cs, labels)

        cp_size = self.cfg.get('context_parallel_size', 1)
        if cp_size > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        query_pos_doc_hs = query_pos_doc_hs.clone().detach()
        query_neg_doc_hs = query_neg_doc_hs.clone().detach()
        logit_diffs = torch.mean(query_pos_doc_hs - query_neg_doc_hs)
        return {
            "loss": loss,
            "query_pos_doc_logit": query_pos_doc_hs,
            "query_neg_doc_logit": query_neg_doc_hs,
            "logit_diff": logit_diffs,
        }

    def gather_and_maybe_write_predictions(self, output, data_cfg, mode, averaged_metric, dataloader_idx=0):
        if not data_cfg.get("write_embeddings_to_file", False):
            return True
        gathered_output_batches = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            gathered_output_batches,
            [
                {
                    'query_pos_doc_logit': batch['query_pos_doc_logit'],
                    'metadata': batch['metadata'],
                }
                for batch in output
            ],
            group=parallel_state.get_data_parallel_group(),
        )

        # Remove duplicate examples due to distributed sampler.
        deduplicated_outputs = {
            'query_pos_doc_logit': [],
            'metadata': [],
        }
        total_size, skipped = 0, 0
        for rank in range(0, parallel_state.get_data_parallel_world_size()):
            for batch in gathered_output_batches[rank]:
                l_q_hs = listify(batch['query_pos_doc_logit'])
                l_m = batch['metadata']
                assert len(l_m) == len(l_q_hs)
                for q_hs, metadata in zip(
                    l_q_hs,
                    l_m,
                ):
                    total_size += 1
                    if not metadata.get("__AUTOGENERATED__", False):
                        deduplicated_outputs['query_pos_doc_logit'].append(q_hs)
                        deduplicated_outputs['metadata'].append(metadata)
                    else:
                        skipped += 1

        logging.info(
            f"{total_size-skipped} deduplicated outputs in dataloader:{dataloader_idx}, (skipped {skipped} autogenerated examples)."
        )
        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        assert metric_name == "loss", "Only loss is supported for now."
        # avg_pos_cs = torch.tensor(deduplicated_outputs['avg_pos_cs']).mean().item()
        # avg_neg_cs = torch.tensor(deduplicated_outputs['avg_neg_cs']).mean().item()
        # diff_cs = torch.tensor(deduplicated_outputs['diff_cs']).mean().item()
        # self.log('val_avg_pos_cs', avg_pos_cs, prog_bar=True, rank_zero_only=True, batch_size=1)
        # self.log('val_avg_neg_cs', avg_neg_cs, prog_bar=True, rank_zero_only=True, batch_size=1)
        # self.log('val_diff_cs', diff_cs, prog_bar=True, rank_zero_only=True, batch_size=1)

        # Write predictions to file
        if self.global_rank == 0 and data_cfg.get("write_embeddings_to_file", False):
            logging.info(
                f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['metadata'])}"
            )

            # Check if the user provided a prefix path to the file(s) they want to write.
            if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                raise ValueError(
                    f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                )
            # (@adithyare) We are not using the log key to write the embeddings to file
            filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
            consumed_samples = self._compute_consumed_samples_after_training_step()
            fldr_path = f"{data_cfg.output_file_path_prefix}/consumed_samples{consumed_samples}/{filename_log_key}"
            self.write_embeddings_to_file(deduplicated_outputs, fldr_path, dataloader_idx)
        return deduplicated_outputs, total_size

    def write_embeddings_to_file(self, outputs, output_file_path, d_idx):
        hs = torch.cat(outputs['query_pos_doc_logit'], dim=0)
        hs_npy = hs.float().numpy()
        emb_fldr = f"{output_file_path}"
        os.makedirs(emb_fldr, exist_ok=True)
        with open(f"{output_file_path}/logits.ids", "w") as f:
            for m in outputs['metadata']:
                f.write(f"{m['query_id'].strip()} {m['doc_id']}\n")
        np.save(f"{emb_fldr}/logits.npy", hs_npy)
        return True
