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

import os
import shutil
import tarfile
import tempfile
from argparse import ArgumentParser
from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.nlp.parts.nlp_overrides import (
    NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState

"""
Usage:

### Tensor Parallelism and Pipeline Parallelism conversion ###

# Megatron GPT
python megatron_change_num_partitions.py \
    --model_file=PATH_TO_SRC_FILE \
    --target_file=PATH_TO_TGT_FILE \
    --tensor_model_parallel_size=-1 \
    --target_tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=-1 \
    --target_pipeline_model_parallel_size=1 \
    --precision=bf16

# Megatron T5
python megatron_change_num_partitions.py \
    --model_file=PATH_TO_SRC_FILE \
    --target_file=PATH_TO_TGT_FILE \
    --model_class="nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model" \
    --tensor_model_parallel_size=-1 \
    --target_tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=-1 \
    --target_pipeline_model_parallel_size=1 \
    --target_pipeline_model_parallel_split_rank=0 \
    --precision=bf16

# Megatron GPT + Virtual Pipeline parallelism

python megatron_change_num_partitions.py \
    --model_extracted_dir="<Directory of Pytorch Lightning ckpt folders>" \
    --target_file="<Name of the target NeMo file>" \
    --ckpt_name="<Name of a single Pytorch Lightning ckpt file inside the extracted dir>" \
    --tensor_model_parallel_size=<TP SIZE> \
    --target_tensor_model_parallel_size=<TARGET TP SIZE> \
    --pipeline_model_parallel_size=<PP SIZE> \
    --target_pipeline_model_parallel_size=<TARGET PP SIZE> \
    --virtual_pipeline_model_parallel_size=<VP SIZE> \
    --hparams_file="<Path to HPARAMS.yaml file>" \
    --precision=bf16

### Only Tensor Parallelism conversion ###

To the above commands, add the following argument: `--tp_conversion_only`

# Note: This requires that the pipeline_model_parallel_size and tgt_pipeline_model_parallel_size is set to 1.

### Large Models conversion ###

When converting large models, ** always ** ensure that you pre-extract the nemo model and then only perform conversion

$ mkdir "unpacked_nemo_file"
$ tar -xvf "<path to nemo file>" -C "<absolute path to pwd>/unpacked_nemo_file/"

python megatron_change_num_partitions.py \
    ...
    --model_extracted_dir="<Absolute path to pwd>/unpacked_nemo_file/"

### Model Classes ###

# NOTE: Conversion of other model types.
# Default model type is MegatronGPTModel, if you want another model you need to pass classpath of the model
# For example - MegatronT5Model -

python megatron_change_num_partitions.py \
    ...
    --model_class="nemo.collections.nlp.models.language_modeling.megatron_t5_model.MegatronT5Model"

# Additional arguments:

--num_gpu_per_node: Number of GPUs per node. Default is 8.
--megatron_legacy: Whether the model is a legacy Megatron model or not. Default is False. May be unsuported for
    Pipeline Parallelism change.
--tokenizer_model_path: Path to tokenizer model. Default is None. When not None, overrides the tokenizer model path
    in the model config.
--tokenizer_vocab_file: Path to tokenizer vocab file. Default is None. When not None, overrides the tokenizer vocab
    file in the model config.

# Comments

Passing --tensor_model_parallel_size=-1 or --pipeline_model_parallel_size=-1 will automatically infer the size from the
model config.

"""


def set_virtual_parallel_rank_safely(rank: int):
    AppState().virtual_pipeline_model_parallel_rank = rank

    try:
        from megatron.core import parallel_state

        parallel_state.set_virtual_pipeline_model_parallel_rank(rank)

        if rank is None:
            parallel_state.set_virtual_pipeline_model_parallel_world_size(None)

    except (ImportError, ModuleNotFoundError):
        logging.warning("`megatron-core` not installed, cannot set virtual parallel rank !")


#################
### Utilities ###
#################


def force_cpu_model(cfg):
    with open_dict(cfg):
        # temporarily set to cpu
        original_cpu_init = cfg.get('use_cpu_initialization', False)
        if 'megatron_amp_O2' in cfg:
            amp_o2_key = 'megatron_amp_O2'
            original_amp_o2 = cfg.megatron_amp_O2
        elif 'megatron_amp_02' in cfg:
            amp_o2_key = 'megatron_amp_02'
            original_amp_o2 = cfg.megatron_amp_02
        else:
            amp_o2_key, original_amp_o2 = None, None

        # Set new values
        cfg.use_cpu_initialization = True
        if amp_o2_key is not None:
            cfg[amp_o2_key] = False

        # Disable sequence parallelism - Not disabling this gives error when converting the the model to TP=1
        original_sequence_parallel = cfg.get('sequence_parallel', None)
        cfg.sequence_parallel = False

    # Setup restore dict
    restore_dict = {'use_cpu_initialization': original_cpu_init}  # 'megatron_amp_O2': original_amp_o2
    if amp_o2_key is not None:
        restore_dict[amp_o2_key] = original_amp_o2
    if original_sequence_parallel is not None:
        restore_dict['sequence_parallel'] = original_sequence_parallel

    return cfg, restore_dict


def restore_model_config(cfg, original_dict):
    with open_dict(cfg):
        for key, val in original_dict.items():
            logging.info(f"Restoring model config key ({key}) from {cfg[key]} to original value of {val}")
            cfg[key] = val
    return cfg


#################
### Utilities ###
#################


def compute_tp_splits(
    param_name, param, partitions, global_idx, tp_size, pp_size, pp_rank, pp_split_rank, megatron_legacy, model_cfg
):
    """
    Function to compute the splits required for tensor-parallelism.

    Args:
        param_name: Name of the current parameter of the current model (TP X PP Y)
        param: Value of the current parameter of the current model (TP X PP Y)
        partitions: Partitions of the flattened parameter of the current model (TP 1 PP 1)
        global_idx: The index used to select the parameter in the global partition.
        tp_size: Int, tensor-parallelism size.
        pp_size: Int, pipeline-parallelism size.
        pp_rank: Int, pipeline-parallelism rank.
        pp_split_rank: Int, pipeline-parallelism split rank. This should be > 1 if TP is being used with EncDec models (T5)
        megatron_legacy: Bool, whether the model is a legacy Megatron model or not.
        model_cfg: The model config as a OmegaConf DictConfig.

    Returns:
        List of torch tensors, each of which is a split of the current parameter.
    """
    # alias the global index to idx
    idx = global_idx

    fast_glu_activation = str(model_cfg.get('activation', '')).lower() in ['fast-geglu', 'fast-swiglu', 'fast-reglu']

    if param.shape == partitions[0][idx].shape:
        split = [partitions[0][idx].data] * tp_size
        logging.debug(">> Perfect match, no splitting needed")
    elif param.shape[0] == partitions[0][idx].shape[0]:
        split = torch.split(partitions[0][idx].data, param.shape[-1], dim=-1)
    else:
        # For T5-converted weights, the splitting needs to be strided such that q,k,v weights are bunched together on each tensor-parallel rank.
        if '.query_key_value.' in param_name and megatron_legacy:  # weight or bias
            split_dim = partitions[0][idx].data.shape[0]
            if split_dim % (tp_size * 3) != 0:
                raise ValueError(
                    f"Can not split Q,K,V parameter {param_name} with shape {param.shape} into tensor parallel size {tp_size}. Not divisible by {tp_size * 3}."
                )
            tp_qkv_splits = torch.chunk(partitions[0][idx].data, tp_size * 3, dim=0)
            split = []
            for i in range(tp_size):
                tp_qkv = torch.cat([tp_qkv_splits[item] for item in range(i, tp_size * 3, tp_size)])
                split.append(tp_qkv)
        elif '.key_value.' in param_name and megatron_legacy:  # weight or bias
            split_dim = partitions[0][idx].data.shape[0]
            if split_dim % (tp_size * 2) != 0:
                raise ValueError(
                    f"Can not split K,V parameter {param_name} with shape {param.shape} into tensor parallel size {tp_size}. Not divisible by {tp_size * 2}."
                )
            tp_qkv_splits = torch.chunk(partitions[0][idx].data, tp_size * 2, dim=0)
            split = []
            for i in range(tp_size):
                tp_qkv = torch.cat([tp_qkv_splits[item] for item in range(i, tp_size * 2, tp_size)])
                split.append(tp_qkv)
        elif ('dense_h_to_4h' in param_name or 'linear_fc1' in param_name) and fast_glu_activation:
            # For Megatron GPT model with Fast Glu activation
            # Handle gated linear units
            # concat all the first halves ('W's) and all the second halves ('V's)
            w_split, k_split = torch.chunk(partitions[0][idx].data, 2, dim=0)
            w_split = torch.chunk(w_split, tp_size, dim=0)
            k_split = torch.chunk(k_split, tp_size, dim=0)
            split = [torch.cat(weights, dim=0) for weights in zip(w_split, k_split)]  # split per tp rank

        # Regular split for Megatron and NeMo-Megatron models.
        else:
            split = torch.split(partitions[0][idx].data, param.shape[0], dim=0)

    return split


def compute_tp_merge(idx, name, param, partitions_pp, model_cfg):
    """
    Function to compute the partition merge required for tensor-parallelism.

    Args:
        idx:  The index used to select the parameter in the current pipeline partition.
        name:
        param: The parameter to be merged under TP 1 PP 1.
        partitions_pp: List of all TP partitions of the flattened parameter of the current model for a given PP rank
            (TP X PP Y). Indexed as partitions_pp[tp_rank][idx].
        model_cfg: The model config as an OmegaConf DictConfig.

    Returns:
        The concatenated parameter for TP 1 PP 1.
    """
    fast_glu_activation = str(model_cfg.get('activation', '')).lower() in ['fast-geglu', 'fast-swiglu', 'fast-reglu']

    # Logic from original TP rank change
    if param.shape == partitions_pp[0][idx].shape:
        concated = partitions_pp[0][idx].data
    elif param.shape[0] == partitions_pp[0][idx].shape[0]:
        concated = torch.cat([partitions_pp[i][idx].data for i in range(len(partitions_pp))], dim=-1)
    else:
        concated = torch.cat([partitions_pp[i][idx].data for i in range(len(partitions_pp))], dim=0)

    # Logic for Fast Glu activation
    if 'dense_h_to_4h' in name and fast_glu_activation:
        # concat all the first halves ('W's) and all the second halves ('V's)
        wk_splits = []
        for tpr in range(len(partitions_pp)):
            wk_splits.append(torch.chunk(partitions_pp[tpr][idx].data, 2, dim=0))

        w_split = torch.cat([w[0] for w in wk_splits], dim=0)
        k_split = torch.cat([w[1] for w in wk_splits], dim=0)
        concated = torch.cat([w_split, k_split], dim=0)

    # Trim padding
    if concated.shape != param.shape:
        logging.info(
            f"Warning: Shape mismatch for parameter {name} required shape: {param.shape}, merged shape: {concated.shape}. Narrowing to match required size."
        )
        if concated.shape[1:] == param.shape[1:]:
            concated = torch.narrow(concated, 0, 0, param.shape[0])
        elif concated.shape[:-1] == param.shape[:-1]:
            concated = torch.narrow(concated, -1, 0, param.shape[-1])
        else:
            raise RuntimeError(
                f"Can not handle parameter {name}, required shape: {param.shape}, merged shape: {concated.shape}."
            )
    return concated


def write_tp_pp_split(model, splits, app_state, tp_size, pp_rank, write_path):
    """
    Function to write the given TP PP split to NeMo File.

    Save each of the TP ranks in reverse order
    This is done so that the last PP rank will save the last TP rank only after all other PP TP ranks are saved
    The final rank will then save a new NeMo file with all other ranks inside.

    Args:
        model: The model corresponding to the current TP PP split. Contains partial parameters.
        splits: Nested List of tensors containing the TP splits of the current model given current PP rank.
            Indexed as splits[idx][tp_rank].
        app_state: AppState object.
        tp_size:  The global tensor-parallel size of the final model.
        pp_rank: The local pipeline parallel rank of the final model.
        write_path: The path to save the NeMo file.
    """
    for tp_rank in range(tp_size - 1, -1, -1):
        app_state.pipeline_model_parallel_rank = pp_rank
        app_state.tensor_model_parallel_rank = tp_rank

        idx = 0
        for name, param in model.named_parameters():
            split_val = splits[idx][tp_rank].clone()

            if param.shape != split_val.shape:
                logging.info(
                    f"Warning: Shape mismatch for parameter {name} required shape: {param.shape}, split shape: {split_val.shape}. Padding to match required size."
                )

                if split_val.shape[1:] == param.shape[1:]:
                    pad = [0, 0] * len(split_val.shape)
                    pad[-1] = param.shape[0] - split_val.shape[0]
                    split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                elif split_val.shape[:-1] == param.shape[:-1]:
                    pad = [0, param.shape[-1] - split_val.shape[-1]]
                    split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                else:
                    raise RuntimeError(
                        f"Can not handle parameter {name}, required shape: {param.shape}, split shape: {split_val.shape}."
                    )

            param.data = split_val
            idx += 1

        if write_path is not None:
            logging.info(f"Writing pp rank {pp_rank} tp rank {tp_rank} to file {write_path}")
            model.save_to(write_path)


def debug_log_split_param_diff(idx, param, param_name, partitions):
    # Log some useful comparison of tensors that are being mapped.
    # Note that the global param index for layers and modules may be different but the shapes
    # and semantics of the layer should match.
    logging.debug(f"Index: {idx} Model Params : {param_name} - {param.shape}")
    logging.debug(f"Index: {idx} Global params: {partitions[1][idx]} - {partitions[0][idx].shape}")


################
### Handlers ###
################


class GPTHandler:
    def __init__(self, megatron_legacy: bool):
        self.duplicate_gpt_word_embedding_offset = 0
        self.untied_gpt_embedding = False
        self.megatron_legacy = megatron_legacy

    def compute_split_index(self, model, idx, tp_rank, pp_rank, pp_split_rank, tp_size, pp_size):
        if pp_rank == (pp_size - 1) and hasattr(model, 'model') and hasattr(model.model, 'word_embeddings'):
            # duplicate embedding copy (tied weights)
            self.duplicate_gpt_word_embedding_offset = 1

        if model.cfg.get('share_embeddings_and_output_weights', True) is False:
            self.untied_gpt_embedding = True

        if self.duplicate_gpt_word_embedding_offset > 0:
            logging.info(f"GPT duplicate_gpt_word_embedding_offset: {self.duplicate_gpt_word_embedding_offset}")

        return idx + self.duplicate_gpt_word_embedding_offset

    def compute_splits(self, model, partitions, idx, tp_rank, pp_rank, pp_split_rank, tp_size, pp_size):
        splits = []

        # This is the PP X TP Y model with partial parameters present in correct order.
        # We need to extract the parameters from the global map in reverse order to fill in the
        # parameters of this model in forward order.
        for param_name, param in model.named_parameters():

            # Since we are moving forward, we may reach the end of the global map
            # but GPT has an additional word embedding as its last parameter
            # Therefore we check for this, and reset the index to the parameter of the PP 0 TP 0 rank
            # which holds the parameters of the embedding.
            if idx == (len(partitions[0])) and self.duplicate_gpt_word_embedding_offset > 0:
                logging.info("Found duplicate embedding copy for GPT model, resetting index")
                idx = 0  # reset idx parameter to 0 if we have duplicate embedding copy

            debug_log_split_param_diff(idx, param, param_name, partitions)

            # Tensor Parallel Splitting
            split = compute_tp_splits(
                param_name,
                param,
                partitions,
                idx,
                tp_size,
                pp_size,
                pp_rank,
                pp_split_rank,
                self.megatron_legacy,
                model.cfg,
            )

            splits.append(split)
            idx += 1

        return idx, splits

    def compute_split_offset(self, offset_diff, tp_rank, pp_rank, pp_split_rank, tp_size, pp_size):
        # GPT offset correction
        if not self.untied_gpt_embedding and pp_size > 1 and pp_rank == (pp_size - 1) and pp_split_rank == 0:
            offset_diff += 1

        return offset_diff


class T5Handler:
    def __init__(self, megatron_legacy: bool):
        self.shared_enc_dec_embeddings = False
        self.shared_enc_dec_embeddings_intermediate = False
        self.enc_dec_share_token_embeddings_count = 0
        self.intermediate_shared_embedding_location = -1
        self.megatron_legacy = megatron_legacy

    def compute_split_index(self, model, idx, tp_rank, pp_rank, pp_split_rank, tp_size, pp_size):
        final_idx = idx

        # Special case for T5 models - where the embeddings are shared between encoder and decoder
        # and the rank of decoder split is arbitrary.
        # Megatron T5 check for pipeline_model_parallel_split_rank in order to inject encoder embeddings
        self.shared_enc_dec_embeddings = (
            pp_split_rank > 0 and pp_split_rank == pp_rank and model.cfg.get('share_token_embeddings', True)
        )
        # If embedding sharing is active, both vocab and position embeddings are shared
        if self.shared_enc_dec_embeddings:
            self.enc_dec_share_token_embeddings_count = 2
        else:
            self.enc_dec_share_token_embeddings_count = 0

        # Start to calculate new idx
        final_idx = final_idx + self.enc_dec_share_token_embeddings_count

        # Special case for T5 models - where the embeddings are shared between encoder and decoder
        # For all decoder ranks which are not the pp_split_rank, we need to inject the vocab embeddings only at
        # an intermediate location of the model (usually second last location).
        # Megatron T5 check for pipeline_model_parallel_split_rank in order to inject encoder embeddings
        # when the pipeline_model_parallel_split_rank is not the last PP rank
        self.shared_enc_dec_embeddings_intermediate = (
            pp_split_rank > 0
            and pp_split_rank < pp_size
            and hasattr(model, 'enc_dec_model')
            and hasattr(model.enc_dec_model, 'word_embeddings')
        )

        if self.shared_enc_dec_embeddings_intermediate:
            # Loop until we get the location of this tensor
            self.intermediate_shared_embedding_location = -1
            for param_name, param in model.named_parameters():  # special case for T5
                if param_name == 'enc_dec_model.word_embeddings.weight':
                    self.intermediate_shared_embedding_location += 1
                    break
                self.intermediate_shared_embedding_location += 1
        else:
            self.intermediate_shared_embedding_location = -1

        # Re-evaluate the intermediate shared embedding flag
        self.shared_enc_dec_embeddings_intermediate = self.shared_enc_dec_embeddings_intermediate and (
            self.intermediate_shared_embedding_location >= 0
        )
        # If module is present, add a module offset to the index
        if self.shared_enc_dec_embeddings_intermediate:
            final_idx += 1

        if self.enc_dec_share_token_embeddings_count:
            logging.info(f"EncDec share_token_embeddings_count: {self.enc_dec_share_token_embeddings_count}")
        if self.shared_enc_dec_embeddings_intermediate:
            logging.info(
                f"EncDec share_enc_dec_embeddings_intermediate: {self.intermediate_shared_embedding_location}"
            )

        return final_idx

    def compute_splits(self, model, partitions, idx, tp_rank, pp_rank, pp_split_rank, tp_size, pp_size):
        splits = []

        # Backup index when EncDec models reset the index to fill in the first embedding matrices (when pp split rank == pp rank)
        computed_index = idx

        # This is the PP X TP Y model with partial parameters present in correct order.
        # We need to extract the parameters from the global map in reverse order to fill in the
        # parameters of this model in forward order.
        for param_name, param in model.named_parameters():

            # Since we are moving forward, we may reach the end of the global map
            # but T5 has an additional word embedding as its first two parameter when pp split rank == pp rank
            # Therefore we check for this, and update the index to the parameter of the PP 0 TP 0 rank
            # which holds the parameters of the embedding.
            if self.enc_dec_share_token_embeddings_count:
                logging.info("EncDec models decoder shares embedding with encoder, resetting index")
                idx = (
                    2 - self.enc_dec_share_token_embeddings_count
                )  # 0th index is vocab embedding, 1 is pos embedding, 2 is embedding count

            # Since we are moving forward, we may reach the end of the global map
            # but T5 has an additional word embedding as randomly located in the decoder when
            # when pp rank > pp_split_rank.
            # Therefore we check for this, and skip the parameter of the current TP X PP Y module
            # and fill this parameter later.
            if self.shared_enc_dec_embeddings_intermediate and param_name == 'enc_dec_model.word_embeddings.weight':
                logging.info(
                    "EncDec models decoder shares embedding with encoder in intermediate pos, skipping module for later update"
                )
                continue

            debug_log_split_param_diff(idx, param, param_name, partitions)

            # Tensor Parallel Splitting
            split = compute_tp_splits(
                param_name,
                param,
                partitions,
                idx,
                tp_size,
                pp_size,
                pp_rank,
                pp_split_rank,
                self.megatron_legacy,
                model.cfg,
            )

            splits.append(split)
            idx += 1

            # When pp split rank is equal to current pp rank, we need to first inject the encoder embeddings
            # and then reset the index to the originally computed index
            if self.enc_dec_share_token_embeddings_count > 0:
                if self.enc_dec_share_token_embeddings_count - 1 == 0:
                    idx = computed_index

                self.enc_dec_share_token_embeddings_count -= 1

        # Inject the EncDec shared embeddings intermediate tensor
        # at one random location in the decoder of this TP PP rank.
        # Note that usually it is the second last tensor, but to avoid specific index we search for it
        # again.
        if self.shared_enc_dec_embeddings_intermediate:
            for param_name, param in model.named_parameters():
                if param_name == 'enc_dec_model.word_embeddings.weight':
                    logging.info("Found intermediate shared embedding, injecting")
                    split = compute_tp_splits(
                        param_name,
                        param,
                        partitions,
                        global_idx=0,
                        tp_size=tp_size,
                        pp_size=pp_size,
                        pp_rank=pp_rank,
                        pp_split_rank=pp_split_rank,
                        megatron_legacy=self.megatron_legacy,
                        model_cfg=model.cfg,
                    )
                    splits.insert(self.intermediate_shared_embedding_location, split)
                    break

        return idx, splits

    def compute_split_offset(self, offset_diff, tp_rank, pp_rank, pp_split_rank, tp_size, pp_size):
        # T5 offset correction for shared embedding when pp split rank == pp rank
        if self.shared_enc_dec_embeddings:
            offset_diff += 2

        # T5 offset correction for intermediate shared embedding when pp rank > pp split rank
        if self.shared_enc_dec_embeddings_intermediate:
            offset_diff += 1

        return offset_diff


##################
### Converters ###
##################


def merge_partition(model, partitions: Dict[int, List[List[torch.Tensor]]], write_path: str = None):
    # Extract the pp_rank and number of modules per tp rank in each pp rank
    pp_ranks = list(partitions.keys())
    pp_lens = []
    for pp_rank in pp_ranks:
        partition_pp = partitions[pp_rank]
        max_len = max([len(x) for x in partition_pp])  # Perform max as we need to iterate through all modules
        pp_lens.append(max_len)

    total_params_merged = len([p for p in model.parameters()])
    pp_total_len = sum(pp_lens)
    logging.info(f"Total layers in Merged Model: {total_params_merged}")

    og_pp_split_rank = 0
    if pp_total_len > total_params_merged:
        og_pp_split_rank = model.cfg.get('pipeline_model_parallel_split_rank', 0)

    idx = 0
    pp_rank = 0
    global_idx = 0

    # During merge - model is TP 1 PP 1 model with all parameters present in correct order.
    # Merge the parameters of the various PP X TP Y models into the TP 1 PP 1 model.
    for name, param in model.named_parameters():
        # Since the PP ranks each contain the list of all their TP rank parameters
        # We need to detect if we need to move to the next PP rank when we run out of tensors in current PP rank
        # Reset the index so that it indexes the new pp rank tensor list correctly
        if idx >= pp_lens[pp_rank]:
            pp_rank += 1
            idx = 0

            # For EncDec models, after the encoder-decoder PP split occurs,
            # the vocab and positional embeddings are duplicated across the PP ranks at the
            # beginning of the decoder rank. We can skip them during the merge step.
            if pp_total_len > total_params_merged:
                if og_pp_split_rank > 0 and og_pp_split_rank == pp_rank:
                    logging.info(
                        f"Skipping duplicate vocab and positional embeddings for EncDec model "
                        f"at the pp split rank: {og_pp_split_rank}"
                    )
                    idx += 2

        # For EncDec models, after the pp split occurs, final pp rank of the decoder
        # has an intermediate embedding tensor at the penultimate positon, skip that.
        if og_pp_split_rank > 0 and global_idx == total_params_merged - 1:
            logging.info(
                f"Skipping intermediate embedding tensor for EncDec model at the final pp split "
                f"rank: {og_pp_split_rank}",
            )
            idx = pp_lens[pp_rank] - 1

        # Extract all TP ranks out of current PP rank
        partitions_pp = partitions[pp_rank]

        logging.debug(
            f"Global idx: {global_idx} Index: {idx} Model Param: {name} "
            f"Partition Params: {[p[idx].shape for p in partitions_pp]}"
        )

        # Original TP rank change logic
        concated = compute_tp_merge(idx, name, param, partitions_pp, model.cfg)

        # Update the model parameter with the merged tensor
        param.data = concated
        idx += 1
        global_idx += 1

    # Save the file iff the original file was PP 1 TP 1
    if write_path is not None:
        model.save_to(write_path)


def split_partition(
    model,
    partitions,
    pp_size: int,
    tp_size: int,
    pp_rank: int,
    offset: int,
    pp_split_rank: int = 0,
    write_path: str = None,
    megatron_legacy: bool = False,
):
    if len(partitions) != 2:
        raise ValueError(
            "Can only split partitions of model with TP=1. For partitions of models with TP>1, merge first."
        )

    if tp_size < 1:
        raise ValueError("TP size must to be >= 1.")

    if pp_size < 1:
        raise ValueError("PP size must to be >= 1.")

    # Setup app state to mimic current PP and TP ranks with single merged module
    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = pp_size
    app_state.tensor_model_parallel_size = tp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    # Go in reverse for TP order, as PP 0 TP 0 will merge all preceding files
    app_state.pipeline_model_parallel_rank = pp_rank
    app_state.tensor_model_parallel_rank = tp_size - 1

    # Compute reverse offset of parameter index from global map
    num_params = sum([1 for _ in model.parameters()])  # Count number of parameters iteratively
    idx = offset - num_params + 1  # start index of current PP TP rank in global map

    assert (
        idx + num_params - 1 == offset
    ), f"idx = {idx}, num_params = {num_params}, sum = {idx + num_params}, offset = {offset}"

    # Special case for GPT models - whose last PP TP rank has a duplicate embedding tensor

    if 'gpt' in model.cfg.target.lower():
        logging.info("Splitting GPT model")
        handler = GPTHandler(megatron_legacy=megatron_legacy)

    elif 't5' in model.cfg.target.lower():
        logging.info("Splitting T5 model")
        handler = T5Handler(megatron_legacy=megatron_legacy)

    else:
        raise ValueError(f"Unsupported model for Pipeline Parallelism change - {model.cfg.target}")

    idx = handler.compute_split_index(model, idx, 0, pp_rank, pp_split_rank, tp_size, pp_size)

    # Print some debug info
    logging.info(f"Start Layer Idx: {idx} Number of layers in current rank: {num_params} Offset: {offset}")
    logging.info("\n")

    # Split the model's parameters according to TP PP ranks
    idx, splits = handler.compute_splits(model, partitions, idx, 0, pp_rank, pp_split_rank, tp_size, pp_size)

    # Compute the new offset for the next PP rank in reverse order
    # Add 1 to offset to account for last PP rank's duplicated Embedding
    offset_diff = offset - num_params
    offset_diff = handler.compute_split_offset(offset_diff, 0, pp_rank, pp_split_rank, tp_size, pp_size)

    # Finalize the new offset
    new_offset = offset_diff

    # Save each of the TP ranks in reverse order
    # This is done so that the last PP rank will save the last TP rank only after all other PP TP ranks are saved
    # The final rank will then save a new NeMo file with all other ranks inside.
    write_tp_pp_split(model, splits, app_state, tp_size, pp_rank, write_path)

    return new_offset


def split_tp_partition_only(model, partitions, tp_size, write_path=None, megatron_legacy=False):
    if len(partitions) != 2:
        raise ValueError(
            "Can only split partitions of model with TP=1. For partitions of models with TP>1, merge first."
        )

    if tp_size < 1:
        raise ValueError("TP size must to be >= 1.")

    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = 1
    app_state.tensor_model_parallel_size = tp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    app_state.pipeline_model_parallel_rank = 0
    app_state.tensor_model_parallel_rank = tp_size - 1

    idx = 0
    splits = []
    for param_name, param in model.named_parameters():
        split = compute_tp_splits(
            param_name,
            param,
            partitions,
            idx,
            tp_size,
            pp_size=1,
            pp_rank=0,
            pp_split_rank=0,
            megatron_legacy=megatron_legacy,
            model_cfg=model.cfg,
        )
        splits.append(split)
        idx += 1

    # Save each of the TP ranks in reverse order
    # This is done so that the last PP rank will save the last TP rank only after all other PP TP ranks are saved
    # The final rank will then save a new NeMo file with all other ranks inside.
    write_tp_pp_split(model, splits, app_state, tp_size, pp_rank=0, write_path=write_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default=None, required=False, help="Path to source .nemo file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to write target .nemo file")
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=-1, required=False, help="TP size of source model"
    )
    parser.add_argument("--target_tensor_model_parallel_size", type=int, required=True, help="TP size of target model")
    parser.add_argument(
        '--pipeline_model_parallel_size', type=int, default=-1, required=False, help='PP size of source model'
    )
    parser.add_argument(
        '--target_pipeline_model_parallel_size', type=int, required=True, help='PP size of target model'
    )
    parser.add_argument(
        '--target_pipeline_model_parallel_split_rank', type=int, default=0, help='PP rank to split for Enc-Dec models'
    )
    parser.add_argument(
        '--virtual_pipeline_model_parallel_size', type=int, default=None, help='Virtual Pipeline parallelism size'
    )
    parser.add_argument(
        '--ckpt_name', type=str, default=None, help='Checkpoint name to load from for Virtual Parallel'
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel",
        help="NeMo model class. This script should support all NeMo megatron models that use Tensor Parallel",
    )
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")
    parser.add_argument('--num_gpu_per_node', default=8, type=int, help='Number of GPUs per node')
    parser.add_argument(
        "--megatron_legacy",
        action="store_true",
        help="Converter for legacy megatron modles that have different q,k,v weight splits",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        required=False,
        default=None,
        help="Path to the tokenizer model path if your model uses a tokenizer model as an artifact. This is needed if your model uses a sentencepiece tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_vocab_file",
        type=str,
        required=False,
        default=None,
        help="Path to the tokenizer model path if your model uses a tokenizer model as an artifact. This is needed if your model uses a sentencepiece tokenizer.",
    )
    parser.add_argument('--hparams_file', type=str, default=None, help='Path to hparams file from PTL training')
    parser.add_argument('--tp_conversion_only', action='store_true', help='Only convert TP model to TP model')
    parser.add_argument('--model_extracted_dir', type=str, default=None, help='Path to pre-extracted model directory')

    args = parser.parse_args()

    precision = args.precision
    num_gpu_per_node = int(args.num_gpu_per_node)
    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))

    if precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            pass
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = precision[2:]

    if precision == 32:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # fallback

    # Built target directory if it does not exist
    target_dir = os.path.split(args.target_file)[0]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    tp_size = args.tensor_model_parallel_size
    tgt_tp_size = args.target_tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    tgt_pp_size = args.target_pipeline_model_parallel_size
    pipeline_model_parallel_split_rank = args.target_pipeline_model_parallel_split_rank
    vp_size = args.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    convert_vp = vp_size > 1
    if convert_vp:
        from megatron.core import parallel_state

        parallel_state.set_virtual_pipeline_model_parallel_world_size(vp_size)

        hparams_filepath = args.hparams_file
        if hparams_filepath is None:
            logging.warning(
                '\n\n\n!!!!!!!!!\n'
                'You are converting a model with virtual pipeline parallelism enabled, \n'
                'but have not passed `hparams_file` argument. \n'
                'This will cause each ckpt file to be temporarily laoded onto GPU memory!\n\n'
                'It is highly recommended to pass `hparams_file` argument to avoid this.\n'
            )
    else:
        hparams_filepath = None

    # Import the class of the model
    cls = model_utils.import_class_by_path(args.model_class)

    if args.model_file is None and args.model_extracted_dir is None:
        raise ValueError("Cannot pass model_file and model_extracted_dir as None at the same time.")

    tmp_cfg = cls.restore_from(
        restore_path=args.model_file,
        trainer=Trainer(devices=1, strategy=NLPDDPStrategy(), accelerator="cpu", precision=precision),
        map_location=torch.device("cpu"),
        return_config=True,
    )
    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=tmp_cfg.get('native_amp_init_scale', 2 ** 32),
                growth_interval=tmp_cfg.get('native_amp_growth_interval', 1000),
                hysteresis=tmp_cfg.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if tmp_cfg.get('megatron_amp_O2', False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        # Set precision None after precision plugins are created as PTL >= 2.1 does not allow both
        # precision plugins and precision to exist
        precision = None
    trainer = Trainer(plugins=plugins, devices=1, strategy=NLPDDPStrategy(), accelerator="cpu")

    if tp_size < 0 or pp_size < 0:
        logging.info(f"Loading model config from {args.model_file} to get TP and PP size")
        model_config_internal = cls.restore_from(
            restore_path=args.model_file, trainer=trainer, map_location=torch.device("cpu"), return_config=True,
        )

        tp_size = model_config_internal.get('tensor_model_parallel_size', 1)
        pp_size = model_config_internal.get('pipeline_model_parallel_size', 1)

    # Check if TP conversion only
    tp_conversion_only = args.tp_conversion_only
    if tp_conversion_only:
        logging.info("Converting TP model to TP model only")

        if pp_size > 1:
            raise ValueError("Provided `--tp_conversion_only` but `--pipeline_model_parallel_size` > 1")

        if tgt_pp_size > 1:
            raise ValueError("Provided `--tp_conversion_only` but `--target_pipeline_model_parallel_size` > 1")

        if pipeline_model_parallel_split_rank > 0:
            raise ValueError("Provided `--tp_conversion_only` but `--target_pipeline_model_parallel_split_rank` > 0")

        # Force PP size to 1
        pp_size = 1
        tgt_pp_size = 1
        pipeline_model_parallel_split_rank = 0

    if vp_size is None or vp_size < 0:
        vp_size = 1

    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = pp_size
    app_state.tensor_model_parallel_size = tp_size

    if vp_size > 1:
        app_state.virtual_pipeline_model_parallel_size = vp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    world_size = pp_size * tp_size  # pseudo world size for simulating load of a specific rank on a single gpu

    app_state.tensor_model_parallel_rank = 0
    app_state.pipeline_model_parallel_rank = 0

    if vp_size > 1:
        set_virtual_parallel_rank_safely(0)

    # Extract tokenizer artifact from the model to temp directory
    logging.info("Extracting tokenizer artifact from NeMo file...")
    temp_dir = tempfile.mkdtemp()
    tokenizer_model_path = None
    with tarfile.open(args.model_file, "r") as tar:
        for member in tar.getmembers():
            if '.model' in member.name:
                extracted_file = tar.extractfile(member)
                extracted_file_path = os.path.join(temp_dir, member.name)

                if tokenizer_model_path is None:
                    logging.info(f"Found tokenizer. Extracting {member.name} to {extracted_file_path}")

                    tokenizer_model_path = extracted_file_path
                    with open(extracted_file_path, "wb") as f:
                        f.write(extracted_file.read())
                else:
                    if args.tokenizer_model_path is None:
                        logging.warning(
                            f"\n\nFound multiple tokenizer artifacts in the model file.\n"
                            f"Using only {tokenizer_model_path}.\n"
                            f"If this is incorrect, manually pass the correct tokenizer using "
                            f"`--tokenizer_model_path`.\n\n"
                        )

    # If input model has TP > 1 or PP > 1
    # Reconstruct the model to have TP = 1 and PP = 1
    # Note that this is a forward loop that will process PP [0..N] TP [0..M] in sequential order.
    if tp_size > 1 or pp_size > 1:
        partitions = {}  # 3d list of VP x PP x TP
        model = None

        # Build partitions structure
        for vp_idx in range(vp_size):
            partitions[vp_idx] = []  # Build first layer - VP

            for pp_idx in range(pp_size):
                # For each VP, build PP x TP holder
                partitions[vp_idx].append({})
                partitions[vp_idx][pp_idx] = []

        for vp_rank in range(vp_size):
            if vp_size > 1:
                set_virtual_parallel_rank_safely(vp_rank)

            for pp_rank in range(pp_size):
                app_state.pipeline_model_parallel_rank = pp_rank

                for tp_rank in range(tp_size):
                    app_state.tensor_model_parallel_rank = tp_rank

                    logging.info(f"Loading ------------ PP Rank: {pp_rank} TP Rank: {tp_rank}")

                    # Override flag that forces Model to use AppState instead of Trainer
                    # to determine the world size, global and local rank
                    # Used for simulating load of a specific rank on a single gpu
                    os.environ[NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE] = "true"

                    # Compute the global rank to load the correct subset of parameters
                    global_rank = pp_rank * tp_size + tp_rank

                    # Update AppState
                    app_state.world_size = world_size
                    app_state.global_rank = global_rank
                    app_state.local_rank = global_rank % num_gpu_per_node
                    app_state.pipeline_model_parallel_size = pp_size
                    app_state.tensor_model_parallel_size = tp_size
                    app_state.pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank
                    app_state.model_parallel_size = (
                        app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
                    )

                    if vp_size > 1:
                        set_virtual_parallel_rank_safely(vp_rank)

                    if vp_rank == 0:
                        save_restore_connector = NLPSaveRestoreConnector()

                        if args.model_extracted_dir is not None:
                            logging.info(f"Using extracted model directory: {args.model_extracted_dir}")
                            save_restore_connector.model_extracted_dir = args.model_extracted_dir

                        if args.model_file is not None:
                            model_filepath = args.model_file
                        else:
                            model_filepath = args.model_extracted_dir

                        if vp_size == 1:

                            # Get model config
                            tmp_cfg = cls.restore_from(
                                restore_path=model_filepath,
                                trainer=trainer,
                                map_location=torch.device("cpu"),
                                save_restore_connector=save_restore_connector,
                                return_config=True,
                            )

                            # Force model onto CPU
                            tmp_cfg, restore_dict = force_cpu_model(tmp_cfg)

                            # Restore model
                            model = cls.restore_from(
                                restore_path=model_filepath,
                                trainer=trainer,
                                map_location=torch.device("cpu"),
                                save_restore_connector=save_restore_connector,
                                override_config_path=tmp_cfg,
                            )
                            model.freeze()

                            # Restore model config
                            restore_model_config(model.cfg, restore_dict)

                        else:
                            if args.ckpt_name is None:
                                raise ValueError(
                                    "For Virtual Parallel, ckpt name is required.\n"
                                    "Please provide `--ckpt_name` argument."
                                )

                            # inject model parallel rank
                            checkpoint_path = model_utils.inject_model_parallel_rank(
                                os.path.join(model_filepath, args.ckpt_name)
                            )

                            vp_state_dict = torch.load(checkpoint_path, map_location="cpu")

                            if hparams_filepath is not None:
                                # Force the model onto CPU
                                tmp_cfg = OmegaConf.load(hparams_filepath)
                                tmp_cfg, restore_dict = force_cpu_model(tmp_cfg)

                                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.yml') as tmp:
                                    OmegaConf.save(tmp_cfg, tmp, resolve=True)
                                    tmp.seek(0)

                                    model = cls.load_from_checkpoint(
                                        checkpoint_path=checkpoint_path,
                                        trainer=trainer,
                                        map_location=torch.device("cpu"),
                                        hparams_file=tmp.name,
                                    )
                                    model.freeze()

                                    restore_model_config(model.cfg, restore_dict)

                            else:
                                model = cls.load_from_checkpoint(
                                    checkpoint_path=checkpoint_path, trainer=trainer, map_location=torch.device("cpu"),
                                )
                                model.freeze()

                        model.to(dtype=dtype)

                        # Reset env flag
                        os.environ.pop(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, None)

                        logging.info(
                            f"<<<<<<<< LOADED MODEL PP={pp_rank + 1} TP={tp_rank + 1} | "
                            f"GLOBAL RANK = {global_rank} >>>>>>>>>"
                        )

                        # Save the parameters
                        if vp_size == 1:
                            params = [p for p in model.parameters()]
                            partitions[vp_rank][pp_rank].append(params)  # vp_rank = 0

                        else:
                            vp_params_tmp = []
                            for vp_idx in range(vp_size):
                                set_virtual_parallel_rank_safely(vp_idx)
                                vp_params = vp_state_dict[f'model{vp_idx}']
                                model.model[vp_idx].module.load_state_dict(vp_params, strict=True)
                                model.model[vp_idx].module.to('cpu')
                                params = [p for p in model.model[vp_idx].module.parameters()]
                                vp_params_tmp.append(params)
                                # partitions[pp_rank][vp_idx].append(params)

                            for vp_idx in range(vp_size):
                                partitions[vp_idx][pp_rank].append(vp_params_tmp[vp_idx])

                            del vp_params_tmp
                            set_virtual_parallel_rank_safely(0)

                        # app_state is being updated incorrectly during restore
                        app_state.data_parallel_rank = 0
                        app_state.pipeline_model_parallel_rank = pp_rank
                        app_state.tensor_model_parallel_rank = tp_rank
                        app_state.pipeline_model_parallel_size = pp_size
                        app_state.tensor_model_parallel_size = tp_size
                        app_state.model_parallel_size = (
                            app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
                        )

                        if vp_size > 1:
                            app_state.virtual_pipeline_model_parallel_size = vp_size
                            set_virtual_parallel_rank_safely(vp_rank)

        # Build a unified model with PP 1 TP 1
        with open_dict(model.cfg):
            model.cfg.tensor_model_parallel_size = 1
            model.cfg.pipeline_model_parallel_size = 1
            model.cfg.virtual_pipeline_model_parallel_size = None

        app_state.global_rank = 0
        app_state.local_rank = 0
        app_state.data_parallel_rank = 0
        app_state.pipeline_model_parallel_rank = 0
        app_state.tensor_model_parallel_rank = 0
        app_state.pipeline_model_parallel_size = 1
        app_state.tensor_model_parallel_size = 1
        app_state.model_parallel_size = 1

        if vp_size > 1:
            set_virtual_parallel_rank_safely(None)

        trainer = Trainer(plugins=plugins, devices=1, strategy=NLPDDPStrategy(), accelerator="cpu")

        with open_dict(model.cfg):
            if args.tokenizer_model_path is not None:
                model.cfg.tokenizer.model = args.tokenizer_model_path
            if args.tokenizer_vocab_file is not None:
                model.cfg.tokenizer.vocab_file = args.tokenizer_vocab_file

            model.cfg, restore_dict = force_cpu_model(model.cfg)

            # Remove Virtual Parallelism
            model.cfg.virtual_pipeline_model_parallel_size = None

        logging.info(f"<<<<<<<< Building TP 1 PP 1 base model >>>>>>>>>")
        model = cls(model.cfg, trainer)  # type: nn.Module
        model.freeze()
        model = model.to('cpu')
        model._save_restore_connector = NLPSaveRestoreConnector()

        restore_model_config(model.cfg, restore_dict)

        vp_param_count = 0
        for vp in range(vp_size):
            for pp in range(pp_size):
                for tp in range(tp_size):
                    vp_param_count += len(partitions[vp][pp][tp])

        if vp_size > 1:
            logging.debug(f"Total params in TP PP VP = 1 : {len(list(model.parameters()))}")
            logging.debug(f"Total params in VP PP TP (og): {vp_param_count}")

        # Flatten Virtual Pipeline
        if vp_size == 1:
            # unpack vp container, pack pp tp container
            partitions = partitions[0]
            partitions = {idx: val for idx, val in enumerate(partitions)}
        else:
            flat_partitions = {idx: [] for idx in range(pp_size)}

            """
            Under VP convention
            Notation :
            Stage  = PP rank
            Number = GPT model / layer index
            Ignore TP - every PP has all TP corresponding to that PP
            chunk_index = the physical index of any [] in the list. Ex idx = 2 in below map corresponds to [2: PP 0 VP 1]]


            For a PP 2 VP 4 model with 8 GPT layers-

            Indices
            # Stage 0: [0:PP 0 VP 0]  [2:PP 0 VP 1]  [4:PP 0 VP 2]  [6:PP 0 VP 3]
            # Stage 1: [1:PP 1 VP 0]  [3:PP 1 VP 1]  [5:PP 1 VP 2]  [7:PP 1 VP 3]

            after conversion will become

            # Stage 0: [0,1,2,3:PP 0]
            # Stage 1: [4,5,6,7:PP 1]

            """
            pp_index = 0
            chunk_counter = 0
            tp_cache = [[] for _ in range(tp_size)]

            for vp in range(vp_size):
                for pp in range(pp_size):
                    # Gather all TP under this VP PP combination.
                    # We will accumulate TP parameters from multiple layers in this cache.
                    for tp in range(tp_size):
                        tp_cache[tp].extend(partitions[vp][pp][tp])

                    # This counter indexes the global selection of a VP PP combination in the above map
                    chunk_counter += 1

                    # Log the mapping from old VP x PP to new PP index
                    logging.info(f"VP Conversion - vp: {vp} pp: {pp} -> pp_idx: {pp_index}")

                    # Every vp_size chunks, we can fill a new PP index in the flat_partitions
                    if chunk_counter % vp_size == 0:
                        flat_partitions[pp_index].extend(tp_cache)
                        tp_cache = [[] for _ in range(tp_size)]
                        pp_index += 1

                        logging.debug(
                            f"VP merge step: \n"
                            f"vp: {vp} pp: {pp} pp_idx: {pp_index - 1} "
                            f"len(flat_partitions): {len(flat_partitions[pp_index - 1])}"
                        )

            logging.debug(f"PP Size len(flat partitions) : {len(flat_partitions)}")
            logging.debug(f"TP Size len(flat partitions[0]): {len(flat_partitions[0])}")
            logging.debug(f"Layers  len(flat partitions[0][0]) : {len(flat_partitions[0][0])}")

            partitions = flat_partitions
            del tp_cache

        if tgt_tp_size > 1 or tgt_pp_size > 1:
            merge_partition(model, partitions)
        else:
            # Write out the PP 1 TP 1 model to disk
            merge_partition(model, partitions, args.target_file)

        # Empty cache memory of all parameters from all PP TP partitions
        partitions.clear()

    else:
        # If input model has TP = 1 and PP = 1
        app_state.model_parallel_size = 1

        save_restore_connector = NLPSaveRestoreConnector()

        if args.model_extracted_dir is not None:
            logging.info(f"Using extracted model directory: {args.model_extracted_dir}")
            save_restore_connector.model_extracted_dir = args.model_extracted_dir

        if args.model_file is not None:
            model_filepath = args.model_file
        else:
            model_filepath = args.model_extracted_dir

        tmp_cfg = cls.restore_from(
            restore_path=model_filepath,
            trainer=trainer,
            map_location=torch.device("cpu"),
            save_restore_connector=save_restore_connector,
            return_config=True,
        )

        tmp_cfg, restore_dict = force_cpu_model(tmp_cfg)

        model = cls.restore_from(
            restore_path=model_filepath,
            trainer=trainer,
            map_location=torch.device("cpu"),
            save_restore_connector=save_restore_connector,
            override_config_path=tmp_cfg,
        )
        model.to(dtype=dtype)

        restore_model_config(model.cfg, restore_dict)

    # If target model has TP > 1 or PP > 1
    if tgt_pp_size > 1 or tgt_tp_size > 1:

        # Preserve the TP 1 PP 1 model parameters and names
        global_params = []
        global_params.append([p for n, p in model.named_parameters()])  # params
        global_params.append([n for n, p in model.named_parameters()])  # names

        logging.debug("Global parameters:")
        for idx, (name, p) in enumerate(zip(global_params[1], global_params[0])):
            logging.debug(f"{name} - {p.shape}")

        logging.info(f"TP 1 PP 1 Number of Parameters : {len(global_params[0])}")

        world_size = (
            tgt_pp_size * tgt_tp_size
        )  # pseudo world size for simulating load of a specific rank on a single gpu
        new_global_batch_size = model.cfg.micro_batch_size * world_size
        old_global_batch_size = model.cfg.get('global_batch_size', model.cfg.micro_batch_size)

        global_offset = len(global_params[0]) - 1  # -1 cause this indexes the array, range [0, L-1]
        logging.info(f"Final layer offset for parameters: {global_offset}")

        for pp_rank in range(tgt_pp_size - 1, -1, -1):  # reverse order

            with open_dict(model.cfg):
                model.cfg.pipeline_model_parallel_size = tgt_pp_size
                model.cfg.tensor_model_parallel_size = tgt_tp_size

                if 'pipeline_model_parallel_split_rank' in model.cfg:
                    if pipeline_model_parallel_split_rank > 0:
                        model.cfg.pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank
                    elif pp_size > 1:
                        logging.warning(
                            f"Model config has `pipeline_model_parallel_split_rank` set to "
                            f"{model.cfg.pipeline_model_parallel_split_rank} and target PP "
                            f"size is {tgt_pp_size}. "
                            f"Provided `pipeline_model_parallel_split_rank` is "
                            f"{pipeline_model_parallel_split_rank}. "
                            f"Be careful that the model config is correct "
                            f"if encoder-decoder models are being converted."
                        )

                model.cfg.global_batch_size = old_global_batch_size  # Used for restoration

            # Override flag that forces Model to use AppState instead of Trainer
            # to determine the world size, global and local rank
            # Used for simulating load of a specific rank on a single gpu
            os.environ[NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE] = "true"

            # Compute the global rank
            global_rank = (
                pp_rank * tgt_tp_size + 0
            )  # tp_rank = 0 needed just for modules, all TP will be merged to this PP rank

            # Update AppState
            app_state.world_size = world_size
            app_state.global_rank = global_rank
            app_state.local_rank = global_rank % num_gpu_per_node
            app_state.pipeline_model_parallel_size = tgt_pp_size
            app_state.tensor_model_parallel_size = tgt_tp_size
            app_state.model_parallel_size = (
                app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
            )

            trainer = Trainer(plugins=plugins, devices=1, strategy=NLPDDPStrategy(), accelerator="cpu")
            if args.tokenizer_model_path is not None:
                with open_dict(model.cfg):
                    model.cfg.tokenizer.model = args.tokenizer_model_path

            else:
                if tokenizer_model_path is None:
                    logging.warning("Could not extract tokenizer model file from checkpoint.")

                else:
                    # Extract tokenizer info
                    with open_dict(model.cfg):
                        model.cfg.tokenizer.model = tokenizer_model_path

            model.cfg, restore_dict = force_cpu_model(model.cfg)

            model = cls(model.cfg, trainer)
            model = model.to('cpu')
            model._save_restore_connector = NLPSaveRestoreConnector()
            model.freeze()
            model.to(dtype=dtype)

            restore_model_config(model.cfg, restore_dict)

            # Update global batch size
            if old_global_batch_size % new_global_batch_size != 0 or old_global_batch_size < new_global_batch_size:
                logging.info(
                    f"Global batch size {old_global_batch_size} is not divisible by new global batch size {new_global_batch_size}."
                    f" The model config will be updated with new global batch size {new_global_batch_size}."
                )
                with open_dict(model.cfg):
                    model.cfg.global_batch_size = new_global_batch_size

            logging.info(f"Global rank: {global_rank} Local rank: {app_state.local_rank} World size: {world_size}")
            logging.info(f"PP rank: {pp_rank} TP rank: {0}")
            logging.info(f"TP 1 PP 1 Number of Layers : {len(global_params[0])}")
            logging.info(f"Remaining layer offset for parameters: {global_offset}")
            logging.info("\n")

            # Special case for TP conversion only mode
            if tp_conversion_only:
                logging.info(f"Skipping PP split due to flag `--tp_conversion_only`")

                split_tp_partition_only(model, global_params, tgt_tp_size, args.target_file, args.megatron_legacy)
                break

            global_offset = split_partition(
                model,
                global_params,
                tgt_pp_size,
                tgt_tp_size,
                pp_rank,
                global_offset,
                pipeline_model_parallel_split_rank,
                args.target_file,
                args.megatron_legacy,
            )

            # Reset env flag
            os.environ.pop(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, None)

        # Check if invalid global offset - after all PP splits, global offset should be -1
        if global_offset < -1 and not tp_conversion_only:
            raise ValueError(
                f"Invalid global offset {global_offset} found for global rank {app_state.global_rank} "
                f"and local rank {app_state.local_rank}. Should be -1 if all parameters have been assigned. "
                f"Currently, seems some parameters were duplicated."
            )
        elif global_offset > -1 and not tp_conversion_only:
            logging.error("\n")
            logging.error("!" * 80)
            logging.error("Error: Some parameters were not correctly added to model partitions.")
            logging.error("Below is list of parameters skipped in reverse order: ")

            for param_id in range(global_offset, -1, -1):
                logging.error(
                    f"Param ID: {param_id} : {global_params[1][param_id]} {global_params[0][param_id].shape}"
                )
            logging.error("!" * 80)

            raise ValueError(
                f"Invalid global offset {global_offset} found for global rank {app_state.global_rank} "
                f"and local rank {app_state.local_rank}. Should be -1 if all parameters have been assigned. "
                f"Currently, seems some parameters were not assigned."
            )

    logging.info("Successfully finished changing partitions!")

    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
