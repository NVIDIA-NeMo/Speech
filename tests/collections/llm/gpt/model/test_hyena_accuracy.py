# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

###########################################################
# BEGIN COPY/pasted bionemo stuff:
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Set, TypeVar

import lightning.pytorch as pl
import megatron.core.num_microbatches_calculator
import pytest
import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.tensor_parallel import random as tp_random
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module, MegatronModule

from nemo.collections import llm
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.io.pl import MegatronCheckpointIO


def _munge_key_megatron_to_nemo2(k: str) -> str:
    return f"module.{k}"


def _munge_sharded_tensor_key_megatron_to_nemo2(v: ShardedTensor) -> ShardedTensor:
    # This works with PP=1, how do we handle PP>1?
    key = v.key
    v.key = _munge_key_megatron_to_nemo2(key)
    return v


def _key_in_filter(k: str, filter: Set[str]) -> bool:
    for prefix in filter:
        if k.startswith(prefix):
            return True
    return False


MegatronModelType = TypeVar("MegatronModelType", bound=MegatronModule)


def _reset_microbatch_calculator():
    """Resets _GLOBAL_NUM_MICROBATCHES_CALCULATOR in megatron which is used in NeMo to initilised model parallel in
    nemo.collections.nlp.modules.common.megatron.megatron_init.initialize_model_parallel_for_nemo
    """  # noqa: D205, D415
    megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None


def _dummy() -> None:
    return


def _teardown_apex_megatron_cuda():
    """Cleans GPU allocation and model and data parallel settings after usage of a model:
    - sets the global variables related to model and data parallelism to None in Apex and Megatron:.
    - releases all unoccupied cached GPU memory currently held by the caching CUDA allocator, see torch.cuda.empty_cache
    """  # noqa: D205, D415
    torch.cuda.empty_cache()
    _reset_microbatch_calculator()
    parallel_state.destroy_model_parallel()


def _initialize_distributed_parallel_state(
    devices: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_split_rank: int = 0,
    context_parallel_size: int = 1,
    interactive: bool = False,
) -> None:
    # initialize pytorch DDP
    # if not interactive and not torch.distributed.is_initialized():
    if not torch.distributed.is_initialized():
        logging.info("pytorch DDP is not initialized. Initializing with pytorch-lightening...")
        trainer = pl.Trainer(devices=devices, strategy="ddp" if not interactive else "auto", num_nodes=1)

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(_dummy, trainer=trainer)
        trainer.strategy.setup_environment()

    if not interactive and parallel_state.is_unitialized():
        logging.info("Megatron DDP is not initialized. Initializing...")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
        )


@contextmanager
def distributed_model_parallel_state(
    seed: Optional[int] = 42,
    devices: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_split_rank: int = 0,
    context_parallel_size: int = 1,
    interactive: bool = False,
) -> Iterator[None]:
    """Context manager for handling creating and cleaning up distributed model parallel state for tests.
    Use like:
    with distributed_model_parallel_state():
        # your test code here
    # After the block your state is cleaned up.
    """  # noqa: D205
    initial_states: Optional[Any] = None

    try:
        _teardown_apex_megatron_cuda()
        _initialize_distributed_parallel_state(
            devices=devices,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
            context_parallel_size=context_parallel_size,
            interactive=interactive,
        )
        # Our goal is to set required state on entry, and then restore current state on exit for the RNGs.
        #  there are two possibilities that are handled below:
        # 1. If the RNG state is not initialized, we need to set it up and then
        #     unset it on exit to restore the current state. We track that this is the case when `initial_states` is `None`.
        # 2. If the RNG state is initialized, we need to track this state and reset it on exit to be what it was on entry.
        #    We track that this is the case when `initial_states` is not `None`.
        if tp_random.get_cuda_rng_tracker().is_initialized():
            initial_states = tp_random.get_cuda_rng_tracker().get_states()
        if seed is not None:
            # Set the seed if provided, this case is valid whether or not the RNG had state previously.
            #  on exit the RNG state will be restored to what it was on entry.
            tp_random.model_parallel_cuda_manual_seed(seed)
        else:
            # This is the case where the RNG state is not initialized and no seed was provided.
            #  We need to raise an error in this case, as we cannot restore the RNG state on exit and we need a seed
            #  to initialize the RNG state to. This only happens if the user overrides the default seed and sets it
            #  to None, and additionally if the RNG state was not initialized externally, as there is a default seed of 42.
            if initial_states is None:
                raise ValueError(
                    "You must provide a seed if the initial parallel state is unset. "
                    "Either provide a seed or leave the default seed (rather setting to None) "
                    "or initialize the RNG state externally."
                )
        yield
    finally:
        if initial_states is not None:
            tp_random.get_cuda_rng_tracker().set_states(initial_states)
        else:
            # Reset to the unset state
            tp_random.get_cuda_rng_tracker().reset()
        _teardown_apex_megatron_cuda()


# END COPY/pasted bionemo stuff
###############################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself


def load_weights_sharded_inplace_nemo2_to_mcore(
    model: MegatronModelType,
    distributed_checkpoint_dir: str | Path,
    skip_keys_with_these_prefixes: Set[str],
    ckpt_format: Literal["zarr", "torch_dist"] = "zarr",
):
    logger.info("Start setting up state dict")
    sharded_state_dict = {
        _munge_key_megatron_to_nemo2(k): _munge_sharded_tensor_key_megatron_to_nemo2(v)
        for k, v in model.sharded_state_dict().items()
        if not _key_in_filter(
            k, skip_keys_with_these_prefixes
        )  # and "_extra_state" not in k  # extra state is needed for fp8 sharded states
    }
    MegatronCheckpointIO(save_ckpt_format=ckpt_format).load_checkpoint(
        distributed_checkpoint_dir, sharded_state_dict=sharded_state_dict
    )


@pytest.mark.skip(reason="Skipping test due to slow runtime and non-availability of model/test data in CI.")
def test_golden_values(use_te: bool = True):
    """Step 1:
    # add local .ssh/*.pub key to eos ~/.ssh/authorized_keys
    mkdir -p arc_model/checkpoints/
    rsync -avz --progress --partial login-eos01.eos.clusters.nvidia.com:/lustre/fsw/healthcareeng_bionemo/arc_evo2/savanna_outputs/interleaved_hyena_7b arc_model/checkpoints/
    rsync -avz --progress --partial login-eos01.eos.clusters.nvidia.com:/lustre/fsw/healthcareeng_bionemo/arc_evo2/savanna_outputs/interleaved_hyena_7b_no_te arc_model/checkpoints/
    mkdir -p arc_model/gold_standards/
    rsync -avz --progress --partial login-eos01.eos.clusters.nvidia.com:/lustre/fsw/healthcareeng_bionemo/arc_evo2/savanna_outputs/interleaved_7b_golden_value.pt arc_model/gold_standards/
    rsync -avz --progress --partial login-eos01.eos.clusters.nvidia.com:/lustre/fsw/healthcareeng_bionemo/arc_evo2/savanna_outputs/final_7b_no_fp8_golden_value.pt arc_model/gold_standards/
    """
    if use_te:
        cfg_path = "arc_model/checkpoints/interleaved_hyena_7b/weights"  # TODO interleaved checkpoint
    else:
        cfg_path = "arc_model/checkpoints/interleaved_hyena_7b_no_te/weights"

    with torch.inference_mode(), distributed_model_parallel_state():
        hyena_config = llm.Hyena7bConfig(use_te=use_te, attention_backend=AttnBackend.fused)
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        raw_megatron_model = hyena_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, cfg_path, {}, "zarr")
        """
        fp8='hybrid', fp8_margin=0, fp8_interval=1, fp8_amax_history_len=16, fp8_amax_compute_algo='max', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False
        """
        model = Float16Module(hyena_config, raw_megatron_model)
        input_seq = "GAAATTAGCGCGTCCGGAATGATACGAGGGGAAACGAAATTTTGAATTAATGGAGAAAAAAGACGAGAAACCTTAAGCAAAAAAATTTTAGCTTCGAATATTTATTAATTTCTGAGATGTTGTTAAACGATTTTCGATTCCAAGTTGTGCGCACGAACGTTATTGCAAATAAATGCTGCTTATTCGGATGTTTCCACGATCTTTGTTGCAATGGTAGTCGAGTACCCGATAACCCAATTTCGTTACATCGGCCTATCTGTAGAATATCCAATCTATGGTTCATAAAAAATCTGATCGTTTGTTTTTAAGAAATTAAACGCGTTAAATTGAACGAATTTCGAATACCGGTCTTAGCGAAGGACCTCCCCTCTTGCTTGCGTATTGCCCCGCGAAATTTCTTTTCGGCGATGAACGATACAAAAAATTCTATCGAATGTTACTTCTATTCTCTGCCTCGTCTATGACTTGGAGATTGGTCTATGTCGTTCGTTTTCTCGCGAGTTTCCAATATGTCCGTAGTATGTGAACGCTGGTATTCGTGAAGATAAATTATTGTTTTTACAATTTCTTTCAAAAATATATAATTTTAATTTATATAAT"
        input_ids = torch.tensor(tokenizer.text_to_ids(input_seq)).int().unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_seq)).unsqueeze(0).to(device)
        attention_mask = None
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        gold_standard_no_fp8 = torch.load("arc_model/gold_standards/final_7b_no_fp8_golden_value.pt").to(
            device=outputs.device, dtype=outputs.dtype
        )
        gold_standard_fp8 = torch.load("arc_model/gold_standards/interleaved_7b_golden_value.pt").to(
            device=outputs.device, dtype=outputs.dtype
        )

        our_generation_str = "".join(
            [chr(idx) for idx in outputs.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy().tolist()]
        )
        their_generation_str_fp8 = "".join(
            [
                chr(idx)
                for idx in gold_standard_fp8.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy().tolist()
            ]
        )
        their_generation_str_no_fp8 = "".join(
            [
                chr(idx)
                for idx in gold_standard_no_fp8.softmax(dim=-1)
                .argmax(dim=-1)
                .flatten()
                .detach()
                .cpu()
                .numpy()
                .tolist()
            ]
        )
        char_matches_ours_v_theirs_no_fp8 = [
            our_generation_str[i] == their_generation_str_no_fp8[i] for i in range(len(their_generation_str_no_fp8))
        ]
        char_matches_ours_v_theirs_fp8 = [
            our_generation_str[i] == their_generation_str_fp8[i] for i in range(len(their_generation_str_fp8))
        ]
        char_matches_theirs_v_theirs_fp8_vs_not = [
            their_generation_str_fp8[i] == their_generation_str_no_fp8[i]
            for i in range(len(their_generation_str_no_fp8))
        ]
        token_similarity_vs_no_fp8 = sum(char_matches_ours_v_theirs_no_fp8) / len(char_matches_ours_v_theirs_no_fp8)
        token_similarity_vs_fp8 = sum(char_matches_ours_v_theirs_fp8) / len(char_matches_ours_v_theirs_fp8)
        token_similarity_theirs = sum(char_matches_theirs_v_theirs_fp8_vs_not) / len(
            char_matches_theirs_v_theirs_fp8_vs_not
        )
        assert (
            token_similarity_vs_no_fp8 >= token_similarity_theirs
            and token_similarity_vs_fp8 >= token_similarity_theirs
        )
        torch.testing.assert_close(outputs, gold_standard_no_fp8)
