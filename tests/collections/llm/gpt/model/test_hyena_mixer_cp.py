# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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

"""Example Usage:
torchrun --nproc_per_node=2 tests/collections/llm/gpt/model/test_hyena_mixer_cp.py --operator_type hyena_short_conv [--use_subquadratic_ops]
"""

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from einops import rearrange
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from torch.distributed.nn.functional import all_gather as functional_all_gather
from torch.nn.parallel import DistributedDataParallel as DDP

from nemo.collections.llm.gpt.model.hyena import HyenaTestConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer
from nemo.utils import logging


def init_parallel_state(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1):
    """Initialize distributed training and megatron parallel state."""

    num_gpus = torch.cuda.device_count()
    required_world_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    assert (
        num_gpus == required_world_size
    ), f"World size {num_gpus} != TP={tensor_model_parallel_size} x PP={pipeline_model_parallel_size} x CP={context_parallel_size}"

    # Set up environment variables
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Get local rank
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)

    # Set up timeout
    timeout_seconds = int(os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800))
    timeout_timedelta = timedelta(seconds=timeout_seconds)

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout_timedelta)
        logging.info(f"Initialized distributed training with local rank {local_rank}")

    # Initialize parallel state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
    )

    # Verify initialization
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_world_size = parallel_state.get_context_parallel_world_size()
    logging.info(f"CP rank: {cp_rank}, CP world size: {cp_world_size}")
    return local_rank


def zigzag_split_across_group_ranks(data, group, seq_dim=0):
    """Distributes tensor data across group ranks using zigzag pattern.

    Divides the input tensor along sequence dimension and distributes chunks
    in an alternating pattern across different ranks.

    Arguments:
        data: original tensor to split across group ranks.
        group: the group to distribute the data across.
        seq_dim: the sequence/context dimension to split.

    Returns:
        Tensor slice for the current rank following zigzag distribution.
    """
    # Get group information
    process_count = len(dist.get_process_group_ranks(group))
    current_rank = dist.get_rank(group)

    # Skip distribution for single process
    if process_count == 1:
        return data

    # Calculate number of chunks for zigzag distribution
    total_chunks = 2 * process_count

    # Divide data into equal chunks
    tensor_chunks = list(torch.chunk(data, total_chunks, dim=seq_dim))

    # Implement zigzag distribution logic:
    # Each rank gets two chunks in specific positions
    # First chunk is at position equal to rank
    first_chunk_idx = current_rank
    # Second chunk is from the end, offset by rank+1
    second_chunk_idx = total_chunks - 1 - current_rank

    # Combine the appropriate chunks for this rank
    rank_data = torch.cat([tensor_chunks[first_chunk_idx], tensor_chunks[second_chunk_idx]], dim=seq_dim)

    return rank_data.contiguous()


def zigzag_gather_from_group_ranks(data, group, seq_dim=0):
    """Reconstructs complete tensor from zigzag-distributed chunks.

    Takes data distributed across ranks in zigzag pattern and reassembles
    the original complete tensor.

    Arguments:
        data: tensor fragment from current rank to be gathered.
        group: the group to gather data from.
        seq_dim: dimension along which to concatenate fragments.

    Returns:
        Reconstructed tensor with fragments from all ranks.
    """
    # Get group information
    process_count = len(dist.get_process_group_ranks(group))

    # Skip gathering for single process
    if process_count == 1:
        return data

    # Gather from all ranks using autograd-enabled all_gather
    gathered_data = functional_all_gather(data, group=group)

    # Initialize a list to store the original sequence chunks with proper tensor type
    seq_chunks = []
    for i in range(2 * process_count):
        seq_chunks.append(None)  # Will be replaced with tensors

    # Process each gathered tensor
    for i, data_i in enumerate(gathered_data):
        chunk_size = data_i.size(seq_dim) // 2

        # Split the data_i back into the original two chunks
        chunk0, chunk1 = torch.split(data_i, chunk_size, dim=seq_dim)

        # Reassign the chunks to their original positions
        seq_chunks[i] = chunk0
        seq_chunks[-(i + 1)] = chunk1

    # Concatenate all chunks to reconstruct the original data
    reconstructed_data = torch.cat(seq_chunks, dim=seq_dim)

    return reconstructed_data


class MixerModuleWrapper(torch.nn.Module):
    def __init__(self, seq_len, operator_type="hyena_short_conv", use_subquadratic_ops=False):
        super().__init__()

        self.use_subquadratic_ops = use_subquadratic_ops
        self.operator_type = operator_type

        # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        # Set the b2b parameter in the config
        hyena_config = HyenaConfig(num_groups_hyena=4096, num_groups_hyena_short=256, num_groups_hyena_medium=256)
        hyena_test_config = HyenaTestConfig(params_dtype=torch.float32, use_subquadratic_ops=use_subquadratic_ops)

        logging.info("Creating HyenaMixer...")
        self.mixer = HyenaMixer(
            transformer_config=hyena_test_config,
            hyena_config=hyena_config,
            max_sequence_length=seq_len,
            submodules=submodules,
            layer_number=1,
            operator_type=operator_type,
        )

    def forward(self, x, _use_cp=True):
        if self.use_subquadratic_ops and self.operator_type != "hyena":
            logging.info(f"Using subquadratic_ops: {self.use_subquadratic_ops}")
            z = self.mixer.b2b_kernel(x, _use_cp=_use_cp)
        else:
            logging.info("Using PyTorch implementation")
            features = self.mixer.hyena_proj_conv(x, _use_cp=_use_cp)
            x1, x2, v = rearrange(
                features, "b (g dg p) l -> b (g dg) p l", p=3, g=self.mixer.num_groups_per_tp_rank
            ).unbind(dim=2)
            z = self.mixer.mixer(x1, x2, v, _hyena_use_cp=_use_cp)
        return z


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test hyena mixer with context parallelism")
    parser.add_argument(
        "--use_subquadratic_ops",
        action="store_true",
        default=False,
        help="Whether to use subquadratic_ops implementation",
    )
    parser.add_argument(
        "--operator_type",
        type=str,
        default="hyena_short_conv",
        choices=["hyena_short_conv", "hyena_medium_conv", "hyena"],
        help="Operator type. Options: hyena_short_conv, hyena_medium_conv, hyena",
    )
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help="Tensor model parallel size",
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="Pipeline model parallel size",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=2,
        help="Context parallel size",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/tmp/nemo2_hyena_results",
        help="Directory for logs",
    )
    args = parser.parse_args()

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up file handler for NeMo logging
    rank = int(os.getenv("RANK", "0"))
    log_file = os.path.join(args.log_dir, f"rank_{rank}.log")
    logging.add_file_handler(log_file)

    # Initialize parallel state
    local_rank = init_parallel_state(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
    )

    logging.info(f"Starting hyena mixer test with args: {args}")

    try:
        # Initialize the model parallel RNG
        model_parallel_cuda_manual_seed(42)

        # Model initialization
        batch_size = 2
        seq_len = 1024  # Increased from 512 to provide more space for overlapping patches
        mixer_module_wrapper = MixerModuleWrapper(
            seq_len=seq_len,
            operator_type=args.operator_type,
            use_subquadratic_ops=args.use_subquadratic_ops,
        )

        ddp_mixer_module_wrapper = DDP(
            mixer_module_wrapper,
            process_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            find_unused_parameters=True,
        )

        input_features = torch.rand(
            (batch_size, mixer_module_wrapper.mixer.hidden_size * 3, seq_len),
            dtype=mixer_module_wrapper.mixer.transformer_config.params_dtype,
            device=torch.cuda.current_device(),
        )

        # Broadcast within each group
        cp_group = parallel_state.get_context_parallel_group()
        dist.broadcast(input_features, min(dist.get_process_group_ranks(cp_group)), group=cp_group)

        logging.info("Running without context parallel")
        output_features = ddp_mixer_module_wrapper(input_features, _use_cp=False)

        if dist.get_rank() == 0:
            try:
                assert output_features.shape == (
                    batch_size,
                    mixer_module_wrapper.mixer.hidden_size,
                    seq_len,
                ), f"output_features.shape: {output_features.shape}, batch_size: {batch_size}, mixer_module_wrapper.mixer.hidden_size: {mixer_module_wrapper.mixer.hidden_size}, seq_len: {seq_len}"
                logging.info(f"Output features shape: {output_features.shape}")
            except AssertionError as e:
                logging.error(f"Assertion error for output features shape: {e}")
                raise

        loss = output_features.float().mean()
        loss.backward()
        dist.barrier()

        # Store the gradients for later comparison.
        grads_without_cp = []
        for n, p in ddp_mixer_module_wrapper.named_parameters():
            if p.grad is not None:
                grads_without_cp.append((n, p.grad.clone()))

        ddp_mixer_module_wrapper.zero_grad()
        dist.barrier()

        logging.info("Running with context parallel")
        # Split the input features across the context parallel group
        input_features_cp = zigzag_split_across_group_ranks(input_features, group=cp_group, seq_dim=2)

        output_features_cp = ddp_mixer_module_wrapper(input_features_cp, _use_cp=True)
        if dist.get_rank() == 0:
            try:
                assert output_features_cp.shape == (
                    batch_size,
                    mixer_module_wrapper.mixer.hidden_size,
                    seq_len // parallel_state.get_context_parallel_world_size(),
                ), f"output_features_cp.shape: {output_features_cp.shape}, batch_size: {batch_size}, mixer_module_wrapper.mixer.hidden_size: {mixer_module_wrapper.mixer.hidden_size}, seq_len: {seq_len}"
                logging.info(f"Output features CP shape: {output_features_cp.shape}")
            except AssertionError as e:
                logging.error(f"Assertion error for output features CP shape: {e}")
                raise

        # Gather from all ranks according to zigzag splitting.
        output_features_cp_gathered = zigzag_gather_from_group_ranks(output_features_cp, group=cp_group, seq_dim=2)
        if dist.get_rank() == 0:
            try:
                # Verify shapes are correct
                assert output_features_cp_gathered.shape == (
                    batch_size,
                    mixer_module_wrapper.mixer.hidden_size,
                    seq_len,
                ), f"output_features_cp_gathered.shape: {output_features_cp_gathered.shape}, batch_size: {batch_size}, mixer_module_wrapper.mixer.hidden_size: {mixer_module_wrapper.mixer.hidden_size}, seq_len: {seq_len}"
                logging.info(f"Output features CP gathered shape: {output_features_cp_gathered.shape}")
            except AssertionError as e:
                logging.error(f"Assertion error for output features CP gathered shape: {e}")
                raise

        loss_with_cp = output_features_cp_gathered.float().mean()
        loss_with_cp.backward()
        dist.barrier()

        # Store the gradients for later comparison.
        grads_with_cp = []
        for n, p in ddp_mixer_module_wrapper.named_parameters():
            if p.grad is not None:
                grads_with_cp.append((n, p.grad.clone()))

        ddp_mixer_module_wrapper.zero_grad()
        dist.barrier()

        # Only perform comparison on rank 0
        if dist.get_rank() == 0:
            logging.info(f"Comparing loss values: without CP = {loss.item()}, with CP = {loss_with_cp.item()}")
            try:
                torch.testing.assert_close(loss, loss_with_cp)
                logging.info("Loss comparison successful")
            except AssertionError as e:
                logging.error(f"Loss comparison failed: {e}")
                raise

            try:
                torch.testing.assert_close(output_features, output_features_cp_gathered)
                logging.info("Output tensor comparison successful")
            except AssertionError as e:
                logging.error(f"Output tensor comparison failed: {e}")
                raise

            # Check gradients with and without CP.
            try:
                assert len(grads_without_cp) == len(grads_with_cp)
                logging.info(f"Comparing {len(grads_without_cp)} gradient tensors")
            except AssertionError as e:
                logging.error(f"Gradient count mismatch: {e}")
                raise

            gradient_mismatch = False
            for (n_without_cp, g_without_cp), (n_with_cp, g_with_cp) in zip(grads_without_cp, grads_with_cp):
                try:
                    torch.testing.assert_close(g_without_cp, g_with_cp)
                except AssertionError as e:
                    gradient_mismatch = True
                    logging.error(f"Gradient mismatch for {n_without_cp}: {e}")

            if gradient_mismatch:
                logging.warning("There were gradient mismatches!")
            else:
                logging.info("All gradients matched successfully!")

    finally:
        # Log final cleanup
        logging.info("Test completed, cleaning up resources")

        # Reset CUDA device
        torch.cuda.empty_cache()

        # Clean up any dangling context or process groups
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

        # Force a small delay to ensure all cleanup is complete
        time.sleep(1)
