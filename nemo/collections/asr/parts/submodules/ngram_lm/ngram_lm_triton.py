# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import triton
import triton.language as tl


@triton.jit
def ngram_advance_triton_kernel(
    vocab_size: "tl.constexpr",
    states_ptr,
    new_states_ptr,
    scores_ptr,
    start_state: int,
    to_states_ptr,
    ilabels_ptr,
    arcs_weights_ptr,
    start_end_arcs_ptr,
    backoff_to_states_ptr,
    backoff_weights_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    """
    Triton kernel for N-Gram LM advance operation.
    Args:
        vocab_size: LM vocabulary size
        states_ptr: pointer to tensor with batch of current states [B]
        new_states_ptr: pointer to tensor [B, V] to store new states
        scores_ptr: pointer to tensor [B, V] to store scores
        start_state: start state of the LM (usually 0)
        to_states_ptr: pointer to the tensor with target states (arcs data)
        ilabels_ptr: pointer to the tensor with labels (arcs data)
        arcs_weights_ptr: pointer to the tensor with weights (arcs data)
        start_end_arcs_ptr: pointer to the tensor with (start, end) indices of arcs (states data)
        backoff_to_states_ptr: pointer to the tensor with backoff target states (states data)
        backoff_weights_ptr: pointer to the tensor with backoff weights (states data)
        BLOCK_SIZE: block size, should be >= vocab_size
    """
    batch_i = tl.program_id(0)  # index of the element in the batch
    cur_state = tl.load(states_ptr + batch_i)  # current state

    # NB: number of arcs in current state is <= vocab_size and BLOCK_SIZE
    vocab_offsets = tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offsets < vocab_size
    # fill in initial values: new_states = -1 (not found yet), scores = 0
    tl.store(new_states_ptr + batch_i * vocab_size + vocab_offsets, -1, mask=vocab_mask)
    tl.store(scores_ptr + batch_i * vocab_size + vocab_offsets, 0.0, mask=vocab_mask)

    accumulated_backoff = 0.0
    start_state_not_processed = True
    # loop until we process start state; it should be guaranteed that in the start state we have all vocabulary tokens
    while start_state_not_processed:
        tl.debug_barrier()  # force threads synchronization
        start_idx, end_idx = tl.load(start_end_arcs_ptr + cur_state * 2 + tl.arange(0, 2)).split()
        indices = start_idx + vocab_offsets
        mask = indices < end_idx

        # load arcs
        cur_ilabels = tl.load(ilabels_ptr + indices, mask=mask)
        cur_weights = tl.load(arcs_weights_ptr + indices, mask=mask)
        cur_to_states = tl.load(to_states_ptr + indices, mask=mask)

        # store scores for arcs reached in the current state (but not processed previously)
        not_final_mask = tl.load(new_states_ptr + batch_i * vocab_size + cur_ilabels, mask=mask, other=0) == -1
        tl.store(
            scores_ptr + batch_i * vocab_size + cur_ilabels,
            cur_weights + accumulated_backoff,
            mask=not_final_mask,
        )
        tl.store(new_states_ptr + batch_i * vocab_size + cur_ilabels, cur_to_states, mask=not_final_mask)

        start_state_not_processed = cur_state != start_state
        # process backoffs
        accumulated_backoff += tl.load(backoff_weights_ptr + cur_state)
        cur_state = tl.load(backoff_to_states_ptr + cur_state).to(states_ptr.dtype.element_ty)


@triton.jit
def ngram_multi_advance_triton_kernel(
    vocab_size: "tl.constexpr",
    states_ptr,
    new_states_out_ptr,
    scores_out_ptr,
    start_state: int,
    model_ids_ptr,
    states_offsets_ptr,
    arcs_offsets_ptr,
    to_states_ptr,
    ilabels_ptr,
    arcs_weights_ptr,
    start_end_arcs_ptr,
    backoff_to_states_ptr,
    backoff_weights_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    """
    Triton kernel for N-Gram LM advance operation.
    Args:
        vocab_size: LM vocabulary size
        states_ptr: pointer to tensor with batch of current states [B]
        new_states_out_ptr: pointer to tensor [B, V] to store new states
        scores_out_ptr: pointer to tensor [B, V] to store scores
        start_state: start state of the LM (usually 0)
        model_ids_ptr: pointer to the tensor with model ids
        states_offsets_ptr: pointer to tensor with mapping model id -> start of states
        arcs_offsets_ptr: pointer to tensor with mapping model id -> start of arcs
        to_states_ptr: pointer to the tensor with target states (arcs data)
        ilabels_ptr: pointer to the tensor with labels (arcs data)
        arcs_weights_ptr: pointer to the tensor with weights (arcs data)
        start_end_arcs_ptr: pointer to the tensor with (start, end) indices of arcs (states data)
        backoff_to_states_ptr: pointer to the tensor with backoff target states (states data)
        backoff_weights_ptr: pointer to the tensor with backoff weights (states data)
        BLOCK_SIZE: block size, should be >= vocab_size
    """
    batch_i = tl.program_id(0)  # index of the element in the batch
    cur_state = tl.load(states_ptr + batch_i)  # current state

    # load model id
    model_id = tl.load(model_ids_ptr + batch_i)
    # model_id < 0 - apply dummy values (0 scores, -1 states, filled in by caller)
    if model_id < 0:
        return

    # load offsets for model states and arcs (based on model id)
    model_states_offset = tl.load(states_offsets_ptr + model_id)
    model_arcs_offset = tl.load(arcs_offsets_ptr + model_id)

    to_states_ptr += model_arcs_offset
    ilabels_ptr += model_arcs_offset
    arcs_weights_ptr += model_arcs_offset
    start_end_arcs_ptr += model_states_offset * 2  # start_end_arcs tensor has 2 elements in each row
    backoff_to_states_ptr += model_states_offset
    backoff_weights_ptr += model_states_offset

    # NB: number of arcs in current state is <= vocab_size and BLOCK_SIZE
    vocab_offsets = tl.arange(0, BLOCK_SIZE)
    # fill in initial values: new_states = -1 (not found yet), scores = 0: moved to caller function
    # (NB: caller function should fill scores with 0, states with -1)

    accumulated_backoff = 0.0
    start_state_not_processed = True
    # loop until we process start state; it should be guaranteed that in the start state we have all vocabulary tokens
    while start_state_not_processed:
        tl.debug_barrier()  # force threads synchronization
        start_idx, end_idx = tl.load(start_end_arcs_ptr + cur_state * 2 + tl.arange(0, 2)).split()
        indices = start_idx + vocab_offsets
        mask = indices < end_idx

        # load arcs
        cur_ilabels = tl.load(ilabels_ptr + indices, mask=mask)
        cur_weights = tl.load(arcs_weights_ptr + indices, mask=mask)
        cur_to_states = tl.load(to_states_ptr + indices, mask=mask)

        # store scores for arcs reached in the current state (but not processed previously)
        not_final_mask = tl.load(new_states_out_ptr + batch_i * vocab_size + cur_ilabels, mask=mask, other=0) == -1
        tl.store(
            scores_out_ptr + batch_i * vocab_size + cur_ilabels,
            cur_weights + accumulated_backoff,
            mask=not_final_mask,
        )
        tl.store(new_states_out_ptr + batch_i * vocab_size + cur_ilabels, cur_to_states, mask=not_final_mask)

        start_state_not_processed = cur_state != start_state
        # process backoffs
        accumulated_backoff += tl.load(backoff_weights_ptr + cur_state)
        cur_state = tl.load(backoff_to_states_ptr + cur_state).to(states_ptr.dtype.element_ty)
