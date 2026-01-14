# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
import torch.nn.functional as F

from nemo.utils import AppState

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    world_size = torch.distributed.get_world_size()
    all_ranks = np.arange(world_size)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    if AppState().use_tp_pp_dp_mapping:
        # [DP, PP, TP]
        all_ranks = all_ranks.reshape(-1, pp_size, tp_size)
        return all_ranks[dp_rank, :, :].min()
    else:
        # [PP, DP, TP]
        all_ranks = all_ranks.reshape(pp_size, -1, tp_size)
        return all_ranks[:, dp_rank, :].min()


def get_computeprob_response(tokenizer, response, inputs):
    if parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage():
        # we only have a response on the first and last pipeline stages
        compute_prob_response = {}
        new_token_ids = []
        new_tokens = []
        new_texts = []
        log_probs = []
        full_logprobs = []
        offsets = []
        for batch_id in range(len(response['tokens'])):
            if isinstance(inputs, (list, tuple)):
                if isinstance(inputs[0], str):
                    new_token_id = tokenizer.text_to_ids(inputs[batch_id])
                    new_text = inputs[batch_id]
                    token_len = len(new_token_id)
                elif isinstance(inputs[0], torch.Tensor):
                    token_len = int(inputs[1][batch_id].item())
                    new_token_id = inputs[0][batch_id][:token_len].tolist()
                    new_text = tokenizer.ids_to_text(new_token_id)
                else:
                    raise TypeError(
                        f"Unsupported type of `inputs[0]`: {type(inputs[0])}. Supported types: `str`, `torch.Tensor`."
                    )
            else:
                raise TypeError(
                    f"Unsupported type of parameter `inputs`: {type(inputs)}. Supported types: `list` and `tuple`"
                )
            new_token_ids.append(new_token_id)
            new_tokens.append(response['tokens'][batch_id][:token_len])
            new_texts.append(new_text)
            log_probs.append(response['logprob'][batch_id][:token_len])
            full_logprobs.append(response['full_logprob'][batch_id][:token_len])
            offsets.append(response['offsets'][batch_id][:-1])
        compute_prob_response['sentences'] = new_texts
        compute_prob_response['tokens'] = new_tokens
        compute_prob_response['token_ids'] = new_token_ids
        compute_prob_response['logprob'] = log_probs
        compute_prob_response['full_logprob'] = full_logprobs
        compute_prob_response['offsets'] = offsets
        return compute_prob_response
    else:
        # intermediate stages
        return None


def repetition_penalty(logits, repetition_penalty, used_tokens):
    """Implement the repetition penalty, check paper
    https://arxiv.org/pdf/1909.05858.pdf
    """
    if used_tokens is not None and repetition_penalty != 1.0:
        logits_update = torch.gather(logits, 1, used_tokens)
        logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
    return logits


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), started=None):
    """
    This function has been mostly taken from huggingface conversational
      ai code at
      https://medium.com/huggingface/how-to-build-a-state-of-the-art-
           conversational-ai-with-transfer-learning-2d818ac26313

     @param logits: logits tensor
     @param top_k: keep only top k tokens with highest probability
     @param top_p: keep the top tokens with cumulative probability
     @filter_value: value to set filtered tokens to
     @started: a tensor of bools indicating whether the text generation starts for the batch
     returns the filtered logits
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        if started is not None:
            for i in np.arange(indices_to_remove.size(0))[started.cpu().numpy()]:
                logits[i, indices_to_remove[i]] = filter_value
        else:
            logits[indices_to_remove] = filter_value

    if 0.0 < top_p < 1.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        if started is not None:
            for i in np.arange(sorted_indices.size(0))[started.cpu().numpy()]:
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = filter_value
        else:
            for i in range(sorted_indices.size(0)):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = filter_value

    return logits
