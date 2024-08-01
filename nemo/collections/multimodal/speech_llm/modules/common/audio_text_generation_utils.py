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

"""Utilities for generating text."""

import pickle
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

import nemo.collections.nlp.modules.common.text_generation_utils as text_generation_utils
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.multimodal.speech_llm.modules.common.audio_text_generation_strategy import (
    model_inference_strategy_dispatcher,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import OutputType
from nemo.utils import AppState, logging

try:
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator as reconfigure_num_microbatches_calculator,
    )

__all__ = [
    "get_computeprob_response",
    "generate",
]


def get_computeprob_response(tokenizer, response, inputs):
    return text_generation_utils.get_computeprob_response(tokenizer, response, inputs)


def send_generate_info(
    context_tokens_tensor,
    context_length_tensor,
    audio_signal,
    audio_signal_length,
    tokens_to_generate,
    all_probs,
    compute_logprob,
    temperature,
    top_k,
    top_p,
    greedy,
    repetition_penalty,
    min_tokens_to_generate,
    end_strings,
    num_audios: Optional[torch.Tensor] = None,
    context_start_idx: Optional[List[List[int]]] = None,
):
    """
    Needs to be synced up with receive_generate_info
    """
    model_parallel_group = parallel_state.get_model_parallel_group()
    src = text_generation_utils.get_model_parallel_src_rank()

    audio_max_len = audio_signal.size(1) if audio_signal is not None else 0

    # Send the sizes of the tensors
    input_info = [
        context_tokens_tensor.size(0),  # batch_size
        context_tokens_tensor.size(1),  # seq_len
        audio_max_len,  # audio_max_len
        tokens_to_generate,
        all_probs,
        compute_logprob,  # whether to compute log probabilities matrix
        temperature,
        top_k,
        top_p,
        greedy,
        repetition_penalty,
        min_tokens_to_generate,
    ]
    input_info_tensor = torch.cuda.FloatTensor(input_info)
    torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)

    # Send variables to all ranks
    torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
    torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)

    torch.distributed.broadcast(audio_signal, src, model_parallel_group)
    torch.distributed.broadcast(audio_signal_length, src, model_parallel_group)

    # send end strings
    string_tensor = torch.as_tensor(
        np.frombuffer(pickle.dumps(end_strings), dtype=np.int8), device=torch.cuda.current_device()
    )
    size = torch.as_tensor([string_tensor.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
    torch.distributed.broadcast(size, src, model_parallel_group)
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)

    if num_audios is not None:
        torch.distributed.broadcast(num_audios, src, model_parallel_group)

    if context_start_idx is not None:
        context_idx_tensor = torch.as_tensor(
            np.frombuffer(pickle.dumps(context_start_idx), dtype=np.int8), device=torch.cuda.current_device()
        )
        ctx_size = torch.as_tensor([context_idx_tensor.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
        torch.distributed.broadcast(ctx_size, src, model_parallel_group)
        torch.distributed.broadcast(context_idx_tensor, src, model_parallel_group)


def receive_generate_info(has_multi_audios=False):
    """
    Needs to be synced up with send_generate_info
    """
    model_parallel_group = parallel_state.get_model_parallel_group()
    src = text_generation_utils.get_model_parallel_src_rank()
    input_info_tensor = torch.empty(12, dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)
    batch_size = int(input_info_tensor[0].item())
    seq_len = int(input_info_tensor[1].item())
    audio_len = int(input_info_tensor[2].item())
    tokens_to_generate = int(input_info_tensor[3].item())
    all_probs = bool(input_info_tensor[4].item())
    compute_logprob = bool(input_info_tensor[5].item())  # whether to compute log probabilities matrix
    temperature = float(input_info_tensor[6].item())
    top_k = int(input_info_tensor[7].item())
    top_p = float(input_info_tensor[8].item())
    greedy = bool(input_info_tensor[9].item())
    repetition_penalty = float(input_info_tensor[10].item())
    min_tokens_to_generate = int(input_info_tensor[11].item())

    context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
    context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.cuda.current_device())
    # Send variables to all ranks
    torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
    torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)

    audio_signal = torch.empty(batch_size, audio_len, dtype=torch.float32, device=torch.cuda.current_device())
    audio_signal_length = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
    # Send variables to all ranks
    torch.distributed.broadcast(audio_signal, src, model_parallel_group)
    torch.distributed.broadcast(audio_signal_length, src, model_parallel_group)

    array_size = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.broadcast(array_size, src, model_parallel_group)

    string_tensor = torch.empty(array_size[0], dtype=torch.int8, device=torch.cuda.current_device())
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)
    bytes = string_tensor.cpu().numpy().tobytes()
    end_strings = pickle.loads(bytes)

    num_audios = None
    context_start_idx = None
    if has_multi_audios:
        num_audios = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(num_audios, src, model_parallel_group)

        array_size = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.broadcast(array_size, src, model_parallel_group)
        context_idx_tensor = torch.empty(array_size[0], dtype=torch.int8, device=torch.cuda.current_device())
        torch.distributed.broadcast(context_idx_tensor, src, model_parallel_group)
        bytes = context_idx_tensor.cpu().numpy().tobytes()
        context_start_idx = pickle.loads(bytes)

    return (
        context_length_tensor,
        context_tokens_tensor,
        audio_signal,
        audio_signal_length,
        tokens_to_generate,
        all_probs,
        compute_logprob,
        temperature,
        top_k,
        top_p,
        greedy,
        repetition_penalty,
        min_tokens_to_generate,
        end_strings,
        num_audios,
        context_start_idx,
    )


def synced_generate(
    model,
    inference_strategy,
    context_tokens_tensor,
    context_length_tensor,
    audio_signal,
    audio_signal_length,
    tokens_to_generate,
    all_probs,
    temperature,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_attention_mask=True,
    compute_logprob=False,
    repetition_penalty=1.2,
    end_strings=[],
    min_tokens_to_generate=0,
    num_audios: Optional[torch.Tensor] = None,
    context_start_idx: Optional[List[List[int]]] = None,
):
    context_length = context_length_tensor.min().item()
    tokenizer = model.tokenizer
    if isinstance(tokenizer, TabularTokenizer):
        raise NotImplementedError("Tabular generation is not supported yet")
    else:
        batch_token_iterator = sample_sequence_batch(
            model,
            inference_strategy,
            context_tokens_tensor,
            context_length_tensor,
            audio_signal,
            audio_signal_length,
            tokens_to_generate,
            all_probs,
            compute_attention_mask=compute_attention_mask,
            compute_logprob=compute_logprob,
            temperature=temperature,
            end_strings=end_strings,
            extra={
                "top_p": top_p,
                "top_k": top_k,
                "greedy": greedy,
                "repetition_penalty": repetition_penalty,
                "min_tokens_to_generate": min_tokens_to_generate,
            },
            num_audios=num_audios,
            context_start_idx=context_start_idx,
        )

    for tokens, lengths, output_logits, full_logits, audio_feat_lens in batch_token_iterator:
        context_length += 1
    context_length += audio_feat_lens.min().item()
    if parallel_state.is_pipeline_last_stage():
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_embedding_group()
        if compute_logprob:
            torch.distributed.broadcast(output_logits, src, group)
        if all_probs:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()
            torch.distributed.broadcast(full_logits, src, group)

    else:
        if parallel_state.is_pipeline_first_stage():
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()

            if compute_logprob:
                precision = model._trainer.precision
                if precision in [16, "16"]:
                    dtype = torch.float16
                elif precision == "bf16":
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
                output_logits = torch.empty(
                    tokens.size(0), context_length - 1, dtype=dtype, device=torch.device("cuda")
                )
                torch.distributed.broadcast(output_logits, src, group)

            if all_probs:
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                full_logits = torch.empty(
                    tokens.size(0),
                    context_length - 1,
                    model.padded_vocab_size,
                    dtype=dtype,
                    device=torch.device("cuda"),
                )
                torch.distributed.broadcast(full_logits, src, group)
    if tokens is not None:
        return tokens[:, :context_length], output_logits, full_logits, audio_feat_lens
    return None


def generate(
    model,
    inputs: Union[Tuple, List[str]],
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    add_BOS=False,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_attention_mask=True,
    compute_logprob=False,
    repetition_penalty=1.0,
    end_strings=['<|endoftext|>'],
    min_tokens_to_generate=0,
    **strategy_args,
) -> OutputType:
    """
    Args:
        model (NLPModel): text generative model
        inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings
        tokens_to_generate (int): The maximum length of the tokens to be generated.
        all_probs (bool): Return the log prob for all the tokens
        temperature (float): sampling temperature
        add_BOS (bool): add the bos token at the begining of the prompt
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        greedy (bool):  Whether or not to use sampling ; use greedy decoding otherwise
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty
        min_tokens_to_generate (int): The minimum length of the tokens to be generated
        strategy_args, the extra arguments are treated as inference strategy arguments
        end_strings, a list of strings to stop generation when they are encountered in the output.
    Returns:
        OutputType: It generates the output in a dictionary type. It has the following keys:
            sentences: List[str], output sentences
            tokens: List[List[str]], output sentences borken into tokens
            logprob: List[Tensor], log prob of generated tokens
            full_logprob: List[Tensor], log prob of all the tokens in the vocab
            token_ids: List[Tensor], output sentence token ids
            offsets: List[List[int]]  # list of tokens start positions in text
    """
    if 'strategy' in strategy_args:
        inference_strategy = strategy_args['strategy']
    else:
        inference_strategy = model_inference_strategy_dispatcher(model)
    tokenizer = model.tokenizer
    has_multi_audios = False
    num_audios = None
    context_start_idx = None
    audio_signal, audio_signal_length = None, None
    if torch.distributed.get_rank() == text_generation_utils.get_model_parallel_src_rank():
        if isinstance(inputs, tuple) and len(inputs) == 2:
            context_tokens_tensor, context_length_tensor = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 4:
            context_tokens_tensor, context_length_tensor, audio_signal, audio_signal_length = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 6:  # multi-audio
            has_multi_audios = True
            (
                context_tokens_tensor,
                context_length_tensor,
                audio_signal,
                audio_signal_length,
                num_audios,
                context_start_idx,
            ) = inputs
        else:
            context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
                inputs, tokens_to_generate, add_BOS
            )

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            audio_signal,
            audio_signal_length,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
            num_audios,
            context_start_idx,
        )
    else:
        (
            context_length_tensor,
            context_tokens_tensor,
            audio_signal,
            audio_signal_length,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
            num_audios,
            context_start_idx,
        ) = receive_generate_info(has_multi_audios)

    output = synced_generate(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        audio_signal,
        audio_signal_length,
        tokens_to_generate,
        all_probs,
        temperature,
        compute_attention_mask=compute_attention_mask,
        compute_logprob=compute_logprob,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        end_strings=end_strings,
        min_tokens_to_generate=min_tokens_to_generate,
        num_audios=num_audios,
        context_start_idx=context_start_idx,
    )
    special_tokens = set()
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        special_tokens.add(tokenizer.pad_token)
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        special_tokens.add(tokenizer.eos_token)
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
        special_tokens.add(tokenizer.bos_token)
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
        special_tokens.add(tokenizer.cls_token)
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
        special_tokens.add(tokenizer.unk_token)
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
        special_tokens.add(tokenizer.sep_token)
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        special_tokens.add(tokenizer.mask_token)
    if output is not None:
        decode_tokens, output_logits, full_logits, audio_feat_lens = output
        resp_sentences = []
        resp_sentences_seg = []

        decode_tokens = decode_tokens.cpu().numpy().tolist()
        for decode_token in decode_tokens:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)
            if not isinstance(tokenizer, TabularTokenizer):
                words = []
                for token in decode_token:
                    if not isinstance(token, Iterable):
                        token = [token]
                    word = tokenizer.ids_to_tokens(token)
                    if isinstance(word, Iterable):
                        word = word[0]
                    if hasattr(tokenizer.tokenizer, 'byte_decoder'):
                        word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                            'utf-8', errors='replace'
                        )
                    words.append(word)
                resp_sentences_seg.append(words)
            else:
                words = tokenizer.text_to_tokens(sentence)
                resp_sentences_seg.append(words)

        # offsets calculation
        all_offsets = []
        for item in resp_sentences_seg:
            offsets = [0]
            for index, token in enumerate(item):
                if index != len(item) - 1:
                    if token in special_tokens:
                        offsets.append(offsets[-1])
                    else:
                        offsets.append(len(token) + offsets[-1])
            all_offsets.append(offsets)

        output = {}
        output['sentences'] = resp_sentences
        output['tokens'] = resp_sentences_seg
        output['logprob'] = output_logits
        output['full_logprob'] = full_logits
        output['token_ids'] = decode_tokens
        output['offsets'] = all_offsets
        output['audio_feat_lens'] = audio_feat_lens
        output = inference_strategy.post_generation_process(output)
        return output
    return None


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    audio_signal,
    audio_signal_length,
    tokens_to_generate,
    all_probs=False,
    compute_attention_mask=True,
    compute_logprob=False,
    type_ids=None,
    temperature=None,
    end_strings=['<|endoftext|>'],
    extra={},
    num_audios: Optional[torch.Tensor] = None,
    context_start_idx: Optional[List[List[int]]] = None,
):
    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    reconfigure_num_microbatches_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    assert tokens_to_generate > 0, "tokens_to_generate should be > 0"
    assert (
        model.cfg.get('sequence_parallel', False) == False
    ), 'sequence_parallel should be False during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_granularity', None) is None
    ), 'activations_checkpoint_granularity should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_method', None) is None
    ), 'activations_checkpoint_method should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        context_tokens, input_embeddings, audio_feat_lens = inference_strategy.init_batch(
            context_tokens,
            context_lengths,
            audio_signal,
            audio_signal_length,
            compute_attention_mask,
            num_audios,
            context_start_idx,
        )
        audio_text_context_lengths = context_lengths + audio_feat_lens
        context_length = audio_text_context_lengths.min().item()
        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + audio_text_context_lengths.max().item()
        maxlen = inference_strategy.clip_max_len(maxlen)
        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        while context_length < maxlen:
            batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                tokens,
                input_embeddings,
                maxlen,
                micro_batch_size,
                counter,
                audio_text_context_lengths,
                context_length,
                compute_attention_mask,
            )
            output = inference_strategy.forward_step(batch, tensor_shape)
            if parallel_state.is_pipeline_last_stage():
                if compute_logprob:
                    output = output[0]['logits']
                    output = tensor_parallel.gather_from_tensor_model_parallel_region(output)
                    assert output is not None
                    logits = output[:, -1].view(batch_size, -1).contiguous()

                else:
                    logits = output[0]['logits'][:, -1].contiguous()
                    logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
                    assert logits is not None
                    logits = logits.view(batch_size, -1)

                # make sure it will generate at least min_length
                min_length = extra.get('min_tokens_to_generate', 0)
                if min_length > 0:
                    within_min_length = (context_length - audio_text_context_lengths) < min_length
                    logits[within_min_length, eod_id] = -float('Inf')
                # make sure it won't sample outside the vocab_size range
                logits[:, tokenizer.vocab_size :] = -float('Inf')

                # started indicates whether the current token step passes the context_length, so we make sure not to overwrite the context tokens
                started = audio_text_context_lengths <= context_length
                if extra.get('greedy', False):
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= temperature
                    # handle repetition penality
                    logits = text_generation_utils.repetition_penalty(
                        logits, extra.get('repetition_penalty', 1.2), all_generated_indices
                    )
                    logits = text_generation_utils.top_k_logits(
                        logits, top_k=extra.get('top_k', 0), top_p=extra.get('top_p', 0.9), started=started
                    )
                    probs = F.softmax(logits, dim=-1)
                    # TODO(zhehuai)
                    probs = probs.nan_to_num(1.0)
                    prev = torch.multinomial(probs, num_samples=1).view(-1)

                # Clamp the predicted out of vocabulary tokens
                prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                # Replace sampled tokens w/ done token if EOD has already been sampled
                new_tokens = switch(new_tokens, eod_id, is_done)

                # post process the inference tokens based on the strategy
                inference_strategy.post_process(tokens, new_tokens, context_length)

                # Insert either new predicted or next prompt token
                tokens[:, context_length] = new_tokens

                if compute_logprob:
                    if output_logits is None:
                        output = F.log_softmax(output[:, :context_length, :], 2)

                        indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                        output_logits = torch.gather(output, 2, indices).squeeze(2)
                        all_generated_indices = indices[:, :, 0]
                        if all_probs:
                            full_logits = output
                    else:
                        output = F.log_softmax(output, 2)
                        indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                        new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                        # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                        output_logits = torch.cat([output_logits, new_output_logits], 1)
                        all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)
                        if all_probs:
                            full_logits = torch.cat([full_logits, output], 1)

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                #                done_token = (prev == eod_id).byte() & started.byte()
                done_token = inference_strategy.end_of_generation_condition(
                    tokens[:, : context_length + 1], prev, eod_id, end_strings
                )
                done_token = done_token.byte() & started.byte()

                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if compute_logprob:
                    if all_probs:
                        yield tokens, lengths, output_logits, full_logits, audio_feat_lens
                    else:
                        yield tokens, lengths, output_logits, None, audio_feat_lens
                else:
                    yield tokens, lengths, None, None, audio_feat_lens

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None, audio_feat_lens
                else:
                    yield None, None, None, None, audio_feat_lens

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break
