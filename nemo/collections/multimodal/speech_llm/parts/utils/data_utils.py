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

from typing import List, Optional

import numpy as np
import torch
from nemo.utils import logging, logging_mode


def maybe_cast_to_list(x):
    if isinstance(x, np.ndarray):
        return [item.tolist() for item in x]
    return x


def ceil_to_nearest(n, m):
    return (n + m - 1) // m * m


def get_num_samples_from_files(file_list):
    if isinstance(file_list, str):
        file_list = file_list.split(',')
    num_samples = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = list(f.readlines())
            num = len(lines)
            if lines[-1] == '\n':
                num -= 1
            num_samples.append(num)
    return num_samples


def shift_tokens_by_multi_audios(
    context_tokens, context_lengths, audio_feat_lens, context_start_idx, encoder_max_length
):
    """
    split and shift the context tokens by the audio segments, then concatenate them back. This function assumes that the whole context
    starts and ends with text tokens, and the audio segments are in between the text tokens. The audio segments are not allowed to be adjacent to each other.
    Args:
        context_tokens: tensor of shape [batch, max_context_len]
        context_lengths: tensor of shape [batch,]
        audio_feat_lens: List[List[int]]
        context_start_idx: List[List[int]]
        encoder_max_length: int
    """
    new_context_tokens = []
    for i in range(context_tokens.shape[0]):
        start_idx_list_i = context_start_idx[i] + [context_lengths[i]]
        input_len_list = [start_idx_list_i[j + 1] - start_idx_list_i[j] for j in range(len(start_idx_list_i) - 1)]
        context_tokens_list = context_tokens[i][: context_lengths[i]].split(input_len_list)
        context_tokens_i = [context_tokens_list[0]]
        for j in range(1, len(context_tokens_list)):
            context_tokens_i.append(
                torch.zeros(audio_feat_lens[i][j - 1], dtype=torch.long, device=context_tokens.device)
            )
            context_tokens_i.append(context_tokens_list[j])
        context_tokens_i = torch.cat(context_tokens_i)
        context_tokens_i = torch.nn.functional.pad(
            context_tokens_i, (0, encoder_max_length - context_tokens_i.shape[0])
        )
        new_context_tokens.append(context_tokens_i)
    new_context_tokens = torch.stack(new_context_tokens)
    return new_context_tokens


def get_nested_dict_value(d, key, sep="."):
    """
    Get the value of a nested dict given a key
    Args:
        d: dict
        key: str
    """
    for k in key.split(sep):
        d = d[k]
    return d


def align_feat_seq_list(
    seq_list: List[torch.Tensor],
    seq_len_list: List[torch.Tensor],
    mode: str = "min",
    pooling: str = 'mean',
    target_len: Optional[int] = None,
):
    """
    Align a list of feature sequences to the same length by repeating or discarding frames.
    Args:
        seq_list: List[torch.Tensor], list of tensors of shape [batch, hidden_size, seq_len]
        seq_len_list: List[torch.Tensor], list of tensors of shape [batch,]
        mode: str, "min" or "max"
        pooling: str, "mean", "max", or "min"
    Returns:
        new_seq_list: List[torch.Tensor], list of tensors of shape [batch, hidden_size, new_seq_len]
        new_seq_len_list: List[torch.Tensor], list of tensors of shape [batch,]
    """
    MODES = ["min", "max"]
    if mode not in MODES:
        raise ValueError(f"mode {mode} not supported, available modes: {MODES}")
    POOLING = ["mean", "max", "min", "avg"]
    if pooling not in POOLING:
        raise ValueError(f"pooling {pooling} not supported, available modes: {POOLING}")

    new_seq_len_list = []
    new_seq_list = []

    if target_len is None:
        target_len = [x.size(-1) for x in seq_list]
        target_len = min(target_len) if mode == "min" else max(target_len)

    for seq, seq_len in zip(seq_list, seq_len_list):
        curr_len = seq.size(-1)
        if curr_len > target_len:
            ratio = round(curr_len / target_len)
            res = abs(ratio * target_len - curr_len)
            if ratio * target_len > curr_len:  # e.g., ratio = 1.9
                # repeat the last res frames
                seq = torch.cat([seq, seq[:, :, -res:]], dim=-1)
                seq_len += res * (seq_len > target_len).long()
            elif ratio * target_len < curr_len:  # e.g., ratio = 2.1
                # discard the last res frames
                seq = seq[:, :, :-res]
                seq_len -= res * (seq_len > target_len).long()
            new_seq = seq.reshape(seq.size(0), seq.size(1), ratio, target_len)
            if pooling == "min":
                new_seq = new_seq.min(dim=2)
            elif pooling == "max":
                new_seq = new_seq.max(dim=2)
            else:
                new_seq = new_seq.mean(dim=2)
            new_seq_len = torch.round(seq_len / ratio).long()
        else:  # curr_len <= target_len
            ratio = round(target_len / curr_len)
            res = abs(ratio * curr_len - target_len)
            new_seq = torch.repeat_interleave(seq, ratio, dim=-1)
            new_seq_len = seq_len * ratio
            if ratio * curr_len > target_len:  # e.g., ratio = 1.9
                new_seq = new_seq[:, :, :target_len]
                new_seq_len = (
                    seq_len * ratio - (ratio * seq_len - target_len) * (ratio * seq_len > target_len).long()
                )  # subtract additional frames
            elif ratio * curr_len < target_len:  # e.g., ratio = 2.1
                new_seq = torch.cat([new_seq, seq[:, :, -res:]], dim=-1)
        new_seq_list.append(new_seq)
        new_seq_len_list.append(new_seq_len)
    return new_seq_list, new_seq_len_list


def build_loss_mask(processed_example: dict, answer_only_loss: bool = True):
    """Pad input_ids in batch to max batch length while building loss mask"""
    # function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
    input_ids = processed_example['input_ids']
    answer_start_idx = processed_example['answer_start_idx']
    if answer_only_loss:
        loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
    else:
        loss_mask = [1.0] * len(input_ids)

    return loss_mask


class TextProcessing:
    """
    Text processing pipeline for speech_llm data loader.
    This class is adapted from the one used in nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_dataset.py
    The class follows the interface of _process_example which takes in a context and an output
      and processes them into a formatted training example.

    Args:
        tokenizer: text tokenizer object
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        sep_id (int): The id of the separation token
        separate_prompt_and_response_with_newline (bool): Whether to separate the prompt and response with a newline character
        answer_only_loss (bool): Whether to compute the loss only on the answer part of the input
        truncation_field (str): Field to use for truncation. (Options: "answer", "context"). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length (bool): Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        prompt_template (str): Prompt template to inject via an fstring. Formatted like Q: {input}\n\nA: {output}
        virtual_tokens (int): Number of virtual tokens to add to the beginning of the input
        tokens_to_generate (int): Number of tokens to generate during inference
        context_key (str): Key to use for the context in your JSONL file
        answer_key (str): Key to use for the label in your JSONL file
        end_string (Optional[str]): If not None, add this string to the end of the answer.
        sample_alpha (Optional[float]): For SPE subword sampling
        input_text_mask_ratio (Optional[float]): If not None, will mask the input text at this ratio.
    """

    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        context_key: str = 'context',
        answer_key: str = 'answer',
        end_string: Optional[str] = None,
        sample_alpha: Optional[float] = None,
        audio_locator: Optional[str] = None,
    ):
        self.context_key = context_key
        self.answer_key = answer_key
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.seed = seed
        self.separate_prompt_and_response_with_newline = separate_prompt_and_response_with_newline
        self.answer_only_loss = answer_only_loss
        self.truncation_field = truncation_field
        self.pad_to_max_length = pad_to_max_length
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.end_string = end_string
        self.sample_alpha = sample_alpha
        self.audio_locator = audio_locator

        if add_bos and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            self.bos_id = tokenizer.bos_id
        else:
            self.bos_id = None

        if add_eos and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            self.eos_id = tokenizer.eos_id
        else:
            self.eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = self.eos_id if self.eos_id is not None else 0

        self.sep_id = sep_id if add_sep else None

        if self.prompt_template is not None:
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        assert self.truncation_field in ["answer", "context"]

    def _process_example(self, context: str, output: str):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.

        function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
        """
        if self.prompt_template is not None:
            if self.context_key not in self.prompt_template or self.answer_key not in self.prompt_template:
                if "input" in self.prompt_template and "output" in self.prompt_template:
                    logging.warning(
                        f"Using 'input' and 'output' as context and answer keys, since given ones ({self.context_key}, {self.answer_key}) are not found in the prompt template: {self.prompt_template}.",
                        mode=logging_mode.ONCE,
                    )
                    self.context_key = "input"
                    self.answer_key = "output"
            assert f'{{{self.context_key}}}' in self.prompt_template
            assert f'{{{self.answer_key}}}' in self.prompt_template
            # Make sure that '{output}' always occurs at the end of the prompt template string
            assert self.prompt_template.index(f'{{{self.answer_key}}}') == len(self.prompt_template) - len(
                f'{{{self.answer_key}}}'
            )
            # Get the context by replacing only the input
            original_context = context
            context = (
                self.prompt_template.replace(f'{{{self.context_key}}}', context)
                .replace(f'{{{self.answer_key}}}', '')
                .strip(' ')
            )
            # Replace the input and output placeholders with the actual input and output
            text = self.prompt_template.replace(f'{{{self.context_key}}}', original_context).replace(
                f'{{{self.answer_key}}}', output
            )

        elif self.separate_prompt_and_response_with_newline:
            text = context + '\n' + output
        else:
            text = context + ' ' + output

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            pre_pad = [self.tokenizer.eos_id] * self.virtual_tokens
        else:
            pre_pad = []
        answer_text = text[len(context) :]
        answer_ids = pre_pad + self.tokenizer.text_to_ids(answer_text, self.sample_alpha)
        if self.end_string:
            answer_ids += self.tokenizer.text_to_ids(self.end_string)

        if self.audio_locator is None:
            # signle audio case
            context_ids = self.tokenizer.text_to_ids(context)
            context_start_idx = [0]
        else:
            # multiple audio case
            context_ids = []
            context_start_idx = []
            for context_seg in context.split(self.audio_locator):
                context_start_idx.append(len(context_ids))
                context_ids.extend(self.tokenizer.text_to_ids(context_seg))
        context_ids = pre_pad + context_ids
        context_start_idx = [x + len(pre_pad) for x in context_start_idx]

        # for the long context cases, collate_fn includes self.tokens_to_generate for padding
        total_ids = len(context_ids) + max(len(answer_ids), self.tokens_to_generate)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        if self.add_eos:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        input_ids = context_ids
        answer_start_idx = len(input_ids)

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.bos_id] + context_ids
            input_ids = [self.bos_id] + input_ids
            answer_start_idx += 1

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        input_ids = input_ids + answer_ids

        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]
            answer_ids = answer_ids + [self.tokenizer.eos_id]

        if len(input_ids) > self.max_seq_length:
            logging.warning(f'Input ids length {len(input_ids)} exceed max sequence length {self.max_seq_length}')
            input_ids = input_ids[: self.max_seq_length]

        processed_example = {
            'input_ids': (input_ids),
            'answer_start_idx': (answer_start_idx),
            'context_ids': (context_ids),
            'context_length': len(context_ids),
            'answer_ids': (answer_ids),
            'context_start_idx': context_start_idx,
        }

        return processed_example
