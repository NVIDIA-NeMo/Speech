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

import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import pytorch_lightning as pl
import torch
from megatron.core import InferenceParams, parallel_state

from nemo.collections.common.tokenizers.chat_template_mixin import explode_chat_template_input, is_chat_input
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import shift_tokens_by_multi_audios

# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


def pad_batch(batch, pad_id, max_len):
    """pad batch seq to the same length"""
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def switch(val1, val2, boolean):
    """switch between two values based on the boolean tensor"""
    boolean = boolean.type_as(val1)
    boolean = boolean.unsqueeze(0).unsqueeze(-1)
    return (1 - boolean) * val1 + boolean * val2


class TextGenerationStrategy:
    """
    Base class for TextGeneration Strategy
    """

    def __init__(self, model):
        self.model = model
        self.tokenizer = getattr(model, "tokenizer", None)
        self.inference_params = None
        if self.model.training:
            # TODO in the future this should raise an exception
            warnings.warn(
                "Generation started while the model is in training mode, switching to eval mode "
                "(this situation may raise an exception in future versions, please call `eval()` before generation)"
            )
            self.model.eval()
        self._end_of_generation_cache = None

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        given input, run the forward step of the language model to get output tensor
        """
        if "labels" in batch:
            batch.pop("labels")

        output_tensor = self.model.language_model(**batch)
        # preds = torch.argmax(output_tensor, dim=-1)

        if self.inference_params:
            # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
            if parallel_state.is_pipeline_last_stage():
                self.inference_params.sequence_len_offset += output_tensor.size(1)
            else:
                self.inference_params.sequence_len_offset += output_tensor.size(0)

        return output_tensor

    def tokenize_batch(self, sentences, max_len, add_BOS):
        """
        convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
        Args:
            sentences (List[str]): list of input sentences in str format.
            max_len (int): max number of tokens to generate.
            add_BOS (bool): whether to add the BOS token at the beginning
        Returns:
            Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
        """
        tokenizer = self.model.tokenizer
        if is_chat_input(sentences):
            assert getattr(
                tokenizer, 'has_chat_template', False
            ), "Got chat-template input but tokenizer does not support chat template formating."
            context_tokens = list(map(tokenizer.text_to_ids, explode_chat_template_input(sentences)))
        elif add_BOS:
            context_tokens = [[tokenizer.bos_id] + tokenizer.text_to_ids(s) for s in sentences]
        elif hasattr(tokenizer.tokenizer, "get_prefix_tokens"):
            # chatglm: add tokenizer.gmask_id, tokenizer.sop_id
            context_tokens = [tokenizer.tokenizer.get_prefix_tokens() + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    @classmethod
    @abstractmethod
    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length
        Args:
            maxlen (int): the max len computed from the context and number of tokens to generate
        returns (int):
            the clip the max length based of the LM model max sequence length
        """
        pass

    @classmethod
    @abstractmethod
    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps.
           It will save the intermediate results as object attributes
           context_length (int): the context token length
           compute_attention_mask: bool: set to True to compute attention mask (not needed for FA)
        Args:
            context_tokens (torch.Tensor):  The padded context tokens including the space for tokens to be generated
        """
        pass

    @classmethod
    @abstractmethod
    def prepare_batch_at_step(
        self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        generate the batch used in inference for each of the steps
        Args:
            tokens  (torch.Tensor): the context tokens
            maxlen (int): the maximum length in the context tokens
            micro_batch_size (int): text generation batch size
            step (int): the inference step count
            context_length (int): the new token position in the tokens
        returns:
            a dict of tensors used in the inference step
        """
        pass

    @classmethod
    @abstractmethod
    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the single step inference, post process the inference results
        Args:
            tokens  (torch.Tensor): the context tokens
            new_token (torch.Tensor): sampled new token id
            context_length (int): the new token position in the tokens
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        if (len(end_strings) == 1 and end_strings[0] == END_OF_SEQ) or not end_strings:
            # Simple scenario: only finish on end of document token.
            return prev == eod_id

        end_tokens, end_strings_to_check = self._get_end_of_generation_tokens_and_strings(eod_id, end_strings)
        assert end_tokens

        is_end = torch.isin(prev, torch.tensor(list(end_tokens), dtype=prev.dtype, device=prev.device))

        if end_strings_to_check:
            # The loop below is inefficient (see warning in `_get_end_of_generation_tokens_and_strings()`)
            # TODO In addition, we will not stop if the model generates an end string followed by extra characters,
            # e.g., if `end_string` is "Done" and there exists a "Done!" token it could generate tokens
            #       [..., ".", "Done!"]
            # which would fail the `endswith("Done")` check. However, stopping when "Done!" is generated would not
            # work either, since we would need to post-process the generated string to truncate the extra "!".
            # ==> this is left for future work if there is a compelling use case requiring this feature.
            for idx, token_seq in enumerate(tokens):
                text = self.model.tokenizer.ids_to_text(token_seq.tolist())
                is_end[idx] |= any(text.endswith(end_string) for end_string in end_strings_to_check)

        return is_end

    def post_generation_process(self, output):
        """
        At the end of the text generation, post process the results
        Args:
            output  (dict): the text generation output dictionary
        """
        return output

    def _get_end_of_generation_tokens_and_strings(
        self, eod_id: int, end_strings: List[str]
    ) -> Tuple[Set[int], List[str]]:
        """
        return the tokens and strings indicating the end of generation
        Args:
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        Returns:
            a pair `(tokens, strings)` where `tokens` is a set of tokens (int) and `strings` is a list of strings,
            which must all be used to identify the end of generation (`tokens` always contains `eod_id`, while
            `strings` may be empty if all end strings are associated to unique tokens)
        """
        tokenizer = self.model.tokenizer
        # A cache is used to remember which end strings are associated to unique tokens vs. which ones
        # require an actual string comparison.
        if self._end_of_generation_cache is None or self._end_of_generation_cache["tokenizer"] is not tokenizer:
            # Invalidate the cache.
            self._end_of_generation_cache = {
                "tokenizer": tokenizer,
                "end_string_to_token": {END_OF_SEQ: eod_id},
                "end_strings_to_check": set(),
            }
        end_string_to_token = self._end_of_generation_cache["end_string_to_token"]

        end_tokens = {eod_id}  # always include `eod_id`, even if `END_OF_SEQ` is not within `end_strings`
        end_strings_to_check = []  # will contain end strings that have no associated special token

        for end_string in end_strings:
            try:
                end_tokens.add(end_string_to_token[end_string])
                continue
            except KeyError:
                if end_string in self._end_of_generation_cache["end_strings_to_check"]:
                    end_strings_to_check.append(end_string)
                    continue

            # `end_string` does not exist in the cache yet: check if `end_string` is a special token for
            # the tokenizer. Ideally, we would simply use `tokenizer.text_to_ids(end_string)`, but some
            # tokenizers (e.g., SentencePiece) may prefix the special token with another token associated
            # to an empty string. The code below is thus meant to extract the special token associated to
            # `end_string` (if it exists). Note that we use "<extra_id_1>" as prefix string to have a low
            # risk of the tokenizer merging it with `end_string`, but this is somewhat arbitrary.
            ids_ref = tokenizer.text_to_ids("<extra_id_1>")
            ids_with_end_string = tokenizer.text_to_ids(f"<extra_id_1>{end_string}")
            if len(ids_with_end_string) == len(ids_ref) + 1 and ids_with_end_string[:-1] == ids_ref:
                # We can assume that the extra token is the one corresponding to `end_string`.
                end_string_to_token[end_string] = ids_with_end_string[-1]
                end_tokens.add(ids_with_end_string[-1])
            else:
                # No special token.
                warnings.warn(
                    f"The end string '{end_string}' has no associated special token: this may slow down "
                    "generation (consider using a different tokenizer or modifying `end_strings`)"
                )
                self._end_of_generation_cache["end_strings_to_check"].add(end_string)
                end_strings_to_check.append(end_string)

        return end_tokens, end_strings_to_check


class SpeechToTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model: pl.LightningModule, max_depth: int = 100):
        cnt = 0
        while not hasattr(model, "language_model") and cnt < max_depth:
            if not hasattr(model, "module"):
                raise ValueError(
                    f"Model does not have `language_model` or `module` attribute: {model.__class__.__name__}"
                )
            model = model.module
            cnt += 1
            if not hasattr(model, "language_model") and cnt == max_depth:
                raise ValueError(
                    f"Cannot find `language_model` attribute after {max_depth} iterations: {model.__class__.__name__}"
                )
        super().__init__(model)

    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool = True,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """initialize the batch data before the inference steps."""

        audio_feats, audio_feat_lens = self.model.perception(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        if num_audios is not None:
            # handle multiple audio files per sample
            audio_feats = audio_feats.split(num_audios.tolist())
            audio_feat_lens = audio_feat_lens.split(num_audios.tolist())

        encoder_input, attention_mask, _, position_ids, encoder_max_length = self.model.inject_perception_input(
            audio_feats, audio_feat_lens, context_tokens, context_lengths, context_start_idx
        )

        self.attention_mask = attention_mask
        self.position_ids = position_ids

        if num_audios is not None:
            # handle multiple audio files per sample
            new_context_tokens = shift_tokens_by_multi_audios(
                context_tokens, context_lengths, audio_feat_lens, context_start_idx, encoder_max_length
            )
            audio_feat_lens = torch.stack([torch.sum(lens) for lens in audio_feat_lens])  # [batch,]
        else:
            new_context_tokens = self.model._shift_labels_by_emb_len(
                context_tokens, context_lengths, audio_feat_lens, encoder_max_length, pad_token=0
            )

        return new_context_tokens, encoder_input, audio_feat_lens

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""
        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.language_model.position_embedding_type == "learned_absolute":
            if maxlen > self.model.language_model.max_sequence_length + 1:
                maxlen = self.model.language_model.max_sequence_length + 1
        return maxlen

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch for each of the inference steps"""
        if step == 0:
            tokens2use = tokens[:, :curr_context_length].contiguous()
            # positions2use = self.position_ids[:, :curr_context_length].contiguous()
            embeddings2use = input_embeddings[:curr_context_length].contiguous()
            # Prepare KV cache for the entire context at first step, and reuse afterwards.
            self.inference_params = InferenceParams(max_batch_size=tokens2use.size(0), max_sequence_length=maxlen)
        else:
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, -1).contiguous()
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, -1).contiguous()
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use).contiguous()
            started = context_lengths <= curr_context_length
            embeddings2use = switch(input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use, started)
            embeddings2use = embeddings2use.contiguous()

        attention_mask = self.attention_mask if compute_attention_mask else None
        batch = {
            "input_ids": None,
            "position_ids": None,
            "attention_mask": attention_mask,
            "decoder_input": embeddings2use,
            "inference_params": self.inference_params,
        }
        return batch

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        if len(end_strings) == 1 and end_strings[0] == END_OF_SEQ:
            return prev == eod_id
        else:
            tokenizer = self.model.tokenizer
            conditions = []
            end_tokens = set()
            end_tokens.add(eod_id)
            for end_string in end_strings:
                if len(end_string) > 1:
                    continue
                ids_1 = tokenizer.text_to_ids(f'<extra_id_1>{end_string}')
                ids_2 = tokenizer.text_to_ids('<extra_id_1>')
                if len(ids_1) <= len(ids_2):
                    continue
                token_id = ids_1[len(ids_2) :][0]

                end_tokens.add(token_id)

            for p, token_item in zip(prev, tokens):
                text = tokenizer.ids_to_text(token_item.tolist())
                conditions.append(
                    any([text.endswith(end_string) for end_string in end_strings] + [p.item() in end_tokens])
                )
            return torch.tensor(conditions, dtype=torch.bool, device=tokens.device)


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.speechlm.models.speech_to_text_llm_model import SpeechToTextLLM

    if isinstance(model, SpeechToTextLLM):
        return SpeechToTextGenerationStrategy(model, **args)
    else:
        raise ValueError(f"Unsupported model type for text generation: {model.__class__.__name__}")
