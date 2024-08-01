# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import abc
import copy
import os
import re
import warnings
from typing import List, Set, Tuple

import torch
from transformers import CLIPImageProcessor

from nemo.collections.common.tokenizers.chat_template_mixin import explode_chat_template_input, is_chat_input
from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging

try:
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches


# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


class TextGenerationStrategy:
    """
    Base class for TextGeneration Strategy
    """

    def __init__(self, model):
        self.model = model
        if self.model.training:
            # TODO in the future this should raise an exception
            warnings.warn(
                "Generation started while the model is in training mode, switching to eval mode "
                "(this situation may raise an exception in future versions, please call `eval()` before generation)"
            )
            self.model.eval()
        self._end_of_generation_cache = None

    def forward_step(self, batch, tensor_shape):
        fwd_bwd_function = get_forward_backward_func()
        output_tensor = fwd_bwd_function(
            forward_step_func=self.model.get_forward_output_only_func(),
            data_iterator=iter(
                [
                    batch,
                ]
            ),
            model=[self.forward_model],
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=tensor_shape[0],
            micro_batch_size=tensor_shape[1],
        )

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

    @abc.abstractclassmethod
    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length
        Args:
            maxlen (int): the max len computed from the context and number of tokens to generate
        returns (int):
            the clip the max length based of the LM model max sequence length
        """
        pass

    @abc.abstractclassmethod
    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps.
           It will save the intermediate results as object attributes
           context_length (int): the context token length
           compute_attention_mask: bool: set to True to compute attention mask (not needed for FA)
        Args:
            context_tokens (torch.Tensor):  The padded context tokens including the space for tokens to be generated
        """
        pass

    @abc.abstractclassmethod
    def prepare_batch_at_step(
        self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        Args:
            tokens  (torch.Tensor): the context tokens
            maxlen (int): the maximum length in the context tokens
            micro_batch_size (int): text generation batch size
            step (int): the inference step count
            context_length (int): the new token position in the tokens
        returns:
            a tuple of list of tensor arguments for the model and a list of tensor shape required by forward method
        """
        pass

    @abc.abstractclassmethod
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


class GPTModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""

        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        # Get the attention mask and postition ids.
        self.attention_mask, _, self.position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eos_id,
            self.model.cfg.get('reset_position_ids', False),
            self.model.cfg.get('reset_attention_mask', False),
            self.model.cfg.get('eod_mask_loss', False),
            compute_attention_mask=compute_attention_mask,
        )

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        """
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


class GriffinModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""

        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        # Get the attention mask and postition ids.
        self.attention_mask, _, self.position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eos_id,
            self.model.cfg.get('reset_position_ids', False),
            self.model.cfg.get('reset_attention_mask', False),
            self.model.cfg.get('eod_mask_loss', False),
            compute_attention_mask=compute_attention_mask,
        )
        self.attention_mask = None

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = False,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        """
        # types2use = None
        # Allocate memory for the entire context.

        tokens2use = tokens

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None

        batch = [tokens2use, attention_mask_repeat]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, (tensor_shape, context_length)

    def forward_step(self, batch, tensor_shape_and_context_length):
        tensor_shape, context_length = tensor_shape_and_context_length
        fwd_bwd_function = get_forward_backward_func()

        output_tensor = fwd_bwd_function(
            forward_step_func=self.model.get_forward_output_only_func(),
            data_iterator=iter(
                [
                    batch,
                ]
            ),
            model=[self.forward_model],
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=tensor_shape[0],
            micro_batch_size=tensor_shape[1],
        )

        output_tensor[0]['logits'] = output_tensor[0]['logits'][:, :context_length, :]
        return output_tensor


def neva_process_prompts(prompt, tokenizer, multimodal_cfg, num_media_latents, conv_template):
    from nemo.collections.multimodal.data.neva.neva_dataset import (
        DEFAULT_IMAGE_TOKEN,
        preprocess_llama_2,
        preprocess_llama_3,
        preprocess_multimodal,
        preprocess_nv_dpo,
        preprocess_nvgpt,
        preprocess_v1,
    )

    list_data_dict = []
    if multimodal_cfg["conv_template"] in ["nvgpt", "nv_steerlm", "nv_dpo"]:
        record = {
            'system': (
                '\n'
                if multimodal_cfg["conv_template"] == 'nv_dpo'
                else 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n\n'
            ),
            'conversations': [
                {'from': 'User', 'value': prompt},
                {
                    'from': 'Assistant',
                    'value': '',
                },
            ],
        }

        for turn in record['conversations']:
            if turn.get('value') is not None:
                turn['value'] = re.sub('<image>', f'{DEFAULT_IMAGE_TOKEN}\n', turn['value'])
        list_data_dict.append(record)

        # overwrite the media_type in multimodal_cfg to image for image inference using video neva
        # if the prompt does not contain video, then the media_type is image
        if list_data_dict[0]['conversations'][0]['value'].find('video') == -1:
            if multimodal_cfg.get('media_type') is not None and multimodal_cfg.get('num_frames') is not None:
                multimodal_cfg['media_type'] = 'image'
                multimodal_cfg['num_frames'] = 1

        sources = preprocess_multimodal(copy.deepcopy(list_data_dict), multimodal_cfg, num_media_latents)
        if multimodal_cfg["conv_template"] in ["nvgpt", "nv_steerlm"]:
            data_dict = preprocess_nvgpt(sources, tokenizer, multimodal_cfg)
        else:
            data_dict = preprocess_nv_dpo(sources, tokenizer, multimodal_cfg)

    elif multimodal_cfg["conv_template"] == "llama_2":
        record = {
            'conversations': [
                {
                    'from': 'human',
                    'value': prompt,
                },
                {
                    'from': 'gpt',
                    'value': '',
                },
            ],
        }

        for turn in record['conversations']:
            if turn.get('value') is not None:
                turn['value'] = re.sub('<image>', f'{DEFAULT_IMAGE_TOKEN}\n', turn['value'])
        list_data_dict.append(record)

        sources = preprocess_multimodal(
            copy.deepcopy(list_data_dict), multimodal_cfg, num_media_latents
        )  # HARDCODED FOR NOW
        data_dict = preprocess_llama_2(sources, tokenizer, multimodal_cfg)
    elif multimodal_cfg["conv_template"] == "llama_3":
        record = {
            'conversations': [
                {
                    'from': 'human',
                    'value': prompt,
                },
                {
                    'from': 'gpt',
                    'value': '',
                },
            ],
        }

        for turn in record['conversations']:
            if turn.get('value') is not None:
                turn['value'] = re.sub('<image>', f'{DEFAULT_IMAGE_TOKEN}\n', turn['value'])
        list_data_dict.append(record)
        sources = preprocess_multimodal(
            copy.deepcopy(list_data_dict), multimodal_cfg, num_media_latents
        )  # HARDCODED FOR NOW
        data_dict = preprocess_llama_3(sources, tokenizer, multimodal_cfg)
    elif multimodal_cfg["conv_template"] == "mistral":
        record = {
            'conversations': [
                {
                    'from': 'human',
                    'value': prompt,
                },
                {
                    'from': 'gpt',
                    'value': '',
                },
            ],
        }
        for turn in record['conversations']:
            if turn.get('value') is not None:
                turn['value'] = re.sub('<image>', f'{DEFAULT_IMAGE_TOKEN}\n', turn['value'])
        list_data_dict.append(record)
        sources = preprocess_multimodal(
            copy.deepcopy(list_data_dict), multimodal_cfg, num_media_latents
        )  # HARDCODED FOR NOW
        data_dict = preprocess_llama_2(sources, tokenizer, multimodal_cfg, is_mistral=True)
    elif multimodal_cfg["conv_template"] == "v1":
        record = {
            'conversations': [
                {
                    'from': 'human',
                    'value': prompt,
                },
                {
                    'from': 'gpt',
                    'value': '',
                },
            ],
        }

        for turn in record['conversations']:
            if turn.get('value') is not None:
                turn['value'] = re.sub('<image>', f'{DEFAULT_IMAGE_TOKEN}\n', turn['value'])
        list_data_dict.append(record)

        sources = preprocess_multimodal(
            copy.deepcopy(list_data_dict), multimodal_cfg, num_media_latents
        )  # HARDCODED FOR NOW
        data_dict = preprocess_v1(sources, tokenizer, multimodal_cfg)
    else:
        raise ValueError(f"Conversation template `{conv_template}` is not supported in Neva now.")
    return data_dict['tokens'].tolist()


class NevaModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model
        self.tokenizer = self.model.tokenizer
        self.image_paths = []
        self.cfg = self.model.cfg
        self.data_cfg = self.model.cfg.data

        add_extra_token = 0
        self.multimodal_cfg = dict(
            is_multimodal=self.data_cfg.is_multimodal,
            sep_image_conv_front=self.data_cfg.sep_image_conv_front,
            conv_template=self.data_cfg.get("conv_template", "nvgpt"),
            model_type=self.cfg.mm_cfg.llm.get("model_type", "nvgpt"),
            patch_dim=self.cfg.mm_cfg.vision_encoder.patch_dim,
            crop_size=self.cfg.mm_cfg.vision_encoder.get("crop_size", None),
            image_folder=self.data_cfg.get('image_folder', None),
            video_folder=self.data_cfg.get('video_folder', None),
            image_aspect_ratio=self.data_cfg.image_aspect_ratio,
            use_im_start_end=getattr(self.cfg.mm_cfg, 'use_im_start_end', False),
            image_processor=None,
            add_extra_token=add_extra_token,
            context_length=self.cfg.encoder_seq_length,
            media_type=getattr(self.data_cfg, 'media_type', 'image'),
            num_frames=getattr(self.data_cfg, 'num_frames', 1),
            mm_mlp_adapter_type=getattr(self.cfg.mm_cfg, 'mm_mlp_adapter_type', 'linear'),
            use_lita=getattr(self.cfg.mm_cfg, 'use_lita', False),
        )
        if self.multimodal_cfg['crop_size'] is None:
            image_processor = CLIPImageProcessor.from_pretrained(
                self.cfg.mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16
            )
            self.multimodal_cfg['crop_size'] = (
                image_processor.crop_size['height'],
                image_processor.crop_size['width'],
            )

        patch_dim = self.multimodal_cfg['patch_dim']
        height_num_patches = self.multimodal_cfg['crop_size'][0] // patch_dim
        width_num_patches = self.multimodal_cfg['crop_size'][1] // patch_dim

        if self.multimodal_cfg['mm_mlp_adapter_type'] == 'mlp_downsample':
            if height_num_patches % 2 != 0:
                height_num_patches += 1
            if width_num_patches % 2 != 0:
                width_num_patches += 1

        self.num_media_latents = height_num_patches * width_num_patches
        # add config for lita
        if self.multimodal_cfg['use_lita']:
            if self.cfg.mm_cfg.get('lita'):
                lita = {
                    'lita_video_arch': getattr(self.cfg.mm_cfg.lita, 'lita_video_arch', 'temporal_spatial_pool'),
                    'visual_token_format': getattr(self.cfg.mm_cfg.lita, 'visual_token_format', 'v1'),
                    'sample_frames': getattr(self.cfg.mm_cfg.lita, 'sample_frames', 1),
                }
                self.multimodal_cfg['lita'] = lita
            else:
                self.multimodal_cfg['use_lita'] = False
                raise Warning(
                    'Use lita has been set True but Lita config not found in the config file'
                    'LITA will be disabled for this run.'
                )

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.cfg.encoder_seq_length + 1:
            maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        # Get the attention mask and postition ids.
        self.attention_mask, _, self.position_ids = get_ltor_masks_and_position_ids(
            tokens,
            eod_token=tokenizer.eos_id,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
            compute_attention_mask=compute_attention_mask,
        )

    def tokenize_batch(self, prompt, max_len, add_BOS):

        if type(prompt) == str:
            context_tokens = neva_process_prompts(
                prompt,
                self.tokenizer,
                self.multimodal_cfg,
                self.num_media_latents,
                self.multimodal_cfg['conv_template'],
            )
        elif type(prompt) == list:
            context_tokens = []
            for p in prompt:
                context_tokens.append(
                    neva_process_prompts(
                        p,
                        self.tokenizer,
                        self.multimodal_cfg,
                        self.num_media_latents,
                        self.multimodal_cfg['conv_template'],
                    )[0]
                )
        else:
            raise ValueError(f'{type(prompt)} is not supported for tokenization')

        context_tokens, context_lengths = pad_batch(context_tokens, self.tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
        media=None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        """
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)
            media = None

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())
        batch = [tokens2use, attention_mask_repeat, positions2use, media, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


class PromptLearningModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model, task_ids):
        super().__init__(model)
        self.task_ids = task_ids
        self.forward_model = self.model

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        # Get the attention mask and postition ids.
        self.attention_mask, _, self.position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eos_id,
            self.model.cfg.get('reset_position_ids', False),
            self.model.cfg.get('reset_attention_mask', False),
            self.model.cfg.get('eod_mask_loss', False),
            compute_attention_mask=compute_attention_mask,
        )

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.frozen_model.cfg.encoder_seq_length + 1:
            maxlen = self.model.frozen_model.cfg.encoder_seq_length + 1
        return maxlen

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            positions2use = self.position_ids[:, :context_length]
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, context_length - 1].view(micro_batch_size, -1)
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, attention_mask_repeat, positions2use, self.task_ids, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.frozen_model.cfg.hidden_size]
        return batch, tensor_shape

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        """
        # Replace special soft prompt token ids with unk token ids
        if (
            self.model.pseudo_token_ids_start is not None
        ):  # TODO: (@adithyare) prompt learning logic can be greatly simplified by removing data preparation logic from model logic.
            tokenizer = self.model.tokenizer
            pseudo_token_ids_start = self.model.pseudo_token_ids_start
            new_tokens[(new_tokens >= pseudo_token_ids_start)] = tokenizer.unk_id
            tokens[:, :context_length][(tokens[:, :context_length] >= pseudo_token_ids_start)] = tokenizer.unk_id


class McoreRetroModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""

        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

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
        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]

        # attention, not pad_batch, padding will be done at init_batch
        context_tokens, context_lengths = pad_batch(batch=context_tokens, pad_id=tokenizer.eos_id, max_len=0)

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def tokenize_neighbors_batch(self, neighbors, retro_args):
        tokenizer = self.model.tokenizer
        r = retro_args['retro_gpt_retrieved_length']
        retro_num_neighbors = retro_args['retro_num_neighbors']
        ft_neighbours = retro_args['ft_neighbours']
        reuse_top = retro_args['reuse_top']

        padded_valid_neighbours_tokens = []
        for i in range(len(neighbors)):
            onesample_neighbors = neighbors[i]

            # tokenize neighbors
            onesample_neighbors_tokens = []
            for neighbor in onesample_neighbors:
                onesample_neighbors_tokens.append(tokenizer.text_to_ids(neighbor))

            # take top k neighbours
            if reuse_top:
                valid_onesample_neighbours_tokens = onesample_neighbors_tokens[:retro_num_neighbors]
            else:
                valid_onesample_neighbours_tokens = onesample_neighbors_tokens[
                    ft_neighbours : retro_num_neighbors + ft_neighbours
                ]

            # pad neighbors
            padded_valid_onesample_neighbours_tokens = []
            for neighbour_tokens in valid_onesample_neighbours_tokens:
                if len(neighbour_tokens) >= r:
                    padded_onesample_neighbour_tokens = neighbour_tokens[:r]
                else:
                    padded_onesample_neighbour_tokens = neighbour_tokens + [tokenizer.eos_id] * (
                        r - len(neighbour_tokens)
                    )
                padded_valid_onesample_neighbours_tokens.append(padded_onesample_neighbour_tokens)

            # check if have enough neighbors
            if len(padded_valid_onesample_neighbours_tokens) < retro_num_neighbors:
                assert ValueError("neighbours are not enough, add empty ones and create mask for those empty ones")

            # append to batch
            padded_valid_neighbours_tokens.append(padded_valid_onesample_neighbours_tokens)

        # cast to torch tensor
        padded_valid_neighbours_tokens = torch.cuda.LongTensor(padded_valid_neighbours_tokens)
        padded_valid_neighbours_tokens_shape = torch.cuda.LongTensor(padded_valid_neighbours_tokens.shape)

        return padded_valid_neighbours_tokens, padded_valid_neighbours_tokens_shape

    def init_batch(self, context_tokens: torch.Tensor, context_length: int, compute_attention_mask: bool, **extra):
        """initialize the batch data before the inference steps."""

        # For Mcore retrieval RETRO model, modify tokens and neighbors to set them into 2 chunks, one for question, and one for answer, both having the same length of context_tokens.shape[1]
        bs, context_tokens_length = context_tokens.shape
        assert bs == 1  # similar to M-LM RETRO inference code, currently only support batch_size=1
        context_tokens = [context_tokens[0].tolist() + [self.model.tokenizer.eos_id] * context_tokens_length]
        context_tokens = torch.cuda.LongTensor(context_tokens)
        self.model.model.config.retro_gpt_chunk_length = context_tokens_length  # set RetroConfig of M-LM's RETRO model
        # reshape tensor extra['neighbors_tokens'] (currently: [k, 1, r]) to [bs, l, k, r]
        neighbors_tokens = extra['neighbors_tokens']
        neighbors_tokens = neighbors_tokens.permute(1, 0, 2)
        neighbors_tokens = neighbors_tokens.unsqueeze(0)
        # duplicate into 2 chunks from [bs, l, k ,r] to [bs, 2*l, k ,r]
        neighbors_tokens = neighbors_tokens.repeat(1, 2, 1, 1)

        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        neighbors_tokens = neighbors_tokens.contiguous().cuda()

        # Get the attention mask and postition ids.
        self.attention_mask, _, self.position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eos_id,
            self.model.cfg.get('reset_position_ids', False),
            self.model.cfg.get('reset_attention_mask', False),
            self.model.cfg.get('eod_mask_loss', False),
            compute_attention_mask=compute_attention_mask,
        )

        # Get the attention mask and postition ids for neighbors (retro_generation.retro_generate_tokens_probs_and_return_on_first_stage)
        # Reshape neighbors_tokens tensor to 2D for get_ltor_masks_and_position_ids and as forward arg of RETRO model, original shape is 3D ([bs, k, r])
        [bs, l, k, r] = neighbors_tokens.shape
        neighbors_tokens = neighbors_tokens.view(-1, r).long()

        _, _, self.neighbor_position_ids = get_ltor_masks_and_position_ids(
            neighbors_tokens,
            tokenizer.eos_id,
            self.model.cfg.get('reset_position_ids', False),
            self.model.cfg.get('reset_attention_mask', False),
            self.model.cfg.get('eod_mask_loss', False),
        )
        self.neighbor_attention_mask = torch.zeros(
            [1, 1]
        )  # dummy value, since the batch neighbor_attention_mask will be set to None in megatron_retro_model.py in Mcore implementation
        self.neighbors_tokens = neighbors_tokens

        # For Mcore retrieval RETRO model, following ADLR's Mcore RETRO inferencing implementation, updating the arguments inside RETRO model (retro_num_neighbors, retro_chunk_length) with the inference's sample
        inference_retro_num_neighbors = k
        inference_retro_chunk_length = context_tokens_length
        inference_retro_retrieved_length = r
        self.forward_model.config.retro_num_neighbors = inference_retro_num_neighbors
        self.forward_model.config.retro_chunk_length = inference_retro_chunk_length
        self.forward_model.config.retro_retrieved_length = inference_retro_retrieved_length
        contain_encoder = True
        if isinstance(self.forward_model, (Float16Module, MCoreFloat16Module)):
            layers = self.forward_model.module.decoder.layers
        else:
            layers = self.forward_model.decoder.layers
        for layer in layers:
            if not (isinstance(layer.cross_attention, IdentityOp)):  # if this is encoder-decoder cross-attention layer
                # updating RetroDecoder (RetroDecoderCrossAttention, RetroDecoderBiasDropoutAdd)
                layer.cross_attention.retro_num_neighbors = inference_retro_num_neighbors
                layer.cross_attention.retro_chunk_length = inference_retro_chunk_length
                layer.cross_attention.retro_retrieved_length = inference_retro_retrieved_length
                layer.cross_attn_bda.retro_chunk_length = inference_retro_chunk_length

                # updating RetroEncoder (RetroEncoderCrossAttention, RetroEncoderBiasDropoutAdd, RetroEncoderLayerNorm)
                if contain_encoder:  # the first cross-attention decoder layer contain encoder
                    layer.cross_attention.encoder.layers[0].cross_attention.retro_num_neighbors = (
                        inference_retro_num_neighbors
                    )
                    layer.cross_attention.encoder.layers[0].cross_attention.retro_chunk_length = (
                        inference_retro_chunk_length
                    )
                    layer.cross_attention.encoder.layers[0].cross_attention.retro_retrieved_length = (
                        inference_retro_retrieved_length
                    )
                    layer.cross_attention.encoder.layers[0].cross_attn_bda.retro_num_neighbors = (
                        inference_retro_num_neighbors
                    )
                    layer.cross_attention.encoder.layers[0].pre_mlp_layernorm.retro_num_neighbors = (
                        inference_retro_num_neighbors
                    )
                    contain_encoder = False

        return context_tokens

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_length: int,
        compute_attention_mask: bool = True,
        **extra,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        generate the batch used in inference for each of the steps
        """

        # For Mcore retrieval RETRO model, currently not support memory caching, always allocate memory for the entire context
        # Allocate memory for the entire context.
        set_inference_key_value_memory = True
        tokens2use = tokens
        positions2use = self.position_ids
        attention_mask2use = self.attention_mask

        """Prepare batch for each of the inference steps"""
        attention_mask_repeat = None
        if compute_attention_mask:
            attention_mask_repeat = torch.concat([attention_mask2use for _ in range(micro_batch_size)])

        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [
            tokens2use,
            attention_mask_repeat,
            positions2use,
            self.neighbors_tokens,
            self.neighbor_attention_mask,
            self.neighbor_position_ids,
            setkey_value_array,
            len_array,
        ]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
        MegatronGPTPromptLearningModel,
    )
    from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
    from nemo.collections.nlp.models.language_modeling.megatron_mamba_model import MegatronMambaModel
    from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
    from nemo.collections.nlp.models.language_modeling.megatron_retro_model import MegatronRetroModel
    from nemo.collections.nlp.modules.common.retro_inference_strategies import (
        RetroFileQAModelTextGenerationStrategy,
        RetroModelTextGenerationStrategy,
        RetroQAModelTextGenerationStrategy,
    )

    if isinstance(model, MegatronGriffinModel):
        return GriffinModelTextGenerationStrategy(model)
    if isinstance(model, MegatronMambaModel):
        return GPTModelTextGenerationStrategy(model)
    if isinstance(model, MegatronNevaModel):
        return NevaModelTextGenerationStrategy(model)
    if isinstance(model, MegatronGPTPromptLearningModel):
        return PromptLearningModelTextGenerationStrategy(model, **args)
    elif isinstance(model, MegatronGPTModel) and not (isinstance(model, MegatronRetroModel)):
        return GPTModelTextGenerationStrategy(model)
    elif isinstance(model, MegatronRetrievalModel):
        strategy_name = args['strategy']
        del args['strategy']
        megatron_lm_compatible = model.model.megatron_lm_compatible
        args['megatron_lm_compatible'] = megatron_lm_compatible
        if strategy_name == 'RetroModelTextGenerationStrategy':
            return RetroModelTextGenerationStrategy(model, **args)
        elif strategy_name == 'RetroQAModelTextGenerationStrategy':
            return RetroQAModelTextGenerationStrategy(model, **args)
        elif strategy_name == 'RetroFileQAModelTextGenerationStrategy':
            return RetroFileQAModelTextGenerationStrategy(model, **args)
        else:
            raise ValueError(f'{strategy_name} is not supported for inference')
    elif isinstance(model, MegatronRetroModel):
        return McoreRetroModelTextGenerationStrategy(model)
    else:
        raise ValueError(f'{model} is not supported for inference')

    # Should call GPTModel or Megatron Retrieval Model's forward method
