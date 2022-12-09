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
from typing import List, Tuple

import torch

from nemo.collections.nlp.modules.common.megatron.retrieval_service import (
    ComboRetrievalService,
    DynamicFaissRetrievalService,
    FaissRetrievalService,
    start_sentence_bert_server,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

try:
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


class TextGenerationStrategy:
    """
    Base class for TextGeneration Strategy
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward_step(self, batch, tensor_shape):

        if self.model.cfg.get('pipeline_model_parallel_size', 1) > 1:
            output_tensor = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.model.get_forward_output_only_func(),
                batch=batch,
                model=self.forward_model,
                forward_only=True,
                tensor_shape=tensor_shape,
                dtype=self.model.autocast_dtype,
                sync_batch_comm=self.model.cfg.get('sync_batch_comm', False),
            )
        else:
            output_tensor = forward_backward_no_pipelining(
                forward_step_func=self.model.get_forward_output_only_func(),
                batch=batch,
                model=self.forward_model,
                forward_only=True,
                tensor_shape=tensor_shape,
                dtype=self.model.autocast_dtype,
                sync_batch_comm=self.model.cfg.get('sync_batch_comm', False),
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
        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + tokenizer.text_to_ids(s) for s in sentences]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    @abc.abstractclassmethod
    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length
        Args:
            maxlen (int): the max len computed from the context and number of tokens to generate
        returns (int):
            the clip the max length based of the LM model max sequence length
        """
        pass

    @abc.abstractclassmethod
    def init_batch(self, context_tokens: torch.Tensor, context_length: int):
        """initialize the batch data before the inference steps.
           It will save the intermediate results as object attributes
           context_length (int): the context token length
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
        At the end of the inference, post process the inference results
        Args:
            tokens  (torch.Tensor): the context tokens
            new_token (torch.Tensor): sampled new token id
            context_length (int): the new token position in the tokens
        """
        pass


class GPTModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model):
        super().__init__(model)
        self.forward_model = self.model.model

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.cfg.encoder_seq_length + 1:
            maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def init_batch(self, context_tokens: torch.Tensor, context_length: int):
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
        )

    def prepare_batch_at_step(
        self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int
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
        attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


class PromptLearningModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model, task_ids):
        super().__init__(model)
        self.task_ids = task_ids
        self.forward_model = self.model

    def init_batch(self, context_tokens: torch.Tensor, context_length: int):
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
        )

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.frozen_model.cfg.encoder_seq_length + 1:
            maxlen = self.model.frozen_model.cfg.encoder_seq_length + 1
        return maxlen

    def prepare_batch_at_step(
        self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int
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


class RetroModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model, **args):
        super().__init__(model)
        self.forward_model = self.model.model
        self.frequent_query = args['frequent_query']
        self.pad_token_for_retrieval = args['pad_tokens']
        self.neighbors = args['neighbors']
        self.store_retrieved = args['store_retrieved']
        weights = args['weights']
        # start the sentence bert server
        start_sentence_bert_server(tokenizer=self.model.tokenizer, **args['sentence_bert'])
        services = []
        for service_conf in args['services']:
            if service_conf['type'] == 'FaissRetrievalService':
                del service_conf['type']
                service = FaissRetrievalService(tokenizer=self.model.tokenizer, **service_conf)
                services.append(service)
            elif service_conf['type'] == 'DynamicFaissRetrievalService':
                del service_conf['type']
                service = DynamicFaissRetrievalService(tokenizer=self.model.tokenizer, **service_conf)
                services.append(service)
            else:
                raise ValueError(f'no such service {service_conf["type"]} implemented')
        self.service = ComboRetrievalService(retrieval_services=services, weights=weights)
        self.retrieved = []
        self.retrieved_text = []
        self.chunk_size = self.service.chunk_size

    def update_neighbors(self, neighbors):
        # dynamically change the number of neighbors during the query
        self.neighbors = neighbors

    def update_weights(self, weights):
        # dynamically change the weights between different retrieval services
        self.service.update_weights(weights)

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
        if self.pad_token_for_retrieval:
            padded = []
            for line in context_tokens:
                if len(line) < self.chunk_size:
                    pad_len = self.chunk_size - len(line)
                    padded.append([tokenizer.pad_id] * pad_len + line)
                else:
                    padded.append(line)
            context_tokens = padded
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def tokenize_batch_with_context_and_completion(self, sentences, max_len, add_BOS):
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
            context_tokens = [
                [[tokenizer.bos_id] + tokenizer.text_to_ids(s[0]), tokenizer.text_to_ids(s[1])] for s in sentences
            ]
        else:
            context_tokens = [[tokenizer.text_to_ids(s[0]), tokenizer.text_to_ids(s[1])] for s in sentences]
        if self.pad_token_for_retrieval:
            padded = []
            for line in context_tokens:
                if len(line[0]) < self.chunk_size:
                    pad_len = self.chunk_size - len(line[0])
                    padded.append([tokenizer.pad_id] * pad_len + line[0] + line[1])
                else:
                    padded.append(line[0] + line[1])
            context_tokens = padded
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def clip_max_len(self, maxlen: int) -> int:
        """ clip the max len based on the LM model max sequence length"""
        if maxlen > self.model.cfg.encoder_seq_length + 1:
            maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def _store_retrieved(self, tokens, neighbors):
        tokenizer = self.model.tokenizer
        for batch_id in range(len(tokens)):
            item = {}
            query_text = tokenizer.ids_to_text(tokens[batch_id])
            item['query'] = query_text
            item['neighbors'] = []
            for context_id in range(len(neighbors[batch_id])):
                neighbor_text = tokenizer.ids_to_text(neighbors[batch_id][context_id])
                item['neighbors'].append(neighbor_text)
            self.retrieved_text.append(item)

    def init_batch(self, context_tokens: torch.Tensor, context_length: int):
        self.retrieved = []
        self.retrieved_text = []
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        self.attention_mask = tokens != tokenizer.pad_id
        for i in range(0, context_length, 64):
            if i > 0:
                tokens = context_tokens[:, i - 64 : i]
                chunks = self.service.get_knn(tokens, self.neighbors)
                if self.store_retrieved:
                    self._store_retrieved(tokens, chunks)
                self.retrieved.append(chunks)

    def prepare_batch_at_step(
        self, tokens: torch.Tensor, maxlen: int, micro_batch_size: int, step: int, context_length: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        tokenizer = self.model.tokenizer

        if context_length % 64 == 0:
            # added a new retrieval context
            token_context = tokens[:, context_length - 64 : context_length]
            chunks = self.service.get_knn(token_context, self.neighbors)
            if self.store_retrieved:
                self._store_retrieved(token_context, chunks)
            self.retrieved.append(chunks)
        elif self.frequent_query and len(self.retrieved) > 0:
            token_context = tokens[:, context_length - 64 : context_length]
            chunks = self.service.get_knn(token_context, self.neighbors)
            if self.store_retrieved:
                self._store_retrieved(token_context, chunks)
            self.retrieved[-1] = chunks

        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :context_length]
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, :context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, context_length - 1].view(micro_batch_size, -1)
            # not using type2use. uncomment it if it is used
            # if type_ids is not None:
            #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)
        retrieved = torch.tensor(self.retrieved, device=torch.cuda.current_device())
        if retrieved.numel() != 0:
            retrieved = retrieved.transpose(0, 1).contiguous()
        retrieved_mask = retrieved != tokenizer.pad_id
        if len(retrieved) == 0:
            retrieved = torch.tensor([-1] * micro_batch_size)
            retrieved_mask = torch.tensor([-1] * micro_batch_size)

        """Prepare batch for each of the inference steps"""
        # attention_mask_repeat = torch.concat([self.attention_mask for _ in range(micro_batch_size)])
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())
        if self.neighbors == 0:
            # no retrieval, use 1 padding
            neighbors_array = torch.tensor([1] * micro_batch_size, device=torch.cuda.current_device())
        else:
            neighbors_array = torch.tensor([self.neighbors] * micro_batch_size, device=torch.cuda.current_device())

        batch = [
            tokens2use,
            self.attention_mask[:, :context_length],
            retrieved,
            retrieved_mask,
            setkey_value_array,
            len_array,
            neighbors_array,
        ]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
        MegatronGPTPromptLearningModel,
    )
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel

    if isinstance(model, MegatronGPTPromptLearningModel):
        return PromptLearningModelTextGenerationStrategy(model, **args)
    elif isinstance(model, MegatronGPTModel):
        return GPTModelTextGenerationStrategy(model)
    elif isinstance(model, MegatronRetrievalModel):
        return RetroModelTextGenerationStrategy(model, **args)
    else:
        raise ValueError(f'{model} is not supported for inference')

    # Should call GPTModel or Megatron Retrieval Model's forward method
