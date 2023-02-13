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

import json
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from nemo.collections.nlp.modules.common.lm_utils import pad_batch
from nemo.collections.nlp.modules.common.megatron.retrieval_services.retrieval_service import ComboRetrievalService
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy


class RetroModelTextGenerationStrategy(TextGenerationStrategy):
    def __init__(self, model, **args):
        super().__init__(model)
        self.forward_model = self.model.model
        self.frequent_query = args['frequent_query']
        self.pad_token_for_retrieval = args['pad_tokens']
        self.store_retrieved = args['store_retrieved']
        self.store = dist.FileStore('/tmp/filestore_eval', -1)
        self.store.set('neighbors', str(args['neighbors']))
        self.megatron_lm_compatible = args['megatron_lm_compatible']
        combo_cfg = args['combo_service']
        self.service = ComboRetrievalService(
            tokenizer=self.model.tokenizer, service_ip=combo_cfg['service_ip'], service_port=combo_cfg['service_port']
        )
        self.retrieved = []
        self.retrieved_text = []
        self.chunk_size = self.model.cfg.chunk_size

    def update_neighbors(self, neighbors):
        # dynamically change the number of neighbors during the query
        self.store.set('neighbors', str(neighbors))

    @property
    def neighbors(self):
        return int(self.store.get('neighbors'))

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
                    if self.megatron_lm_compatible:
                        # megatron lm use eos to pad
                        padded.append([tokenizer.eos_id] * pad_len + line)
                    else:
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
                    if self.megatron_lm_compatible:
                        # megatron lm use eos to pad
                        padded.append([tokenizer.eos_id] * pad_len + line[0] + line[1])
                    else:
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
        micro_batch_size, seq_length = tokens.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        self.position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        if self.megatron_lm_compatible:
            # all TRUE for megatron lm, there is no attention mask
            self.attention_mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
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
        retrieved = torch.tensor(np.array(self.retrieved), device=torch.cuda.current_device())
        if retrieved.numel() != 0:
            retrieved = retrieved.transpose(0, 1).contiguous()
        if self.megatron_lm_compatible:
            # all TRUE for megatron lm, there is no attention mask
            retrieved_mask = torch.ones_like(retrieved, dtype=torch.bool)
        else:
            retrieved_mask = retrieved != tokenizer.pad_id
        if retrieved.numel() == 0:
            # add empty retrieved
            retrieved = (
                torch.tensor(self.service.get_knn(['a'], 0), device=torch.cuda.current_device())
                .unsqueeze(0)
                .repeat(1, len(self.retrieved), 1, 1)
            )
            retrieved_mask = retrieved != tokenizer.pad_id
            # retrieved = torch.tensor([-1] * micro_batch_size)
            # retrieved_mask = torch.tensor([-1] * micro_batch_size)

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
            positions2use,
        ]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


class RetroQAModelTextGenerationStrategy(RetroModelTextGenerationStrategy):
    def tokenize_batch(self, questions, max_len, add_BOS):
        """
        convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
        Args:
            questions (List[str]): list of input questions in str format.
            max_len (int): max number of tokens to generate.
            add_BOS (bool): whether to add the BOS token at the beginning
        Returns:
            Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
        """
        tokenizer = self.model.tokenizer
        all_lookups = self.service.get_knn(questions, 1 + self.neighbors)
        # hack to add "source: " tag
        prepend_ids = np.array(tokenizer.text_to_ids('source: '))
        all_lookups = np.pad(all_lookups, ((0, 0), (0, 0), (len(prepend_ids), 0)))
        all_lookups[:, :, : len(prepend_ids)] = prepend_ids
        all_lookups = all_lookups[:, :, : -len(prepend_ids)]
        reuse_neighbors = all_lookups[:, 1:]
        self.store.set('reuse_neighbors', pickle.dumps(reuse_neighbors))
        neighbor_tokens = [neighbors[0].tolist() for neighbors in all_lookups]

        # combine question and context
        context_tokens = [
            n + tokenizer.text_to_ids('\nquestion: ' + q + ' \nanswer:') for n, q in zip(neighbor_tokens, questions)
        ]

        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + s for s in context_tokens]
        if self.pad_token_for_retrieval:
            padded = []
            for line in context_tokens:
                pad_len = (self.chunk_size - len(line) % self.chunk_size) % self.chunk_size
                if self.megatron_lm_compatible:
                    padded.append([tokenizer.eos_id] * pad_len + line)
                else:
                    padded.append([tokenizer.pad_id] * pad_len + line)
            context_tokens = padded
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor

    def init_batch(self, context_tokens: torch.Tensor, context_length: int):
        self.retrieved = []
        self.retrieved_text = []
        self.reuse_neighbors = pickle.loads(self.store.get('reuse_neighbors'))
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        tokenizer = self.model.tokenizer
        tokens = context_tokens.contiguous().cuda()
        micro_batch_size, seq_length = tokens.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        self.position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        if self.megatron_lm_compatible:
            # all TRUE for megatron lm, there is no attention mask
            self.attention_mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            self.attention_mask = tokens != tokenizer.pad_id
        for i in range(0, context_length, 64):
            if i > 0:
                tokens = context_tokens[:, i - 64 : i]
                chunks = self.reuse_neighbors
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
            chunks = self.reuse_neighbors
            if self.store_retrieved:
                self._store_retrieved(token_context, chunks)
            self.retrieved.append(chunks)
        elif self.frequent_query and len(self.retrieved) > 0:
            token_context = tokens[:, context_length - 64 : context_length]
            chunks = self.reuse_neighbors
            if self.store_retrieved:
                self._store_retrieved(token_context, chunks)
            self.retrieved[-1] = chunks

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
        retrieved = torch.tensor(np.array(self.retrieved), device=torch.cuda.current_device())
        if retrieved.numel() != 0:
            retrieved = retrieved.transpose(0, 1).contiguous()
        if self.megatron_lm_compatible:
            # all TRUE for megatron lm, there is no attention mask
            retrieved_mask = torch.ones_like(retrieved, dtype=torch.bool)
        else:
            retrieved_mask = retrieved != tokenizer.pad_id
        if retrieved.numel() == 0:
            # add empty retrieved
            retrieved = (
                torch.tensor(self.service.get_knn(['a'], 0), device=torch.cuda.current_device())
                .unsqueeze(0)
                .repeat(1, len(self.retrieved), 1, 1)
            )
            retrieved_mask = retrieved != tokenizer.pad_id

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
            positions2use,
        ]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

    def post_generation_process(self, output):
        sentences = output['sentences']
        modified = []
        for sentence in sentences:
            sentence = 'answer:' + sentence.split(' \nanswer:')[1]
            modified.append(sentence)
        output['sentences'] = modified
        return output


class RetroFileQAModelTextGenerationStrategy(RetroQAModelTextGenerationStrategy):
    def __init__(self, model, **args):
        super().__init__(model, **args)
        # load the DPR to memory
        self.context_db = {}
        with open('/dataset/FiD/test.jsonl_title', 'r') as f:
            for line in f:
                obj = json.loads(line)
                self.context_db[obj['question']] = obj

    def tokenize_batch(self, questions, max_len, add_BOS):
        """
        convert the sentences into lists of tokens, pad them to the same length, add bos tokens if it is needed
        Args:
            questions (List[str]): list of input questions in str format.
            max_len (int): max number of tokens to generate.
            add_BOS (bool): whether to add the BOS token at the beginning
        Returns:
            Tuple[torch.Tensor], the tokenized and padded torch tensor and the token context length tensor.
        """

        tokenizer = self.model.tokenizer

        # get context from memory
        chunks = []
        first_context = []
        for question in questions:
            hash_code = question
            if hash_code not in self.context_db:
                raise ValueError(f"wrong question is fed: {question}")
            contexts = self.context_db[hash_code]['ctxs']
            for i, neighbor in enumerate(contexts[: self.neighbors + 1]):
                text = "title: " + neighbor["title"] + ", source: " + neighbor["text"]
                if i == 0:
                    first_context.append(text)
                tokens = tokenizer.text_to_ids(text)
                tokens = tokens[:128]
                if len(tokens) < 128:
                    tokens = tokens + [tokenizer.eos_id] * (128 - len(tokens))
                chunks.append(tokens)
        all_lookups = np.array(chunks).reshape(1, self.neighbors + 1, -1).astype(np.int64)
        reuse_neighbors = all_lookups[:, 1:]
        self.store.set('reuse_neighbors', pickle.dumps(reuse_neighbors))
        # combine question and context
        context_tokens = [
            tokenizer.text_to_ids(n + '\nquestion: ' + q + ' \nanswer:') for n, q in zip(first_context, questions)
        ]

        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + s for s in context_tokens]
        if self.pad_token_for_retrieval:
            padded = []
            for line in context_tokens:
                pad_len = (self.chunk_size - len(line) % self.chunk_size) % self.chunk_size
                padded.append([tokenizer.eos_id] * pad_len + line)
            context_tokens = padded
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)
        return context_tokens_tensor, context_length_tensor
