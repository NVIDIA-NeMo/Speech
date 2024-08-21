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

import copy

import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.utils import logging

__all__ = ['GPTSFTChatDataset', 'get_prompt_template_example']


PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

IGNORE_INDEX = -100
SYSTEM_TOKEN = "System"

TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}


def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    data_type = None
    if 'type' in source:
        data_type = source['type']
        if data_type is not None:
            assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if data_type is not None:
        if TYPE_INSTRUCTION[data_type] != '':
            conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
    mask_role = source.get('mask', 'User')
    header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)
    return header, conversation, data_type, mask_role


def get_prompt_template_example(special_tokens):
    source = {
        'system': '{system message}',
        'conversations': [
            {'from': 'User', 'value': '{turn 1 user message}', 'label': None},
            {'from': 'Assistant', 'value': '{turn 1 assistant message}', 'label': '{turn 1 assistant label}'},
            {'from': 'User', 'value': '{turn 2 user message}', 'label': None},
            {'from': 'Assistant', 'value': '{turn 2 assistant message}', 'label': '{turn 2 assistant label}'},
        ],
        "mask": "User",
        "type": "VALUE_TO_TEXT",
    }
    _, conversation, _, _ = _get_header_conversation_type_mask_role(source, special_tokens)
    return conversation


def identify_start_index_of_subsequence(subsequence, sequence):
    """find the location of the small tensor in the large tensor.
        e.g.  small = [1,3], large = [2,3,1,3], returns 2
              small = [3,2], large = [2,3,1,3], returns -1
    Args:
        small (tensor): small tensor
        large (tensor): large tensor
    """
    for i in range(sequence.size(0) - subsequence.size(0) + 1):
        if torch.equal(sequence[i : i + subsequence.size(0)], subsequence):
            return i
    return -1


def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    gtype,
    name_end_token_ids,
    special_tokens,
    label_start_ids,
    num_turn_start_tokens,
):
    """This function masks the tokens so the loss is computed only on the non-masked role's responses.
    For 'TEXT_TO_VALUE' type, the loss is computed on the value attributes.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation
        gtype (str): either 'TEXT_TO_VALUE' or 'VALUE_TO_TEXT'
        name_end_token_ids (int): end of name token ids
        special_tokens (dict): special tokens used for the chat prompt. It has the keys: system_turn_start, turn_start, label_start, end_of_turn
        label_start_ids (list): list of label start token ids,
        num_turn_start_tokens (int): number of tokens of the turn_start str
    """
    TURN_TOKEN = special_tokens['turn_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    label_start_ids = torch.tensor(label_start_ids)
    name_end_token_ids = torch.tensor(name_end_token_ids)

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. has to compute the diff
        id1 = tokenizer.text_to_ids(PREFIX_STR)
        id2 = tokenizer.text_to_ids(PREFIX_STR + TURN_TOKEN + speaker + END_NAME_SIGNAL)
        skip_name_len = len(id2) - len(
            id1
        )  # s_ids[:skip_name_len] is the name part of the prompt 'TURN_TOKEN + speaker + END_NAME_SIGNAL'
        # get the position of the label start string in this turn
        location = identify_start_index_of_subsequence(label_start_ids, s_id)

        if location >= 0:
            # if it contains the label start tokens
            if gtype == 'VALUE_TO_TEXT':
                # handles the case that condition on labels to generate respone
                # the next token after the name part of the prompt is the beginning of the label start tokens
                assert skip_name_len == location
                # find the first new line token after the label part, which indicates the end of the whole label string
                # newline_loc = torch.where((s_id[skip_name_len:] == name_end_token_ids))[0]
                newline_loc = identify_start_index_of_subsequence(name_end_token_ids, s_id[skip_name_len:])
                if newline_loc < 0:
                    # cannot find new line token, which means the the whole turn is just a partial label string. Mask the whole turn
                    target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
                    continue
                # skip the label part and the new line token
                more_skip_len = newline_loc + len(name_end_token_ids)
                # skip the name part and the label part
                skip_name_len += more_skip_len
            elif gtype == 'TEXT_TO_VALUE':
                # handles the case that condition on response to generate label
                # skip the name part, response and the label start tokens part, the remainder is the label string without label start, e.g. 'quality:9,toxicity:8...'
                skip_name_len = location + len(label_start_ids)
        if cur_idx >= tgt_len:
            break
        elif cur_idx + tokenized_len < tgt_len:
            # Check whether the mask is applied to the correct position, the first token is turn start tokens
            if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[1:]):
                logging.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0 and (gtype == 'VALUE_TO_TEXT' or gtype is None):
            # mask the first turn completely to provide at least one turn as context for the rest
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and i == 1 and gtype == 'TEXT_TO_VALUE':
            # leave the first turn start tag unmasked, servers severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i > 1):
            # leave the first turn start tag unmasked, which severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i <= 1):
            # mask out everything in the second turn
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask up to name part, label part for VALUE_TO_TEXT, or name part, response and label start tokens for TEXT_TO_VALUE, or just the name part if gtype is None
            target[cur_idx : cur_idx + skip_name_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')


def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = ""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        role_token = TURN_TOKEN
        if gtype is None:
            sentence["value"] = (
                BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
            )
        elif gtype == "VALUE_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
                + sentence["value"]
                + END_SIGNAL
            )
        elif gtype == "TEXT_TO_VALUE":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
            )
        else:
            raise ValueError(
                f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
            )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
        if sentence_from != mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation


def preprocess(
    source: dict,
    tokenizer: TokenizerSpec,
    name_end_token_ids: int,
    label_start_ids: list,
    special_tokens: dict,
    num_turn_start_tokens: int,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    header_tokens = tokenizer.text_to_ids(header)
    header_len = len(header_tokens)

    ids = []
    tokenized_lens = []
    assert torch.equal(torch.tensor(target[:header_len]), torch.tensor(header_tokens))
    for s in source['conversations']:
        # hack to remove the extra empty token in front
        id1 = tokenizer.text_to_ids(PREFIX_STR + s["value"])
        id2 = tokenizer.text_to_ids(PREFIX_STR)
        tokenized_sentence = id1[len(id2) :]
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source['conversations']]
    assert mask_role in speakers, "mask role not in the conversation"
    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)
    _mask_targets(
        target,
        tokenized_lens,
        speakers,
        header_len,
        ids,
        tokenizer,
        mask_role,
        data_type,
        name_end_token_ids,
        special_tokens,
        label_start_ids,
        num_turn_start_tokens,
    )
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    # Choose the last conversation as answer other history are context
    last_ignore_index_pos = torch.nonzero(target == IGNORE_INDEX)[-1].item() + 1
    context_ids = input_ids[:last_ignore_index_pos]
    answer_ids = input_ids[last_ignore_index_pos:]
    return dict(input_ids=input_ids, mask=mask, context_ids=context_ids, answer_ids=answer_ids)


class GPTSFTChatDataset(GPTSFTDataset):
    def _maybe_validate_prompt_template(self):
        pass

    def _build_samples_mapping(self):
        super()._build_samples_mapping()
        assert hasattr(self.tokenizer, "vocab"), "tokenizer should have vocab property, not supported"
        LABEL_START = self.special_tokens['label_start']
        END_NAME_SIGNAL = self.special_tokens['end_of_name']

        id1 = self.tokenizer.text_to_ids(PREFIX_STR)
        id2 = self.tokenizer.text_to_ids(PREFIX_STR + LABEL_START)
        self.label_start_tokens = id2[len(id1) :]

        id1 = self.tokenizer.text_to_ids(PREFIX_STR + END_NAME_SIGNAL)
        id2 = self.tokenizer.text_to_ids(PREFIX_STR)
        self.name_end_token_ids = id1[len(id2) :]

        id1 = self.tokenizer.text_to_ids(PREFIX_STR + self.special_tokens['turn_start'])
        id2 = self.tokenizer.text_to_ids(PREFIX_STR)
        self.num_turn_start_tokens = len(id1) - len(id2)

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        result = preprocess(
            example,
            self.tokenizer,
            self.name_end_token_ids,
            self.label_start_tokens,
            self.special_tokens,
            self.num_turn_start_tokens,
        )

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in ['conversations']}
        result['metadata'] = metadata
        if self.output_original_text:
            result['metadata']['conversations'] = example['conversations']

        return result

    def collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1].tolist() for item in batch]
        labels = [item['input_ids'][1:].tolist() for item in batch]
        contexts = [item['context_ids'].tolist() for item in batch]
        answers = [item['answer_ids'].tolist() for item in batch]
        loss_mask = [item['mask'][1:].tolist() for item in batch]
        metadata = [item['metadata'] for item in batch]

        max_length = max(max([len(x) for x in input_ids]), max([len(x) for x in contexts]) + self.tokens_to_generate)
        if max_length > self.max_seq_length:
            # truncate the sequences if it is longer than max_seq_length
            input_ids = [x[: self.max_seq_length] for x in input_ids]
            labels = [x[: self.max_seq_length] for x in labels]
            loss_mask = [x[: self.max_seq_length] for x in loss_mask]
            contexts = [x[: self.max_seq_length] for x in contexts]
            answers = [x[: self.max_seq_length] for x in answers]

        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        if not self.get_attention_mask_from_fusion:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        context_lengths = torch.LongTensor([len(x) for x in contexts])
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=self.tokenizer.eos_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=self.tokenizer.eos_id))

        processed_batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'contexts': contexts,
            'context_lengths': context_lengths,
            'answers': answers,
            'metadata': metadata,
        }

        if not self.get_attention_mask_from_fusion:
            processed_batch['attention_mask'] = attention_mask

        return processed_batch
