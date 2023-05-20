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

__all__ = ['GPTSFTChatDataset']

IGNORE_INDEX = -100
END_SIGNAL = "\n"
END_NAME_SIGNAL = "\n"

SYSTEM_TOKEN = "<extra_id_0>System\n"
TURN_TOKEN = "<extra_id_1>"

GUARD_RAIL_INSTRUCTION = {
    "TEXT_TO_CANONICAL_FORM": "Given a dialogue, for each turn you need to generate a short summary called a canonical form. Generate the canonical form for the last turn in the dialogue.",
    "CANONICAL_FORM_TO_TEXT": "Given a dialogue, for each turn we also have a short summary called a canonical form. Generate the canonical form given the last turn message and canonical form. Then generate the message.",
}


def _mask_targets(target, tokenized_lens, speakers, header_len, s_ids, tokenizer, mask_role):
    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. s_id has that extra token too
        skip_name_len = len(tokenizer.text_to_ids(TURN_TOKEN + speaker + END_NAME_SIGNAL))
        if cur_idx >= tgt_len:
            break
        elif cur_idx + tokenized_len < tgt_len:
            # Check whether the mask is applied to the correct position, the first token is turn token: <extra_id_1>
            # s_id[2:] skips the artifact empty token and the turn token
            # target[cur_idx + 1:cur_idx + tokenized_len] skip the turn token
            if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[2:]):
                logging.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0:
            # mask the first turn completely to provide at least one turn as context
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role:
            # leave the first human tag unmasked
            target[cur_idx + 1 : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask up to the name end, need to remove one as skip name has an extra artifact empty token
            target[cur_idx : cur_idx + skip_name_len - 1] = IGNORE_INDEX
        cur_idx += tokenized_len


def cannonical_form_formater(cannoical_form):
    return f'<extra_id_2>{cannoical_form}\n'


def _add_speaker_and_signal(header, source, mask_role, gtype):
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
        elif gtype == "TEXT_TO_CANONICAL_FORM":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + cannonical_form_formater(sentence['canonical_form'])
            )
        elif gtype == "CANONICAL_FORM_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + cannonical_form_formater(sentence['canonical_form'])
                + sentence["value"]
                + END_SIGNAL
            )
        else:
            raise ValueError(f"source type {gtype} not supported")
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
        if sentence_from != mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation


def preprocess(
    source: dict, tokenizer: TokenizerSpec,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    canonical_type = None
    if 'type' in source:
        canonical_type = source['type']
        assert canonical_type in GUARD_RAIL_INSTRUCTION, f"source type {canonical_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if canonical_type is not None:
        conversation = conversation + '\n' + GUARD_RAIL_INSTRUCTION[canonical_type]
    mask_role = source.get('mask', 'User')
    header = f"{SYSTEM_TOKEN}{conversation}\n\n"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, canonical_type)
    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    header_len = len(tokenizer.text_to_ids(header))

    ids = []
    tokenized_lens = []
    for s in source['conversations']:
        tokenized_sentence = tokenizer.text_to_ids(s["value"])
        ids.append(torch.tensor(tokenized_sentence))
        # remove one token as it adds an empty token in front
        tokenized_lens.append(len(tokenized_sentence) - 1)
    speakers = [sentence["from"] for sentence in source['conversations']]
    assert mask_role in speakers, "mask role not in the conversation"
    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)

    _mask_targets(target, tokenized_lens, speakers, header_len, ids, tokenizer, mask_role)
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    return dict(input_ids=input_ids, mask=mask)


class GPTSFTChatDataset(GPTSFTDataset):
    def _build_samples_mapping(self):
        super()._build_samples_mapping()
        assert hasattr(self.tokenizer, "vocab"), "tokenizer should have vocab property, not supported"
        assert '<extra_id_0>' in self.tokenizer.vocab, "<extra_id_0> not in the tokenizer vocab. not supported"
        assert '<extra_id_1>' in self.tokenizer.vocab, "<extra_id_1> not in the tokenizer vocab. not supported"

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        result = preprocess(example, self.tokenizer)

        return result

    def collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1].tolist() for item in batch]
        labels = [item['input_ids'][1:].tolist() for item in batch]
        loss_mask = [item['mask'][1:].tolist() for item in batch]

        max_length = max([len(x) for x in input_ids])
        if max_length > self.max_seq_length:
            # truncate the sequences if it is longer than max_seq_length
            input_ids = [x[: self.max_seq_length] for x in input_ids]
            labels = [x[: self.max_seq_length] for x in labels]
            loss_mask = [x[: self.max_seq_length] for x in loss_mask]
        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask(max_length) for _ in batch]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))

        processed_batch = {
            'tokens': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
        }

        return processed_batch
