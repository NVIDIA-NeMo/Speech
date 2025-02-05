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

from typing import Union

import torch.utils.data
from lhotse.cut import Cut, CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.common.data.lhotse.text_adapters import NeMoSFTExample, SourceTargetTextExample
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import (
    PromptFormatterTextProcessing,
    build_loss_mask,
)


def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    Args:
        text_processor: TextProcessing object
        default_context: Default question to use if no question is provided
        tokens_to_generate: Number of tokens to generate during inference
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        max_seq_length: Maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        context_key: Key to use for the context in your JSONL file
        default_context_key: Key to use for the default context in lhotse yaml
    """

    def __init__(
        self,
        text_processor: PromptFormatterTextProcessing,
        default_context: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        context_key: str = "context",
        default_context_key: str = "default_context",
    ):
        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.default_context = default_context
        self.context_key = context_key
        self.default_context_key = default_context_key

    def __getitem__(self, all_cuts: CutSet) -> dict[str, Union[torch.Tensor, list[str], dict]]:
        ans = {}

        # convert audio cuts to mini-batch
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        if cuts:
            audio, audio_lens, cuts = self.load_audio(cuts)

            return_batch = {}
            audio_ratio = [1.0] * len(cuts)
            for _, cut in enumerate(cuts):
                if hasattr(cut, self.context_key):
                    cut.context = getattr(cut, self.context_key)
                elif hasattr(cut, self.default_context_key):
                    cut.context = getattr(cut, self.default_context_key)
                else:
                    cut.context = self.default_context

            metadata = []
            for id, cut in enumerate(cuts):
                metadata.append({'audio_filepath': cut.id + '.wav'})

            collated_text_data = collate_text_data(
                cuts=cuts,
                default_context=self.default_context,
                text_processor=self.text_processor,
                tokens_to_generate=self.tokens_to_generate,
                pad_to_max_length=self.pad_to_max_length,
                max_seq_length=self.max_seq_length,
            )
            return_batch.update(
                {
                    "sample_ids": list(cuts.ids),
                    "audio_signal": audio,
                    "audio_signal_length": audio_lens,
                    "audio_ratio": torch.FloatTensor(audio_ratio),
                    "metadata": metadata,
                    **collated_text_data,
                }
            )
            ans.update(return_batch)

        # convert text examples to tensors
        text_examples = all_cuts.filter(lambda c: isinstance(c, (SourceTargetTextExample, NeMoSFTExample)))
        if text_examples:
            pad_id = self.text_processor.pad_id
            text_minibatch = dict(
                text_input_ids=collate_vectors_lhotse([e.input_ids for e in text_examples], padding_value=pad_id),
                text_input_lens=torch.tensor([len(e.input_ids) for e in text_examples], dtype=torch.int64),
                text_answer_ids=collate_vectors_lhotse([e.answer_ids for e in text_examples], padding_value=pad_id),
                text_answer_lens=torch.tensor([len(e.answer_ids) for e in text_examples], dtype=torch.int64),
                text_context_ids=collate_vectors_lhotse([e.context_ids for e in text_examples], padding_value=pad_id),
                text_context_lens=torch.tensor([len(e.context_ids) for e in text_examples], dtype=torch.int64),
                text_masks=collate_vectors_lhotse([e.mask for e in text_examples], padding_value=0),
            )
            ans.update(text_minibatch)

        return ans


def collate_text_data(
    cuts,
    default_context: str,
    text_processor: PromptFormatterTextProcessing,
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
) -> dict:
    """Perform text collation equivalent to nemo/collections/multimodal/data/audio_text_qa_dataset.py:121"""
    batch_size = len(cuts)
    pad_id = text_processor.pad_id
    examples = [{k: torch.as_tensor(v) for k, v in text_processor._process_example(cut).items()} for cut in cuts]
    fields = as_dict(examples)

    def get_max_len(input_list):
        return max([len(x) for x in input_list])

    input_id_maxlen = get_max_len(fields["input_ids"])
    context_id_maxlen = tokens_to_generate + get_max_len(fields["context_ids"])
    answer_id_maxlen = get_max_len(fields["answer_ids"])
    if pad_to_max_length:
        input_id_maxlen = max_seq_length
        context_id_maxlen = max_seq_length
        answer_id_maxlen = max_seq_length

    all_tokens = collate_vectors(fields["input_ids"], max_length=input_id_maxlen, padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in fields["input_ids"]])

    assert input_id_maxlen <= max_seq_length, f"{input_id_maxlen=} <= {max_seq_length=}"

    return {
        "tokens": all_tokens[:, :-1],
        "tokens_length": full_lengths - 1,
        "labels": all_tokens[:, 1:],
        "loss_mask": collate_vectors(
            [torch.as_tensor(build_loss_mask(item)) for item in examples], max_length=input_id_maxlen, padding_value=0
        )[:, 1:],
        "position_ids": torch.arange(input_id_maxlen, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], max_length=context_id_maxlen, padding_value=pad_id),
        "context_lengths": torch.LongTensor([len(seq) for seq in fields["context_ids"]]),
        "answers": collate_vectors(fields["answer_ids"], max_length=answer_id_maxlen, padding_value=pad_id),
        "max_length": torch.LongTensor([input_id_maxlen] * batch_size),
    }


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}
