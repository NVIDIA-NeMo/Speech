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

from typing import Any, List, Tuple, Union

import torch.utils.data
from lhotse.cut import Cut, CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversation, TextTurn
from nemo.collections.speechlm.data.dataset.data_utils import build_loss_mask
from nemo.collections.speechlm.data.text_processing import MultimodalConversationTextProcessor, TextProcessorOutput


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


class MultimodalConversationDataset(torch.utils.data.Dataset):
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
        text_processor: MultimodalConversationTextProcessor,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        default_context: str = "listen to the audio",
        context_key: str = "context",
        default_context_key: str = "default_context",
        answer_key: str = "answer",
        answer_only_loss: bool = True,
        is_train: bool = False,
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
        self.answer_key = answer_key
        self.answer_only_loss = answer_only_loss
        self.is_train = is_train

    def __getitem__(self, all_cuts: CutSet) -> dict[str, Union[torch.Tensor, list[str], dict]]:
        audio_samples = []
        text_samples = []
        for sample in all_cuts:
            audio_data, text_data = self._process_sample(sample)
            audio_samples.append(audio_data)
            text_samples.append(text_data)

        audio_batch = collate_audio_data(audio_samples, pad_val=0.0)

        text_batch = collate_text_data(
            text_samples,
            self.tokens_to_generate,
            self.pad_to_max_length,
            self.max_seq_length,
            pad_id=self.text_processor.pad_id,
            answer_only_loss=self.answer_only_loss,
        )

        sample_ids = list(all_cuts.ids)
        metadata = self._get_metadata(all_cuts)
        batch = {
            "sample_ids": sample_ids,
            "metadata": metadata,
            "audio_signal": audio_batch["audio_signal"],
            "audio_signal_length": audio_batch["audio_length"],
            "tokens": text_batch["tokens"],
            "tokens_length": text_batch["tokens_length"],
            "labels": text_batch["labels"],
            "loss_mask": text_batch["loss_mask"],
            "position_ids": text_batch["position_ids"],
            "contexts": text_batch["contexts"],
            "context_lengths": text_batch["context_lengths"],
            "answers": text_batch["answers"],
            "max_length": text_batch["max_length"],
            "context_start_idx": text_batch["context_start_idx"],  # used for multi-audio per sample
            "num_audios": text_batch["num_audios"],  # used for multi-audio per sample
        }

        if self.is_train:
            # drop the context and answer that are not used in training
            batch.pop("contexts")
            batch.pop("context_lengths")
            batch.pop("answers")

        return batch

    def _get_metadata(self, all_cuts: CutSet) -> List[dict]:
        metadata = []
        for cut in all_cuts:
            metadata.append({"type": type(cut).__name__, "id": getattr(cut, "id", "n/a"), "cut": str(cut)})
        return metadata

    def _process_sample(self, sample: Any) -> dict:
        if isinstance(sample, Cut):
            return self._process_cut(sample)
        elif isinstance(sample, NeMoMultimodalConversation):
            return self._process_multimodal_conversation(sample)
        else:
            raise ValueError(f"Unsupported input type: {type(sample)}")

    def _convert_cut_sample(self, cut: Cut) -> NeMoMultimodalConversation:
        """
        Transform Cut input to as single turn MultiModalConversation for backward compatability.
        """
        if hasattr(cut, self.context_key):
            context = getattr(cut, self.context_key)
        elif hasattr(cut, self.default_context_key):
            context = getattr(cut, self.default_context_key)
        else:
            context = self.default_context

        if hasattr(cut, self.answer_key):
            answer = getattr(cut, self.answer_key)
        else:
            answer = cut.supervisions[0].text

        if context is None:
            raise ValueError(
                f"Context is None for cut {cut}, please double check the `context_key` and the actual content of that field in your data"
            )
        if answer is None:
            raise ValueError(
                f"Answer is None for cut {cut}, please double check the `answer_key` and the actual content of that field in your data"
            )

        sample = NeMoMultimodalConversation(
            id=cut.id,
            turns=[
                TextTurn(
                    role="user",
                    value=context,
                ),
                AudioTurn(
                    role="user",
                    audio_locator_tag="[audio]",
                    cut=cut,
                ),
                TextTurn(
                    role="assistant",
                    value=answer,
                ),
            ],
        )
        return sample

    def _process_cut(self, cut: Cut) -> Tuple[dict, TextProcessorOutput]:
        audio_signal = cut.load_audio().reshape(-1)
        audio_data = {
            "audio_signal": [audio_signal],
            "audio_length": [len(audio_signal)],
        }

        # Process text
        sample = self._convert_cut_sample(cut)
        text_data = self.text_processor(sample)

        return audio_data, text_data

    def _process_multimodal_conversation(self, sample: NeMoMultimodalConversation) -> Tuple[dict, TextProcessorOutput]:
        # Load audio
        audio_turns = [turn for turn in sample.turns if isinstance(turn, AudioTurn)]
        audio_signal = [turn.cut.load_audio().reshape(-1) for turn in audio_turns]
        audio_length = [len(audio) for audio in audio_signal]
        audio_data = {
            "audio_signal": audio_signal,
            "audio_length": audio_length,
        }

        # Process text
        text_data = self.text_processor(sample)

        return audio_data, text_data


def collate_audio_data(samples: List[dict], pad_val: int = 0) -> dict:
    """
    Collate audio data for all samples in the batch
    """
    audio_signal = []
    audio_length = []
    for sample in samples:
        audio_signal.extend(sample["audio_signal"])
        audio_length.extend(sample["audio_length"])

    if len(audio_signal) == 0:
        audio_signal = torch.tensor([0.0])
        audio_length = [0]

    max_audio_length = max(audio_length)
    audio_signal_tensor = collate_vectors(audio_signal, max_audio_length, pad_val)
    audio_length_tensor = torch.tensor(audio_length).long()
    return {
        "audio_signal": audio_signal_tensor,  # [batch_size, max_audio_length]
        "audio_length": audio_length_tensor,  # [batch_size]
    }


def collate_text_data(
    samples: List[TextProcessorOutput],
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
    pad_id: int,
    answer_only_loss: bool = True,
) -> dict:
    """
    Perform text collation on results from text processor
    """
    batch_size = len(samples)

    def get_max_len(input_list):
        return max([len(x) for x in input_list])

    input_ids = [sample.input_ids for sample in samples]
    context_ids = [sample.context_ids for sample in samples]
    context_lengths = [sample.context_length for sample in samples]
    answer_ids = [sample.answer_ids for sample in samples]
    context_start_idx = [sample.context_start_idx.numpy().tolist() for sample in samples]
    num_audios = [sample.num_audios for sample in samples]
    answer_start_idx = [sample.answer_start_idx for sample in samples]

    input_id_maxlen = get_max_len(input_ids)
    context_id_maxlen = tokens_to_generate + get_max_len(context_ids)
    answer_id_maxlen = get_max_len(answer_ids)
    if pad_to_max_length:
        input_id_maxlen = max_seq_length
        context_id_maxlen = max_seq_length
        answer_id_maxlen = max_seq_length

    all_tokens = collate_vectors(input_ids, max_length=input_id_maxlen, padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in input_ids])

    assert input_id_maxlen <= max_seq_length, f"{input_id_maxlen=} <= {max_seq_length=}"

    loss_mask = [
        torch.as_tensor(build_loss_mask(inp_id, ans_idx, answer_only_loss))
        for inp_id, ans_idx in zip(input_ids, answer_start_idx)
    ]
    return {
        "tokens": all_tokens[:, :-1],  # [batch_size, input_id_maxlen - 1]
        "tokens_length": full_lengths - 1,  # [batch_size]
        "labels": all_tokens[:, 1:],  # [batch_size, input_id_maxlen - 1]
        "loss_mask": collate_vectors(loss_mask, max_length=input_id_maxlen, padding_value=0)[
            :, 1:
        ],  # [batch_size, input_id_maxlen - 1]
        "position_ids": torch.arange(input_id_maxlen, dtype=torch.long).repeat(
            batch_size, 1
        ),  # [batch_size, input_id_maxlen]
        "contexts": collate_vectors(
            context_ids, max_length=context_id_maxlen, padding_value=pad_id
        ),  # [batch_size, context_id_maxlen]
        "context_lengths": torch.LongTensor(context_lengths),  # [batch_size]
        "answers": collate_vectors(
            answer_ids, max_length=answer_id_maxlen, padding_value=pad_id
        ),  # [batch_size, answer_id_maxlen]
        "max_length": torch.LongTensor([input_id_maxlen] * batch_size),  # [batch_size]
        "context_start_idx": context_start_idx,  # List[List[int]],
        "num_audios": torch.stack(num_audios),  # torch.Tensor, shape (batch_size,)
    }
