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
from dataclasses import dataclass
from typing import Callable, Sequence

import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors
from lhotse.utils import ifnone

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.prompts.canary import CanaryPromptFormatter
from nemo.collections.common.tokenizers import CanaryTokenizer, TokenizerSpec
from nemo.collections.common.tokenizers.canary_tokenizer import CANARY_SPECIAL_TOKENIZER


@dataclass
class PromptedAudioToTextMiniBatch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    transcript: torch.Tensor
    transcript_lens: torch.Tensor
    prompt: torch.Tensor
    prompt_lens: torch.Tensor
    prompted_transcript: torch.Tensor
    prompted_transcript_lens: torch.Tensor

    def get_decoder_inputs_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the inputs and outputs of transformer decoder for training.
        The input is ``prompted_transcript`` (minus last token),
        and the output is ``prompted_transcript`` (minus first token).
        """
        return self.prompted_transcript[:, :-1], self.prompted_transcript[:, 1:]


class PromptedAudioToTextLhotseDataset(torch.utils.data.Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``CutSet`` which it will internally iterate over for utterances, and
    * a ``TokenizerWrapper`` object that will be internally used to tokenize the utterances

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt_format_fn: Callable[
            [CutSet, TokenizerWrapper], tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        ],
    ):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer._tokenizer.pad_id
        self.prompt_format_fn = prompt_format_fn

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
        audio, audio_lens, cuts = self.load_audio(cuts)

        prompts_with_answers, prompts, answers = self.prompt_format_fn(cuts, self.tokenizer)

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(prompts_with_answers)
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
        )

    def _collate_tokens(self, tokens: list[list[int] | torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens


# Mapping from a string name to a known prompt formatter function.
PROMPT_FORMAT_FNS = {}


def registered_prompt_format_fn(prompt_fn: Callable[[CutSet, TokenizerWrapper], Sequence[Sequence[int]]]):
    """
    Decorator for registering prompt functions under a name.

    Example::

        >>> @registered_prompt_format_fn
        ... def my_prompt(cuts, tokenizer):
        ...     pass
        ...
        ... prompt_fn = get_prompt_format_fn("my_prompt")
    """
    global PROMPT_FORMAT_FNS

    PROMPT_FORMAT_FNS[prompt_fn.__name__] = prompt_fn
    return prompt_fn


def get_prompt_format_fn(name: str) -> Callable[[CutSet, TokenizerWrapper], Sequence[Sequence[int]]]:
    if name not in PROMPT_FORMAT_FNS:
        raise ValueError(
            f"Unknown prompt format function name: {name} " f"(must be one of: {list(PROMPT_FORMAT_FNS.keys())}"
        )
    return PROMPT_FORMAT_FNS[name]


@registered_prompt_format_fn
def canary(
    cuts: CutSet, tokenizer: TokenizerWrapper
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Prepend and append control tokens to the token sequence as per Canary format.

    We use the following special tokens:
    * <|startoftranscript|>
    * <|transcribe|>
    * <|translate|>
    * <|nopnc|>
    * <|pnc|>
    * <|endoftext|>
    * <|LANG|> - for each supported language.
    * <|nospeech|>

    The prompt format syntax is as follows:

        <|startoftranscript|> [ <|nospeech|> | <|LANG|> [ <|transcribe|> | <|translate|> ] <|LANG|> [ <|pnc|> | <|nopnc|> ] TEXT <|endoftext|> ]

    Where expression ``[ a | b ]`` denotes expression ``a`` or expression ``b``, and can be nested.
    Note that ``<|LANG|>`` appears twice: the first occurrence is for the "source" language
    (i.e., spoken language in the recording) and the second occurrence is for the "target" language
    (i.e., the language in which we are going to output the text).
    """

    assert isinstance(
        tokenizer._tokenizer, CanaryTokenizer
    ), "To use 'canary' prompt format, you must use the CanaryTokenizer."
    formatter = CanaryPromptFormatter(tokenizer._tokenizer)

    prompts_with_answers, prompts, answers = [], [], []
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut._first_non_padding_cut
        if not isinstance(cut, MonoCut):
            raise TypeError(
                f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
            )

        # first, validate the utterance
        expected_slots = set(formatter.get_slots("user"))
        missing_keys = expected_slots - set(cut.custom)
        if "task" in missing_keys and "taskname" in cut.custom:
            # Compatibility with "old" Canary manifest format.
            # For compatbility with inference options, this slot is now called "task".
            cut.custom["task"] = cut.custom["taskname"]
            missing_keys.remove("task")
        if missing_keys:
            raise RuntimeError(
                f"We found cut with ID {cut.id} that is missing the following keys: {missing_keys}"
                f"Please ensure that every utterance in the input manifests contains these keys."
            )

        turns = [
            dict(
                role="user",
                slots={
                    **{slot: cut.custom[slot] for slot in expected_slots},
                    formatter.PROMPT_LANGUAGE_SLOT: CANARY_SPECIAL_TOKENIZER,
                },
            )
        ]
        # If data has no transcript, create empty response with <eos> only.
        text = ' '.join(s.text for s in cut.supervisions if s.text is not None)
        turns.append(
            dict(
                role="assistant",
                slots={
                    "text": text,
                    formatter.PROMPT_LANGUAGE_SLOT: ifnone(
                        cut.supervisions[0].language, cut.custom.get("target_lang")
                    ),
                },
            ),
        )
        encoded = formatter.encode_dialog(turns)
        prompts_with_answers.append(encoded["input_ids"])
        prompts.append(encoded["context_ids"])
        if "answer_ids" in encoded:
            assert (
                encoded["answer_ids"][-1].item() == formatter.tokenizer.eos
            ), f"Expected the last token in answer_ids to be EOS, but we got {encoded['answer_ids']=}"
            answers.append(encoded["answer_ids"][:-1])  # Strip Canary's EOS
        else:
            answers.append([])

    return prompts_with_answers, prompts, answers


class ProbablyIncorrectLanguageKeyError(RuntimeError):
    pass
