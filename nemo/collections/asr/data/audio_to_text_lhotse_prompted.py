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
from typing import Callable, Sequence

import omegaconf
import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.tokenizers import CanaryTokenizer, TokenizerSpec


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
        prompt_format_fn: Callable[[CutSet, TokenizerWrapper, bool], Sequence[Sequence[int]]],
        inference: bool = False,
    ):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer._tokenizer.pad_id
        self.prompt_format_fn = prompt_format_fn
        self.inference = inference

    def __getitem__(self, cuts: CutSet) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio, audio_lens, cuts = self.load_audio(cuts)

        tokens, prompt_tokens = self.prompt_format_fn(cuts, self.tokenizer, inference=self.inference)

        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)

        if self.inference:
            prompt_tokens = [torch.as_tensor(t) for t in prompt_tokens]
            prompt_token_lens = torch.tensor([t.size(0) for t in prompt_tokens], dtype=torch.long)
            prompt_tokens = collate_vectors(prompt_tokens, padding_value=self.padding_value)
        else:
            prompt_tokens = None
            prompt_token_lens = None

        return audio, audio_lens, tokens, token_lens, prompt_tokens, prompt_token_lens


# Mapping from a string name to a known prompt formatter function.
PROMPT_FORMAT_FNS = {}


def registered_prompt_format_fn(prompt_fn: Callable[[CutSet, TokenizerWrapper, bool], Sequence[Sequence[int]]]):
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


def get_prompt_format_fn(name: str) -> Callable[[CutSet, TokenizerWrapper, bool], Sequence[Sequence[int]]]:
    if name not in PROMPT_FORMAT_FNS:
        raise ValueError(
            f"Unknown prompt format function name: {name} " f"(must be one of: {list(PROMPT_FORMAT_FNS.keys())}"
        )
    return PROMPT_FORMAT_FNS[name]


@registered_prompt_format_fn
def canary(cuts: CutSet, tokenizer: TokenizerWrapper, inference: bool = False) -> Sequence[Sequence[int]]:
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
    tokenizer = tokenizer._tokenizer

    tokens, prompts = [], []
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut._first_non_padding_cut
        assert isinstance(cut, MonoCut), "Expected MonoCut."

        # first, validate the utterance
        missing_keys = [k for k in ("source_lang", "target_lang", "taskname", "pnc") if k not in cut.custom]
        if missing_keys:
            raise RuntimeError(
                f"We found cut with ID {cut.id} that is missing the following keys: {missing_keys}"
                f"Please ensure that every utterance in the input manifests contains these keys."
            )

        # Actual tokenization. If a cut has multiple supervisions, we'll stitch their tokenized texts together.
        texts = [sup.text for sup in cut.supervisions]
        langs = [sup.language for sup in cut.supervisions]
        taskname = cut.custom['taskname']
        pnc = cut.custom['pnc']
        source_lang = cut.custom['source_lang']
        target_lang = cut.custom['target_lang']

        tokens.append(canary_prompt(tokenizer, texts, langs, source_lang, target_lang, taskname, pnc))
        if inference:
            prompts.append(canary_prompt(tokenizer, None, None, source_lang, target_lang, taskname, pnc))
    return tokens, prompts


def canary_prompt(
    tokenizer: CanaryTokenizer,
    text: str | list[str] | None,
    language: str | list[str] | None,
    source_language: str,
    target_language: str,
    taskname: str,
    pnc: str,
) -> list[int]:
    if isinstance(text, str):
        text = [text]
    if isinstance(language, str):
        language = [language]

    if text is not None:
        try:
            tokens = sum((tokenizer.text_to_ids(text_, lang_) for text_, lang_ in zip(text, language)), start=[])
        except omegaconf.errors.KeyValidationError as e:
            raise ProbablyIncorrectLanguageKeyError(
                "We couldn't select the right tokenizer, which could be due to issues with reading "
                "the language from the manifest. "
                "If you're training, try setting lang_field='' to a different value (probably 'target_lang' or 'lang'). "
                "If you're using model.transcribe() directly, please use override_config kwarg to set this. "
                "If you're using transcribe_speech.py, use option gt_lang_attr_name='...' "
            ) from e
    else:
        tokens = None  # create prompt for inference

    # bos
    prompted_tokens = [tokenizer.bos_id]

    if tokens is not None and len(tokens) == 0:
        # no speech token
        prompted_tokens.append(tokenizer.nospeech_id)
    else:
        # first, validate the utterance
        if source_language is None or target_language is None or taskname is None or pnc is None:
            raise RuntimeError(
                f"Missing keys provided to prompt: "
                f"source_langauge={source_language},\n"
                f"target_language={target_language},\n"
                f"taskname={taskname},\n"
                f"pnc={pnc}\n"
                f"Please ensure that every utterance in the input manifests contains these keys."
            )

        # src_lang_id/no_speech
        src_lang_id = tokenizer.spl_token_to_id(source_language)
        prompted_tokens.append(src_lang_id)

        # task
        task = taskname
        if task == 'asr' or task == "transcribe":
            prompted_tokens.append(tokenizer.spl_token_to_id("transcribe"))
        elif task == 's2t_translation' or task == 'ast' or task == "translate":
            prompted_tokens.append(tokenizer.spl_token_to_id("translate"))
        else:
            raise ValueError(f"Unknown task: {task}")

        # tgt_lang_id
        tgt_lang_id = tokenizer.spl_token_to_id(target_language)
        prompted_tokens.append(tgt_lang_id)

        # PnC
        pnc = f"{pnc}".lower().strip()  # to account for bool or str
        if pnc in {'yes', 'true'}:
            prompted_tokens.append(tokenizer.spl_token_to_id("pnc"))
        elif pnc in {'no', 'false'}:
            prompted_tokens.append(tokenizer.spl_token_to_id("nopnc"))
        else:
            raise ValueError(f"Unknown value for key 'pnc': {pnc}")

        # text (only in training)
        if tokens is not None:
            prompted_tokens.extend(tokens)

    # eos (only in training)
    if tokens is not None:
        prompted_tokens.append(tokenizer.eos_id)
    return prompted_tokens


class ProbablyIncorrectLanguageKeyError(RuntimeError):
    pass
