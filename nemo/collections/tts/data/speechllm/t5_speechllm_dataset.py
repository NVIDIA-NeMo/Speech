# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import enum
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import get_ipa_punctuation_list
from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import any_locale_text_preprocessing
from nemo.collections.nlp.data.language_modeling.megatron.base_prompt_learning_dataset import BasePromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import T5Sentinel
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    BetaBinomialInterpolator,
    beta_binomial_prior_distribution,
    general_padding,
    get_base_dir,
)
from nemo.utils import logging

__all__ = ['T5SpeechLMDataset', "Lang"]


def get_full_list_puncts():
    punct_set = set()
    for locale_id in ["en-US", "de-DE", "fr-FR"]:
        punct_list = get_ipa_punctuation_list(locale=locale_id)
        punct_set.update(punct_list)
    return sorted(punct_set)


@dataclass
class G2PConfig:
    _target_: str = "nemo.collections.tts.g2p.models.en_us_arpabet.EnglishG2p"
    phoneme_dict: str = "scripts/tts_dataset_files/cmudict-0.7b_nv22.10"
    heteronyms: str = "scripts/tts_dataset_files/heteronyms-052722"
    phoneme_probability: float = 0.5


@dataclass
class EnglishIpaG2pConfig:
    _target_: str = "nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p"
    phoneme_dict: str = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
    locale: str = "en-US"
    heteronyms: str = "scripts/tts_dataset_files/heteronyms-052722"
    phoneme_probability: float = 0.5
    grapheme_case: str = "upper"
    use_stresses: bool = True
    use_chars: bool = True
    ignore_ambiguous_words: bool = False


@dataclass
class TextTokenizer:
    _target_: str = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishPhonemesTokenizer"
    punct: bool = True
    stresses: bool = True
    chars: bool = True
    apostrophe: bool = True
    pad_with_space: bool = True
    add_blank_at: bool = True
    g2p: G2PConfig = G2PConfig()


@dataclass
class EnglishIpaTextTokenizer:
    _target_: str = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer"
    locale: str = "en-US"
    punct: bool = True
    # Define non_default_punct_list as a ClassVar to explicitly mark it as a class variable
    non_default_punct_list: ClassVar[List[str]] = get_full_list_puncts()
    apostrophe: bool = True
    pad_with_space: bool = True
    add_blank_at: bool = True
    g2p: EnglishIpaG2pConfig = EnglishIpaG2pConfig()


@dataclass
class TextTokenizerConfig:
    text_tokenizer: TextTokenizer = TextTokenizer()


@dataclass
class EnglishIpaTextTokenizerConfig:
    text_tokenizer: EnglishIpaTextTokenizer = EnglishIpaTextTokenizer()


def _get_default_text_tokenizer_conf(phoneme_probability: float = 0.5, use_ipa: bool = False):
    if use_ipa:
        g2p = EnglishIpaG2pConfig(phoneme_probability=phoneme_probability)
        _text_tokenizer = EnglishIpaTextTokenizer(g2p=g2p)
        text_tokenizer: EnglishIpaTextTokenizerConfig = EnglishIpaTextTokenizerConfig(text_tokenizer=_text_tokenizer)
    else:
        g2p = G2PConfig(phoneme_probability=phoneme_probability)
        _text_tokenizer = TextTokenizer(g2p=g2p)
        text_tokenizer: TextTokenizerConfig = TextTokenizerConfig(text_tokenizer=_text_tokenizer)
    return OmegaConf.create(OmegaConf.to_yaml(text_tokenizer))


def pad_text_to_speech_dims(text_tensor, pad_id, pad_size=7):
    token_len = text_tensor.shape[0]
    empty_padding = torch.ones((pad_size, token_len), dtype=text_tensor.dtype, device=text_tensor.device) * pad_id
    return torch.cat((text_tensor.unsqueeze(0), empty_padding), dim=0)


class Lang(enum.Enum):
    en = 1
    es = 2
    fr = 3
    zh = 4
    de = 4


class T5SpeechLMDataset(BasePromptLearningDataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained T5 SpeechLM models.
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        sample_rate: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
        decoder_starts_with_pad: bool = False,
        add_eos_to_decoder_output: bool = True,
        add_sentinel_to_input: bool = True,
        ul2_prompt_token: str = None,
        segment_max_duration: Optional[int] = None,
        trim: bool = False,
        trim_ref: Optional[float] = None,
        trim_top_db: Optional[int] = None,
        trim_frame_length: Optional[int] = None,
        trim_hop_length: Optional[int] = None,
        pad_multiple: int = 1,
        pitch_augment: bool = False,
        sup_data_path: Optional[Union[Path, str]] = None,
        speech_offset: Optional[int] = None,
        train_task: Optional[str] = None,
        seq_pattern: Optional[str] = "parallel",
        use_attention_prior: Optional[bool] = False,
        attention_prior_scaling_factor: Optional[float] = 1.0,
        spec_aug=False,
        spec_aug_time_width=0.2,
        spec_aug_time_masks=2,
        cross_attention_epsilon: Optional[float] = 0.0,
        lm_vocab_size: Optional[int] = None,
        num_speech_codebooks: Optional[int] = 8,
        codebook_fps: Optional[int] = 86,
        add_special_tokens_to_only_first_codebook: Optional[bool] = False,
        context_pattern: Optional[str] = "parallel",
        context_duration_min: Optional[float] = 3.0,
        context_duration_max: Optional[float] = 5.0,
        skip_datasets: Optional[List[str]] = [],  # substrings of dataset names to skip
        english_only_model: Optional[bool] = False,
        context_conditioning: Optional[str] = "decoder",  # encoder or decoder
        use_beta_binomial_interpolator: Optional[str] = False,  # encoder or decoder
        context_slice_method: Optional[str] = "random",  # random or fixed
        phoneme_probability: Optional[float] = 0.5,
        encoder_type: Optional[str] = "single_transformer",
        use_ipa: bool = False,
        **kwargs,
    ):
        """
        Only speech parameters are explained here.
        segment_max_duration: Optional[int] = None, - Speech max segment duration
        trim: bool = False, - speech parameter
        trim_ref: Optional[float] = None, - speech parameter
        trim_top_db: Optional[int] = None, - speech parameter
        trim_frame_length: Optional[int] = None, - speech parameter
        trim_hop_length: Optional[int] = None, - speech parameter
        pad_multiple: int = 1, - speech parameter
        pitch_augment: bool = False, - speech parameter
        sup_data_path: Optional[Union[Path, str]] = None, - Supplementary folder path where codecs are stored.
        speech_offset: Optional[int] = None, - if speech tokens then add this offset to the token indices to distinguish between text and speech tokens.
        lm_vocab_size: Optional[int] = None, - vocab size of the original language model (phoneme tokens start from this index)
        english_only_model: Optional[bool] = False, specify if monolingual or multi-lingual modeling.
        use_ipa: bool = False, specify if using IPA tokens or default ARPABET tokens. Either choice still mixes chars.
        **kwargs,
        """
        # These two variables need to be set before calling super().__init__() because the parent class calls `load_data()` which requires these attributes.
        self._rng = random.Random()
        self.spec_aug = spec_aug if for_train else False
        self.time_width = spec_aug_time_width
        self.time_masks = spec_aug_time_masks
        self.decoder_starts_with_pad = decoder_starts_with_pad
        self.add_eos_to_decoder_output = add_eos_to_decoder_output
        self.add_sentinel_to_input = add_sentinel_to_input
        self.ul2_prompt_token = ul2_prompt_token
        # Speech related variables
        self.base_data_dir = None
        self.segment_max_duration = segment_max_duration
        self.sample_rate = sample_rate
        self.featurizer = WaveformFeaturizer(sample_rate=self.sample_rate)
        self.pad_multiple = pad_multiple
        self.pitch_augment = pitch_augment
        self.trim = trim
        self.trim_ref = trim_ref if trim_ref is not None else np.max
        self.trim_top_db = trim_top_db if trim_top_db is not None else 60
        self.trim_frame_length = trim_frame_length if trim_frame_length is not None else 2048
        self.trim_hop_length = trim_hop_length if trim_hop_length is not None else 512
        self.speech_offset = speech_offset if speech_offset is not None else 3
        self.seq_pattern = seq_pattern
        self.use_attention_prior = use_attention_prior
        self.attention_prior_scaling_factor = attention_prior_scaling_factor
        self.cross_attention_epsilon = cross_attention_epsilon  # value of prior for context tokens (b/w 0 and 1)
        assert self.cross_attention_epsilon >= 0.0 and self.cross_attention_epsilon <= 1.0
        self.lm_vocab_size = tokenizer.vocab_size if lm_vocab_size is None else lm_vocab_size
        self.num_speech_codebooks = num_speech_codebooks
        self.codebook_fps = codebook_fps
        self.add_special_tokens_to_only_first_codebook = add_special_tokens_to_only_first_codebook
        # context_pattern and duration arguments are supported only if context_type is REFSPEAKERCODEC in the manifest
        self.context_pattern = context_pattern
        self.context_duration_min = context_duration_min
        self.context_duration_max = context_duration_max
        self.english_only_model = english_only_model
        self.phoneme_tokenizer = None
        if english_only_model:
            self.phoneme_tokenizer = instantiate(
                _get_default_text_tokenizer_conf(phoneme_probability=phoneme_probability, use_ipa=use_ipa)
            ).text_tokenizer
        else:
            self.g2p = {"fr": lambda x: x}
            if kwargs.get("g2p", None):
                if "english" in kwargs["g2p"]:
                    english_g2p = instantiate(kwargs["g2p"]["english"])
                    self.g2p["en"] = lambda x: english_g2p(x)
                if "spanish" in kwargs["g2p"]:
                    spanish_g2p = instantiate(kwargs["g2p"]["spanish"])
                    self.g2p["es"] = lambda x: spanish_g2p(x)
                if "mandarin" in kwargs["g2p"]:
                    mandarin_g2p = instantiate(kwargs["g2p"]["mandarin"])
                    self.g2p["zh"] = lambda x: mandarin_g2p(x)
                if "german" in kwargs["g2p"]:
                    german_g2p = instantiate(kwargs["g2p"]["german"])
                    self.g2p["de"] = lambda x: german_g2p(x)

        self.context_conditioning = context_conditioning
        if self.context_conditioning == "decoder":
            assert (
                self.context_duration_min == self.context_duration_max
            ), "For decoder conditioning, context_duration_min and context_duration_max should be same"
            self.decoder_context_len = int(
                self.context_duration_min * self.codebook_fps
            )  # TODO: Just take from model var?

        # Initialize sup_data_path, sup_data_types and run preprocessing methods for every supplementary data type\
        self.sup_data_path = None
        if sup_data_path is not None:
            Path(sup_data_path).mkdir(parents=True, exist_ok=True)
            self.sup_data_path = sup_data_path

        self.codec_folder = kwargs.pop('codec_folder', None)
        self.train_task = train_task
        if self.codec_folder is None and self.sup_data_path is not None:
            self.codec_folder = Path(self.sup_data_path) / "codec"
        elif isinstance(self.codec_folder, str):
            self.codec_folder = Path(self.codec_folder)

        self.codec_folder.mkdir(exist_ok=True, parents=True)

        self.context_length = kwargs.pop('context_length', None)  # only used in gpt dataset atm
        # self.attention_prior_strength = attention_prior_strength
        self.transformer_type = kwargs.pop('transformer_type', 'T5')
        self.skip_datasets = skip_datasets

        self.beta_binomial_interpolator = (
            BetaBinomialInterpolator(scaling_factor=self.attention_prior_scaling_factor)
            if use_beta_binomial_interpolator
            else None
        )
        self.context_slice_method = context_slice_method
        self.encoder_type = encoder_type
        super().__init__(
            datasets=datasets,
            tokenizer=tokenizer,
            virtual_prompt_source=virtual_prompt_source,
            task_templates=task_templates,
            pseudo_tokens=pseudo_tokens,
            pad_token_id=pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
        )

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in
        the task templates with the actual virtual prompt token ids.

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        copy_dataset = list(dataset)
        audio_filelist = []
        # This loop is needed to calculate self.base_data_dir.
        for json_line in copy_dataset:
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)
            taskname = doc["taskname"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]

            for p in prompt_template_fields:
                if f"{p}_type" in doc and doc[f"{p}_type"] == "SPEECH":
                    audio_filelist.append(doc[p])
        self.base_data_dir = get_base_dir(audio_filelist)

        skipped = 0
        tts = 0
        asr = 0
        i = 0
        logging.info(f"copy_dataset len === {len(copy_dataset)}")
        examples = []
        for json_line in tqdm(copy_dataset):
            i += 1

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            if self.context_conditioning == "decoder":
                # Modify doc to make combine context and anwer
                assert ";" not in doc['context'], "Multiple contexts not supported in decoder conditioning"
                doc['answer'] = "{};{}".format(doc['context'], doc['answer'])
                doc['answer_duration'] = self.context_duration_min + doc['answer_duration']
                doc['answer_type'] = "CONTEXTANSWER"
                doc['context_type'] = "DUMMYCONTEXT"
                doc['context'] = "DUMMYCONTEXT"

            question_in_manifest = doc['question']

            if "Text to speech this" in question_in_manifest or "Phoneme TTS" in question_in_manifest:
                tts += 1
                if self.train_task not in ['tts', 'all']:
                    continue
            elif "Next token prediction" in question_in_manifest:
                if self.train_task != 'tts':
                    asr += 1
                else:
                    tts += 1
                continue
            else:
                if self.train_task == 'tts':
                    continue
                asr += 1

            if doc["context_type"] == "SPEECH":
                assert "context_duration" in doc, f"context_duration key not in document {doc}"
                approx_context_len = 3 * (self.codebook_fps + 1)  # +1 just to be safe
                if self.context_length is not None and doc["context_duration"] < self.context_length:
                    logging.debug(
                        f"skipped as context_length of {doc['context_duration']} is less than {self.context_length}"
                    )
                    skipped += 1
                    continue
            elif "Remove Noise" in question_in_manifest:
                approx_context_len = doc["answer_duration"] * (self.codebook_fps + 1)
            elif "Extract Speaker Audio" in question_in_manifest:
                approx_context_len = (
                    doc["answer_duration"] * (self.codebook_fps + 1) + 400
                )  # 400 is the max ref speaker audio
            elif ("Text to speech this" in question_in_manifest) or ('Phoneme TTS' in question_in_manifest):
                # approx_context_len = 400
                approx_context_len = 5 * (
                    self.codebook_fps + 1
                )  # better than 400. TODO: pneekhara: Need to change things for multi-encoder vs single encoder based filtering.
            elif "Edit Speech" in question_in_manifest:
                approx_context_len = doc["answer_duration"] * (self.codebook_fps + 1)
            else:
                raise NotImplementedError(f"Unknown context type {doc['context_type']}")

            approx_question_len = len(doc["question"].split(' ')) + 3
            if 'Phoneme TTS' in question_in_manifest:
                # approx len is equal to num of characters
                approx_question_len = len(question_in_manifest)

            if doc["answer_type"] in ["SPEECH", "AUDIOCODEC", "CONTEXTANSWER"]:
                assert "answer_duration" in doc, f"answer_duration key not in document {doc}"
                approx_answer_len = doc["answer_duration"] * (self.codebook_fps + 1) + 3  # +3 for EOS, BOS padding
                if self.seq_pattern == "delay_parallel":
                    # In delay parallel, there is padding so add 8 frames
                    approx_answer_len = approx_answer_len + self.num_speech_codebooks
            else:
                approx_answer_len = len(doc["answer"].split(' ')) + 3

            skip_record = False
            for skip_dataset in self.skip_datasets:
                if skip_dataset in doc['answer']:
                    skip_record = True

            if not skip_record:
                if (self.transformer_type == "GPT") and (
                    self.min_seq_length
                    < approx_context_len + approx_question_len + approx_answer_len
                    < self.max_seq_length
                ):
                    examples.append(doc)
                elif (self.transformer_type == "T5") and (
                    self.min_seq_length < approx_context_len + approx_question_len < self.max_seq_length
                    and self.min_seq_length < approx_answer_len < self.max_seq_length
                ):
                    examples.append(doc)
                else:
                    logging.debug(f"skipped for {approx_context_len + approx_question_len} {approx_answer_len} len")
                    skipped += 1
            else:
                print("Skipping", doc['answer'])
                logging.debug(f"skipped for {doc['answer']} as it is in skip_datasets")
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

        return examples

    def __getitem__(self, idx):
        doc = self.examples[idx]
        taskname = doc["taskname"]
        prompt_template = self.task_templates[taskname]["prompt_template"]
        prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
        total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
        virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
        truncation_field = self.task_templates[taskname]['truncate_field']
        answer_field = self.task_templates[taskname]["answer_field"]

        input_example = prompt_template

        self._input_sanity_checks(
            total_virtual_tokens=total_virtual_tokens,
            virtual_token_splits=virtual_token_splits,
            prompt_template=prompt_template,
            prompt_template_fields=doc.keys(),  # Skip this check as we don't need it for TTS
            truncation_field=truncation_field,
            answer_field=answer_field,
            doc=doc,
        )
        question_in_manifest = doc['question']

        # Format the input example according to the template
        # Get context, question and answer codes in a dict.
        # TODO @xueyang: declare the instructions when initializing the dataset so that they can be re-used. Temporally
        #  hardcode them here.
        question_text = doc["question"].strip()
        instructions = ["Phoneme TTS", "Text to speech this"]
        for prefix in instructions:
            if doc["question"].startswith(prefix):
                question_text = doc["question"][len(prefix) :].strip()
                break

        input_dict = self._insert_data_in_template(prompt_template_fields, doc, answer_field)
        lang = Lang[doc.get("lang", "en")]
        context_tokens = input_dict['context']
        question_tokens = input_dict['question']

        # Logic to prune context
        # In case of TTS task, the entire reference speech is not required, so we randomly select a portion
        # of the reference audio.
        # In case of Next token prediction, We want context[:T] to go in the encoder and context[T+1:] to be
        # predicted by the decoder.
        start_token_index = 0
        end_token_index = -1
        if ("Text to speech this" in question_in_manifest) and (doc["context_type"] == "SPEECH"):
            total_context_len = context_tokens[0].size()[1]
            reduced_len = min(
                400,
                (
                    int(total_context_len * 0.2)
                    if total_context_len > 600
                    else int(total_context_len * random.uniform(0.2, 0.5))
                ),
            )
            start_token_index = random.randint(
                0, total_context_len - reduced_len
            )  # start index can be greater than 440
            context_tokens[0] = context_tokens[0][
                :, start_token_index : min(start_token_index + 440, start_token_index + reduced_len)
            ]
        elif "Next token prediction" in question_in_manifest:
            total_context_len = context_tokens[0].size()[1]
            end_token_index = int(total_context_len * random.uniform(0.01, 0.2))
            context_tokens[0] = context_tokens[0][:, :end_token_index]

        # Get virtual tokens
        # `virtual_tokens` is "<prompt_0><prompt_1><prompt_2>".
        virtual_tokens = self._insert_virtual_token_placeholders(input_example.split(' ')[0], virtual_token_splits)

        # a trick to align with the data format in t5 pretraining
        virtual_tokens = self.tokenizer.text_to_ids(virtual_tokens)
        if self.add_sentinel_to_input:
            question_tokens = question_tokens + self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)

        # Add BOS/EOS to the input of encoder if desired, adds EOS by default
        if self.ul2_prompt_token is not None:
            ul2_prompt_token_id = self.tokenizer.text_to_ids(self.ul2_prompt_token)
            assert len(ul2_prompt_token_id) == 1
            context_tokens = ul2_prompt_token_id + context_tokens
        if self.add_bos:
            context_tokens = [self.tokenizer.bos_id] + context_tokens
        if self.add_eos:
            question_tokens = question_tokens + [self.tokenizer.eos_id]

        # Try to truncate input text to fit into the max sequence length
        if self._get_len(context_tokens, question_tokens, virtual_tokens) > self.max_seq_length:
            context_tokens, question_tokens, virtual_tokens = self._truncate_input_speech(
                context_tokens, question_tokens, virtual_tokens
            )

        virtual_tokens, virtual_tokens_len = self.list_to_tensor(virtual_tokens)
        context_tokens, context_tokens_len = self.list_to_tensor(context_tokens)
        question_tokens, question_tokens_len = self.list_to_tensor(question_tokens)

        if doc["question_type"] == "TEXT" and doc["context_type"] != "TEXT":
            question_tokens = pad_text_to_speech_dims(
                question_tokens, self.tokenizer.pad_id, self.num_speech_codebooks - 1
            )
        if doc["context_type"] == "TEXT" and doc["question_type"] != "TEXT":
            context_tokens = pad_text_to_speech_dims(
                context_tokens, self.tokenizer.pad_id, self.num_speech_codebooks - 1
            )
        if doc["context_type"] == "TEXT" and doc["question_type"] == "TEXT":
            context_tokens = pad_text_to_speech_dims(
                context_tokens, self.tokenizer.pad_id, self.num_speech_codebooks - 1
            )
            question_tokens = pad_text_to_speech_dims(
                question_tokens, self.tokenizer.pad_id, self.num_speech_codebooks - 1
            )

        # context_tokens: tensor, (num_speech_codebooks, audio_context_len)
        # question_tokens: tensor, (num_speech_codebooks, instruction token len + question token len + 1 (<extra_id_0> + 1 ([SEP])), only first row includes token ids while all other rows are all zeros (pad)
        if self.encoder_type == "multi_transformer":
            context_and_question_tokens = [context_tokens, question_tokens]
        else:
            context_and_question_tokens = torch.cat([context_tokens, question_tokens], dim=1)

        # get answer ids
        if answer_field in doc.keys():  # training and validation
            answer_ids = self._get_tokens(doc, answer_field, doc[answer_field])
            if end_token_index > -1:
                answer_ids[0] = answer_ids[0][:, end_token_index:]

            if self.decoder_starts_with_pad:
                answer_text_ids = [self.tokenizer.pad_id]
            else:
                answer_text_ids = [self.tokenizer.bos_id]
            # a trick to align with the data format in t5 pretraining
            # if self.add_sentinel_to_input:
            #     answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            answer_text_ids += answer_ids

            if self.add_eos_to_decoder_output:
                answer_text_ids += [self.tokenizer.eos_id]
            else:
                answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.END.value)

        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            taskname_id = self.tokenizer.text_to_ids(taskname)
        elif (
            self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT
        ):  # TODO (@adithyare) this class and GPTPromptLearningDataset should be merged.
            taskname_id = -1
        else:
            raise ValueError("Invalid virtual prompt source specified")

        dec_input = None
        dec_labels = None

        # if single-encoder and context_condition is decoder, answer_text_ids = [CLS_id, context audio code tensors, zero-pad, answer audio code tensor, SEP_id]
        # if multi-encoder, answer_text_ids = [CLS_id, answer audio codec tensor, SEP_id], so dec_input will not include audio context anymore.
        if answer_field in doc.keys():  # training and validation
            dec_input = answer_text_ids[:-1]
            dec_labels = answer_text_ids[1:]

        # if single-encoder and context_condition is decoder:
        #   dec_input: shape=(self.num_speech_codebooks, 1([CLS]) + len(context audio frames) + 1([PAD]) + len(answer audio frames))
        #   dec_labels: shape=(self.num_speech_codebooks, len(context audio frames) + 1([PAD]) + len(answer audio frames) + 1([SEP]))
        # if multi-encoder:
        #   dec_input: (num_speech_codebooks, 1([CLS]) + len(answer audio frames))
        #   dec_labels: (num_speech_codebooks, len(answer audio frames) + 1([SEP]))
        dec_input, dec_input_len = self.list_to_tensor(dec_input, True)
        dec_labels, dec_labels_len = self.list_to_tensor(dec_labels, True)
        is_speech = True if doc["answer_type"] != "TEXT" else False
        if is_speech:
            assert dec_input.dim() == 2 and dec_labels.dim() == 2
            if self.seq_pattern == "delay_parallel":
                num_codebooks = dec_input.shape[0]
                dec_input_padded = torch.cat(
                    [
                        torch.zeros_like(dec_input[:, 0:num_codebooks]),
                        dec_input,
                        torch.zeros_like(dec_input[:, 0:num_codebooks]),
                    ],
                    dim=1,
                )
                dec_labels_padded = torch.cat(
                    [
                        torch.zeros_like(dec_labels[:, 0:num_codebooks]),
                        dec_labels,
                        torch.zeros_like(dec_labels[:, 0:num_codebooks]),
                    ],
                    dim=1,
                )
                dec_input_new = []
                dec_labels_new = []
                for _c in range(self.num_speech_codebooks):
                    st = num_codebooks - _c
                    et_decoder_input = dec_input_padded.shape[1] - _c
                    et_decoder_labels = dec_labels_padded.shape[1] - _c
                    dec_input_new.append(dec_input_padded[_c, st:et_decoder_input])
                    dec_labels_new.append(dec_labels_padded[_c, st:et_decoder_labels])
                dec_input = torch.stack(dec_input_new, dim=0)
                dec_labels = torch.stack(dec_labels_new, dim=0)
                dec_input_len = torch.tensor(dec_input.shape[1]).long()
                dec_labels_len = torch.tensor(dec_labels.shape[1]).long()

        if self.encoder_type == "multi_transformer":
            enc_len = question_tokens_len + virtual_tokens_len
        else:
            enc_len = context_tokens_len + question_tokens_len + virtual_tokens_len
        # TODO: Remove hardcoding
        start_of_question_offset = 4  # For both "Text to Speech this" and "Phoneme TTS"
        end_of_question_offset = 2
        cross_attention_prior = torch.zeros(dec_labels_len, enc_len) + self.cross_attention_epsilon
        if self.use_attention_prior:
            prior_dec_len = dec_labels_len.item()
            prior_dec_start_idx = 0
            if self.context_conditioning == "decoder":
                prior_dec_len = dec_labels_len.item() - (self.decoder_context_len + 1)
                prior_dec_start_idx = self.decoder_context_len + 1
            text_len = question_tokens_len.item() - start_of_question_offset - end_of_question_offset
            audio_len = prior_dec_len
            if self.beta_binomial_interpolator is not None:
                cross_attention_question_prior = torch.from_numpy(self.beta_binomial_interpolator(audio_len, text_len))
            else:
                cross_attention_question_prior = torch.from_numpy(
                    beta_binomial_prior_distribution(
                        text_len,
                        audio_len,
                        scaling_factor=self.attention_prior_scaling_factor,
                    )
                )
            if self.encoder_type == "multi_transformer":
                cross_attention_prior[
                    prior_dec_start_idx:, virtual_tokens_len + start_of_question_offset : -end_of_question_offset
                ] = cross_attention_question_prior
            else:
                cross_attention_prior[
                    prior_dec_start_idx:,
                    virtual_tokens_len + context_tokens_len + start_of_question_offset : -end_of_question_offset,
                ] = cross_attention_question_prior

        if self.encoder_type == "multi_transformer":
            context_and_question_len = [context_tokens_len, question_tokens_len]
        else:
            context_and_question_len = context_tokens_len + question_tokens_len
        return (
            taskname_id,  # List, only one item. token id for "squad"
            virtual_tokens,  # Tensor, shape=(3,). token id for ['<prompt_0>', '<prompt_1>', '<prompt_2>']
            virtual_tokens_len,  # tensor, 3
            context_tokens_len,  # tensor, 1
            # tensor if single encoder and context_condition is encoder, shape=(self.num_speech_codebooks, 1(context) + question len + 1(<extra_id_0>) + 1([SEP])). only first row includes token ids while all other rows are all zeros (pad).
            # list if multi-encoder and context_condition is encoder.
            context_and_question_tokens,
            # tensor scalar if single encoder and context_condition is decoder, 1 + (question len + 1 + 1).
            # list if multi-encoder and context_condition is encoder.
            context_and_question_len,
            dec_input,  # tensor, shape=(self.num_speech_codebooks, 1 CLS + context audio frame len + 1 pad + answer audio frame len), first column is [CLS_id, 0*7]^T
            dec_input_len,  # scalar tensor, 1 CLS + context audio frame len + 1 pad + answer audio frame len. 1 corresponds to CLS id
            dec_labels,  # tensor, shape=(self.num_speech_codebooks, context audio frame len + 1 pad + answer frame len + 1 SEP).
            dec_labels_len,  # tensor, context audio frame len + 1 PAD + answer frame len + 1 SEP.  1 corresponds to SEP id.
            is_speech,  # True
            cross_attention_prior,  # tensor, shape=(dec_labels_len, context_tokens_len + question_tokens_len + virtual_tokens_len).
            lang.value,  # int,
            question_text,  # str, answer transcript without question type (Phoneme TTS or Text to speech this).
        )

    def _truncate_input_speech(self, context_tokens, question_tokens, virtual_tokens):
        total_len = self._get_len(context_tokens, question_tokens, virtual_tokens)
        context_len = self._get_element_len(context_tokens)
        truncation_length = total_len - self.max_seq_length + 1
        context_tokens[0] = context_tokens[0][:, min(truncation_length, context_len) :]
        return context_tokens, question_tokens, virtual_tokens

    def list_to_tensor(self, element, fill=False):
        """
        Convert list to tensor. The list might contain integers, 2D-tensors (speech tokens) and combination of two.
        If all of them are ints, simply convert to tensor
        If combination of 2D-tensor and ints. Convert int to the dimension of the tensor.
        example: [2, 4, 5] -> torch.tensor([2, 4, 5])
        example: [2, torch.tensor([[4, 5, 6], [6, 7, 8]])] -> torch.tensor( [[-1, 4, 5, 6], [2, 6, 7, 8]] )
        """
        ret, ln = None, None
        if element is None:
            return ret, ln

        max_len = max([1 if isinstance(item, int) else len(item) for item in element])
        if max_len == 1:
            ret = torch.as_tensor(element).long()
            ln = torch.tensor(ret.size()[0]).long()
        else:
            ret = []
            for e in element:
                if isinstance(e, int):
                    tmp = torch.full((self.num_speech_codebooks, 1), e if fill else -1)
                    tmp[self.num_speech_codebooks - 1] = e
                    if self.add_special_tokens_to_only_first_codebook:
                        # Fill zeros in all other codebooks (to avoid out of range when getting embeddings)
                        tmp[1:] = 0
                else:
                    tmp = e
                ret.append(tmp)
            ret = torch.cat(ret, dim=1)
            ln = torch.tensor(ret.size()[1]).long()
        return ret, ln

    def _get_text_tokens(self, text):
        input_ids = self.tokenizer.text_to_ids(text)
        return input_ids

    def _get_phoneme_tokens(self, text, lang="en"):
        if self.english_only_model:
            input_ids = self.phoneme_tokenizer.encode(text)
            input_ids_adjusted = [_id + self.lm_vocab_size for _id in input_ids]
            return input_ids_adjusted
        else:
            text = any_locale_text_preprocessing(text)
            input_ids = self.g2p[lang](text)
            input_ids_adjusted = []
            for i in input_ids:
                input_ids_adjusted.append(f"p{{{i}}}")
            input_ids_adjusted = self.tokenizer.text_to_ids("".join(input_ids_adjusted))
            return input_ids_adjusted

    def _pad_wav_to_multiple(self, wav):
        if self.pad_multiple > 1:
            if wav.shape[0] % self.pad_multiple != 0:
                wav = torch.cat(
                    [wav, torch.zeros(self.pad_multiple - wav.shape[0] % self.pad_multiple, dtype=torch.float)]
                )
        return wav

    def _get_element_len(self, element):
        length = 0
        if isinstance(element, list):
            for e in element:
                if isinstance(e, int):
                    length += 1
                else:
                    if e.dim() > 1:
                        length += e.size()[1]
                    else:
                        length += e.size()[0]
        else:
            if element.dim() > 1:
                length += element.size()[1]
            else:
                length += element.size()[0]
        return length

    def _get_len(self, context_tokens, question_tokens, virtual_tokens):
        length = 0
        length += self._get_element_len(context_tokens)
        length += self._get_element_len(question_tokens)
        length += self._get_element_len(virtual_tokens)
        return length

    def _load_audio(self, audio_filepath, dur=-1):
        if self.segment_max_duration is not None and dur > 0 and dur > self.segment_max_duration:
            # this case has been added for segmenting audio for speaker verification task of SSLDisentangler
            n_segments = int(self.segment_max_duration * self.sample_rate)
            features = AudioSegment.segment_from_file(
                audio_filepath, target_sr=self.sample_rate, n_segments=n_segments, trim=self.trim
            )

            features = torch.tensor(features.samples)
            if self.pad_multiple > 1:
                features = self._pad_wav_to_multiple(features)
            audio, audio_length = features, torch.tensor(features.shape[0]).long()
        else:
            features = self.featurizer.process(
                audio_filepath,
                trim=self.trim,
                trim_ref=self.trim_ref,
                trim_top_db=self.trim_top_db,
                trim_frame_length=self.trim_frame_length,
                trim_hop_length=self.trim_hop_length,
            )

            if self.pad_multiple > 1:
                features = self._pad_wav_to_multiple(features)

            audio, audio_length = features, torch.tensor(features.shape[0]).long()

        return audio, audio_length

    def convert_audio(self, audio, sample_rate, target_sample_rate, target_channels):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        assert audio.shape[1] in [1, 2], "Audio must be mono or stereo."
        # assert sample_rate == target_sample_rate, "sample rate of FastPitch and Encodec model has to be same"
        if target_channels == 2:
            *shape, _, length = audio.shape
            audio = audio.expand(*shape, target_channels, length)
        return audio

    def get_codec(self, audio):
        wav1 = self.convert_audio(audio, self.sample_rate, self.encodec_model.sample_rate, self.encodec_model.channels)
        encoded_frames = self.encodec_model.encode(wav1)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        return codes.squeeze(0)

    def get_quantizer_codebook(self, reference_codec, reference_codec_length):
        out = torch.zeros((1, 128, reference_codec_length.item()))
        for i in range(reference_codec.size()[0]):
            out += self.encodec_model.quantizer.vq.layers[i].decode(reference_codec[i, :].unsqueeze(0))
        return out.squeeze(0)

    def _get_speech_tokens(self, audio_filepath, dur=-1):
        # Let's keep audio name and all internal directories in rel_audio_path_as_text_id to avoid any collisions
        rel_audio_path = Path(audio_filepath).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")

        # Load audio features
        audio, audio_length = self._load_audio(audio_filepath, dur)

        # Convert to codes
        codec_path = self.codec_folder / f"{rel_audio_path_as_text_id}.pt"

        if codec_path.exists():
            try:
                codec_codes = torch.load(codec_path).long()
            except Exception as e:
                print(f"[ERROR IN LOADING {codec_path}] e")
                codec_codes = self.get_codec(audio).long()
                torch.save(codec_codes, codec_path)
        else:
            codec_codes = self.get_codec(audio).long()
            torch.save(codec_codes, codec_path)

        # Convert codes to codes corresponding to megatron embedding layer
        codec_codes[0] = (codec_codes[0] + self.speech_offset).long()

        return codec_codes

    def _get_tokens(self, doc, field, field_data):
        if self.context_slice_method == "random":
            # During training, we want a random slice of the context
            rng = random.Random()  # Custom random generator (since random uses fixed seeds)
        elif self.context_slice_method == "fixed":
            # During inference, we want a fixed slice of the context
            rng = random
        else:
            raise ValueError(f"Invalid context_slice_method {self.context_slice_method}")
        if f"{field}_type" not in doc.keys():
            field_tokens = self._get_text_tokens(field_data.strip(" "))  # list of ids
        elif doc[f"{field}_type"] == 'TEXT':
            _text = field_data.strip(" ")
            if _text.startswith("Phoneme TTS"):
                lang = doc.get("lang", "en")
                instruction_tokens = self._get_text_tokens("Phoneme TTS")
                field_tokens = self._get_phoneme_tokens(_text[len("Phoneme TTS") :].strip(), lang=lang)
                field_tokens = instruction_tokens + field_tokens
            elif _text.startswith("Edit Speech"):
                # Always use phoneme tokenizer for edit speech
                instruction_tokens = self._get_text_tokens("Edit Speech")
                field_tokens = self._get_phoneme_tokens(_text[len("Edit Speech") :].strip())
                field_tokens = instruction_tokens + field_tokens
            elif _text.startswith("TEXT CONTEXT:"):
                # Speaker id conditioning
                field_tokens = self._get_text_tokens(_text)
                # pad field tokens to fixed length
                # assert self.context_duration_min == self.context_duration_max, "TEXT CONTEXT only supports fixed context duration"
                # To keep context length the same for audio or tex context
                # _fixed_context_len = int(self.context_duration_min * self.codebook_fps)
                field_tokens = field_tokens + [self.tokenizer.eos_id]
            else:
                # if starts with Text to speech this
                field_tokens = self._get_text_tokens(field_data.strip(" "))  # list of ids
        elif doc[f"{field}_type"] == 'SPEECH':
            dur = -1
            if f"{field}_duration" in doc:
                dur = doc[f"{field}_duration"]
            field_tokens = self._get_speech_tokens(field_data, dur)  # list of ids
            if not isinstance(field_tokens, list):
                field_tokens = [field_tokens]
        elif doc[f"{field}_type"] == 'AUDIOCODEC':
            reference_codec_paths = field_data.split(";")
            reference_codec_path = rng.choice(reference_codec_paths)
            if self.codec_folder is not None:
                reference_codec_path = self.codec_folder / reference_codec_path
            field_tokens = torch.load(reference_codec_path).long()
            field_tokens[0] = (field_tokens[0] + self.speech_offset).long()
            field_tokens = [field_tokens]
            # print("AUDIOCODEC", field_tokens.shape)
        elif doc[f"{field}_type"] == 'REFSPEAKERCODEC':
            reference_codec_paths = field_data.split(";")
            reference_codec_path = rng.choice(reference_codec_paths)
            if self.codec_folder is not None:
                reference_codec_path = self.codec_folder / reference_codec_path
            field_tokens = torch.load(reference_codec_path).long()
            field_tokens[0] = (field_tokens[0] + self.speech_offset).long()
            _min_len = int(self.context_duration_min * self.codebook_fps)
            _max_len = int(self.context_duration_max * self.codebook_fps)
            reference_codec_len = rng.randint(_min_len, _max_len)
            reference_codec_len = min(reference_codec_len, field_tokens.shape[1])
            si = rng.randint(0, field_tokens.shape[1] - reference_codec_len)
            field_tokens = field_tokens[:, si : si + reference_codec_len]
            if self.context_pattern == "delay_parallel":
                field_tokens = torch.cat(
                    [
                        torch.zeros(self.num_speech_codebooks, self.num_speech_codebooks).long(),
                        field_tokens,
                        torch.zeros(self.num_speech_codebooks, self.num_speech_codebooks).long(),
                    ],
                    dim=1,
                )
                new_field_tokens = []
                for _c in range(self.num_speech_codebooks):
                    st = self.num_speech_codebooks - _c
                    et = field_tokens.shape[1] - _c
                    new_field_tokens.append(field_tokens[_c, st:et])
                field_tokens = torch.stack(new_field_tokens, dim=0)
            field_tokens = [field_tokens]
        elif doc[f"{field}_type"] == 'DUMMYCONTEXT':
            field_tokens = torch.zeros(self.num_speech_codebooks, 1).long()
            return [field_tokens]
        elif doc[f"{field}_type"] == 'CONTEXTANSWER':
            # Both Context and Answer are in the field
            context_info, answer_codec_path = field_data.split(";")
            if self.codec_folder is not None:
                context_codec_path = self.codec_folder / context_info
                answer_codec_path = self.codec_folder / answer_codec_path
            if context_info.startswith("TEXT CONTEXT:"):
                context_tokens = self._get_text_tokens(context_info.strip(" "))
                # pad field tokens to fixed length
                assert (
                    self.context_duration_min == self.context_duration_max
                ), "TEXT CONTEXT only supports fixed context duration"
                _fixed_context_len = int(self.context_duration_min * self.codebook_fps)
                context_tokens = context_tokens + [self.tokenizer.pad_id] * (_fixed_context_len - len(context_tokens))

                answer_tokens = torch.load(answer_codec_path).long()
                answer_tokens[0] = (answer_tokens[0] + self.speech_offset).long()
                field_tokens = context_tokens + [self.tokenizer.pad_id] + [answer_tokens]
            else:
                context_tokens = torch.load(context_codec_path).long()
                context_tokens[0] = (context_tokens[0] + self.speech_offset).long()
                assert (
                    self.context_duration_min == self.context_duration_max
                ), "CONTEXTANSWER only supports fixed context duration"
                reference_codec_len = int(self.context_duration_min * self.codebook_fps)
                if context_tokens.shape[1] < reference_codec_len:
                    # Repeat the context to match the reference_codec_len
                    context_tokens = torch.cat(
                        [context_tokens] * (reference_codec_len // context_tokens.shape[1] + 1), dim=1
                    )
                assert (
                    context_tokens.shape[1] >= reference_codec_len
                ), "CONTEXTANSWER context duration is less than min duration {} {} {}".format(
                    context_tokens.shape[1], reference_codec_len, context_codec_path
                )
                si = rng.randint(0, context_tokens.shape[1] - reference_codec_len)
                context_tokens = context_tokens[:, si : si + reference_codec_len]

                answer_tokens = torch.load(answer_codec_path).long()
                answer_tokens[0] = (answer_tokens[0] + self.speech_offset).long()
                pad_tokens = torch.zeros(self.num_speech_codebooks, 1).long()
                # padding between context and answer
                field_tokens = torch.cat([context_tokens, pad_tokens, answer_tokens], dim=1)
                field_tokens = [field_tokens]
        elif doc[f"{field}_type"] == 'SEPARATIONCODECS':
            mixed_codec_path, reference_codec_paths = field_data.split(",")
            reference_codec_paths = reference_codec_paths.split(";")
            reference_codec_path = rng.choice(reference_codec_paths)
            mixed_codec = torch.load(mixed_codec_path).long()
            reference_codec = torch.load(reference_codec_path).long()
            reference_codec_len = rng.randint(240, 400)
            reference_codec = reference_codec[:, :reference_codec_len]
            # MIXED AUDIO AND REF AUDIO ARE SEPARATED BY 8 TIMESTEPS OF 1023 TOKENS IN ALL CODEBOOKS
            mask_tokens = (torch.ones(self.num_speech_codebooks, self.num_speech_codebooks) * 1023).long()
            field_tokens = torch.cat([mixed_codec, mask_tokens, reference_codec], dim=1)
            field_tokens[0] = (field_tokens[0] + self.speech_offset).long()
            field_tokens = [field_tokens]
        elif doc[f"{field}_type"] == 'EDITINGCODECS':
            reference_audio_path = field_data
            reference_codec = torch.load(reference_audio_path).long()
            assert reference_codec.shape[1] > 80  # ensure reference audio is atleast 1 second
            mask_len = rng.randint(40, 320)  # ~0.5 second to 4 seconds
            mask_len = min(mask_len, reference_codec.shape[1] - 80)
            mask_start = rng.randint(0, reference_codec.shape[1] - mask_len)
            mask_end = mask_start + mask_len
            mask_tokens = (torch.ones(self.num_speech_codebooks, self.num_speech_codebooks) * 1023).long()
            seg1 = reference_codec[:, :mask_start]
            seg2 = reference_codec[:, mask_end:]
            field_tokens = torch.cat([seg1, mask_tokens, seg2], dim=1)
            # MISSING AUDIO IS REPLACED WITH 8 TIMESTEPS OF 1023 TOKENS IN ALL CODEBOOKS
            field_tokens[0] = (field_tokens[0] + self.speech_offset).long()
            field_tokens = [field_tokens]
        else:
            raise Exception(f"{field}_type not recognized")
        return field_tokens

    def _insert_data_in_template(self, prompt_template_fields, doc, answer_field):
        """Format the input example according to the template"""
        out_dict = {}
        for field in prompt_template_fields:
            # discard the last one, {label} / {answer}
            # Or if some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            if field == answer_field or field not in doc.keys():
                continue
                #  out_dict[field] = ""

            elif field in doc.keys():
                field_data = doc[field]
                if f"{field}_type" not in doc.keys():
                    doc[f"{field}_type"] = "TEXT"
                    raise Exception(f"{field}_type does not exist in doc")
                else:
                    out_dict[field] = self._get_tokens(doc, field, field_data)
        return out_dict

    def get_position_ids(self, virtual_token, context_and_qquestion):
        enc_input = []
        enc_input.append(virtual_token)
        if context_and_qquestion.dim() > 2:
            enc_input.append(context_and_qquestion[:, 0, :])
        else:
            enc_input.append(context_and_qquestion)

        enc_input = torch.cat(enc_input, dim=1)

        enc_input_p = enc_input[:, 0, :] if enc_input.dim() == 3 else enc_input
        return build_position_ids(enc_input_p).contiguous()

    def collate_fn(self, batch):
        """Prepares enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids for global batch"""

        data_dict = self.pad_batch_and_build_loss_mask(batch)

        if self.encoder_type == "multi_transformer":
            position_ids = [
                self.get_position_ids(data_dict['virtual_tokens'], data_dict['context_and_question_tokens'][0]),
                self.get_position_ids(data_dict['virtual_tokens'], data_dict['context_and_question_tokens'][1]),
            ]
        else:
            position_ids = self.get_position_ids(data_dict['virtual_tokens'], data_dict['context_and_question_tokens'])

        return (
            data_dict['virtual_tokens'],
            data_dict['context_and_question_tokens'],
            data_dict['enc_mask'],
            data_dict['dec_input'],
            data_dict['dec_input_mask'],
            data_dict['dec_labels'],
            data_dict['dec_labels_mask'],
            position_ids,
            data_dict['taskname_id'],
            data_dict['speech_mask'],
            data_dict['context_and_question_tokens_lens'],
            data_dict['cross_attention_prior'],
            data_dict['text_limits'],
            data_dict['lang'],
            data_dict['question_texts'],
        )

    def pad_batch_and_build_loss_mask(self, batch):
        """Pad enc_input, dec_input, labels in batch to max batch length while building loss_mask, enc_mask, and dec_mask"""
        (
            taskname_ids,
            _,
            virtual_tokens_len,
            _,
            _,
            context_and_question_tokens_len,
            _,
            dec_input_len,
            _,
            dec_labels_len,
            _,
            _,
            _,
            question_texts,
        ) = zip(*batch)

        taskname_ids = self.pad_taskname_ids(taskname_ids)

        max_virtual_tokens_len = max(virtual_tokens_len).item() if virtual_tokens_len is not None else 0
        if isinstance(virtual_tokens_len, tuple):
            virtual_tokens_len = torch.stack(virtual_tokens_len)
        virtual_mask = get_mask_from_lengths(virtual_tokens_len)

        if self.encoder_type == "multi_transformer":
            max_context_len = (
                max(_c[0] for _c in context_and_question_tokens_len)
                if context_and_question_tokens_len is not None
                else 0
            )
            max_question_len = (
                max(_c[1] for _c in context_and_question_tokens_len)
                if context_and_question_tokens_len is not None
                else 0
            )
            max_context_and_question_tokens_len = [max_context_len, max_question_len]
            context_len = torch.stack([_c[0] for _c in context_and_question_tokens_len])
            question_len = torch.stack([_c[1] for _c in context_and_question_tokens_len])
            context_mask = get_mask_from_lengths(context_len)
            question_mask = get_mask_from_lengths(question_len)
            context_and_question_tokens_len = [context_len, question_len]
            context_and_question_mask = [context_mask, question_mask]
            enc_mask = [
                torch.cat([virtual_mask, context_and_question_mask[0]], dim=1),
                torch.cat([virtual_mask, context_and_question_mask[1]], dim=1),
            ]
            # import ipdb; ipdb.set_trace()
        else:
            max_context_and_question_tokens_len = (
                max(context_and_question_tokens_len).item() if context_and_question_tokens_len is not None else 0
            )
            if isinstance(context_and_question_tokens_len, tuple):
                context_and_question_tokens_len = torch.stack(context_and_question_tokens_len)
            context_and_question_mask = get_mask_from_lengths(context_and_question_tokens_len)
            enc_mask = torch.cat([virtual_mask, context_and_question_mask], dim=1)

        max_dec_input_len = max(dec_input_len).item() if dec_input_len is not None else 0
        max_dec_labels_len = max(dec_labels_len).item() if dec_labels_len is not None else 0

        (
            virtual_tokens_list,
            context_question_tokens_list,
            dec_input_list,
            dec_input_mask_list,
            dec_labels_list,
            dec_labels_mask_list,
            speech_mask_list,
            cross_attention_prior_list,
            text_limits,
            lang_list,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i, sample_tuple in enumerate(batch):
            (
                _,
                virtual_token,
                virtual_token_len,
                context_token_len,
                context_and_question_token,
                context_and_question_token_len,
                dec_input,
                dec_input_len,
                dec_label,
                dec_label_len,
                is_speech,
                cross_attention_prior,
                lang,
                _,
            ) = sample_tuple

            virtual_tokens_list.append(
                general_padding(
                    virtual_token, virtual_token_len.item(), max_virtual_tokens_len, pad_value=self.tokenizer.pad_id
                )
            )

            if self.encoder_type == "multi_transformer":
                context_tokens_padded = general_padding(
                    context_and_question_token[0],
                    context_and_question_token_len[0].item(),
                    max_context_and_question_tokens_len[0],
                    pad_value=self.tokenizer.pad_id,
                )
                if len(context_tokens_padded.shape) < 2:
                    context_tokens_padded = pad_text_to_speech_dims(
                        context_tokens_padded, self.tokenizer.pad_id, self.num_speech_codebooks - 1
                    )
                question_tokens_padded = general_padding(
                    context_and_question_token[1],
                    context_and_question_token_len[1].item(),
                    max_context_and_question_tokens_len[1],
                    pad_value=self.tokenizer.pad_id,
                )
                if len(question_tokens_padded.shape) < 2:
                    question_tokens_padded = pad_text_to_speech_dims(
                        question_tokens_padded, self.tokenizer.pad_id, self.num_speech_codebooks - 1
                    )
                context_question_tokens_list.append([context_tokens_padded, question_tokens_padded])
            else:
                # This means context and questions are concatenated together
                context_tokens_padded = general_padding(
                    context_and_question_token,
                    context_and_question_token_len.item(),
                    max_context_and_question_tokens_len,
                    pad_value=self.tokenizer.pad_id,
                )
                if len(context_tokens_padded.shape) < 2:
                    context_tokens_padded = pad_text_to_speech_dims(
                        context_tokens_padded, self.tokenizer.pad_id, self.num_speech_codebooks - 1
                    )
                context_question_tokens_list.append(context_tokens_padded)

            if max_dec_input_len > 0:
                dec_input_padded = general_padding(
                    dec_input, dec_input_len.item(), max_dec_input_len, pad_value=self.tokenizer.pad_id
                )
                if len(dec_input_padded.shape) < 2:
                    dec_input_padded = pad_text_to_speech_dims(
                        dec_input_padded, self.tokenizer.pad_id, self.num_speech_codebooks - 1
                    )
                dec_input_list.append(dec_input_padded)
                dec_mask = (
                    torch.as_tensor(([1] * dec_input_len) + ([0] * (max_dec_input_len - dec_input_len)))
                    .long()
                    .contiguous()
                )
                dec_input_mask_list.append(dec_mask)
                speech_mask = dec_mask if is_speech else torch.zeros(dec_mask.shape)
                speech_mask_list.append(speech_mask)

            if max_dec_labels_len > 0:
                loss_mask = (
                    torch.as_tensor(([1] * dec_label_len) + ([0] * (max_dec_labels_len - dec_label_len)))
                    .long()
                    .contiguous()
                )
                dec_label_padded = general_padding(
                    dec_label, dec_label_len.item(), max_dec_labels_len, pad_value=self.tokenizer.pad_id
                )
                if len(dec_label_padded.shape) < 2:
                    dec_label_padded = pad_text_to_speech_dims(
                        dec_label_padded, self.tokenizer.pad_id, self.num_speech_codebooks - 1
                    )
                dec_labels_list.append(dec_label_padded)
                dec_labels_mask_list.append(loss_mask)

                _p0 = max_dec_labels_len - dec_label_len
                if self.encoder_type == "multi_transformer":
                    _p1 = (
                        max_virtual_tokens_len
                        + max_context_and_question_tokens_len[1]
                        - context_and_question_token_len[1]
                        - virtual_token_len
                    )
                else:
                    _p1 = (
                        max_virtual_tokens_len
                        + max_context_and_question_tokens_len
                        - context_and_question_token_len
                        - virtual_token_len
                    )

                cross_attention_prior_padded = torch.nn.functional.pad(
                    cross_attention_prior,
                    pad=(0, _p1, 0, _p0),
                    mode="constant",
                    value=1,
                )
                cross_attention_prior_list.append(cross_attention_prior_padded)

                if self.encoder_type == "multi_transformer":
                    _start_of_text_id = virtual_token_len + 4
                    _end_of_text_id = _start_of_text_id + (
                        context_and_question_token_len[1] - 2 - 4
                    )  # -2 for some end tokens
                else:
                    _start_of_text_id = virtual_token_len + context_token_len + 4
                    _end_of_text_id = _start_of_text_id + (
                        context_and_question_token_len - context_token_len - 2 - 4
                    )  # -2 for some end tokens
                text_limits.append(torch.tensor([_start_of_text_id.item(), _end_of_text_id.item()]))
                lang_list.append(torch.tensor(lang))

        dec_labels_mask = torch.stack(dec_labels_mask_list) if len(dec_labels_mask_list) > 0 else None
        if dec_labels_mask is not None and self.context_conditioning == 'decoder':
            # Mask out context tokens from loss computation. +1 for bos/pad in the beginning
            dec_labels_mask[:, : self.decoder_context_len + 1] = 0

        if self.encoder_type == "multi_transformer":
            context_batch = torch.stack([c[0] for c in context_question_tokens_list])
            question_batch = torch.stack([c[1] for c in context_question_tokens_list])
            context_and_question_tokens = [context_batch, question_batch]
        else:
            context_and_question_tokens = torch.stack(context_question_tokens_list)

        data_dict = {
            "taskname_id": taskname_ids,
            "virtual_tokens": torch.stack(virtual_tokens_list),
            "context_and_question_tokens": context_and_question_tokens,
            "enc_mask": enc_mask,
            "dec_input": torch.stack(dec_input_list) if len(dec_input_list) > 0 else None,
            "dec_input_mask": torch.stack(dec_input_mask_list) if len(dec_input_mask_list) > 0 else None,
            "dec_labels": torch.stack(dec_labels_list) if len(dec_labels_list) > 0 else None,
            "dec_labels_mask": dec_labels_mask,
            "speech_mask": torch.stack(speech_mask_list) if len(speech_mask_list) > 0 else None,
            "context_and_question_tokens_lens": context_and_question_tokens_len,
            "cross_attention_prior": (
                torch.stack(cross_attention_prior_list) if len(cross_attention_prior_list) > 0 else None
            ),
            "text_limits": (
                torch.stack(text_limits) if len(text_limits) > 0 else None
            ),  # tensor, valid range of answer transcripts without virtual/instruction/end tokens.
            "lang": torch.stack(lang_list),
            "question_texts": question_texts,
        }

        return data_dict
