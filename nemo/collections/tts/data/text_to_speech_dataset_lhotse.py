# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random
import re
from typing import Dict, List, Union

import numpy as np
import torch
from hydra.utils import instantiate
from lhotse import CutSet
from lhotse.dataset.collation import collate_matrices, collate_vectors
from omegaconf import DictConfig
from transformers import AutoTokenizer, T5Tokenizer

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    beta_binomial_prior_distribution,
    normalize_volume,
    stack_tensors,
)
from nemo.utils import logging


def setup_tokenizers(all_tokenizers_config, mode='train'):
    # Being used in both model and worker_init_fn, so it is defined here
    # Returns two tokenizers: one for TTS transcript and one for conditioning text (if needed)
    tokenizers = []
    tokenizer_names = []
    for tokenizer_name in all_tokenizers_config:
        tokenizer_config = all_tokenizers_config[tokenizer_name]
        if tokenizer_config._target_ == 'AutoTokenizer':
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.pretrained_model)
        elif tokenizer_config._target_ == 'T5Tokenizer':
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_config.pretrained_model)
        else:
            text_tokenizer_kwargs = {}
            if "g2p" in tokenizer_config:
                text_tokenizer_kwargs["g2p"] = instantiate(tokenizer_config.g2p)
            tokenizer = instantiate(tokenizer_config, **text_tokenizer_kwargs)
            # TODO @xueyang: is it really necessary to set phone probability to 1.0 for test mode?
            if mode == 'test' and hasattr(tokenizer, "set_phone_prob"):
                tokenizer.set_phone_prob(1.0)
        tokenizers.append(tokenizer)
        tokenizer_names.append(tokenizer_name)

    aggregated_tokenizer = AggregatedTTSTokenizer(tokenizers, tokenizer_names)  # TTS Transcript tokenizer

    return aggregated_tokenizer


def check_speaker_format(item: str):
    # enforce the format as example like "| Language:en Dataset:HiFiTTS Speaker:9136_other |".
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    return bool(re.match(pattern, item))


class MagpieTTSLhotseDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading and processing Text-to-Speech data for
    MagpieTTS models using Lhotse CutSets, specifically designed for datasets
    with text or audio context. But either context can be optional.

    This dataset expects Lhotse Cut objects where each cut represents a
    target utterance along with its preceding context. Context can be
    audio (preferred) or text. It handles loading either pre-computed audio
    codes or raw audio waveforms, applying volume normalization, and tokenizing
    text transcripts. Context audio/codes are sliced or repeated to fit within
    a specified duration range. Optionally, it loads 16kHz audio suitable for
    speaker verification models and calculates alignment priors.

    Tokenizers (for target text and optional context text) are initialized lazily
    within each dataloader worker process upon first access.

    Args:
        sample_rate (int): Target sample rate for loading audio. Audio will be
            resampled if necessary.
        volume_norm (bool): If True, applies peak volume normalization to audio
            waveforms. Defaults to True.
        codec_model_samples_per_frame (int): The total downsampling factor of the
            audio codec model used to generate codes. Used for padding audio
            and calculating number of codec frames.
        audio_bos_id (int): Token ID representing the beginning-of-sequence (BOS) for
            target audio codes.
        audio_eos_id (int): Token ID representing the end-of-sequence (EOS) for target
            audio codes.
        context_audio_bos_id (int): Token ID representing the beginning-of-sequence (BOS)
            for context audio codes.
        context_audio_eos_id (int): Token ID representing the end-of-sequence (EOS)for
            context audio codes.
        num_audio_codebooks (int): Number of codebooks used by the audio codec model.
            Needed for creating dummy context codes if necessary.
        prior_scaling_factor (Optional[float]): Scaling factor for the beta-binomial
            alignment prior calculation. If None, priors are not computed. Defaults to None.
        load_cached_codes_if_available (bool): If True, attempts to load pre-computed
            audio codes from custom fields in the Lhotse Cut (e.g., 'codes_21fpsCausalDecoder',
            'context_codes_21fpsCausalDecoder'). Falls back to loading audio if codes
            are not found. Defaults to True.
        dataset_type (str): Specifies the mode ('train' or 'test'), mainly affecting
            tokenizer settings like phoneme probability. Defaults to 'train'.
        load_16khz_audio (bool): If True, loads 16kHz audio suitable for speaker
            verification models. It prioritizes context audio ('context_recording' field)
            if available, otherwise uses the target audio ('recording' field).
            Defaults to True.
        pad_context_text_to_max_duration (bool): If True and `use_text_conditioning_tokenizer`
            is True, pads the tokenized context text to a length derived from
            `context_duration_max`. Defaults to False.
        context_duration_min (float): Minimum duration (in seconds) for the context
            audio/codes. Context shorter than this will be repeated. Defaults to 3.0.
        context_duration_max (float): Maximum duration (in seconds) for the context
            audio/codes. Context longer than this will be sliced randomly. Defaults to 10.0.
        use_text_conditioning_tokenizer (bool): If True, enables processing of context
            text using a separate tokenizer (currently T5Tokenizer). Expects context text
            in `cut.supervisions[0].custom['context_text']`. Defaults to False.
        tokenizer_config (Optional[DictConfig]): Configuration for the text tokenizers.
            Used for lazy initialization within workers. Must be provided if tokenizers
            are not set externally. Defaults to None.
        text_context_remapping: Dict defining mapping of multiple text contexts to a single text context.
        text_context_remapping_prob: Probability of remapping the original text context to a remapped text context.
    """

    def __init__(
        self,
        sample_rate: int,
        volume_norm: bool = True,
        codec_model_samples_per_frame: int = None,
        audio_bos_id: int = None,
        audio_eos_id: int = None,
        context_audio_bos_id: int = None,
        context_audio_eos_id: int = None,
        num_audio_codebooks: int = None,
        prior_scaling_factor: float = None,
        load_cached_codes_if_available: bool = True,
        dataset_type: str = 'train',
        load_16khz_audio: bool = True,
        pad_context_text_to_max_duration: bool = False,
        context_duration_min: float = 3.0,
        context_duration_max: float = 10.0,
        use_text_conditioning_tokenizer: bool = False,
        text_conditioning_tokenizer_name: str = None,
        tokenizer_config: DictConfig = None,
        text_context_remapping: Dict[str, str] = None,
        text_context_remapping_prob: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.volume_norm = volume_norm
        self.audio_bos_id = audio_bos_id
        self.audio_eos_id = audio_eos_id
        self.context_audio_bos_id = context_audio_bos_id
        self.context_audio_eos_id = context_audio_eos_id

        self.codec_model_samples_per_frame = codec_model_samples_per_frame
        self.num_audio_codebooks = num_audio_codebooks

        self.include_align_prior = prior_scaling_factor is not None
        self.prior_scaling_factor = prior_scaling_factor
        self.load_cached_codes_if_available = load_cached_codes_if_available
        self.dataset_type = dataset_type  # 'train' or 'test'
        self.load_16khz_audio = load_16khz_audio
        self.use_text_conditioning_tokenizer = use_text_conditioning_tokenizer
        self.text_conditioning_tokenizer_name = text_conditioning_tokenizer_name
        self.pad_context_text_to_max_duration = pad_context_text_to_max_duration
        self.context_duration_min = context_duration_min
        self.context_duration_max = context_duration_max
        self.tokenizer_config = tokenizer_config
        self.text_tokenizer = None
        self.text_context_remapping = text_context_remapping
        self.text_context_remapping_prob = text_context_remapping_prob

    def get_num_audio_samples_to_slice(self, duration, sample_rate):
        num_codec_frames = int(duration * sample_rate / self.codec_model_samples_per_frame)
        num_audio_samples = num_codec_frames * self.codec_model_samples_per_frame
        return num_audio_samples

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List]]:
        # layze initialize tokenizers. The first time any specific worker
        # process calls this function, on its copy of the dataset, the
        # tokenizers are created for that worker. All subsequent calls
        # to this function will reuse the tokenizers. This equivilent to
        # the `worker_init_fn` in MagpieTTSModel.
        if self.text_tokenizer is None:
            # First time this worker is accessing the dataset, initialize the
            # tokenizers. If called by the main process (num_workers=0), worker_info will be None.
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            logging.info(f"Worker {worker_id} initializing tokenizers...")
            self.text_tokenizer = setup_tokenizers(
                all_tokenizers_config=self.tokenizer_config,
                mode=self.dataset_type,
            )
            self.bos_id = len(self.text_tokenizer.tokens)
            self.eos_id = self.bos_id + 1
            self.pad_id = self.text_tokenizer.pad

        # define list to store batched information
        dataset_name_list = []
        audio_list = []
        audio_len_list = []
        audio_list_16khz = []
        audio_len_list_16khz = []
        token_list = []
        token_len_list = []
        prior_list = []
        audio_codes_list = []
        audio_codes_len_list = []
        context_audio_list = []
        context_audio_len_list = []
        context_audio_codes_list = []
        context_audio_codes_len_list = []
        context_text_tokens_list = []
        context_text_tokens_len_list = []
        context_has_text_context_list = []
        reward_list = []
        raw_text_list = (
            []
        )  # raw text here is the string of normalized text or text stored in the supervision segment. Used to distinguish from text tokens.
        for cut in cuts:
            speaker = cut.supervisions[0].speaker
            if not check_speaker_format(speaker):
                raise ValueError(f"Invalid format in cut.supervisions[0].speaker: {speaker}")
            dataset_name = speaker.strip().split()[2].split(":")[-1]
            dataset_name_list.append(dataset_name)

            # target audio or target codes
            if self.load_cached_codes_if_available and cut.has_custom("target_codes"):
                # TODO @xueyang: applying Tensor.long(), i.e. torch.int64, is not necessary.

                # Note that we have segmented the audio according to offset and duration so that the audio codes should
                # not specify start and duration again when calling TemporalArray.load(start, duration). Ensure start
                # and duration are None to the load function.
                audio_codes = torch.from_numpy(cut.target_codes.load()).long()  # (C, T)
                spec_len = audio_codes.shape[1] + 1  # +1 for EOS
                audio_bos_tensor = torch.full((audio_codes.shape[0], 1), self.audio_bos_id, dtype=audio_codes.dtype)
                audio_eos_tensor = torch.full((audio_codes.shape[0], 1), self.audio_eos_id, dtype=audio_codes.dtype)
                audio_codes = torch.cat([audio_bos_tensor, audio_codes, audio_eos_tensor], dim=1)
                audio_codes_len = audio_codes.shape[1]
                audio_codes_list.append(audio_codes.T)  # transpose to (T, C) to use collate_matrices to process batch.
                audio_codes_len_list.append(audio_codes_len)
            else:
                # Only load audio if codes are not available
                audio_array = cut.recording.resample(self.sample_rate).load_audio().squeeze(0)
                if self.volume_norm:
                    audio_array = normalize_volume(audio_array)
                audio = torch.from_numpy(audio_array)
                # Pad audio to be multiple of downsample factor
                audio = torch.nn.functional.pad(
                    audio,
                    (0, self.codec_model_samples_per_frame - (audio.shape[0] % self.codec_model_samples_per_frame)),
                    value=0,
                )
                audio_len = audio.shape[0]
                spec_len = int(audio_len / self.codec_model_samples_per_frame) + 1  # +1 for EOS
                audio_list.append(audio)
                audio_len_list.append(audio_len)

            # context audio or context codes
            if self.load_cached_codes_if_available and cut.has_custom("context_codes"):
                # TODO @xueyang: applying Tensor.long(), i.e. torch.int64, is not necessary.

                # Note that we have segmented the audio according to offset and duration so that the audio codes should
                # not specify start and duration again when calling TemporalArray.load(start, duration). Ensure start
                # and duration are None to the load function.
                context_audio_codes = torch.from_numpy(cut.context_codes.load()).long()  # (8, T)
                # Sample random duration between self.context_duration_min and self.context_duration_max
                _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
                _num_frames_to_slice = int(
                    _context_duration_to_slice * self.sample_rate / self.codec_model_samples_per_frame
                )
                if _num_frames_to_slice < context_audio_codes.shape[1]:
                    start_idx = random.randint(0, context_audio_codes.shape[1] - _num_frames_to_slice)
                    context_audio_codes = context_audio_codes[:, start_idx : start_idx + _num_frames_to_slice]
                else:
                    # Repeat the audio if it is shorter than the desired duration
                    _num_repeats = int(np.ceil(_num_frames_to_slice / context_audio_codes.shape[1]))
                    # context_audio_codes is a tensor of shape (num_codebooks, T)
                    context_audio_codes_repeated = context_audio_codes.repeat(1, _num_repeats)
                    context_audio_codes = context_audio_codes_repeated[:, :_num_frames_to_slice]

                context_bos_tensor = torch.full(
                    (context_audio_codes.shape[0], 1), self.context_audio_bos_id, dtype=context_audio_codes.dtype
                )
                context_eos_tensor = torch.full(
                    (context_audio_codes.shape[0], 1), self.context_audio_eos_id, dtype=context_audio_codes.dtype
                )
                context_audio_codes = torch.cat([context_bos_tensor, context_audio_codes, context_eos_tensor], dim=1)
                context_audio_codes_len = context_audio_codes.shape[1]
                context_audio_codes_list.append(
                    context_audio_codes.T
                )  # transpose to (T, 8) in order to use collate_matrices to process batch.
                context_audio_codes_len_list.append(context_audio_codes_len)
            elif cut.has_custom("context_recording"):
                # Only load audio if codes are not available
                context_audio_array = cut.context_recording.resample(self.sample_rate).load_audio().squeeze(0)
                if self.volume_norm:
                    context_audio_array = normalize_volume(context_audio_array)
                _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
                _num_samples_to_slice = self.get_num_audio_samples_to_slice(
                    _context_duration_to_slice, self.sample_rate
                )
                if _num_samples_to_slice < len(context_audio_array):
                    start_idx = random.randint(0, len(context_audio_array) - _num_samples_to_slice)
                    context_audio_array = context_audio_array[start_idx : start_idx + _num_samples_to_slice]
                else:
                    # Repeat the audio if it is shorter than the desired duration
                    _num_repeats = int(np.ceil(_num_samples_to_slice / len(context_audio_array)))
                    context_audio_array = np.tile(context_audio_array, _num_repeats)
                    context_audio_array = context_audio_array[:_num_samples_to_slice]
                context_audio = torch.from_numpy(context_audio_array)
                context_audio_len = context_audio.shape[0]
                context_audio_list.append(context_audio)
                context_audio_len_list.append(context_audio_len)
            else:
                # We always want to have context_audio_codes if available for multi-encoder model. These are ignored for single-encoder model.
                # If context audio is not available, just use a dummy context_audio_codes
                # (Will be used in text context scenario)
                # TODO @xueyang: verified that this block should cover below 3 conditions which were handled well.
                #  1. load_cached_codes_if_available and ["context_audio_codes_path", "context_audio_filepath"] not in data.manifest_entry;
                #        assign to example["context_audio_codes"] and example["context_audio_codes_len"]
                #  2. load_cached_codes_if_available is not True and "context_audio_codes_path" in data.manifest_entry;
                #        assign to example["context_audio"] and example["context_audio_len"]
                #  3. load_cached_codes_if_available is not True and ["context_audio_codes_path", "context_audio_filepath"] not in data.manifest_entry;
                #        assign to example["context_audio"] and example["context_audio_len"]
                if self.load_cached_codes_if_available:
                    context_bos_tensor = torch.full(
                        (self.num_audio_codebooks, 1), self.context_audio_bos_id, dtype=torch.int32
                    )
                    context_eos_tensor = torch.full(
                        (self.num_audio_codebooks, 1), self.context_audio_eos_id, dtype=torch.int32
                    )
                    context_audio_codes = torch.cat([context_bos_tensor, context_eos_tensor], dim=1)
                    context_audio_codes_len = context_audio_codes.shape[1]
                    context_audio_codes_list.append(
                        context_audio_codes.T
                    )  # transpose to (T, C) to use collate_matrices to process batch.
                    context_audio_codes_len_list.append(context_audio_codes_len)
                else:
                    # @shehzeenh: Added this condition so that a batch does not have a mix of context_audio and context_audio_codes
                    context_audio = torch.zeros(self.codec_model_samples_per_frame, dtype=torch.float32)
                    context_audio_len = context_audio.shape[0]
                    context_audio_list.append(context_audio)
                    context_audio_len_list.append(context_audio_len)

            if self.load_16khz_audio:
                if cut.has_custom("context_recording"):
                    # use context audio for SV model
                    audio_array_16khz = cut.context_recording.resample(16_000).load_audio().squeeze(0)
                    if self.volume_norm:
                        audio_array_16khz = normalize_volume(audio_array_16khz)
                else:
                    # Otherwise, load the target audio for SV model.
                    audio_array_16khz = cut.recording.resample(16_000).load_audio().squeeze(0)
                    if self.volume_norm:
                        audio_array_16khz = normalize_volume(audio_array_16khz)
                _context_duration_to_slice = random.uniform(self.context_duration_min, self.context_duration_max)
                _num_samples_to_slice = int(_context_duration_to_slice * 16_000)
                if _num_samples_to_slice < len(audio_array_16khz):
                    start_idx = random.randint(0, len(audio_array_16khz) - _num_samples_to_slice)
                    audio_array_16khz = audio_array_16khz[start_idx : start_idx + _num_samples_to_slice]
                audio_16khz = torch.from_numpy(audio_array_16khz)
                audio_len_16khz = audio_16khz.shape[0]
                audio_list_16khz.append(audio_16khz)
                audio_len_list_16khz.append(audio_len_16khz)

            if self.use_text_conditioning_tokenizer:
                if cut.supervisions[0].has_custom("context_text"):
                    context_text = cut.supervisions[0].context_text
                    if self.text_context_remapping is not None and context_text in self.text_context_remapping:
                        if self.dataset_type == 'train' and random.random() < self.text_context_remapping_prob:
                            # Only remap during training. Give the exact text context during inference.
                            context_text = self.text_context_remapping[context_text]
                    context_text_tokens = self.text_tokenizer.encode(
                        context_text, tokenizer_name=self.text_conditioning_tokenizer_name
                    )
                    has_text_context = True
                else:
                    context_text_tokens = self.text_tokenizer.encode(
                        "[NO TEXT CONTEXT]", tokenizer_name=self.text_conditioning_tokenizer_name
                    )
                    has_text_context = False
                if self.pad_context_text_to_max_duration:
                    _required_len = (
                        int(self.context_duration_max * self.sample_rate / self.codec_model_samples_per_frame) + 2
                    )  # +2 for BOS and EOS
                    if len(context_text_tokens) < _required_len:
                        _pad_id = self.text_tokenizer.tokenizer_pad_ids[self.text_conditioning_tokenizer_name]
                        context_text_tokens += [_pad_id] * (_required_len - len(context_text_tokens))
                    else:
                        # TODO @xueyang: It seems counter intuition if trimming the text context tokens to the required
                        #  context length. For example, the context_tokens after trimming may correspond to the partial
                        #  context_text like "Speaker and Emotion: | Language:en Dataset" where the following string is trimmed: ":Riva Speaker:Rodney_DROP |".
                        context_text_tokens = context_text_tokens[:_required_len]
                context_text_tokens = torch.tensor(context_text_tokens, dtype=torch.int32)
                context_text_tokens_len = context_text_tokens.shape[0]
                context_text_tokens_list.append(context_text_tokens)
                context_text_tokens_len_list.append(context_text_tokens_len)
                context_has_text_context_list.append(has_text_context)

            # tokenize transcript
            # there may exist "normalized_text" in the suprvisionsegement. Prioritize it over "text" if available.
            if cut.supervisions[0].has_custom("normalized_text"):
                text_str = cut.supervisions[0].normalized_text
            else:
                text_str = cut.supervisions[0].text
            raw_text_list.append(text_str)
            if cut.has_custom("tokenizer_names"):
                # Pick a random tokenizer from the list of tokenizers
                tokenizer_name = random.choice(cut.tokenizer_names)
            else:
                tokenizer_name = "english_phoneme"  # Default to english phoneme tokenizer
            tokens = self.text_tokenizer.encode(text=text_str, tokenizer_name=tokenizer_name)
            tokens = tokens + [self.eos_id]  # Not adding BOS id
            tokens = torch.tensor(tokens, dtype=torch.int32)
            text_len = tokens.shape[0]
            token_list.append(tokens)
            token_len_list.append(text_len)

            if self.include_align_prior:
                align_prior = beta_binomial_prior_distribution(
                    phoneme_count=text_len, mel_count=spec_len, scaling_factor=self.prior_scaling_factor
                )
                align_prior = torch.tensor(align_prior, dtype=torch.float32)
                prior_list.append(align_prior)

            if cut.supervisions[0].has_custom("reward"):
                reward = cut.supervisions[0].reward
                reward_list.append(reward)

        # collate vectors and matrices here.
        batch_dict = {
            "dataset_names": dataset_name_list,
            "raw_texts": raw_text_list,
            "text": collate_vectors(token_list, padding_value=self.pad_id),  # (B, max_len)
            "text_lens": torch.IntTensor(token_len_list),
        }

        # audio for SV.
        if len(audio_list_16khz) > 0:
            batch_dict["audio_16khz"] = collate_vectors(audio_list_16khz, padding_value=0.0)
            batch_dict["audio_lens_16khz"] = torch.IntTensor(audio_len_list_16khz)

        # target audio and codes
        if len(audio_list) > 0:
            batch_dict["audio"] = collate_vectors(audio_list, padding_value=0.0)
            batch_dict["audio_lens"] = torch.IntTensor(audio_len_list)
        if len(audio_codes_list) > 0:
            # transpose back to (B, 8, T) from (B, T, 8).
            batch_dict["audio_codes"] = collate_matrices(audio_codes_list, padding_value=0).transpose(1, 2)
            batch_dict["audio_codes_lens"] = torch.IntTensor(audio_codes_len_list)

        # context audio and codes
        if len(context_audio_list) > 0:
            batch_dict["context_audio"] = collate_vectors(context_audio_list, padding_value=0.0)
            batch_dict["context_audio_lens"] = torch.IntTensor(context_audio_len_list)
        if len(context_audio_codes_list) > 0:
            # transpose back to (B, 8, T) from (B, T, 8).
            batch_dict["context_audio_codes"] = collate_matrices(context_audio_codes_list, padding_value=0).transpose(
                1, 2
            )
            batch_dict["context_audio_codes_lens"] = torch.IntTensor(context_audio_codes_len_list)

        if self.use_text_conditioning_tokenizer:
            batch_dict['context_text_tokens'] = collate_vectors(
                tensors=context_text_tokens_list,
                padding_value=self.text_tokenizer.tokenizer_pad_ids[self.text_conditioning_tokenizer_name],
            )
            batch_dict['context_text_tokens_lens'] = torch.IntTensor(context_text_tokens_len_list)
            batch_dict['has_text_context'] = torch.BoolTensor(context_has_text_context_list)

        if self.include_align_prior:
            spec_max_len = max([prior.shape[0] for prior in prior_list])
            text_max_len = max([prior.shape[1] for prior in prior_list])
            batch_dict["align_prior_matrix"] = stack_tensors(prior_list, max_lens=[text_max_len, spec_max_len])

        if len(reward_list) > 0:
            batch_dict['rewards'] = torch.FloatTensor(reward_list)

        # Assert only ONE of context_audio or context_audio_codes in the batch
        assert ('audio' in batch_dict) ^ ('audio_codes' in batch_dict)

        # Assert only ONE of context_audio or context_audio_codes in the batch
        if 'context_audio' in batch_dict:
            assert 'context_audio_codes' not in batch_dict
        if 'context_audio_codes' in batch_dict:
            assert 'context_audio' not in batch_dict

        return batch_dict
