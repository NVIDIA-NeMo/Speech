# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from hydra.utils import instantiate
from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_matrices, collate_vectors, collate_audio
from lhotse.utils import ifnone
from omegaconf import DictConfig
from transformers import AutoTokenizer, T5Tokenizer

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPABPETokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    beta_binomial_prior_distribution,
    normalize_volume,
    stack_tensors,
)
from nemo.utils import logging


def setup_tokenizers(all_tokenizers_config, mode='train'):
    tokenizers = []
    tokenizer_names = []
    for tokenizer_name in all_tokenizers_config:
        tokenizer_config = all_tokenizers_config[tokenizer_name]
        if tokenizer_config._target_ == 'AutoTokenizer':
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.pretrained_model, trust_remote_code=True)
        elif tokenizer_config._target_ == 'T5Tokenizer':
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_config.pretrained_model)
        else:
            text_tokenizer_kwargs = {}
            if "g2p" in tokenizer_config:
                text_tokenizer_kwargs["g2p"] = instantiate(tokenizer_config.g2p)
            tokenizer = instantiate(tokenizer_config, **text_tokenizer_kwargs)
            if mode == 'test' and hasattr(tokenizer, "set_phone_prob"):
                tokenizer.set_phone_prob(1.0)
        tokenizers.append(tokenizer)
        tokenizer_names.append(tokenizer_name)

    aggregated_tokenizer = AggregatedTTSTokenizer(tokenizers, tokenizer_names)
    return aggregated_tokenizer


def check_speaker_format(item: str):
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    return bool(re.match(pattern, item))


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    if text is None:
        return ""
    text = _TIMESTAMP_PATTERN.sub("", text)
    return _SPACE_PATTERN.sub(" ", text).strip()


def _get_supervision_ipa_text(supervision) -> str:
    """Return IPA for a supervision, preferring top-level field over custom."""
    ipa_text = getattr(supervision, "ipa", None)
    if isinstance(ipa_text, str) and ipa_text.strip():
        return ipa_text

    custom = getattr(supervision, "custom", None)
    if isinstance(custom, dict):
        custom_ipa = custom.get("ipa")
        if isinstance(custom_ipa, str):
            return custom_ipa

    return ""


class MagpieTTSLhotseMultiturnDataset(torch.utils.data.Dataset):
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
            verification models. It prioritizes context audio ('context_audio' field)
            if available, otherwise uses the target audio ('target_audio' field).
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
        codec_model_input_sample_rate: int = None,
        frame_stacking_factor: int = None,
        num_audio_codebooks: int = None,
        prior_scaling_factor: float = None,
        load_cached_codes_if_available: bool = True,
        dataset_type: str = 'train',
        load_16khz_audio: bool = False,
        pad_context_text_to_max_duration: bool = False,
        context_duration_min: float = 3.0,
        context_duration_max: float = 10.0,
        use_text_conditioning_tokenizer: bool = False,
        text_conditioning_tokenizer_name: str = None,
        tokenizer_config: DictConfig = None,
        text_context_remapping: Dict[str, str] = None,
        text_context_remapping_prob: float = 0.0,
        phoneme_tokenizer_config: DictConfig = None,
        ignore_phoneme_languages: List[str] = None,
        add_language_to_context_text: bool = False,
        source_sample_rate: int = 16000,
        input_roles: List[str] = ["user", "User"],
        output_roles: List[str] = ["assistant", "Assistant", "agent", "Agent"],
        add_text_bos: bool = False,
        remove_user_turns_prob: float = None,
    ):
        # super().__init__()
        self.sample_rate = sample_rate
        self.volume_norm = volume_norm

        self.codec_model_samples_per_frame = codec_model_samples_per_frame
        self.num_audio_codebooks = num_audio_codebooks
        self.remove_user_turns_prob = remove_user_turns_prob

        self.include_align_prior = prior_scaling_factor is not None
        self.prior_scaling_factor = prior_scaling_factor
        self.load_cached_codes_if_available = load_cached_codes_if_available
        self.dataset_type = dataset_type
        self.load_16khz_audio = load_16khz_audio
        self.use_text_conditioning_tokenizer = use_text_conditioning_tokenizer
        self.text_conditioning_tokenizer_name = text_conditioning_tokenizer_name
        self.pad_context_text_to_max_duration = pad_context_text_to_max_duration
        self.context_duration_min = context_duration_min
        self.context_duration_max = context_duration_max
        self.tokenizer_config = tokenizer_config
        self.text_tokenizer = None
        self.phoneme_tokenizer = None
        self.text_context_remapping = text_context_remapping
        self.text_context_remapping_prob = text_context_remapping_prob
        self.phoneme_tokenizer_config = phoneme_tokenizer_config
        self.ignore_phoneme_languages = ignore_phoneme_languages or []
        self.add_language_to_context_text = add_language_to_context_text

        self.source_sample_rate = source_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.add_text_bos = add_text_bos

        self.frame_length = (self.codec_model_samples_per_frame / codec_model_input_sample_rate) * frame_stacking_factor

    def get_num_audio_samples_to_slice(self, duration, sample_rate):
        num_codec_frames = int(duration * sample_rate / self.codec_model_samples_per_frame)
        num_audio_samples = num_codec_frames * self.codec_model_samples_per_frame
        return num_audio_samples

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List]]:
        if self.text_tokenizer is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            logging.info(f"Worker {worker_id} initializing tokenizers...")
            self.text_tokenizer = setup_tokenizers(
                all_tokenizers_config=self.tokenizer_config,
                mode=self.dataset_type,
            )
            num_tokens = len(self.text_tokenizer.tokens)
            self.bos_id = num_tokens
            self.eos_id = num_tokens + 1
            self.cfg_unk_token_id = num_tokens + 2
            self.interruption_token_id = num_tokens + 3
            self.pad_id = self.text_tokenizer.pad

        if self.phoneme_tokenizer is None and self.phoneme_tokenizer_config is not None:
            self.phoneme_tokenizer = instantiate(self.phoneme_tokenizer_config)

        cuts = cuts.transform_text(_strip_timestamps)

        batch_tokenizer_names = []
        remove_user_turn_flags = []
        for cut in cuts:
            if cut.has_custom("tokenizer_names"):
                batch_tokenizer_names.append(random.choice(cut.tokenizer_names))
            else:
                batch_tokenizer_names.append("english_phoneme")

            # Get all agent supervisions in this cut
            agent_sups = [sup for sup in cut.supervisions if sup.speaker in self.output_roles]

            # It is a multiturn if there's more than 1 agent turn
            is_multiturn = not (len(agent_sups) == 1)

            # Apply augmentation only if it's multiturn AND passes the probability check
            if is_multiturn and self.remove_user_turns_prob and random.random() < self.remove_user_turns_prob:
                remove_user_turn_flags.append(True)
            else:
                remove_user_turn_flags.append(False)

        def _align_codebooks(t):
            C = t.shape[1]
            if C < self.num_audio_codebooks:
                return F.pad(t, (0, self.num_audio_codebooks - C))
            elif C > self.num_audio_codebooks:
                return t[:, :self.num_audio_codebooks]
            return t

        with fp32_precision():
            target_audio, target_audio_lens = collate_audio(
                cuts.resample(self.sample_rate, recording_field="target_audio"), recording_field="target_audio"
            )
            source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
            target_audio_list = []
            source_audio_list = []
            # normalize volume and apply audio the removal of user turn if needed

            for i, cut in enumerate(cuts):
                remove_user_turn_this_cut = remove_user_turn_flags[i]
                
                # Extract the raw, unpadded 1D numpy array for this specific cut
                t_audio = target_audio[i, :target_audio_lens[i]].numpy()
                s_audio = source_audio[i, :source_audio_lens[i]].numpy()
                if remove_user_turn_this_cut:
                    collapsed_t, collapsed_s = [], []
                    for sup in cut.supervisions:
                        if sup.speaker in self.output_roles:
                            start_t = int(round(max(0, sup.start) * self.sample_rate))
                            end_t = int(round(sup.end * self.sample_rate))
                            start_s = int(round(max(0, sup.start) * self.source_sample_rate))
                            end_s = int(round(sup.end * self.source_sample_rate))
                            
                            # Clamp safely inside the array
                            start_t, end_t = min(start_t, len(t_audio)), min(end_t, len(t_audio))
                            start_s, end_s = min(start_s, len(s_audio)), min(end_s, len(s_audio))
                            
                            if end_t > start_t: collapsed_t.append(t_audio[start_t:end_t])
                            if end_s > start_s: collapsed_s.append(s_audio[start_s:end_s])
                                
                    t_audio = np.concatenate(collapsed_t) if collapsed_t else np.zeros(1, dtype=np.float32)
                    s_audio = np.concatenate(collapsed_s) if collapsed_s else np.zeros(1, dtype=np.float32)

                # Apply volume norm locally (so we only normalize the stitched audio, saving math ops)
                if self.volume_norm:
                    t_audio = normalize_volume(t_audio)
                    s_audio = normalize_volume(s_audio)

                target_audio_list.append(torch.from_numpy(t_audio))
                source_audio_list.append(torch.from_numpy(s_audio))

            # 3. Re-pad the newly stitched arrays to the batch's new maximum length
            target_audio = collate_vectors(target_audio_list, padding_value=0.0)
            target_audio_lens = torch.tensor([len(a) for a in target_audio_list], dtype=torch.long)
            
            source_audio = collate_vectors(source_audio_list, padding_value=0.0)
            source_audio_lens = torch.tensor([len(a) for a in source_audio_list], dtype=torch.long)
        
        target_text_tokens, target_token_lens = collate_token_channel(
            cuts, self.text_tokenizer, self.frame_length, roles=self.output_roles,
            add_text_bos=self.add_text_bos, tokenizer_names=batch_tokenizer_names,
            pad_id=self.pad_id, eos_id=self.eos_id, bos_id=self.bos_id, interruption_token_id=self.interruption_token_id, remove_user_turn_flags=remove_user_turn_flags
        )
        source_tokens, source_token_lens = collate_token_channel(
            cuts, self.text_tokenizer, self.frame_length, roles=self.input_roles,
            add_text_bos=self.add_text_bos, tokenizer_names=batch_tokenizer_names,
            pad_id=self.pad_id, eos_id=self.eos_id, bos_id=self.bos_id, interruption_token_id=self.interruption_token_id, remove_user_turn_flags=remove_user_turn_flags
        )

        if self.phoneme_tokenizer is not None:
            target_phoneme_tokens, target_phoneme_lens = collate_phoneme_channel(
                cuts, self.phoneme_tokenizer, self.frame_length, roles=self.output_roles,
                ignore_phoneme_languages=self.ignore_phoneme_languages, pad_id=self.phoneme_tokenizer.pad, eos_id=self.phoneme_tokenizer.eos_token_id, bos_id=self.phoneme_tokenizer.bos_token_id,
                remove_user_turn_flags=remove_user_turn_flags,
            )
        else:
            target_phoneme_tokens, target_phoneme_lens = None, None

        dataset_name_list = []
        audio_list_16khz = []
        audio_len_list_16khz = []
        prior_list = []
        
        target_codes_list = []
        source_codes_list = []
        
        context_audio_list = []
        context_audio_len_list = []
        context_audio_codes_list = []
        context_audio_codes_len_list = []
        context_text_tokens_list = []
        context_text_tokens_len_list = []
        context_has_text_context_list = []
        reward_list = []
        language_list = []

        def _sample_context_duration_with_available_limit(available_duration_sec: float) -> float:
            effective_duration_max = min(self.context_duration_max, available_duration_sec)
            effective_duration_max = max(self.context_duration_min, effective_duration_max)
            return random.uniform(self.context_duration_min, effective_duration_max)

        for i, cut in enumerate(cuts):
            remove_user_turn_this_cut = remove_user_turn_flags[i]
            speaker_found = False
            for sup in reversed(cut.supervisions):
                if check_speaker_format(sup.speaker):
                    dataset_name = sup.speaker.strip().split()[2].split(":")[-1]
                    speaker_found = True
                    break

            if not speaker_found:
                dataset_name = "unknown"
            dataset_name_list.append(dataset_name)
            # print("Language is available?", cut.has_custom("lang"), " Has codes?", cut.has_custom("target_codes"), "Has context audio?", cut.has_custom("context_audio"), "Has context codes?", cut.has_custom("context_codes"))

            language = cut.lang if cut.has_custom("lang") else next((sup.language for sup in reversed(cut.supervisions) if sup.has_custom("language")), "en")
            language_list.append(language)

            # Target and Source Codes
            if self.load_cached_codes_if_available:
                if cut.has_custom("target_codes"):
                    codes_array = cut.target_codes.load().astype(np.int32)
                    if remove_user_turn_this_cut:
                        raise RuntimeError("Remove user turn augmentation is not implemented for cached codes!")
                    target_codes_list.append(torch.from_numpy(codes_array).T)

                if cut.has_custom("source_codes"):
                    source_codes_list.append(torch.from_numpy(cut.source_codes.load().astype(np.int32)).T)
                    if remove_user_turn_this_cut:
                        raise RuntimeError("Remove user turn augmentation is not implemented for cached codes!")

            # Context Audio or Context Codes
            if self.load_cached_codes_if_available and cut.has_custom("context_codes"):
                context_audio_codes_array = cut.context_codes.load().astype(np.int32)
                context_audio_codes = torch.from_numpy(context_audio_codes_array)
                _available_context_duration = (context_audio_codes.shape[1] * self.codec_model_samples_per_frame / self.sample_rate)
                _context_duration_to_slice = _sample_context_duration_with_available_limit(_available_context_duration)
                _num_frames_to_slice = int(_context_duration_to_slice * self.sample_rate / self.codec_model_samples_per_frame)
                
                if _num_frames_to_slice < context_audio_codes.shape[1]:
                    start_idx = random.randint(0, context_audio_codes.shape[1] - _num_frames_to_slice)
                    context_audio_codes = context_audio_codes[:, start_idx : start_idx + _num_frames_to_slice]
                else:
                    _num_repeats = int(np.ceil(_num_frames_to_slice / context_audio_codes.shape[1]))
                    context_audio_codes = context_audio_codes.repeat(1, _num_repeats)[:, :_num_frames_to_slice]

                context_audio_codes = _align_codebooks(context_audio_codes.T)
                context_audio_codes_list.append(context_audio_codes)
                context_audio_codes_len_list.append(context_audio_codes.shape[0])
                
            elif cut.has_custom("context_audio"):
                with fp32_precision():
                    context_audio_array = cut.context_audio.resample(self.sample_rate).load_audio().squeeze(0)
                if self.volume_norm:
                    context_audio_array = normalize_volume(context_audio_array)

                _available_context_duration = len(context_audio_array) / self.sample_rate
                _context_duration_to_slice = _sample_context_duration_with_available_limit(_available_context_duration)
                _num_samples_to_slice = self.get_num_audio_samples_to_slice(_context_duration_to_slice, self.sample_rate)

                if _num_samples_to_slice < len(context_audio_array):
                    start_idx = random.randint(0, len(context_audio_array) - _num_samples_to_slice)
                    context_audio_array = context_audio_array[start_idx : start_idx + _num_samples_to_slice]
                else:
                    _num_repeats = int(np.ceil(_num_samples_to_slice / len(context_audio_array)))
                    context_audio_array = np.tile(context_audio_array, _num_repeats)[:_num_samples_to_slice]
                    
                context_audio = torch.from_numpy(context_audio_array)
                context_audio_list.append(context_audio)
                context_audio_len_list.append(context_audio.shape[0])

            else:
                matching_supervisions = [s for s in cut.supervisions if s.speaker in self.output_roles]
                
                if self.load_cached_codes_if_available:
                    if len(matching_supervisions) > 0 and cut.has_custom("target_codes"):
                        sup = random.choice(matching_supervisions)
                        codes_array = cut.target_codes.load().astype(np.int32)
                        start_frame = int(max(0, sup.start) * self.sample_rate / self.codec_model_samples_per_frame)
                        num_frames = int(sup.duration * self.sample_rate / self.codec_model_samples_per_frame)
                        context_audio_codes = torch.from_numpy(codes_array)[:, start_frame : start_frame + num_frames].T
                        context_audio_codes = _align_codebooks(context_audio_codes)
                    else:
                        context_audio_codes = torch.zeros([0, self.num_audio_codebooks], dtype=torch.int32)
                    context_audio_codes_list.append(context_audio_codes)
                    context_audio_codes_len_list.append(context_audio_codes.shape[0])
                else:
                    if len(matching_supervisions) > 0:
                        sup = random.choice(matching_supervisions)
                        with fp32_precision():
                            turn_cut = cut.resample(self.sample_rate, recording_field="target_audio").truncate(offset=max(0, sup.start), duration=sup.duration)
                            context_audio_array = turn_cut.load_custom("target_audio").squeeze(0)
                        if self.volume_norm:
                            context_audio_array = normalize_volume(context_audio_array)
                        context_audio = torch.from_numpy(context_audio_array)
                    else:
                        context_audio = torch.zeros(self.codec_model_samples_per_frame, dtype=torch.float32)
                    
                    context_audio_list.append(context_audio)
                    context_audio_len_list.append(context_audio.shape[0])

            # 16khz audio for SV
            if self.load_16khz_audio:
                with fp32_precision():
                    if cut.has_custom("context_audio"):
                        audio_array_16khz = cut.context_audio.resample(16_000).load_audio().squeeze(0)
                        if self.volume_norm:
                            audio_array_16khz = normalize_volume(audio_array_16khz)
                            
                        _available_context_duration = len(audio_array_16khz) / 16_000
                        _context_duration_to_slice = _sample_context_duration_with_available_limit(_available_context_duration)
                        _num_samples_to_slice = int(_context_duration_to_slice * 16_000)
                        if _num_samples_to_slice < len(audio_array_16khz):
                            start_idx = random.randint(0, len(audio_array_16khz) - _num_samples_to_slice)
                            audio_array_16khz = audio_array_16khz[start_idx : start_idx + _num_samples_to_slice]
                    else:
                        matching_supervisions = [s for s in cut.supervisions if s.speaker in self.output_roles]
                        if len(matching_supervisions) > 0:
                            sup = random.choice(matching_supervisions)
                            turn_cut = cut.resample(16_000, recording_field="target_audio").truncate(offset=max(0, sup.start), duration=sup.duration)
                            audio_array_16khz = turn_cut.load_custom("target_audio").squeeze(0)
                        else:
                            audio_array_16khz = np.zeros(16000, dtype=np.float32)

                        if self.volume_norm:
                            audio_array_16khz = normalize_volume(audio_array_16khz)

                audio_16khz = torch.from_numpy(audio_array_16khz)
                audio_list_16khz.append(audio_16khz)
                audio_len_list_16khz.append(audio_16khz.shape[0])

            # Context Text
            if self.use_text_conditioning_tokenizer:
                context_text = next((sup.context_text for sup in cut.supervisions if sup.has_custom("context_text")), None)
                if context_text is not None:
                    if self.text_context_remapping is not None and context_text in self.text_context_remapping:
                        if self.dataset_type == 'train' and random.random() < self.text_context_remapping_prob:
                            context_text = self.text_context_remapping[context_text]
                    context_text_tokens = self.text_tokenizer.encode(context_text, tokenizer_name=self.text_conditioning_tokenizer_name)
                    has_text_context = True
                else:
                    context_text = f"[{language.upper()}]" if self.add_language_to_context_text else "[NO TEXT CONTEXT]"
                    context_text_tokens = self.text_tokenizer.encode(context_text, tokenizer_name=self.text_conditioning_tokenizer_name)
                    has_text_context = False
                    
                if self.pad_context_text_to_max_duration:
                    _required_len = int(self.context_duration_max * self.sample_rate / self.codec_model_samples_per_frame) + 2
                    if len(context_text_tokens) < _required_len:
                        _pad_id = self.text_tokenizer.tokenizer_pad_ids[self.text_conditioning_tokenizer_name]
                        context_text_tokens += [_pad_id] * (_required_len - len(context_text_tokens))
                    else:
                        context_text_tokens = context_text_tokens[:_required_len]

                context_text_tokens = torch.tensor(context_text_tokens, dtype=torch.int32)
                context_text_tokens_list.append(context_text_tokens)
                context_text_tokens_len_list.append(context_text_tokens.shape[0])
                context_has_text_context_list.append(has_text_context)

            # Align Prior (Note: Using full target length to preserve shape compatibility)
            if self.include_align_prior:
                tok_name = batch_tokenizer_names[i]
                full_text_len = sum([len(self.text_tokenizer.encode(sup.text, tokenizer_name=tok_name)) for sup in cut.supervisions if sup.speaker in self.output_roles])

                if self.add_text_bos:
                    full_text_len += 2 * sum([1 for sup in cut.supervisions if sup.speaker in self.output_roles])
                else:
                    # cont eos token
                    full_text_len += sum([1 for sup in cut.supervisions if sup.speaker in self.output_roles])

                full_text_len = max(1, full_text_len)

                if self.load_cached_codes_if_available and cut.has_custom("target_codes"):
                    spec_len = int(target_codes_list[-1].shape[0]) + 1
                else:
                    spec_len = int(target_audio_lens[i] / self.codec_model_samples_per_frame) + 2 # +1 extra in case it was truncated

                align_prior = beta_binomial_prior_distribution(phoneme_count=full_text_len, mel_count=spec_len, scaling_factor=self.prior_scaling_factor)
                prior_list.append(torch.tensor(align_prior, dtype=torch.float32))

            reward = next((sup.reward for sup in reversed(cut.supervisions) if sup.has_custom("reward")), None)
            if reward is not None:
                reward_list.append(reward)

        batch_dict = {
            "sample_id": [str(cut.id) for cut in cuts],
            "dataset_names": dataset_name_list,
            "languages": language_list,
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "audio": target_audio,
            "audio_lens": target_audio_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "text": target_text_tokens,
            "text_lens": target_token_lens,
            "raw_texts": [" ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts],
            "task": [getattr(cut, "task", "tts") for cut in cuts],
        }

        if target_codes_list:
            batch_dict["audio_codes"] = collate_matrices(target_codes_list, padding_value=0).transpose(1, 2)
            batch_dict["audio_codes_lens"] = torch.IntTensor([c.shape[0] for c in target_codes_list])
            
        if source_codes_list:
            batch_dict["source_codes"] = collate_matrices(source_codes_list, padding_value=0).transpose(1, 2)
            batch_dict["source_codes_lens"] = torch.IntTensor([c.shape[0] for c in source_codes_list])

        if self.phoneme_tokenizer is not None:
            batch_dict["phoneme_tokens"] = target_phoneme_tokens
            batch_dict["phoneme_tokens_lens"] = target_phoneme_lens

        if len(audio_list_16khz) > 0:
            batch_dict["audio_16khz"] = collate_vectors(audio_list_16khz, padding_value=0.0)
            batch_dict["audio_lens_16khz"] = torch.IntTensor(audio_len_list_16khz)

        if len(context_audio_list) > 0:
            batch_dict["context_audio"] = collate_vectors(context_audio_list, padding_value=0.0)
            batch_dict["context_audio_lens"] = torch.IntTensor(context_audio_len_list)
            
        if len(context_audio_codes_list) > 0:
            batch_dict["context_audio_codes"] = collate_matrices(context_audio_codes_list, padding_value=0).transpose(1, 2)
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

        agent_mask, agent_mask_lens = collate_speaker_mask_channel(
            cuts,
            self.frame_length,
            self.output_roles,
        )

        batch_dict["agent_mask"] = agent_mask
        batch_dict["agent_mask_lens"] = agent_mask
        return batch_dict


def collate_token_channel(
    cuts: CutSet,
    tokenizer,
    frame_length: Seconds,
    roles: set[str],
    add_text_bos: bool = True,
    tokenizer_names: list[str] = None,
    pad_id: int = None,
    eos_id: int = None,
    bos_id: int = None,
    interruption_token_id: int = None,
    remove_user_turn_flags: list[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and collate token channels aligned to the audio frame grid."""
    tokens = []

    for i, c in enumerate(cuts):
        tok_name = tokenizer_names[i] if tokenizer_names else "english_phoneme"
        flag = remove_user_turn_flags[i] if remove_user_turn_flags else False
        tokens.append(
            build_token_channel(
                c, tokenizer, frame_length, roles, pad_id, eos_id, bos_id, interruption_token_id,
                add_text_bos, tok_name, remove_user_turns=flag,
            )
        )
    token_lens = torch.tensor([len(tt) for tt in tokens])
    return collate_vectors(tokens, padding_value=pad_id), token_lens


def build_speaker_mask_channel(
    cut: Cut,
    frame_length: Seconds,
    output_roles: set[str],
) -> torch.Tensor:
    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    mask = torch.zeros(total, dtype=torch.float32)

    for supervision in cut.supervisions:
        start = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
        end = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)

        if supervision.speaker in output_roles:
            mask[start:end] = 1.0

    return mask

def collate_speaker_mask_channel(
    cuts: CutSet,
    frame_length: Seconds,
    output_roles: set[str],
):
    masks = [
        build_speaker_mask_channel(cut, frame_length, output_roles)
        for cut in cuts
    ]
    mask_lens = torch.tensor([len(m) for m in masks])
    return collate_vectors(masks, padding_value=0.0), mask_lens


def build_token_channel(
    cut: Cut,
    tokenizer,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
    eos_id: int = -2,
    bos_id: int = -3,
    interruption_token_id: int = -4,
    add_text_bos: bool = True,
    tokenizer_name: str = "english_phoneme",
    remove_user_turns: bool = False,
) -> torch.Tensor:
    if remove_user_turns:
        turn_chunks = []
        for supervision in cut.supervisions:
            if supervision.speaker in roles:
                # 1. Get exact frame length of THIS turn
                start_f = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
                end_f = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
                turn_frames = max(0, end_f - start_f)
                
                turn_tokens = torch.ones(turn_frames, dtype=torch.long) * pad_id
                
                if turn_frames == 0:
                    continue

                # 2. Encode text
                text = supervision.text
                if hasattr(tokenizer, "encode"):
                    try:
                        raw_ids = tokenizer.encode(text=text, tokenizer_name=tokenizer_name)
                    except TypeError:
                        raw_ids = tokenizer.encode(text)
                else:
                    raw_ids = tokenizer.text_to_ids(text)

                if add_text_bos:
                    text_ids = [bos_id] + raw_ids + [eos_id]
                else:
                    text_ids = raw_ids + [eos_id]
                
                # 3. Place text at the start, keeping the rest as pad_id
                text_len = len(text_ids)
                if text_len > turn_frames:
                    text_ids = text_ids[:turn_frames]
                    text_len = turn_frames
                
                turn_tokens[0:text_len] = torch.as_tensor(text_ids, dtype=torch.long)
                
                # 4. Place interruption token at the exact end of the turn
                turn_tokens[-1] = interruption_token_id
                turn_chunks.append(turn_tokens)

        if turn_chunks:
            return torch.cat(turn_chunks, dim=0)
        else:
            return torch.tensor([pad_id], dtype=torch.long)

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            text = supervision.text
            
            if hasattr(tokenizer, "encode"):
                try:
                    raw_ids = tokenizer.encode(text=text, tokenizer_name=tokenizer_name)
                except TypeError:
                    raw_ids = tokenizer.encode(text)
            else:
                raw_ids = tokenizer.text_to_ids(text)

            if add_text_bos:
                text_ids = torch.as_tensor([bos_id] + raw_ids + [eos_id])
            else:
                text_ids = torch.as_tensor(raw_ids + [eos_id])

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):
                continue

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                text_ids = text_ids[:len(tokens) - pos]
            tokens[pos:pos+len(text_ids)] = text_ids

            # add interruption token, used for add speech eos and interrupt the model
            interruption_pos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            if interruption_pos < len(tokens):
                tokens[interruption_pos] = interruption_token_id

    return tokens


def collate_phoneme_channel(
    cuts: CutSet,
    phoneme_tokenizer,
    frame_length: Seconds,
    roles: set[str],
    ignore_phoneme_languages: list[str],
    pad_id: int = -1,
    eos_id: int = -2,
    bos_id: int = -3,
    remove_user_turn_flags: list[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = []
    for i, c in enumerate(cuts):
        flag = remove_user_turn_flags[i] if remove_user_turn_flags else False
        tokens.append(
            build_phoneme_channel(
                c, phoneme_tokenizer, frame_length, roles,
                ignore_phoneme_languages, pad_id, eos_id, bos_id, remove_user_turns=flag
            )
        )
    token_lens = torch.tensor([len(tt) for tt in tokens])
    return collate_vectors(tokens, padding_value=pad_id), token_lens


def build_phoneme_channel(
    cut: Cut,
    phoneme_tokenizer,
    frame_length: Seconds,
    roles: set[str],
    ignore_phoneme_languages: list[str],
    pad_id: int = -1,
    eos_id: int = -2,
    bos_id: int = -3,
    remove_user_turns: bool = False,
) -> torch.Tensor:
    language = cut.lang if cut.has_custom("lang") else next((sup.language for sup in cut.supervisions if sup.has_custom("language")), "en")

    if remove_user_turns:
        turn_chunks = []
        for supervision in cut.supervisions:
            if supervision.speaker in roles:
                start_f = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
                end_f = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
                turn_frames = max(0, end_f - start_f)
                
                turn_tokens = torch.ones(turn_frames, dtype=torch.long) * pad_id
                
                if turn_frames == 0:
                    continue

                if isinstance(phoneme_tokenizer, IPABPETokenizer):
                    ipa_text = _get_supervision_ipa_text(supervision)
                    if language in ignore_phoneme_languages:
                        ipa_text = ""
                else:
                    ipa_text = supervision.text

                phoneme_ids = phoneme_tokenizer.encode(ipa_text)
                phoneme_ids = [bos_id] + phoneme_ids + [eos_id]
                
                text_len = len(phoneme_ids)
                if text_len > turn_frames:
                    phoneme_ids = phoneme_ids[:turn_frames]
                    text_len = turn_frames
                
                turn_tokens[0:text_len] = torch.as_tensor(phoneme_ids, dtype=torch.long)
                turn_chunks.append(turn_tokens)

        if turn_chunks:
            return torch.cat(turn_chunks, dim=0)
        else:
            return torch.tensor([pad_id], dtype=torch.long)

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            if isinstance(phoneme_tokenizer, IPABPETokenizer):
                ipa_text = _get_supervision_ipa_text(supervision)
                if language in ignore_phoneme_languages:
                    ipa_text = ""
            else:
                ipa_text = supervision.text

            phoneme_ids = phoneme_tokenizer.encode(ipa_text)
            phoneme_ids = [bos_id] + phoneme_ids + [eos_id]
            
            phoneme_ids = torch.as_tensor(phoneme_ids, dtype=torch.long)
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):
                continue

            endpos = pos + len(phoneme_ids)
            if endpos > len(tokens):
                phoneme_ids = phoneme_ids[:len(tokens) - pos]
            tokens[pos:pos+len(phoneme_ids)] = phoneme_ids

    return tokens
