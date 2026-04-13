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
import random
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors, collate_matrices
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.utils import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import IPABPETokenizer

class MagpieTTSLhotseMultiturnDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

        p_drop_description (float, optional):
            Probability of dropping text descriptions. Default: `0.0`.

        add_text_bos_and_eos_in_each_turn (bool, optional):
            If True, each conversational turn from any speaker is explicitly delimited
            with BOS and EOS tokens in the text stream.
            Default: `True`.

     Returns:
        A dictionary with the following keys:
            - sample_id: List of sample IDs for each cut in the batch [B]

            - non_prompt_mask: Bool tensor [B, T] marking positions that are not part of the prompt
            - prompt_lens: Tensor of description + audio prompt lengths [B]

            - aligned_attention_mask: Bool tensor [B, T] used by alignment-aware transformer models
            - aligned_position_ids: Tensor of position indices aligned to audio frames [B, T]

            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]

            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]

            - target_text_tokens: Tensor of frame-aligned input text tokens [B, T],
                including BOS/EOS/PAD when enabled
            - target_token_lens: Tensor of target token sequence lengths [B]

            - source_tokens: Tensor of frame-aligned source text tokens [B, T],
                including BOS/EOS/PAD
            - source_token_lens: Tensor of source token sequence lengths [B]

            - target_texts: List of full target texts joined from output_roles supervisions [B]

            - audio_prompt: Tensor of optional speaker reference waveform samples [B, T]
            - audio_prompt_lens: Tensor of speaker reference audio lengths [B]

            - task: List indicating the task to use for each cut (default "s2s_duplex") [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
    """

    def __init__(
        self,
        tokenizer,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        p_drop_description: float = 0.0,
        add_text_bos_and_eos_in_each_turn: bool = False,
        add_audio_prompt: bool = False,
        audio_prompt_duration: float = 3.0,
        num_delay_speech_tokens: int = 0,
        add_system_prompt: bool = False,
        ignore_data_system_prompt: bool = True,
        phoneme_tokenizer_config: DictConfig = None,
        ignore_phoneme_languages: list[str] = None,
        load_cached_codes_if_available: bool = False,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.p_drop_description = p_drop_description
        self.add_text_bos_and_eos_in_each_turn = add_text_bos_and_eos_in_each_turn
        self.add_audio_prompt = add_audio_prompt
        self.audio_prompt_duration = audio_prompt_duration
        self.num_delay_speech_tokens = num_delay_speech_tokens
        self.add_system_prompt = add_system_prompt
        self.ignore_data_system_prompt = ignore_data_system_prompt

        self.phoneme_tokenizer_config = phoneme_tokenizer_config
        self.ignore_phoneme_languages = ignore_phoneme_languages or []
        self.phoneme_tokenizer = None
        self.load_cached_codes_if_available = load_cached_codes_if_available

        self.source_samples_per_frame = int(self.source_sample_rate * self.frame_length)
        self.target_samples_per_frame = int(self.target_sample_rate * self.frame_length)

        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def __getitem__(self, cuts: CutSet) -> dict:
        if self.phoneme_tokenizer is None and getattr(self, "phoneme_tokenizer_config", None) is not None:
            self.phoneme_tokenizer = instantiate(self.phoneme_tokenizer_config)

        cuts = cuts.transform_text(_strip_timestamps)
        
        batch_tokenizer_names = []
        for cut in cuts:
            if cut.has_custom("tokenizer_names"):
                batch_tokenizer_names.append(random.choice(cut.tokenizer_names))
            else:
                batch_tokenizer_names.append("english_phoneme")
        
        target_codes_list = []
        source_codes_list = []
        if self.load_cached_codes_if_available:
            for cut in cuts:
                if cut.has_custom("target_codes"):
                    codes_array = cut.target_codes.load().astype(np.int32)
                    target_codes_list.append(torch.from_numpy(codes_array).T) 
                if cut.has_custom("source_codes"):
                    codes_array = cut.source_codes.load().astype(np.int32)
                    source_codes_list.append(torch.from_numpy(codes_array).T)

        if target_codes_list:
            target_codes = collate_matrices(target_codes_list, padding_value=0).transpose(1, 2)
            target_codes_lens = torch.tensor([c.shape[0] for c in target_codes_list], dtype=torch.int32)
        else:
            target_codes, target_codes_lens = None, None

        if source_codes_list:
            source_codes = collate_matrices(source_codes_list, padding_value=0).transpose(1, 2)
            source_codes_lens = torch.tensor([c.shape[0] for c in source_codes_list], dtype=torch.int32)
        else:
            source_codes, source_codes_lens = None, None

        with fp32_precision():
            source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
            target_audio, target_audio_lens = collate_audio(
                cuts.resample(self.target_sample_rate, recording_field="target_audio"), recording_field="target_audio"
            )

        target_text_tokens, target_token_lens = collate_token_channel(
            cuts,
            self.tokenizer,
            self.frame_length,
            roles=self.output_roles,
            add_text_bos_and_eos_in_each_turn=self.add_text_bos_and_eos_in_each_turn,
            tokenizer_names=batch_tokenizer_names,
        )
        source_tokens, source_token_lens = collate_token_channel(
            cuts,
            self.tokenizer,
            self.frame_length,
            roles=self.input_roles,
            add_text_bos_and_eos_in_each_turn=self.add_text_bos_and_eos_in_each_turn,
            tokenizer_names=batch_tokenizer_names,
        )

        if self.phoneme_tokenizer is not None:
            target_phoneme_tokens, target_phoneme_lens = collate_phoneme_channel(
                cuts,
                self.phoneme_tokenizer,
                self.frame_length,
                roles=self.output_roles,
                ignore_phoneme_languages=self.ignore_phoneme_languages,
                add_text_bos_and_eos_in_each_turn=self.add_text_bos_and_eos_in_each_turn,
            )
        else:
            target_phoneme_tokens, target_phoneme_lens = None, None

        with fp32_precision():
            audio_prompt, audio_prompt_lens = get_audio_prompt(
                cuts, self.target_sample_rate, roles=self.output_roles, recording_field="target_audio"
            )

        if self.num_delay_speech_tokens:
            (
                source_audio, 
                source_audio_lens, 
                target_audio, 
                target_audio_lens,
                source_codes,
                source_codes_lens,
                target_codes,
                target_codes_lens
            ) = add_speech_delay(
                source_audio,
                source_audio_lens,
                target_audio,
                target_audio_lens,
                self.num_delay_speech_tokens,
                self.target_samples_per_frame,
                self.source_samples_per_frame,
                source_codes=source_codes,
                source_codes_lens=source_codes_lens,
                target_codes=target_codes,
                target_codes_lens=target_codes_lens,
            )

        if self.add_system_prompt:
            with fp32_precision():
                system_prompts, system_prompts_lens, system_prompts_raw = collate_system_prompt(
                    cuts, 
                    self.tokenizer, 
                    ignore_data_system_prompt=self.ignore_data_system_prompt,
                    tokenizer_names=batch_tokenizer_names,
                )
        else:
            system_prompts = None
            system_prompts_lens = None
            system_prompts_raw = None

        dataset_type = [getattr(c, "type", "") for c in cuts]

        (
            target_text_tokens,
            target_token_lens,
            source_tokens,
            source_token_lens,
            source_audio,
            source_audio_lens,
            target_audio,
            target_audio_lens,
            prompt_lens,
            target_phoneme_tokens,   
            target_phoneme_lens,
            source_codes,
            source_codes_lens,
            target_codes,
            target_codes_lens,
        ) = self.maybe_add_audio_prompt(
            target_text_tokens, target_token_lens, source_tokens, source_token_lens,
            target_audio, target_audio_lens, source_audio, source_audio_lens,
            audio_prompt, audio_prompt_lens, system_prompts, system_prompts_lens,
            target_phoneme_tokens=target_phoneme_tokens, target_phoneme_lens=target_phoneme_lens,
            source_codes=source_codes, source_codes_lens=source_codes_lens,
            target_codes=target_codes, target_codes_lens=target_codes_lens,
        )

        non_prompt_mask = get_mask_from_lengths(target_token_lens)
        for i, frame in enumerate(prompt_lens):
            non_prompt_mask[i, : frame - 1] = 0.0

        max_len = max(target_token_lens)
        aligned_segment_ids = torch.stack(
            [torch.nn.functional.pad(torch.full((seq_len,), i), (0, max_len - seq_len), value=-1) for i, seq_len in enumerate(target_token_lens)], dim=0,
        )
        aligned_attention_mask = (aligned_segment_ids.unsqueeze(-2) == aligned_segment_ids.unsqueeze(-1)) & (
            torch.arange(max_len).unsqueeze(0).unsqueeze(1) <= torch.arange(max_len).unsqueeze(0).unsqueeze(-1)
        )
        aligned_attention_mask = aligned_attention_mask.unsqueeze(1)
        aligned_position_ids = torch.stack(
            [torch.nn.functional.pad(torch.arange(seq_len), (0, max(target_token_lens) - seq_len), value=0) for seq_len in target_token_lens], dim=0,
        )

        batch_dict = {
            "sample_id": [str(cut.id) for cut in cuts],
            "non_prompt_mask": non_prompt_mask.bool(),
            "prompt_lens": prompt_lens,
            "aligned_attention_mask": aligned_attention_mask.bool(),
            "aligned_position_ids": aligned_position_ids,
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_text_tokens": target_text_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "audio_prompt": audio_prompt,
            "audio_prompt_lens": audio_prompt_lens,
            "system_prompts_raw": system_prompts_raw,
            "dataset_type": dataset_type,
            "phoneme_tokens": target_phoneme_tokens,
            "phoneme_tokens_lens": target_phoneme_lens,
            "task": [getattr(cut, "task", "s2s_duplex") for cut in cuts],
        }

        if target_codes is not None:
            batch_dict["target_codes"] = target_codes
            batch_dict["target_codes_lens"] = target_codes_lens
        if source_codes is not None:
            batch_dict["source_codes"] = source_codes
            batch_dict["source_codes_lens"] = source_codes_lens

        return batch_dict

    def maybe_add_audio_prompt(
        self,
        target_text_tokens: torch.Tensor,
        target_token_lens: torch.Tensor,
        source_tokens: torch.Tensor,
        source_token_lens: torch.Tensor,
        target_audio: torch.Tensor,
        target_audio_lens: torch.Tensor,
        source_audio: torch.Tensor,
        source_audio_lens: torch.Tensor,
        audio_prompt: torch.Tensor,
        audio_prompt_lens: torch.Tensor,
        system_prompts: torch.Tensor = None,
        system_prompts_lens: torch.Tensor = None,
        target_phoneme_tokens: torch.Tensor = None,
        target_phoneme_lens: torch.Tensor = None,
        source_codes: torch.Tensor = None,
        source_codes_lens: torch.Tensor = None,
        target_codes: torch.Tensor = None,
        target_codes_lens: torch.Tensor = None,
    ):
        text_pad_id = get_pad_id(self.tokenizer)

        target_text_tokens_ = []
        source_tokens_ = []
        source_audio_ = []
        target_audio_ = []
        prompt_lens = []

        target_phoneme_tokens_ = []
        phoneme_pad_id = self.phoneme_tokenizer.pad if self.phoneme_tokenizer else -1

        source_codes_ = []
        target_codes_ = []

        for i in range(target_text_tokens.size(0)):
            if system_prompts is not None:
                text_prompt = system_prompts[i][: system_prompts_lens[i]]
            else:
                text_prompt = torch.tensor(
                    [self.tokenizer.eos],
                    dtype=torch.long,
                    device=target_text_tokens.device,
                )

            if self.add_audio_prompt:
                prompt_audio_size = int(
                    ((self.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame)
                    * self.target_samples_per_frame
                )

                prompt_audio = sample_audio_segments_repeat(
                    audio_prompt, audio_prompt_lens, prompt_audio_size, sample=True
                )

                prompt_audio[:, -int(self.target_samples_per_frame * 2) :] = 0

                prompt_audio_text_pad_size = prompt_audio_size // self.target_samples_per_frame
                prompt_audio_text_pad = (
                    torch.ones(prompt_audio_text_pad_size, device=target_text_tokens.device, dtype=target_text_tokens.dtype)
                    * text_pad_id
                )
                prompt_audio_text_pad[-1] = self.tokenizer.eos

                new_target_text_tokens = torch.cat(
                    [text_prompt.to(target_text_tokens.dtype), prompt_audio_text_pad, target_text_tokens[i]]
                )
                target_text_tokens_.append(new_target_text_tokens)
                target_token_lens[i] += len(text_prompt) + prompt_audio_text_pad_size

                new_source_tokens = torch.cat([text_prompt, prompt_audio_text_pad, source_tokens[i]])
                source_tokens_.append(new_source_tokens)
                source_token_lens[i] += len(text_prompt) + prompt_audio_text_pad_size

                if target_phoneme_tokens is not None:
                    phoneme_pad_size = len(text_prompt) + prompt_audio_text_pad_size
                    phoneme_pad = torch.full((phoneme_pad_size,), phoneme_pad_id, device=target_phoneme_tokens.device, dtype=target_phoneme_tokens.dtype)
                    target_phoneme_tokens_.append(torch.cat([phoneme_pad, target_phoneme_tokens[i]]))
                    target_phoneme_lens[i] += phoneme_pad_size

                code_pad_size = len(text_prompt) + prompt_audio_text_pad_size
                if target_codes is not None:
                    pad_codes = torch.zeros((target_codes.size(1), code_pad_size), device=target_codes.device, dtype=target_codes.dtype)
                    target_codes_.append(torch.cat([pad_codes, target_codes[i]], dim=1))
                    target_codes_lens[i] += code_pad_size
                
                if source_codes is not None:
                    pad_codes = torch.zeros((source_codes.size(1), code_pad_size), device=source_codes.device, dtype=source_codes.dtype)
                    source_codes_.append(torch.cat([pad_codes, source_codes[i]], dim=1))
                    source_codes_lens[i] += code_pad_size

                pad_size_src = (len(text_prompt) * self.source_samples_per_frame) + prompt_audio.size(1)
                pad_audio_src = torch.zeros(pad_size_src, device=source_audio.device, dtype=source_audio.dtype)
                source_audio_.append(torch.cat([pad_audio_src, source_audio[i]]))
                source_audio_lens[i] += pad_size_src

                pad_size_tgt = len(text_prompt) * self.target_samples_per_frame
                pad_audio_tgt = torch.zeros(pad_size_tgt, device=target_audio.device, dtype=target_audio.dtype)
                target_audio_.append(torch.cat([pad_audio_tgt, prompt_audio[i], target_audio[i]]))
                target_audio_lens[i] += pad_size_tgt + prompt_audio.size(1)

                prompt_lens.append(len(text_prompt) + prompt_audio_text_pad_size - 1)

            else:
                target_text_tokens_.append(torch.cat([text_prompt, target_text_tokens[i]]))
                target_token_lens[i] += len(text_prompt)

                source_tokens_.append(torch.cat([text_prompt, source_tokens[i]]))
                source_token_lens[i] += len(text_prompt)

                if target_phoneme_tokens is not None:
                    phoneme_pad_size = len(text_prompt)
                    phoneme_pad = torch.full((phoneme_pad_size,), phoneme_pad_id, device=target_phoneme_tokens.device, dtype=target_phoneme_tokens.dtype)
                    target_phoneme_tokens_.append(torch.cat([phoneme_pad, target_phoneme_tokens[i]]))
                    target_phoneme_lens[i] += phoneme_pad_size

                code_pad_size = len(text_prompt)
                if target_codes is not None:
                    pad_codes = torch.zeros((target_codes.size(1), code_pad_size), device=target_codes.device, dtype=target_codes.dtype)
                    target_codes_.append(torch.cat([pad_codes, target_codes[i]], dim=1))
                    target_codes_lens[i] += code_pad_size
                
                if source_codes is not None:
                    pad_codes = torch.zeros((source_codes.size(1), code_pad_size), device=source_codes.device, dtype=source_codes.dtype)
                    source_codes_.append(torch.cat([pad_codes, source_codes[i]], dim=1))
                    source_codes_lens[i] += code_pad_size

                pad_size_src = len(text_prompt) * self.source_samples_per_frame
                pad_audio_src = torch.zeros(pad_size_src, device=source_audio.device, dtype=source_audio.dtype)
                source_audio_.append(torch.cat([pad_audio_src, source_audio[i]]))
                source_audio_lens[i] += pad_size_src

                pad_size_tgt = len(text_prompt) * self.target_samples_per_frame
                pad_audio_tgt = torch.zeros(pad_size_tgt, device=target_audio.device, dtype=target_audio.dtype)
                target_audio_.append(torch.cat([pad_audio_tgt, target_audio[i]]))
                target_audio_lens[i] += pad_size_tgt

                prompt_lens.append(len(text_prompt))

        target_text_tokens = collate_vectors(target_text_tokens_, padding_value=text_pad_id)
        source_tokens = collate_vectors(source_tokens_, padding_value=text_pad_id)
        source_audio = collate_vectors(source_audio_, padding_value=0)
        target_audio = collate_vectors(target_audio_, padding_value=0)

        if target_phoneme_tokens is not None:
            target_phoneme_tokens = collate_vectors(target_phoneme_tokens_, padding_value=phoneme_pad_id)

        if target_codes is not None:
            max_len = max([c.size(1) for c in target_codes_])
            target_codes = torch.stack([F.pad(c, (0, max_len - c.size(1))) for c in target_codes_])
        if source_codes is not None:
            max_len = max([c.size(1) for c in source_codes_])
            source_codes = torch.stack([F.pad(c, (0, max_len - c.size(1))) for c in source_codes_])

        return (
            target_text_tokens,
            target_token_lens,
            source_tokens,
            source_token_lens,
            source_audio,
            source_audio_lens,
            target_audio,
            target_audio_lens,
            prompt_lens,
            target_phoneme_tokens,
            target_phoneme_lens,
            source_codes,
            source_codes_lens,
            target_codes,
            target_codes_lens,
        )


def build_phoneme_channel(
    cut: Cut,
    phoneme_tokenizer,
    frame_length: Seconds,
    roles: set[str],
    ignore_phoneme_languages: list[str],
    pad_id: int = -1,
    add_text_bos_and_eos_in_each_turn: bool = True,
) -> torch.Tensor:
    """
    Build a frame-aligned phoneme sequence for a single cut, mirroring text token logic.
    """
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    if cut.has_custom("lang"):
        language = cut.lang
    else:
        language = cut.supervisions[0].language if cut.supervisions[0].has_custom("language") else "en"

    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            if isinstance(phoneme_tokenizer, IPABPETokenizer):
                if not supervision.has_custom("ipa"):
                    logging.warning(f"'ipa' field not found in cut {cut.id}. Using empty string.")
                    ipa_text = ""
                else:
                    ipa_text = supervision.ipa
                    
                if language in ignore_phoneme_languages:
                    ipa_text = ""
            else:
                ipa_text = supervision.text

            phoneme_ids = phoneme_tokenizer.encode(ipa_text)
            if add_text_bos_and_eos_in_each_turn:
                phoneme_ids = [phoneme_tokenizer.bos_token_id] + phoneme_ids
            
            phoneme_ids = torch.as_tensor(phoneme_ids, dtype=torch.long)

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(f"Supervision offset {pos} larger than {len(tokens)}. {diagnostic}")
                continue

            endpos = pos + len(phoneme_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(f"Truncating phoneme_ids by {trunc_len}. {diagnostic}")
                phoneme_ids = phoneme_ids[:trunc_len]
            
            try:
                tokens[pos:endpos] = phoneme_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {phoneme_ids.shape=} {diagnostic}") from e

            if add_text_bos_and_eos_in_each_turn:
                eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
                if eospos < len(tokens):
                    tokens[eospos] = phoneme_tokenizer.eos_token_id

    return tokens


def collate_phoneme_channel(
    cuts: CutSet,
    phoneme_tokenizer,
    frame_length: Seconds,
    roles: set[str],
    ignore_phoneme_languages: list[str],
    add_text_bos_and_eos_in_each_turn: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate frame-aligned phoneme channels.
    """
    pad_id = phoneme_tokenizer.pad
    tokens = [
        build_phoneme_channel(
            c,
            phoneme_tokenizer=phoneme_tokenizer,
            frame_length=frame_length,
            roles=roles,
            ignore_phoneme_languages=ignore_phoneme_languages,
            pad_id=pad_id,
            add_text_bos_and_eos_in_each_turn=add_text_bos_and_eos_in_each_turn,
        )
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def add_speech_delay(
    source_audio: torch.Tensor,
    source_audio_lens: torch.Tensor,
    target_audio: torch.Tensor,
    target_audio_lens: torch.Tensor,
    num_delay_speech_tokens: int,
    target_samples_per_frame: int,
    source_samples_per_frame: int,
    source_codes: torch.Tensor = None,
    source_codes_lens: torch.Tensor = None,
    target_codes: torch.Tensor = None,
    target_codes_lens: torch.Tensor = None,
):
    """
    Apply a speech delay by padding audio waveforms based on the number of delay speech tokens.
    """
    extra_target_samples = int(num_delay_speech_tokens * target_samples_per_frame)
    target_audio = F.pad(target_audio, (extra_target_samples, 0))
    target_audio_lens = target_audio_lens + extra_target_samples

    extra_source_samples = int(num_delay_speech_tokens * source_samples_per_frame)
    source_audio = F.pad(source_audio, (0, extra_source_samples))
    source_audio_lens = source_audio_lens + extra_source_samples

    if target_codes is not None:
        target_codes = F.pad(target_codes, (num_delay_speech_tokens, 0))
        target_codes_lens = target_codes_lens + num_delay_speech_tokens
    
    if source_codes is not None:
        source_codes = F.pad(source_codes, (0, num_delay_speech_tokens))
        source_codes_lens = source_codes_lens + num_delay_speech_tokens

    return (
        source_audio, source_audio_lens, target_audio, target_audio_lens,
        source_codes, source_codes_lens, target_codes, target_codes_lens
    )


def collate_system_prompt(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    ignore_data_system_prompt: bool = False,
    tokenizer_names: list[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate system prompts from cuts.
    System prompts should be stored in cut.custom['system_prompt'].
    """
    pad_id = get_pad_id(tokenizer)
    tokens = []
    system_prompts_raw = []
    
    for i, c in enumerate(cuts):
        tok_name = tokenizer_names[i] if tokenizer_names else "english_phoneme"
        
        def _encode(txt):
            if hasattr(tokenizer, "encode"):
                try:
                    return tokenizer.encode(text=txt, tokenizer_name=tok_name)
                except TypeError:
                    return tokenizer.encode(text=txt)
            return tokenizer.text_to_ids(txt)

        if c.custom and c.custom.get("system_prompt", None) and not ignore_data_system_prompt:
            prompt_text = c.custom["system_prompt"]
            tokens.append(
                torch.as_tensor(
                    [tokenizer.bos] + _encode(prompt_text) + [tokenizer.eos], dtype=torch.long
                )
            )
            system_prompts_raw.append(prompt_text)
        else:
            if getattr(c, "type", None):
                prompt_text = c.type
                tokens.append(
                    torch.as_tensor(
                        [tokenizer.bos] + _encode(prompt_text) + [tokenizer.eos], dtype=torch.long
                    )
                )
                system_prompts_raw.append(prompt_text)
            else:
                logging.warning(
                    "No system prompt or dataset type defined on the config! Using a eos token as system prompt!"
                )
                tokens.append(torch.as_tensor([tokenizer.eos], dtype=torch.long))
                system_prompts_raw.append("")

    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens, system_prompts_raw


def get_audio_prompt(
    cuts: CutSet,
    target_sample_rate: int,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve an audio prompt for speaker conditioning.
    """
    if hasattr(cuts[0], "context_audio"):
        audio_prompt = []
        audio_prompt_lens = []

        for cut in cuts:
            ref_audio = cut.context_audio.resample(target_sample_rate).load_audio()
            ref_audio = torch.tensor(ref_audio).float()
            ref_audio_len = ref_audio.shape[1]

            audio_prompt.append(ref_audio.squeeze(0))
            audio_prompt_lens.append(ref_audio_len)

        audio_prompt = collate_vectors(audio_prompt, padding_value=0).float()
        audio_prompt_lens = torch.tensor(audio_prompt_lens).long()

    else:
        cuts = sanitize_cuts(cuts)
        audio_prompt, audio_prompt_lens = collate_random_turn_audio(
            cuts.resample(target_sample_rate, recording_field=recording_field),
            roles=roles,
            recording_field=recording_field,
        )

    return audio_prompt, audio_prompt_lens


def sanitize_cuts(cuts: CutSet) -> CutSet:
    """
    Adjusts supervisions to fit within the cut's truncated duration.
    """
    sanitized_list = []

    for cut in cuts:
        valid_supervisions = []
        for sup in cut.supervisions:
            if sup.start >= cut.duration:
                continue

            if sup.end > cut.duration:
                new_duration = cut.duration - sup.start

                if new_duration <= 0:
                    continue

                new_sup = deepcopy(sup)
                new_sup.duration = new_duration
                valid_supervisions.append(new_sup)

            else:
                valid_supervisions.append(sup)

        cut.supervisions = valid_supervisions
        sanitized_list.append(cut)

    return cuts.from_cuts(sanitized_list)


def collate_random_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and collate reference audio from random speaker turns.
    """
    selected_turn_audios = []
    selected_turn_audios_lens = []
    for cut in cuts:
        matching_supervisions = [s for s in cut.supervisions if s.speaker in roles]
        if len(matching_supervisions) == 0:
            target_duration = 5.0
            num_samples = int(target_duration * cut.sampling_rate)

            silence_tensor = torch.zeros(num_samples, dtype=torch.float32)
            selected_turn_audios.append(silence_tensor)
            selected_turn_audios_lens.append(num_samples)
            logging.warning(
                "There is no target speaker supervision available on this sample! Using a silence audio as audio prompt!"
            )
        else:
            selected_supervision = random.choice(matching_supervisions)
            truncated_audio = cut.truncate(
                offset=max(0, selected_supervision.start), duration=selected_supervision.duration
            ).load_custom(recording_field)

            selected_turn_audios.append(truncated_audio.squeeze(0))
            selected_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(selected_turn_audios, padding_value=0), torch.tensor(selected_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    add_text_bos_and_eos_in_each_turn: bool = True,
    tokenizer_names: list[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build and collate token channels aligned to the audio frame grid.
    """
    pad_id = get_pad_id(tokenizer)
    tokens = []
    
    for i, c in enumerate(cuts):
        tok_name = tokenizer_names[i] if tokenizer_names else "english_phoneme"
        tokens.append(
            build_token_channel(
                c,
                tokenizer=tokenizer,
                frame_length=frame_length,
                roles=roles,
                pad_id=pad_id,
                add_text_bos_and_eos_in_each_turn=add_text_bos_and_eos_in_each_turn,
                tokenizer_name=tok_name,
            )
        )
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
    add_text_bos_and_eos_in_each_turn: bool = True,
    tokenizer_name: str = "english_phoneme",
) -> torch.Tensor:
    """
    Build a frame-aligned token sequence for a single cut.
    """
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

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

            if add_text_bos_and_eos_in_each_turn:
                text_ids = torch.as_tensor([tokenizer.bos] + raw_ids)
            else:
                text_ids = torch.as_tensor(raw_ids)

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            if add_text_bos_and_eos_in_each_turn:
                eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
                if eospos < len(tokens):
                    tokens[eospos] = tokenizer.eos

    return tokens


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text.
    """
    text = _TIMESTAMP_PATTERN.sub("", text)
    return _SPACE_PATTERN.sub(" ", text).strip()


def sample_audio_segments_repeat(
    prompt_audio: torch.Tensor,
    prompt_audio_lens: torch.Tensor,
    n_sample: int,
    sample: bool = True,
) -> torch.Tensor:
    """
    Extract audio segments of length n_sample.
    """
    B, T = prompt_audio.shape
    device = prompt_audio.device
    out = torch.zeros(B, n_sample, device=device, dtype=prompt_audio.dtype)

    for b in range(B):
        length = min(prompt_audio_lens[b].item(), T)

        if length <= 0:
            continue

        if length >= n_sample:
            if sample:
                max_start = max(1, length - n_sample + 1)
                start = torch.randint(0, max_start, (1,), device=device).item()
            else:
                start = 0
            out[b] = prompt_audio[b, start : start + n_sample]

        else:
            start = 0
            segment = prompt_audio[b, start:length]

            repeat_times = (n_sample + (length - start) - 1) // (length - start)
            repeated = segment.repeat(repeat_times)[:n_sample]
            out[b] = repeated

    return out
