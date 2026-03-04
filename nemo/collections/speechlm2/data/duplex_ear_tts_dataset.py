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

import torch
import torch.nn.functional as F
import torch.utils.data
from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.utils import logging


class DuplexEARTTSDataset(torch.utils.data.Dataset):
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

        add_audio_prompt (bool, optional):
            If True, an optional audio/speaker prompt is appended after the description.
            Default: `True`.

        audio_prompt_duration (float, optional):
            Duration (in seconds) of the audio prompt appended when
            `add_audio_prompt=True`. Default: `3.0`.

        num_delay_speech_tokens (int, optional):
            Number of PAD tokens to insert before speech tokens to artificially
            delay the start of speech. Default: `0`.

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
        add_text_bos_and_eos_in_each_turn: bool = True,
        add_audio_prompt: bool = True,
        audio_prompt_duration: float = 3.0,
        num_delay_speech_tokens: int = 0,
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

        # compute source and target samples_per_frame
        self.source_samples_per_frame = int(self.source_sample_rate * self.frame_length)
        self.target_samples_per_frame = int(self.target_sample_rate * self.frame_length)

        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.transform_text(_strip_timestamps)
        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        target_audio, target_audio_lens = collate_audio(
            cuts.resample(self.target_sample_rate), recording_field="target_audio"
        )
        target_text_tokens, target_token_lens = collate_token_channel(
            cuts,
            self.tokenizer,
            self.frame_length,
            roles=self.output_roles,
            add_text_bos_and_eos_in_each_turn=self.add_text_bos_and_eos_in_each_turn,
        )
        source_tokens, source_token_lens = collate_token_channel(
            cuts,
            self.tokenizer,
            self.frame_length,
            roles=self.input_roles,
            add_text_bos_and_eos_in_each_turn=self.add_text_bos_and_eos_in_each_turn,
        )

        audio_prompt, audio_prompt_lens = get_audio_prompt(
            cuts, self.target_sample_rate, roles=self.output_roles, recording_field="target_audio"
        )

        # add speech channel delay if needed
        if self.num_delay_speech_tokens:
            source_audio, source_audio_lens, target_audio, target_audio_lens = add_speech_delay(
                source_audio,
                source_audio_lens,
                target_audio,
                target_audio_lens,
                self.num_delay_speech_tokens,
                self.target_samples_per_frame,
                self.source_samples_per_frame,
            )

        # add audio prompt if needed
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
        ) = self.maybe_add_audio_prompt(
            target_text_tokens,
            target_token_lens,
            source_tokens,
            source_token_lens,
            target_audio,
            target_audio_lens,
            source_audio,
            source_audio_lens,
            audio_prompt,
            audio_prompt_lens,
        )

        # create non_prompt_mask that should mask desc plus audio prompt if used
        non_prompt_mask = get_mask_from_lengths(target_token_lens)
        for i, frame in enumerate(prompt_lens):
            non_prompt_mask[i, : frame - 1] = 0.0

        max_len = max(target_token_lens)

        # Segment IDs per sequence (padded)
        aligned_segment_ids = torch.stack(
            [
                torch.nn.functional.pad(torch.full((seq_len,), i), (0, max_len - seq_len), value=-1)  # -1 for padding
                for i, seq_len in enumerate(target_token_lens)
            ],
            dim=0,
        )  # [B, max_len]

        # Attention mask: same-segment & causal
        aligned_attention_mask = (
            aligned_segment_ids.unsqueeze(-2) == aligned_segment_ids.unsqueeze(-1)
        ) & (  # [B, max_len, max_len]
            torch.arange(max_len).unsqueeze(0).unsqueeze(1) <= torch.arange(max_len).unsqueeze(0).unsqueeze(-1)
        )  # causal tril

        aligned_attention_mask = aligned_attention_mask.unsqueeze(1)  # [B, 1, max_len, max_len]

        # create position IDs from the aligned length
        aligned_position_ids = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.arange(seq_len), (0, max(target_token_lens) - seq_len), value=0
                )  # value=0 is safe for padding
                for seq_len in target_token_lens
            ],
            dim=0,
        )

        return {
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
            "task": [getattr(cut, "task", "s2s_duplex") for cut in cuts],
        }

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
    ):
        """
        Prepend an audio-based speaker prompt and aligned text tokens to the Duplex S2S inputs.

        This method optionally injects a speaker-reference audio prompt at the beginning of each
        sample in the batch. The prompt is inserted in the target-audio channel and aligned text
        padding is inserted into the text-token streams (input text tokens and source tokens).

        Args:
            target_text_tokens (torch.Tensor):
                Tensor of input text tokens with shape [B, T_text].
                dtype: torch.long.

            target_token_lens (torch.Tensor):
                Lengths of target_text_tokens per batch element (before padding). shape [B].

            source_tokens (torch.Tensor):
                Source-side text tokens, shape [B, T_src_text], dtype torch.long.

            source_token_lens (torch.Tensor):
                Source text token lengths per batch element, shape [B].

            target_audio (torch.Tensor):
                Target-side audio waveforms, shape [B, T_audio], dtype torch.float32/float16.

            target_audio_lens (torch.Tensor):
                Target audio lengths per batch element, shape [B].

            source_audio (torch.Tensor):
                Source-side audio waveforms, shape [B, T_audio], dtype torch.float32/float16.

            source_audio_lens (torch.Tensor):
                Source audio lengths per batch element, shape [B].

            audio_prompt (torch.Tensor):
                Audio prompt waveforms to sample from, shape [B, T_prompt_audio].

            audio_prompt_lens (torch.Tensor):
                Valid lengths for audio_prompt per batch element.

        Returns:
            Tuple containing:
                target_text_tokens (torch.Tensor):
                    Updated text tokens with prepended prompt-aligned tokens. Shape [B, T'].

                target_token_lens (torch.Tensor):
                    Updated token lengths per batch element.

                source_tokens (torch.Tensor):
                    Updated source text tokens with prompt padding included. Shape [B, T'].

                source_token_lens (torch.Tensor):
                    Updated source token lengths per batch element.

                source_audio (torch.Tensor):
                    Updated source audio with silence padding. Shape [B, T_audio'].

                source_audio_lens (torch.Tensor):
                    Updated source audio lengths.

                target_audio (torch.Tensor):
                    Updated target audio with prompt audio and silence padding. Shape [B, T_audio'].

                target_audio_lens (torch.Tensor):
                    Updated target audio lengths.

                prompt_lens (list[int]):
                    Length (in text-token units) of the prompt region per batch item.
        """

        text_pad_id = get_pad_id(self.tokenizer)

        target_text_tokens_ = []
        source_tokens_ = []
        source_audio_ = []
        target_audio_ = []
        prompt_lens = []

        for i in range(target_text_tokens.size(0)):
            first_text_frame = torch.tensor(
                [self.tokenizer.eos],
                dtype=torch.long,
                device=target_text_tokens.device,
            )

            if self.add_audio_prompt:
                # Compute audio prompt duration in samples (rounded to frame boundaries)
                prompt_audio_size = int(
                    ((self.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame)
                    * self.target_samples_per_frame
                )

                prompt_audio = sample_audio_segments_repeat(
                    audio_prompt, audio_prompt_lens, prompt_audio_size, sample=True
                )

                # add silence at the end of the prompt
                prompt_audio[:, -int(self.target_samples_per_frame * 2) :] = 0

                # Number of text tokens to insert to align with prompt_audio frames
                prompt_audio_text_pad_size = prompt_audio_size // self.target_samples_per_frame

                prompt_audio_text_pad = (
                    torch.ones(
                        prompt_audio_text_pad_size,
                        device=target_text_tokens.device,
                        dtype=target_text_tokens.dtype,
                    )
                    * text_pad_id
                )
                prompt_audio_text_pad[-1] = self.tokenizer.eos

                new_target_text_tokens = torch.cat(
                    [
                        first_text_frame.to(target_text_tokens.dtype),
                        prompt_audio_text_pad,
                        target_text_tokens[i],
                    ]
                )
                target_text_tokens_.append(new_target_text_tokens)
                target_token_lens[i] += len(first_text_frame) + prompt_audio_text_pad_size

                new_source_tokens = torch.cat([first_text_frame, prompt_audio_text_pad, source_tokens[i]])
                source_tokens_.append(new_source_tokens)
                source_token_lens[i] += len(first_text_frame) + prompt_audio_text_pad_size

                # Silence in source audio during prompt processing
                pad_size_src = (len(first_text_frame) * self.source_samples_per_frame) + prompt_audio.size(1)
                pad_audio_src = torch.zeros(
                    pad_size_src,
                    device=source_audio.device,
                    dtype=source_audio.dtype,
                )
                source_audio_.append(torch.cat([pad_audio_src, source_audio[i]]))
                source_audio_lens[i] += pad_size_src

                # Insert prompt audio in the target channel
                pad_size_tgt = len(first_text_frame) * self.target_samples_per_frame
                pad_audio_tgt = torch.zeros(
                    pad_size_tgt,
                    device=target_audio.device,
                    dtype=target_audio.dtype,
                )
                target_audio_.append(torch.cat([pad_audio_tgt, prompt_audio[i], target_audio[i]]))
                target_audio_lens[i] += pad_size_tgt + prompt_audio.size(1)

                prompt_lens.append(len(first_text_frame) + prompt_audio_text_pad_size - 1)

            else:
                # Add only a single text-frame (EOS) as prompt
                target_text_tokens_.append(torch.cat([first_text_frame, target_text_tokens[i]]))
                target_token_lens[i] += len(first_text_frame)

                source_tokens_.append(torch.cat([first_text_frame, source_tokens[i]]))
                source_token_lens[i] += len(first_text_frame)

                pad_size_src = len(first_text_frame) * self.source_samples_per_frame
                pad_audio_src = torch.zeros(
                    pad_size_src,
                    device=source_audio.device,
                    dtype=source_audio.dtype,
                )
                source_audio_.append(torch.cat([pad_audio_src, source_audio[i]]))
                source_audio_lens[i] += pad_size_src

                pad_size_tgt = len(first_text_frame) * self.target_samples_per_frame
                pad_audio_tgt = torch.zeros(
                    pad_size_tgt,
                    device=target_audio.device,
                    dtype=target_audio.dtype,
                )
                target_audio_.append(torch.cat([pad_audio_tgt, target_audio[i]]))
                target_audio_lens[i] += pad_size_tgt

                prompt_lens.append(len(first_text_frame))

        target_text_tokens = collate_vectors(target_text_tokens_, padding_value=text_pad_id)
        source_tokens = collate_vectors(source_tokens_, padding_value=text_pad_id)
        source_audio = collate_vectors(source_audio_, padding_value=0)
        target_audio = collate_vectors(target_audio_, padding_value=0)

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
        )


def add_speech_delay(
    source_audio: torch.Tensor,
    source_audio_lens: torch.Tensor,
    target_audio: torch.Tensor,
    target_audio_lens: torch.Tensor,
    num_delay_speech_tokens: int,
    target_samples_per_frame: int,
    source_samples_per_frame: int,
):
    """
    Apply a speech delay by padding audio waveforms based on the number of delay speech tokens.

    Behavior:
        - Target audio is *left padded* to force the model to predict initial silence.
        - Source audio is *right padded* to maintain size consistency for attention alignment.

    Args:
        source_audio (FloatTensor [B, T_src]):
            Source/input audio waveforms.

        source_audio_lens (LongTensor [B]):
            Lengths of source audio in samples.

        target_audio (FloatTensor [B, T_tgt]):
            Target/output audio waveforms.

        target_audio_lens (LongTensor [B]):
            Lengths of target audio in samples.

        num_delay_speech_tokens (int):
            Number of delay tokens inserted on the text side.

        target_samples_per_frame (int):
            Number of audio samples per frame for target audio.

        source_samples_per_frame (int):
            Number of audio samples per frame for source audio.

    Returns:
        Tuple containing:
            - source_audio (FloatTensor [B, T_src + pad])
            - source_audio_lens (LongTensor [B])
            - target_audio (FloatTensor [B, T_tgt + pad])
            - target_audio_lens (LongTensor [B])
    """
    # Compute target-side left padding for the delay
    extra_target_samples = int(num_delay_speech_tokens * target_samples_per_frame)

    # Left-pad target audio: forces model to generate silence initially
    target_audio = F.pad(target_audio, (extra_target_samples, 0))
    target_audio_lens = target_audio_lens + extra_target_samples

    # Compute source-side right padding to maintain alignment
    extra_source_samples = int(num_delay_speech_tokens * source_samples_per_frame)

    # Right-pad source audio: avoids mismatch when aligning source/target frames
    source_audio = F.pad(source_audio, (0, extra_source_samples))
    source_audio_lens = source_audio_lens + extra_source_samples

    return source_audio, source_audio_lens, target_audio, target_audio_lens


def get_audio_prompt(
    cuts: CutSet,
    target_sample_rate: int,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve an audio prompt for speaker conditioning.

    This function returns:
        - context audio if available (per-cut),
        - otherwise a random turn belonging to the desired speaker roles.

    Behavior:
        1. If `cut.context_audio` exists, use it as the reference.
        2. Otherwise, select a random audio turn from the same target-role speakers.

    Args:
        cuts (CutSet):
            Batch of cuts from which to extract reference audio.

        target_sample_rate (int):
            Sample rate to which reference audio is resampled.

        roles (set[str]):
            Set of speaker roles to sample from when selecting random turns.

        recording_field (str, optional):
            Name of the audio field in the cut ("recording", "target_audio", etc.).
            Used when sampling random reference turns.

    Returns:
        Tuple containing:
        - audio_prompt (FloatTensor [B, T]):
            Padded batch of reference waveforms.
        - audio_prompt_lens (LongTensor [B]):
            Lengths of each reference waveform before padding.
    """
    # use provided context audio directly
    if hasattr(cuts[0], "context_audio"):
        audio_prompt = []
        audio_prompt_lens = []

        for cut in cuts:
            ref_audio = cut.context_audio.resample(target_sample_rate).load_audio()
            ref_audio = torch.tensor(ref_audio).float()  # shape: [1, T]
            ref_audio_len = ref_audio.shape[1]

            audio_prompt.append(ref_audio.squeeze(0))  # [T]
            audio_prompt_lens.append(ref_audio_len)

        audio_prompt = collate_vectors(audio_prompt, padding_value=0).float()
        audio_prompt_lens = torch.tensor(audio_prompt_lens).long()

    else:
        # sample a reference turn from the target-role speakers
        audio_prompt, audio_prompt_lens = collate_random_turn_audio(
            cuts.resample(target_sample_rate),
            roles=roles,
            recording_field=recording_field,
        )

    return audio_prompt, audio_prompt_lens


def collate_random_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and collate reference audio from random speaker turns.

    For each cut in the batch, this function:
        - selects a random supervision belonging to one of the specified roles,
        - extracts the corresponding audio segment,
        - collates all segments into a padded batch.

    Args:
        cuts (CutSet):
            Batch of cuts to sample from.

        roles (set[str]):
            Set of speaker roles to consider when selecting random turns.

        recording_field (str, optional):
            Name of the audio field to load from the cut
            (e.g., "recording", "target_audio").

    Returns:
        Tuple containing:
        - audio (FloatTensor [B, T]):
            Padded batch of sampled reference waveforms.
        - audio_lens (LongTensor [B]):
            Lengths of each waveform before padding.
    """
    selected_turn_audios = []
    selected_turn_audios_lens = []
    for cut in cuts:
        # Filter supervisions matching roles
        matching_supervisions = [s for s in cut.supervisions if s.speaker in roles]

        # Randomly select one supervision
        selected_supervision = random.choice(matching_supervisions)

        # Truncate audio according to supervision
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build and collate token channels aligned to the audio frame grid.

    This function converts text supervisions into frame-level token
    representations for each cut, pads them to a uniform length, and
    returns both the padded tokens and their true lengths.

    Args:
        cuts (CutSet):
            Batch of cuts from which to extract token channels.

        tokenizer (TokenizerSpec):
            Tokenizer used to convert text into token IDs.

        frame_length (Seconds):
            Duration of a single audio frame, used to align text tokens
            to audio frames.

        roles (set[str]):
            Speaker roles whose text will be included in the token channel.

        add_text_bos_and_eos_in_each_turn (bool, optional):
            Whether to insert BOS at the beginning and EOS at the end of
            each speaking turn.

    Returns:
        Tuple containing:
        - tokens (LongTensor [B, T]):
            Padded batch of frame-aligned token sequences.
        - token_lens (LongTensor [B]):
            Length of each sequence before padding.
    """
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(
            c,
            tokenizer=tokenizer,
            frame_length=frame_length,
            roles=roles,
            pad_id=pad_id,
            add_text_bos_and_eos_in_each_turn=add_text_bos_and_eos_in_each_turn,
        )
        for c in cuts
    ]
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
) -> torch.Tensor:
    """
    Build a frame-aligned token sequence for a single cut.

    This function maps speaking turns into a token channel aligned to the
    audio frame grid. Tokens are inserted at frame positions corresponding
    to supervision start times, with optional BOS and EOS insertion.

    Any region not covered by text is filled with `pad_id`.

    Args:
        cut (Cut):
            Input cut containing audio and supervisions.

        tokenizer (TokenizerSpec):
            Tokenizer used to encode text into tokens.

        frame_length (Seconds):
            Duration of one frame, used to align text to the audio grid.

        roles (set[str]):
            Speaker roles whose text should be included.

        pad_id (int, optional):
            Token ID used for padding empty frames.

        add_text_bos_and_eos_in_each_turn (bool, optional):
            Whether to insert BOS before and EOS after each supervision.

    Returns:
        tokens (LongTensor [T]):
            Frame-aligned token sequence for the cut.
    """

    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            text = supervision.text
            if add_text_bos_and_eos_in_each_turn:
                text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(text))
            else:
                text_ids = torch.as_tensor(tokenizer.text_to_ids(text))

            # Determine the frame offset for the start of the supervision to insert the text tokens.
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            # Determine the frame offset for the last non-EOS text token to form a valid range for insertion;
            # Note that EOS will be placed possibly much later, at the frame that coincides with end of speech,
            # rather than end of text. The gap between last non-EOS token and EOS token will be filled with `pad_id`.
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

            # Insert EOS at the end of the supervision segment.
            if add_text_bos_and_eos_in_each_turn:
                eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
                if eospos < len(tokens):  # skip otherwise - unfinished turn
                    tokens[eospos] = tokenizer.eos

    return tokens


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces


def sample_audio_segments_repeat(
    prompt_audio: torch.Tensor,
    prompt_audio_lens: torch.Tensor,
    n_sample: int,
    sample: bool = True,
) -> torch.Tensor:
    """
    Extract audio segments of length n_sample.
    If sample=True: randomly sample segments (repeating if shorter).
    If sample=False: always take from the beginning (repeating if shorter).

    Args:
        prompt_audio: Tensor [B, T]
        prompt_audio_lens: Tensor [B] with valid lengths
        n_sample: int, target length per segment
        sample: bool, whether to randomly sample (True) or take first seconds (False)

    Returns:
        Tensor [B, n_sample]
    """
    B, T = prompt_audio.shape
    device = prompt_audio.device
    out = torch.zeros(B, n_sample, device=device, dtype=prompt_audio.dtype)

    for b in range(B):
        length = min(prompt_audio_lens[b].item(), T)

        # Case: empty audio
        if length <= 0:
            continue

        if length >= n_sample:
            if sample:
                # Random start (safe bounds)
                max_start = max(1, length - n_sample + 1)
                start = torch.randint(0, max_start, (1,), device=device).item()
            else:
                # Deterministic: take from start
                start = 0
            out[b] = prompt_audio[b, start : start + n_sample]

        else:
            # Audio shorter than target → repeat
            start = 0
            segment = prompt_audio[b, start:length]

            repeat_times = (n_sample + (length - start) - 1) // (length - start)
            repeated = segment.repeat(repeat_times)[:n_sample]
            out[b] = repeated

    return out
