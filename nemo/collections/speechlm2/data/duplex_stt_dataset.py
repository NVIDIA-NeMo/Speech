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
import json
import random
import re

import torch
import torch.utils.data
from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.data.lhotse.text_adapters import Formattable
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.force_align import ForceAligner
from nemo.collections.speechlm2.data.s2s_dataset import _strip_timestamps
from nemo.collections.speechlm2.data.utils import (
    MCQ_SYSTEM_PROMPT_NOTHINK,
    MCQ_SYSTEM_PROMPT_THINK,
    get_pad_id,
    is_asr_cut,
    is_mcq_cut_train,
    is_mcq_cut_val,
    normalize_numbers,
)
from nemo.collections.speechlm2.parts.augmentation import AudioAugmenter
from nemo.utils import logging


class DuplexSTTDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-text models.

    Unlike DuplexS2SDataset, this dataset does not require target audio and is suitable
    for training on standard ASR, AST, and SpeechQA datasets in addition to duplex data.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs. Must support BOS and EOS tokens.

        frame_length (Seconds):
            Duration of a single frame in seconds.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        input_roles (list[str], optional):
            Speaker roles to treat as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            Speaker roles to treat as outputs. Defaults to ["agent"].

        aug_by_swap_role (bool, optional):
            Whether to augment data by swapping user/agent roles. Defaults to False.
            Note: Enabling this requires agent audio to be available in cut.custom['target_audio'].

        cfg (dict, optional):
            Dataset configuration (e.g., word_align_position).

        model_cfg (dict, optional):
            Model configuration (e.g., predict_user_text, force_align_user_text).
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        aug_by_swap_role: bool = False,
        cfg: dict = None,
        model_cfg: dict = None,
        is_training: bool = True,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.aug_by_swap_role = aug_by_swap_role
        self.is_training = is_training

        self.word_align_position = cfg.get("word_align_position", "left") if cfg is not None else "left"
        self.predict_user_text = model_cfg.get("predict_user_text", False) if model_cfg is not None else False
        self.force_align_user_text = model_cfg.get("force_align_user_text", False) if model_cfg is not None else None
        self.force_align_device = model_cfg.get("force_align_device", "cuda") if model_cfg is not None else "cuda"

        self.prepend_word_space = cfg.get("prepend_word_space", True) if cfg is not None else True
        self.early_interruption_prob = cfg.get("early_interruption_prob", 0.0) if cfg is not None else 0.0
        self.add_mcq_prompt = cfg.get("add_mcq_prompt", False) if cfg is not None else False

        self.add_filler_response_for_asr = cfg.get("add_filler_response_for_asr", False) if cfg is not None else False
        self.filler_response_delay = cfg.get("filler_response_delay", 0) if cfg is not None else 0
        self.filler_responses = None

        self.use_numbers_norm = model_cfg.get("use_numbers_norm", True) if model_cfg is not None else True

        self.cfg = cfg
        self.model_cfg = model_cfg

        self.force_aligner = None
        if self.force_align_user_text:
            self.force_aligner = ForceAligner(device=self.force_align_device, frame_length=self.frame_length)

        self.audio_augmenter = None
        if cfg is not None and (
            cfg.get('use_noise_aug', None)
            or cfg.get('use_room_ir_aug', None)
            or cfg.get('use_mic_ir_aug', None)
            or cfg.get('use_codec_aug', None)
        ):
            self.audio_augmenter = AudioAugmenter(sample_rate=source_sample_rate)

        assert tokenizer.bos is not None, "BOS support in the tokenizer is required."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required."
        
        user_bos_token = '^'
        user_eos_token = '$'
        self.user_bos_id = self.tokenizer.text_to_ids(user_bos_token)[0]
        self.user_eos_id = self.tokenizer.text_to_ids(user_eos_token)[0]
        self.agent_bos_id = self.tokenizer.bos
        self.agent_eos_id = self.tokenizer.eos
        self.pad_id = get_pad_id(self.tokenizer)

    def _is_augmentation_task(self, task: str) -> bool:
        if self.cfg is not None and self.cfg.get('force_use_noise_augmentation', False):
            return True
        return task not in ('s2s_duplex_overlap_as_s2s_duplex', 'asr')

    def _load_filler_responses(self):
        if self.filler_responses is not None:
            return
        filler_response_file = self.cfg.get("filler_response_file", None)
        if filler_response_file is None:
            raise ValueError("filler_response_file must be specified when add_filler_response_for_asr is True")
        with open(filler_response_file, 'r') as f:
            self.filler_responses = json.load(f)["agent_responses"]
        logging.info(f"Loaded {len(self.filler_responses)} filler responses from {filler_response_file}")

    def _inject_filler_responses(self, target_tokens: torch.Tensor, target_token_lens: torch.Tensor) -> None:
        """Inject random filler agent responses into target_tokens for ASR cuts.

        For each sample, finds the existing agent BOS, clears it, then writes
        [BOS filler_tok1 ... filler_tokN] at (old_bos_pos + filler_response_delay).
        Truncates if the filler extends beyond the sequence length.
        Updates target_token_lens based on filler duration_ms when available.
        """
        self._load_filler_responses()
        bos_id = self.agent_bos_id
        seq_len = target_tokens.shape[1]

        for i in range(target_tokens.shape[0]):
            bos_positions = (target_tokens[i] == bos_id).nonzero(as_tuple=True)[0]
            if bos_positions.numel() == 0:
                logging.warning(f"[_inject_filler] sample {i}: no BOS found, skipping")
                continue

            old_bos_pos = bos_positions[0].item()
            target_tokens[i, old_bos_pos] = self.pad_id

            insert_pos = old_bos_pos + self.filler_response_delay
            if insert_pos >= seq_len:
                logging.warning(
                    f"[_inject_filler] sample {i}: insert_pos={insert_pos} >= seq_len={seq_len}, skipping"
                )
                continue

            filler = random.choice(self.filler_responses)
            filler_token_ids = self.tokenizer.text_to_ids(filler["text"])
            turn_tokens = [bos_id] + filler_token_ids
            turn_tensor = torch.tensor(turn_tokens, dtype=torch.long)

            available = seq_len - insert_pos
            if len(turn_tensor) > available:
                logging.warning(
                    f"[_inject_filler] sample {i}: truncating filler from {len(turn_tensor)} to {available} tokens"
                )
                turn_tensor = turn_tensor[:available]

            target_tokens[i, insert_pos:insert_pos + len(turn_tensor)] = turn_tensor

            filler_end = insert_pos + len(turn_tensor)
            duration_ms = filler.get("duration_ms", None)
            if duration_ms is not None:
                duration_frames = max(1, int(duration_ms / (self.frame_length * 1000)))
                duration_end = insert_pos + duration_frames
            else: # fallback to filler end
                duration_end = filler_end
            target_token_lens[i] = min(max(duration_end, filler_end), seq_len)

    def _apply_early_interruption_augmentation(
        self,
        target_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        source_audio: torch.Tensor,
        source_audio_lens: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Simulate early interruption by randomly truncating an agent turn with overlap."""
        target_seq = target_tokens[batch_idx]
        bos_id = self.agent_bos_id
        eos_id = self.agent_eos_id
        pad_id = self.pad_id
        overlap_tokens = self.cfg.get("early_interruption_overlap_tokens", 8) if self.cfg is not None else 8

        bos_positions = (target_seq == bos_id).nonzero(as_tuple=True)[0]
        eos_positions = (target_seq == eos_id).nonzero(as_tuple=True)[0]

        if len(bos_positions) == 0 or len(eos_positions) == 0:
            return

        turns = []
        for bos_pos in bos_positions:
            matching_eos = eos_positions[eos_positions > bos_pos]
            if len(matching_eos) > 0:
                eos_pos = matching_eos[0]
                turn_tokens = target_seq[bos_pos + 1 : eos_pos]
                non_pad_mask = turn_tokens != pad_id
                all_non_pad_positions = (bos_pos + 1 + non_pad_mask.nonzero(as_tuple=True)[0]).tolist()
                non_pad_positions = [pos for pos in all_non_pad_positions if (eos_pos - pos) > overlap_tokens]

                if len(non_pad_positions) > 0:
                    turns.append(
                        {'bos_pos': bos_pos.item(), 'eos_pos': eos_pos.item(), 'non_pad_positions': non_pad_positions}
                    )

        if len(turns) == 0:
            return

        selected_turn = random.choice(turns)
        cutoff_pos = random.choice(selected_turn['non_pad_positions'])
        original_eos_pos = selected_turn['eos_pos']

        new_eos_pos = min(cutoff_pos + overlap_tokens, original_eos_pos)
        frames_to_remove = original_eos_pos - cutoff_pos
        if frames_to_remove <= 0:
            return

        # Update target_tokens: place eos at new_eos_pos, shift tail, pad at end
        target_tokens[batch_idx, new_eos_pos] = eos_id
        seq_len = target_tokens.shape[1]
        cont_start_pos = original_eos_pos + overlap_tokens
        tail_length = seq_len - (cont_start_pos + 1)
        if tail_length > 0:
            target_tokens[batch_idx, new_eos_pos + 1 : new_eos_pos + 1 + tail_length] = target_tokens[
                batch_idx, cont_start_pos + 1 : cont_start_pos + 1 + tail_length
            ].clone()
        target_tokens[batch_idx, -frames_to_remove:] = pad_id

        # Update source_tokens: shift tail (from cutoff_pos)
        src_frames_to_remove = original_eos_pos - cutoff_pos
        source_seq_len = source_tokens.shape[1]
        source_tail_length = source_seq_len - (original_eos_pos + 1)
        if source_tail_length > 0:
            source_tokens[batch_idx, cutoff_pos + 1 : cutoff_pos + 1 + source_tail_length] = source_tokens[
                batch_idx, original_eos_pos + 1 : original_eos_pos + 1 + source_tail_length
            ].clone()
        source_tokens[batch_idx, -src_frames_to_remove:] = pad_id

        # Update source_audio: shift and pad with silence
        old_source_len = source_audio_lens[batch_idx].item()
        new_bos_source_sample = min(int(cutoff_pos * self.frame_length * self.source_sample_rate), old_source_len)
        original_eos_source_sample = min(
            int(original_eos_pos * self.frame_length * self.source_sample_rate), old_source_len
        )

        source_tail_audio_length = old_source_len - original_eos_source_sample
        if source_tail_audio_length > 0:
            source_audio[batch_idx, new_bos_source_sample : new_bos_source_sample + source_tail_audio_length] = (
                source_audio[batch_idx, original_eos_source_sample:old_source_len].clone()
            )

        source_samples_to_remove = original_eos_source_sample - new_bos_source_sample
        if new_bos_source_sample + source_tail_audio_length < source_audio.shape[1]:
            source_audio[
                batch_idx,
                new_bos_source_sample
                + source_tail_audio_length : new_bos_source_sample
                + source_tail_audio_length
                + source_samples_to_remove,
            ] = 0

    def _create_minimal_batch(self) -> dict:
        """Create a minimal valid batch when all cuts are filtered out."""
        return {
            "sample_id": ["empty_batch"],
            "source_audio": torch.zeros((1, 1000), dtype=torch.float32),
            "source_audio_lens": torch.tensor([1000], dtype=torch.long),
            "target_tokens": torch.full((1, 50), self.pad_id, dtype=torch.long),
            "target_token_lens": torch.tensor([1], dtype=torch.long),
            "source_tokens": torch.full((1, 50), self.pad_id, dtype=torch.long),
            "source_token_lens": torch.tensor([1], dtype=torch.long),
            "source_texts": [""],
            "target_texts": [""],
            "task": ["s2s_duplex"],
        }

    def __getitem__(self, all_cuts: CutSet) -> dict:
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        audio_data = None

        if cuts and getattr(cuts[0], 'task', None) == 'asr':
            filtered_cuts = []
            skipped_cuts = []
            for cut in cuts:
                if self._has_valid_input(cut):
                    filtered_cuts.append(cut)
                else:
                    skipped_cuts.append(cut.id)
            if skipped_cuts:
                logging.info(
                    f"Skipped {len(skipped_cuts)} cuts with empty input text. Skipped cut ids: {', '.join(skipped_cuts)}"
                )
            if not filtered_cuts:
                logging.warning(
                    f"All cuts were filtered out! Original batch size: {len(cuts)}. Returning minimal valid batch."
                )
                return self._create_minimal_batch()
            cuts = CutSet.from_cuts(filtered_cuts)

        if cuts:
            swapped_cuts = []

            if self.aug_by_swap_role:
                for cut in cuts:
                    total_turns = cut.custom.get('total_turns', len(cut.supervisions))

                    if total_turns > 4 and total_turns % 2 == 0:
                        swapped_cut = self._create_role_swapped_cut(cut)
                        if swapped_cut:
                            swapped_cuts.append(swapped_cut)

            if swapped_cuts:
                all_cuts_combined = CutSet.from_cuts(list(cuts) + swapped_cuts)
            else:
                all_cuts_combined = cuts

            prompt_tokens, prompt_token_lens = collate_system_prompt(
                all_cuts_combined,
                self.tokenizer,
                bos_id=self.agent_bos_id,
                eos_id=self.agent_eos_id,
                pad_id=self.pad_id,
                add_mcq_prompt=self.add_mcq_prompt,
            )
            source_audio, source_audio_lens = collate_audio(all_cuts_combined.resample(self.source_sample_rate))

            target_tokens, target_token_lens = collate_token_channel(
                all_cuts_combined,
                self.tokenizer,
                self.frame_length,
                roles=self.output_roles,
                bos_id=self.agent_bos_id,
                eos_id=self.agent_eos_id,
                pad_id=self.pad_id,
                remove_timestamps=True,
                skip_eos=True, # agent eos is placed when source tokens are collated
                use_numbers_norm=self.use_numbers_norm,
            )

            # Inject filler agent responses for ASR cuts (training only)
            if self.add_filler_response_for_asr and self.is_training and is_asr_cut(all_cuts_combined[0]):
                self._inject_filler_responses(target_tokens, target_token_lens)

            # Force align user text (runs in dataloader worker, training only)
            if self.force_align_user_text and self.is_training:
                all_cuts_combined = self.force_aligner.batch_force_align_user_audio(
                    all_cuts_combined, source_sample_rate=self.source_sample_rate
                )

            source_tokens, source_token_lens = collate_token_channel(
                all_cuts_combined,
                self.tokenizer,
                self.frame_length,
                roles=self.input_roles,
                bos_id=self.user_bos_id,
                eos_id=self.agent_eos_id,
                pad_id=self.pad_id,
                word_align_position=self.word_align_position,
                remove_timestamps=not self.predict_user_text,
                prepend_word_space=self.prepend_word_space,
                agent_token_channel=target_tokens,
                agent_token_channel_lengths=target_token_lens.clone(),
                agent_eos_id=self.agent_eos_id,
            )

            # Audio augmentation (runs in dataloader workers for performance, training only)
            if (
                self.audio_augmenter is not None
                and self.is_training
                and self._is_augmentation_task(getattr(all_cuts_combined[0], 'task', 's2s_duplex'))
            ):
                source_audio = self.audio_augmenter.augment_batch(self.cfg, source_audio, source_audio_lens)

            # Early interruption augmentation (training only)
            if self.early_interruption_prob > 0 and self.is_training:
                for batch_idx in range(target_tokens.shape[0]):
                    if random.random() < self.early_interruption_prob:
                        self._apply_early_interruption_augmentation(
                            target_tokens,
                            source_tokens,
                            source_audio,
                            source_audio_lens,
                            batch_idx,
                        )

            audio_data = {
                "sample_id": [str(cut.id) for cut in all_cuts_combined],
                "source_audio": source_audio,
                "source_audio_lens": source_audio_lens,
                "target_tokens": target_tokens,
                "target_token_lens": target_token_lens,
                "source_tokens": source_tokens,
                "source_token_lens": source_token_lens,
                "source_texts": [
                    " ".join(_strip_timestamps(s.text) for s in cut.supervisions if s.speaker in self.input_roles)
                    for cut in all_cuts_combined
                ],
                "target_texts": [
                    " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles)
                    for cut in all_cuts_combined
                ],
                "task": [getattr(cut, "task", "s2s_duplex") for cut in all_cuts_combined],
            }

            if torch.sum(prompt_token_lens) > 0:
                audio_data['prompt_tokens'] = prompt_tokens
                audio_data['prompt_token_lens'] = prompt_token_lens

        text_cuts = all_cuts.filter(lambda c: isinstance(c, Formattable))
        text_data = None
        if text_cuts:
            text_tokens = []
            text_token_lens = []
            for c in text_cuts:
                text_ids = c.input_ids
                text_tokens.append(text_ids)
                text_token_lens.append(text_ids.shape[0])

            text_tokens = collate_vectors(text_tokens, padding_value=self.pad_id)
            text_token_lens = torch.tensor(text_token_lens, dtype=torch.long)
            text_data = {
                "text_tokens": text_tokens,
                "text_token_lens": text_token_lens,
            }

        return {
            "audio_data": audio_data,
            "text_data": text_data,
        }

    def _create_role_swapped_cut(self, cut):
        from io import BytesIO

        import numpy as np
        import soundfile as sf
        from lhotse import AudioSource

        assert (
            'target_audio' in cut.custom
        ), f"Role swapping requires target_audio in cut.custom, but cut {cut.id} does not have it. Disable aug_by_swap_role or ensure your data includes target audio."

        swapped_supervisions = []
        for sup in cut.supervisions:
            if sup.speaker == 'User':
                new_speaker = 'Assistant'
            elif sup.speaker == 'Assistant':
                new_speaker = 'User'
            else:
                continue

            swapped_sup = SupervisionSegment(
                id=sup.id + "_swapped",
                recording_id=sup.recording_id,
                start=sup.start,
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=new_speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment,
            )
            swapped_supervisions.append(swapped_sup)

        swapped_supervisions = sorted(swapped_supervisions, key=lambda s: s.start)

        first_agent_idx = None
        last_user_idx = None

        for i, sup in enumerate(swapped_supervisions):
            if sup.speaker == 'Assistant' and first_agent_idx is None:
                first_agent_idx = i
            if sup.speaker == 'User':
                last_user_idx = i

        filtered_supervisions = []
        for i, sup in enumerate(swapped_supervisions):
            if i != first_agent_idx and i != last_user_idx:
                filtered_supervisions.append(sup)

        if not filtered_supervisions:
            return None

        first_remaining_start = filtered_supervisions[0].start
        adjusted_supervisions = []
        for sup in filtered_supervisions:
            adjusted_sup = SupervisionSegment(
                id=sup.id,
                recording_id=sup.recording_id,
                start=sup.start - first_remaining_start,
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=sup.speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment,
            )
            adjusted_supervisions.append(adjusted_sup)

        total_duration = max(s.start + s.duration for s in adjusted_supervisions)
        total_samples = int(total_duration * cut.sampling_rate)

        new_source_audio = np.zeros(total_samples, dtype=np.float32)

        for sup in adjusted_supervisions:
            start_sample = int(sup.start * cut.sampling_rate)
            end_sample = int((sup.start + sup.duration) * cut.sampling_rate)

            if sup.speaker == 'User':
                # New User was originally Assistant — audio lives in target_audio channel
                original_start = sup.start + first_remaining_start
                agent_audio = (
                    cut.custom['target_audio']
                    .to_cut()
                    .truncate(offset=original_start, duration=sup.duration)
                    .load_audio()
                )
                if len(agent_audio.shape) > 1:
                    agent_audio = agent_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(agent_audio))
                new_source_audio[start_sample:actual_end] = agent_audio[: actual_end - start_sample]

        source_buffer = BytesIO()
        sf.write(source_buffer, new_source_audio, cut.sampling_rate, format='wav')
        source_buffer.seek(0)

        new_source_recording = Recording(
            id=f"{cut.id}_swapped_source",
            sampling_rate=cut.sampling_rate,
            num_samples=len(new_source_audio),
            duration=total_duration,
            sources=[AudioSource(type="memory", channels=[0], source=source_buffer.getvalue())],
        )

        swapped_cut = MonoCut(
            id=f"{cut.id}_swapped",
            start=0,
            duration=total_duration,
            channel=0,
            supervisions=adjusted_supervisions,
            recording=new_source_recording,
            custom={
                **cut.custom,
                'total_turns': len(adjusted_supervisions),
                'role_swapped': True,
            },
        )

        return swapped_cut

    def _has_valid_input(self, cut: Cut) -> bool:
        return any(s.text.strip() for s in cut.supervisions if s.speaker in self.input_roles)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    bos_id: int = None,
    eos_id: int = None,
    pad_id: int = None,
    word_align_position: str = 'left',
    remove_timestamps: bool = False,
    prepend_word_space: bool = True,
    skip_eos: bool = False,
    agent_token_channel: torch.Tensor = None,
    agent_token_channel_lengths: torch.Tensor = None,
    agent_eos_id: int = None,
    use_numbers_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = []
    for cut_idx, c in enumerate(cuts):
        token_seq = _build_token_channel(
            c,
            tokenizer=tokenizer,
            frame_length=frame_length,
            roles=roles,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            word_align_position=word_align_position,
            remove_timestamps=remove_timestamps,
            prepend_word_space=prepend_word_space,
            skip_eos=skip_eos,
            cut_agent_token_channel=agent_token_channel[cut_idx] if agent_token_channel is not None else None,
            cut_agent_token_channel_length=agent_token_channel_lengths[cut_idx] if agent_token_channel_lengths is not None else None,
            agent_eos_id=agent_eos_id,
            use_numbers_norm=use_numbers_norm,
        )
        tokens.append(token_seq)
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def collate_system_prompt(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    bos_id: int = None,
    eos_id: int = None,
    pad_id: int = None,
    add_mcq_prompt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate system prompts from cuts.

    Prompt selection priority:
      1. Per-cut custom prompt (cut.custom['system_prompt'])
      2. MCQ training cut -> THINK prompt for think-cuts, NOTHINK prompt for others
      3. MCQ validation cut (when add_mcq_prompt=True) -> THINK prompt
      4. No prompt (empty tensor)
    """
    tokens = []
    for c in cuts:
        prompt_text = None

        if c.custom and c.custom.get("system_prompt", None):
            prompt_text = c.custom["system_prompt"]
        elif add_mcq_prompt and is_mcq_cut_train(c):
            if is_mcq_cut_train(c, check_think=True):
                prompt_text = MCQ_SYSTEM_PROMPT_THINK
            else:
                prompt_text = MCQ_SYSTEM_PROMPT_NOTHINK
        elif add_mcq_prompt and is_mcq_cut_val(c):
            prompt_text = MCQ_SYSTEM_PROMPT_THINK

        if prompt_text:
            tokens.append(
                torch.as_tensor(
                    [bos_id] + tokenizer.text_to_ids(prompt_text) + [eos_id], dtype=torch.long
                )
            )
        else:
            tokens.append(torch.as_tensor([], dtype=torch.long))

    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def _build_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
    bos_id: int = None,
    eos_id: int = None,
    word_align_position: str = 'left',
    remove_timestamps: bool = False,
    prepend_word_space: bool = True,
    skip_eos: bool = False,
    cut_agent_token_channel: torch.Tensor = None,
    cut_agent_token_channel_length: torch.Tensor = None,
    agent_eos_id: int = None,
    eos_offset_frames: int = 8,
    use_numbers_norm: bool = False,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
                )
                continue
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            available_frames_for_text = eospos - pos

            text_ids = torch.as_tensor(
                [bos_id]
                + _text_to_ids(
                    supervision.text,
                    tokenizer,
                    available_frames_for_text=available_frames_for_text,
                    word_align_position=word_align_position,
                    remove_timestamps=remove_timestamps,
                    prepend_word_space=prepend_word_space,
                    pad_id=pad_id,
                    use_numbers_norm=use_numbers_norm,
                )
            )

            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                text_ids = torch.tensor([], dtype=torch.long)

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = pos + len(text_ids)

            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e
            
            if not skip_eos:
                # place agent eos in the agent token channel
                if cut_agent_token_channel is not None:
                    assert agent_eos_id is not None, "Agent EOS ID is not set"
                    user_bospos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
                    agent_eospos = user_bospos + eos_offset_frames
                    if agent_eospos < cut_agent_token_channel_length.item():
                        cut_agent_token_channel[agent_eospos] = agent_eos_id
                    else:
                        logging.warning(f"Agent EOS position {agent_eospos} is out of bounds for agent token channel {cut_agent_token_channel.shape}")

                # place user eos in the user token channel
                if eospos < len(tokens) and eos_id is not None:
                    tokens[eospos] = eos_id

    return tokens


def _text_to_ids(
    text: str,
    tokenizer: TokenizerSpec,
    _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
    available_frames_for_text=None,
    word_align_position='left',
    remove_timestamps=False,
    prepend_word_space=True,
    pad_id: int = None,
    use_numbers_norm: bool = False,
):
    if not remove_timestamps and re.compile(_TIMESTAMP_PATTERN_STR).search(text):
        # Timestamp-aligned path: normalization is skipped because some expansions
        # change word count (e.g. "$3.50" → 4 words), breaking word-to-timestamp pairing.
        text_ids = _text_with_timestamps_to_ids(
            text,
            tokenizer,
            _TIMESTAMP_PATTERN_STR,
            available_frames_for_text,
            word_align_position,
            prepend_word_space=prepend_word_space,
            pad_id=pad_id,
        )
    else:
        _TIMESTAMP_PATTERN = re.compile(_TIMESTAMP_PATTERN_STR)
        text = _TIMESTAMP_PATTERN.sub("", text)
        text = " ".join(text.strip().split())
        if use_numbers_norm:
            text = normalize_numbers(text)
        text_ids = tokenizer.text_to_ids(text)
    return text_ids


def _text_with_timestamps_to_ids(
    text: str,
    tokenizer: TokenizerSpec,
    _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
    available_frames_for_text=None,
    word_align_position='left',
    prepend_word_space=True,
    pad_id: int = None,
) -> list[int]:
    text_ids, start_times, end_times, word_lens = _extract_text_and_time_tokens(
        text,
        tokenizer,
        _TIMESTAMP_PATTERN_STR,
        prepend_word_space=prepend_word_space,
    )
    text_ids_with_timestamps = _expand_text_with_timestamps_and_word_lengths(
        text_ids,
        word_lens,
        start_times,
        end_times,
        available_frames_for_text,
        frame_rate=0.08,
        pad_id=pad_id,
        word_align_position=word_align_position,
    )
    return text_ids_with_timestamps


def _extract_text_and_time_tokens(
    text, tokenizer: TokenizerSpec, _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>", prepend_word_space=True
):
    time_tokens = re.findall(_TIMESTAMP_PATTERN_STR, text)
    start_time = [int(time_tokens[i]) for i in range(0, len(time_tokens), 2)]
    end_time = [int(time_tokens[i]) for i in range(1, len(time_tokens), 2)]
    words = re.sub(_TIMESTAMP_PATTERN_STR, '', text).split()
    text_ids = []
    word_lens = []
    for i, word in enumerate(words):
        word_with_space = word if i == 0 or not prepend_word_space else ' ' + word
        word_ids = tokenizer.text_to_ids(word_with_space)
        word_len = len(word_ids)
        text_ids.extend(word_ids)
        word_lens.append(word_len)
    return text_ids, start_time, end_time, word_lens


def _expand_text_with_timestamps_and_word_lengths(
    text_ids,
    word_lens,
    start_time,
    end_time,
    available_frames_for_text,
    frame_rate=0.08,
    pad_id=None,
    word_align_position='left',
):
    def discretize_time(start_token, speech_frame_rate=0.08, timestamp_frame_rate=0.08):
        return int(start_token * timestamp_frame_rate / speech_frame_rate)

    if pad_id is None:
        raise ValueError("pad_id must be provided.")

    max_length = available_frames_for_text
    text_ids_with_timestamps = [pad_id] * max_length

    cur_word_idx = 0
    for word_idx, word_len in enumerate(word_lens):
        start_idx = discretize_time(start_time[word_idx], speech_frame_rate=frame_rate)
        end_idx = discretize_time(end_time[word_idx], speech_frame_rate=frame_rate)
        if word_align_position == 'left':
            end_idx = min(start_idx + word_len, end_idx)
        elif word_align_position == 'right':
            start_idx = max(start_idx, end_idx - word_len)
        else:
            raise ValueError(f"Unknown word_align_position: {word_align_position}")

        word_ids = text_ids[cur_word_idx : cur_word_idx + word_len]

        for i in range(start_idx, end_idx + 1):
            if i - start_idx < len(word_ids) and i < max_length:
                token_id = word_ids[i - start_idx]
                text_ids_with_timestamps[i] = token_id

        cur_word_idx += word_len

    return text_ids_with_timestamps
