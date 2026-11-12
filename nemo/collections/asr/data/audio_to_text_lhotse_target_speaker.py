# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lhotse dataset for enrollment-conditioned target-speaker ASR.

Each sampled input cut is the *only* supervised target utterance.  Background
cuts are selected separately, stripped of supervisions, and mixed on the fly.
This deliberately differs from the multitalker ASR dataset, whose target is a
random speaker from an already-supervised multi-speaker recording.
"""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset import AudioSamples
from omegaconf import ListConfig

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.data.lhotse.audio_loading import LhotseAudioLoadingDatasetMixin
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseTargetSpeakerBpeDataset(LhotseAudioLoadingDatasetMixin, torch.utils.data.Dataset):
    """Create target-plus-background mixtures from clean supervised cuts.

    ``cuts`` passed to :meth:`__getitem__` must be clean, single-speaker target
    cuts.  The background and enrollment pools are loaded from separate config
    paths, allowing the sampler to keep its usual duration-based batching over
    the clean core utterances.

    Required config keys are ``background_cuts_path`` and
    ``enrollment_embeddings_path``. ``enrollment_cuts_path`` defaults to the
    background pool path.  The embedding file is a torch mapping from clean cut
    id to a 1-D Titanet embedding.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'enrollment_embeddings': NeuralType(('B', 'D'), LabelsType()),
            'target_activity': NeuralType(('B', 'T'), LabelsType()),
            'target_activity_length': NeuralType(tuple('B'), LengthsType()),
            'target_is_absent': NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(self, cfg, tokenizer: TokenizerSpec):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=cfg.get('fault_tolerant_audio_loading', False))

        background_path = cfg.get('background_cuts_path')
        if not background_path:
            raise ValueError("Target-speaker training requires train_ds.background_cuts_path.")
        enrollment_path = cfg.get('enrollment_cuts_path', background_path)
        embeddings_path = cfg.get('enrollment_embeddings_path')
        if not embeddings_path:
            raise ValueError("Target-speaker training requires train_ds.enrollment_embeddings_path.")

        self.background_by_speaker = self._index_cuts_by_speaker(guess_parse_cutset(background_path))
        self.enrollment_by_speaker = self._index_cuts_by_speaker(guess_parse_cutset(enrollment_path))
        self.enrollment_embeddings = self._load_embeddings(embeddings_path)
        self.embedding_dim = next(iter(self.enrollment_embeddings.values())).numel()

        self.num_background_clips = cfg.get('num_background_clips', 1)
        self.background_snr = tuple(float(x) for x in cfg.get('background_snr', (-5.0, 5.0)))
        if len(self.background_snr) != 2 or self.background_snr[0] > self.background_snr[1]:
            raise ValueError("background_snr must contain [minimum, maximum] dB.")
        self.absent_target_prob = float(cfg.get('absent_target_prob', 0.1))
        if not 0.0 <= self.absent_target_prob < 1.0:
            raise ValueError("absent_target_prob must be in [0, 1).")
        self.num_sample_per_mel_frame = int(cfg.get('num_sample_per_mel_frame', 160))
        self.num_mel_frame_per_asr_frame = int(cfg.get('num_mel_frame_per_asr_frame', 8))
        if self.num_sample_per_mel_frame <= 0 or self.num_mel_frame_per_asr_frame <= 0:
            raise ValueError("Frame-stride values must be positive integers.")

    @staticmethod
    def _speaker_id(cut) -> str:
        speakers = {supervision.speaker for supervision in cut.supervisions if supervision.speaker is not None}
        if len(speakers) != 1:
            raise ValueError(
                "Target-speaker ASR requires exactly one speaker per clean cut; "
                f"cut {cut.id!r} has {sorted(speakers)}."
            )
        return next(iter(speakers))

    @classmethod
    def _index_cuts_by_speaker(cls, cuts) -> Dict[str, list]:
        result: Dict[str, list] = {}
        for cut in cuts:
            if isinstance(cut, MixedCut):
                raise ValueError("Enrollment and background pools must contain clean cuts, not MixedCut instances.")
            result.setdefault(cls._speaker_id(cut), []).append(cut)
        if not result:
            raise ValueError("Target-speaker cut pool is empty.")
        return result

    @staticmethod
    def _load_embeddings(path: str) -> Dict[str, torch.Tensor]:
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("enrollment_embeddings_path must contain a non-empty mapping from cut id to embedding.")
        embeddings = {}
        for cut_id, value in payload.items():
            embedding = torch.as_tensor(value, dtype=torch.float32).flatten()
            if embedding.numel() == 0:
                raise ValueError(f"Enrollment embedding for {cut_id!r} is empty.")
            embeddings[str(cut_id)] = F.normalize(embedding, dim=0)
        dims = {embedding.numel() for embedding in embeddings.values()}
        if len(dims) != 1:
            raise ValueError(f"Enrollment embeddings must have one common width, got {sorted(dims)}.")
        return embeddings

    @staticmethod
    def _rng_for_cut(cut_id: str) -> random.Random:
        # ``torch.initial_seed`` is worker-specific; no global RNG is touched.
        seed_material = f"{torch.initial_seed()}:{cut_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        return random.Random(seed)

    def _sample_enrollment(self, speaker: str, target_id: str, rng: random.Random) -> Tuple[str, torch.Tensor]:
        choices = [cut for cut in self.enrollment_by_speaker.get(speaker, []) if cut.id != target_id]
        choices = [cut for cut in choices if cut.id in self.enrollment_embeddings]
        if not choices:
            raise ValueError(
                f"No distinct embedded enrollment cut is available for target cut {target_id!r} speaker {speaker!r}."
            )
        enrollment_cut = rng.choice(choices)
        return enrollment_cut.id, self.enrollment_embeddings[enrollment_cut.id]

    def _sample_backgrounds(self, target_speaker: str, rng: random.Random) -> list:
        candidates = [
            cut for speaker, cuts in self.background_by_speaker.items() if speaker != target_speaker for cut in cuts
        ]
        if not candidates:
            raise ValueError(f"No different-speaker background clip is available for speaker {target_speaker!r}.")
        count = self.num_background_clips
        if isinstance(count, (list, tuple, ListConfig)):
            if len(count) != 2:
                raise ValueError("num_background_clips must be an integer or [minimum, maximum].")
            if int(count[0]) > int(count[1]):
                raise ValueError("num_background_clips minimum cannot exceed its maximum.")
            count = rng.randint(int(count[0]), int(count[1]))
        count = int(count)
        if count < 1:
            raise ValueError("num_background_clips must be at least one.")
        return [rng.choice(candidates) for _ in range(count)]

    @staticmethod
    def _without_supervisions(cut):
        """Copy a background cut without text/speaker labels before mixing."""
        background = deepcopy(cut)
        background.supervisions = []
        return background

    def _mix(self, core, backgrounds: Sequence, rng: random.Random, target_absent: bool):
        if target_absent:
            mixed = self._without_supervisions(backgrounds[0])
            remaining_backgrounds = backgrounds[1:]
        else:
            mixed = core
            remaining_backgrounds = backgrounds

        for background in remaining_backgrounds:
            background = self._without_supervisions(background)
            max_offset = max(0.0, mixed.duration - background.duration)
            offset = rng.uniform(0.0, max_offset)
            snr = rng.uniform(*self.background_snr)
            mixed = mixed.mix(background, snr=snr, offset_other_by=offset)
        return mixed

    def _tokenize(self, core) -> torch.Tensor:
        tokens = [
            torch.as_tensor(
                supervision.tokens
                if hasattr(supervision, 'tokens')
                else self.tokenizer(supervision.text or "", supervision.language)
            )
            for supervision in core.supervisions
        ]
        if not tokens:
            raise ValueError(f"Core target cut {core.id!r} has no supervision text.")
        return torch.cat(tokens, dim=0)

    def _activity_target(self, core, mixed, target_absent: bool) -> torch.Tensor:
        samples_per_asr_frame = self.num_sample_per_mel_frame * self.num_mel_frame_per_asr_frame
        mixed_frames = (mixed.num_samples + samples_per_asr_frame - 1) // samples_per_asr_frame
        target = torch.zeros(mixed_frames, dtype=torch.float32)
        if not target_absent:
            core_frames = (core.num_samples + samples_per_asr_frame - 1) // samples_per_asr_frame
            target[: min(core_frames, mixed_frames)] = 1.0
        return target

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        mixtures, tokens, enrollments, activities, absent_flags = [], [], [], [], []
        for core in cuts:
            if isinstance(core, MixedCut):
                raise ValueError(
                    "Target-speaker dataset expects clean core cuts. Disable generic noise_path mixing "
                    "for this dataset."
                )
            speaker = self._speaker_id(core)
            rng = self._rng_for_cut(core.id)
            _, enrollment = self._sample_enrollment(speaker, core.id, rng)
            target_absent = rng.random() < self.absent_target_prob
            backgrounds = self._sample_backgrounds(speaker, rng)
            mixed = self._mix(core, backgrounds, rng, target_absent)

            mixtures.append(mixed)
            tokens.append(torch.empty(0, dtype=torch.long) if target_absent else self._tokenize(core).long())
            enrollments.append(enrollment)
            activities.append(self._activity_target(core, mixed, target_absent))
            absent_flags.append(target_absent)

        audio, audio_lens, _ = self.load_audio_with_cuts(CutSet.from_cuts(mixtures))
        token_lens = torch.tensor([token.numel() for token in tokens], dtype=torch.long)
        max_tokens = max(1, int(token_lens.max().item()))
        padded_tokens = torch.zeros(len(tokens), max_tokens, dtype=torch.long)
        for index, token in enumerate(tokens):
            padded_tokens[index, : token.numel()] = token

        activity_lens = torch.tensor([activity.numel() for activity in activities], dtype=torch.long)
        padded_activity = torch.zeros(len(activities), int(activity_lens.max().item()), dtype=torch.float32)
        for index, activity in enumerate(activities):
            padded_activity[index, : activity.numel()] = activity

        return (
            audio,
            audio_lens,
            padded_tokens,
            token_lens,
            torch.stack(enrollments),
            padded_activity,
            activity_lens,
            torch.tensor(absent_flags, dtype=torch.bool),
        )
