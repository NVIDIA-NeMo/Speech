# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Classify the end-of-utterance (EoU) audio as: good (natural ending), cutoff (abrupt
ending), silence (long trailing region that is quiet), or noise (significant trailing
region with high energy).

Uses NeMo Forced Aligner's viterbi_decoding() for CTC forced alignment of audio to
transcript text with a Wav2Vec2 acoustic model.

Usage:
    from nemo.collections.tts.metrics.eou_classifier import EoUClassifier

    classifier = EoUClassifier()  # loads model once

    # Single-sample inference (file path — any sample rate is handled automatically)
    result = classifier.classify("output.wav", "Hello world.")
    print(result.eou_type, result.trailing_duration)

    # Single-sample inference (numpy array — sample_rate is required)
    result = classifier.classify(samples, "Hello world.", sample_rate=22050)

    # Batched inference (same outputs, better throughput)
    results = classifier.classify_batch([
        ("output1.wav", "Hello world."),
        ("output2.wav", "Goodbye."),
    ])
"""

import math
from dataclasses import dataclass, field

# StrEnum is part of enum in python >= 3.11, for backward compatibility
# to python < 3.11 we import StrEnum from strenum. Use-case: Huggignface
# demo works on python version 3.10
try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum
from typing import Union

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from nemo.collections.asr.parts.utils.aligner_utils import viterbi_decoding

# Spelling patterns at end of a word that produce sibilant fricatives
# (/s/, /z/, /ʃ/, /tʃ/) whose noise-like energy tends to extend past the
# forced-alignment boundary.
_SIBILANT_ENDINGS = ("sh", "ch", "s", "z", "x", "ce", "se", "ze")


def _ends_with_sibilant(text: str) -> bool:
    """Return True if the last word in *text* ends with a sibilant sound."""
    words = text.strip().rstrip(".,!?;:\"'").split()
    if not words:
        return False
    last_word = words[-1].lower()
    return last_word.endswith(_SIBILANT_ENDINGS)


class EoUType(StrEnum):
    GOOD = "good"  # natural ending
    CUTOFF = "cutoff"  # speech ends abruptly
    SILENCE = "silence"  # long trailing region with near-zero energy
    NOISE = "noise"  # significant trailing region with high energy relative to speech

    @classmethod
    def error_types(cls) -> tuple["EoUType", ...]:
        """All types that represent an error (everything except GOOD)."""
        return tuple(t for t in cls if t != cls.GOOD)


@dataclass
class TokenSegment:
    token: str
    start: float  # seconds
    end: float  # seconds
    duration: float  # seconds
    confidence: float


@dataclass
class AlignmentFeatures:
    """Information about the end of the utterance, derived from forced alignment."""

    speech_end: float  # seconds — end of the last aligned token
    last_token_duration: float  # seconds
    last_token_confidence: float
    last_token: str
    last_token_gap: float  # blank gap (seconds) between last and second-to-last speech token
    last_two_token_avg_confidence: float  # average confidence of last two alphanumeric tokens
    token_segments: list[TokenSegment] = field(default_factory=list)


@dataclass
class EoUClassification:
    """Classification of the end of the utterance along with associated metadata."""

    eou_type: EoUType
    alignment: AlignmentFeatures
    audio_duration: float  # seconds
    trailing_duration: float  # seconds
    trail_rms_ratio: float


class EoUClassifier:
    """
    Classifies end-of-utterance (EoU) audio as good (natural ending), cutoff, silence, or noise.

    The model is loaded once at construction time. Call `classify()`
    repeatedly to process files without reloading, or `classify_batch()`
    for batched inference with better throughput.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str | None = None):
        self.sr = 16000  # We will resample all inputs to this rate internally.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.blank_id = self.processor.tokenizer.pad_token_id
        self.vocab = self.processor.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.frame_duration = math.prod(self.model.config.conv_stride) / self.sr

    def _text_to_tokens(self, text: str) -> list[int]:
        # Wav2Vec2 uses uppercase characters; normalize to match its vocabulary
        text = text.upper().strip()
        tokens = []
        for i, word in enumerate(text.split()):
            # "|" is the word-boundary token in Wav2Vec2's CTC vocabulary
            if i > 0:
                tokens.append(self.vocab["|"])
            for char in word:
                # Skip characters not in vocab (punctuation, accents, etc.)
                if char in self.vocab:
                    tokens.append(self.vocab[char])
        return tokens

    def extract_alignment_features(
        self,
        log_probs: torch.Tensor,
        token_ids_with_blanks: list[int],
        alignment_path: list[int],
    ) -> AlignmentFeatures:
        """Extract alignment-derived end-of-utterance features from per-frame log_probs and a Viterbi alignment path.

        Args:
            log_probs: (T, V) log-probability tensor for a single sample.
            token_ids_with_blanks: Interleaved-blank token sequence.
            alignment_path: Viterbi path indices into token_ids_with_blanks.

        Returns:
            AlignmentFeatures with information about when the speech ended, what the last token was,
            what its confidence was, and detailed per-segment information.
        """
        T = log_probs.shape[0]
        frame_duration = self.frame_duration

        # Reconstruct per-frame token IDs and confidence scores from the
        # alignment path.
        aligned_ids = np.array([token_ids_with_blanks[s] for s in alignment_path])
        scores = (
            torch.exp(
                log_probs[
                    torch.arange(T, device=log_probs.device),
                    torch.tensor([token_ids_with_blanks[s] for s in alignment_path], device=log_probs.device),
                ]
            )
            .cpu()
            .numpy()
        )

        # Walk through the frame-level alignment and merge consecutive frames of the
        # same token into TokenSegment objects.
        segments: list[TokenSegment] = []
        cur_id = -1  # indicates that no segment is open
        seg_start = 0
        for i, aid in enumerate(aligned_ids):
            tid = int(aid)
            if tid == self.blank_id:
                # Blank frame: close the current segment if one is open
                if cur_id != -1:
                    seg_scores = scores[seg_start:i]
                    segments.append(
                        TokenSegment(
                            token=self.id_to_token.get(cur_id, f"<id:{cur_id}>"),
                            start=seg_start * frame_duration,
                            end=i * frame_duration,
                            duration=(i - seg_start) * frame_duration,
                            confidence=float(seg_scores.mean()),
                        )
                    )
                    cur_id = -1
            elif tid != cur_id:
                # New non-blank token: close previous segment (if any) and start a new one
                if cur_id != -1:
                    seg_scores = scores[seg_start:i]
                    segments.append(
                        TokenSegment(
                            token=self.id_to_token.get(cur_id, f"<id:{cur_id}>"),
                            start=seg_start * frame_duration,
                            end=i * frame_duration,
                            duration=(i - seg_start) * frame_duration,
                            confidence=float(seg_scores.mean()),
                        )
                    )
                cur_id = tid
                seg_start = i
            # else: same non-blank token continues — keep extending the segment
        # Flush the last open segment if the alignment ends on a non-blank token
        if cur_id != -1:
            seg_scores = scores[seg_start : len(aligned_ids)]
            segments.append(
                TokenSegment(
                    token=self.id_to_token.get(cur_id, f"<id:{cur_id}>"),
                    start=seg_start * frame_duration,
                    end=len(aligned_ids) * frame_duration,
                    duration=(len(aligned_ids) - seg_start) * frame_duration,
                    confidence=float(seg_scores.mean()),
                )
            )

        # No tokens were aligned — return zeroed-out defaults
        if not segments:
            return AlignmentFeatures(
                speech_end=0.0,
                last_token_duration=0.0,
                last_token_confidence=0.0,
                last_token="",
                last_token_gap=0.0,
                last_two_token_avg_confidence=0.0,
            )

        last = segments[-1]

        # Skip trailing punctuation/non-letter tokens for cutoff analysis, since they
        # don't correspond to real speech sounds and get unreliably short durations from
        # forced alignment.
        last_speech = last
        for seg in reversed(segments):
            if seg.token.isalnum():
                last_speech = seg
                break

        # Measure the blank gap between the last speech token and the preceding token. A
        # large gap (especially with low final token confidence) tends to indicate a
        # noisy ending where alignment has broken down.
        last_idx = segments.index(last_speech)
        if last_idx > 0:
            last_token_gap = last_speech.start - segments[last_idx - 1].end
        else:
            # First (and only) token — gap is measured from audio start
            last_token_gap = last_speech.start

        # Average confidence of the last two alphanumeric tokens;
        # used as a fallback when the single last-token confidence is near zero.
        last_two_alnum = [s for s in segments if s.token.isalnum()][-2:]
        last_two_avg = float(np.mean([s.confidence for s in last_two_alnum]))

        return AlignmentFeatures(
            speech_end=last.end,
            last_token_duration=last_speech.duration,
            last_token_confidence=last_speech.confidence,
            last_token=last_speech.token,
            last_token_gap=last_token_gap,
            last_two_token_avg_confidence=last_two_avg,
            token_segments=segments,
        )

    def classify_from_alignment(
        self, samples: np.ndarray, text: str, features: AlignmentFeatures
    ) -> EoUClassification:
        """Apply the EoU decision tree given audio samples and forced-alignment features."""
        audio_dur = len(samples) / self.sr
        trailing = audio_dur - features.speech_end
        last_letter_pad = 0.15 if _ends_with_sibilant(text) else 0.1
        trail_start = int((features.speech_end + last_letter_pad) * self.sr)
        trailing_audio = samples[trail_start:]

        if len(trailing_audio) > 0:
            rms_trail = np.sqrt(np.mean(trailing_audio**2))
            rms_full = np.sqrt(np.mean(samples**2))
            trail_rms_ratio = float(rms_trail / (rms_full + 1e-10))
        else:
            trail_rms_ratio = 0.0

        last_conf = features.last_token_confidence
        if last_conf < 0.01:
            last_conf = features.last_two_token_avg_confidence

        # --- Decision tree for EoU classification ---
        conf_threshold = 0.07
        if trailing < 0.1 and last_conf < conf_threshold and features.last_token_gap <= 0.4:
            eou_type = EoUType.CUTOFF
        elif (trailing > 0.2 and trail_rms_ratio > 0.4) or (features.last_token_gap > 0.4 and last_conf < 0.15):
            eou_type = EoUType.NOISE
        elif trailing > 1.4:
            eou_type = EoUType.SILENCE
        else:
            eou_type = EoUType.GOOD

        return EoUClassification(
            eou_type=eou_type,
            alignment=features,
            audio_duration=audio_dur,
            trailing_duration=trailing,
            trail_rms_ratio=trail_rms_ratio,
        )

    def classify(
        self,
        audio: Union[str, np.ndarray],
        text: str,
        sample_rate: int | None = None,
    ) -> EoUClassification:
        """
        Classify the end-of-utterance quality of utterance audio.

        Args:
            audio: Path to a WAV file, or a numpy array of audio samples.
            text: The target text that was supposed to be spoken.
            sample_rate: Required when `audio` is a numpy array. The audio will
                be resampled to 16 kHz internally. Ignored for file paths.

        Returns:
            EoUClassification with the predicted eou_type and supporting features.
        """
        return self.classify_batch([(audio, text)], sample_rate=sample_rate)[0]

    def _forced_align_batch(self, audios: list[np.ndarray], texts: list[str]) -> list[AlignmentFeatures]:
        """Run forced alignment on a batch.

        Args:
            audios: List of 1-D numpy audio arrays at self.sr.
            texts: Corresponding transcripts.

        Returns:
            List of AlignmentFeatures, one per input audio.
        """
        B = len(audios)

        # --- CNN feature extraction ---
        # We run the CNN feature extractor part of Wav2Vec2 at batch size 1 because its
        # outputs were found to be batch-size-dependent, likely due to the GroupNorm
        # layer being unable to ignore padding.
        cnn_outputs: list[torch.Tensor] = []
        for audio in audios:
            iv = self.processor(audio, return_tensors="pt", sampling_rate=self.sr).input_values.to(self.device)
            with torch.no_grad():
                feat = self.model.wav2vec2.feature_extractor(iv)  # (1, C, T_i)
                cnn_outputs.append(feat.squeeze(0))  # (C, T_i)

        # --- Pad CNN outputs and build attention mask ---
        feat_lengths = [f.shape[1] for f in cnn_outputs]
        max_feat_len = max(feat_lengths)
        C = cnn_outputs[0].shape[0]

        padded = torch.zeros(B, C, max_feat_len, device=self.device)
        attention_mask = torch.zeros(B, max_feat_len, dtype=torch.bool, device=self.device)
        for i, f in enumerate(cnn_outputs):
            padded[i, :, : feat_lengths[i]] = f
            attention_mask[i, : feat_lengths[i]] = True
        padded = padded.transpose(1, 2)  # (B, T_max, C)

        # --- Feature projection + transformer encoder + LM head (batched) ---
        with torch.no_grad():
            hidden, _ = self.model.wav2vec2.feature_projection(padded)
            encoder_out = self.model.wav2vec2.encoder(hidden, attention_mask=attention_mask)
            hidden = encoder_out[0]
            hidden = self.model.dropout(hidden)
            logits = self.model.lm_head(hidden)  # (B, T_max, V)

        log_probs_all = torch.log_softmax(logits, dim=-1)  # (B, T_max, V)

        # --- Batched Viterbi decoding ---
        V = log_probs_all.shape[-1]
        VITERBI_PAD = -3.4e38

        all_token_ids_with_blanks: list[list[int]] = []
        for text in texts:
            target_tokens = self._text_to_tokens(text)
            tids = [self.blank_id]
            for tok in target_tokens:
                tids.extend([tok, self.blank_id])
            all_token_ids_with_blanks.append(tids)

        U_lengths = [len(tids) for tids in all_token_ids_with_blanks]
        U_max = max(U_lengths)

        # Pad log probabilities with VITERBI_PAD to match max_feat_len
        log_probs_padded = log_probs_all.clone()
        for i in range(B):
            if feat_lengths[i] < max_feat_len:
                log_probs_padded[i, feat_lengths[i] :, :] = VITERBI_PAD

        # Pad y_batch with V to match U_max
        y_batch = torch.full((B, U_max), V, dtype=torch.int64, device=self.device)
        for i, tids in enumerate(all_token_ids_with_blanks):
            y_batch[i, : len(tids)] = torch.tensor(tids, dtype=torch.int64, device=self.device)

        T_batch = torch.tensor(feat_lengths, device=self.device)
        U_batch = torch.tensor(U_lengths, device=self.device)

        alignments = viterbi_decoding(log_probs_padded, y_batch, T_batch, U_batch)

        # --- Extract alignment features ---
        results: list[AlignmentFeatures] = []
        for i in range(B):
            sample_log_probs = log_probs_all[i, : feat_lengths[i]]
            alignment_info = self.extract_alignment_features(
                sample_log_probs, all_token_ids_with_blanks[i], alignments[i]
            )
            results.append(alignment_info)

        return results

    def classify_batch(
        self,
        items: list[tuple[Union[str, np.ndarray], str]],
        sample_rate: int | None = None,
    ) -> list[EoUClassification]:
        """
        Classifies a batch of utterances.

        Args:
            items: List of (audio, text) pairs. Audio can be a file path or numpy array.
            sample_rate: Required when any item's audio is a numpy array. All numpy
                arrays in the batch are assumed to share this rate and will be
                resampled to 16 kHz internally. Ignored for file paths.

        Returns:
            List of EoUClassification results, one per input item.
        """
        audios: list[np.ndarray] = []
        for audio, _text in items:
            if isinstance(audio, np.ndarray):
                if sample_rate is None:
                    raise ValueError(
                        "sample_rate is required when audio is a numpy array. "
                        "Pass the sample rate of the array so it can be resampled to 16 kHz."
                    )
                if sample_rate != self.sr:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sr)
                audios.append(audio)
            else:
                samples, _ = librosa.load(audio, sr=self.sr)
                audios.append(samples)
        texts = [text for _, text in items]

        infos = self._forced_align_batch(audios, texts)

        results: list[EoUClassification] = []
        for i in range(len(audios)):
            results.append(self.classify_from_alignment(audios[i], texts[i], infos[i]))

        return results
