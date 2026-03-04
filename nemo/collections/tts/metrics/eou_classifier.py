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
    result = classifier.classify("output.wav", "Hello world.")
    print(result.eou_type, result.trailing_duration)
"""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Union

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from nemo.collections.asr.parts.utils.aligner_utils import viterbi_decoding

SR = 16000

# Spelling patterns at end of a word that produce sibilant fricatives
# (/s/, /z/, /ʃ/, /tʃ/) whose noise-like energy extends past the
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
class EoUClassification:
    eou_type: EoUType
    speech_end: float  # seconds
    audio_duration: float  # seconds
    trailing_duration: float  # seconds
    trail_rms_ratio: float
    last_token_duration: float
    last_token_confidence: float
    last_token: str
    last_token_gap: float  # blank gap (seconds) between last and second-to-last speech token
    last_two_phoneme_avg_confidence: float  # average confidence of last two alphanumeric tokens
    token_segments: list[TokenSegment] = field(default_factory=list)


class EoUClassifier:
    """
    Classifies end-of-utterance (EoU) audio as good (natural ending), cutoff, silence, or noise.

    The model is loaded once at construction time. Call `classify()`
    repeatedly to process files without reloading.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", sr: int = SR):
        self.sr = sr
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
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

    def _forced_align(self, audio: np.ndarray, text: str) -> dict:
        """Run forced alignment and return speech boundary info."""
        # Tokenize audio into Wav2Vec2 input features
        input_values = self.processor(audio, return_tensors="pt", sampling_rate=self.sr).input_values

        # Forward pass through the CTC model to get per-frame logits
        with torch.no_grad():
            logits = self.model(input_values).logits[0]

        log_probs = torch.log_softmax(logits, dim=-1)
        frame_duration = self.frame_duration

        # Convert target text to CTC token IDs
        target_tokens = self._text_to_tokens(text)

        # Build interleaved-blank sequence: [blank, T1, blank, T2, blank, ...]
        token_ids_with_blanks = [self.blank_id]
        for tok in target_tokens:
            token_ids_with_blanks.extend([tok, self.blank_id])

        T = log_probs.shape[0]
        y = torch.tensor([token_ids_with_blanks], dtype=torch.int64)

        alignment = viterbi_decoding(
            log_probs.unsqueeze(0),
            y,
            torch.tensor([T]),
            torch.tensor([len(token_ids_with_blanks)]),
        )
        alignment_path = alignment[0]

        # Reconstruct per-frame token IDs and confidence scores from the
        # alignment path, matching the format that torchaudio.forced_align returns.
        aligned_ids = np.array([token_ids_with_blanks[s] for s in alignment_path])
        scores = torch.exp(
            log_probs[torch.arange(T), torch.tensor([token_ids_with_blanks[s] for s in alignment_path])]
        ).numpy()

        # Walk through the frame-level alignment and merge consecutive
        # frames of the same token into TokenSegment objects.
        # Transitions: blank-to-token (start new segment), token-to-blank (end segment),
        # token-to-different token (end old, start new).
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
            return {
                "speech_end": 0.0,
                "last_token_duration": 0.0,
                "last_token_confidence": 0.0,
                "last_token": "",
                "last_token_gap": 0.0,
                "last_two_phoneme_avg_confidence": 0.0,
                "token_segments": [],
            }

        last = segments[-1]

        # Skip trailing punctuation/non-letter tokens for cutoff analysis,
        # since they don't correspond to real speech sounds and get
        # unreliably short durations from forced alignment.
        last_speech = last
        for seg in reversed(segments):
            if seg.token.isalnum():
                last_speech = seg
                break

        # Measure the blank gap between the last speech token and its predecessor.
        # A large gap can indicate noise or misalignment before the final sound.
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

        return {
            "speech_end": last.end,
            "last_token_duration": last_speech.duration,
            "last_token_confidence": last_speech.confidence,
            "last_token": last_speech.token,
            "last_token_gap": last_token_gap,
            "last_two_phoneme_avg_confidence": last_two_avg,
            "token_segments": segments,
        }

    def classify(
        self,
        audio: Union[str, np.ndarray],
        text: str,
    ) -> EoUClassification:
        """
        Classify the end-of-utterance quality of utterance audio.

        Args:
            audio: Path to a WAV file, or a numpy array of audio samples at self.sr.
            text: The target text that was supposed to be spoken.

        Returns:
            EoUClassification with the predicted eou_type and supporting features.
        """
        # Accept either a file path or a pre-loaded numpy array
        if isinstance(audio, np.ndarray):
            samples = audio
        else:
            samples, _ = librosa.load(audio, sr=self.sr)

        audio_dur = len(samples) / self.sr
        # Run forced alignment and collect information about speech segments
        info = self._forced_align(samples, text)

        speech_end = info["speech_end"]
        trailing = audio_dur - speech_end
        last_letter_pad = 0.15 if _ends_with_sibilant(text) else 0.1
        trail_start = int((speech_end + last_letter_pad) * self.sr)
        trailing_audio = samples[trail_start:]

        # Compute RMS energy ratio between the trailing region and the full
        # utterance — a high ratio means the tail is loud
        if len(trailing_audio) > 0:
            rms_trail = np.sqrt(np.mean(trailing_audio**2))
            rms_full = np.sqrt(np.mean(samples**2))
            trail_rms_ratio = rms_trail / (rms_full + 1e-10)
        else:
            trail_rms_ratio = 0.0

        last_dur = info["last_token_duration"]
        last_conf = info["last_token_confidence"]
        if last_conf < 0.01:
            last_conf = info["last_two_phoneme_avg_confidence"]
        last_tok = info["last_token"]
        last_gap = info["last_token_gap"]
        last_two_avg = info["last_two_phoneme_avg_confidence"]
        token_segments = info["token_segments"]

        # --- Decision tree for EoU classification ---
        conf_threshold = 0.07
        # Short tail with low confidence and not due to gap (which could indicate noise) --> cutoff
        if trailing < 0.1 and last_conf < conf_threshold and not last_gap > 0.4:
            eou_type = EoUType.CUTOFF
        # Long noisy tail OR a gap between last two segements and low confidence --> noisy
        elif (trailing > 0.15 and trail_rms_ratio > 0.4) or (last_gap > 0.4 and last_conf < 0.15):
            eou_type = EoUType.NOISE
        # Long tail without much energy (or it would captured by the previous condition) --> silence
        elif trailing > 1.0:
            eou_type = EoUType.SILENCE
        else:
            # everything else --> good
            eou_type = EoUType.GOOD

        return EoUClassification(
            eou_type=eou_type,
            speech_end=speech_end,
            audio_duration=audio_dur,
            trailing_duration=trailing,
            trail_rms_ratio=trail_rms_ratio,
            last_token_duration=last_dur,
            last_token_confidence=last_conf,
            last_token=last_tok,
            last_token_gap=last_gap,
            last_two_phoneme_avg_confidence=last_two_avg,
            token_segments=token_segments,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify end-of-utterance audio quality")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("text", help="Target text")
    args = parser.parse_args()

    classifier = EoUClassifier()
    result = classifier.classify(args.audio, args.text)
    print(f"eou_type:           {result.eou_type}")
    print(f"speech_end:         {result.speech_end:.3f}s")
    print(f"audio_duration:     {result.audio_duration:.3f}s")
    print(f"trailing_duration:  {result.trailing_duration:.3f}s")
    print(f"trail_rms_ratio:    {result.trail_rms_ratio:.4f}")
    print(f"last_token_dur:     {result.last_token_duration:.3f}s")
    print(f"last_token_conf:    {result.last_token_confidence:.3f}")
    print(f"last_token_gap:     {result.last_token_gap:.3f}s")
    print(f"last_2_ph_avg_conf: {result.last_two_phoneme_avg_confidence:.3f}")
    print(f"last_token:         {result.last_token!r}")
    print(f"\nToken segments ({len(result.token_segments)}):")
    for seg in result.token_segments:
        print(f"  {seg.token!r:<6} {seg.start:.3f}-{seg.end:.3f}s  dur={seg.duration:.3f}s  conf={seg.confidence:.3f}")
