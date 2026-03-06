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

    # Single-sample inference
    result = classifier.classify("output.wav", "Hello world.")
    print(result.eou_type, result.trailing_duration)

    # Batched inference (same outputs, better throughput)
    results = classifier.classify_batch([
        ("output1.wav", "Hello world."),
        ("output2.wav", "Goodbye."),
    ])
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
    repeatedly to process files without reloading, or `classify_batch()`
    for batched inference with better throughput.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", sr: int = SR, device: str | None = None):
        self.sr = sr
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

    def _extract_eou_features(
        self,
        log_probs: torch.Tensor,
        token_ids_with_blanks: list[int],
        alignment_path: list[int],
    ) -> dict:
        """Extract end-of-utterance features from per-frame log_probs and a Viterbi alignment path.

        Args:
            log_probs: (T, V) log-probability tensor for a single sample.
            token_ids_with_blanks: Interleaved-blank token sequence.
            alignment_path: Viterbi path indices into token_ids_with_blanks.

        Returns:
            Dict with speech_end, last_token_*, token_segments, etc.
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
        # same token into TokenSegment objects. Transitions: blank-to-token (start new
        # segment), token-to-blank (end segment), token-to-different token (end old,
        # start new).
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

    def _classify_from_alignment(self, samples: np.ndarray, text: str, info: dict) -> EoUClassification:
        """Apply the EoU decision tree given audio samples and forced-alignment info."""
        audio_dur = len(samples) / self.sr
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
            trail_rms_ratio = float(rms_trail / (rms_full + 1e-10))
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
        # Long noisy tail OR a gap between last two segments and low confidence --> noisy
        elif (trailing > 0.15 and trail_rms_ratio > 0.4) or (last_gap > 0.4 and last_conf < 0.15):
            eou_type = EoUType.NOISE
        # Long tail without much energy (or it would be captured by the previous condition) --> silence
        elif trailing > 1.4:
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
        return self.classify_batch([(audio, text)])[0]

    def _forced_align_batch(self, audios: list[np.ndarray], texts: list[str]) -> list[dict]:
        """Run forced alignment on a batch.

        Args:
            audios: List of 1-D numpy audio arrays at self.sr.
            texts: Corresponding transcripts.

        Returns:
            List of alignment-info dicts (same format as _forced_align).
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

        log_probs_all = torch.log_softmax(logits, dim=-1)

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

        log_probs_padded = log_probs_all.clone()
        for i in range(B):
            if feat_lengths[i] < max_feat_len:
                log_probs_padded[i, feat_lengths[i] :, :] = VITERBI_PAD

        y_batch = torch.full((B, U_max), V, dtype=torch.int64, device=self.device)
        for i, tids in enumerate(all_token_ids_with_blanks):
            y_batch[i, : len(tids)] = torch.tensor(tids, dtype=torch.int64, device=self.device)

        T_batch = torch.tensor(feat_lengths, device=self.device)
        U_batch = torch.tensor(U_lengths, device=self.device)

        alignments = viterbi_decoding(log_probs_padded, y_batch, T_batch, U_batch)

        # --- Extract EoU features ---
        results: list[dict] = []
        for i in range(B):
            sample_log_probs = log_probs_all[i, : feat_lengths[i]]
            info = self._extract_eou_features(sample_log_probs, all_token_ids_with_blanks[i], alignments[i])
            results.append(info)

        return results

    def classify_batch(
        self,
        items: list[tuple[Union[str, np.ndarray], str]],
    ) -> list[EoUClassification]:
        """
        Classifies a batch of utterances.

        Args:
            items: List of (audio, text) pairs. Audio can be a file path or numpy array.

        Returns:
            List of EoUClassification results, one per input item.
        """
        audios: list[np.ndarray] = []
        for audio, _text in items:
            if isinstance(audio, np.ndarray):
                audios.append(audio)
            else:
                samples, _ = librosa.load(audio, sr=self.sr)
                audios.append(samples)
        texts = [text for _, text in items]

        infos = self._forced_align_batch(audios, texts)

        results: list[EoUClassification] = []
        for i in range(len(audios)):
            results.append(self._classify_from_alignment(audios[i], texts[i], infos[i]))

        return results


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
