"""
Classify the end-of-utterance (EoU) audio as: good (natural ending), cutoff (abrupt
ending), silence (long trailing region that is quiet), or noise (significant trailing
region with high energy).

Uses Wav2Vec2 forced alignment against the known target text to find the end of the
speech, then applies simple threshold rules based on the trailing duration, last token
duration and confidence, and relative RMS energy.

Usage:
    from eou_classifier import EoUClassifier

    classifier = EoUClassifier()  # loads model once
    result = classifier.classify("output.wav", "Hello world.")
    print(result.eou_type, result.trailing_duration)
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Union

import librosa
import numpy as np
import torch
import torchaudio.functional as taf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SR = 16000


class EoUType(StrEnum):
    GOOD = "good"  # natural ending
    CUTOFF = "cutoff"  # speech ends abruptly
    SILENCE = "silence"  # long trailing region with near-zero energy
    NOISE = "noise"  # significant trailing region with high energy relative to speech


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

    def _text_to_tokens(self, text: str) -> list[int]:
        text = text.upper().strip()
        tokens = []
        for i, word in enumerate(text.split()):
            if i > 0:
                tokens.append(self.vocab["|"])
            for char in word:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
        return tokens

    def _find_speech_end(self, audio: np.ndarray, text: str) -> dict:
        """Run forced alignment and return speech boundary info."""
        input_values = self.processor(audio, return_tensors="pt", sampling_rate=self.sr).input_values

        with torch.no_grad():
            logits = self.model(input_values).logits[0]

        log_probs = torch.log_softmax(logits, dim=-1)
        n_frames = len(logits)
        frame_duration = len(audio) / n_frames / self.sr

        target_tokens = self._text_to_tokens(text)
        fa_ids, fa_scores = taf.forced_align(
            log_probs.unsqueeze(0),
            torch.tensor([target_tokens]),
            blank=self.blank_id,
        )
        aligned_ids = fa_ids[0].numpy()
        scores = torch.exp(fa_scores[0]).numpy()

        segments: list[TokenSegment] = []
        cur_id = -1
        seg_start = 0
        for i, aid in enumerate(aligned_ids):
            tid = int(aid)
            if tid == self.blank_id:
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
            # else: same non-blank token continues
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

        last_idx = segments.index(last_speech)
        if last_idx > 0:
            last_token_gap = last_speech.start - segments[last_idx - 1].end
        else:
            last_token_gap = last_speech.start

        last_two_alnum = [s for s in segments if s.token.isalnum()][-2:]
        last_two_avg = float(np.mean([s.confidence for s in last_two_alnum]))

        return {
            "speech_end": last.end,  # + 0.05, # add 50ms of tolerance
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
        Classify the end-of-utterance quality of a TTS audio sample.

        Args:
            audio: Path to a WAV file, or a numpy array of audio samples at self.sr.
            text: The target text that was supposed to be spoken.

        Returns:
            EoUClassification with the predicted eou_type and supporting features.
        """
        if isinstance(audio, np.ndarray):
            samples = audio
        else:
            samples, _ = librosa.load(audio, sr=self.sr)

        audio_dur = len(samples) / self.sr
        info = self._find_speech_end(samples, text)

        speech_end = info["speech_end"]
        trailing = audio_dur - speech_end
        last_letter_pad = 0.15
        trail_start = int((speech_end + last_letter_pad) * self.sr)
        trailing_audio = samples[trail_start:]
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

        # if trailing < 0.06 and (last_dur < 0.025 and last_conf < 0.15):
        # if trailing < 0.06 and (last_gap < 0.1 and last_conf < 0.15): short trail and not due to gap
        conf_threshold = 0.07
        # short tail with low confidence and not due to gap (which could indicate noise) --> cutoff
        if trailing < 0.1 and last_conf < conf_threshold and not last_gap > 0.4:
            # speech ends abruptly, with a very short last token or low confidence --> cutoff
            eou_type = EoUType.CUTOFF
        # long noisy tail OR odd gap --> noisy
        elif (trailing > 0.25 and trail_rms_ratio > 0.3) or (last_gap > 0.4 and last_conf < conf_threshold):
            # significant trailing region with high energy relative to speech --> noise
            eou_type = EoUType.NOISE
        # very long trailing region with near-zero energy --> silence
        elif trailing > 1.0:  # and trail_rms_ratio < 0.10:
            eou_type = EoUType.SILENCE
        else:
            # everything else (moderate trailing, natural energy decay) --> good
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
