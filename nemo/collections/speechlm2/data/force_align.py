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

import logging
import multiprocessing as mp
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lhotse import CutSet

# Use NeMo's force alignment utilities instead of torchaudio
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.utils.aligner_utils import (
    add_t_start_end_to_utt_obj,
    get_batch_variables,
    viterbi_decoding,
)


class ForceAligner:
    """
    Force alignment utility using NeMo CTC-based ASR models for speech-to-text alignment.
    """

    def __init__(
        self,
        asr_model: Optional[ASRModel] = None,
        device: str = None,
        frame_length: float = 0.02,
        asr_model_name: str = "stt_en_fastconformer_ctc_large",
    ):
        """
        Initialize the ForceAligner.

        Args:
            asr_model: NeMo ASR model instance for alignment. If None, will load from asr_model_name
            device: Device to run alignment on (default: auto-detect)
            frame_length: Frame length in seconds for timestamp conversion
            asr_model_name: Name of the NeMo ASR model to load if asr_model is None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_length = frame_length
        self.asr_model_name = asr_model_name

        self.asr_model = asr_model
        self.output_timestep_duration = None
        self._model_loaded = False

    def _load_asr_model(self):
        """Load the NeMo ASR model."""
        try:
            if self.device == 'cuda' and mp.get_start_method(allow_none=True) == 'fork':
                logging.warning(
                    "Detected 'fork' multiprocessing start method with CUDA device. "
                    "To avoid CUDA re-initialization errors in worker processes, "
                    "falling back to CPU for force alignment. "
                    "To use CUDA, set mp.set_start_method('spawn', force=True) in your main training script "
                    "before creating the DataLoader."
                )
                self.device = 'cpu'

            device = torch.device(self.device)
            logging.info(f"Loading NeMo ASR model '{self.asr_model_name}' for force alignment on device {device}")

            if self.asr_model is None:
                # Load ASR model from pretrained
                self.asr_model = ASRModel.from_pretrained(self.asr_model_name, map_location=device)
            else:
                self.asr_model = self.asr_model.to(device)

            self.asr_model.eval()

            # Calculate output timestep duration
            try:
                self.output_timestep_duration = (
                    self.asr_model.cfg['preprocessor']['window_stride'] * self.asr_model.encoder.subsampling_factor
                )
            except Exception as e:
                # Default fallback based on typical FastConformer settings
                self.output_timestep_duration = 0.04
                logging.warning(
                    f"Could not calculate output_timestep_duration from model config: {e}. "
                    f"Using default {self.output_timestep_duration}s"
                )

            logging.info(
                f"NeMo ASR model loaded successfully for force alignment. "
                f"Output timestep duration: {self.output_timestep_duration}s"
            )
        except Exception as e:
            logging.error(f"Failed to load NeMo ASR model for force alignment: {e}")
            self.asr_model = None
            raise

    def batch_force_align_user_audio(self, cuts: CutSet, source_sample_rate: int = 16000) -> CutSet:
        """
        Perform batched force alignment on all user audio segments.

        Collects all user segments, writes temp files, runs a single batched
        get_batch_variables + viterbi_decoding call, then maps results back.

        Args:
            cuts: CutSet containing all cuts to process
            source_sample_rate: Source sample rate of the audio

        Returns:
            CutSet with updated supervision texts (timestamped where alignment succeeded)
        """
        if not self._model_loaded:
            self._load_asr_model()
            self._model_loaded = True

        if self.asr_model is None:
            logging.warning("ASR model not available for force alignment, returning empty cutset")
            return CutSet.from_cuts([])

        # Collect all user supervisions
        user_supervisions = []
        user_cuts = []
        for cut in cuts:
            for supervision in cut.supervisions:
                if supervision.speaker.lower() == "user":
                    user_supervisions.append(supervision)
                    user_cuts.append(cut)

        if not user_supervisions:
            logging.info("No user supervisions found for force alignment")
            return cuts

        logging.info(f"Performing batched force alignment on {len(user_supervisions)} user audio segments")

        # Prepare all audio arrays and texts for batched processing
        audio_arrays = []
        normalized_texts = []
        valid_indices = []  # track which supervisions have valid audio/text
        target_sample_rate = 16000

        for i, (supervision, cut) in enumerate(zip(user_supervisions, user_cuts)):
            try:
                text = self._strip_timestamps(supervision.text)
                normalized_text = self._normalize_transcript(text)
                if not normalized_text.strip():
                    logging.warning(f"Text became empty after normalization: {supervision.text}")
                    continue

                user_cut = cut.truncate(offset=supervision.start, duration=supervision.duration)
                audio = user_cut.load_audio()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)

                if source_sample_rate != target_sample_rate:
                    from scipy import signal

                    num_samples = int(len(audio) * target_sample_rate / source_sample_rate)
                    audio = signal.resample(audio, num_samples)

                # Add silence padding for better alignment at the end
                silence_samples = int(0.64 * target_sample_rate)
                audio = np.concatenate([audio, np.zeros(silence_samples)])

                audio_arrays.append(audio)
                normalized_texts.append(normalized_text)
                valid_indices.append(i)
            except Exception as e:
                logging.error(f"Failed to prepare segment {i} for alignment: {e}")

        if not audio_arrays:
            logging.warning("No valid segments to align")
            return cuts

        # Batched ASR inference + Viterbi decoding
        success_count = 0
        failed_count = 0
        try:
            (
                log_probs_batch,
                y_batch,
                T_batch,
                U_batch,
                utt_obj_batch,
                output_timestep_duration,
            ) = get_batch_variables(
                audio=audio_arrays,
                model=self.asr_model,
                gt_text_batch=normalized_texts,
                align_using_pred_text=False,
                output_timestep_duration=self.output_timestep_duration,
            )

            alignments_batch = viterbi_decoding(
                log_probs_batch=log_probs_batch,
                y_batch=y_batch,
                T_batch=T_batch,
                U_batch=U_batch,
                viterbi_device=torch.device(self.device),
            )

            # Map results back to supervisions
            for batch_idx, orig_idx in enumerate(valid_indices):
                try:
                    if batch_idx >= len(alignments_batch) or batch_idx >= len(utt_obj_batch):
                        failed_count += 1
                        continue

                    utt_obj = utt_obj_batch[batch_idx]
                    if not utt_obj.token_ids_with_blanks:
                        failed_count += 1
                        continue

                    alignment = alignments_batch[batch_idx]
                    utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment, output_timestep_duration)
                    word_segments = self._extract_word_timestamps(utt_obj)

                    if word_segments:
                        timestamped_text = self._convert_alignment_to_timestamped_text(
                            word_segments, user_supervisions[orig_idx].text
                        )
                        user_supervisions[orig_idx].text = timestamped_text
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logging.error(f"Failed to process alignment for segment {orig_idx}: {e}")
                    failed_count += 1

        except Exception as e:
            logging.error(f"Batched force alignment failed: {e}")
            failed_count = len(valid_indices)
        finally:
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if failed_count > 0:
            logging.warning(
                f"Force alignment failed for {failed_count}/{len(user_supervisions)} user segments. "
                f"Keeping original text for failed alignments."
            )
        else:
            logging.info(f"Force alignment succeeded for all {success_count} user segments.")

        return cuts

    def _extract_word_timestamps(self, utt_obj) -> List[Dict[str, Any]]:
        """
        Extract word-level timestamps from the utterance object returned by NeMo aligner.

        Args:
            utt_obj: Utterance object with timing information

        Returns:
            List of word segments with timing information
        """
        word_segments = []

        for segment_or_token in utt_obj.segments_and_tokens:
            # Check if this is a Segment object (has words_and_tokens attribute)
            if hasattr(segment_or_token, 'words_and_tokens'):
                segment = segment_or_token
                for word_or_token in segment.words_and_tokens:
                    # Check if this is a Word object (has 'text' and timing attributes)
                    if hasattr(word_or_token, 'text') and hasattr(word_or_token, 't_start'):
                        word = word_or_token
                        # Skip CTC blank tokens and include only words with valid timing
                        if (
                            word.text not in ('<b>', '')
                            and word.t_start is not None
                            and word.t_end is not None
                            and word.t_start >= 0
                            and word.t_end >= 0
                        ):
                            word_segments.append(
                                {
                                    'word': word.text,
                                    'start': word.t_start,
                                    'end': word.t_end,
                                    'score': 1.0,  # NeMo CTC alignment doesn't provide confidence scores
                                }
                            )

        return word_segments

    def _normalize_transcript(self, transcript: str) -> str:
        """
        Normalize transcript for the ASR model's tokenizer.
        Keeps it simple to match common ASR preprocessing.
        """
        text = transcript.lower()
        # Remove special characters except apostrophes and spaces
        text = re.sub(r"[^a-z' ]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _convert_alignment_to_timestamped_text(
        self, alignment_result: List[Dict[str, Any]], original_text: str
    ) -> str:
        """
        Convert alignment results to timestamped text format.

        Args:
            alignment_result: List of word segments with timing information
            original_text: Original text without timestamps

        Returns:
            Text with timestamp tokens in the format <|start_frame|>word<|end_frame|>
        """
        timestamped_words = []

        for word_seg in alignment_result:
            word = word_seg["word"]
            start_frame = int(word_seg["start"] / self.frame_length)
            end_frame = int(word_seg["end"] / self.frame_length)
            timestamped_words.append(f"<|{start_frame}|> {word} <|{end_frame}|>")

        return " ".join(timestamped_words)

    def _strip_timestamps(self, text: str) -> str:
        """
        Strip timestamp tokens from text.

        Args:
            text: Text that may contain timestamp tokens

        Returns:
            Text with timestamp tokens removed
        """
        text = re.sub(r'<\|[0-9]+\|>', '', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()
