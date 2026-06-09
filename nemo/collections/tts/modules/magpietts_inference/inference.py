# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Core inference logic for MagpieTTS models.

This module provides a strategy-pattern based inference framework with:
- BaseInferenceConfig / MagpieInferenceConfig / EasyMagpieInferenceConfig
- BaseInferenceRunner / MagpieInferenceRunner / EasyMagpieInferenceRunner

MagpieInferenceRunner handles the encoder-decoder MagpieTTSModel
(chunked text, generate_speech + codes_to_audio).

EasyMagpieInferenceRunner handles the decoder-only EasyMagpieTTSInferenceModel
(infer_batch, returns audio directly).
"""
from __future__ import annotations

import abc
import glob
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer
from nemo.collections.tts.data.text_to_speech_dataset import ChunkedTTSInferenceDataset, MagpieTTSDataset
from nemo.collections.tts.models.easy_magpietts_inference import EasyModelInferenceParameters
from nemo.collections.tts.models.magpietts import ModelInferenceParameters
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.tts.parts.utils.tts_dataset_utils import normalize_volume, stack_tensors
from nemo.utils import logging


@dataclass
class BaseInferenceConfig(abc.ABC):
    """Shared inference configuration fields.

    Subclasses must declare their own ``model_inference_parameters`` field
    with the appropriate type (ModelInferenceParameters or
    EasyModelInferenceParameters).

    Attributes:
        batch_size: Batch size for inference.
        use_cfg: Whether to use classifier-free guidance.
        use_local_transformer: Whether to use local transformer for inference.
    """

    batch_size: int = 32
    use_cfg: bool = False
    use_local_transformer: bool = False

    @abc.abstractmethod
    def build_identifier(self) -> str:
        """Build a unique identifier string for naming output directories."""
        pass

    @staticmethod
    def _format_layer_list(layers: Optional[List[int]]) -> str:
        """Format a list of layer indices as a compact string."""
        if layers is None:
            return "None"
        return "".join(str(_layer) for _layer in layers)


@dataclass
class MagpieInferenceConfig(BaseInferenceConfig):
    """Configuration for encoder-decoder MagpieTTSModel inference.

    Attributes:
        # Model specific inference parameters
        model_inference_parameters: See ModelInferenceParameters dataclass

        # MaskGit parameters
        maskgit_n_steps: Number of MaskGit refinement steps.
        maskgit_noise_scale: Noise scale for MaskGit sampling.
        maskgit_fixed_schedule: Fixed schedule for MaskGit (optional).
        maskgit_sampling_type: Type of MaskGit sampling.
    """

    model_inference_parameters: ModelInferenceParameters = field(default_factory=ModelInferenceParameters)
    apply_attention_prior: bool = False

    # MaskGit parameters
    maskgit_n_steps: int = 3
    maskgit_noise_scale: float = 0.0
    maskgit_fixed_schedule: Optional[List[int]] = None
    maskgit_sampling_type: Optional[str] = None

    def build_identifier(self) -> str:
        """Build a unique identifier string for this configuration.

        Used for naming output directories and files.

        Returns:
            String identifier incorporating key config values.
        """
        parts = [
            f"Temp{self.model_inference_parameters.temperature}",
            f"Topk{self.model_inference_parameters.topk}",
            f"Cfg_{self.use_cfg}_{self.model_inference_parameters.cfg_scale}",
            f"Prior_{self.apply_attention_prior}",
        ]

        if self.apply_attention_prior:
            parts.extend(
                [
                    f"{self.model_inference_parameters.attention_prior_epsilon}",
                    f"{self.model_inference_parameters.attention_prior_lookahead_window}",
                    f"{self.model_inference_parameters.start_prior_after_n_audio_steps}",
                    self._format_layer_list(self.model_inference_parameters.estimate_alignment_from_layers),
                    self._format_layer_list(self.model_inference_parameters.apply_prior_to_layers),
                ]
            )

        parts.extend(
            [
                f"LT_{self.use_local_transformer}",
                f"MaskGit_{self.maskgit_n_steps}_{self.maskgit_sampling_type}",
                self._format_layer_list(self.maskgit_fixed_schedule),
                f"EOS_{self.model_inference_parameters.eos_detection_method}",
                f"IgnoreFST_{self.model_inference_parameters.ignore_finished_sentence_tracking}",
            ]
        )

        return "_".join(parts)


@dataclass
class EasyMagpieInferenceConfig(BaseInferenceConfig):
    """Configuration for decoder-only EasyMagpieTTSInferenceModel inference.

    Attributes:
        model_inference_parameters: See EasyModelInferenceParameters dataclass
        phoneme_input_type: Type of phoneme input ('gt' or 'predicted').
        phoneme_sampling_method: Method of sampling phonemes ('argmax' or 'multinomial').
        dropout_text_input: Whether to dropout text input.
    """

    model_inference_parameters: EasyModelInferenceParameters = field(default_factory=EasyModelInferenceParameters)
    phoneme_input_type: str = "gt"
    phoneme_sampling_method: str = "argmax"
    dropout_text_input: bool = False

    def build_identifier(self) -> str:
        parts = [
            f"Temp{self.model_inference_parameters.temperature}",
            f"Topk{self.model_inference_parameters.topk}",
            f"Cfg_{self.use_cfg}_{self.model_inference_parameters.cfg_scale}",
            f"LT_{self.use_local_transformer}",
            f"Phoneme_{self.phoneme_input_type}_{self.phoneme_sampling_method}",
        ]
        return "_".join(parts)


# Backwards-compatible aliases
InferenceConfig = MagpieInferenceConfig


class BaseInferenceRunner(abc.ABC):
    """Abstract base for TTS inference runners.

    Provides shared utilities (batch-to-cuda, file cleanup, reference audio
    copying, RTF metrics) and declares the interface that concrete runners
    must implement.
    """

    def __init__(self, model, config: BaseInferenceConfig):
        """Initialize the inference runner.

        Args:
            model: Loaded TTS model (should be on GPU and in eval mode).
            config: Inference configuration.
        """
        self.model = model
        self.config = config

        # Set phoneme probability to 1 for inference
        self._configure_tokenizer()

        # Cached state from create_dataset (set when create_dataset is called)
        self._manifest_records: Optional[List[dict]] = None
        self._audio_base_dir: Optional[str] = None

    @abc.abstractmethod
    def create_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: Optional[float] = None,
        context_duration_max: Optional[float] = None,
    ) -> Union[ChunkedTTSInferenceDataset, MagpieTTSDataset]:
        """Create an inference dataset from dataset metadata.

        Args:
            dataset_meta: Dataset metadata dictionary with manifest and audio root.
            context_duration_min: Minimum context duration in seconds, or None to use defaults.
            context_duration_max: Maximum context duration in seconds, or None to use defaults.

        Returns:
            A model-compatible inference dataset implementation.
        """
        pass

    @abc.abstractmethod
    def run_inference_on_dataset(
        self,
        dataset,
        output_dir: str,
        manifest_records: Optional[List[dict]] = None,
        audio_base_dir: Optional[str] = None,
        save_cross_attention_maps: bool = True,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        """Run inference on a dataset and persist generated artifacts.

        Args:
            dataset: The inference dataset created by ``create_dataset``.
            output_dir: Directory to save generated outputs.
            manifest_records: Original manifest records (uses cached records if None).
            audio_base_dir: Base directory for audio paths (uses cached value if None).
            save_cross_attention_maps: Whether to save attention maps, if supported.
            save_context_audio: Whether to copy context/target reference audio to output.
            save_predicted_codes: Whether to save predicted codec tokens.

        Returns:
            Tuple of:
                - rtf_metrics: Per-batch real-time factor metrics.
                - generated_audio_paths: Paths to generated audio files.
                - codec_file_paths: Paths to predicted codec token files.
        """
        pass

    def _configure_tokenizer(self) -> None:
        """Configure the tokenizer for inference (phoneme prob = 1.0)."""
        g2p = None
        if isinstance(self.model.tokenizer, AggregatedTTSTokenizer):
            if "english_phoneme" in self.model.tokenizer.tokenizers and hasattr(
                self.model.tokenizer.tokenizers["english_phoneme"], "g2p"
            ):
                g2p = self.model.tokenizer.tokenizers["english_phoneme"].g2p
        elif isinstance(self.model.tokenizer, IPATokenizer):
            g2p = self.model.tokenizer.g2p

        if g2p is not None:
            g2p.phoneme_probability = 1.0

    def _resolve_manifest_and_audio_dir(
        self,
        manifest_records: Optional[List[dict]],
        audio_base_dir: Optional[str],
    ) -> Tuple[List[dict], str]:
        if manifest_records is None:
            if self._manifest_records is None:
                raise ValueError("manifest_records not provided and not cached from create_dataset()")
            manifest_records = self._manifest_records
        if audio_base_dir is None:
            if self._audio_base_dir is None:
                raise ValueError("audio_base_dir not provided and not cached from create_dataset()")
            audio_base_dir = self._audio_base_dir
        return manifest_records, audio_base_dir

    def _read_and_cache_manifest(self, dataset_meta: dict) -> Tuple[str, str]:
        """Read manifest from dataset_meta, cache records, return (manifest_path, audio_dir)."""
        dataset_name = list(dataset_meta.keys())[0]
        dataset_info = dataset_meta[dataset_name]
        manifest_path = dataset_info.get('manifest_path')
        audio_dir = dataset_info.get('audio_dir', '')
        logging.info(f"Dataset name: {dataset_name}, manifest_path: {manifest_path}, audio_dir: {audio_dir}")
        self._manifest_records = read_manifest(manifest_path)
        self._audio_base_dir = audio_dir
        return manifest_path, audio_dir

    def _get_context_durations(
        self,
        context_duration_min: Optional[float],
        context_duration_max: Optional[float],
    ) -> Tuple[float, float]:
        if context_duration_min is None:
            context_duration_min = self.model.cfg.get('context_duration_min', 5.0)
        if context_duration_max is None:
            context_duration_max = self.model.cfg.get('context_duration_max', 5.0)
        # For multi-encoder models, use fixed 5s context for fair evaluation
        if context_duration_min < 5.0 and context_duration_max > 5.0:
            context_duration_min = 5.0
            context_duration_max = 5.0
        return context_duration_min, context_duration_max

    @staticmethod
    def _batch_to_cuda(batch: dict) -> dict:
        """Move batch tensors to CUDA device."""
        batch_cuda = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_cuda[key] = value.cuda()
            else:
                batch_cuda[key] = value
        return batch_cuda

    @staticmethod
    def _delete_old_generated_files(output_dir: str) -> None:
        """Delete leftover generated files from previous runs."""
        logging.info(f"Cleaning up old generated files in: {output_dir}")
        patterns = [
            "predicted_codes*.pt",
            "predicted_audio*.wav",
            "cross_attn_map_*.png",
        ]
        for pattern in patterns:
            for f in glob.glob(os.path.join(output_dir, pattern)):
                os.remove(f)

    @staticmethod
    def _copy_reference_audio(
        record: dict,
        audio_base_dir: str,
        output_dir: str,
        item_idx: int,
    ) -> None:
        """Copy context and target audio files to output directory."""
        context_path = record.get('context_audio_filepath')
        target_path = record.get('audio_filepath')

        if context_path is not None:
            full_context_path = os.path.join(audio_base_dir, context_path)
            if os.path.exists(full_context_path):
                dest = os.path.join(output_dir, f"context_audio_{item_idx}.wav")
                shutil.copy(full_context_path, dest)

        if target_path is not None:
            full_target_path = os.path.join(audio_base_dir, target_path)
            if os.path.exists(full_target_path):
                dest = os.path.join(output_dir, f"target_audio_{item_idx}.wav")
                shutil.copy(full_target_path, dest)

    @staticmethod
    def compute_mean_rtf_metrics(rtf_metrics_list: List[dict]) -> Dict[str, float]:
        """Compute mean RTF metrics across batches."""
        if not rtf_metrics_list or not rtf_metrics_list[0]:
            return {}
        mean_metrics = {}
        for key in rtf_metrics_list[0]:
            values = [m[key] for m in rtf_metrics_list if key in m]
            mean_metrics[key] = float(sum(values) / len(values)) if values else 0.0
        return mean_metrics


class MagpieInferenceRunner(BaseInferenceRunner):
    """Runner for encoder-decoder MagpieTTSModel.

    Uses ChunkedTTSInferenceDataset and model.generate_speech() per chunk,
    then codes_to_audio() to produce waveforms.
    """

    def __init__(self, model, config: MagpieInferenceConfig):
        super().__init__(model, config)

    def create_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: Optional[float] = None,
        context_duration_max: Optional[float] = None,
    ) -> ChunkedTTSInferenceDataset:
        """Create a unified dataset for inference.

        Always creates ChunkedTTSInferenceDataset which uses language-aware chunking
        to automatically handle both short and long texts:
        - Short text (below threshold): processed as single chunk
        - Long text (above threshold): split into sentence chunks

        Args:
            dataset_meta: Dataset metadata dictionary with 'manifest_path' and 'audio_dir'.
            context_duration_min: Minimum context duration (uses model default if None).
            context_duration_max: Maximum context duration (uses model default if None).

        Returns:
            Configured ChunkedTTSInferenceDataset instance.
        """
        context_duration_min, context_duration_max = self._get_context_durations(
            context_duration_min, context_duration_max
        )
        self._read_and_cache_manifest(dataset_meta)

        # Always use unified dataset (handles both short and long texts automatically)
        # Language for chunking thresholds is determined per-sample from manifest
        logging.info("Creating unified inference dataset")
        dataset = self._create_chunked_inference_dataset(dataset_meta, context_duration_min, context_duration_max)
        return dataset

    def run_inference_on_dataset(
        self,
        dataset: ChunkedTTSInferenceDataset,
        output_dir: str,
        manifest_records: Optional[List[dict]] = None,
        audio_base_dir: Optional[str] = None,
        save_cross_attention_maps: bool = True,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        """Use the unified chunked inference path for encoder-decoder models."""
        manifest_records, audio_base_dir = self._resolve_manifest_and_audio_dir(manifest_records, audio_base_dir)
        logging.info("Using unified inference path")
        return self._run_unified_inference(
            dataset, output_dir, manifest_records, audio_base_dir, save_context_audio, save_predicted_codes
        )

    def _create_chunked_inference_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: float,
        context_duration_max: float,
    ) -> ChunkedTTSInferenceDataset:
        """Create a unified inference dataset.

        Creates ChunkedTTSInferenceDataset which uses language-aware chunking
        to automatically handle both short and long texts.

        Args:
            dataset_meta: Dataset metadata dictionary (same format as MagpieTTSDataset).
            context_duration_min: Minimum context duration.
            context_duration_max: Maximum context duration.

        Returns:
            Configured ChunkedTTSInferenceDataset instance.
        """
        # Create unified dataset - language and tokenizer are determined per-sample from manifest
        dataset = ChunkedTTSInferenceDataset(
            dataset_meta=dataset_meta,
            sample_rate=self.model.output_sample_rate,
            codec_model_samples_per_frame=self.model.codec_model_samples_per_frame,
            eos_id=self.model.eos_id,
            num_audio_codebooks=self.model.num_audio_codebooks,
            context_duration_min=context_duration_min,
            context_duration_max=context_duration_max,
            use_text_conditioning_tokenizer=self.model.use_text_conditioning_encoder,
            text_conditioning_tokenizer_name=self.model.text_conditioning_tokenizer_name,
            pad_context_text_to_max_duration=self.model.pad_context_text_to_max_duration,
            load_16khz_audio=self.model.model_type == 'single_encoder_sv_tts',
        )

        # Attach model's tokenizer
        dataset.text_tokenizer = self.model.tokenizer
        return dataset

    def _run_unified_inference(
        self,
        dataset: ChunkedTTSInferenceDataset,
        output_dir: str,
        manifest_records: List[dict],
        audio_base_dir: str,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        """Run unified inference with automatic single/multi-chunk handling.

        Processes all samples through generate_speech, passing
        beginning_of_text and end_of_text so the model can handle both
        single-chunk (short text) and multi-chunk (long text) cases correctly.

        Args:
            dataset: ChunkedTTSInferenceDataset created by create_dataset().
            output_dir: Directory to save generated audio and artifacts.
            manifest_records: List of manifest record dictionaries.
            audio_base_dir: Base directory for resolving audio paths.
            save_context_audio: Whether to copy context audio files.
            save_predicted_codes: Whether to save predicted code files.

        Returns:
            Tuple of:
                - rtf_metrics: List of real-time factor metrics per batch.
                - generated_audio_paths: List of paths to generated audio files.
                - codec_file_paths: List of paths to predicted codes files.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._delete_old_generated_files(output_dir)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=0,  # Avoid multiprocessing issues with CUDA
            shuffle=False,
        )

        all_rtf_metrics = []
        generated_audio_paths = []
        codec_file_paths = []
        global_item_idx = 0

        for batch_idx, batch in enumerate(dataloader):
            logging.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            # Move batch tensors to CUDA
            batch = self._batch_to_cuda(batch)
            batch['sample_rate'] = self.model.output_sample_rate
            batch['context_sample_rate'] = self.model.output_sample_rate

            batch_size = len(batch['chunked_tokens'])
            max_num_chunks = max(len(tokens) for tokens in batch['chunked_tokens'])

            # Clear stale KV cache from prior inference calls (e.g., the previous batch or dataset
            # may have left with populated tensors).
            logging.info(f"Resetting KV cache for decoder: {self.model.use_kv_cache_for_inference}")
            use_kv_cache_for_this_batch = self.model.use_kv_cache_for_inference if max_num_chunks == 1 else False
            self.model.decoder.reset_cache(use_cache=use_kv_cache_for_this_batch)

            # Create chunk state for this batch
            chunk_state = self.model.create_chunk_state(batch_size=batch_size)

            # Accumulators for predicted codes
            predicted_codes_per_sample = [[] for _ in range(batch_size)]
            predicted_codes_lens = [0 for _ in range(batch_size)]

            # Overwrite the model's parameters since we want to use the arguments from the commandline
            self.model.inference_parameters = self.config.model_inference_parameters

            start_time = time.time()
            # Iterate over text chunks (1 for short text, N for long text)
            for chunk_idx in range(max_num_chunks):
                # Extract current chunk tokens for each sample
                current_tokens = []
                current_tokens_lens = []
                for b_idx in range(batch_size):
                    current_tokens.append(batch['chunked_tokens'][b_idx][chunk_idx])
                    current_tokens_lens.append(batch['chunked_tokens_lens'][b_idx][chunk_idx])

                # Pad tokens to max length in this chunk
                max_len = max(current_tokens_lens)
                batch['text'] = stack_tensors(current_tokens, max_lens=[max_len]).cuda()
                batch['text_lens'] = torch.tensor(current_tokens_lens, dtype=torch.int32).cuda()

                # Compute is_end_of_text flags (per-sample)
                is_end_of_text = self._compute_end_of_text_flags(
                    batch, chunk_idx, max_num_chunks, current_tokens_lens, batch_size
                )

                beginning_of_text = chunk_idx == 0

                # Call generate_speech (unified entry point)
                output = self.model.generate_speech(
                    batch,
                    chunk_state=chunk_state,
                    end_of_text=is_end_of_text,
                    beginning_of_text=beginning_of_text,
                    use_cfg=self.config.use_cfg,
                    use_local_transformer_for_inference=self.config.use_local_transformer,
                    maskgit_n_steps=self.config.maskgit_n_steps,
                    maskgit_noise_scale=self.config.maskgit_noise_scale,
                    maskgit_fixed_schedule=self.config.maskgit_fixed_schedule,
                    maskgit_sampling_type=self.config.maskgit_sampling_type,
                )

                # Unpack output
                chunk_codes = output.predicted_codes
                chunk_codes_lens = output.predicted_codes_lens

                # Accumulate codes for each sample
                for b_idx in range(batch_size):
                    # Skip if this sample's text has ended (padding chunks)
                    if is_end_of_text[b_idx] and current_tokens_lens[b_idx] == 1:
                        continue
                    code_len = chunk_codes_lens[b_idx]
                    if code_len > 0:
                        codes_slice = chunk_codes[b_idx][:, :code_len]
                        predicted_codes_per_sample[b_idx].append(codes_slice)
                        predicted_codes_lens[b_idx] += code_len

            elapsed = time.time() - start_time
            logging.info(f"Batch inference time: {elapsed:.2f}s")

            # Concatenate codes and convert to audio
            predicted_codes_list = []
            for b_idx in range(batch_size):
                if predicted_codes_per_sample[b_idx]:
                    concatenated = torch.cat(predicted_codes_per_sample[b_idx], dim=1).cuda()
                else:
                    # Empty placeholder
                    concatenated = torch.zeros((self.model.num_audio_codebooks, 1), dtype=torch.long, device='cuda')
                predicted_codes_list.append(concatenated)

            # Stack and convert to audio
            max_code_len = max(predicted_codes_lens) if any(predicted_codes_lens) else 1
            predicted_codes = stack_tensors(predicted_codes_list, max_lens=[max_code_len]).cuda()
            predicted_codes_lens_tensor = torch.tensor(predicted_codes_lens, dtype=torch.long, device='cuda')

            predicted_audio, predicted_audio_lens, predicted_codes = self.model._codec_helper.codes_to_audio(
                predicted_codes,
                predicted_codes_lens_tensor,
            )

            # Compute RTF metrics
            total_audio_samples = sum(predicted_audio_lens.cpu().tolist())
            total_audio_seconds = total_audio_samples / self.model.output_sample_rate
            rtf = elapsed / total_audio_seconds if total_audio_seconds > 0 else 0.0
            rtf_metrics = {
                'inference_time': elapsed,
                'audio_seconds': total_audio_seconds,
                'rtf': rtf,
            }
            all_rtf_metrics.append(rtf_metrics)

            # Save outputs
            predicted_audio_np = predicted_audio.float().detach().cpu().numpy()

            for b_idx in range(batch_size):
                sample_idx = batch['idx'][b_idx]
                audio_len = predicted_audio_lens[b_idx].item()
                audio_np = predicted_audio_np[b_idx, :audio_len]

                audio_path = os.path.join(output_dir, f"predicted_audio_{sample_idx}.wav")
                sf.write(audio_path, audio_np, self.model.output_sample_rate)
                generated_audio_paths.append(audio_path)

                # Copy reference audio if requested
                if save_context_audio and sample_idx < len(manifest_records):
                    self._copy_reference_audio(
                        manifest_records[sample_idx],
                        audio_base_dir,
                        output_dir,
                        sample_idx,
                    )

                if save_predicted_codes:
                    codes_path = os.path.join(output_dir, f"predicted_codes_{sample_idx}.pt")
                    predicted_codes_current = predicted_codes[b_idx, :, : predicted_codes_lens[b_idx]]  # C, T
                    torch.save(predicted_codes_current, codes_path)
                    codec_file_paths.append(codes_path)

                global_item_idx += 1

        return all_rtf_metrics, generated_audio_paths, codec_file_paths

    @staticmethod
    def _compute_end_of_text_flags(
        batch: Dict[str, Any],
        chunk_idx: int,
        max_num_chunks: int,
        current_tokens_lens: List[int],
        batch_size: int,
    ) -> List[bool]:
        """Compute end-of-text flags for each sample in batch.

        Args:
            batch: Current batch dictionary.
            chunk_idx: Current chunk index.
            max_num_chunks: Maximum number of chunks in this batch.
            current_tokens_lens: Token lengths for current chunk per sample.
            batch_size: Number of samples in batch.

        Returns:
            List of booleans indicating if each sample has reached end of text.
        """
        is_end_of_text = []
        for b_idx in range(batch_size):
            if chunk_idx == max_num_chunks - 1:
                # Last chunk
                is_end_of_text.append(True)
            elif current_tokens_lens[b_idx] == 1:
                # Current chunk is padding
                is_end_of_text.append(True)
            elif batch['chunked_tokens_lens'][b_idx][chunk_idx + 1] == 1:
                # Next chunk is padding
                is_end_of_text.append(True)
            else:
                is_end_of_text.append(False)
        return is_end_of_text




@dataclass
class EasyMagpieMultiturnUserAudioInferenceConfig(EasyMagpieInferenceConfig):
    """Configuration for EasyMagpie multiturn user-audio inference.

    This mode keeps the standard EasyMagpie/MagpieTTS evaluation contract by
    writing one evaluation row per generated agent turn:

      predicted_audio_<idx>.wav
      predicted_codes_<idx>.pt
      target_audio_<idx>.wav
      context_audio_<idx>.wav

    The generated turn-level manifest is cached on the runner as
    ``evaluation_manifest_path`` so the top-level inference script can pass it
    to ``evaluate_generated_audio_dir``.
    """

    max_eval_turns: int = 6
    save_debug_multiturn_audio: bool = True

    def build_identifier(self) -> str:
        return super().build_identifier() + f"_MTUserAudio_MaxTurns{self.max_eval_turns}"


class EasyMagpieMultiturnUserAudioDataset(torch.utils.data.Dataset):
    """Manifest dataset for turn-level multiturn user-audio EasyMagpie inference."""

    def __init__(
        self,
        manifest_path: str,
        audio_dir: str,
        model,
        max_eval_turns: int = 6,
        normalize_audio: bool = True,
    ):
        self.manifest_path = manifest_path
        self.audio_dir = audio_dir or ""
        self.model = model
        self.max_eval_turns = max_eval_turns
        self.normalize_audio = normalize_audio
        self.records = read_manifest(manifest_path)
        # debug
        self.records = self.records[:7]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        item = dict(self.records[idx])
        item["idx"] = idx
        return item

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None or path == "":
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(self.audio_dir, path)

    def _load_audio_1d(self, path: str, sample_rate: int) -> torch.Tensor:
        path = self._resolve_path(path)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Missing audio path: {path}")

        audio, sr = sf.read(path, dtype="float32", always_2d=False)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if self.normalize_audio:
            audio = normalize_volume(audio)

        wav = torch.as_tensor(audio, dtype=torch.float32).flatten()

        if sr != sample_rate:
            wav = resample(wav.unsqueeze(0), sr, sample_rate).squeeze(0)

        return wav.contiguous()

    @staticmethod
    def _as_turn_list(value) -> List[str]:
        if isinstance(value, list):
            return [str(x) for x in value]
        return [str(value)]

    def collate_fn(self, batch: List[dict]) -> Dict[str, Any]:
        if len(batch) != 1:
            raise RuntimeError("multiturn_user_audio inference currently requires batch_size=1.")

        sample = batch[0]
        model = self.model
        sample_rate = model.sample_rate
        main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]

        raw_turn_texts = self._as_turn_list(sample["text"])[: self.max_eval_turns]
        max_turns = len(raw_turn_texts)

        batched_turns = []
        batched_turn_lens = []
        valid_turn_masks = []
        for turn_text in raw_turn_texts:
            ids = model.tokenizer.encode(turn_text, tokenizer_name=main_tokenizer_name) + [model.eos_id]
            batched_turns.append(torch.tensor([ids], dtype=torch.long))
            batched_turn_lens.append(torch.tensor([len(ids)], dtype=torch.long))
            valid_turn_masks.append(torch.tensor([True], dtype=torch.bool))

        context_path = self._resolve_path(sample.get("context_audio_filepath"))
        context_audio = self._load_audio_1d(context_path, sample_rate).unsqueeze(0)
        context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long)

        user_audio_paths = sample.get("user_audio_file_path", None)
        if not isinstance(user_audio_paths, list):
            user_audio_paths = []

        user_audio_turns = []
        user_audio_turns_lens = []
        for turn_id in range(max_turns):
            if turn_id < len(user_audio_paths) and user_audio_paths[turn_id]:
                wav = self._load_audio_1d(user_audio_paths[turn_id], sample_rate)
            else:
                wav = torch.zeros(int(2 * sample_rate), dtype=torch.float32)
            user_audio_turns.append(wav.unsqueeze(0))
            user_audio_turns_lens.append(torch.tensor([wav.numel()], dtype=torch.long))

        target_turn_audio_paths = sample.get("target_audio_file_path", sample.get("target_audio_filepath", None))
        if target_turn_audio_paths is not None and not isinstance(target_turn_audio_paths, list):
            target_turn_audio_paths = [target_turn_audio_paths]

        return {
            "idx": torch.tensor([int(sample["idx"])], dtype=torch.long),
            "raw_record": sample,
            "raw_turn_texts": [raw_turn_texts],
            "batched_turns": batched_turns,
            "batched_turn_lens": batched_turn_lens,
            "valid_turn_masks": valid_turn_masks,
            "context_audio": context_audio,
            "context_audio_lengths": context_audio_lens,
            "user_audio_turns": user_audio_turns,
            "user_audio_turns_lens": user_audio_turns_lens,
            "target_audio_path": sample.get("audio_filepath"),
            "target_turn_audio_paths": target_turn_audio_paths,
        }



class _InferenceSubset(torch.utils.data.Dataset):
    """Subset wrapper that preserves the wrapped dataset collate_fn."""

    def __init__(self, dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = list(indices)
        self.collate_fn = getattr(dataset, "collate_fn", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

class EasyMagpieInferenceRunner(BaseInferenceRunner):
    """Runner for decoder-only EasyMagpieTTSInferenceModel.

    Uses MagpieTTSDataset and model.infer_batch() which returns audio directly.
    """

    def __init__(self, model, config: EasyMagpieInferenceConfig):
        super().__init__(model, config)

    def create_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: Optional[float] = None,
        context_duration_max: Optional[float] = None,
    ) -> MagpieTTSDataset:
        context_duration_min, context_duration_max = self._get_context_durations(
            context_duration_min, context_duration_max
        )
        self._read_and_cache_manifest(dataset_meta)

        logging.info("Creating inference dataset for decoder-only model")
        dataset = MagpieTTSDataset(
            dataset_meta=dataset_meta,
            sample_rate=self.model.sample_rate,
            min_duration=0.5,
            max_duration=20,
            codec_model_samples_per_frame=self.model.codec_model_samples_per_frame,
            bos_id=getattr(self.model, "bos_id", None),
            eos_id=self.model.eos_id,
            num_audio_codebooks=self.model.num_audio_codebooks,
            prior_scaling_factor=None,
            load_cached_codes_if_available=False,
            dataset_type='test',
            tokenizer_config=None,
            load_16khz_audio=False,
            use_text_conditioning_tokenizer=True,
            text_conditioning_tokenizer_name=self.model.text_conditioning_tokenizer_name,
            pad_context_text_to_max_duration=False,
            context_duration_min=context_duration_min,
            context_duration_max=context_duration_max,
            ignore_phoneme_languages=self.model.cfg.get('ignore_phoneme_languages', []),
            add_language_to_context_text=self.model.add_language_to_context_text,
        )
        dataset.text_tokenizer = self.model.tokenizer

        if hasattr(self.model, 'phoneme_tokenizer'):
            dataset.phoneme_tokenizer = self.model.phoneme_tokenizer

        return dataset

    def run_inference_on_dataset(
        self,
        dataset: MagpieTTSDataset,
        output_dir: str,
        manifest_records: Optional[List[dict]] = None,
        audio_base_dir: Optional[str] = None,
        save_cross_attention_maps: bool = True,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        manifest_records, audio_base_dir = self._resolve_manifest_and_audio_dir(manifest_records, audio_base_dir)
        logging.info("Using decoder-only inference path")
        return self._run_decoder_only_inference(
            dataset, output_dir, manifest_records, audio_base_dir, save_context_audio, save_predicted_codes
        )

    def _run_decoder_only_inference(
        self,
        dataset: MagpieTTSDataset,
        output_dir: str,
        manifest_records: List[dict],
        audio_base_dir: str,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        os.makedirs(output_dir, exist_ok=True)
        self._delete_old_generated_files(output_dir)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=0,
            shuffle=False,
        )

        all_rtf_metrics = []
        generated_audio_paths = []
        codec_file_paths = []
        item_idx = 0
        phoneme_sampling_method = (
            "argmax" if self.config.phoneme_sampling_method == "greedy" else self.config.phoneme_sampling_method
        )

        for batch_idx, batch in enumerate(dataloader):
            logging.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            batch = self._batch_to_cuda(batch)
            output = self.model.infer_batch(
                batch,
                max_decoder_steps=self.config.model_inference_parameters.max_decoder_steps,
                temperature=self.config.model_inference_parameters.temperature,
                topk=self.config.model_inference_parameters.topk,
                use_cfg=self.config.use_cfg,
                cfg_scale=self.config.model_inference_parameters.cfg_scale,
                use_local_transformer_for_inference=self.config.use_local_transformer,
                phoneme_input_type=self.config.phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                force_dropout_text=self.config.dropout_text_input,
            )
            predicted_audio = output.predicted_audio
            predicted_audio_lens = output.predicted_audio_lens
            predicted_codes = output.predicted_codes
            predicted_codes_lens = output.predicted_codes_lens
            rtf_metrics = output.rtf_metrics

            all_rtf_metrics.append(rtf_metrics)
            logging.info(f"Output shape: {predicted_audio.size()}")

            for idx in range(predicted_audio.size(0)):
                audio_len = predicted_audio_lens[idx].item()
                audio_np = predicted_audio[idx].float().detach().cpu().numpy()[:audio_len]
                audio_path = os.path.join(output_dir, f"predicted_audio_{item_idx}.wav")
                sample_rate = getattr(self.model, "output_sample_rate", self.model.sample_rate)
                sf.write(audio_path, audio_np, sample_rate)
                generated_audio_paths.append(audio_path)

                if save_context_audio and item_idx < len(manifest_records):
                    self._copy_reference_audio(
                        manifest_records[item_idx],
                        audio_base_dir,
                        output_dir,
                        item_idx,
                    )

                if save_predicted_codes:
                    code_len = predicted_codes_lens[idx].item()
                    codes_path = os.path.join(output_dir, f"predicted_codes_{item_idx}.pt")
                    torch.save(predicted_codes[idx, :, :code_len].detach().cpu(), codes_path)
                    codec_file_paths.append(codes_path)

                item_idx += 1

        return all_rtf_metrics, generated_audio_paths, codec_file_paths

class EasyMagpieMultiturnUserAudioInferenceRunner(BaseInferenceRunner):
    """Runner for decoder-only EasyMagpieTTS multiturn user-audio inference.

    It generates one agent turn at a time using user-audio prefill, but writes
    outputs using the standard EasyMagpie evaluation contract. Therefore the
    existing magpietts_inference.py evaluation code can call
    evaluate_generated_audio_dir() unchanged.
    """

    produces_turn_level_evaluation: bool = True

    def __init__(self, model, config: EasyMagpieMultiturnUserAudioInferenceConfig):
        if config.batch_size != 1:
            raise ValueError("EasyMagpie multiturn user-audio inference requires batch_size=1.")
        super().__init__(model, config)
        self.evaluation_manifest_path: Optional[str] = None
        self.evaluation_audio_dir: Optional[str] = None
        self.evaluation_manifest_records: Optional[List[dict]] = None

        # Used by examples/tts/magpietts_inference.py for torchrun sharding.
        self.distributed_rank: int = int(os.environ.get("RANK", "0"))
        self.distributed_world_size: int = int(os.environ.get("WORLD_SIZE", "1"))

    def create_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: Optional[float] = None,
        context_duration_max: Optional[float] = None,
    ) -> EasyMagpieMultiturnUserAudioDataset:
        manifest_path, audio_dir = self._read_and_cache_manifest(dataset_meta)
        logging.info("Creating multiturn user-audio inference dataset for decoder-only model")
        return EasyMagpieMultiturnUserAudioDataset(
            manifest_path=manifest_path,
            audio_dir=audio_dir,
            model=self.model,
            max_eval_turns=self.config.max_eval_turns,
            normalize_audio=True,
        )

    def set_distributed_context(self, rank: int, world_size: int) -> None:
        self.distributed_rank = int(rank)
        self.distributed_world_size = int(world_size)

    def run_inference_on_dataset(
        self,
        dataset: EasyMagpieMultiturnUserAudioDataset,
        output_dir: str,
        manifest_records: Optional[List[dict]] = None,
        audio_base_dir: Optional[str] = None,
        save_cross_attention_maps: bool = True,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        manifest_records, audio_base_dir = self._resolve_manifest_and_audio_dir(manifest_records, audio_base_dir)
        return self._run_multiturn_user_audio_inference(
            dataset=dataset,
            output_dir=output_dir,
            manifest_records=manifest_records,
            audio_base_dir=audio_base_dir,
            save_context_audio=save_context_audio,
            save_predicted_codes=save_predicted_codes,
        )

    @staticmethod
    def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device)
            elif isinstance(value, list):
                out[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
            else:
                out[key] = value
        return out

    @staticmethod
    def _copy_or_link(src: Optional[str], dst: str, required: bool = False, description: str = "audio") -> Optional[str]:
        """Copy/symlink an audio artifact and optionally fail fast if missing.

        Evaluation later expects target_audio_*.wav/context_audio_*.wav to exist.
        Silently skipping those files makes evaluate_generated_audio_dir fail much
        later with a less useful FileNotFoundError, so target/context paths should
        call this with required=True.
        """
        if src is None or src == "":
            if required:
                raise FileNotFoundError(f"Missing required {description}: source path is empty for destination {dst}")
            return None
        if not os.path.exists(src):
            if required:
                raise FileNotFoundError(
                    f"Missing required {description}: source path does not exist: {src}; destination would be: {dst}"
                )
            return None

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copyfile(src, dst)

        if required and not os.path.exists(dst):
            raise FileNotFoundError(
                f"Failed to materialize required {description}: src={src}, dst={dst}. "
                "The destination may be a broken symlink."
            )
        return dst

    def _resolve_audio_path(self, path: Optional[str], audio_base_dir: str) -> Optional[str]:
        if path is None or path == "":
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(audio_base_dir, path)

    def _ensure_codec_silence_codes(self) -> torch.Tensor:
        """Ensure silence codec codes exist before streaming_prefill_profile.

        Newer EasyMagpieTTSInferenceModel exposes codec_sil_codes as a @property
        backed by _codec_sil_codes_buffer. Some older branches/checkpoints do not
        have that property, but still have _generate_codec_silence_buffer(). This
        helper supports both cases and creates a plain module attribute fallback
        when the property is absent.
        """
        if not hasattr(self.model, "_codec_sil_codes_buffer"):
            if not hasattr(self.model, "_generate_codec_silence_buffer"):
                raise AttributeError(
                    "Model does not have _codec_sil_codes_buffer or _generate_codec_silence_buffer(); "
                    "cannot run multiturn_user_audio prefill."
                )
            self.model._generate_codec_silence_buffer()

        class_codec_sil_codes = getattr(type(self.model), "codec_sil_codes", None)
        if class_codec_sil_codes is None:
            # Compatibility with branches where streaming_prefill_profile expects
            # self.codec_sil_codes but the @property was not added.
            self.model.codec_sil_codes = self.model._codec_sil_codes_buffer
            if hasattr(self.model, "_codec_sil_codes_buffer_unconverted"):
                self.model.codec_sil_codes_unconverted = self.model._codec_sil_codes_buffer_unconverted

        return self.model._codec_sil_codes_buffer.to(self.model.device).long()

    def _run_multiturn_generation(self, batch: Dict[str, Any]):
        model = self.model
        device = model.device
        B = int(batch["context_audio"].size(0))
        if B != 1:
            raise RuntimeError("multiturn_user_audio generation requires batch_size=1.")

        with torch.inference_mode():
            # streaming_prefill_profile reads self.model.codec_sil_codes, so make
            # sure the silence buffer/property exists before entering the turn loop.
            self._ensure_codec_silence_codes()

            wav = batch["context_audio"]
            wav_len = batch["context_audio_lengths"]
            codes, codes_lens = model._codec_helper.audio_to_codes(wav, wav_len)

            use_lang = bool(getattr(model, "add_language_to_context_text", False))
            language = getattr(self.config, "language", "en")
            ctx_text = f"[{language.upper()}]" if use_lang else "[NO TEXT CONTEXT]"
            ctx_text_ids = model.tokenizer.encode(ctx_text, tokenizer_name=model.text_conditioning_tokenizer_name)
            ctx_toks = torch.tensor([ctx_text_ids], dtype=torch.long, device=device).expand(B, -1)
            ctx_toks_lens = torch.tensor([len(ctx_text_ids)] * B, dtype=torch.long, device=device)

            params = self.config.model_inference_parameters
            state = model.streaming_init(
                context_audio_codes=codes,
                context_audio_codes_lens=codes_lens,
                context_text_tokens=ctx_toks,
                context_text_tokens_lens=ctx_toks_lens,
                use_cfg=self.config.use_cfg,
                cfg_scale=params.cfg_scale,
                use_local_transformer=self.config.use_local_transformer,
                temperature=params.temperature,
                topk=params.topk,
                phoneme_input_type="pred",
                phoneme_sampling_method=self.config.phoneme_sampling_method,
                use_inference_mode=True,
            )

            turn_frame_ranges = []
            decode_start_frame = 0
            max_decoder_steps = params.max_decoder_steps

            for turn_id in range(len(batch["batched_turns"])):
                turn_text = batch["batched_turns"][turn_id].to(device)
                turn_lens = batch["batched_turn_lens"][turn_id].to(device)
                valid_mask = batch["valid_turn_masks"][turn_id].to(device)
                if not bool(valid_mask[0].item()):
                    continue

                state.finished.zero_()
                state.text_finished.zero_()
                state.audio_prediction_end_idx.fill_(-1)
                for attr in [
                    "turn_text_tokens_seen",
                    "phoneme_steps",
                    "phoneme_stream_ended",
                    "phoneme_eos_detected",
                ]:
                    if hasattr(state, attr):
                        getattr(state, attr).zero_()
                state.last_phoneme_tokens = None

                if not model.cfg.get("condition_on_user_speech", False):
                    user_audio = batch["user_audio_turns"][turn_id]
                    user_audio_prefill_steps = int(round(user_audio.size(-1) / model.input_samples_per_frame))
                    user_audio_prefill_tokens = torch.full(
                        (1, user_audio_prefill_steps), model.pad_id, dtype=torch.long, device=device
                    )
                    user_audio_channel_embedding = None
                else:
                    user_audio = batch["user_audio_turns"][turn_id]
                    user_audio_lens = batch["user_audio_turns_lens"][turn_id]
                    user_audio_codes, user_audio_codes_lens = model._codec_helper.audio_to_codes(
                        user_audio, user_audio_lens
                    )

                    if model._codec_converter is not None:
                        user_audio_codes = model._codec_converter.convert_original_to_new(
                            audio_tokens=user_audio_codes,
                            audio_lens=user_audio_codes_lens,
                        ).long()

                    user_audio_codes, user_audio_codes_lens = model.stack_codes(
                        user_audio_codes,
                        user_audio_codes_lens,
                        model.audio_bos_id,
                        model.audio_eos_id,
                        model.frame_stacking_factor,
                        model.num_audio_codebooks,
                    )

                    user_audio_embedded = model.embed_audio_tokens(user_audio_codes)
                    boundary_trim = model.cfg.get("user_audio_boundary_trim", 0)
                    boundary_trim = 0 if boundary_trim is None else int(boundary_trim)

                    if boundary_trim == 0:
                        real_start = 0
                        real_end = int(user_audio_codes_lens[0].item())
                    else:
                        real_start = 1
                        real_end = max(real_start, int(user_audio_codes_lens[0].item()) - 1)

                    user_audio_embedded = user_audio_embedded[:, real_start:real_end]
                    copy_len = user_audio_embedded.size(1)
                    if boundary_trim > 0:
                        trim = min(boundary_trim, copy_len // 2)
                        if trim > 0:
                            user_audio_embedded[:, :trim] = 0.0
                            user_audio_embedded[:, copy_len - trim :] = 0.0

                    bos_user_pad = torch.zeros(
                        user_audio_embedded.size(0),
                        1,
                        user_audio_embedded.size(2),
                        device=user_audio_embedded.device,
                        dtype=user_audio_embedded.dtype,
                    )
                    user_audio_embedded = torch.cat([bos_user_pad, user_audio_embedded], dim=1)
                    user_audio_prefill_steps = user_audio_embedded.size(1)
                    user_audio_prefill_tokens = torch.full(
                        (B, user_audio_prefill_steps), model.pad_id, dtype=torch.long, device=device
                    )
                    user_audio_channel_embedding = user_audio_embedded

                delay_tokens = int(state.config.training_mode.streaming_speech_delay)
                delay_tokens = min(delay_tokens, int(turn_lens[0].item()), user_audio_prefill_steps)

                warmup_tokens = turn_text[:, :delay_tokens]
                turn_text = turn_text[:, delay_tokens:]
                turn_lens = torch.clamp(turn_lens - delay_tokens, min=0)

                if user_audio_channel_embedding is not None and delay_tokens > 0:
                    warmup_user_audio = user_audio_channel_embedding[:, -delay_tokens:]
                    user_audio_channel_embedding = user_audio_channel_embedding[:, :-delay_tokens]
                    user_audio_prefill_tokens = user_audio_prefill_tokens[:, :-delay_tokens]
                else:
                    warmup_user_audio = None

                if user_audio_prefill_tokens.size(1) > 0:
                    state = model.streaming_prefill_profile(
                        state=state,
                        text_tokens=user_audio_prefill_tokens,
                        use_inference_mode=True,
                        user_audio_channel_embedding=user_audio_channel_embedding,
                    )

                for i in range(delay_tokens):
                    user_step_emb = warmup_user_audio[:, i] if warmup_user_audio is not None else None
                    state.finished.zero_()
                    state, _, _ = model.streaming_step(
                        state=state,
                        text_tokens=warmup_tokens[:, i],
                        user_audio_channel_embedding=user_step_emb,
                        prefill_like_step=not bool(model.cfg.get("agent_mask_include_transition_prefix", False)),
                        prefill_like_is_last_step=(i == delay_tokens - 1),
                        use_inference_mode=True,
                    )

                turn_start_frame = sum(p.size(-1) for p in state.all_predictions)
                if turn_id == 0:
                    state.audio_prediction_start_idx.fill_(turn_start_frame)
                    decode_start_frame = turn_start_frame

                turn_offset = state.text_tokens_seen.clone()
                steps = 0
                while steps < max_decoder_steps:
                    steps += 1
                    state.finished.zero_()
                    relative_position = state.text_tokens_seen - turn_offset
                    text_exhausted = relative_position >= turn_lens

                    if turn_text.size(1) == 0:
                        current_tokens = torch.full((B,), model.eos_id, dtype=torch.long, device=device)
                    else:
                        position = relative_position.clamp(min=0, max=turn_text.size(1) - 1)
                        current_tokens = turn_text[torch.arange(B, device=device), position]
                        current_tokens = torch.where(
                            text_exhausted,
                            torch.full_like(current_tokens, model.eos_id),
                            current_tokens,
                        )

                    state, _, _ = model.streaming_step(
                        state=state,
                        text_tokens=current_tokens,
                        use_inference_mode=True,
                    )

                    if bool(text_exhausted[0].item()) and bool(state.finished[0].item()):
                        break

                state.audio_prediction_end_idx.fill_(-1)
                state.finished.zero_()
                turn_end_frame = sum(p.size(-1) for p in state.all_predictions)
                turn_frame_ranges.append((turn_id, turn_start_frame, turn_end_frame))

            codec_sil_codes = self._ensure_codec_silence_codes()
            bos_id = getattr(model, "audio_bos_id", -1)
            eos_id = getattr(model, "audio_eos_id", -1)
            speaking_id = getattr(model, "audio_user_speaking_id", -1)
            speaking_end_id = getattr(model, "audio_user_speaking_end_id", -1)
            sil_injection = codec_sil_codes.view(1, -1, 1)

            for step_idx in range(len(state.all_predictions)):
                pred = state.all_predictions[step_idx]
                mask = (pred == bos_id) | (pred == eos_id) | (pred == speaking_id) | (pred == speaking_end_id)
                frame_mask = mask.any(dim=1, keepdim=True)
                if frame_mask.any():
                    state.all_predictions[step_idx] = torch.where(frame_mask, sil_injection.expand_as(pred), pred)

            state.audio_prediction_end_idx.fill_(-1)
            generated_codes = None
            if getattr(state, "all_predictions", None):
                generated_codes = torch.cat(state.all_predictions, dim=-1).detach()

            finalize_output = model.streaming_finalize(state, use_inference_mode=True)

        return finalize_output, turn_frame_ranges, decode_start_frame, generated_codes

    @staticmethod
    def _save_code_slice(generated_codes, batch_idx: int, start_frame: int, end_frame: int, path: str) -> Optional[str]:
        if generated_codes is None:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        total_frames = int(generated_codes.size(-1))
        start_frame = max(0, min(int(start_frame), total_frames))
        end_frame = max(start_frame, min(int(end_frame), total_frames))
        if end_frame <= start_frame:
            return None
        codes = generated_codes[batch_idx, :, start_frame:end_frame].detach().cpu().long()
        torch.save(codes, path)
        return path

    def _resolve_target_audio_for_turn(
        self,
        raw_record: dict,
        target_turn_audio_paths,
        local_turn_idx: int,
        audio_base_dir: str,
    ) -> Optional[str]:
        """Resolve the GT target audio for one evaluation turn.

        Prefer per-turn GT if present; otherwise fall back to sample-level audio_filepath.
        If no candidate exists, returns None so the caller can fall back to
        context audio and keep EasyMagpie evaluation from failing on missing
        target_audio_*.wav.
        """
        candidates = []

        if isinstance(target_turn_audio_paths, list) and local_turn_idx < len(target_turn_audio_paths):
            candidates.append(target_turn_audio_paths[local_turn_idx])
        elif isinstance(target_turn_audio_paths, str):
            candidates.append(target_turn_audio_paths)

        # Common manifest keys.
        candidates.extend(
            [
                raw_record.get("target_audio_file_path"),
                raw_record.get("target_audio_filepath"),
                raw_record.get("audio_filepath"),
            ]
        )

        tried = []
        for candidate in candidates:
            if candidate is None or candidate == "":
                continue
            if isinstance(candidate, list):
                if local_turn_idx < len(candidate):
                    candidate = candidate[local_turn_idx]
                else:
                    continue
            resolved = self._resolve_audio_path(candidate, audio_base_dir)
            tried.append(resolved)
            if resolved is not None and os.path.exists(resolved):
                return resolved

        logging.warning(
            "Could not resolve target audio for multiturn_user_audio evaluation turn; "
            "caller will fall back to context audio. "
            f"sample_idx={raw_record.get('idx')}, local_turn_idx={local_turn_idx}, "
            f"audio_base_dir={audio_base_dir}, tried={tried}, "
            f"raw_record_keys={sorted(raw_record.keys())}"
        )
        return None

    def _run_multiturn_user_audio_inference(
        self,
        dataset: EasyMagpieMultiturnUserAudioDataset,
        output_dir: str,
        manifest_records: List[dict],
        audio_base_dir: str,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        os.makedirs(output_dir, exist_ok=True)
        self._delete_old_generated_files(output_dir)

        debug_user_dir = os.path.join(output_dir, "debug_user_turns")
        debug_mixed_dir = os.path.join(output_dir, "debug_mixed_user_agent")
        if self.config.save_debug_multiturn_audio:
            os.makedirs(debug_user_dir, exist_ok=True)
            os.makedirs(debug_mixed_dir, exist_ok=True)

        rank = int(getattr(self, "distributed_rank", 0))
        world_size = int(getattr(self, "distributed_world_size", 1))
        if world_size > 1:
            rank_indices = list(range(rank, len(dataset), world_size))
            logging.info(
                f"multiturn_user_audio distributed sharding: rank={rank}/{world_size}, "
                f"local_samples={len(rank_indices)}, total_samples={len(dataset)}"
            )
            dataset_for_rank = _InferenceSubset(dataset, rank_indices)
        else:
            dataset_for_rank = dataset

        dataloader = torch.utils.data.DataLoader(
            dataset_for_rank,
            batch_size=1,
            collate_fn=dataset_for_rank.collate_fn,
            num_workers=0,
            shuffle=False,
        )

        all_rtf_metrics = []
        generated_audio_paths = []
        codec_file_paths = []
        turn_manifest_records = []
        item_idx = 0

        sample_rate = getattr(self.model, "output_sample_rate", self.model.sample_rate)

        for batch_idx, batch in enumerate(dataloader):
            logging.info(f"Processing multiturn user-audio sample {batch_idx + 1}/{len(dataloader)}")
            batch = self._move_batch_to_device(batch, self.model.device)
            sample_idx = int(batch["idx"][0].item())
            raw_record = batch["raw_record"]
            raw_turn_texts = batch["raw_turn_texts"][0]

            start_time = time.time()
            output, turn_frame_ranges, decode_start_frame, generated_codes = self._run_multiturn_generation(batch)
            elapsed = time.time() - start_time

            predicted_audio = output.audio.float().detach().cpu()
            predicted_audio_lens = output.audio_len.int().detach().cpu()
            full_len = int(predicted_audio_lens[0].item())
            full_wav = predicted_audio[0, :full_len]

            samples_per_prediction_frame = self.model.codec_model_samples_per_frame / (
                self.model.sample_rate / sample_rate
            )
            aligned_agent = torch.zeros_like(full_wav)

            context_len = int(batch["context_audio_lengths"][0].detach().cpu().item())
            context_wav = batch["context_audio"][0, :context_len].detach().cpu().float()
            context_audio_path = os.path.join(output_dir, f"context_audio_sample_{sample_idx}.wav")
            # Always write context audio because evaluate_generated_audio_dir reads
            # context_audio_filepath from the generated turn-level manifest for every repeat.
            sf.write(context_audio_path, context_wav.numpy(), self.model.sample_rate)

            target_turn_audio_paths = batch.get("target_turn_audio_paths")

            for local_turn_idx, (turn_id, start_frame, end_frame) in enumerate(turn_frame_ranges):
                rel_start_frame = start_frame - decode_start_frame
                rel_end_frame = end_frame - decode_start_frame
                start_sample = int(round(rel_start_frame * samples_per_prediction_frame))
                end_sample = int(round(rel_end_frame * samples_per_prediction_frame))
                start_sample = max(0, min(start_sample, full_len))
                end_sample = max(start_sample, min(end_sample, full_len))

                aligned_agent[start_sample:end_sample] = full_wav[start_sample:end_sample]
                turn_wav = aligned_agent[start_sample:end_sample].float()

                predicted_audio_path = os.path.join(output_dir, f"predicted_audio_{item_idx}.wav")
                sf.write(predicted_audio_path, turn_wav.numpy(), sample_rate)
                generated_audio_paths.append(predicted_audio_path)

                if save_predicted_codes:
                    code_path = os.path.join(output_dir, f"predicted_codes_{item_idx}.pt")
                    saved_code_path = self._save_code_slice(generated_codes, 0, start_frame, end_frame, code_path)
                    if saved_code_path is not None:
                        codec_file_paths.append(saved_code_path)

                turn_context_path = os.path.join(output_dir, f"context_audio_{item_idx}.wav")
                self._copy_or_link(
                    context_audio_path,
                    turn_context_path,
                    required=True,
                    description=f"context audio for sample_idx={sample_idx}, turn_id={turn_id}",
                )

                target_src = self._resolve_target_audio_for_turn(
                    raw_record=raw_record,
                    target_turn_audio_paths=target_turn_audio_paths,
                    local_turn_idx=local_turn_idx,
                    audio_base_dir=audio_base_dir,
                )
                if target_src is None or not os.path.exists(target_src):
                    logging.warning(
                        "Target audio is missing for multiturn_user_audio evaluation; "
                        "using context audio as target fallback to avoid evaluator failure. "
                        f"sample_idx={sample_idx}, turn_id={turn_id}, missing_target={target_src}, "
                        f"context_audio_path={context_audio_path}"
                    )
                    target_src = context_audio_path

                target_dst = os.path.join(output_dir, f"target_audio_{item_idx}.wav")
                self._copy_or_link(
                    target_src,
                    target_dst,
                    required=True,
                    description=f"target audio fallback/context for sample_idx={sample_idx}, turn_id={turn_id}",
                )

                turn_manifest_records.append(
                    {
                        "audio_filepath": f"target_audio_{item_idx}.wav",
                        "context_audio_filepath": f"context_audio_{item_idx}.wav",
                        "text": raw_turn_texts[local_turn_idx] if local_turn_idx < len(raw_turn_texts) else "",
                        "speaker": str(sample_idx),
                        "source_sample_idx": sample_idx,
                        "turn_id": int(turn_id),
                    }
                )
                item_idx += 1

            full_agent_path = os.path.join(output_dir, f"predicted_audio_sample_{sample_idx}_full_agent.wav")
            sf.write(full_agent_path, aligned_agent.numpy(), sample_rate)

            if self.config.save_debug_multiturn_audio and "user_audio_turns" in batch:
                self._save_debug_user_agent_audio(
                    batch=batch,
                    sample_idx=sample_idx,
                    turn_frame_ranges=turn_frame_ranges,
                    decode_start_frame=decode_start_frame,
                    aligned_agent=aligned_agent,
                    samples_per_prediction_frame=samples_per_prediction_frame,
                    output_dir=output_dir,
                    debug_user_dir=debug_user_dir,
                    debug_mixed_dir=debug_mixed_dir,
                )

            audio_seconds = sum(
                max(0, int(round((end - start) * samples_per_prediction_frame))) for _, start, end in turn_frame_ranges
            ) / sample_rate
            all_rtf_metrics.append(
                {
                    "inference_time": elapsed,
                    "audio_seconds": audio_seconds,
                    "rtf": elapsed / audio_seconds if audio_seconds > 0 else 0.0,
                }
            )

        self.evaluation_audio_dir = output_dir
        rank = int(getattr(self, "distributed_rank", 0))
        self.evaluation_manifest_path = os.path.join(output_dir, f"multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl")
        self.evaluation_manifest_records = turn_manifest_records
        with open(self.evaluation_manifest_path, "w", encoding="utf-8") as f:
            for record in turn_manifest_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(f"Wrote multiturn turn-level evaluation manifest: {self.evaluation_manifest_path}")
        return all_rtf_metrics, generated_audio_paths, codec_file_paths

    def _save_debug_user_agent_audio(
        self,
        batch: Dict[str, Any],
        sample_idx: int,
        turn_frame_ranges: List[Tuple[int, int, int]],
        decode_start_frame: int,
        aligned_agent: torch.Tensor,
        samples_per_prediction_frame: float,
        output_dir: str,
        debug_user_dir: str,
        debug_mixed_dir: str,
    ) -> None:
        sample_rate = getattr(self.model, "output_sample_rate", self.model.sample_rate)
        first_user_len_in = int(batch["user_audio_turns_lens"][0][0].detach().cpu().item())
        first_user_delay_out = int(round(first_user_len_in * sample_rate / self.model.sample_rate))

        user_segments = []
        for turn_id, _, _ in turn_frame_ranges:
            if turn_id >= len(batch["user_audio_turns"]):
                continue
            turn_audio = batch["user_audio_turns"][turn_id][0].detach().cpu().float()
            turn_audio_len = int(batch["user_audio_turns_lens"][turn_id][0].detach().cpu().item())
            turn_audio = turn_audio[:turn_audio_len]
            turn_audio_out = resample(turn_audio.unsqueeze(0), self.model.sample_rate, sample_rate).squeeze(0)

            user_turn_path = os.path.join(debug_user_dir, f"sample_{sample_idx}_user_turn_{turn_id}.wav")
            sf.write(user_turn_path, turn_audio_out.numpy(), sample_rate)

            if turn_id == 0:
                user_start_sample = 0
            else:
                prev_turn_end_frame = turn_frame_ranges[turn_id - 1][2]
                rel_prev_end_frame = prev_turn_end_frame - decode_start_frame
                user_start_sample = first_user_delay_out + int(round(rel_prev_end_frame * samples_per_prediction_frame))
            user_segments.append((user_start_sample, turn_audio_out))

        total_user_len = 0
        for start, wav in user_segments:
            total_user_len = max(total_user_len, start + wav.numel())
        user_ch = torch.zeros(total_user_len)
        for start, wav in user_segments:
            user_ch[start : start + wav.numel()] += wav

        agent_ch = torch.cat([torch.zeros(first_user_delay_out, dtype=aligned_agent.dtype), aligned_agent])
        stereo_len = max(user_ch.numel(), agent_ch.numel())
        user_pad = torch.zeros(stereo_len)
        agent_pad = torch.zeros(stereo_len)
        user_pad[: user_ch.numel()] = user_ch
        agent_pad[: agent_ch.numel()] = agent_ch

        mono_mix = torch.clamp(user_pad + agent_pad, min=-1.0, max=1.0)
        sf.write(os.path.join(debug_mixed_dir, f"sample_{sample_idx}_user_agent_mixed_mono.wav"), mono_mix.numpy(), sample_rate)
        stereo = torch.stack([user_pad, agent_pad], dim=1).numpy()
        sf.write(os.path.join(debug_mixed_dir, f"sample_{sample_idx}_user_agent_aligned_stereo.wav"), stereo, sample_rate)
