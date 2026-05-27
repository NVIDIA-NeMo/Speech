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
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
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
