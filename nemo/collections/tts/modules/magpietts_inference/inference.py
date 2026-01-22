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
Core inference logic for MagpieTTS.

This module provides:
- InferenceConfig: Dataclass for inference hyperparameters
- MagpieInferenceRunner: Class for running batch inference with a loaded model
  (supports auto-detection of longform text via longform_mode="auto")
"""
from __future__ import annotations

import glob
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch
from PIL import Image

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer
from nemo.collections.tts.data.text_to_speech_dataset import LongFormTTSInferenceDataset, MagpieTTSDataset
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.models.magpietts import ModelInferenceParameters
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from nemo.utils import logging


@dataclass
class InferenceConfig:
    """Configuration for MagpieTTS inference.

    Attributes:
        batch_size: Batch size for inference.
        use_cfg: Whether to use classifier-free guidance.
        apply_attention_prior: Whether to apply attention prior during decoding.

        # Model specific inference parameters
        model_inference_parameters: See ModelInferenceParameters dataclass

        # Local transformer / MaskGit parameters
        use_local_transformer: Whether to use local transformer for inference.
        maskgit_n_steps: Number of MaskGit refinement steps.
        maskgit_noise_scale: Noise scale for MaskGit sampling.
        maskgit_fixed_schedule: Fixed schedule for MaskGit (optional).
        maskgit_sampling_type: Type of MaskGit sampling.

        # Longform inference mode
        longform_mode: Longform inference mode ("auto", "always", "never").
        longform_word_threshold: Word threshold for auto-detection.
    """

    # Core sampling parameters
    batch_size: int = 32
    use_cfg: bool = False
    apply_attention_prior: bool = False
    model_inference_parameters: ModelInferenceParameters = field(default_factory=ModelInferenceParameters)

    # Local transformer / MaskGit parameters
    use_local_transformer: bool = False
    maskgit_n_steps: int = 3
    maskgit_noise_scale: float = 0.0
    maskgit_fixed_schedule: Optional[List[int]] = None
    maskgit_sampling_type: Optional[str] = None

    # Longform inference mode
    longform_mode: str = "auto"  # "auto" | "always" | "never"
    longform_word_threshold: int = 40  # Word threshold for auto-detection

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

    @staticmethod
    def _format_layer_list(layers: Optional[List[int]]) -> str:
        """Format a list of layer indices as a compact string."""
        if layers is None:
            return "None"
        return "".join(str(_layer) for _layer in layers)


class MagpieInferenceRunner:
    """Runner class for MagpieTTS batch inference.

    Encapsulates the logic for running inference on a dataset, saving outputs,
    and collecting metrics.
    """

    def __init__(
        self,
        model: MagpieTTSModel,
        config: InferenceConfig,
    ):
        """Initialize the inference runner.

        Args:
            model: Loaded MagpieTTS model (should be on GPU and in eval mode).
            config: Inference configuration.
        """
        self.model = model
        self.config = config

        # Set phoneme probability to 1 for inference
        self._configure_tokenizer()

        # Cached state from create_dataset (set when create_dataset is called)
        self._use_longform: Optional[bool] = None
        self._manifest_records: Optional[List[dict]] = None
        self._audio_base_dir: Optional[str] = None

    def _configure_tokenizer(self) -> None:
        """Configure the tokenizer for inference (phoneme prob = 1.0)."""
        g2p = None
        if isinstance(self.model.tokenizer, AggregatedTTSTokenizer):
            g2p = self.model.tokenizer.tokenizers["english_phoneme"].g2p
        elif isinstance(self.model.tokenizer, IPATokenizer):
            g2p = self.model.tokenizer.g2p

        if g2p is not None:
            g2p.phoneme_probability = 1.0

    def _needs_longform_inference(self, manifest_records: List[dict]) -> bool:
        """Determine if any manifest entry needs longform inference.

        Checks if text exceeds character threshold OR has multiple sentences.

        Args:
            manifest_records: List of manifest record dictionaries.

        Returns:
            True if longform inference should be used, False otherwise.
        """
        if self.config.longform_mode == "always":
            return True
        if self.config.longform_mode == "never":
            return False

        # Auto-detection based on text characteristics
        for record in manifest_records:
            text = record.get('text', '')

            # Check word count
            word_count = len(text.split())
            if word_count >= self.config.longform_word_threshold:
                return True

        return False

    def create_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: Optional[float] = None,
        context_duration_max: Optional[float] = None,
    ) -> Union[MagpieTTSDataset, LongFormTTSInferenceDataset]:
        """Create a dataset for inference.

        Automatically creates the appropriate dataset type based on longform detection:
        - LongFormTTSInferenceDataset if longform text is detected
        - MagpieTTSDataset for standard inference

        Args:
            dataset_meta: Dataset metadata dictionary with 'manifest_path' and 'audio_dir'.
            context_duration_min: Minimum context duration (uses model default if None).
            context_duration_max: Maximum context duration (uses model default if None).

        Returns:
            Configured dataset instance (MagpieTTSDataset or LongFormTTSInferenceDataset).
        """
        # Use model defaults if not specified
        if context_duration_min is None:
            context_duration_min = self.model.cfg.get('context_duration_min', 5.0)
        if context_duration_max is None:
            context_duration_max = self.model.cfg.get('context_duration_max', 5.0)

        # For multi-encoder models, use fixed 5s context for fair evaluation
        if context_duration_min < 5.0 and context_duration_max > 5.0:
            context_duration_min = 5.0
            context_duration_max = 5.0

        # Read manifest and cache for later use
        dataset_name = list(dataset_meta.keys())[0]
        dataset_info = dataset_meta[dataset_name]
        manifest_path = dataset_info.get('manifest_path')
        audio_dir = dataset_info.get('audio_dir', '')
        logging.info(f"Dataset name: {dataset_name}, manifest_path: {manifest_path}, audio_dir: {audio_dir}")

        self._manifest_records = read_manifest(manifest_path)
        self._audio_base_dir = audio_dir
        # Determine longform mode and cache
        self._use_longform = self._needs_longform_inference(self._manifest_records)
        logging.info(f"Longform detection: {self._use_longform} (mode: {self.config.longform_mode})")

        # Create appropriate dataset type based on longform detection
        if self._use_longform:
            logging.info("Creating LongFormTTSInferenceDataset for longform inference")
            dataset = self._create_longform_dataset(dataset_meta, context_duration_min, context_duration_max)
        else:
            logging.info("Creating MagpieTTSDataset for standard inference")
            dataset = MagpieTTSDataset(
                dataset_meta=dataset_meta,
                sample_rate=self.model.output_sample_rate,
                min_duration=0.5,
                max_duration=20,
                codec_model_samples_per_frame=self.model.codec_model_samples_per_frame,
                bos_id=self.model.bos_id,
                eos_id=self.model.eos_id,
                num_audio_codebooks=self.model.num_audio_codebooks,
                prior_scaling_factor=None,
                load_cached_codes_if_available=False,
                dataset_type='test',
                tokenizer_config=None,
                load_16khz_audio=self.model.model_type == 'single_encoder_sv_tts',
                use_text_conditioning_tokenizer=self.model.use_text_conditioning_encoder,
                text_conditioning_tokenizer_name=self.model.text_conditioning_tokenizer_name,
                pad_context_text_to_max_duration=self.model.pad_context_text_to_max_duration,
                context_duration_min=context_duration_min,
                context_duration_max=context_duration_max,
            )
            # Attach model's tokenizer for standard dataset
            dataset.text_tokenizer = self.model.tokenizer

        return dataset

    def run_inference_on_dataset(
        self,
        dataset: Union[MagpieTTSDataset, LongFormTTSInferenceDataset],
        output_dir: str,
        manifest_records: Optional[List[dict]] = None,
        audio_base_dir: Optional[str] = None,
        save_cross_attention_maps: bool = True,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str]]:
        """Run inference on a dataset.

        Routes to standard or longform inference based on the cached detection
        from create_dataset(). Uses cached manifest_records and audio_base_dir
        if not provided.

        Args:
            dataset: The inference dataset (created by create_dataset()).
            output_dir: Directory to save generated audio and artifacts.
            manifest_records: Original manifest records (uses cached if None).
            audio_base_dir: Base directory for audio paths (uses cached if None).
            save_cross_attention_maps: Whether to save attention map images.
            save_context_audio: Whether to copy context audio files.
            save_predicted_codes: Whether to save predicted code files.

        Returns:
            Tuple of:
                - rtf_metrics: List of real-time factor metrics per batch.
                - generated_audio_paths: List of paths to generated audio files.
                - codec_file_paths: List of paths to predicted codes files.
        """
        # Use cached values if not provided
        if manifest_records is None:
            if self._manifest_records is None:
                raise ValueError("manifest_records not provided and not cached from create_dataset()")
            manifest_records = self._manifest_records

        if audio_base_dir is None:
            if self._audio_base_dir is None:
                raise ValueError("audio_base_dir not provided and not cached from create_dataset()")
            audio_base_dir = self._audio_base_dir

        # Route based on cached longform detection
        if self._use_longform:
            logging.info("Using longform inference path")
            return self._run_longform_inference(
                dataset, output_dir, manifest_records, audio_base_dir, save_context_audio, save_predicted_codes
            )
        else:
            logging.info("Using standard inference path")
            return self._run_standard_inference(
                dataset,
                output_dir,
                manifest_records,
                audio_base_dir,
                save_cross_attention_maps,
                save_context_audio,
                save_predicted_codes,
            )

    def _run_standard_inference(
        self,
        dataset: MagpieTTSDataset,
        output_dir: str,
        manifest_records: List[dict],
        audio_base_dir: str,
        save_cross_attention_maps: bool = True,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str]]:
        """Run standard single-pass inference on a dataset.

        Args:
            dataset: The inference dataset.
            output_dir: Directory to save generated audio and artifacts.
            manifest_records: Original manifest records for metadata.
            audio_base_dir: Base directory for resolving audio paths.
            save_cross_attention_maps: Whether to save attention map images.
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
            num_workers=2,
            shuffle=False,
        )

        item_idx = 0
        all_rtf_metrics = []
        generated_audio_paths = []
        codec_file_paths = []

        for batch_idx, batch in enumerate(dataloader):
            logging.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            # Move batch to GPU
            batch = self._batch_to_cuda(batch)

            batch['sample_rate'] = self.model.output_sample_rate
            batch['context_sample_rate'] = self.model.output_sample_rate

            # Overwrite the model's parameters since we want to use the arguments from the commandline
            self.model.inference_parameters = self.config.model_inference_parameters

            # Run inference
            start_time = time.time()
            output = self.model.infer_batch(
                batch,
                use_cfg=self.config.use_cfg,
                return_cross_attn_probs=save_cross_attention_maps,
                use_local_transformer_for_inference=self.config.use_local_transformer,
                maskgit_n_steps=self.config.maskgit_n_steps,
                maskgit_noise_scale=self.config.maskgit_noise_scale,
                maskgit_fixed_schedule=self.config.maskgit_fixed_schedule,
                maskgit_sampling_type=self.config.maskgit_sampling_type,
            )

            predicted_audio = output.predicted_audio
            predicted_audio_lens = output.predicted_audio_lens
            predicted_codes = output.predicted_codes
            predicted_codes_lens = output.predicted_codes_lens
            rtf_metrics = output.rtf_metrics
            cross_attention_maps = output.cross_attention_maps

            all_rtf_metrics.append(rtf_metrics)
            elapsed = time.time() - start_time
            logging.info(f"Batch inference time: {elapsed:.2f}s, output shape: {predicted_audio.size()}")

            # Save outputs for each item in batch
            for idx in range(predicted_audio.size(0)):
                # Save cross attention map
                if save_cross_attention_maps and cross_attention_maps is not None:
                    attn_map_image = Image.fromarray(cross_attention_maps[idx])
                    attn_map_path = os.path.join(output_dir, f"cross_attn_map_{item_idx}.png")
                    attn_map_image.save(attn_map_path)

                # Save predicted audio
                audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                audio_np = audio_np[: predicted_audio_lens[idx]]
                audio_path = os.path.join(output_dir, f"predicted_audio_{item_idx}.wav")
                sf.write(audio_path, audio_np, self.model.output_sample_rate)
                generated_audio_paths.append(audio_path)

                # Copy context and target audio if available
                if save_context_audio:
                    self._copy_reference_audio(
                        manifest_records[item_idx],
                        audio_base_dir,
                        output_dir,
                        item_idx,
                    )

                if save_predicted_codes:
                    codes_path = os.path.join(output_dir, f"predicted_codes_{item_idx}.pt")
                    predicted_codes_current = predicted_codes[idx, :, : predicted_codes_lens[idx]]  # C, T
                    torch.save(predicted_codes_current, codes_path)
                    codec_file_paths.append(codes_path)
                item_idx += 1

        return all_rtf_metrics, generated_audio_paths, codec_file_paths

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

    def _create_longform_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: Optional[float] = None,
        context_duration_max: Optional[float] = None,
    ) -> LongFormTTSInferenceDataset:
        """Create a longform dataset for inference.

        Args:
            dataset_meta: Dataset metadata dictionary (same format as MagpieTTSDataset).
            context_duration_min: Minimum context duration (uses model default if None).
            context_duration_max: Maximum context duration (uses model default if None).

        Returns:
            Configured LongFormTTSInferenceDataset instance.
        """
        # Use model defaults if not specified
        if context_duration_min is None:
            context_duration_min = self.model.cfg.get('context_duration_min', 5.0)
        if context_duration_max is None:
            context_duration_max = self.model.cfg.get('context_duration_max', 5.0)

        # For multi-encoder models, use fixed 5s context for fair evaluation
        if context_duration_min < 5.0 and context_duration_max > 5.0:
            context_duration_min = 5.0
            context_duration_max = 5.0

        # Determine tokenizer name
        tokenizer_name = "english_phoneme"
        if isinstance(self.model.tokenizer, AggregatedTTSTokenizer):
            tokenizer_name = "english_phoneme"

        # Create dataset - inherits from MagpieTTSDataset, so uses same dataset_meta format
        dataset = LongFormTTSInferenceDataset(
            dataset_meta=dataset_meta,
            sample_rate=self.model.output_sample_rate,
            tokenizer_name=tokenizer_name,
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

    def _run_longform_inference(
        self,
        dataset: LongFormTTSInferenceDataset,
        output_dir: str,
        manifest_records: List[dict],
        audio_base_dir: str,
        save_context_audio: bool = True,
        save_predicted_codes: bool = True,
    ) -> Tuple[List[dict], List[str], List[str]]:
        """Run longform inference with automatic sentence chunking.

        Processes text sentence-by-sentence using generate_long_form_speech().

        Args:
            dataset: LongFormTTSInferenceDataset created by create_dataset().
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
            logging.info(f"Processing batch {batch_idx + 1}/{len(dataloader)} (longform)")

            # Move batch tensors to CUDA
            batch = self._batch_to_cuda(batch)

            batch['sample_rate'] = self.model.output_sample_rate
            batch['context_sample_rate'] = self.model.output_sample_rate

            batch_size = len(batch['chunked_tokens'])
            max_num_chunks = max(len(tokens) for tokens in batch['chunked_tokens'])

            # Create longform chunk state for this batch
            chunk_state = self.model.create_longform_chunk_state(batch_size=batch_size)

            # Accumulators for predicted codes
            predicted_codes_per_sample = [[] for _ in range(batch_size)]
            predicted_codes_lens = [0 for _ in range(batch_size)]

            # Overwrite the model's parameters since we want to use the arguments from the commandline
            self.model.inference_parameters = self.config.model_inference_parameters

            start_time = time.time()
            # Iterate over text chunks (sentences)
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

                # Compute is_end_of_text flags
                is_end_of_text = self._compute_end_of_text_flags(
                    batch, chunk_idx, max_num_chunks, current_tokens_lens, batch_size
                )

                beginning_of_text = chunk_idx == 0

                # Call generate_long_form_speech
                output = self.model.generate_long_form_speech(
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

                # Unpack output - generate_long_form_speech returns InferBatchOutput
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
            logging.info(f"Batch longform inference time: {elapsed:.2f}s")

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

            predicted_audio, predicted_audio_lens, _ = self.model.codes_to_audio(
                predicted_codes, predicted_codes_lens_tensor
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

    def _compute_end_of_text_flags(
        self,
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
