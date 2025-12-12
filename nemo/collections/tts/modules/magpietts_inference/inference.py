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
"""
from __future__ import annotations

import glob
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
from PIL import Image

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
from nemo.collections.tts.models import MagpieTTSModel
from nemo.utils import logging


@dataclass
class InferenceConfig:
    """Configuration for MagpieTTS inference.

    Attributes:
        temperature: Sampling temperature for token generation.
        topk: Top-k sampling parameter.
        max_decoder_steps: Maximum number of decoder steps.
        use_cfg: Whether to use classifier-free guidance.
        cfg_scale: Scale factor for classifier-free guidance.
        batch_size: Batch size for inference.

        # Attention prior parameters
        apply_attention_prior: Whether to apply attention prior during decoding.
        attention_prior_epsilon: Epsilon value for attention prior.
        attention_prior_lookahead_window: Lookahead window size for prior.
        estimate_alignment_from_layers: Layer indices for alignment estimation.
        apply_prior_to_layers: Layer indices to apply prior to.
        start_prior_after_n_audio_steps: When to start applying the prior.

        # Local transformer / MaskGit parameters
        use_local_transformer: Whether to use local transformer for inference.
        maskgit_n_steps: Number of MaskGit refinement steps.
        maskgit_noise_scale: Noise scale for MaskGit sampling.
        maskgit_fixed_schedule: Fixed schedule for MaskGit (optional).
        maskgit_sampling_type: Type of MaskGit sampling.

        # EOS detection
        eos_detection_method: Method for detecting end-of-sequence.
        ignore_finished_sentence_tracking: Whether to ignore sentence tracking.
    """

    # Core sampling parameters
    temperature: float = 0.6
    topk: int = 80
    max_decoder_steps: int = 440
    use_cfg: bool = False
    cfg_scale: float = 2.5
    batch_size: int = 32

    # Attention prior parameters
    apply_attention_prior: bool = False
    attention_prior_epsilon: float = 0.1
    attention_prior_lookahead_window: int = 5
    estimate_alignment_from_layers: Optional[List[int]] = None
    apply_prior_to_layers: Optional[List[int]] = None
    start_prior_after_n_audio_steps: int = 0

    # Local transformer / MaskGit parameters
    use_local_transformer: bool = False
    maskgit_n_steps: int = 3
    maskgit_noise_scale: float = 0.0
    maskgit_fixed_schedule: Optional[List[int]] = None
    maskgit_sampling_type: Optional[str] = None

    # EOS detection
    eos_detection_method: str = "argmax_or_multinomial_any"
    ignore_finished_sentence_tracking: bool = False

    def build_identifier(self) -> str:
        """Build a unique identifier string for this configuration.

        Used for naming output directories and files.

        Returns:
            String identifier incorporating key config values.
        """
        parts = [
            f"Temp{self.temperature}",
            f"Topk{self.topk}",
            f"Cfg_{self.use_cfg}_{self.cfg_scale}",
            f"Prior_{self.apply_attention_prior}",
        ]

        if self.apply_attention_prior:
            parts.extend(
                [
                    f"{self.attention_prior_epsilon}",
                    f"{self.attention_prior_lookahead_window}",
                    f"{self.start_prior_after_n_audio_steps}",
                    self._format_layer_list(self.estimate_alignment_from_layers),
                    self._format_layer_list(self.apply_prior_to_layers),
                ]
            )

        parts.extend(
            [
                f"LT_{self.use_local_transformer}",
                f"MaskGit_{self.maskgit_n_steps}_{self.maskgit_sampling_type}",
                self._format_layer_list(self.maskgit_fixed_schedule),
                f"EOS_{self.eos_detection_method}",
                f"IgnoreFST_{self.ignore_finished_sentence_tracking}",
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

    def _configure_tokenizer(self) -> None:
        """Configure the tokenizer for inference (phoneme prob = 1.0)."""
        g2p = None
        if isinstance(self.model.tokenizer, AggregatedTTSTokenizer):
            g2p = self.model.tokenizer.tokenizers["english_phoneme"].g2p
        elif isinstance(self.model.tokenizer, IPATokenizer):
            g2p = self.model.tokenizer.g2p

        if g2p is not None:
            g2p.phoneme_probability = 1.0

    def create_dataset(
        self,
        dataset_meta: dict,
        context_duration_min: Optional[float] = None,
        context_duration_max: Optional[float] = None,
    ) -> MagpieTTSDataset:
        """Create a dataset for inference.

        Args:
            dataset_meta: Dataset metadata dictionary.
            context_duration_min: Minimum context duration (uses model default if None).
            context_duration_max: Maximum context duration (uses model default if None).

        Returns:
            Configured MagpieTTSDataset instance.
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

        dataset = MagpieTTSDataset(
            dataset_meta=dataset_meta,
            sample_rate=self.model.sample_rate,
            min_duration=0.5,
            max_duration=20,
            codec_model_samples_per_frame=self.model.codec_model_samples_per_frame,
            bos_id=self.model.bos_id,
            eos_id=self.model.eos_id,
            context_audio_bos_id=self.model.context_audio_bos_id,
            context_audio_eos_id=self.model.context_audio_eos_id,
            audio_bos_id=self.model.audio_bos_id,
            audio_eos_id=self.model.audio_eos_id,
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

        # Attach model's tokenizer
        dataset.text_tokenizer = self.model.tokenizer

        return dataset

    def run_inference_on_dataset(
        self,
        dataset: MagpieTTSDataset,
        output_dir: str,
        manifest_records: List[dict],
        audio_base_dir: str,
        save_cross_attention_maps: bool = True,
        save_context_audio: bool = True,
    ) -> Tuple[List[dict], List[str]]:
        """Run inference on a dataset and save outputs.

        Args:
            dataset: The inference dataset.
            output_dir: Directory to save generated audio and artifacts.
            manifest_records: Original manifest records for metadata.
            audio_base_dir: Base directory for resolving audio paths.
            save_cross_attention_maps: Whether to save attention map images.
            save_context_audio: Whether to copy context audio files.

        Returns:
            Tuple of:
                - rtf_metrics: List of real-time factor metrics per batch.
                - generated_audio_paths: List of paths to generated audio files.
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

        for batch_idx, batch in enumerate(dataloader):
            logging.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

            # Move batch to GPU
            batch_cuda = self._batch_to_cuda(batch)

            # Run inference
            start_time = time.time()
            output = self.model.infer_batch(
                batch_cuda,
                max_decoder_steps=self.config.max_decoder_steps,
                temperature=self.config.temperature,
                topk=self.config.topk,
                use_cfg=self.config.use_cfg,
                cfg_scale=self.config.cfg_scale,
                return_cross_attn_probs=save_cross_attention_maps,
                apply_attention_prior=self.config.apply_attention_prior,
                prior_epsilon=self.config.attention_prior_epsilon,
                lookahead_window_size=self.config.attention_prior_lookahead_window,
                estimate_alignment_from_layers=self.config.estimate_alignment_from_layers,
                apply_prior_to_layers=self.config.apply_prior_to_layers,
                start_prior_after_n_audio_steps=self.config.start_prior_after_n_audio_steps,
                use_local_transformer_for_inference=self.config.use_local_transformer,
                maskgit_n_steps=self.config.maskgit_n_steps,
                maskgit_noise_scale=self.config.maskgit_noise_scale,
                maskgit_fixed_schedule=self.config.maskgit_fixed_schedule,
                maskgit_sampling_type=self.config.maskgit_sampling_type,
                ignore_finished_sentence_tracking=self.config.ignore_finished_sentence_tracking,
                eos_detection_method=self.config.eos_detection_method,
            )

            predicted_audio = output.predicted_audio
            predicted_audio_lens = output.predicted_audio_lens
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
                sf.write(audio_path, audio_np, self.model.sample_rate)
                generated_audio_paths.append(audio_path)

                # Copy context and target audio if available
                if save_context_audio:
                    self._copy_reference_audio(
                        manifest_records[item_idx],
                        audio_base_dir,
                        output_dir,
                        item_idx,
                    )

                item_idx += 1

        return all_rtf_metrics, generated_audio_paths

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
