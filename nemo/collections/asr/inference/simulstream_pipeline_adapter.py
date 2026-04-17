#!/usr/bin/env python3
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

"""
Adapter to use NeMo's native streaming pipelines with simulstream evaluation.

This adapter properly interfaces with NeMo's internal streaming API (transcribe_step)
rather than duplicating chunking/buffering logic. NeMo handles all buffering internally.

Key Insight:
    NeMo's pipelines already have complete streaming infrastructure:
    - Frame/FeatureBuffer creation
    - Buffering logic (BufferedPipeline / CacheAwarePipeline)
    - State management (StreamingState)
    - Translation integration (LLMTranslator)
    
    We just need to:
    1. Create Frame objects from audio chunks
    2. Call pipeline.transcribe_step()
    3. Convert TranscribeStepOutput → IncrementalOutput
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.utils import logging
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig


try:
    from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor
    from simulstream.server.speech_processors.incremental_output import IncrementalOutput

    SIMULSTREAM_AVAILABLE = True
except ImportError:
    SIMULSTREAM_AVAILABLE = False
    SpeechProcessor = object
    SAMPLE_RATE = 16000

    # Mock IncrementalOutput for type hints when simulstream not available
    class IncrementalOutput:
        def __init__(self, asr_partial="", asr_final="", translation_partial="", translation_final=""):
            pass


def load_nemo_config(config_path: str):
    """
    Load NeMo config using OmegaConf (NeMo's native config system).

    Args:
        config_path: Path to YAML config file

    Returns:
        DictConfig: OmegaConf configuration object
    """
    return OmegaConf.load(config_path)


def create_nemo_pipeline_from_config(config_path: str):
    """
    Create NeMo streaming pipeline directly from config file.

    Args:
        config_path: Path to NeMo YAML config

    Returns:
        BasePipeline: NeMo streaming pipeline
    """
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    cfg = load_nemo_config(config_path)
    return PipelineBuilder.build_pipeline(cfg)


class NeMoStreamingPipelineAdapter(SpeechProcessor):
    """
    Adapter to use NeMo's streaming pipelines with simulstream evaluation.

    Architecture:
        audio_chunk → Frame → pipeline.transcribe_step() → TranscribeStepOutput → IncrementalOutput

    The pipeline internally handles:
        - Buffering (cache-aware or buffered mode)
        - Feature extraction
        - ASR decoding (CTC/RNN-T)
        - Translation (optional, via LLMTranslator)
        - State management per stream
    """

    pipeline = None  # Class-level pipeline (shared across instances)
    output_manifest_path: Optional[str] = None
    wav_names: list[str] = []
    
    def __init__(self, config: SimpleNamespace):
        """
        Initialize adapter.

        Args:
            config: Configuration from simulstream (SimpleNamespace)
                    Note: Will be converted to OmegaConf DictConfig for NeMo
        """
        if not SIMULSTREAM_AVAILABLE:
            raise ImportError("simulstream is required. Install with: pip install simulstream")

        super().__init__(config)

        # Stream state
        self.stream_id = 0
        self.frame_count = 0
        self.is_first_chunk = True
        self._final_transcript_acc = ""
        self._final_translation_acc = ""
        self._last_partial_transcript = ""
        self._last_partial_translation = ""
        # Determine request type from config
        self.request_type = getattr(config, 'request_type', 'frame')
        if hasattr(config, 'streaming') and hasattr(config.streaming, 'request_type'):
            self.request_type = config.streaming.request_type
        self.latency_unit = getattr(config, 'latency_unit', 'word')
        if isinstance(self.latency_unit, str):
            self.latency_unit = self.latency_unit.lower()
        if self.latency_unit not in ("word", "char"):
            logging.warning(f"Unsupported latency_unit='{self.latency_unit}', defaulting to 'word'")
            self.latency_unit = "word"

        # Language settings (from runtime args)
        self.src_lang = None
        self.tgt_lang = None

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        """
        Load NeMo pipeline once (class-level, shared).

        Args:
            config: Configuration from simulstream
        """
        if cls.pipeline is not None:
            return  # Already loaded

        import atexit

        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

        # Convert SimpleNamespace to DictConfig
        # SimulStream uses SimpleNamespace for configuration, so we need to convert it to use in NeMo.
        cfg = OmegaConf.create(cls._namespace_to_dict(config))
        cls.cfg = cfg

        # Build pipeline using NeMo's factory
        cls.pipeline = PipelineBuilder.build_pipeline(cfg)
        cls.pipeline.open_session()

        cls.detailed_log_path = getattr(config, "detailed_log_path", None)

        # Output manifest path (optional, but enabled by default when metrics_log_file is available).
        cls.output_manifest_path = getattr(config, 'output_manifest_file', None) or getattr(
            config, 'output_manifest', None
        )
        # cls.detailed_output_path = Path(cls.output_manifest_path).parent / "detailed_output.jsonl"
        if cls.output_manifest_path is None:
            metrics_log_file = getattr(config, 'metrics_log_file', None)
            if metrics_log_file:
                metrics_path = Path(metrics_log_file)
                cls.output_manifest_path = str(metrics_path.parent / f"{metrics_path.stem}_pred_manifest.jsonl")

        if cls.output_manifest_path:
            # Truncate at start of run.
            Path(cls.output_manifest_path).write_text("", encoding="utf-8")
            logging.info(f"Prediction manifest output: {cls.output_manifest_path}")
        cls._wer_calculated = False

        # Load wav names from wav list if available.
        cls.wav_names = []
        wav_list_file = getattr(config, 'wav_list_file', None)
        if wav_list_file and Path(wav_list_file).exists():
            with open(wav_list_file, 'r', encoding='utf-8') as f:
                cls.wav_names = [line.strip() for line in f if line.strip()]

        cls._load_reference_manifest(config)

        # Register cleanup handler to properly shutdown vLLM on exit
        # Attempting to gracefully shut down vLLM engine, to get "ERROR 02-09 16:53:28 [core_client.py:610] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client."
        # Works for now, but returns warning.
        # TODO: Find a better way to gracefully shut down vLLM engine.
        atexit.register(cls.cleanup_model)

        logging.info(f"Loaded NeMo pipeline: {type(cls.pipeline).__name__}")
        logging.info(f"  ASR model: {cfg.asr.model_name}")
        if cfg.get('enable_nmt', False):
            logging.info(f"  NMT model: {cfg.nmt.model_name}")
            logging.info(f"  Translation: {cfg.nmt.source_language} → {cfg.nmt.target_language}")

        if cfg.get("per_stream_boosting") and cfg.per_stream_boosting.get("phrases_file"):
            boosting_model_alpha = cfg.per_stream_boosting.get("alpha", 1.0)
            with open(cfg.per_stream_boosting.phrases_file, "r", encoding="utf-8") as f:
                boosting_requests_raw = json.load(f)
                cls.per_stream_boosting_requests = [
                    BiasingRequestItemConfig(
                        BoostingTreeModelConfig(key_phrases_list=item["key_phrases_list"]),
                        boosting_model_alpha=boosting_model_alpha,
                    )
                    for item in boosting_requests_raw
                ]
            logging.info(
                f"Per-stream boosting enabled with weight {boosting_model_alpha:.2g}, "
                f"expected {len(cls.per_stream_boosting_requests)} ordered streams"
            )
        else:
            logging.info(
                "Per-stream boosting disabled; to enable, "
                "specify `per_stream_boosting.phrases_file` and `per_stream_boosting.alpha`"
            )


    @classmethod
    def _load_reference_manifest(cls, config: SimpleNamespace) -> None:
        """Load optional input manifest to copy reference text fields and enable WER calculation."""
        cls.reference_manifest_by_audio = {}
        cls.reference_manifest_by_basename = {}
        cls.reference_manifest_items_ordered = []

        # import pdb; pdb.set_trace()

        manifest_path = None
        for key in (
            "manifest",
            "manifest_file",
            "input_manifest",
            "input_manifest_file",
            "reference_manifest",
        ):
            value = getattr(config, key, None)
            if value:
                manifest_path = value
                break

        # import pdb; pdb.set_trace()
        if not manifest_path:
            return

        manifest_path = str(manifest_path)
        if not Path(manifest_path).exists():
            logging.warning(f"Reference manifest path not found: {manifest_path}")
            return

        manifest_dir = Path(manifest_path).parent
        loaded = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                audio = item.get("audio_filepath", "")
                if not audio:
                    continue
                audio_path = Path(audio)
                if not audio_path.is_absolute():
                    audio_path = manifest_dir / audio_path
                audio_abs = str(audio_path.resolve())
                cls.reference_manifest_by_audio[audio_abs] = item
                cls.reference_manifest_by_basename[audio_path.name] = item
                cls.reference_manifest_items_ordered.append(item)
                loaded += 1

        # import pdb; pdb.set_trace()
        logging.info(f"Loaded reference manifest entries: {loaded}")

    @staticmethod
    def _namespace_to_dict(obj):
        """Recursively convert SimpleNamespace to dict."""
        if isinstance(obj, SimpleNamespace):
            return {k: NeMoStreamingPipelineAdapter._namespace_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: NeMoStreamingPipelineAdapter._namespace_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NeMoStreamingPipelineAdapter._namespace_to_dict(item) for item in obj]
        return obj

    def set_source_language(self, language: str) -> None:
        """Set source language (simulstream interface)."""
        self.src_lang = language

    def set_target_language(self, language: str) -> None:
        """Set target language (simulstream interface)."""
        self.tgt_lang = language

    def process_chunk(self, audio: np.ndarray) -> IncrementalOutput:
        """
        Process audio chunk using NeMo's native streaming API.

        This creates a Frame or FeatureBuffer request (depending on config) and
        calls pipeline.transcribe_step(), which internally handles all buffering,
        feature extraction, and decoding.

        Auto-detects the last chunk by comparing chunk size to expected size.
        If chunk is smaller than expected, it's treated as the last chunk.
        NOTE: works only with batch size 1 (so does SimulStream).

        Args:
            audio: Audio chunk (numpy array, float32, mono, 16kHz)

        Returns:
            IncrementalOutput: Streaming results (partial/final ASR + translation)
        """
        from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

        # import pdb; pdb.set_trace()
        if audio.ndim > 1:
            raise ValueError("Simulstream processes only one audio at a time (batch size 1).")

        expected_chunk_size = int(16000 * self.speech_chunk_size)
        audio_length = len(audio)
        if audio_length < expected_chunk_size:
            audio = np.concatenate([audio, np.zeros(expected_chunk_size - audio_length)])
        # Convert audio to torch tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.pipeline.device)

        if self.is_first_chunk and self.per_stream_boosting_requests is not None:
            biasing_cfg = self.per_stream_boosting_requests[self.stream_id]
        else:
            biasing_cfg = None

        # Create request based on config's request_type
        if self.request_type == "feature_buffer":
            # Extract features first, then create FeatureBuffer
            features = self.pipeline.preprocessor(
                input_signal=audio_tensor.unsqueeze(0),
                length=torch.tensor([audio_length], device=self.pipeline.device),
            )[0].squeeze(0)

            request = FeatureBuffer(
                stream_id=self.stream_id,
                features=features,
                is_first=self.is_first_chunk,
                is_last=False,  # simulstream does not tell wether chunk is last or not, we handle right context with return_full_right_context
                length=features.shape[1],  # Valid feature length
                options=ASRRequestOptions(biasing_cfg=biasing_cfg) if self.is_first_chunk else None,
            )
        else:  # frame
            # Create Frame request (NeMo's native streaming input)
            request = Frame(
                stream_id=self.stream_id,
                samples=audio_tensor,
                is_first=self.is_first_chunk,
                is_last=False,  # simulstream does not tell wether chunk is last or not, we handle right context with return_full_right_context
                length=audio_length,  # Valid audio length (without padding)
                options=ASRRequestOptions(biasing_cfg=biasing_cfg) if self.is_first_chunk else None,
            )

        # Call NeMo's native streaming API
        # This internally handles: buffering → encoding → decoding → translation
        step_outputs = self.pipeline.transcribe_step([request])
        step_output = step_outputs[0]

        # Track final and latest partial outputs to write a NeMo-style prediction manifest line.
        self._final_transcript_acc += step_output.final_transcript or ""
        self._final_translation_acc += step_output.final_translation or ""
        self._last_partial_transcript = step_output.partial_transcript or ""
        if step_output.final_translation:
            self._last_partial_translation = step_output.final_translation
        elif step_output.partial_translation:
            self._last_partial_translation = step_output.partial_translation

        # Convert NeMo's output to simulstream's IncrementalOutput
        result = self._convert_to_incremental_output(step_output)

        self.is_first_chunk = False
        self.frame_count += 1

        if self.detailed_log_path is not None:
            with open(self.detailed_log_path, "a", encoding="utf-8") as f:
                print(
                    json.dumps(
                        {
                            "final_transcript": step_output.final_transcript,
                            "partial_transcript": step_output.partial_transcript,
                            "final_translation": step_output.final_translation,
                            "partial_translation": step_output.partial_translation,
                            "new_tokens": result.new_tokens,
                            "new_string": result.new_string,
                            "deleted_tokens": result.deleted_tokens,
                            "deleted_string": result.deleted_string,
                        }
                    ),
                    file=f,
                )

        return result

    def _convert_to_incremental_output(self, step_output) -> IncrementalOutput:
        """
        Convert NeMo's TranscribeStepOutput to simulstream's IncrementalOutput.

        Calculate generated and deleted tokens by comparing previous and current partial outputs.
        Uses tokenization based on latency_unit:
          - word: split by whitespace
          - char: split into individual characters
        TODO: Think more on how actualyy this tokenization should be done.

        Args:
            step_output: NeMo's TranscribeStepOutput object with:
                - previous_partial_transcript: Previous step's partial transcript
                - previous_partial_translation: Previous step's partial translation
                - partial_transcript: Current step's partial transcript
                - partial_translation: Current step's partial translation (if NMT enabled)

        Returns:
            IncrementalOutput: Simulstream format with generated/deleted token lists
        """

        prev_partial = step_output.previous_partial_translation
        if step_output.final_translation:
            current_partial = step_output.final_translation
        elif step_output.partial_translation:
            current_partial = step_output.partial_translation
        else:
            current_partial = ""

        print(f"Current partial: {current_partial}")
        prev_tokens = self._tokenize_text(prev_partial)
        curr_tokens = self._tokenize_text(current_partial)

        # Find longest common prefix to identify what changed
        common_prefix_len = 0
        for i in range(min(len(prev_tokens), len(curr_tokens))):
            if prev_tokens[i] == curr_tokens[i]:
                common_prefix_len += 1
            else:
                break

        # Calculate deleted and generated token lists
        deleted_tokens = prev_tokens[common_prefix_len:]  # Tokens removed from previous
        generated_tokens = curr_tokens[common_prefix_len:]  # Tokens added in current

        # Construct strings from token lists
        deleted_string = self._join_tokens(deleted_tokens)
        generated_string = self._join_tokens(generated_tokens)

        return IncrementalOutput(
            new_tokens=generated_tokens,  # List of string tokens added
            new_string=generated_string,
            deleted_tokens=deleted_tokens,  # List of string tokens removed
            deleted_string=deleted_string,
        )

    def end_of_stream(self) -> IncrementalOutput:
        """
        Called at the end of audio stream to finalize output.

        In most cases, the last chunk is auto-detected by size and processed with
        is_last=True in process_chunk(), so this returns empty output.

        This is kept as required by SpeechProcessor interface and serves as a
        fallback for edge cases where the last chunk has the same size as others.

        Returns:
            IncrementalOutput: Empty output in most cases
        """
        pred_text = (self._final_transcript_acc + self._last_partial_transcript).strip()
        pred_translation = (self._final_translation_acc + self._last_partial_translation).strip()
        self._write_prediction_manifest_line(pred_text, pred_translation)

        # NOTE: Last chunk was already processed with is_last=False in process_chunk().
        # We only finalize stream state and emit empty incremental output here.
        self.pipeline.delete_state(self.stream_id)
        return IncrementalOutput(
            new_tokens=[],
            new_string="",
            deleted_tokens=[],
            deleted_string="",
        )

    def clear(self) -> None:
        """
        Clear stream state and prepare for next audio (simulstream interface).

        This finalizes the current stream and resets state for a new one.
        """
        # Finalize current stream if we've processed anything
        if not self.is_first_chunk:
            self.end_of_stream()

        # Reset for next stream
        self.stream_id += 1
        self.frame_count = 0
        self.is_first_chunk = True
        self._final_transcript_acc = ""
        self._final_translation_acc = ""
        self._last_partial_transcript = ""
        self._last_partial_translation = ""

    def _write_prediction_manifest_line(self, pred_text: str, pred_translation: str) -> None:
        """Write one NeMo-style manifest line with model predictions."""
        if not self.output_manifest_path:
            return

        audio_filepath = ""
        if self.stream_id < len(self.wav_names):
            audio_filepath = self.wav_names[self.stream_id]

        reference_item = self._get_reference_item(audio_filepath)
        if not audio_filepath and reference_item is not None:
            audio_filepath = str(reference_item.get("audio_filepath", "") or "")
        reference_text = ""
        reference_translation = ""
        if reference_item is not None:
            reference_text = reference_item.get("text", "")
            reference_translation = reference_item.get("answer", "")

        item = {
            "audio_filepath": audio_filepath,
            # Reference fields
            "text": reference_text,
            "translation": reference_translation,
            # Prediction fields
            "pred_text": pred_text,
            "pred_translation": pred_translation,
        }

        if reference_item is not None:
            for key, value in reference_item.items():
                if key not in item:
                    item[key] = value

        with open(self.output_manifest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Compute WER once when we flush the final stream.
        if self.wav_names and self.stream_id == len(self.wav_names) - 1:
            self._calculate_and_write_wer()

    def _get_reference_item(self, audio_filepath: str) -> Optional[dict]:
        """Get reference manifest item by absolute path, basename, or stream order."""
        if not audio_filepath:
            if self.stream_id < len(self.reference_manifest_items_ordered):
                return self.reference_manifest_items_ordered[self.stream_id]
            return None
        try:
            audio_abs = str(Path(audio_filepath).resolve())
        except Exception:
            audio_abs = audio_filepath
        item = self.reference_manifest_by_audio.get(audio_abs)
        if item is not None:
            return item
        item = self.reference_manifest_by_basename.get(Path(audio_filepath).name)
        if item is not None:
            return item
        if self.stream_id < len(self.reference_manifest_items_ordered):
            return self.reference_manifest_items_ordered[self.stream_id]
        return None

    @classmethod
    def _calculate_and_write_wer(cls) -> None:
        """Calculate WER from output manifest and write summary artifacts."""
        if cls._wer_calculated or not cls.output_manifest_path:
            return

        gt_text_attr_name = "text"
        clean_groundtruth_text = False
        langid = "en"
        use_cer = False
        ignore_capitalization = False
        ignore_punctuation = False

        try:
            if cls.cfg is not None and cls.cfg.get("metrics") and cls.cfg.metrics.get("asr"):
                asr_cfg = cls.cfg.metrics.asr
                gt_text_attr_name = asr_cfg.get("gt_text_attr_name", gt_text_attr_name)
                clean_groundtruth_text = asr_cfg.get("clean_groundtruth_text", clean_groundtruth_text)
                langid = asr_cfg.get("langid", langid)
                use_cer = asr_cfg.get("use_cer", use_cer)
                ignore_capitalization = asr_cfg.get("ignore_capitalization", ignore_capitalization)
                ignore_punctuation = asr_cfg.get("ignore_punctuation", ignore_punctuation)
        except Exception as e:
            logging.warning(f"Failed to read ASR metric config, using defaults: {e}")

        try:
            output_manifest_w_wer, total_res, _ = cal_write_wer(
                pred_manifest=cls.output_manifest_path,
                gt_text_attr_name=gt_text_attr_name,
                pred_text_attr_name="pred_text",
                output_filename=None,
                clean_groundtruth_text=clean_groundtruth_text,
                langid=langid,
                use_cer=use_cer,
                ignore_capitalization=ignore_capitalization,
                ignore_punctuation=ignore_punctuation,
            )

            if output_manifest_w_wer:
                metrics_summary_path = str(Path(cls.output_manifest_path).with_suffix(".wer.txt"))
                with open(metrics_summary_path, "w", encoding="utf-8") as f:
                    f.write(str(total_res) + "\n")
                logging.info(f"WER manifest: {output_manifest_w_wer}")
                logging.info(f"WER summary: {metrics_summary_path}")
            else:
                logging.warning(
                    "WER calculation skipped because ground-truth text is unavailable in output manifest."
                )
        except Exception as e:
            logging.warning(f"Failed to calculate WER: {e}")
        finally:
            cls._wer_calculated = True

    def tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens to string using NeMo's tokenizer.

        Args:
            tokens: List of token strings (BPE/SentencePiece tokens)

        Returns:
            Detokenized string
        """
        return self._join_tokens(tokens)

    def _tokenize_text(self, text: Optional[str]) -> List[str]:
        """Tokenize text according to configured latency unit. For char-level, removes
        all spaces so emitted token count matches simulstream eval (MWER path does
        .replace(" ", "") on resegmented text, so delay count must be non-space chars only)."""
        if not text:
            return []
        # for compatability with omnisteval
        text = text.replace("…", "")
        if self.latency_unit == "char":
            return list(text.strip())
        return text.strip().split()

    def _join_tokens(self, tokens: List[str]) -> str:
        """Join tokens according to configured latency unit."""
        if not tokens:
            return ""
        if self.latency_unit == "char":
            return "".join(tokens)
        return " ".join(tokens)

    @classmethod
    def cleanup_model(cls):
        """
        Explicitly cleanup vLLM and release resources.
        Call this when done with inference to properly shutdown vLLM engine.
        """
        if cls.pipeline is not None:
            cls._calculate_and_write_wer()
        if cls.pipeline is not None and cls.pipeline.nmt_model is not None:
            try:
                # vLLM cleanup - destroy the engine to release Ray resources
                if hasattr(cls.pipeline.nmt_model, 'nmt_model'):
                    vllm_engine = cls.pipeline.nmt_model.nmt_model
                    if hasattr(vllm_engine, 'llm_engine'):
                        # Destroy the engine core
                        from vllm.distributed import destroy_model_parallel

                        destroy_model_parallel()
                    del vllm_engine
                    cls.pipeline.nmt_model.nmt_model = None
                    print("[NeMo Adapter] vLLM engine cleaned up")
            except Exception as e:
                print(f"[NeMo Adapter] Warning during vLLM cleanup: {e}")

    def __del__(self):
        """Cleanup when adapter is destroyed"""
        # Note: cleanup_model() is class-level, should be called explicitly
        # since multiple adapter instances share the same pipeline
        pass
