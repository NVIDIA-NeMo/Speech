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

from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.utils import logging

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
        # Determine request type from config
        self.request_type = getattr(config, 'request_type', 'frame')
        if hasattr(config, 'streaming') and hasattr(config.streaming, 'request_type'):
            self.request_type = config.streaming.request_type
        
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
        
        # Build pipeline using NeMo's factory
        cls.pipeline = PipelineBuilder.build_pipeline(cfg)
        cls.pipeline.open_session()
        
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
    
    @staticmethod
    def _namespace_to_dict(obj):
        """Recursively convert SimpleNamespace to dict."""
        if isinstance(obj, SimpleNamespace):
            return {k: NeMoStreamingPipelineAdapter._namespace_to_dict(v) 
                    for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: NeMoStreamingPipelineAdapter._namespace_to_dict(v) 
                    for k, v in obj.items()}
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

        # DEBUG: First chunk difference vs regular cache_aware script.
        # Simulstream uses speech_chunk_size (may differ from pipeline chunk_size_in_secs).
        # Frame length=audio_length, size=expected_chunk_size; bufferer gets [zeros, this chunk]
        # and uses valid_size for preprocess. If length/valid_size are wrong you can get
        # all LOG_MEL_ZERO (-16.635) in the first feature chunk.

        # Create request based on config's request_type
        if self.request_type == "feature_buffer":
            # Extract features first, then create FeatureBuffer
            features = self.pipeline.preprocessor(
                input_signal=audio_tensor.unsqueeze(0),
                length=torch.tensor([audio_length], device=self.pipeline.device)
            )[0].squeeze(0)
            
            request = FeatureBuffer(
                stream_id=self.stream_id,
                features=features,
                is_first=self.is_first_chunk,
                is_last=False, # simulstream does not tell wether chunk is last or not, we handle right context with return_full_right_context
                length=features.shape[1],  # Valid feature length
                options=ASRRequestOptions() if self.is_first_chunk else None
            )
        else:  # frame
            # Create Frame request (NeMo's native streaming input)
            request = Frame(
                stream_id=self.stream_id,
                samples=audio_tensor,
                is_first=self.is_first_chunk,
                is_last=False,  # simulstream does not tell wether chunk is last or not, we handle right context with return_full_right_context
                length=audio_length,  # Valid audio length (without padding)
                options=ASRRequestOptions() if self.is_first_chunk else None
            )
        
        # Call NeMo's native streaming API
        # This internally handles: buffering → encoding → decoding → translation
        step_outputs = self.pipeline.transcribe_step([request])
        step_output = step_outputs[0]
        
        # Convert NeMo's output to simulstream's IncrementalOutput
        result = self._convert_to_incremental_output(step_output)
        
        self.is_first_chunk = False
        self.frame_count += 1
        
        return result
    
    def _convert_to_incremental_output(self, step_output) -> IncrementalOutput:
        """
        Convert NeMo's TranscribeStepOutput to simulstream's IncrementalOutput.
        
        Calculate generated and deleted tokens by comparing previous and current partial outputs.
        Uses word-level tokenization (split by whitespace) for token-level tracking.
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
        
        # Tokenize by whitespace (word-level tokens)
        prev_tokens = prev_partial.split() if prev_partial else []
        curr_tokens = current_partial.split() if current_partial else []
        
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
        deleted_string = " ".join(deleted_tokens) if deleted_tokens else ""
        generated_string = " ".join(generated_tokens) if generated_tokens else ""
        
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
        # NOTE: Last chunk was already processed with is_last=True in process_chunk()
        # Nothing more to do - NeMo's buffers were already flushed
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
    
    def tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens to string using NeMo's tokenizer.
        
        Args:
            tokens: List of token strings (BPE/SentencePiece tokens)
            
        Returns:
            Detokenized string
        """
        # Use NeMo's tokenizer to properly detokenize
        # Just creating text as tokens are words for now.
        return " ".join(tokens)
    
    @classmethod
    def cleanup_model(cls):
        """
        Explicitly cleanup vLLM and release resources.
        Call this when done with inference to properly shutdown vLLM engine.
        """
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