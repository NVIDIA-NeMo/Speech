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

import copy
import os
import time

import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.inference.model_wrappers.decode_state import (
    InferenceStepResult,
    IntermediateResultLogger,
    NullIntermediateResultLogger,
    NullTimingSummary,
    StreamingDecodeState,
    TimingSummary,
)
from nemo.collections.speechlm2.inference.model_wrappers.factory import create_model
from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import (
    PerceptionCacheManager,
    PerceptionCacheState,
)
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.text_utils import (
    _decode_tokens_with_specials,
    get_special_token_ids,
    get_special_token_strings,
)
from nemo.utils import logging, str_to_dtype

# --- Configuration ---
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming Parameters ---
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples

TTS_SAMPLE_RATE = 22050


class NemotronVoicechatInferenceWrapper:
    """
    Inference wrapper for NemotronVoiceChat models.
    Uses a sliding window buffer and processes audio frame by frame.
    """

    def __init__(self, model_cfg: DictConfig):
        """
        Initialize the model for realtime streaming inference.

        Args:
            model_cfg (DictConfig): Configuration describing the model paths and inference parameters.
        """
        if model_cfg is None:
            raise ValueError("model_cfg must be provided")
        if not isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.create(model_cfg)

        # Precision settings (applied here so they take effect before model loading)
        allow_tf32 = bool(model_cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        matmul_precision = str(model_cfg.get("matmul_precision", "medium"))
        torch.set_float32_matmul_precision(matmul_precision)

        # Deterministic mode: guarantees identical *text* outputs (from the STT/LLM
        # heads) across runs for the same inputs, even when sampling is enabled
        # (top_p < 1, temperature != 1, repetition_penalty != 1).  This works
        # because we fix the global PyTorch RNG seeds and force all CUDA ops
        # to use deterministic algorithm implementations.
        # Not compatible with vLLM engines (raises an error below).
        self._deterministic = bool(model_cfg.get("deterministic", False))
        if self._deterministic:
            engine_type = model_cfg.get("engine_type", "native")
            if "vllm" in engine_type.lower():
                raise ValueError(
                    "`deterministic` is not compatible with vLLM engines because vLLM uses custom "
                    "CUDA kernels (PagedAttention, FlashAttention) that do not support deterministic mode. "
                    f"Got engine_type='{engine_type}'. Use engine_type='native' for deterministic inference."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.use_deterministic_algorithms(False)

        self.model_cfg = model_cfg

        self.model_path = model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("`model_cfg.model_path` must be provided.")

        self.decode_audio = bool(model_cfg.get("decode_audio", True))

        self.speaker_reference = model_cfg.get("speaker_reference")
        self.speaker_name = model_cfg.get("speaker_name", None)
        if self.decode_audio and not self.speaker_reference and not self.speaker_name:
            raise ValueError(
                "`model_cfg.speaker_reference` or `model_cfg.speaker_name` must be provided when decode_audio is enabled."
            )

        self.tts_system_prompt = model_cfg.get("tts_system_prompt", None)
        logging.info(f"TTS system prompt: {self.tts_system_prompt}")

        self.dtype = str_to_dtype(model_cfg.get("compute_dtype", "bfloat16"))

        device = model_cfg.get("device")
        device_id = model_cfg.get("device_id")
        if device is None:
            self.device = DEFAULT_DEVICE
        else:
            device_str = str(device)
            if device_id is not None and device_str.startswith("cuda") and ":" not in device_str:
                device_str = f"{device_str}:{device_id}"
            self.device = torch.device(device_str)

        logging.info("=" * 70)
        logging.info("INITIALIZING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)
        logging.info(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        logging.info(f"Device: {self.device}")
        logging.info(f"Compute dtype: {self.dtype}")
        logging.info(f"Decode audio: {self.decode_audio}")
        logging.info(f"Engine type: {model_cfg.get('engine_type', 'native')}")
        logging.info(
            f"Sampling - top_p: {model_cfg.get('top_p', 1.0)}, repetition_penalty: {model_cfg.get('repetition_penalty', 1.0)}, temperature: {model_cfg.get('temperature', 1.0)}"
        )
        logging.info(
            f"Precision (configured): matmul_precision={matmul_precision}, allow_tf32={allow_tf32}, deterministic={self._deterministic}"
        )
        logging.info(
            f"Precision (effective): float32_matmul_precision={torch.get_float32_matmul_precision()}, cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
        )
        logging.info("=" * 70)

        # Profiling: when True, a TimingSummary (extending NamedTimer with
        # sync_cuda=True) is attached to each decode state, recording
        # per-stage wall-clock times.  Disabled by default to avoid
        # unnecessary GPU stalls in production.
        self._profile_timing = bool(model_cfg.get("profile_timing", False))

        # Cached TTS helpers populated during initialization/warmup
        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None

        self.model = None
        self.model_llm_interface = None
        self.tokenizer = None

        # Engine configuration.
        # engine_type is a user-facing config value that selects which combination
        # of backends to use for the LLM and TTS components:
        #   "native"                 -> native_llm  + native_eartts
        #   "vllm_llm"              -> vllm_llm    + native_eartts
        #   "vllm_eartts"           -> native_llm  + vllm_eartts
        #   "vllm_llm_vllm_eartts"  -> vllm_llm    + vllm_eartts
        # The factory (create_model) uses the specific {backend}_{component}
        # names: native_llm, native_eartts, vllm_llm, vllm_eartts.
        self.engine_type = model_cfg.get("engine_type", "native")
        self.use_vllm_llm = "vllm_llm" in self.engine_type.lower()
        self.use_vllm_eartts = "vllm_eartts" in self.engine_type.lower()
        self.vllm_llm_config = model_cfg.get("vllm_llm_config", None)
        self.vllm_tts_config = model_cfg.get("vllm_tts_config", None)
        self.request_id = "streaming_request_0"  # For vLLM streaming

        # Sampling parameters
        self.top_p = float(model_cfg.get("top_p", 1.0))
        self.repetition_penalty = float(model_cfg.get("repetition_penalty", 1.0))
        self.temperature = float(model_cfg.get("temperature", 1.0))

        # Perception cache configuration
        self.use_perception_cache = bool(model_cfg.get("use_perception_cache", False))
        use_perception_cudagraph = bool(model_cfg.get("use_perception_cudagraph", False))
        if use_perception_cudagraph and not self.use_perception_cache:
            raise ValueError(
                "use_perception_cudagraph requires use_perception_cache to be enabled. "
                "Please also set use_perception_cache=True."
            )
        self.perception_cache_mgr: PerceptionCacheManager | None = None
        self._use_perception_cudagraph = use_perception_cudagraph

        self._initialize_model()

        logging.info("NemotronVoicechatInferenceWrapper initialized successfully.")

    def _initialize_model(self):
        """Initialize the NemotronVoiceChat model from an HF checkpoint."""
        logging.info("Initializing model structure...")
        start_model_init = time.time()

        # Tell from_pretrained to skip loading checkpoint weights for
        # submodules that vLLM will replace — avoids wasted I/O and memory.
        skip_prefixes = set()
        if self.use_vllm_llm:
            skip_prefixes.add("stt_model.llm.")
        if self.use_vllm_eartts:
            skip_prefixes.add("tts_model.tts_model.")

        self.model = NemotronVoiceChat.from_pretrained(
            self.model_path,
            local_files_only=True,
            skip_prefixes=skip_prefixes or None,
        )
        logging.info(f"NemotronVoiceChat initialized in {time.time() - start_model_init:.1f}s")

        # Remove skipped submodules (still on meta device / uninitialized)
        if self.use_vllm_llm:
            del self.model.stt_model.llm
            self.model.stt_model.llm = None
        if self.use_vllm_eartts:
            del self.model.tts_model.tts_model

        # Setup model on device and cast to configured dtype
        self.model.to(self.device)
        self.model.safe_cast_to(self.dtype)
        self.model.eval()

        self.tokenizer = self.model.stt_model.tokenizer

        # Allow runtime wrapper overrides into the STT config used by shared inference helpers.
        _BOOST_KEYS = (
            "inference_pad_boost",
            "inference_bos_boost",
            "inference_eos_boost",
            "inference_user_pad_boost",
            "inference_user_bos_boost",
            "inference_user_eos_boost",
        )
        _FORCE_TURN_TAKING_KEYS = (
            "force_turn_taking",
            "force_turn_taking_threshold",
            "force_turn_taking_pad_window",
        )
        for key in _BOOST_KEYS + _FORCE_TURN_TAKING_KEYS:
            val = self.model_cfg.get(key, None)
            if val is not None:
                OmegaConf.update(self.model.stt_model.cfg, key, val, force_add=True)
        boost_values = {k: self.model.stt_model.cfg.get(k, None) for k in _BOOST_KEYS}
        logging.info(f"Inference logit boosts: {boost_values}")
        force_turn_taking_values = {k: self.model.stt_model.cfg.get(k, None) for k in _FORCE_TURN_TAKING_KEYS}
        logging.info(f"Forced turn-taking config: {force_turn_taking_values}")

        # Create LLM backend
        if self.use_vllm_llm:
            logging.info("Creating VLLMLLM backend...")
            if self.vllm_llm_config is None:
                raise ValueError("vllm_llm_config must be provided when engine_type contains 'vllm_llm'")

            # Set logit boosts as env vars BEFORE creating the vLLM engine,
            # so they are inherited by the forked worker process.  The modified
            # nemotron_h.py reads VLLM_ASR_BOOST_<token_id> and
            # VLLM_TEXT_BOOST_<token_id> in __init__.
            stt = self.model.stt_model
            asr_boost_map = {
                "inference_user_pad_boost": stt.text_pad_id,
                "inference_user_bos_boost": stt.text_bos_id,
                "inference_user_eos_boost": stt.text_eos_id,
            }
            for cfg_key, token_id in asr_boost_map.items():
                val = self.model_cfg.get(cfg_key, None)
                if val is not None and float(val) != 0.0:
                    env_key = f"VLLM_ASR_BOOST_{token_id}"
                    os.environ[env_key] = str(float(val))
                    logging.info(f"Set env {env_key}={val} (from {cfg_key})")

            text_boost_map = {
                "inference_pad_boost": stt.text_pad_id,
                "inference_bos_boost": stt.text_bos_id,
                "inference_eos_boost": stt.text_eos_id,
            }
            for cfg_key, token_id in text_boost_map.items():
                val = self.model_cfg.get(cfg_key, None)
                if val is not None and float(val) != 0.0:
                    env_key = f"VLLM_TEXT_BOOST_{token_id}"
                    os.environ[env_key] = str(float(val))
                    logging.info(f"Set env {env_key}={val} (from {cfg_key})")

        stt = self.model.stt_model
        special_ids = get_special_token_ids(stt.tokenizer, stt.text_pad_id, model_cfg=stt.cfg)
        self.model_llm_interface = create_model(
            model=self.model_path if self.use_vllm_llm else self.model,
            engine_type="vllm_llm" if self.use_vllm_llm else "native_llm",
            vllm_config=self.vllm_llm_config if self.use_vllm_llm else None,
            special_token_ids=special_ids,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
        )
        logging.info(f"LLM backend: {type(self.model_llm_interface).__name__}")

        # Create TTS backend
        self.model_eartts_interface = create_model(
            model=self.model_path if self.use_vllm_eartts else self.model.tts_model,
            engine_type="vllm_eartts" if self.use_vllm_eartts else "native_eartts",
            vllm_config=self.vllm_tts_config if self.use_vllm_eartts else None,
            special_token_ids=special_ids,
        )
        logging.info(f"TTS backend: {type(self.model_eartts_interface).__name__}")

        # torch.compile and subword cache (no-ops for vLLM, delegated to TTS backend)
        if bool(self.model_cfg.get("use_tts_torch_compile", False)):
            self.model_eartts_interface.compile()
        self.model_eartts_interface.setup_subword_cache(self.model_cfg)

        # Get TTS info
        if hasattr(self.model, 'tts_model'):
            self.target_fps = self.model.tts_model.target_fps
            self.target_sample_rate = self.model.tts_model.target_sample_rate
            logging.info(f"TTS model initialized: target_fps={self.target_fps}, sample_rate={self.target_sample_rate}")
            if self.decode_audio:
                self._prepare_tts_initial_state()
        else:
            logging.warning("Warning: TTS model not found in the model")

        # Setup perception cache if enabled
        if self.use_perception_cache:
            self.perception_cache_mgr = PerceptionCacheManager(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                use_cudagraph=self._use_perception_cudagraph,
            )
            if not self.perception_cache_mgr.setup():
                self.use_perception_cache = False
                self.perception_cache_mgr = None

    def _prepare_system_prompt_embeddings(
        self,
        system_prompt: str,
    ) -> tuple[torch.Tensor | None, int]:
        if not system_prompt or not system_prompt.strip():
            return None, 0

        prompt_token_ids = self._build_prompt_token_ids(system_prompt)
        prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        prompt_embedded = self.model.stt_model.embed_tokens(prompt_tokens).to(dtype=self.dtype)
        prompt_len = prompt_tokens.shape[1]

        pad_id = self.model.stt_model.text_pad_id
        pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
        pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)
        pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)

        if prompt_len > 1:
            prompt_embedded[:, 1:, :] += pad_emb
            prompt_embedded[:, 1:, :] += pad_asr_emb

        stt = self.model.stt_model
        bos_emb = stt._get_bos_embedding().to(dtype=self.dtype)
        asr_bos_emb = stt._get_asr_bos_embedding().to(dtype=self.dtype)
        prompt_embedded[:, 0, :] += bos_emb.squeeze(0)
        prompt_embedded[:, 0, :] += asr_bos_emb.squeeze(0)

        return prompt_embedded, prompt_len

    def _clone_cache(self, cache):
        """Deep clone cache structures to ensure complete isolation between streams."""
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            return cache.detach().clone()
        if isinstance(cache, (list, tuple)):
            return type(cache)(self._clone_cache(x) for x in cache)
        if isinstance(cache, dict):
            return {k: self._clone_cache(v) for k, v in cache.items()}
        if hasattr(cache, '__dict__'):
            return copy.deepcopy(cache)
        return cache

    def _build_prompt_token_ids(self, system_prompt: str | None) -> list[int]:
        if not system_prompt or not system_prompt.strip():
            return []
        return [self.tokenizer.bos_id] + self.tokenizer.text_to_ids(system_prompt) + [self.tokenizer.eos_id]

    def _init_token_buffers(self, max_len: int):
        stt_model = self.model.stt_model
        gen_text = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_asr_text = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        return gen_text, gen_asr_text

    def _prepare_tts_initial_state(self):
        if not self.decode_audio:
            return
        if not hasattr(self.model, 'tts_model'):
            return

        logging.info("Preparing TTS warmup state...")

        if self.speaker_name is not None:
            logging.info(f"Using registered speaker name: {self.speaker_name}")
            speaker_audio = None
            speaker_audio_lens = None
        else:
            with fp32_precision():
                speaker_audio, speaker_sr = torchaudio.load(self.speaker_reference)
                speaker_audio = resample(speaker_audio, speaker_sr, self.model.tts_model.target_sample_rate)
            speaker_audio = speaker_audio.to(self.device)
            speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long()

        self.model.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
            system_prompt=self.tts_system_prompt,
            speaker_name=self.speaker_name,
        )
        init_inputs = self.model.tts_model.get_init_inputs(B=1)

        self.generation_config = self.model.tts_model._get_generation_config(guidance_enabled=True)
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})

        with torch.no_grad():
            if self.use_vllm_eartts:
                self.tts_prompt_token_ids = init_inputs["subword_ids"].squeeze().cpu().numpy().tolist()
                self.tts_init_inputs = init_inputs
            outputs = self.model_eartts_interface.prefill_prompt(
                init_inputs,
                prompt_token_ids=getattr(self, 'tts_prompt_token_ids', None),
                request_id="tts_warmup",
            )
            self.model_eartts_interface.abort_request("tts_warmup")

            code = init_inputs["code"][:, -1:]

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)

        logging.info("TTS warmup state prepared")

    def create_decode_state(self, max_len: int) -> StreamingDecodeState:
        gen_text, gen_asr_text = self._init_token_buffers(max_len)
        llm_cache = self.model_llm_interface.create_cache()
        if self.decode_audio:
            subword_mask, tts_codec_cache = self.model_eartts_interface.create_codec_state(max_len, self.device)
        else:
            subword_mask, tts_codec_cache = None, None
        perception_cache = None
        if self.use_perception_cache and self.perception_cache_mgr is not None:
            perception_cache = self.perception_cache_mgr.get_initial_state(batch_size=1)

        tts_past_key_values = None
        tts_code = None
        if self.decode_audio and self.first_tts_code_input is not None:
            tts_past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
            tts_code = self.first_tts_code_input.detach().clone()

        return StreamingDecodeState(
            frame_idx=0,
            gen_text=gen_text,
            gen_asr_text=gen_asr_text,
            input_embeds_history=[],
            llm_cache=llm_cache,
            tts_past_key_values=tts_past_key_values,
            tts_code=tts_code,
            subword_mask=subword_mask,
            perception_cache=perception_cache,
            tts_codec_cache=tts_codec_cache,
            llm_cache_position_offset=0,
            timing=TimingSummary() if self._profile_timing else NullTimingSummary(),
        )

    def infer_one_step(
        self,
        audio_input: torch.Tensor,
        num_frames_per_chunk: int,
        state: StreamingDecodeState,
        *,
        request_id: str | None = None,
        has_prompt: bool = False,
        return_debug: bool = False,
        sampling_params: dict[str, float] | None = None,
    ) -> InferenceStepResult:
        """Run one streaming inference step: perception -> LLM -> TTS -> audio decode.

        All mutable decode state (caches, gen_text, gen_asr_text, code, etc.) is
        updated **in-place** on *state*.  The returned :class:`InferenceStepResult`
        carries only per-step outputs needed by the pipeline.

        Args:
            audio_input (torch.Tensor): Raw audio tensor for this chunk, shape ``(1, samples)``.
            num_frames_per_chunk (int): Number of 80 ms frames in this chunk.
            state (StreamingDecodeState): Mutable decode state (KV caches, token workspaces, etc.).
            request_id (str | None): Unique ID for this stream (used by vLLM engines).
            has_prompt (bool): Whether the LLM state already contains a prefilled
                system prompt. Affects the first-frame embedding (PAD vs BOS).
            return_debug (bool): If True, attach per-step debug info to the result.
            sampling_params (dict[str, float] | None): Optional per-stream sampling overrides
                (``top_p``, ``temperature``, ``repetition_penalty``).
                Keys that are absent fall back to the pipeline-level defaults.
        """
        effective_request_id = request_id or self.request_id
        frame_idx = state.frame_idx

        state.timing.start("total_step")
        has_llm_cache = state.llm_cache is not None
        B = state.gen_text.shape[0]

        predicted_tokens = torch.empty(
            (B, num_frames_per_chunk), dtype=state.gen_text.dtype, device=state.gen_text.device
        )
        asr_predicted_tokens = torch.empty(
            (B, num_frames_per_chunk), dtype=state.gen_text.dtype, device=state.gen_text.device
        )

        debug_logger = IntermediateResultLogger() if return_debug else NullIntermediateResultLogger()

        # --- Stage 1: Perception ---
        state.timing.start("perception")
        source_encoded, state.perception_cache = self._run_perception(
            audio_input,
            frame_idx,
            num_frames_per_chunk,
            state.perception_cache,
        )
        state.timing.stop("perception")
        total_encoded_frames = source_encoded.shape[1]
        if (
            self.use_perception_cache
            and state.perception_cache is not None
            and state.perception_cache.is_initialized()
        ):
            # With cache: we get exactly num_frames_per_chunk output frames
            base_frame_index = 0
        else:
            # Without cache: use the second-to-last encoded frame as the
            # "newest" because the model expects 10ms / 80ms / 80ms ... framing
            # but we always feed 80ms chunks, so the final frame contains
            # silence padding.
            newest = total_encoded_frames - 2
            base_frame_index = max(newest - (num_frames_per_chunk - 1), 0)

        # --- Stage 2: Per-frame generation loop ---
        new_input_embeds = []
        new_codes_for_decode = []
        for frame_offset in range(num_frames_per_chunk):
            current_frame_idx = frame_idx + frame_offset
            current_frame_index = min(base_frame_index + frame_offset, source_encoded.shape[1] - 1)
            debug_logger.log_selected_frame_index(current_frame_index)
            frame_embedding = source_encoded[:, current_frame_index : current_frame_index + 1, :]

            input_emb = self.model.stt_model.build_input_embedding(
                frame_embedding,
                current_frame_idx,
                state.gen_text,
                state.gen_asr_text,
                has_prompt,
            )
            debug_logger.log_input_embeds(input_emb)

            ans = self._run_llm_step(
                input_emb,
                state,
                frame_offset,
                effective_request_id,
                current_frame_idx,
                has_llm_cache,
                return_debug,
                new_input_embeds,
                sampling_params=sampling_params,
            )

            if "text_logits" in ans:
                debug_logger.log_text_logits(ans["text_logits"][:, -1])
            asr_logits = ans.get("asr_logits")
            if asr_logits is not None:
                debug_logger.log_asr_logits(asr_logits[:, -1])

            state.gen_text[:, current_frame_idx] = ans["predicted_token"]
            predicted_tokens[:, frame_offset] = ans["predicted_token"]
            state.gen_asr_text[:, current_frame_idx] = ans["asr_predicted_token"]
            asr_predicted_tokens[:, frame_offset] = ans["asr_predicted_token"]

            self.model.stt_model.streaming_inference._maybe_apply_forced_turn_taking(
                current_frame_idx, state.gen_text, state.gen_asr_text
            )
            predicted_tokens[:, frame_offset] = state.gen_text[:, current_frame_idx]

            if self.decode_audio:
                new_code = self._run_tts_step(
                    state,
                    current_frame_idx,
                    effective_request_id,
                )
                new_codes_for_decode.append(new_code)

        # --- Stage 3: Audio decode ---
        # No-op when self.decode_audio is False: _decode_audio returns None immediately.
        decoded_audio_new = self._decode_audio(new_codes_for_decode, state, frame_idx, num_frames_per_chunk)

        # --- Stage 4: Token -> string conversion ---
        predicted_text_strs = self._tokens_to_strings(predicted_tokens)
        asr_predicted_text_strs = self._tokens_to_strings(asr_predicted_tokens)

        logging.debug(f'frame {frame_idx}: USER asr: {asr_predicted_text_strs}')
        logging.debug(f'frame {frame_idx}: AGENT txt: {predicted_text_strs}')

        # --- Update remaining state fields ---
        if not has_llm_cache:
            # WARNING: Without the KV cache, input_embeds_history grows
            # linearly with session length and the full-sequence LLM forward
            # pass is O(n^2).  This path is currently used for native
            # Nemotron until upstream cache handling is reliable.
            state.input_embeds_history = state.input_embeds_history + new_input_embeds
        if has_llm_cache:
            state.llm_cache_position_offset += num_frames_per_chunk

        state.timing.stop("total_step")

        debug = debug_logger.build_debug_dict(source_encoded, state.gen_text, state.gen_asr_text)

        return InferenceStepResult(
            predicted_text_tokens=predicted_tokens,
            asr_predicted_text_tokens=asr_predicted_tokens,
            predicted_text_strs=predicted_text_strs,
            asr_predicted_text_strs=asr_predicted_text_strs,
            decoded_audio=decoded_audio_new,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # infer_one_step sub-stages
    # ------------------------------------------------------------------

    def _run_llm_step(
        self,
        input_emb: torch.Tensor,
        state: StreamingDecodeState,
        frame_offset: int,
        request_id: str,
        current_frame_idx: int,
        has_llm_cache: bool,
        return_debug: bool,
        new_input_embeds: list,
        sampling_params: dict[str, float] | None = None,
    ) -> dict:
        """Run one LLM forward pass (native cache, vLLM, or full-history).

        Updates ``state.llm_cache`` in-place for cached paths.  For the
        no-cache fallback, appends to *new_input_embeds* (list, mutated).
        """
        state.timing.start("stt_model")

        if has_llm_cache or self.use_vllm_llm:
            if self.use_vllm_llm:
                ans = self.model_llm_interface(
                    input_emb,
                    request_id=request_id,
                    generated_tokens=state.gen_text,
                    current_step=current_frame_idx,
                    sampling_params=sampling_params,
                )
            else:
                ans = self.model_llm_interface(
                    input_emb,
                    cache=state.llm_cache,
                    cache_position_offset=state.llm_cache_position_offset + frame_offset,
                    generated_tokens=state.gen_text,
                    current_step=current_frame_idx,
                    return_logits=return_debug,
                    sampling_params=sampling_params,
                )
            state.llm_cache = ans["cache"]
        else:
            new_input_embeds.append(input_emb)
            full_input_embeds = torch.cat(state.input_embeds_history + new_input_embeds, dim=1)
            ans = self.model_llm_interface(
                full_input_embeds,
                cache=None,
                generated_tokens=state.gen_text,
                current_step=current_frame_idx,
                return_logits=return_debug,
                sampling_params=sampling_params,
            )

        state.timing.stop("stt_model")

        return ans

    def _run_tts_step(
        self,
        state: StreamingDecodeState,
        current_frame_idx: int,
        request_id: str,
    ) -> torch.Tensor:
        """Run one TTS code-generation step.

        Mutates ``state.tts_code`` and ``state.tts_past_key_values`` in-place.
        Returns the new code tensor (cloned) for later batch decoding.
        """
        current_subword_id = state.gen_text[:, current_frame_idx].unsqueeze(-1)

        if current_frame_idx == 0:
            if self.first_context_subword_id is None:
                raise RuntimeError("first_context_subword_id is not initialized. Ensure TTS warmup ran successfully.")
            prev_subword_id = self.first_context_subword_id
        else:
            prev_subword_id = state.gen_text[:, current_frame_idx - 1].unsqueeze(-1)

        current_subword_mask = state.subword_mask[:, current_frame_idx].unsqueeze(-1)

        if self.generation_config is None:
            raise RuntimeError("generation_config is not initialized. Ensure TTS warmup ran successfully.")

        state.timing.start("tts_model")
        if self.use_vllm_eartts:
            tts_inputs = {
                "code": state.tts_code,
                "context_hidden_state": None,
                "subword_ids": current_subword_id,
                "subword_mask": current_subword_mask,
                "past_key_values": state.tts_past_key_values,
                "use_cache": True,
                "guidance_enabled": True,
                "generation_config": self.generation_config,
                "ignore_eos_flag_stop": True,
            }
        else:
            tts_inputs = {
                "current_subword_id": current_subword_id,
                "prev_subword_id": prev_subword_id,
                "current_subword_mask": current_subword_mask,
                "prev_audio_tokens": state.tts_code,
                "past_key_values": state.tts_past_key_values,
                "guidance_enabled": True,
                "generation_config": self.generation_config,
                "ignore_eos_flag_stop": True,
            }
        result = self.model_eartts_interface(tts_inputs, request_id=request_id)
        state.tts_code = result.codes
        state.tts_past_key_values = result.past_key_values

        state.timing.stop("tts_model")

        new_code = state.tts_code.clone()

        if self.model.cfg.get('inference_force_speech_silence_on_eos', None):
            silence_codes = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(state.tts_code.shape)
            state.tts_code = torch.where(
                current_subword_id.unsqueeze(-1) == self.model.tts_model.text_eos_id,
                silence_codes,
                state.tts_code,
            )

        return new_code

    def _decode_audio(
        self,
        new_codes_for_decode: list[torch.Tensor],
        state: StreamingDecodeState,
        frame_idx: int,
        num_frames_per_chunk: int,
    ) -> torch.Tensor | None:
        """Decode accumulated TTS codes into a waveform.

        Returns the decoded audio tensor or *None* when ``decode_audio``
        is disabled or no codes were produced.
        """
        if not self.decode_audio or not new_codes_for_decode:
            return None

        logging.debug(f"Decoding audio for {frame_idx}-th frame  ({num_frames_per_chunk=})")

        state.timing.start("audio_codec")
        with fp32_precision(), torch.no_grad():
            new_codes_tensor = torch.cat(new_codes_for_decode, dim=1)
            if hasattr(self.model.tts_model, '_control_codes'):
                from nemo.collections.speechlm2.models.duplex_ear_tts import replace_control_speech_codes

                new_codes_tensor = replace_control_speech_codes(
                    new_codes_tensor,
                    self.model.tts_model._control_codes,
                    getattr(self.model.tts_model, 'codec_silence_tokens', None),
                )
            new_code_len = torch.tensor(
                [new_codes_tensor.shape[1]],
                dtype=torch.long,
                device=self.device,
            )
            decoded_audio, _ = self.model.tts_model.audio_codec.decode(
                new_codes_tensor,
                new_code_len,
                cache=state.tts_codec_cache,
            )

        state.timing.stop("audio_codec")

        return decoded_audio

    def _run_perception(
        self,
        audio_input: torch.Tensor,
        frame_idx: int,
        num_frames_per_chunk: int,
        perception_cache: PerceptionCacheState | None,
    ) -> tuple[torch.Tensor, PerceptionCacheState | None]:
        """Run the perception encoder and return (source_encoded, updated_cache)."""
        if self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized():
            source_encoded, perception_cache = self.perception_cache_mgr.step(
                audio_input=audio_input,
                frame_idx=frame_idx,
                num_frames_per_chunk=num_frames_per_chunk,
                perception_cache=perception_cache,
            )
        else:
            buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)
            source_encoded, _, _ = self.model.stt_model.perception(
                input_signal=audio_input,
                input_signal_length=buffer_len,
                return_encoder_emb=True,
            )

        source_encoded = source_encoded.to(self.dtype)
        return source_encoded, perception_cache

    def _tokens_to_strings(self, token_ids: torch.Tensor) -> list[str]:
        """Convert a [B, T] tensor of token IDs to a list of strings.

        Uses ``_decode_tokens_with_specials`` so byte-level BPE is decoded
        properly (e.g. ``âĢĻ`` -> ``'``) via HF ``convert_tokens_to_string``.

        Leading spaces are preserved in the output: in byte-level BPE,
        word-initial tokens carry a space prefix that ``convert_tokens_to_string``
        keeps intact.  So callers can concatenate successive chunk strings to
        recover properly spaced text.  A leading space means "new word"; no
        leading space means the token continues the previous word.  For
        example, three chunks producing ``"Hi"``, ``" how can"``,
        ``" I help"`` concatenate to ``"Hi how can I help"`` (not
        ``"Hihow canI help"``).

        NOTE: multi-byte UTF-8 characters whose BPE tokens span two frames
        will show as replacement chars (U+FFFD) because each frame is decoded
        independently.
        """
        pad_token_str = self.tokenizer.ids_to_tokens([self.model.stt_model.text_pad_id])[0]
        result = []
        for tok_ids_b in token_ids:
            toks = self.tokenizer.ids_to_tokens(tok_ids_b.tolist())
            result.append(
                _decode_tokens_with_specials(
                    toks,
                    self.tokenizer,
                    pad_token_str=pad_token_str,
                    keep_pad=False,
                )
            )
        return result

    @property
    def special_token_strings(self) -> set[str]:
        """Token strings that should be stripped from decoded text for clean output."""
        stt = self.model.stt_model
        return get_special_token_strings(stt.tokenizer, stt.text_pad_id, model_cfg=stt.cfg)
