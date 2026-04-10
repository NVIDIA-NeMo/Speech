"""
TRT-LLM Backbone Runner for EasyMagpieTTS.

Wraps the custom backbone TRT engine (inputs_embeds → last_hidden_state)
with a HuggingFace-compatible interface:

    runner = TRTBackboneRunner(engine_dir, device)
    out = runner(inputs_embeds=ctx_embeds, use_cache=True)
    last_hidden = out.last_hidden_state   # [B, T, E]
    state = out.past_key_values           # runner-internal opaque handle

    out2 = runner(inputs_embeds=step_embed, use_cache=True, past_key_values=state)
    last_hidden2 = out2.last_hidden_state  # [B, 1, E]

KV cache is managed inside the runner as pre-allocated continuous buffers.
`past_key_values` returned to the caller is a lightweight SequenceState
object that records the current cache length — not the full KV tensors.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import ctypes

import torch
import tensorrt as trt


def _register_trt_llm_plugins():
    """Register gpt_attention_plugin via ctypes without importing full tensorrt_llm.

    Works in environments that have the TRT-LLM .so but not the Python package.
    Set TRTLLM_PLUGIN_LIB env var to override the default library path.
    """
    lib_path = os.environ.get(
        "TRTLLM_PLUGIN_LIB",
        "/home/subhankarg/miniconda3/envs/em_trtllm/lib/python3.10/site-packages/"
        "tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so",
    )
    handle = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initTrtLlmPlugins.restype = ctypes.c_bool
    ok = handle.initTrtLlmPlugins(None, b"tensorrt_llm")
    assert ok, f"initTrtLlmPlugins returned False for lib: {lib_path}"


_register_trt_llm_plugins()


# ─── output container matching HuggingFace BaseModelOutputWithPast ────────────

@dataclass
class BackboneOutput:
    last_hidden_state: torch.Tensor
    past_key_values: "SequenceState"


@dataclass
class SequenceState:
    """Opaque state returned to the caller; tracks KV cache position only."""
    cache_len: int   # number of tokens in the KV cache after this step


# ─── main runner ──────────────────────────────────────────────────────────────

class TRTBackboneRunner:
    """
    Execute the backbone TRT engine with inputs_embeds injection.

    Engine path: em_trtllm_engine_backbone/rank0.engine
    - Inputs:  inputs_embeds [B, T, E], position_ids [B, T], attention metadata,
               past_key_value_{i} [B, 2, num_kv_heads, max_seq_len, head_dim]
    - Outputs: last_hidden_state [B, T, E],
               present_key_value_{i} [B, 2, num_kv_heads, max_seq_len, head_dim]
    """

    N_LAYERS = 28
    NUM_KV_HEADS = 2
    HEAD_DIM = 128
    HIDDEN_SIZE = 1536

    def __init__(self, engine_dir: str, device: torch.device, max_seq_len: int = 512):
        self.device = device
        self.max_seq_len = max_seq_len

        # Ensure TRT engine is created on the correct CUDA device
        with torch.cuda.device(device):
            # Load engine
            engine_path = os.path.join(engine_dir, "rank0.engine")
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            with open(engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            assert self._engine is not None, f"Failed to load engine from {engine_path}"

            self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=device)

        # Read build config for max_seq_len
        cfg_path = os.path.join(engine_dir, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        bc = cfg.get("build_config", {})
        self.max_seq_len = bc.get("max_seq_len", max_seq_len)
        self.max_batch_size = bc.get("max_batch_size", 2)

        # Pre-allocate KV cache buffers: [B, 2, num_kv_heads, max_seq_len, head_dim]
        # B=1 for streaming TTS
        B = 1
        kv_shape = (B, 2, self.NUM_KV_HEADS, self.max_seq_len, self.HEAD_DIM)
        self._kv_cache = [
            torch.zeros(kv_shape, dtype=torch.float16, device=device)
            for _ in range(self.N_LAYERS)
        ]

        # Pre-allocate attention plugin host tensors (on CPU)
        self._host_max_attn_window = torch.full(
            (self.N_LAYERS,), self.max_seq_len, dtype=torch.int32)
        self._host_sink_token_length = torch.zeros(1, dtype=torch.int32)
        self._host_runtime_perf_knobs = torch.zeros(16, dtype=torch.int64)
        self._host_context_progress = torch.zeros(1, dtype=torch.int64)

        # Dummy cache_indirection (no beam search)
        self._cache_indirection = torch.zeros(
            B, 1, self.max_seq_len, dtype=torch.int32, device=device)

    def reset_cache(self):
        """Clear KV cache (call before processing a new utterance)."""
        for buf in self._kv_cache:
            buf.zero_()

    def __call__(
        self,
        inputs_embeds: torch.Tensor,       # [B, T, hidden_size]
        use_cache: bool = True,
        past_key_values: Optional[SequenceState] = None,
        attention_mask=None,               # ignored (causal attention in engine)
        **kwargs,
    ) -> BackboneOutput:
        """Run one forward step through the backbone engine."""
        B, T, E = inputs_embeds.shape
        assert E == self.HIDDEN_SIZE, f"Expected hidden_size={self.HIDDEN_SIZE}, got {E}"
        assert B == 1, "TRTBackboneRunner currently supports B=1 only"

        prev_len = past_key_values.cache_len if past_key_values is not None else 0
        is_context = (prev_len == 0)

        # ── position_ids ─────────────────────────────────────────────────────
        pos_ids = torch.arange(prev_len, prev_len + T,
                               dtype=torch.int32, device=self.device).unsqueeze(0)  # [1, T]

        # ── attention plugin metadata (CPU) ───────────────────────────────────
        # GenerationSession semantics (from generation.py source):
        #   context:
        #     sequence_length = host_past_kv_len = context_lengths = T
        #   generation step k (prev_len = T + k):
        #     sequence_length    = prev_len  (= sequence_length_buffer before increment)
        #     host_past_kv_len   = prev_len  (= max_context_length + step)
        #     context_lengths    = prev_len  (carried from context phase, unchanged)
        #     host_request_types = 1
        total_len = prev_len + T
        kv_ref_len = total_len if is_context else prev_len
        seq_len = torch.tensor([kv_ref_len], dtype=torch.int32)
        host_past_kv_len = torch.tensor([kv_ref_len], dtype=torch.int32)
        # context_lengths = total sequence length (prev_len + T) for BOTH phases.
        # Scan confirmed: this is the only value that affects correctness.
        context_lengths = torch.tensor([total_len], dtype=torch.int32)
        host_request_types = torch.tensor([0 if is_context else 1], dtype=torch.int32)

        # ── dummy input_ids (engine requires it but ignores it) ───────────────
        dummy_input_ids = torch.zeros(B, T, dtype=torch.int32, device=self.device)
        dummy_last_token_ids = torch.tensor([T], dtype=torch.int32, device=self.device)

        # ── output buffer ─────────────────────────────────────────────────────
        out_hidden = torch.empty(B, T, E, dtype=torch.float16, device=self.device)

        # ── bind all engine tensors ───────────────────────────────────────────
        ctx = self._context

        def bind(name, tensor, shape=None):
            if shape is None:
                shape = tuple(tensor.shape)
            ctx.set_input_shape(name, shape) if self._engine.get_tensor_mode(
                name) == trt.TensorIOMode.INPUT else None
            ctx.set_tensor_address(name, tensor.data_ptr())

        # Primary inputs
        ctx.set_input_shape("inputs_embeds", (B, T, E))
        ctx.set_tensor_address("inputs_embeds", inputs_embeds.contiguous().data_ptr())

        ctx.set_input_shape("input_ids", (B, T))
        ctx.set_tensor_address("input_ids", dummy_input_ids.data_ptr())

        ctx.set_input_shape("position_ids", (B, T))
        ctx.set_tensor_address("position_ids", pos_ids.data_ptr())

        ctx.set_input_shape("last_token_ids", (B,))
        ctx.set_tensor_address("last_token_ids", dummy_last_token_ids.data_ptr())

        # Attention plugin metadata (host tensors — TRT copies to device internally)
        # Keep explicit references to device tensors so they are not GC'd before
        # execute_async_v3 reads them from the GPU.
        seq_len_dev = seq_len.to(self.device)
        context_lengths_dev = context_lengths.to(self.device)

        ctx.set_input_shape("sequence_length", (B,))
        ctx.set_tensor_address("sequence_length", seq_len_dev.data_ptr())

        ctx.set_input_shape("host_past_key_value_lengths", (B,))
        ctx.set_tensor_address("host_past_key_value_lengths",
                               host_past_kv_len.data_ptr())

        ctx.set_input_shape("context_lengths", (B,))
        ctx.set_tensor_address("context_lengths", context_lengths_dev.data_ptr())

        ctx.set_input_shape("host_request_types", (B,))
        ctx.set_tensor_address("host_request_types",
                               host_request_types.data_ptr())

        ctx.set_input_shape("host_max_attention_window_sizes", (self.N_LAYERS,))
        ctx.set_tensor_address("host_max_attention_window_sizes",
                               self._host_max_attn_window.data_ptr())

        ctx.set_input_shape("host_sink_token_length", (1,))
        ctx.set_tensor_address("host_sink_token_length",
                               self._host_sink_token_length.data_ptr())

        ctx.set_input_shape("host_runtime_perf_knobs", (16,))
        ctx.set_tensor_address("host_runtime_perf_knobs",
                               self._host_runtime_perf_knobs.data_ptr())

        ctx.set_input_shape("host_context_progress", (1,))
        ctx.set_tensor_address("host_context_progress",
                               self._host_context_progress.data_ptr())

        ctx.set_input_shape("cache_indirection", (B, 1, self.max_seq_len))
        ctx.set_tensor_address("cache_indirection",
                               self._cache_indirection.data_ptr())

        # KV cache: past inputs and present outputs share the same buffer
        kv_shape = (B, 2, self.NUM_KV_HEADS, self.max_seq_len, self.HEAD_DIM)
        for i in range(self.N_LAYERS):
            buf = self._kv_cache[i]
            ctx.set_input_shape(f"past_key_value_{i}", kv_shape)
            ctx.set_tensor_address(f"past_key_value_{i}", buf.data_ptr())
            # present_key_value_{i} is written in-place to the same buffer
            ctx.set_tensor_address(f"present_key_value_{i}", buf.data_ptr())

        # Output
        ctx.set_tensor_address("last_hidden_state", out_hidden.data_ptr())

        # ── execute ───────────────────────────────────────────────────────────
        with torch.cuda.device(self.device):
            ok = ctx.execute_async_v3(self._stream.cuda_stream)
        assert ok, "TRT engine execution failed"
        self._stream.synchronize()

        new_state = SequenceState(cache_len=prev_len + T)
        return BackboneOutput(last_hidden_state=out_hidden, past_key_values=new_state)


# ─── HuggingFace-compatible decoder wrapper ───────────────────────────────────

class TRTBackboneDecoder:
    """
    Drop-in replacement for `EasyMagpieTTSDecoder.decoder` (normally a HF Qwen2Model).

    Satisfies the interface expected by easy_magpietts_inference.py:
      - decoder.set_input_embeddings(emb)
      - decoder.get_input_embeddings()         → nn.Embedding
      - decoder(inputs_embeds, attention_mask,
                use_cache, past_key_values, **kwargs)
                                               → BackboneOutput

    KV cache is reset automatically on the first call of each utterance
    (i.e., when past_key_values is None).
    """

    def __init__(self, engine_dir: str, device: torch.device):
        self._runner = TRTBackboneRunner(engine_dir, device)
        self._embeddings = None   # set via set_input_embeddings()

    # ── embedding helpers (mirrors nn.Module interface) ───────────────────────

    def set_input_embeddings(self, embeddings):
        self._embeddings = embeddings

    def get_input_embeddings(self):
        return self._embeddings

    # ── forward pass ─────────────────────────────────────────────────────────

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask=None,          # ignored; engine uses causal attention
        use_cache: bool = True,
        past_key_values=None,         # None → start of utterance; SequenceState → decode step
        **kwargs,
    ) -> BackboneOutput:
        # Reset KV cache at the start of each utterance
        if past_key_values is None:
            self._runner.reset_cache()

        input_dtype = inputs_embeds.dtype
        emb = inputs_embeds.half().contiguous()
        out = self._runner(
            inputs_embeds=emb,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        # Cast output back to the input dtype (TRT engine always outputs fp16)
        out.last_hidden_state = out.last_hidden_state.to(input_dtype)
        return out
