"""FastAPI sidecar that drives the vLLM-backed SmallMamba TTS backbone.

Lives in ``vllm_omni_env``. The voice agent (running in
``nemo_virtual_environment``, which cannot import vLLM) is the client and
talks to this server over HTTP.

Run with (after `pip install -e .` in vllm_omni_env):
    source /mnt/n1_mount/personal/vllm_omni/vllm_omni_env/bin/activate
    export TMPDIR=/mnt/n1_mount/personal/tmp_nemo
    CUDA_VISIBLE_DEVICES=1 python -m easymagpie_server.server --port 18765
    # or: easymagpie-sidecar --port 18765  (entry-point script)

All required env vars (VLLM_ALLOW_INSECURE_SERIALIZATION, EASYMAGPIE_LT_*,
EM_LT_*) are set internally via os.environ.setdefault — no shell prelude
needed beyond TMPDIR and CUDA_VISIBLE_DEVICES.

Streaming protocol: POST /tts/stream returns ``application/x-ndjson`` -- one
JSON object per line. Each line is either:
    {"frame": <int>, "codes": [16 ints]}   -- one audio frame
    {"done": true, "total_frames": <int>}  -- terminator

Request body (JSON):
    {
        "context_audio_codes": [[c, ...], ...],   # shape (16, T_ctx); int IDs
        "context_text_token_ids": [...],          # shape (L_text,); int IDs
        "max_frames": 200,
        "temperature": 0.7,
        "topk": 80
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time as _time_mod
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


logger = logging.getLogger("easymagpie_server")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ---------------------------------------------------------------------------
# vLLM lifecycle: build LLM once at startup. The whole TTS streaming chain
# (LT + audio-code sampling + next-frame embedding) runs INSIDE the engine
# subprocess via our ``embed_input_ids`` override; the server just submits
# requests and pulls captured codes via ``apply_model``.
# ---------------------------------------------------------------------------
class _LLMHolder:
    llm = None
    lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls.llm is None:
            with cls.lock:
                if cls.llm is None:
                    cls.llm = cls._build()
        return cls.llm

    @classmethod
    def _setup_trt_lt_env(cls):
        """Configure the LT (local-transformer) backend.

        Default: torch.compile ``max-autotune`` with FP16 LT input dtype. This
        combo gives UTMOS ~4.40 (near source-reference quality) at RTF ~0.96.
        See refactor_baselines/expA_compile_fp16_megan.json for the A/B that
        produced this default.

        Alternatives via env (kept for diagnostics):
          * ``EASYMAGPIE_LT_BACKEND=trt_fused`` — uses the categorical-sampling
            TRT engine. FP16 internal (default) or set
            ``EASYMAGPIE_LT_FUSED_PRECISION=fp32`` to force FP32. Both have
            been observed to produce warbly audio (UTMOS ~3.0) compared to
            torch.compile, so they're NOT recommended.
          * ``EM_LT_COMPILE_MODE=reduce-overhead`` — faster compile,
            slightly lower runtime quality.
          * ``EM_LT_DTYPE=fp32`` — compile path in FP32 (UTMOS ~4.4 same as
            FP16 but RTF ~1.3).
        """
        # EASYMAGPIE_CATEGORICAL_PLUGIN_PATH is set by main() from the
        # --categorical-plugin-path CLI flag.
        os.environ.setdefault("EASYMAGPIE_LT_BACKEND", "compile")
        os.environ.setdefault("EM_LT_COMPILE_MODE", "max-autotune")
        os.environ.setdefault("EM_LT_DTYPE", "fp16")
        # vLLM 0.19.1 routes apply_model(func) over an msgpack-encoded IPC
        # bus to the EngineCore subprocess; msgpack can't serialize a Python
        # function, so we need the pickle fallback. We use apply_model once
        # at startup to fetch the virtual-EOS token id from the loaded model
        # (see _build below). The trust boundary is fine for a local sidecar.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        logger.info(
            "LT backend = %s  compile_mode=%s  dtype=%s",
            os.environ["EASYMAGPIE_LT_BACKEND"],
            os.environ["EM_LT_COMPILE_MODE"],
            os.environ["EM_LT_DTYPE"],
        )

    @classmethod
    def _build(cls):
        # Preload CUDA-12 cudart + opt-in the TRT-fused LT backend BEFORE
        # importing vllm so the EngineCore subprocess inherits the env.
        # Matches the production agent's recipe in ``EasyMagpieTTSService._setup_model``.
        cls._setup_trt_lt_env()

        from vllm import LLM
        # Set by main() from the --model-dir CLI flag.
        model_dir = os.environ["EASYMAGPIE_MODEL_DIR"]
        logger.info("Building LLM with model_dir=%s ...", model_dir)
        # NOTE: gpu_memory_utilization default is 0.4 (vs vLLM's 0.6).
        # max_num_seqs=1 with at most ~500-token sequence really only needs
        # <1 GB KV cache; vLLM's default 0.6 grabs ~28 GB on an A6000 and
        # OOMs the agent process sharing the same GPU.
        #
        # ``enforce_eager`` default: FALSE (compile path on).
        # vLLM 0.19.1's compile path tripped on
        # ``_update_hybrid_attention_mamba_layout`` asserting that the
        # attention KV cache's ``num_blocks != 2`` (our hybrid Mamba+Attn
        # checkpoint lands at exactly num_blocks=2). We patched that
        # assertion in vLLM site-packages per PR #37679 (FlashAttention 2
        # backend assumes dim 0 is K/V; safe for our setup). Combined with
        # ``cudagraph_mode=NONE`` (below) so the captured graph doesn't
        # bypass our Python TTS hooks. Set EM_ENFORCE_EAGER=1 to revert
        # to the legacy eager path for diagnostics.
        enforce_eager = os.environ.get("EM_ENFORCE_EAGER", "0") == "1"
        # ``dtype`` default bf16 to match production's bf16 autocast --
        # fp16 strict diverges enough numerically that the phoneme stream
        # collapses to EOS within 1-2 steps. Override with EM_DTYPE.
        dtype_str = os.environ.get("EM_DTYPE", "bfloat16")
        # When compile mode is active (enforce_eager=False), disable
        # CUDA-graph capture: vLLM captures the forward graph during
        # warmup when self._request_state is None, then replays at
        # request time — bypassing our Python TTS hooks entirely. With
        # cudagraph_mode=NONE, vLLM still uses torch.compile for kernel
        # fusion but re-enters Python forward() on every step.
        compilation_config = None
        if not enforce_eager:
            from vllm.config import CompilationConfig
            compilation_config = CompilationConfig(
                cudagraph_mode="NONE",
            )
        llm = LLM(
            model=model_dir, tokenizer=model_dir, skip_tokenizer_init=True,
            dtype=dtype_str, max_model_len=int(os.environ.get("MAX_MODEL_LEN", "512")),
            max_num_seqs=1,
            gpu_memory_utilization=float(os.environ.get("EM_GPU_MEM_UTIL", "0.4")),
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
            seed=0, trust_remote_code=False, load_format="safetensors",
            enable_prompt_embeds=True,
            # Force sync scheduling. vLLM 0.19.1 V1 async scheduling has a
            # bug in the prompt_embeds H2D copy path (gpu_model_runner.py
            # line 1679-1685): on the 2nd-and-later request, the
            # `inputs_embeds.copy_to_gpu()` is gated behind
            # `num_common_tokens < total_without_spec`, which evaluates
            # False for our single-request-at-a-time TTS workload, so the
            # H2D copy is skipped. GPU buffer keeps stale/zero data,
            # backbone forward sees zero input, audio degrades.
            # Sync path (line 1627-1633) unconditionally copies.
            # For one-request-at-a-time TTS this is a strict win:
            # correctness without throughput cost.
            async_scheduling=False,
        )
        logger.info("LLM built (dtype=%s, enforce_eager=%s)", dtype_str, enforce_eager)
        # NOTE: the NemotronHMoE shared_experts SiLU fix lives inside V2's
        # __init__ — see easymagpie_vllm/backbone_patches.py. It's an
        # architectural quirk of running SmallMamba on vLLM's NemotronH,
        # so it belongs with the model, not the server orchestration.

        # Sampler-driven EOS: V2 returns real lm_head logits + boosts
        # ``VIRTUAL_EOS_TOKEN_ID`` when streaming state's audio_finished
        # flips True. The standard vLLM sampler picks that token, the
        # request hits ``stop_token_ids``, and the engine finishes the
        # request cleanly. The virtual EOS id is a class constant on V2.
        cls._virtual_eos_token_id = int(llm.apply_model(
            lambda m: int(m.VIRTUAL_EOS_TOKEN_ID)
        )[0])
        logger.info(
            "Sampler-driven EOS: virtual_eos_token_id=%d",
            cls._virtual_eos_token_id,
        )
        logger.info("LLM ready.")
        return llm


# ---------------------------------------------------------------------------
# apply_model helpers -- run inside the engine subprocess.
# ---------------------------------------------------------------------------
def _build_prefill_and_reset(
    model, ctx_codes_list, ctx_text_list,
    phoneme_token_ids, text_eos_id,
    audio_bos_id, audio_eos_id,
    phoneme_bos_id, phoneme_eos_id,
    streaming_speech_delay, streaming_phonemes_delay,
    lt_temperature, lt_topk,
):
    """Build the prefill embedding (speaker+context_text) and initialise
    the streaming state machine with the model-specific constants the agent
    extracted from the production checkpoint. Returns the prefill on CPU.
    """
    device = next(model.parameters()).device
    ctx_codes = torch.tensor(ctx_codes_list, dtype=torch.long, device=device)
    if ctx_codes.dim() == 2:
        ctx_codes = ctx_codes.unsqueeze(0)               # (1, n_tables, T_ctx)
    ctx_lens = torch.tensor([ctx_codes.shape[-1]], dtype=torch.long, device=device)
    text_ids = torch.tensor([ctx_text_list], dtype=torch.long, device=device)
    text_lens = torch.tensor([text_ids.shape[-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        prefill, _prefill_lens = model.build_prefill_combined_embeddings(
            ctx_codes, ctx_lens, text_ids, text_lens,
        )

    # Clear any lingering per-request state.
    model._last_hidden_state = None

    # Defensive: force-zero vLLM's per-layer KV caches (Mamba conv_states +
    # ssm_states, Attention K/V blocks) before each request. With the
    # eos-abort + inline LLMEngine.step() fixes this is no longer strictly
    # required for correctness (aborted requests release their cache
    # slots cleanly), but it's a microsecond-cost defense in depth.
    for _mod in model.modules():
        kv = getattr(_mod, "kv_cache", None)
        if kv is None:
            continue
        if isinstance(kv, list):
            for _t in kv:
                if torch.is_tensor(_t):
                    _t.zero_()
        elif torch.is_tensor(kv):
            kv.zero_()

    # Initialise the full streaming state machine to production-equivalent
    # values. After this, every embed_input_ids + compute_logits call drives
    # the AR with proper phase masks (needs_text/needs_phoneme/needs_audio),
    # audio_bos for the first audio step, last_audio_codes thereafter, etc.
    model.init_streaming_request(
        audio_bos_id=audio_bos_id,
        audio_eos_id=audio_eos_id,
        phoneme_bos_id=phoneme_bos_id,
        phoneme_eos_id=phoneme_eos_id,
        streaming_speech_delay=streaming_speech_delay,
        streaming_phonemes_delay=streaming_phonemes_delay,
        phoneme_token_ids=phoneme_token_ids,
        text_eos_id=text_eos_id,
        lt_temperature=lt_temperature,
        lt_topk=lt_topk,
    )

    return prefill.squeeze(0).detach().to("cpu").float().contiguous()


def _pull_codes(model):
    """Return list-of-frame-codes captured so far, then clear. Each frame is
    a list of 16 ints. We return a copy and clear so streaming progress can
    be polled incrementally."""
    codes = getattr(model, "_streaming_codes", []) or []
    if not codes:
        return []
    frames = [c.flatten().tolist() for c in codes]
    model._streaming_codes = []
    return frames


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="EasyMagpie SmallMamba vLLM sidecar", version="0.1.0")


class TTSRequest(BaseModel):
    """Full per-utterance request. Includes the model-specific constants
    production reads from its loaded checkpoint (audio/phoneme BOS+EOS
    IDs, streaming delays) so the sidecar can replicate the
    ``streaming_step`` phase state machine without hard-coding them."""

    context_audio_codes: list[list[int]] = Field(
        ..., description="(n_tables=16, T_ctx) audio code IDs for context.")
    context_text_token_ids: list[int] = Field(
        ..., description="(L,) subword token IDs for context text.")
    phoneme_token_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Per-step CAS-subword token IDs for the utterance, ending "
            "with the text EOS id. Consumed one per AR step until "
            "exhausted; production ``streaming_step(text_tokens=...)``."
        ))
    text_eos_id: int = Field(
        ..., description="The subword-tokenizer EOS id. Used to set "
        "text_finished when consumed.")
    audio_bos_id: int = Field(
        ..., description="Audio codebook BOS id (production's "
        "``self.audio_bos_id``). Used for the FIRST audio step embedding.")
    audio_eos_id: int = Field(
        ..., description="Audio codebook EOS id. Triggers stream finish "
        "when any codebook samples it.")
    phoneme_bos_id: int = Field(
        ..., description="Phoneme tokenizer BOS id (``self.phoneme_tokenizer."
        "bos_token_id``). Used for the FIRST phoneme step embedding.")
    phoneme_eos_id: int = Field(
        ..., description="Phoneme tokenizer EOS id. Triggers "
        "phoneme_stream_ended when predicted.")
    streaming_speech_delay: int = Field(
        ..., description="Number of post-context text steps before audio "
        "emission begins. Comes from the checkpoint's training_mode "
        "(``streaming_3_5`` -> 5).")
    streaming_phonemes_delay: int = Field(
        ..., description="Number of post-context text steps before "
        "auto-predicted phoneme stream begins.")
    max_frames: int = 200
    # LT sampling knobs. The TRT-fused LT engine bakes its temperature +
    # topk into the compiled graph at build time (via
    # LocalTransformerHelper._fused_temperature / _fused_topk, set from
    # EM_LT_TEMPERATURE / EM_LT_TOPK at sidecar startup). When the per-
    # request values match the baked-in ones, the engine's FUSED path
    # is taken; otherwise the helper falls back to its pytorch AR loop.
    # Defaults here match production's PHONEME_SAMPLING_METHOD context
    # (temperature=0.7, topk=80) and the sidecar's TRT engine build.
    temperature: float = 0.7
    topk: int = 80


class WireTokenizerRequest(BaseModel):
    """Send the BPE tokenizer info needed by the CAS encoder. Must be POSTed
    once at agent startup before the first /tts/stream call. Without this the
    CAS encoder's subword->char map is empty and any non-trivial
    context_text_token_ids will trigger a CUDA out-of-range assert."""
    subword_vocab: dict[str, int]
    bos_id: int
    eos_id: int
    cfg_unk_token_id: int
    subword_padding_idx: Optional[int] = None


def _wire_tokenizer(model, subword_vocab, bos_id, eos_id, cfg_unk_token_id,
                    subword_padding_idx):
    model.wire_tokenizer(
        subword_vocab=subword_vocab,
        bos_id=bos_id,
        eos_id=eos_id,
        cfg_unk_token_id=cfg_unk_token_id,
        subword_padding_idx=subword_padding_idx,
    )
    return True


@app.on_event("startup")
def _startup():
    llm = _LLMHolder.get()
    # Pre-warm the compiled LT (torch.compile + CUDA graph capture). The
    # first invocation otherwise takes minutes -- longer than the HTTP
    # client's read timeout -- so we burn that cost here, before any
    # client request lands.
    if os.environ.get("EASYMAGPIE_LT_BACKEND", "compile").lower() == "compile":
        try:
            llm.apply_model(lambda m: m.warmup_compiled_lt(iters=3))
            logger.info("[startup] compiled-LT warmup complete")
        except Exception as e:
            logger.exception("[startup] compiled-LT warmup failed: %s", e)


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    from vllm import SamplingParams

    llm = _LLMHolder.get()

    # Build prefill embed inside the engine subprocess (so the speaker_encoder
    # + cas_encoder weights stay on GPU) and reset streaming state. Also
    # stashes the per-utterance phoneme stream on the model so embed_input_ids
    # can consume one token per AR step.
    try:
        # Hoist into locals so the lambda doesn't reference req across
        # the apply_model pickle boundary.
        ctx_codes = req.context_audio_codes
        ctx_text = req.context_text_token_ids
        phoneme_ids = req.phoneme_token_ids
        text_eos = req.text_eos_id
        audio_bos = req.audio_bos_id
        audio_eos = req.audio_eos_id
        phoneme_bos = req.phoneme_bos_id
        phoneme_eos = req.phoneme_eos_id
        speech_delay = req.streaming_speech_delay
        phonemes_delay = req.streaming_phonemes_delay
        lt_temp = req.temperature
        lt_topk = req.topk
        prefill_list = llm.apply_model(
            lambda m: _build_prefill_and_reset(  # noqa: E501
                m, ctx_codes, ctx_text, phoneme_ids, text_eos,
                audio_bos, audio_eos, phoneme_bos, phoneme_eos,
                speech_delay, phonemes_delay, lt_temp, lt_topk,
            )
        )
    except Exception as e:
        logger.exception("Prefill build failed")
        raise HTTPException(status_code=500, detail=f"prefill build failed: {e}") from e
    prefill = prefill_list[0]  # tp_size=1 => single worker
    if prefill is None:
        raise HTTPException(500, "engine returned no prefill")

    # max_frames is a hard cap; the AR also exits early via the
    # sampler's stop_token_ids (set when any audio codebook samples audio_eos).
    _max_frames = req.max_frames
    if req.phoneme_token_ids:
        # The cap is a safety net; audio_eos sampled by the model is the
        # real exit signal. Be generous (8 frames per subword + 80 head-
        # room) so natural-length English utterances reach EOS before the
        # cap. Old formula (len+50) was too tight: a 19-subword utterance
        # was capped at 69 frames (~2.76s) and got cut mid-word.
        _max_frames = min(_max_frames, len(req.phoneme_token_ids) * 8 + 80)

    def stream():
        # Sampler-driven EOS: compute_logits boosts VIRTUAL_EOS_TOKEN_ID
        # when streaming state's audio_finished flips True. The standard
        # vLLM sampler picks that token, the engine sees stop_token_ids
        # match, request terminates. No abort_request, no per-step EOS
        # polling, fewer apply_model roundtrips per engine.step().
        virtual_eos_id = _LLMHolder._virtual_eos_token_id
        sp = SamplingParams(
            temperature=req.temperature, top_k=req.topk,
            max_tokens=_max_frames, seed=0,
            ignore_eos=False, stop_token_ids=[virtual_eos_id],
        )

        # Drive LLMEngine inline (no background thread). When this loop
        # exits the engine is quiescent, so the next request's apply_model
        # doesn't race against tail steps from this one.
        engine = llm.llm_engine
        req_id = f"em-{int(_time_mod.monotonic()*1e6)}"
        engine.add_request(
            request_id=req_id,
            prompt={"prompt_embeds": prefill},
            params=sp,
        )
        emitted = 0
        try:
            while engine.has_unfinished_requests():
                engine.step()
                # Drain frames captured by predict_codes_and_advance.
                try:
                    frames = llm.apply_model(_pull_codes)[0]
                except Exception:
                    frames = []
                for f in frames or []:
                    yield json.dumps({"frame": emitted, "codes": f}) + "\n"
                    emitted += 1
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
            return
        # Final drain (no race possible -- the engine is idle now).
        try:
            frames = llm.apply_model(_pull_codes)[0]
        except Exception:
            frames = []
        for f in frames or []:
            yield json.dumps({"frame": emitted, "codes": f}) + "\n"
            emitted += 1
        yield json.dumps({"done": True, "total_frames": emitted}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/tts/wire_tokenizer")
def tts_wire_tokenizer(req: WireTokenizerRequest):
    """Wire the BPE tokenizer's subword vocab into the sidecar's CAS encoder.

    The agent's TTS service must POST this once at startup -- before the
    first /tts/stream call -- so the CAS encoder can map subword IDs to
    char IDs. Without this, any non-trivial context_text_token_ids
    triggers a CUDA out-of-range assert from the CAS encoder.
    """
    llm = _LLMHolder.get()
    try:
        llm.apply_model(lambda m: _wire_tokenizer(
            m, req.subword_vocab, req.bos_id, req.eos_id,
            req.cfg_unk_token_id, req.subword_padding_idx,
        ))
    except Exception as e:
        logger.exception("wire_tokenizer failed")
        raise HTTPException(500, f"wire_tokenizer failed: {e}") from e
    return {"status": "ok", "vocab_size": len(req.subword_vocab)}



@app.get("/healthz")
def healthz():
    # Touch the LLM holder to confirm it loaded.
    _LLMHolder.get()
    return {"status": "ok"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--model-dir",
        default="/mnt/n1_mount/personal/vllm_omni/smallmamba_vllm_format",
        help="vLLM-format model checkpoint directory (must contain "
             "config.json + model.safetensors + tts_extras.pt). "
             "Sets $EASYMAGPIE_MODEL_DIR.",
    )
    p.add_argument(
        "--categorical-plugin-path",
        default="/tmp/categorical_plugin/libcategorical_sampling_plugin_v2.so",
        help="Path to libcategorical_sampling_plugin_v2.so. Only used when "
             "EASYMAGPIE_LT_BACKEND=trt_fused; the default compile LT path "
             "ignores this. Sets $EASYMAGPIE_CATEGORICAL_PLUGIN_PATH.",
    )
    args = p.parse_args()

    # CLI -> env var. Downstream code (model load, LT backend) reads the env
    # vars, so this keeps the indirection single-source-of-truth.
    os.environ["EASYMAGPIE_MODEL_DIR"] = args.model_dir
    os.environ["EASYMAGPIE_CATEGORICAL_PLUGIN_PATH"] = args.categorical_plugin_path

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                workers=1)


if __name__ == "__main__":
    main()
