"""FastAPI sidecar that drives the vLLM-backed SmallMamba TTS backbone.

Lives in ``vllm_omni_env``. The voice agent (in ``nemo_virtual_environment``,
which cannot import vLLM) is the client and talks to this server over HTTP.

Run::

    source /mnt/n1_mount/personal/vllm_omni/vllm_omni_env/bin/activate
    export TMPDIR=/mnt/n1_mount/personal/tmp_nemo
    CUDA_VISIBLE_DEVICES=1 easymagpie-sidecar --port 18765

Streaming protocol: POST /tts/stream returns ``application/x-ndjson`` — one
JSON object per line, either ``{"frame": N, "codes": [16 ints]}`` for each
emitted frame or ``{"done": true, "total_frames": N}`` as the terminator.

Env knobs:
  * ``EM_CUDAGRAPH=PIECEWISE`` — enable CUDA-graph capture (default ``NONE``).
  * ``EM_ENFORCE_EAGER=1`` — disable torch.compile entirely (diagnostics).
  * ``EM_DTYPE`` / ``EM_GPU_MEM_UTIL`` / ``MAX_MODEL_LEN`` — vLLM knobs.
  * ``EASYMAGPIE_LT_BACKEND`` / ``EM_LT_DTYPE`` / ``EM_LT_COMPILE_MODE`` — LT path.
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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


class _LLMHolder:
    """Lazy-singleton holder for the vLLM ``LLM`` instance + virtual EOS id."""
    llm = None
    lock = threading.Lock()
    _virtual_eos_token_id: int = 0

    @classmethod
    def get(cls):
        """Return (lazily building on first call) the singleton ``LLM`` instance."""
        if cls.llm is None:
            with cls.lock:
                if cls.llm is None:
                    cls.llm = cls._build()
        return cls.llm

    @classmethod
    def _setup_lt_env(cls) -> None:
        """Apply default LT backend env vars before vLLM imports."""
        os.environ.setdefault("EASYMAGPIE_LT_BACKEND", "compile")
        os.environ.setdefault("EM_LT_COMPILE_MODE", "max-autotune")
        os.environ.setdefault("EM_LT_DTYPE", "fp16")
        # apply_model RPCs across the EngineCore subprocess use pickle fallback.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        logger.info(
            "LT backend=%s compile_mode=%s dtype=%s",
            os.environ["EASYMAGPIE_LT_BACKEND"],
            os.environ["EM_LT_COMPILE_MODE"],
            os.environ["EM_LT_DTYPE"],
        )

    @classmethod
    def _build(cls):
        """Build the vLLM ``LLM`` and cache the virtual-EOS id."""
        cls._setup_lt_env()

        from vllm import LLM

        model_dir = os.environ["EASYMAGPIE_MODEL_DIR"]
        logger.info("Building LLM with model_dir=%s ...", model_dir)

        enforce_eager = os.environ.get("EM_ENFORCE_EAGER", "0") == "1"
        dtype_str = os.environ.get("EM_DTYPE", "bfloat16")
        cudagraph_mode = os.environ.get("EM_CUDAGRAPH", "NONE").upper()

        compilation_config = None
        if not enforce_eager:
            from vllm.config import CompilationConfig
            compilation_config = CompilationConfig(cudagraph_mode=cudagraph_mode)
            logger.info("Compilation: cudagraph_mode=%s", cudagraph_mode)

        llm = LLM(
            model=model_dir,
            tokenizer=model_dir,
            skip_tokenizer_init=True,
            dtype=dtype_str,
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "512")),
            max_num_seqs=1,
            gpu_memory_utilization=float(os.environ.get("EM_GPU_MEM_UTIL", "0.4")),
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
            seed=0,
            trust_remote_code=False,
            load_format="safetensors",
            enable_prompt_embeds=True,
            # vLLM 0.19.1 V1 async scheduling skips the H2D copy of inputs_embeds
            # for second-and-later requests; sync path is correct.
            async_scheduling=False,
        )
        logger.info("LLM built (dtype=%s, enforce_eager=%s)", dtype_str, enforce_eager)

        cls._virtual_eos_token_id = int(
            llm.apply_model(lambda m: int(m.VIRTUAL_EOS_TOKEN_ID))[0]
        )
        logger.info("virtual_eos_token_id=%d", cls._virtual_eos_token_id)
        return llm


def _build_prefill_and_reset(
    model,
    ctx_codes_list,
    ctx_text_list,
    phoneme_token_ids,
    text_eos_id,
    audio_bos_id,
    audio_eos_id,
    phoneme_bos_id,
    phoneme_eos_id,
    streaming_speech_delay,
    streaming_phonemes_delay,
    lt_temperature,
    lt_topk,
):
    """Build the prefill embedding and initialise per-request streaming state.

    Runs inside the engine subprocess via ``apply_model``. Combines the
    speaker-encoded context audio with the CAS-encoded context text into a
    single prefill tensor, zeroes the per-layer KV caches defensively, and
    seeds the model's request state with the per-request constants.

    Args:
        model: the ``EasyMagpieSmallMamba`` instance.
        ctx_codes_list: ``(n_tables, T_ctx)`` audio codes.
        ctx_text_list: ``(L,)`` subword ids.
        phoneme_token_ids / text_eos_id / audio_bos_id / audio_eos_id /
            phoneme_bos_id / phoneme_eos_id / streaming_speech_delay /
            streaming_phonemes_delay / lt_temperature / lt_topk: per-request
            constants forwarded to ``init_streaming_request``.

    Returns:
        ``(T_ctx + L, embedding_dim)`` prefill embedding on CPU.
    """
    device = next(model.parameters()).device
    ctx_codes = torch.tensor(ctx_codes_list, dtype=torch.long, device=device)
    if ctx_codes.dim() == 2:
        ctx_codes = ctx_codes.unsqueeze(0)
    ctx_lens = torch.tensor([ctx_codes.shape[-1]], dtype=torch.long, device=device)
    text_ids = torch.tensor([ctx_text_list], dtype=torch.long, device=device)
    text_lens = torch.tensor([text_ids.shape[-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        prefill, _ = model.build_prefill_combined_embeddings(
            ctx_codes, ctx_lens, text_ids, text_lens,
        )

    model._last_hidden_state = None

    # Defensive KV-cache zero — microsecond cost, prevents cross-request bleed.
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
    """Drain newly-emitted audio frames from the request state's ring buffer.

    Runs inside the engine subprocess via ``apply_model``. The ``.cpu()``
    sync happens here, outside the per-step predict.

    Args:
        model: the ``EasyMagpieSmallMamba`` instance.

    Returns:
        List of frames, each a flat list of ``n_tables`` ints. Empty list
        when no new frames since the previous call.
    """
    state = getattr(model, "_request_state", None)
    if state is None:
        return []
    count = int(state.codes_count_dev.item())
    if count <= 0:
        return []
    frame_t = state.codes_buf_dev[:count].detach().cpu()
    state.codes_count_dev.zero_()
    return frame_t.flatten(start_dim=1).tolist()


app = FastAPI(title="EasyMagpie SmallMamba vLLM sidecar", version="0.1.0")


class TTSRequest(BaseModel):
    """One streaming TTS request.

    Per-request constants (BOS/EOS ids, delays) come from the agent so the
    sidecar doesn't have to hard-code them. ``temperature`` / ``topk`` must
    match the values baked into the LT engine at build time for the fused
    fast path to be taken.
    """

    context_audio_codes: list[list[int]] = Field(
        ..., description="(n_tables=16, T_ctx) audio code ids for context.",
    )
    context_text_token_ids: list[int] = Field(
        ..., description="(L,) subword token ids for context text.",
    )
    phoneme_token_ids: list[int] = Field(
        default_factory=list,
        description="Per-AR-step CAS subword ids for the utterance.",
    )
    text_eos_id: int = Field(..., description="Subword tokenizer EOS id.")
    audio_bos_id: int = Field(..., description="Audio codebook BOS id.")
    audio_eos_id: int = Field(..., description="Audio codebook EOS id.")
    phoneme_bos_id: int = Field(..., description="Phoneme tokenizer BOS id.")
    phoneme_eos_id: int = Field(..., description="Phoneme tokenizer EOS id.")
    streaming_speech_delay: int = Field(..., description="AR-step delay before audio emission.")
    streaming_phonemes_delay: int = Field(..., description="AR-step delay before phoneme emission.")
    max_frames: int = 200
    temperature: float = 0.7
    topk: int = 80


class WireTokenizerRequest(BaseModel):
    """One-shot setup payload for the CAS encoder's subword→char map.

    POST once at agent startup before any /tts/stream call.
    """
    subword_vocab: dict[str, int]
    bos_id: int
    eos_id: int
    cfg_unk_token_id: int
    subword_padding_idx: Optional[int] = None


def _wire_tokenizer(model, subword_vocab, bos_id, eos_id, cfg_unk_token_id, subword_padding_idx):
    """apply_model trampoline for ``WireTokenizerRequest``."""
    model.wire_tokenizer(
        subword_vocab=subword_vocab,
        bos_id=bos_id,
        eos_id=eos_id,
        cfg_unk_token_id=cfg_unk_token_id,
        subword_padding_idx=subword_padding_idx,
    )
    return True


@app.on_event("startup")
def _startup() -> None:
    """Build the LLM and warm the compiled LT before any client request lands."""
    llm = _LLMHolder.get()
    if os.environ.get("EASYMAGPIE_LT_BACKEND", "compile").lower() == "compile":
        try:
            llm.apply_model(lambda m: m.warmup_compiled_lt(iters=3))
            logger.info("[startup] compiled-LT warmup complete")
        except Exception as e:
            logger.exception("[startup] compiled-LT warmup failed: %s", e)


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    """Stream one TTS utterance as NDJSON.

    Args:
        req: the parsed request body.

    Returns:
        NDJSON ``StreamingResponse`` with one frame per line.
    """
    from vllm import SamplingParams

    llm = _LLMHolder.get()

    try:
        # Hoist into locals so the lambda doesn't capture ``req`` across the
        # apply_model pickle boundary.
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
            lambda m: _build_prefill_and_reset(
                m, ctx_codes, ctx_text, phoneme_ids, text_eos,
                audio_bos, audio_eos, phoneme_bos, phoneme_eos,
                speech_delay, phonemes_delay, lt_temp, lt_topk,
            )
        )
    except Exception as e:
        logger.exception("Prefill build failed")
        raise HTTPException(status_code=500, detail=f"prefill build failed: {e}") from e
    prefill = prefill_list[0]
    if prefill is None:
        raise HTTPException(500, "engine returned no prefill")

    # Phoneme-count-aware cap; audio_eos sampling is the real exit signal.
    _max_frames = req.max_frames
    if req.phoneme_token_ids:
        _max_frames = min(_max_frames, len(req.phoneme_token_ids) * 8 + 80)

    def stream():
        """Generator: drive ``engine.step()`` and yield NDJSON frame lines."""
        virtual_eos_id = _LLMHolder._virtual_eos_token_id
        sp = SamplingParams(
            temperature=req.temperature,
            top_k=req.topk,
            max_tokens=_max_frames,
            seed=0,
            ignore_eos=False,
            stop_token_ids=[virtual_eos_id],
        )

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
        # Final drain — engine is idle, no race.
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

    Args:
        req: the parsed payload.

    Returns:
        ``{"status": "ok", "vocab_size": N}``.
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
    """Return ``{"status": "ok"}`` once the LLM has loaded."""
    _LLMHolder.get()
    return {"status": "ok"}


def main() -> None:
    """CLI entry point — parse args, set env, run uvicorn."""
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--model-dir",
        default="/mnt/n1_mount/personal/vllm_omni/smallmamba_vllm_format",
        help="vLLM-format checkpoint directory. Sets $EASYMAGPIE_MODEL_DIR.",
    )
    p.add_argument(
        "--categorical-plugin-path",
        default="/tmp/categorical_plugin/libcategorical_sampling_plugin_v2.so",
        help="Path to libcategorical_sampling_plugin_v2.so (trt_fused LT only). "
             "Sets $EASYMAGPIE_CATEGORICAL_PLUGIN_PATH.",
    )
    args = p.parse_args()

    os.environ["EASYMAGPIE_MODEL_DIR"] = args.model_dir
    os.environ["EASYMAGPIE_CATEGORICAL_PLUGIN_PATH"] = args.categorical_plugin_path

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
