"""HTTP client adapter for the vLLM-backed EasyMagpie SmallMamba sidecar.

Lives in ``nemo_virtual_environment`` (which cannot import vLLM). Talks to
``/mnt/n1_mount/personal/vllm_omni/easymagpie_server/server.py`` over HTTP.

The client takes responsibility for:
  * Building ``context_audio_codes`` from a context wav file using the
    codec encoder (caller passes in a callable that does this so we don't
    depend on a specific codec class here).
  * Building ``context_text_token_ids`` from text using the caller-supplied
    tokenizer (same reason).
  * Submitting the request, streaming back per-frame audio codes, and
    yielding them one at a time to the TTS service so the codec-decode +
    pipecat audio-out loop stays exactly the same as for the in-process
    PyTorch path.

This adapter intentionally does NOT load the SmallMamba AR model on the
client side -- that model lives in the sidecar. The voice agent's TTS
service still needs the codec model + phoneme tokenizer (locally), since
those are used both for context encoding (input) and for audio decoding
(output) and live on the same GPU as the pipecat audio pipeline.

Protocol mirrors ``easymagpie_server.server`` (NDJSON, one frame per
line). See that file for the request/response schema.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Iterator

import httpx


logger = logging.getLogger(__name__)


class EasyMagpieVllmClient:
    """Streaming client for the vLLM sidecar."""

    def __init__(
        self,
        server_url: str,
        timeout_s: float = 300.0,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._timeout_s = timeout_s
        # One persistent client per service instance; httpx pools connections.
        self._http = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._http.close()

    def healthz(self) -> bool:
        try:
            r = self._http.get(f"{self._server_url}/healthz", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def wire_tokenizer(
        self,
        *,
        subword_vocab: dict,
        bos_id: int,
        eos_id: int,
        cfg_unk_token_id: int,
        subword_padding_idx: int,
    ) -> None:
        """Tell the sidecar what BPE tokenizer the CAS encoder should use.

        Must be called once at agent startup (after the agent has built its
        local tokenizer) before any /tts/stream call. Idempotent on the
        sidecar -- subsequent calls overwrite the previous wiring.
        """
        body = {
            "subword_vocab": subword_vocab,
            "bos_id": int(bos_id),
            "eos_id": int(eos_id),
            "cfg_unk_token_id": int(cfg_unk_token_id),
            "subword_padding_idx": int(subword_padding_idx),
        }
        r = self._http.post(
            f"{self._server_url}/tts/wire_tokenizer", json=body, timeout=60.0,
        )
        r.raise_for_status()

    def stream_frames(
        self,
        *,
        context_audio_codes: list[list[int]],
        context_text_token_ids: list[int],
        phoneme_token_ids: list[int],
        text_eos_id: int,
        audio_bos_id: int,
        audio_eos_id: int,
        phoneme_bos_id: int,
        phoneme_eos_id: int,
        streaming_speech_delay: int,
        streaming_phonemes_delay: int,
        max_frames: int = 300,
    ) -> Iterator[list[int]]:
        """POST a streaming request and yield codes per frame.

        Args:
            context_audio_codes: shape ``(n_tables=16, T_ctx)`` int audio
                codes for the speaker-reference context. Build with your
                codec encoder.
            context_text_token_ids: ``(L,)`` subword IDs from your
                tokenizer (matches what the SmallMamba checkpoint trained on).
            phoneme_token_ids: per-step phoneme token IDs for the
                utterance. Consumed one-per-AR-step by the sidecar's
                ``embed_input_ids`` hook and added to the audio embedding.
                Append ``eos_token_id`` to the end so the sidecar can
                stop streaming after the utterance finishes.
            eos_token_id: if set, sidecar stops streaming after consuming
                the first phoneme token equal to this value.
            max_frames: cap on number of audio frames to emit (each frame
                is ``1/25 s`` at the 25 fps codec). Default 300 = 12 s.

        Yields:
            ``list[int]`` of length 16 -- the codebook codes for one
            audio frame. Pass these straight into your codec decoder.
        """
        body = {
            "context_audio_codes": context_audio_codes,
            "context_text_token_ids": context_text_token_ids,
            "phoneme_token_ids": list(phoneme_token_ids),
            "text_eos_id": int(text_eos_id),
            "audio_bos_id": int(audio_bos_id),
            "audio_eos_id": int(audio_eos_id),
            "phoneme_bos_id": int(phoneme_bos_id),
            "phoneme_eos_id": int(phoneme_eos_id),
            "streaming_speech_delay": int(streaming_speech_delay),
            "streaming_phonemes_delay": int(streaming_phonemes_delay),
            "max_frames": max_frames,
        }
        with self._http.stream(
            "POST", f"{self._server_url}/tts/stream", json=body,
            timeout=self._timeout_s,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if "frame" in obj:
                    yield obj["codes"]
                elif "error" in obj:
                    raise RuntimeError(f"vLLM sidecar error: {obj['error']}")
                elif obj.get("done"):
                    return


def build_client_from_url(
    server_url: str, ping_first: bool = True,
) -> EasyMagpieVllmClient:
    """Convenience factory that optionally pings ``/healthz`` first so we
    fail fast at service setup if the sidecar isn't reachable."""
    c = EasyMagpieVllmClient(server_url)
    if ping_first:
        if not c.healthz():
            c.close()
            raise RuntimeError(
                f"vLLM sidecar at {server_url} did not respond to /healthz; "
                "start the server with: "
                "cd /mnt/n1_mount/personal/vllm_omni && "
                "source vllm_omni_env/bin/activate && "
                "VLLM_ALLOW_INSECURE_SERIALIZATION=1 "
                "CUDA_VISIBLE_DEVICES=1 "
                "PYTHONPATH=/mnt/n1_mount/personal/vllm_omni:"
                "/home/subhankarg/Projects/open_source/worktrees/easymp_voiceagent"
                ":$PYTHONPATH "
                "python -m easymagpie_server.server --port 18765"
            )
    return c
