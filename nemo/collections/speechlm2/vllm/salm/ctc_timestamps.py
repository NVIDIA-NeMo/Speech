# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Keep CTC timestamp inputs per request inside vLLM and align finished transcripts.

Only requests that opt in keep anything: they set ``"capture_ctc_timestamps": True`` in
vLLM's ``mm_processor_kwargs``, offline in ``LLM.chat`` or ``LLM.generate`` and served
in the chat completion request body.

The encoder returns ``CTCTimestampInputs`` (ASR states, Sortformer speaker
probabilities, 10 ms diarization labels) from the same forward that feeds the LLM; the
CTC head runs only when a finished transcript is aligned, in the checkpoint's
``MultiSpeakerSOTWordTimestampAligner``. This module holds those inputs between the two
points. vLLM makes that awkward in two ways:

* ``embed_multimodal(**mm_kwargs_batch)`` receives only tensors, so the inputs have
  no request id or ``mm_hash`` to be keyed by when they are produced.
* The scheduler only runs the encoder for audio whose ``mm_hash`` is absent from the
  encoder cache, so a request served from that cache produces no inputs of its own.

Both are solved by keying on ``mm_hash``, the identity vLLM already uses. A hook on
``_execute_mm_encoder`` reads each step's hashes before the encoder runs and hands them
to the forward through per-thread state, in the order vLLM encodes the items, so each
row's inputs are stored under its audio's hash. Every new request is mapped to its
hashes, so a cache hit finds the inputs its audio produced earlier. The hook calls
``_batch_mm_inputs_from_scheduler``, whose signature matches in vLLM 0.23 and 0.28.

The inputs move to pinned host memory without blocking the engine. An ASR state is
1,280 values per 80 ms frame, far less than the vocabulary-wide CTC output, so the
copy overlaps with compute instead of stalling it. Once a copy lands, each row is
compacted to its own valid frames, so retaining it does not keep its whole padded
batch alive.

vLLM's own encoder cache drops a request's claim once prefill has consumed the
embeddings; a capture is read later, after the request has finished, by a caller vLLM
does not know about. So each capture records the requests that own it (repeated audio
shares one through the encoder cache). A capture is deleted once the last of them is
aligned with ``release=True`` or released through :func:`ctc_release`, and vLLM has
evicted its audio from the encoder cache: until then a new request with that audio is
served from the cache and captures nothing, so it needs the old capture. Stored captures
are bounded by ``NEMO_CTC_TIMESTAMP_RETAIN_GB`` of host memory (default 8): captures no
request owns are evicted first, least recently used first, and when owned captures alone
fill the budget, new captures are refused, so no request loses a capture it already has.
A result without word timestamps names the reason in its ``error``; its 10 ms
diarization, which needs only the capture, is still returned.

Alignment is split where its work changes kind. The engine process, where the inputs
live, runs the CTC head on the device and keeps only the log-prob columns each
transcript's tokens use, window by window, in batches bounded by ``_HEAD_FRAME_BUDGET``
(:func:`prepare_finished_requests`). Both modes reach it through one worker method and
``collective_rpc``. The caller then runs the CPU-bound
alignment search on that compact batch (:func:`align_prepared_requests`), so the engine
is busy only for the head. The client functions :func:`align` and
:func:`release_captures` serve a synchronous ``LLM``, which is what
:func:`ctc_timestamps`, :func:`ctc_word_timestamps` and :func:`ctc_release` do;
:func:`align_async` and :func:`release_captures_async` serve a server's async engine
client, which is what ``ctc_serving`` does. Callers pass the external request id, which
vLLM's scheduler knows with a random suffix appended.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import threading
import weakref
from collections import Counter, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode

# Bounds the host memory captures take; an hour of audio keeps on the order of
# 0.1-0.2 GB of encoder states.
_RETENTION_GB_ENV = "NEMO_CTC_TIMESTAMP_RETAIN_GB"
_DEFAULT_RETENTION_GB = 8.0

# Requests nobody aligns or releases would otherwise stay mapped forever.
_MAX_TRACKED_REQUESTS = 1 << 16

# Worker method names reached through collective_rpc.
WORKER_PREPARE_METHOD = "nemo_ctc_prepare_requests"
WORKER_RELEASE_METHOD = "nemo_ctc_release_requests"

# Marks a tensor packed as dtype, shape and bytes in a worker reply. vLLM rebuilds
# tensors in a collective_rpc result only with VLLM_ALLOW_INSECURE_SERIALIZATION; bytes
# always arrive intact.
_PACKED_TENSOR = "__tensor__"

# vLLM forms the scheduler's request id as f"{external_id}-{random_uuid():.8}".
_INTERNAL_ID_SUFFIX_LEN = 8

# Requests decoded together in one deferred-head call, at most.
_ALIGN_BATCH = 16

# Bounds a deferred-head call by its requests times the frames the head decodes at
# once: a window, or the longest request when it fits in one. The head's
# vocabulary-wide output for those frames dominates the device memory preparing
# takes, about 1.6 GiB at this budget with a 32,769-class BF16 head.
_HEAD_FRAME_BUDGET = 8192

# Records a server merges into one alignment search (see _AlignmentBatcher). The
# search's per-frame loop is shared, so more records cost little more, short ones
# padded to the longest.
_SERVER_ALIGN_RECORDS = 32

# Weight of the Sortformer speaker-activity prior in CTC alignment, overridable per
# checkpoint as ctc_timestamps.speaker_logprob_weight. The aligner's own default is
# 0.0, which lets words in overlapped speech drift out of their speaker's turns.
DEFAULT_SPEAKER_PRIOR_WEIGHT = 0.25

# CTCTimestampInputs tensor fields: the time axis to trim and pad along, and the
# lengths field marking its valid frames (None for the lengths fields themselves).
_FIELDS = (
    ("asr_encoded", 2, "asr_encoded_lengths"),
    ("asr_encoded_lengths", None, None),
    ("sortformer_sigmoids", 1, "sortformer_lengths"),
    ("sortformer_lengths", None, None),
    ("diarization_labels", 2, "diarization_lengths"),
    ("diarization_lengths", None, None),
)

# The current encoder step's hashes stay thread-local: the hook hands them to the
# forwards it runs, on the engine thread.
_state = threading.local()

# Process-wide, unlike _state, and guarded by _lock. With an in-process engine the
# alignment call runs on the caller's thread while the encoder runs on the engine
# thread, so anything shared between them must not be thread-local.
_registry: dict[str, Any] = {}
_store: dict[str, dict] = {}
_request_hashes: dict[str, list[str]] = {}
# Request -> how many multimodal items its prompt holds. Identical clips share one hash,
# so _request_hashes alone cannot tell two of them from one.
_request_items: dict[str, int] = {}
# mm_hash -> the requests that still own its capture.
_hash_owners: dict[str, set[str]] = {}
# Hashes whose audio vLLM's encoder cache holds. A new request with that audio is
# served from the cache and captures nothing, so it needs the existing capture.
_engine_cached: set[str] = set()
# External request id -> the internal id the scheduler reported for it.
_external_ids: dict[str, str] = {}
# Stored rows that still view their forward's padded host buffers, oldest first.
_uncompacted: deque[dict] = deque()
# Reentrant because the alignment lookup resolves request ids while holding it.
_lock = threading.RLock()
# One alignment batcher per event loop; a server has one.
_batchers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def register_aligner(aligner: Any, special_tokens: Sequence[str] = ()) -> None:
    """Publish the checkpoint's CTC timestamp aligner, which also turns input capture on.

    The runner hook and the worker methods reached through ``collective_rpc`` never
    receive the model instance. One engine hosts one model per process, so a
    module-level reference is enough to bridge that.

    Args:
        aligner (Any): The ``MultiSpeakerSOTWordTimestampAligner`` holding the deferred CTC head.
        special_tokens (Sequence[str]): The LLM tokenizer's special tokens, which preparing removes
            from each transcript, except ``<spk:N>`` speaker tags.
    """
    _registry["aligner"] = aligner
    _registry["special_tokens"] = tuple(special_tokens)


def active_aligner() -> Any:
    """Return the CTC timestamp aligner, or ``None`` when timestamps are not enabled."""
    return _registry.get("aligner")


def require_v1_model_runner(vllm_config: Any) -> None:
    """Refuse CTC timestamps on vLLM's Model Runner V2.

    Captured inputs are keyed, trimmed and compacted by a hook on the V1 runner's
    ``GPUModelRunner._execute_mm_encoder``. Under V2, the default from vLLM 0.30 on,
    that hook never runs: nothing would be aligned, and the inputs would pile up in
    host memory.

    Args:
        vllm_config (Any): The engine's ``VllmConfig``; without ``use_v2_model_runner``
            (older vLLM) the V1 runner is assumed.
    """
    if getattr(vllm_config, "use_v2_model_runner", False):
        raise ValueError(
            "CTC timestamps need vLLM's V1 GPU model runner, but this engine uses Model Runner V2; "
            "set VLLM_USE_V2_MODEL_RUNNER=0."
        )


def read_speaker_prior_weight(ctc_config: Any) -> float:
    """Read the Sortformer prior weight from a checkpoint's ``ctc_timestamps`` block.

    The config key is ``speaker_logprob_weight``, after the aligner's constructor
    parameter. The aligner validates it only in that constructor, and the plugin sets
    it on an aligner that is already built, so it is validated here instead.

    Args:
        ctc_config (Any): The ``ctc_timestamps`` block, as a dict or an attribute object.

    Returns:
        float: The configured weight, or ``DEFAULT_SPEAKER_PRIOR_WEIGHT`` when unset.
    """
    value = (
        ctc_config.get("speaker_logprob_weight")
        if isinstance(ctc_config, dict)
        else getattr(ctc_config, "speaker_logprob_weight", None)
    )
    speaker_prior_weight = float(DEFAULT_SPEAKER_PRIOR_WEIGHT if value is None else value)
    if speaker_prior_weight < 0:
        raise ValueError(f"ctc_timestamps.speaker_logprob_weight must be non-negative; got {speaker_prior_weight}.")
    return speaker_prior_weight


def ctc_adapter_path(ctc_config: Any) -> str | None:
    """Return the adapter path of a checkpoint's ``ctc_timestamps`` block; set means timestamps are on.

    Args:
        ctc_config (Any): The ``ctc_timestamps`` block, as a dict or an attribute object.
    """
    if not ctc_config:
        return None
    if isinstance(ctc_config, dict):
        return ctc_config.get("adapter_path")
    return getattr(ctc_config, "adapter_path", None)


def ctc_timestamp_config(model_config: Any) -> Any:
    """Return the ``ctc_timestamps`` block that turns timestamps on for a model, or ``None`` when they are off.

    ``"enabled": false`` turns them off. Otherwise the checkpoint config's block when it names an
    adapter. Otherwise a checkpoint that bundles the CTC head, in ``model.safetensors`` or in one
    shard of a sharded checkpoint (``ctc_timestamp_format`` in its metadata, the head under
    ``ctc_timestamp.``), is its own adapter, with the block's speaker prior weight. Such a block is ``"optional"`` unless the
    config sets ``"enabled": true``: where the engine cannot produce timestamps, the model then
    serves without them instead of failing to start.

    Args:
        model_config (Any): vLLM's ``ModelConfig``: its ``hf_config`` and the local ``model`` directory.

    Raises:
        ValueError: ``"enabled": true``, but the config names no adapter and the checkpoint
            bundles no head.
    """
    ctc_config = getattr(model_config.hf_config, "ctc_timestamps", None)
    enabled = ctc_config.get("enabled") if isinstance(ctc_config, dict) else getattr(ctc_config, "enabled", None)
    if enabled is False:
        return None
    if ctc_adapter_path(ctc_config):
        return ctc_config
    bundled = _bundled_ctc_head(str(getattr(model_config, "model", None) or ""))
    if bundled is None:
        if enabled:
            raise ValueError(
                "ctc_timestamps.enabled is true, but the config names no adapter_path and the checkpoint "
                "bundles no CTC timestamp head."
            )
        return None
    return {
        "adapter_path": bundled,
        "speaker_logprob_weight": read_speaker_prior_weight(ctc_config),
        "optional": not enabled,
    }


@functools.lru_cache(maxsize=None)
def _bundled_ctc_head(model: str) -> str | None:
    """The safetensors file of a local checkpoint directory that holds its bundled CTC timestamp head.

    That is ``model.safetensors``, or in a sharded checkpoint the one shard its index maps every
    ``ctc_timestamp.`` tensor to, as quantized exports write it. The file's metadata must name the
    head's format.
    """
    from safetensors import safe_open

    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
        CTC_TIMESTAMP_ARTIFACT_FORMAT,
        CTC_TIMESTAMP_BUNDLE_PREFIX,
    )
    from nemo.collections.speechlm2.parts.hf_hub import SAFETENSORS_SINGLE_FILE

    if not model:
        return None
    path = Path(model) / SAFETENSORS_SINGLE_FILE
    index = Path(model) / f"{SAFETENSORS_SINGLE_FILE}.index.json"
    if not path.is_file() and index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        shards = sorted({shard for name, shard in weight_map.items() if name.startswith(CTC_TIMESTAMP_BUNDLE_PREFIX)})
        if len(shards) > 1:
            logging.warning(
                "[NeMoSpeechLM] The checkpoint's CTC timestamp head spans %d shards (%s), so it is not loaded; "
                "point ctc_timestamps.adapter_path at one file that holds the whole head.",
                len(shards),
                ", ".join(shards),
            )
        if len(shards) != 1:
            return None
        path = Path(model) / shards[0]
    if not path.is_file():
        return None
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    return str(path.resolve()) if metadata.get("ctc_timestamp_format") == CTC_TIMESTAMP_ARTIFACT_FORMAT else None


def tokenizer_special_tokens(model_config: Any) -> tuple[str, ...]:
    """Return the special tokens of the model's tokenizer, which decoding can leave in a transcript.

    Args:
        model_config (Any): vLLM's ``ModelConfig``. ``None``, or a config with
            ``skip_tokenizer_init``, has no tokenizer and so no special tokens.
    """
    if model_config is None or getattr(model_config, "skip_tokenizer_init", False):
        return ()
    try:
        from vllm.tokenizers import cached_tokenizer_from_config
    except ImportError:  # vLLM releases before vllm.tokenizers
        from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
    return tuple(getattr(cached_tokenizer_from_config(model_config), "all_special_tokens", ()))


def take_step_hashes(count: int) -> list[str] | None:
    """Return the ``mm_hash`` of each item in one encoder forward, in item order.

    The hook around vLLM's ``_execute_mm_encoder`` lists the step's hashes before the
    encoder runs, in the order vLLM encodes the items, and each forward takes the next
    ``count`` of them. One step can run several forwards, because vLLM batches
    consecutive items only while they share a modality and fields, so a forward must
    leave the rest of the hashes to the forwards after it. An audio-only model
    normally runs one forward per step.

    Args:
        count (int): Number of multimodal items in the forward.

    Returns:
        list[str] | None: One hash per item, or ``None`` when nothing should be stored:
        the forward runs outside the hook (vLLM's startup profiling pass), or it has
        more items than the step has hashes left.
    """
    step = getattr(_state, "step", None)
    if step is None:
        return None
    step["rows"] += count
    if count > len(step["hashes"]):
        return None
    keys, step["hashes"] = step["hashes"][:count], step["hashes"][count:]
    step["taken"].extend(keys)
    return keys


def store_alignment_states(
    mm_hashes: Sequence[str | None], alignment_states: Any, audio_durations: torch.Tensor | Sequence[float]
) -> None:
    """Keep one forward's alignment states per item, in host memory, under each item's ``mm_hash``.

    The alignment states are the encoder's ``CTCTimestampInputs``: what the deferred CTC
    head and the aligner read once the transcript is known.

    Args:
        mm_hashes (Sequence[str | None]): One ``mm_hash`` per batch row, from
            :func:`take_step_hashes`; ``None`` skips the row.
        alignment_states (CTCTimestampInputs): The encoder's ``CTCTimestampInputs`` for the batch.
        audio_durations (torch.Tensor | Sequence[float]): Audio duration per row, in
            seconds. A device tensor is copied along with the states, so the forward
            never waits to read it.
    """
    if not isinstance(audio_durations, torch.Tensor):
        audio_durations = torch.tensor(audio_durations, dtype=torch.float64)
    host = {name: _to_host(getattr(alignment_states, name)) for name, _, _ in _FIELDS}
    durations = _to_host(audio_durations)
    on_device = audio_durations.is_cuda or any(
        getattr(alignment_states, name) is not None and getattr(alignment_states, name).is_cuda
        for name, _, _ in _FIELDS
    )
    # Rows are read only after this event, so the copies never block the forward.
    ready = torch.cuda.current_stream().record_event() if on_device else None
    with _lock:
        for row, mm_hash in enumerate(mm_hashes):
            if mm_hash is None:
                continue
            entry = {name: None if tensor is None else tensor[row : row + 1] for name, tensor in host.items()}
            entry.update(
                diarization_frame_seconds=alignment_states.diarization_frame_seconds,
                duration=durations[row : row + 1],
                ready=ready,
            )
            entry["nbytes"] = _entry_nbytes(entry)
            # Re-encoded audio replaces its older capture and becomes the most recent,
            # which assigning to the existing key would not do.
            _drop_capture(mm_hash)
            _store[mm_hash] = entry
            _uncompacted.append(entry)


def _to_host(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Copy a tensor to host memory; pinned and non-blocking when it is on the GPU."""
    if tensor is None:
        return None
    if not tensor.is_cuda:
        return tensor.detach().clone()
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor.detach(), non_blocking=True)
    return host


def _entry_nbytes(entry: dict) -> int:
    """Host bytes of an entry's tensors, each counted at its own (row) size."""
    return sum(entry[name].numel() * entry[name].element_size() for name, _, _ in _FIELDS if entry[name] is not None)


def _resolve_request_id(request_id: str) -> str | None:
    """Return the recorded key for a request id, accepting the external form.

    Callers holding a ``RequestOutput`` see the external id, while the scheduler
    reports the internal one; exact matches win so ids without a suffix (older vLLM,
    in-process callers) resolve unchanged. Must hold ``_lock``.
    """
    if request_id in _request_hashes:
        return request_id
    internal = _external_ids.get(request_id)
    return internal if internal in _request_hashes else None


def _external_id(internal_id: str) -> str | None:
    """Strip vLLM's ``-{8 chars}`` request id suffix, or ``None`` when there is none."""
    cut = len(internal_id) - _INTERNAL_ID_SUFFIX_LEN - 1
    return internal_id[:cut] if cut > 0 and internal_id[cut] == "-" else None


def mm_hashes_for_request(request_id: str) -> list[str]:
    """Return the multimodal hashes seen for a request, in arrival order.

    Args:
        request_id (str): vLLM request id, internal or external.

    Returns:
        list[str]: Hashes whose inputs belong to this request.
    """
    with _lock:
        key = _resolve_request_id(request_id)
        return list(_request_hashes[key]) if key is not None else []


def _record_request_hashes(pairs) -> None:
    """Associate a request with the hashes whose inputs belong to it, and make it an owner."""
    with _lock:
        for req_id, mm_hash in pairs:
            hashes = _request_hashes.setdefault(req_id, [])
            if mm_hash not in hashes:
                hashes.append(mm_hash)
            _hash_owners.setdefault(mm_hash, set()).add(req_id)
            external = _external_id(req_id)
            if external is not None:
                _external_ids[external] = req_id
        while len(_request_hashes) > _MAX_TRACKED_REQUESTS:
            _forget_request(next(iter(_request_hashes)))


def _record_new_requests(scheduler_output: Any) -> None:
    """Map newly scheduled requests to their hashes, and count each one's multimodal items."""
    pairs = list(_new_request_hashes(scheduler_output))
    with _lock:
        # Counted first, so a request the hash map then forgets loses its count too.
        _request_items.update(Counter(req_id for req_id, _ in pairs))
    _record_request_hashes(pairs)


def _drop_capture(mm_hash: str) -> None:
    """Delete a capture and tell compaction to skip it. Must hold ``_lock``."""
    entry = _store.pop(mm_hash, None)
    if entry is not None:
        entry["dropped"] = True


def _forget_request(req_id: str) -> None:
    """Drop a request's claims and delete the captures nobody needs anymore. Must hold ``_lock``."""
    _request_items.pop(req_id, None)
    for mm_hash in _request_hashes.pop(req_id, ()):
        owners = _hash_owners.get(mm_hash)
        if owners is None:
            continue
        owners.discard(req_id)
        if not owners:
            del _hash_owners[mm_hash]
            if mm_hash not in _engine_cached:
                _drop_capture(mm_hash)
    external = _external_id(req_id)
    if external is not None and _external_ids.get(external) == req_id:
        del _external_ids[external]


def _follow_engine_cache(freed: Sequence[str] = (), encoded: Sequence[str] = ()) -> None:
    """Track which audio vLLM's encoder cache holds, deleting unowned captures it evicted."""
    with _lock:
        for mm_hash in freed:
            _engine_cached.discard(mm_hash)
            if mm_hash not in _hash_owners:
                _drop_capture(mm_hash)
        _engine_cached.update(encoded)


def _release_requests(request_ids: Sequence[str]) -> None:
    """Drop these requests' claims on their captures, accepting external ids."""
    with _lock:
        for request_id in request_ids:
            key = _resolve_request_id(request_id)
            if key is not None:
                _forget_request(key)


def _byte_limit() -> int:
    """How many bytes of stored inputs to hold before evicting the least recently used."""
    try:
        gigabytes = float(os.environ.get(_RETENTION_GB_ENV, _DEFAULT_RETENTION_GB))
    except ValueError:
        gigabytes = _DEFAULT_RETENTION_GB
    return int(max(gigabytes, 0.0) * 1e9)


def _begin_step(mm_hashes: Sequence[str]) -> None:
    """Hand an encoder step's hashes, in encoding order, to the forwards it runs."""
    _state.step = {"hashes": list(mm_hashes), "scheduled": len(mm_hashes), "rows": 0, "taken": []}


def _end_step() -> None:
    """Close the encoder step, dropping its captures when its rows did not match its hashes.

    vLLM pairs its own encoder outputs with ``mm_hashes`` by position, and so does
    :func:`take_step_hashes`; a different number of rows means some captures may sit
    under another item's hash, and no timestamps are better than wrong ones.
    """
    step, _state.step = getattr(_state, "step", None), None
    if step is None or step["rows"] == step["scheduled"]:
        return
    logging.warning(
        "[NeMoSpeechLM] The encoder produced %d rows for %d scheduled audio items; dropping this step's CTC "
        "timestamp captures.",
        step["rows"],
        step["scheduled"],
    )
    with _lock:
        for mm_hash in step["taken"]:
            _drop_capture(mm_hash)


def _trim_store(new_hashes: Sequence[str] = ()) -> int:
    """Bring stored inputs under the byte budget without deleting a capture a request owns.

    Captures normally go when their last owner is aligned or released, so the least
    recently used captures no request owns go first: audio vLLM's encoder cache still
    holds after its requests were aligned, and the captures of forgotten requests. Python
    dicts preserve insertion order and alignment re-inserts what it keeps, so the first
    keys are the least recently used. When owned captures alone exceed the budget, the
    captures of ``new_hashes`` are refused, newest first, so that no request loses a
    capture it already has; each refused request reports ``capture_unavailable``. The
    newest capture is kept even when it alone exceeds the budget.

    Args:
        new_hashes (Sequence[str]): The hashes the current encoder step stored for the first time,
            in encoding order. Audio encoded again replaces a capture already counted, so it is
            not among them.

    Returns:
        int: How many captures were deleted.
    """
    byte_limit = _byte_limit()
    evicted = refused = 0
    with _lock:
        stored = sum(entry["nbytes"] for entry in _store.values())
        unowned = [mm_hash for mm_hash in _store if mm_hash not in _hash_owners] if stored > byte_limit else []
        for mm_hash in unowned:
            if stored <= byte_limit or len(_store) <= 1:
                break
            stored -= _store[mm_hash]["nbytes"]
            _drop_capture(mm_hash)
            evicted += 1
        for mm_hash in reversed(new_hashes):
            if stored <= byte_limit or len(_store) <= 1:
                break
            if mm_hash in _store:
                stored -= _store[mm_hash]["nbytes"]
                _drop_capture(mm_hash)
                refused += 1
    if evicted:
        logging.warning(
            "[NeMoSpeechLM] Evicting CTC timestamp captures no request owns to stay under %.1f GB (logged "
            "once); raise %s to keep more.",
            byte_limit / 1e9,
            _RETENTION_GB_ENV,
            mode=LogMode.ONCE,
        )
    if refused:
        logging.warning(
            "[NeMoSpeechLM] Requests not yet aligned or released hold %.1f GB of CTC timestamp captures, so new "
            "captures are refused and their requests get no timestamps (logged once). Align or release requests "
            "sooner (offline, ctc_release()), or raise %s.",
            byte_limit / 1e9,
            _RETENTION_GB_ENV,
            mode=LogMode.ONCE,
        )
    return evicted + refused


def _compact_ready() -> int:
    """Compact stored rows whose host copies have landed, oldest first, without waiting.

    A row starts as a view of its forward's padded host buffers, which would stay
    alive for as long as any row of that forward is retained. Copies complete in
    stream order, so the first row still in flight ends the pass.
    """
    compacted = 0
    with _lock:
        while _uncompacted:
            entry = _uncompacted[0]
            if entry["ready"] is not None and not entry["ready"].query():
                break
            _uncompacted.popleft()
            if not entry.get("dropped"):
                _compact(entry)
                compacted += 1
    return compacted


def _compact(entry: dict) -> None:
    """Replace an entry's views of its forward's buffers with copies of its valid frames."""
    for name, time_axis, lengths_name in _FIELDS:
        tensor = entry[name]
        if tensor is None:
            continue
        if time_axis is not None and entry[lengths_name] is not None:
            valid = min(int(entry[lengths_name][0]), tensor.shape[time_axis])
            tensor = tensor.narrow(time_axis, 0, valid)
        entry[name] = tensor.clone()
    entry["duration"] = float(entry["duration"])
    entry["ready"] = None
    entry["nbytes"] = _entry_nbytes(entry)


def _empty_result(error: str | None = None, diarization: Sequence[dict] = ()) -> dict:
    """A result without words; ``diarization`` comes from the capture alone, when there is one."""
    return {"words": [], "diarization": list(diarization), "speaker_tag_to_diarization_speaker": {}, "error": error}


def align_finished_requests(finished: Sequence[tuple[str, str]], *, release: bool) -> list[dict]:
    """Align the transcripts of finished requests against the inputs captured while they ran.

    :func:`prepare_finished_requests` followed by :func:`align_prepared_requests`, in one
    process.

    Args:
        finished (Sequence[tuple[str, str]]): ``(request_id, transcript)`` pairs of
            requests that have finished generating; the transcript is aligned exactly
            as given, speaker tags included.
        release (bool): Afterwards drop each request's claim on its capture, so the
            capture can be deleted. Pass ``False`` to align the same requests again.

    Returns:
        list[dict]: Per item, ``words`` (``word``/``start``/``end``/``speaker`` in
        start-time order, speaker being the transcript's ``<spk:N>`` tag),
        ``diarization`` (10 ms Sortformer activity segments, ``speaker`` being the
        Sortformer speaker index), ``speaker_tag_to_diarization_speaker`` linking the
        two, and ``error``. Score diarization (DER) with ``diarization`` only: word
        timestamps mark where each word is, not continuous speech activity, so a DER
        computed from them is poor. ``diarization`` needs only the capture, so it is
        returned even when the words cannot be aligned. ``error`` is ``None`` when the
        words are complete, as for an empty transcript, which has none; otherwise it says
        why ``words`` is empty: ``"not_enabled"`` (timestamps are not enabled),
        ``"no_capture"`` (no capture is recorded for the request: it did not opt in,
        carried no audio, or was already released), ``"multiple_audio"`` (it has more
        than one audio item), ``"capture_unavailable"`` (its capture was not kept, to stay
        under ``NEMO_CTC_TIMESTAMP_RETAIN_GB``), or ``"alignment_failed"`` (the aligner
        cannot align the transcript; ``diarization`` is still returned).

    Raises:
        Exception: Any aligner error other than ``ValueError``, which is how the
            aligner rejects a transcript it cannot align.
    """
    return align_prepared_requests(prepare_finished_requests(finished, release=release))


def prepare_finished_requests(finished: Sequence[tuple[str, str]], *, release: bool) -> dict:
    """Run the deferred CTC head for finished requests and keep what aligning them needs.

    The half of :func:`align_finished_requests` that needs the captured inputs, the CTC head
    and the tokenizer, so it runs where the inputs live. Each prepared batch holds, per
    request, only the log-prob columns its transcript's tokens use; once prepared, the
    captures are no longer needed. Each transcript first loses the special tokens the
    decoder kept, except ``<spk:N>`` speaker tags (``clean_sot_transcript``).

    Args:
        finished (Sequence[tuple[str, str]]): As in :func:`align_finished_requests`.
        release (bool): Afterwards drop each request's claim on its capture, also when
            preparing fails.

    Returns:
        dict: ``count``, the number of items; ``errors``, per item the ``error`` its result
        reports (see :func:`align_finished_requests`); ``diarization``, per item in no
        batch its diarization from the capture alone, or ``None``; and ``batches``: each
        ``items``, the indices it covers, and ``prepared``, a batch for
        :func:`align_prepared_requests`. Items in no batch get results without words.

    Raises:
        Exception: Any error other than ``ValueError`` while preparing.
    """
    try:
        return _prepare_finished_requests(finished)
    finally:
        if release:
            _release_requests([request_id for request_id, _ in finished])


def _prepare_finished_requests(finished: Sequence[tuple[str, str]]) -> dict:
    """:func:`prepare_finished_requests` without releasing, which that function does on every exit."""
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import MultiSpeakerSOTWordTimestampAligner

    errors: list[str | None] = [None] * len(finished)
    diarization: list[list[dict] | None] = [None] * len(finished)
    reply = {"count": len(finished), "batches": [], "errors": errors, "diarization": diarization}
    aligner = active_aligner()
    if aligner is None:
        return {**reply, "errors": ["not_enabled"] * len(finished)}
    special_tokens = _registry.get("special_tokens", ())
    pending, wordless, no_audio, multi_audio, evicted = [], [], [], [], []
    with _lock:
        for index, (request_id, text) in enumerate(finished):
            hashes = mm_hashes_for_request(request_id)
            if not hashes:
                no_audio.append(request_id)
                errors[index] = "no_capture"
                continue
            if len(hashes) > 1 or _request_items.get(_resolve_request_id(request_id), 0) > 1:
                # The transcript spans all of the request's audio, but each item has its
                # own inputs and timeline, so aligning it to any one item would be wrong.
                # The same clip twice is two items with one hash.
                multi_audio.append(request_id)
                errors[index] = "multiple_audio"
                continue
            key = next((h for h in hashes if h in _store), None)
            if key is None:
                evicted.append(request_id)
                errors[index] = "capture_unavailable"
                continue
            # Re-insert to mark it recently used; releasing happens after preparing.
            entry = _store.pop(key)
            _store[key] = entry
            text = MultiSpeakerSOTWordTimestampAligner.clean_sot_transcript(text, special_tokens)
            # A snapshot, so compaction on the engine thread cannot swap its tensors
            # while they are being read. An empty transcript has no words to align.
            (pending if text else wordless).append((index, text, dict(entry)))
    for index, _, entry in wordless:
        diarization[index] = _capture_diarization(entry)
    if multi_audio:
        logging.warning(
            "[NeMoSpeechLM] CTC timestamps need one audio item per request; skipped %d requests with more "
            "(first: %s).",
            len(multi_audio),
            multi_audio[0],
        )
    if no_audio:
        logging.warning(
            "[NeMoSpeechLM] No capture is recorded for %d of %d requests (first: %s): they did not opt in with "
            'mm_processor_kwargs={"capture_ctc_timestamps": True}, carried no audio, were already aligned or '
            "released with release=True, or the request id is unknown.",
            len(no_audio),
            len(finished),
            no_audio[0],
        )
    if evicted:
        logging.warning(
            "[NeMoSpeechLM] The captures of %d of %d requests (first: %s) were not kept, to stay under %.1f GB; "
            "align or ctc_release() requests sooner, or raise %s.",
            len(evicted),
            len(finished),
            evicted[0],
            _byte_limit() / 1e9,
            _RETENTION_GB_ENV,
        )
    # A transcript without any tag is aligned as one speaker: the model wrote none, or decoding removed them.
    untagged = [finished[index][0] for index, text, _ in pending if "<spk:" not in text]
    if untagged:
        logging.warning(
            "[NeMoSpeechLM] %d of %d transcripts have no <spk:N> speaker tags (first: %s) and are aligned as "
            "one speaker: the model wrote none, or skip_special_tokens=True removed them.",
            len(untagged),
            len(pending),
            untagged[0],
        )

    device = next(aligner.ctc_decoder.parameters()).device
    for chunk in _head_batches(aligner, pending):
        for items, prepared in _prepare_chunk(aligner, chunk, device):
            reply["batches"].append({"items": items, "prepared": prepared})
    prepared_items = {index for batch in reply["batches"] for index in batch["items"]}
    for index, _, entry in pending:
        if index not in prepared_items:
            errors[index] = "alignment_failed"
            diarization[index] = _capture_diarization(entry)
    return reply


def align_prepared_requests(reply: dict) -> list[dict]:
    """Align the batches :func:`prepare_finished_requests` prepared, in any process.

    Needs no model, tokenizer or device, so a caller can run the CPU-bound alignment search
    outside the engine.

    Args:
        reply (dict): The reply of :func:`prepare_finished_requests`.

    Returns:
        list[dict]: Per item, the result described in :func:`align_finished_requests`.

    Raises:
        Exception: Any aligner error other than ``ValueError``.
    """
    return _results_from(reply, [_align_batch(batch["prepared"]) for batch in reply["batches"]])


def _results_from(reply: dict, aligned: Sequence[list[dict | None]]) -> list[dict]:
    """Public results per item of ``reply``, from each of its batches' aligner results."""
    errors = reply.get("errors") or [None] * reply["count"]
    diarization = reply.get("diarization") or [None] * reply["count"]
    results: list[dict] = [_empty_result(error, segments or ()) for error, segments in zip(errors, diarization)]
    for batch, batch_results in zip(reply["batches"], aligned):
        prepared = batch["prepared"]
        for index, record, result in zip(batch["items"], prepared["records"], batch_results):
            if result is None:
                results[index] = _empty_result("alignment_failed", _record_diarization(prepared, record))
            else:
                results[index] = _public_result(result)
    return results


def _capture_diarization(entry: dict) -> list[dict]:
    """Public diarization of a stored capture, from its 10 ms labels alone."""
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import MultiSpeakerSOTWordTimestampAligner

    labels = entry["diarization_labels"]
    if labels is None:
        return []
    if entry["ready"] is not None:
        entry["ready"].synchronize()
    lengths = entry["diarization_lengths"]
    return _public_diarization(
        MultiSpeakerSOTWordTimestampAligner.diarization_segments(
            labels[0, :, : labels.shape[-1] if lengths is None else int(lengths[0])],
            frame_seconds=entry["diarization_frame_seconds"],
            time_offset=0.0,
            audio_duration=float(entry["duration"]),
        )
    )


def _record_diarization(prepared: dict, record: dict) -> list[dict]:
    """Public diarization of a prepared record, from its 10 ms labels alone."""
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import MultiSpeakerSOTWordTimestampAligner

    if record.get("diarization_labels") is None:
        return []
    return _public_diarization(
        MultiSpeakerSOTWordTimestampAligner.diarization_segments(
            record["diarization_labels"],
            frame_seconds=prepared["diarization_frame_seconds"],
            time_offset=record["time_offset"],
            audio_duration=record["audio_duration"],
        )
    )


def _head_batches(aligner: Any, pending: list) -> list[list]:
    """Split pending requests, in order, into deferred-head batches.

    A batch holds at most ``_ALIGN_BATCH`` requests, and its requests times the frames the
    head decodes at once stay within ``_HEAD_FRAME_BUDGET``; a request over the budget on
    its own forms a batch by itself.
    """
    core = int(getattr(aligner, "online_inference_length", 0))
    window = core + int(getattr(aligner, "chunk_left_context", 0)) + int(getattr(aligner, "chunk_right_context", 0))

    def head_frames(frames: int) -> int:
        # Collation pads every request to the longest, which the head decodes whole when it
        # fits in one core window, and otherwise window by window.
        return frames if core <= 0 or frames <= core else min(frames, window)

    batches: list[list] = []
    longest = 0
    for item in pending:
        frames = item[2]["asr_encoded"].shape[-1]
        if (
            batches
            and len(batches[-1]) < _ALIGN_BATCH
            and (len(batches[-1]) + 1) * head_frames(max(longest, frames)) <= _HEAD_FRAME_BUDGET
        ):
            batches[-1].append(item)
            longest = max(longest, frames)
        else:
            batches.append([item])
            longest = frames
    return batches


def _prepare_chunk(aligner: Any, chunk: list, device: torch.device) -> list[tuple[list[int], dict]]:
    """Run the deferred head for one batch from :func:`_head_batches` and prepare its alignment."""
    entries = [entry for _, _, entry in chunk]
    prepared = None
    try:
        prepared = aligner.prepare_from_inputs(
            _collate(entries, device),
            [text for _, text, _ in chunk],
            # Read only after _collate has waited for the host copies.
            [float(entry["duration"]) for entry in entries],
        )
    except Exception as error:  # noqa: BLE001
        if len(chunk) == 1:
            if not isinstance(error, ValueError):
                raise
            logging.warning("[NeMoSpeechLM] CTC alignment failed: %s", error)
            return []
    if prepared is None:
        # Retry one by one, so that a single bad transcript, or a batch too large for
        # device memory, does not cost the rest their timestamps. Not from inside the
        # handler: its traceback keeps the failed batch's device tensors alive.
        return [batch for item in chunk for batch in _prepare_chunk(aligner, [item], device)]
    return [([index for index, _, _ in chunk], prepared)]


def _align_batch(prepared: dict) -> list[dict | None]:
    """Align one prepared batch; ``None`` for a request the aligner rejects."""
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import align_prepared_batch

    try:
        return align_prepared_batch(prepared)
    except Exception as error:  # noqa: BLE001
        if len(prepared["records"]) > 1:
            # Retry one by one, so that a single unalignable transcript does not cost the
            # rest their timestamps.
            return [_align_batch({**prepared, "records": [record]})[0] for record in prepared["records"]]
        if not isinstance(error, ValueError):
            raise
        logging.warning("[NeMoSpeechLM] CTC alignment failed: %s", error)
        return [None]


class _AlignmentBatcher:
    """Aligns the prepared batches of concurrent requests together, one search at a time.

    A server prepares each request on its own, but the alignment search pays its per-frame
    loop once per batch, so a one-record search costs several times more per record than a
    merged one. The first batch starts a search at once; the batches that arrive while it
    runs on a worker thread queue up, and the next search takes all of them, up to
    ``_SERVER_ALIGN_RECORDS`` records.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[dict, asyncio.Future]] = []
        self._drain_task: asyncio.Task | None = None

    async def align(self, prepared: dict) -> list[dict | None]:
        """Align one prepared batch, merged with whatever else is waiting."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append((prepared, future))
        if self._drain_task is None:
            self._drain_task = loop.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        group: list[tuple[dict, asyncio.Future]] = []
        try:
            while self._pending:
                group = [self._pending.pop(0)]
                records = len(group[0][0]["records"])
                while self._pending and records + len(self._pending[0][0]["records"]) <= _SERVER_ALIGN_RECORDS:
                    records += len(self._pending[0][0]["records"])
                    group.append(self._pending.pop(0))
                try:
                    outcomes = await asyncio.to_thread(_align_group, [prepared for prepared, _ in group])
                except Exception as error:  # noqa: BLE001
                    outcomes = [error] * len(group)
                for (_, future), outcome in zip(group, outcomes):
                    if future.done():
                        continue
                    if isinstance(outcome, BaseException):
                        future.set_exception(outcome)
                    else:
                        future.set_result(outcome)
                group = []
        finally:
            self._drain_task = None
            # A drain cancelled with its event loop must not leave requests waiting forever.
            for _, future in group + self._pending:
                if not future.done():
                    future.cancel()
            self._pending.clear()


def _batcher() -> _AlignmentBatcher:
    """The alignment batcher of the running event loop."""
    loop = asyncio.get_running_loop()
    batcher = _batchers.get(loop)
    if batcher is None:
        batcher = _batchers[loop] = _AlignmentBatcher()
    return batcher


def _align_group(group: list[dict]) -> list[list[dict | None] | BaseException]:
    """Align queued prepared batches in one search; per batch after an error, so it reaches only its request."""
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import merge_prepared_batches

    merged = merge_prepared_batches(group)
    if merged is not None:
        try:
            results = _align_batch(merged)
        except Exception:  # noqa: BLE001
            pass
        else:
            outcomes, start = [], 0
            for prepared in group:
                outcomes.append(results[start : start + len(prepared["records"])])
                start += len(prepared["records"])
            return outcomes
    outcomes = []
    for prepared in group:
        try:
            outcomes.append(_align_batch(prepared))
        except Exception as error:  # noqa: BLE001
            outcomes.append(error)
    return outcomes


def _pack(value: Any) -> Any:
    """Replace every tensor in a worker reply with its dtype, shape and bytes."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            _PACKED_TENSOR: [str(tensor.dtype).removeprefix("torch."), list(tensor.shape)],
            "data": tensor.numpy().tobytes(),
        }
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    return value


def _unpack(value: Any) -> Any:
    """Rebuild the tensors :func:`_pack` replaced."""
    if isinstance(value, dict):
        if _PACKED_TENSOR in value:
            dtype_name, shape = value[_PACKED_TENSOR]
            dtype = getattr(torch, dtype_name)
            if not value["data"]:
                return torch.empty(shape, dtype=dtype)
            return torch.frombuffer(bytearray(value["data"]), dtype=dtype).reshape(shape)
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


def _collate(entries: list[dict], device: torch.device) -> Any:
    """Batch stored single-row inputs, padding each time axis to the longest row."""
    from nemo.collections.asr.modules.parallel_expert_encoder import CTCTimestampInputs

    for entry in entries:
        if entry["ready"] is not None:
            entry["ready"].synchronize()
    fields = {}
    for name, time_axis, _ in _FIELDS:
        tensors = [entry[name] for entry in entries]
        fields[name] = None if any(t is None for t in tensors) else _collate_field(tensors, time_axis, device)
    return CTCTimestampInputs(**fields, diarization_frame_seconds=entries[0]["diarization_frame_seconds"])


def _collate_field(tensors: list[torch.Tensor], time_axis: int | None, device: torch.device) -> torch.Tensor:
    """Stack single-row tensors, zero-padding ``time_axis``; lengths mark the valid frames.

    The batch is assembled in pinned memory when it is bound for the GPU, so the copy
    to the device does not block.
    """
    shape = [len(tensors), *tensors[0].shape[1:]]
    if time_axis is not None:
        shape[time_axis] = max(tensor.shape[time_axis] for tensor in tensors)
    batch = torch.zeros(shape, dtype=tensors[0].dtype, pin_memory=device.type == "cuda")
    for row, tensor in enumerate(tensors):
        target = batch[row : row + 1]
        if time_axis is not None:
            target = target.narrow(time_axis, 0, tensor.shape[time_axis])
        target.copy_(tensor)
    return batch.to(device=device, non_blocking=True)


def _public_result(result: dict) -> dict:
    """Plain, serializable view of one aligner result (see :func:`align_finished_requests`).

    Times are rounded to milliseconds; further digits are float noise from frame
    arithmetic.
    """
    words = [
        {
            "word": word["word"],
            "start": _seconds(word["start"]),
            "end": _seconds(word["end"]),
            "speaker": str(word["speaker"]),
        }
        for speaker_words in (result.get("speaker_word_timestamps") or {}).values()
        for word in speaker_words
    ]
    words.sort(key=lambda w: (w["start"], w["end"]))
    mapping = {
        str(tag): None if column is None else int(column)
        for tag, column in (result.get("speaker_tag_to_sortformer_column") or {}).items()
    }
    return {
        "words": words,
        "diarization": _public_diarization(result.get("diarization_timestamps") or ()),
        "speaker_tag_to_diarization_speaker": mapping,
        "error": None,
    }


def _public_diarization(segments: Sequence[dict]) -> list[dict]:
    """Diarization segments with the Sortformer speaker index, and times rounded to milliseconds."""
    return [
        {"speaker": int(segment["speaker"]), "start": _seconds(segment["start"]), "end": _seconds(segment["end"])}
        for segment in segments
    ]


def _seconds(value: Any) -> float:
    return round(float(value), 3)


def _new_request_hashes(scheduler_output: Any):
    """Yield ``(req_id, mm_hash)`` for each multimodal item of newly scheduled requests."""
    for new_req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
        features = getattr(new_req, "mm_features", None)
        if features is not None:
            hashes = [getattr(feature, "identifier", None) for feature in features]
        else:
            hashes = list(getattr(new_req, "mm_hashes", None) or ())
        for mm_hash in hashes:
            if mm_hash:
                yield new_req.req_id, mm_hash


def _holds_captures() -> bool:
    """Whether this worker stores captures and prepares them for alignment.

    Every tensor-parallel rank of the first pipeline stage runs the encoder on the same
    audio, so only its rank 0 keeps the inputs and prepares them; ``collective_rpc`` lists that
    worker's reply first. Outside an initialized engine, as in unit tests, every caller does.
    """
    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    except ImportError:  # pragma: no cover - vLLM absent
        return True
    try:
        return get_tp_group().rank_in_group == 0 and get_pp_group().is_first_rank
    except (AssertionError, AttributeError):
        return True


def _worker_prepare_requests(worker: Any, items: list, release: bool = True, require_enabled: bool = True) -> dict:
    """``collective_rpc`` entry point for both modes: prepare finished requests where their inputs live.

    ``collective_rpc`` calls every worker; the ones that hold no captures answer with no
    batches, which callers skip.

    Args:
        worker (Any): The vLLM worker the method is attached to; unused.
        items (list): ``(request_id, transcript)`` pairs of finished requests; a server
            sends one.
        release (bool): Afterwards drop each request's claim on its capture.
        require_enabled (bool): Raise when the model was loaded without timestamps. A
            server, which asks for every timestamped request, passes ``False`` to get
            empty results instead.

    Returns:
        dict: The reply of :func:`prepare_finished_requests`, with its tensors packed so
        that ``collective_rpc`` returns them intact.
    """
    del worker
    if not _holds_captures():
        return {"count": len(items), "batches": []}
    if require_enabled and active_aligner() is None:
        raise RuntimeError(
            "CTC timestamps are not enabled for this model: its config has no ctc_timestamps.adapter_path "
            "(check the hf_overrides key)."
        )
    return _pack(prepare_finished_requests([(request_id, text) for request_id, text in items], release=release))


def _worker_release_requests(worker: Any, request_ids: list) -> None:
    """``collective_rpc`` entry point: release captures that will not be aligned."""
    del worker
    _release_requests(list(request_ids))


def install_worker_methods() -> None:
    """Expose preparing and releasing to callers outside the engine as worker methods.

    ``collective_rpc`` resolves a method name on the worker, so the entry points are
    attached to vLLM's GPU worker class under ``WORKER_PREPARE_METHOD`` and
    ``WORKER_RELEASE_METHOD``.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:  # pragma: no cover - vLLM absent
        return
    for name, method in (
        (WORKER_PREPARE_METHOD, _worker_prepare_requests),
        (WORKER_RELEASE_METHOD, _worker_release_requests),
    ):
        if getattr(Worker, name, None) is not method:
            setattr(Worker, name, method)


def align(
    rpc: Any,
    items: Sequence[tuple[str, str]],
    release: bool = True,
    require_enabled: bool = True,
    chunk_size: int = 256,
) -> list[dict]:
    """Align finished requests through an engine's ``collective_rpc``, one RPC per ``chunk_size`` items.

    The client side shared by both modes; :func:`align_async` is the same for an async
    engine client. Each RPC runs the deferred CTC head on the engine thread, which stalls
    requests decoding meanwhile for that long; the alignment search then runs here.

    Args:
        rpc (Any): The engine's ``collective_rpc``, e.g. ``LLM.collective_rpc``.
        items (Sequence[tuple[str, str]]): ``(request_id, transcript)`` pairs, with the
            external request ids the caller sees.
        release (bool): Drop each request's claim on its capture after aligning it.
        require_enabled (bool): Raise when the model was loaded without timestamps,
            instead of returning empty results.
        chunk_size (int): Items aligned per RPC.

    Returns:
        list[dict]: Per item, the result described in :func:`align_finished_requests`.
    """
    results: list[dict] = []
    for batch in _rpc_batches(items, chunk_size):
        results.extend(_align_reply(rpc(WORKER_PREPARE_METHOD, args=(batch, release, require_enabled))))
    return results


async def align_async(
    rpc: Any,
    items: Sequence[tuple[str, str]],
    release: bool = True,
    require_enabled: bool = True,
    chunk_size: int = 256,
) -> list[dict]:
    """:func:`align` for an async engine client, e.g. a server's ``EngineClient.collective_rpc``.

    The alignment search runs on a worker thread, so the event loop keeps serving other
    requests, and is merged with the searches of the other requests aligned meanwhile
    (``_AlignmentBatcher``), so a busy server pays its per-frame cost once per batch.
    """
    batcher = _batcher()
    results: list[dict] = []
    for batch in _rpc_batches(items, chunk_size):
        replies = await rpc(WORKER_PREPARE_METHOD, args=(batch, release, require_enabled))
        reply = _unpack(replies[0])
        aligned = await asyncio.gather(*(batcher.align(entry["prepared"]) for entry in reply["batches"]))
        results.extend(_results_from(reply, aligned))
    return results


def release_captures(rpc: Any, request_ids: Sequence[str]) -> None:
    """Release requests that will not be aligned, through ``collective_rpc``.

    Their captures are deleted once no other request or vLLM's encoder cache needs them.

    Args:
        rpc (Any): The engine's ``collective_rpc``, e.g. ``LLM.collective_rpc``.
        request_ids (Sequence[str]): External request ids.
    """
    rpc(WORKER_RELEASE_METHOD, args=(list(request_ids),))


async def release_captures_async(rpc: Any, request_ids: Sequence[str]) -> None:
    """:func:`release_captures` for an async engine client."""
    await rpc(WORKER_RELEASE_METHOD, args=(list(request_ids),))


def _rpc_batches(items: Sequence[tuple[str, str]], chunk_size: int) -> list[list[tuple[str, str]]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1; got {chunk_size}.")
    return [list(items[start : start + chunk_size]) for start in range(0, len(items), chunk_size)]


def _align_reply(replies: list) -> list[dict]:
    """Align the preparing worker's reply; ``collective_rpc`` returns one reply per worker, in rank order."""
    return align_prepared_requests(_unpack(replies[0]))


def ctc_timestamps(
    llm: Any,
    outputs: Sequence[Any],
    texts: Sequence[str] | None = None,
    chunk_size: int = 256,
    release: bool = True,
) -> list[dict]:
    """CTC word timestamps and diarization for finished outputs of an offline vLLM ``LLM``.

    The engine runs the deferred CTC head where the stored inputs live, with one RPC per
    ``chunk_size`` outputs, and the alignment search then runs in this process. Call it
    after generation: the RPC runs on the engine thread, so preparing while other
    requests decode would stall them.

    Each output's capture is deleted once it has been aligned with ``release=True``.
    Pass ``release=False`` to align the same outputs again, e.g. with ``texts``, and
    release outputs that will never be aligned with :func:`ctc_release`. Until then their
    captures count against ``NEMO_CTC_TIMESTAMP_RETAIN_GB`` (default 8); once they fill it,
    new outputs get no timestamps.

    Args:
        llm (Any): The ``vllm.LLM`` that produced ``outputs``, with CTC timestamps enabled.
        outputs (Sequence[Any]): Its ``RequestOutput`` objects, one audio item each,
            generated with ``mm_processor_kwargs={"capture_ctc_timestamps": True}``.
            Decode speaker-tagged prompts with ``skip_special_tokens=False`` so
            ``<spk:N>`` reaches the aligner.
        texts (Sequence[str] | None): Transcripts to align instead of the generated
            ones, one per output, e.g. a reference or corrected transcript.
        chunk_size (int): Outputs aligned per RPC.
        release (bool): Delete the outputs' captures after aligning them.

    Returns:
        list[dict]: Per output, the result described in :func:`align_finished_requests`.

    Raises:
        RuntimeError: When ``llm`` was loaded without CTC timestamps, e.g. because the
            ``ctc_timestamps`` key in ``hf_overrides`` is misspelled.
    """
    if texts is not None and len(texts) != len(outputs):
        raise ValueError(f"Got {len(texts)} texts for {len(outputs)} outputs.")
    items = [
        (out.request_id, out.outputs[0].text if texts is None else texts[index]) for index, out in enumerate(outputs)
    ]
    return align(llm.collective_rpc, items, release=release, chunk_size=chunk_size)


def ctc_word_timestamps(
    llm: Any, outputs: Sequence[Any], chunk_size: int = 256, release: bool = True
) -> list[list[dict]]:
    """Just the word timestamps of :func:`ctc_timestamps`, one list per output."""
    return [result["words"] for result in ctc_timestamps(llm, outputs, chunk_size=chunk_size, release=release)]


def ctc_release(llm: Any, outputs: Sequence[Any]) -> None:
    """Release outputs that will not be aligned.

    Their captures are deleted once no other request or vLLM's encoder cache needs them.

    Args:
        llm (Any): The ``vllm.LLM`` that produced ``outputs``.
        outputs (Sequence[Any]): Its ``RequestOutput`` objects.
    """
    release_captures(llm.collective_rpc, [out.request_id for out in outputs])


def install_encoder_cache_binding() -> None:
    """Key captured inputs by ``mm_hash`` and record the request-to-hash map.

    The hook passes straight through while no encoder is registered.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:  # pragma: no cover - vLLM absent
        return

    original = GPUModelRunner._execute_mm_encoder
    if getattr(original, "_nemo_ctc_timestamp_bound", False):
        return

    def _execute_mm_encoder(self, scheduler_output, *args, **kwargs):
        # Other tensor-parallel ranks run the same encoder forward but take no hashes,
        # so they store nothing.
        if active_aligner() is None or not _holds_captures():
            return original(self, scheduler_output, *args, **kwargs)

        # New requests resolve to their hashes here, including those whose audio is
        # an encoder-cache hit and therefore never reaches the batch below; without
        # this, repeated audio would find no inputs.
        _record_new_requests(scheduler_output)
        # vLLM evicting audio only ends cache hits on its capture; owners that have
        # not been aligned yet keep it.
        _follow_engine_cache(freed=getattr(scheduler_output, "free_encoder_mm_hashes", ()))
        # Audio encoded again replaces a capture an earlier request may still own, so only
        # captures new to the store can be refused.
        with _lock:
            retained = set(_store)

        # The forward receives only tensors, so it takes its items' hashes from here, in
        # the order vLLM encodes them.
        mm_hashes, _, item_refs = self._batch_mm_inputs_from_scheduler(scheduler_output)
        _begin_step(mm_hashes)
        try:
            outputs = original(self, scheduler_output, *args, **kwargs)
        finally:
            _end_step()

        _follow_engine_cache(encoded=mm_hashes)
        _record_request_hashes((req_id, mm_hash) for mm_hash, (req_id, _position) in zip(mm_hashes, item_refs))
        _trim_store([mm_hash for mm_hash in mm_hashes if mm_hash not in retained])
        _compact_ready()
        return outputs

    _execute_mm_encoder._nemo_ctc_timestamp_bound = True
    GPUModelRunner._execute_mm_encoder = _execute_mm_encoder
    logging.info("[NeMoSpeechLM] CTC timestamp inputs bound to vLLM encoder-cache identity.")
