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

"""The vLLM plugin's per-request store for deferred CTC timestamp inputs, on CPU."""

import asyncio
import contextlib
import json
import sys
import threading
import weakref
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo.collections.asr.modules.parallel_expert_encoder import CTCTimestampInputs
from nemo.collections.speechlm2.vllm.salm import ctc_timestamps as ct


class _FakeAligner:
    """Records collated inputs; its prepared batches align to one word per token of each transcript."""

    def __init__(self):
        self.ctc_decoder = nn.Linear(1, 1)
        self.calls = []

    def prepare_from_inputs(self, timestamp_inputs, sot_transcripts, audio_durations):
        self.calls.append((timestamp_inputs, list(sot_transcripts), list(audio_durations)))
        if any("crash" in text for text in sot_transcripts):
            raise RuntimeError("kernel failure")
        if any("untokenizable" in text for text in sot_transcripts):
            raise ValueError("tokenizer disagreement")
        labels, lengths = timestamp_inputs.diarization_labels, timestamp_inputs.diarization_lengths
        return {
            "config": {},
            "num_speaker_columns": 2,
            "diarization_frame_seconds": 0.01,
            "diarization_max_speaker_count": 4,
            "records": [
                {
                    "text": text,
                    "ctc_log_probs": torch.zeros(1, 2),
                    "diarization_labels": None if labels is None else labels[row, :, : int(lengths[row])],
                    "audio_duration": duration,
                    "time_offset": 0.0,
                }
                for row, (text, duration) in enumerate(zip(sot_transcripts, audio_durations))
            ],
        }


def _align_fake_batch(prepared):
    """Stands in for align_prepared_batch on a _FakeAligner batch, which arrives with its tensors intact."""
    records = prepared["records"]
    assert all(isinstance(record["ctc_log_probs"], torch.Tensor) for record in records)
    if any("unalignable" in record["text"] for record in records):
        raise ValueError("tokenizer disagreement")
    if any("explode" in record["text"] for record in records):
        raise RuntimeError("search failure")
    return [
        {
            "speaker_word_timestamps": {
                0: [
                    {"word": word, "speaker": 0, "start": 0.1 * (index + 1), "end": 0.1 * (index + 1) + 0.08}
                    for index, word in enumerate(record["text"].split())
                ]
            },
            "diarization_timestamps": [{"speaker": 1, "start": 0.05, "end": 0.93}],
            "speaker_tag_to_sortformer_column": {0: 1},
        }
        for record in records
    ]


def _words(result):
    return [w["word"] for w in result["words"]]


def _inputs(batch, frames):
    return CTCTimestampInputs(
        asr_encoded=torch.randn(batch, 4, frames),
        asr_encoded_lengths=torch.full((batch,), frames),
        sortformer_sigmoids=torch.rand(batch, frames, 2),
        sortformer_lengths=torch.full((batch,), frames),
        diarization_labels=torch.zeros(batch, 4, frames * 8, dtype=torch.bool),
        diarization_lengths=torch.full((batch,), frames * 8),
    )


def _capture(batch, frames, hashes, durations=None, inputs=None):
    """Store one forward's states under its items' hashes, as during a runner-hook step."""
    ct._begin_step(hashes)
    ct.store_alignment_states(
        ct.take_step_hashes(batch),
        _inputs(batch, frames) if inputs is None else inputs,
        [1.0] * batch if durations is None else durations,
    )
    ct._end_step()


def _align_one(request_id, text, *, release):
    return ct.align_finished_requests([(request_id, text)], release=release)[0]


@pytest.fixture
def aligner(monkeypatch):
    monkeypatch.setattr(ct, "_store", {})
    monkeypatch.setattr(ct, "_request_hashes", {})
    monkeypatch.setattr(ct, "_request_items", {})
    monkeypatch.setattr(ct, "_hash_owners", {})
    monkeypatch.setattr(ct, "_engine_cached", set())
    monkeypatch.setattr(ct, "_external_ids", {})
    monkeypatch.setattr(ct, "_registry", {})
    monkeypatch.setattr(ct, "_uncompacted", deque())
    monkeypatch.setattr(ct, "_state", SimpleNamespace())
    monkeypatch.setattr("nemo.collections.speechlm2.parts.ctc_timestamp_utils.align_prepared_batch", _align_fake_batch)
    fake = _FakeAligner()
    ct.register_aligner(fake)
    return fake


def test_external_request_id_resolves_and_rows_from_two_forwards_are_padded(aligner):
    _capture(1, 5, ["hash-a"], durations=[0.4])
    _capture(1, 3, ["hash-b"], durations=[0.24])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a"), ("req-b-89abcdef", "hash-b")])

    results = ct.align_finished_requests([("req-a", "one two"), ("req-b", "three")], release=True)

    assert [_words(result) for result in results] == [["one", "two"], ["three"]]
    inputs, texts, durations = aligner.calls[0]
    assert texts == ["one two", "three"] and durations == [0.4, 0.24]
    assert inputs.asr_encoded.shape == (2, 4, 5)
    assert inputs.asr_encoded_lengths.tolist() == [5, 3]
    assert inputs.sortformer_sigmoids.shape == (2, 5, 2)
    assert inputs.diarization_labels.shape == (2, 4, 40)
    assert torch.count_nonzero(inputs.asr_encoded[1, :, 3:]) == 0


def test_cache_hit_request_and_repeated_alignment_find_the_same_inputs(aligner):
    _capture(1, 4, ["hash-a"])
    first = SimpleNamespace(req_id="req-1", mm_features=[SimpleNamespace(identifier="hash-a")])
    repeat = SimpleNamespace(req_id="req-2", mm_features=[SimpleNamespace(identifier="hash-a")])
    ct._record_request_hashes(ct._new_request_hashes(SimpleNamespace(scheduled_new_reqs=[first, repeat])))

    kept = _align_one("req-1", "a b", release=False)
    assert kept == _align_one("req-1", "a b", release=True)
    assert _words(_align_one("req-2", "c", release=True)) == ["c"]


def test_forwards_outside_an_encoder_step_get_no_hashes(aligner):
    # vLLM's startup profiling pass runs the encoder outside the runner hook.
    assert ct.take_step_hashes(1) is None
    _capture(1, 4, ["hash-a"], durations=[0.32])
    assert ct.take_step_hashes(1) is None
    ct._record_request_hashes([("req", "hash-a")])

    _align_one("req", "word", release=True)

    assert aligner.calls[0][0].asr_encoded.shape[-1] == 4
    assert aligner.calls[0][2] == [0.32]


def test_a_shared_capture_is_deleted_once_its_last_owner_is_aligned(aligner, caplog):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req-1", "hash-a"), ("req-2", "hash-a")])

    assert _words(_align_one("req-1", "a", release=True)) == ["a"]
    assert "hash-a" in ct._store and _align_one("req-1", "a", release=True) == ct._empty_result("no_capture")
    assert "already aligned or released" in caplog.text

    assert _words(_align_one("req-2", "b", release=True)) == ["b"]
    assert ct._store == {} and ct._hash_owners == {} and ct._request_hashes == {}


def test_an_unowned_capture_lives_while_vllm_caches_its_audio(aligner):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._follow_engine_cache(encoded=["hash-a", "hash-b"])
    ct._record_request_hashes([("req-1", "hash-a"), ("req-2", "hash-b")])

    _align_one("req-1", "a", release=True)
    # A later request with the same audio is an encoder-cache hit and captures nothing.
    ct._record_request_hashes([("req-3", "hash-a")])
    assert _words(_align_one("req-3", "c", release=True)) == ["c"]
    assert "hash-a" in ct._store

    ct._follow_engine_cache(freed=["hash-a", "hash-b"])
    assert list(ct._store) == ["hash-b"]
    _align_one("req-2", "b", release=True)
    assert ct._store == {}


def test_one_unalignable_transcript_does_not_cost_the_batch(aligner):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b"), ("req-c", "hash-c")])

    results = ct.align_finished_requests([("req-a", "x"), ("req-b", "unalignable"), ("req-c", "y z")], release=True)

    assert [_words(result) for result in results] == [["x"], [], ["y", "z"]]
    assert [result["error"] for result in results] == [None, "alignment_failed", None]


def test_diarization_segments_and_speaker_mapping_pass_through(aligner):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    result = _align_one("req", "<spk:0> hi", release=True)

    assert result["diarization"] == [{"speaker": 1, "start": 0.05, "end": 0.93}]
    assert result["speaker_tag_to_diarization_speaker"] == {"0": 1}
    assert result["words"][0]["speaker"] == "0"


def test_release_false_keeps_captures_until_the_worker_releases_them(aligner):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a"), ("req-b-0123abcd", "hash-b")])

    ct.align_finished_requests([("req-a", "x"), ("req-b", "y")], release=False)
    assert set(ct._store) == {"hash-a", "hash-b"}

    ct._worker_release_requests(None, ["req-a", "req-b", "never-seen"])
    assert ct._store == {} and ct._request_hashes == {} and ct._external_ids == {}


def test_ranks_that_hold_no_captures_still_encode_but_store_nothing(aligner, monkeypatch):
    class StubRunner:
        def _batch_mm_inputs_from_scheduler(self, step):
            return step.hashes, None, [(req_id, None) for req_id in step.req_ids]

        def _execute_mm_encoder(self, step):
            # What _encode_with_ctc_capture does: the forward always runs, and states
            # are stored only under the step's hashes.
            count = len(step.hashes)
            hashes = ct.take_step_hashes(count)
            if hashes is not None:
                ct.store_alignment_states(hashes, _inputs(count, 4), [1.0] * count)
            return "encoded"

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", SimpleNamespace(GPUModelRunner=StubRunner))
    ct.install_encoder_cache_binding()
    monkeypatch.setattr(ct, "_holds_captures", lambda: False)
    step = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req-a-0123abcd", mm_features=[SimpleNamespace(identifier="hash-a")])
        ],
        hashes=["hash-a"],
        req_ids=["req-a-0123abcd"],
        free_encoder_mm_hashes=[],
    )

    assert StubRunner()._execute_mm_encoder(step) == "encoded"

    assert ct._store == {} and ct._request_hashes == {}
    assert ct._worker_prepare_requests(None, [("req-a", "x")]) == {"count": 1, "batches": []}
    assert aligner.calls == []


def test_offline_api_aligns_repeatedly_then_releases(aligner):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([(f"req-{name}-0123abcd", f"hash-{name}") for name in "abc"])
    methods = {
        ct.WORKER_PREPARE_METHOD: ct._worker_prepare_requests,
        ct.WORKER_RELEASE_METHOD: ct._worker_release_requests,
    }
    llm = SimpleNamespace(collective_rpc=lambda method, args: [methods[method](None, *args)])
    outputs = [SimpleNamespace(request_id=f"req-{name}", outputs=[SimpleNamespace(text=name)]) for name in "ab"]

    generated = ct.ctc_word_timestamps(llm, outputs, release=False)
    reference = ct.ctc_timestamps(llm, outputs, texts=["ref a", "ref b"])

    assert [[word["word"] for word in words] for words in generated] == [["a"], ["b"]]
    assert [_words(result) for result in reference] == [["ref", "a"], ["ref", "b"]]
    assert list(ct._store) == ["hash-c"]

    ct.ctc_release(llm, [SimpleNamespace(request_id="req-c")])
    assert ct._store == {}


def test_sync_and_async_clients_send_the_same_rpcs_and_get_the_same_results(aligner):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([(f"req-{name}-0123abcd", f"hash-{name}") for name in "abc"])
    methods = {
        ct.WORKER_PREPARE_METHOD: ct._worker_prepare_requests,
        ct.WORKER_RELEASE_METHOD: ct._worker_release_requests,
    }
    sent = []

    def rpc(method, args):
        sent.append((method, args))
        reply = methods[method](None, *args)
        # Every rank replies; only the first one prepares.
        other_rank = None if reply is None else {"count": reply["count"], "batches": []}
        return [reply, other_rank]

    async def rpc_async(method, args):
        return rpc(method, args)

    items = [("req-a", "a"), ("req-b", "b c"), ("req-c", "d")]
    sync = ct.align(rpc, items, release=False, chunk_size=2)
    sync_sent = sent[:]
    sent.clear()
    via_async = asyncio.run(ct.align_async(rpc_async, items, release=False, chunk_size=2))

    assert via_async == sync
    assert [_words(result) for result in sync] == [["a"], ["b", "c"], ["d"]]
    assert sent == sync_sent and [len(args[0]) for _, args in sent] == [2, 1]

    asyncio.run(ct.release_captures_async(rpc_async, ["req-a", "req-b"]))
    assert list(ct._store) == ["hash-c"]
    ct.release_captures(rpc, ["req-c"])
    assert ct._store == {}

    with pytest.raises(ValueError, match="chunk_size"):
        ct.align(rpc, items, chunk_size=0)


def test_async_alignment_runs_off_the_event_loop_thread(aligner, monkeypatch):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a")])
    threads = []

    def align_and_record_thread(prepared):
        threads.append(threading.current_thread())
        return _align_fake_batch(prepared)

    monkeypatch.setattr(
        "nemo.collections.speechlm2.parts.ctc_timestamp_utils.align_prepared_batch", align_and_record_thread
    )

    async def rpc(method, args):
        return [ct._worker_prepare_requests(None, *args)]

    (result,) = asyncio.run(ct.align_async(rpc, [("req-a", "a b")]))

    assert _words(result) == ["a", "b"]
    assert threads and threads[0] is not threading.main_thread()


def _serve_concurrently(names):
    """Align one request per name, all at once on one event loop, as a server's middleware does."""

    async def rpc(method, args):
        return [ct._worker_prepare_requests(None, *args)]

    async def serve():
        requests = (ct.align_async(rpc, [(f"req-{name}", name)]) for name in names)
        # Bounded, so that a request left waiting fails the test instead of hanging it.
        return await asyncio.wait_for(asyncio.gather(*requests, return_exceptions=True), timeout=30)

    return asyncio.run(serve())


def _record_searches(monkeypatch):
    searches = []

    def align_and_record(prepared):
        searches.append([record["text"] for record in prepared["records"]])
        return _align_fake_batch(prepared)

    monkeypatch.setattr("nemo.collections.speechlm2.parts.ctc_timestamp_utils.align_prepared_batch", align_and_record)
    return searches


def test_concurrent_server_alignments_share_one_search(aligner, monkeypatch):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([(f"req-{name}-0123abcd", f"hash-{name}") for name in "abc"])
    searches = _record_searches(monkeypatch)

    results = _serve_concurrently("abc")

    assert [_words(result) for (result,) in results] == [["a"], ["b"], ["c"]]
    assert searches == [["a", "b", "c"]]


def test_server_searches_take_at_most_the_record_cap(aligner, monkeypatch):
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._record_request_hashes([(f"req-{name}-0123abcd", f"hash-{name}") for name in "abc"])
    searches = _record_searches(monkeypatch)
    monkeypatch.setattr(ct, "_SERVER_ALIGN_RECORDS", 2)

    results = _serve_concurrently("abc")

    assert [_words(result) for (result,) in results] == [["a"], ["b"], ["c"]]
    assert searches == [["a", "b"], ["c"]]


def test_an_error_in_a_shared_search_reaches_only_its_request(aligner):
    _capture(2, 4, ["hash-fine", "hash-explode"])
    ct._record_request_hashes([("req-fine-0123abcd", "hash-fine"), ("req-explode-0123abcd", "hash-explode")])

    fine, exploded = _serve_concurrently(["fine", "explode"])

    assert _words(fine[0]) == ["fine"]
    assert isinstance(exploded, RuntimeError) and str(exploded) == "search failure"


def test_an_unexpected_batcher_error_fails_the_waiting_requests_instead_of_hanging_them(aligner, monkeypatch):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a"), ("req-b-0123abcd", "hash-b")])

    def broken(group):
        raise KeyError("broken batch")

    monkeypatch.setattr(ct, "_align_group", broken)

    results = _serve_concurrently("ab")

    assert all(isinstance(result, KeyError) for result in results)


def test_a_prepared_reply_survives_vllms_rpc_serialization(aligner):
    pytest.importorskip("vllm")
    from vllm.v1.engine import UtilityOutput
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req-a-0123abcd", "hash-a")])
    reply = ct._worker_prepare_requests(None, [("req-a", "a b")])

    # How a collective_rpc result travels from the engine process to the client.
    sent = MsgpackEncoder().encode(UtilityOutput(call_id=1, result=UtilityResult(reply)))
    received = MsgpackDecoder(UtilityOutput).decode(sent).result.result

    assert _words(ct._align_reply([received])[0]) == ["a", "b"]


def test_packed_worker_replies_carry_bytes_and_rebuild_their_tensors():
    reply = {
        "values": torch.arange(6.0).reshape(2, 3),
        "flags": torch.tensor([[True, False]]),
        "nested": [torch.zeros(0, 3)],
        "text": "x",
        "missing": None,
    }

    packed = ct._pack(reply)
    unpacked = ct._unpack(packed)

    assert isinstance(packed["values"]["data"], bytes)
    assert torch.equal(unpacked["values"], reply["values"])
    assert torch.equal(unpacked["flags"], reply["flags"]) and unpacked["flags"].dtype == torch.bool
    assert unpacked["nested"][0].shape == (0, 3)
    assert unpacked["text"] == "x" and unpacked["missing"] is None


def test_request_with_several_audio_items_gets_no_timestamps(aligner):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("two-clips", "hash-a"), ("two-clips", "hash-b"), ("one-clip", "hash-b")])

    results = ct.align_finished_requests([("two-clips", "one two"), ("one-clip", "three")], release=True)

    assert results[0] == ct._empty_result("multiple_audio")
    assert _words(results[1]) == ["three"]
    assert [texts for _, texts, _ in aligner.calls] == [["three"]]


def test_a_request_with_the_same_clip_twice_gets_no_timestamps(aligner):
    _capture(1, 4, ["hash-a"])
    clip = SimpleNamespace(identifier="hash-a")
    twice = SimpleNamespace(req_id="twice", mm_features=[clip, clip])
    once = SimpleNamespace(req_id="once", mm_features=[clip])
    ct._record_new_requests(SimpleNamespace(scheduled_new_reqs=[twice, once]))

    results = ct.align_finished_requests([("twice", "one two"), ("once", "three")], release=True)

    assert results[0] == ct._empty_result("multiple_audio")
    assert _words(results[1]) == ["three"]
    assert ct._request_items == {}


def test_a_step_whose_rows_do_not_match_its_hashes_keeps_no_captures(aligner, caplog):
    ct._begin_step(["hash-a", "hash-b"])
    ct.store_alignment_states(ct.take_step_hashes(1), _inputs(1, 4), [1.0])
    ct._end_step()
    assert ct._store == {} and "dropping this step's CTC timestamp captures" in caplog.text

    ct._begin_step(["hash-c"])
    assert ct.take_step_hashes(2) is None
    ct._end_step()

    _capture(1, 3, ["hash-d"])
    ct._record_request_hashes([("req", "hash-d")])
    assert _words(_align_one("req", "ok", release=True)) == ["ok"]


def test_compaction_trims_each_row_to_its_valid_frames_in_its_own_storage(aligner):
    inputs = CTCTimestampInputs(
        asr_encoded=torch.randn(2, 4, 5),
        asr_encoded_lengths=torch.tensor([5, 3]),
        sortformer_sigmoids=torch.rand(2, 5, 2),
        sortformer_lengths=torch.tensor([5, 3]),
        diarization_labels=torch.ones(2, 4, 40, dtype=torch.bool),
        diarization_lengths=torch.tensor([40, 24]),
    )
    _capture(2, 5, ["hash-a", "hash-b"], [0.4, 0.24], inputs=inputs)

    assert ct._compact_ready() == 2
    short = ct._store["hash-b"]
    assert short["asr_encoded"].shape == (1, 4, 3)
    assert short["sortformer_sigmoids"].shape == (1, 3, 2)
    assert short["diarization_labels"].shape == (1, 4, 24)
    asr = short["asr_encoded"]
    assert asr.untyped_storage().nbytes() == asr.numel() * asr.element_size()
    assert torch.equal(asr[0], inputs.asr_encoded[1, :, :3])

    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])
    ct.align_finished_requests([("req-a", "one"), ("req-b", "two")], release=True)
    collated = aligner.calls[0][0]
    assert collated.asr_encoded.shape == (2, 4, 5)
    assert collated.diarization_labels[0].all()
    assert torch.count_nonzero(collated.diarization_labels[1, :, 24:]) == 0


def test_evicted_rows_are_not_compacted(aligner, monkeypatch):
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", "0")
    _capture(2, 4, ["hash-a", "hash-b"])

    ct._trim_store()

    assert ct._compact_ready() == 1
    assert list(ct._store) == ["hash-b"]


def test_speaker_prior_weight_defaults_and_rejects_negative_values():
    assert ct.read_speaker_prior_weight({}) == 0.25
    assert ct.read_speaker_prior_weight(SimpleNamespace(speaker_logprob_weight=0)) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        ct.read_speaker_prior_weight({"speaker_logprob_weight": -0.1})


def test_special_tokens_come_from_the_model_tokenizer(monkeypatch):
    vllm_tokenizers = pytest.importorskip("vllm.tokenizers")
    tokenizer = SimpleNamespace(all_special_tokens=["<s>", "<spk:0>"])
    monkeypatch.setattr(vllm_tokenizers, "cached_tokenizer_from_config", lambda model_config: tokenizer)

    assert ct.tokenizer_special_tokens(SimpleNamespace(skip_tokenizer_init=False)) == ("<s>", "<spk:0>")
    assert ct.tokenizer_special_tokens(SimpleNamespace(skip_tokenizer_init=True)) == ()
    assert ct.tokenizer_special_tokens(None) == ()


def test_model_runner_v2_is_refused():
    ct.require_v1_model_runner(SimpleNamespace())
    ct.require_v1_model_runner(SimpleNamespace(use_v2_model_runner=False))
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        ct.require_v1_model_runner(SimpleNamespace(use_v2_model_runner=True))


def test_byte_budget_evicts_unowned_captures_oldest_first_but_keeps_the_newest(aligner, monkeypatch, caplog):
    monkeypatch.setattr(ct.logging, "once_logged", set())
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"])
    ct._compact_ready()
    row_bytes = ct._store["hash-a"]["nbytes"]

    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", str(2.5 * row_bytes / 1e9))
    ct._trim_store()
    assert list(ct._store) == ["hash-b", "hash-c"]

    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", "0")
    ct._trim_store()
    assert list(ct._store) == ["hash-c"]
    # A server evicts routinely, so this is said once rather than per step.
    assert caplog.text.count("Evicting CTC timestamp captures") == 1


def test_owned_captures_are_kept_and_new_ones_refused_when_they_fill_the_budget(aligner, monkeypatch, caplog):
    monkeypatch.setattr(ct.logging, "once_logged", set())
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req-a", "hash-a")])
    ct._compact_ready()
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", str(1.5 * ct._store["hash-a"]["nbytes"] / 1e9))

    # The next step encodes req-b's audio while req-a still waits for alignment.
    _capture(1, 4, ["hash-b"])
    ct._record_request_hashes([("req-b", "hash-b")])
    ct._trim_store(["hash-b"])

    assert list(ct._store) == ["hash-a"] and "new captures are refused" in caplog.text
    results = ct.align_finished_requests([("req-a", "kept"), ("req-b", "refused")], release=True)
    assert _words(results[0]) == ["kept"] and results[0]["error"] is None
    assert results[1] == ct._empty_result("capture_unavailable")


def test_captures_no_request_owns_are_evicted_before_new_ones_are_refused(aligner, monkeypatch):
    _capture(2, 4, ["hash-old", "hash-a"])
    ct._follow_engine_cache(encoded=["hash-old"])
    ct._record_request_hashes([("req-old", "hash-old"), ("req-a", "hash-a")])
    _align_one("req-old", "done", release=True)
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", str(2.5 * ct._store["hash-a"]["nbytes"] / 1e9))

    _capture(1, 4, ["hash-b"])
    ct._record_request_hashes([("req-b", "hash-b")])
    ct._trim_store(["hash-b"])

    # vLLM still caches hash-old's audio, but no request needs its capture.
    assert list(ct._store) == ["hash-a", "hash-b"]


def test_an_empty_transcript_has_no_words_and_no_error(aligner):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    assert _align_one("req", " ", release=True) == ct._empty_result()
    assert aligner.calls == []


def test_requests_whose_words_are_not_aligned_still_get_their_diarization(aligner):
    labels = torch.zeros(3, 4, 32, dtype=torch.bool)
    labels[:, 1, 5:15] = True
    inputs = CTCTimestampInputs(
        asr_encoded=torch.randn(3, 4, 4),
        asr_encoded_lengths=torch.full((3,), 4),
        sortformer_sigmoids=torch.rand(3, 4, 2),
        sortformer_lengths=torch.full((3,), 4),
        diarization_labels=labels,
        diarization_lengths=torch.full((3,), 32),
    )
    _capture(3, 4, ["hash-a", "hash-b", "hash-c"], inputs=inputs)
    ct._record_request_hashes([("empty", "hash-a"), ("untokenizable", "hash-b"), ("unalignable", "hash-c")])
    finished = [("empty", " "), ("untokenizable", "untokenizable"), ("unalignable", "unalignable")]

    results = ct.align_finished_requests(finished, release=True)

    # Speaker 1 is active from 10 ms label frame 5 to 15, whatever happened to the words:
    # nothing to align, rejected while preparing, and rejected by the search.
    segment = [{"speaker": 1, "start": 0.05, "end": 0.15}]
    assert results == [
        ct._empty_result(None, segment),
        ct._empty_result("alignment_failed", segment),
        ct._empty_result("alignment_failed", segment),
    ]


def test_reencoded_audio_replaces_its_entry_as_the_most_recent(aligner):
    _capture(1, 4, ["hash-a"])
    first = ct._store["hash-a"]
    _capture(1, 4, ["hash-b"])
    _capture(1, 3, ["hash-a"])

    assert list(ct._store) == ["hash-b", "hash-a"]
    assert first["dropped"] and ct._store["hash-a"]["asr_encoded"].shape[-1] == 3


def test_deferred_head_batches_stay_within_the_frame_budget(aligner, monkeypatch):
    monkeypatch.setattr(ct, "_HEAD_FRAME_BUDGET", 12)
    _capture(1, 5, ["hash-a"])
    _capture(1, 5, ["hash-b"])
    _capture(1, 3, ["hash-c"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b"), ("req-c", "hash-c")])
    finished = [("req-a", "a"), ("req-b", "b"), ("req-c", "c")]

    ct.align_finished_requests(finished, release=False)
    # Two requests padded to five frames take ten; a third would take fifteen.
    assert [texts for _, texts, _ in aligner.calls] == [["a", "b"], ["c"]]

    aligner.calls.clear()
    aligner.online_inference_length, aligner.chunk_left_context, aligner.chunk_right_context = 2, 1, 1
    ct.align_finished_requests(finished, release=True)
    # Decoded in windows of four frames, all three take twelve.
    assert [texts for _, texts, _ in aligner.calls] == [["a", "b", "c"]]


def test_a_failed_batch_is_freed_before_its_requests_are_retried_one_by_one(aligner, monkeypatch):
    failed_inputs = []

    def prepare(timestamp_inputs, sot_transcripts, audio_durations):
        if len(sot_transcripts) > 1:
            failed_inputs.append(weakref.ref(timestamp_inputs.asr_encoded))
            raise RuntimeError("CUDA out of memory")
        assert all(ref() is None for ref in failed_inputs)
        return _FakeAligner.prepare_from_inputs(aligner, timestamp_inputs, sot_transcripts, audio_durations)

    monkeypatch.setattr(aligner, "prepare_from_inputs", prepare)
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])

    results = ct.align_finished_requests([("req-a", "x"), ("req-b", "y")], release=True)

    assert [_words(result) for result in results] == [["x"], ["y"]] and len(failed_inputs) == 1


@pytest.mark.parametrize("release", [True, False])
def test_unexpected_alignment_errors_are_raised_not_hidden(aligner, release):
    _capture(2, 4, ["hash-a", "hash-b"])
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b"), ("req-other", "hash-a")])

    with pytest.raises(RuntimeError, match="kernel failure"):
        ct.align_finished_requests([("req-a", "fine"), ("req-b", "crash")], release=release)

    if release:
        # The failed call still drops its claims; req-other keeps the capture it shares.
        assert set(ct._request_hashes) == {"req-other"} and ct._hash_owners == {"hash-a": {"req-other"}}
        assert set(ct._store) == {"hash-a"}
    else:
        # Without release the claims stay, so the caller can retry.
        assert set(ct._request_hashes) == {"req-a", "req-b", "req-other"}
        assert set(ct._store) == {"hash-a", "hash-b"}


def test_tensor_durations_survive_compaction(aligner):
    _capture(2, 4, ["hash-a", "hash-b"], torch.tensor([0.5, 0.25], dtype=torch.float64))
    ct._compact_ready()
    ct._record_request_hashes([("req-a", "hash-a"), ("req-b", "hash-b")])

    ct.align_finished_requests([("req-a", "x"), ("req-b", "y")], release=False)

    assert aligner.calls[0][2] == [0.5, 0.25]
    assert ct._store["hash-b"]["duration"] == 0.25


def test_external_ids_resolve_through_an_index_that_follows_forgotten_requests(aligner, monkeypatch):
    monkeypatch.setattr(ct, "_MAX_TRACKED_REQUESTS", 4)

    ct._record_request_hashes([(f"req{i}-0123abcd", f"hash-{i}") for i in range(5)])

    assert ct.mm_hashes_for_request("req0") == []
    assert ct.mm_hashes_for_request("req4") == ct.mm_hashes_for_request("req4-0123abcd") == ["hash-4"]
    assert ct._external_ids == {f"req{i}": f"req{i}-0123abcd" for i in range(1, 5)}


def test_only_the_first_tensor_parallel_rank_prepares(aligner, monkeypatch):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    monkeypatch.setattr(ct, "_holds_captures", lambda: False)
    assert ct._worker_prepare_requests(None, [("req", "x")], False) == {"count": 1, "batches": []}
    assert aligner.calls == []

    monkeypatch.setattr(ct, "_holds_captures", lambda: True)
    assert _words(ct._align_reply([ct._worker_prepare_requests(None, [("req", "x")])])[0]) == ["x"]


def test_output_times_are_rounded_to_milliseconds(aligner):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    words = _align_one("req", "a b c", release=True)["words"]

    assert [(w["start"], w["end"]) for w in words] == [(0.1, 0.18), (0.2, 0.28), (0.3, 0.38)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_collate_assembles_padded_rows_on_the_gpu(aligner):
    _capture(1, 5, ["hash-a"])
    _capture(1, 3, ["hash-b"])

    batch = ct._collate([ct._store["hash-a"], ct._store["hash-b"]], torch.device("cuda"))

    assert batch.asr_encoded.is_cuda and batch.asr_encoded.shape == (2, 4, 5)
    assert torch.equal(batch.asr_encoded[1, :, :3].cpu(), ct._store["hash-b"]["asr_encoded"][0])
    assert torch.count_nonzero(batch.asr_encoded[1, :, 3:]) == 0


def _hooked_runner(monkeypatch):
    """A stand-in for vLLM's GPU model runner with the capture hook installed, and a builder for its steps."""

    class StubRunner:
        def _batch_mm_inputs_from_scheduler(self, step):
            return step.hashes, None, [(req_id, None) for req_id in step.req_ids]

        def _execute_mm_encoder(self, step):
            # What _process_audio does for the items vLLM encodes this step.
            if step.hashes:
                count = len(step.hashes)
                ct.store_alignment_states(ct.take_step_hashes(count), _inputs(count, 4), [1.0] * count)
            return "encoded"

    def step(new_requests, encoded, freed=()):
        return SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id=req_id, mm_features=[SimpleNamespace(identifier=mm_hash)])
                for req_id, mm_hash in new_requests
            ],
            hashes=[mm_hash for _, mm_hash in encoded],
            req_ids=[req_id for req_id, _ in encoded],
            free_encoder_mm_hashes=list(freed),
        )

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", SimpleNamespace(GPUModelRunner=StubRunner))
    ct.install_encoder_cache_binding()
    return StubRunner(), step


def test_runner_hook_keys_captures_by_hash_maps_cache_hits_and_ignores_outside_forwards(aligner, monkeypatch):
    runner, step = _hooked_runner(monkeypatch)

    first = step([("req-a-0123abcd", "hash-a")], [("req-a-0123abcd", "hash-a")])
    assert runner._execute_mm_encoder(first) == "encoded"
    # The same audio again is an encoder-cache hit: scheduled, but never encoded.
    runner._execute_mm_encoder(step([("req-b-0123abcd", "hash-a")], []))
    # A forward outside the hook, like vLLM's startup profiling pass on dummy audio, gets no hashes.
    assert ct.take_step_hashes(1) is None
    runner._execute_mm_encoder(step([("req-c-0123abcd", "hash-c")], [("req-c-0123abcd", "hash-c")]))

    assert set(ct._store) == {"hash-a", "hash-c"} and not ct._uncompacted
    results = ct.align_finished_requests([("req-a", "a"), ("req-b", "b"), ("req-c", "c")], release=True)
    assert [_words(result) for result in results] == [["a"], ["b"], ["c"]]
    assert aligner.calls[0][2] == [1.0, 1.0, 1.0]

    # Released, but kept until vLLM evicts the audio from its encoder cache.
    assert set(ct._store) == {"hash-a", "hash-c"}
    runner._execute_mm_encoder(step([], [], freed=["hash-a"]))
    assert list(ct._store) == ["hash-c"]


def test_audio_encoded_again_keeps_the_capture_an_earlier_request_owns(aligner, monkeypatch):
    runner, step = _hooked_runner(monkeypatch)
    both = [("req-a-0123abcd", "hash-a"), ("req-b-0123abcd", "hash-b")]
    runner._execute_mm_encoder(step(both, both))
    monkeypatch.setenv("NEMO_CTC_TIMESTAMP_RETAIN_GB", str(2.5 * ct._store["hash-a"]["nbytes"] / 1e9))

    # vLLM evicts hash-a's audio while req-a waits for alignment, then encodes it again for
    # req-a2 in the same step as req-c's new audio.
    again = [("req-c-0123abcd", "hash-c"), ("req-a2-0123abcd", "hash-a")]
    runner._execute_mm_encoder(step(again, again, freed=["hash-a"]))

    assert set(ct._store) == {"hash-a", "hash-b"}
    results = ct.align_finished_requests([("req-a", "a"), ("req-b", "b"), ("req-c", "c")], release=True)
    assert [result["error"] for result in results] == [None, None, "capture_unavailable"]


def test_offline_alignment_refuses_a_model_without_timestamps(monkeypatch):
    monkeypatch.setattr(ct, "_registry", {})

    with pytest.raises(RuntimeError, match="not enabled"):
        ct._worker_prepare_requests(None, [("req", "<spk:0> hi")])
    reply = ct._worker_prepare_requests(None, [("req", "<spk:0> hi")], True, False)
    assert ct._align_reply([reply]) == [ct._empty_result("not_enabled")]


def test_adapter_on_an_encoder_without_timestamp_support_is_refused():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

    model = object.__new__(NeMoSpeechLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.perception = SimpleNamespace(encoder=SimpleNamespace())
    model._uses_pe_encoder = False
    model.encoder_chunk_size_seconds = None

    model._maybe_enable_ctc_timestamps(None, SimpleNamespace())
    with pytest.raises(ValueError, match="cannot produce CTC timestamp inputs"):
        model._maybe_enable_ctc_timestamps({"adapter_path": "/adapter.pt"}, SimpleNamespace())


def test_timestamps_refuse_encoder_chunking_outside_the_parallel_expert_encoder():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

    model = object.__new__(NeMoSpeechLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.perception = SimpleNamespace(encoder=SimpleNamespace(supports_ctc_timestamp_inputs=True))
    model._uses_pe_encoder = False
    model.encoder_chunk_size_seconds = 30.0

    with pytest.raises(ValueError, match="unchunked"):
        model._maybe_enable_ctc_timestamps({"adapter_path": "/adapter.pt"}, SimpleNamespace())


def test_perception_built_for_another_sample_rate_is_refused():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.model import _require_resampling_rate

    def perception(sample_rate):
        return SimpleNamespace(preprocessor=SimpleNamespace(featurizer=SimpleNamespace(sample_rate=sample_rate)))

    _require_resampling_rate(SimpleNamespace())
    _require_resampling_rate(perception(16000))
    with pytest.raises(ValueError, match="8000 Hz"):
        _require_resampling_rate(perception(8000))


class _FakePerception(nn.Module):
    """A perception module whose encoder is not a ParallelExpertEncoder; records each forward."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.forwards = []

    def forward(self, input_signal, input_signal_length, return_ctc_timestamp_inputs=False):
        self.forwards.append((tuple(input_signal.shape), return_ctc_timestamp_inputs))
        batch = input_signal.shape[0]
        outputs = (torch.ones(batch, 3, 4), torch.full((batch,), 3))
        return (*outputs, _inputs(batch, 4)) if return_ctc_timestamp_inputs else outputs


def _model_with_fake_perception():
    from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

    model = object.__new__(NeMoSpeechLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.perception = _FakePerception()
    model._uses_pe_encoder = False
    model.encoder_chunk_size_seconds = None
    return model


def _audio(capture=None):
    capture = None if capture is None else torch.tensor(capture)
    return SimpleNamespace(
        audio_signal=torch.ones(2, 16), audio_signal_length=torch.tensor([16, 8]), capture_ctc_timestamps=capture
    )


def test_any_encoder_with_the_flag_captures_through_one_unchunked_forward(aligner):
    pytest.importorskip("vllm")
    model = _model_with_fake_perception()

    ct._begin_step(["hash-a", "hash-b"])
    embeddings = model._process_audio(_audio([True, True]))
    ct._end_step()

    assert [tuple(e.shape) for e in embeddings] == [(3, 4), (3, 4)]
    assert model.perception.forwards == [((2, 16), True)]
    assert list(ct._store) == ["hash-a", "hash-b"]


def test_a_forward_outside_the_hook_produces_states_but_stores_none(aligner):
    pytest.importorskip("vllm")
    model = _model_with_fake_perception()

    model._process_audio(_audio([True, True]))

    # Memory is profiled as served, but there is no hash to store anything under.
    assert model.perception.forwards == [((2, 16), True)]
    assert ct._store == {}


def test_items_that_opt_out_take_their_hash_but_store_nothing(aligner):
    pytest.importorskip("vllm")
    model = _model_with_fake_perception()

    ct._begin_step(["hash-a", "hash-b", "hash-c", "hash-d", "hash-e", "hash-f"])
    model._process_audio(_audio([False, True]))
    model._process_audio(_audio([False, False]))
    # Audio that arrives without a flag counts as not opted in.
    model._process_audio(_audio())
    ct._end_step()

    assert [capture for _, capture in model.perception.forwards] == [True, False, False]
    # Rows take the step's hashes by position, so an opted-out row still takes one.
    assert list(ct._store) == ["hash-b"] and float(ct._store["hash-b"]["duration"]) == 8 / 16000


def test_processor_marks_the_audio_of_requests_that_opt_into_capture():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

    class _Tokenizer:
        def get_vocab(self):
            return {"<|audio|>": 0}

        def encode(self, prompt, add_special_tokens=True):
            return [0] * len(prompt.split())

    processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_tokenizer=_Tokenizer,
        _estimate_audio_tokens=lambda samples, chunk_size_seconds=None, estimator_config=None: 2,
        _get_encoder_chunk_size_seconds=lambda: None,
        _get_audio_token_estimator_config=lambda: None,
    )

    def capture_flags(mm_kwargs):
        result = processor._call_hf_processor(
            prompt="<|audio|> <|audio|>",
            mm_data={"audios": [[0.0] * 8, [0.0] * 4]},
            mm_kwargs=mm_kwargs,
            tok_kwargs={},
        )
        return result["capture_ctc_timestamps"].tolist()

    assert capture_flags({}) == [False, False]
    assert capture_flags({"capture_ctc_timestamps": True}) == [True, True]
    # Read on the host while encoding, where a device copy would need a sync.
    assert processor._get_mm_fields_config(None, {})["capture_ctc_timestamps"].field.keep_on_cpu


def test_startup_profiling_inputs_opt_into_capture(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder

    from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMDummyInputsBuilder

    monkeypatch.setattr(
        BaseDummyInputsBuilder,
        "get_dummy_processor_inputs",
        lambda self, seq_len, mm_counts, mm_options: SimpleNamespace(hf_processor_mm_kwargs={}),
    )
    builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)

    inputs = builder.get_dummy_processor_inputs(64, {"audio": 1}, {})

    assert inputs.hf_processor_mm_kwargs == {"capture_ctc_timestamps": True}


_WORKER_METHODS = {
    ct.WORKER_PREPARE_METHOD: ct._worker_prepare_requests,
    ct.WORKER_RELEASE_METHOD: ct._worker_release_requests,
}


@pytest.fixture
def server(aligner):
    """The chat middleware in front of a stand-in for vLLM's chat route, over a stub engine and the real store."""
    pytest.importorskip("vllm")
    import uuid

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.testclient import TestClient

    from nemo.collections.speechlm2.vllm.salm.ctc_serving import ctc_timestamp_middleware

    def capture(request_id):
        _capture(1, 4, [f"hash-{request_id}"])
        ct._record_request_hashes([(f"{request_id}-0123abcd", f"hash-{request_id}")])

    script = {
        "transcript": "hi there",
        "status": 200,
        "id_prefix": "chatcmpl-",
        "capture": capture,
        "cancelled": False,
        "stream_fragment_bytes": None,
        "finish_reason": "stop",
        "stream_error": None,
    }
    rpcs = []

    async def collective_rpc(method, args):
        rpcs.append(method)
        return [_WORKER_METHODS[method](None, *args)]

    hf_config = SimpleNamespace(ctc_timestamps={"adapter_path": "/adapter.pt"})
    engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config), collective_rpc=collective_rpc)
    app = FastAPI()
    app.state.engine_client = engine

    @app.post("/v1/chat/completions")
    async def vllm_chat_route(request: Request):  # stands in for vLLM's own chat route
        body = await request.json()
        base = request.headers.get("x-request-id") or body.get("request_id") or uuid.uuid4().hex
        request_id = f"{script['id_prefix']}{base}"
        if (body.get("mm_processor_kwargs") or {}).get("capture_ctc_timestamps"):
            # The capture the engine keeps for an opted-in request while it generates.
            script["capture"](request_id)
        if script["cancelled"]:
            return None  # what vLLM's route returns once its client disconnects
        if body.get("stream") and script["status"] == 200:

            async def stream():
                base_chunk = {"id": request_id, "object": "chat.completion.chunk", "created": 1, "model": "hr9a"}
                chunks = [
                    {"delta": {"role": "assistant", "content": ""}, "finish_reason": None},
                    {"delta": {"content": script["transcript"]}, "finish_reason": script["finish_reason"]},
                ]
                data = b""
                for choice in chunks:
                    chunk = {**base_chunk, "choices": [{"index": 0, "logprobs": None, **choice}]}
                    data += _sse(chunk)
                if script["stream_error"]:
                    data += _sse({"error": script["stream_error"]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    data += _sse(
                        {
                            **base_chunk,
                            "choices": [],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                        }
                    )
                data += b"data: [DONE]\n\n"
                step = script["stream_fragment_bytes"] or len(data)
                for start in range(0, len(data), step):
                    yield data[start : start + step]

            return StreamingResponse(stream(), media_type="text/event-stream", headers={"x-upstream": "preserved"})
        message = {"role": "assistant", "content": script["transcript"]}
        content = {"id": request_id, "object": "chat.completion", "choices": [{"index": 0, "message": message}]}
        return JSONResponse(content, status_code=script["status"])

    app.middleware("http")(ctc_timestamp_middleware)
    client = TestClient(app)

    def post(opt_in=True, path="/v1/chat/completions", headers=None, **fields):
        body = {"model": "hr9a", "messages": [{"role": "user", "content": "Transcribe."}], **fields}
        if opt_in:
            body["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
        rpcs.clear()
        return client.post(path, json=body, headers=headers or {})

    return SimpleNamespace(post=post, script=script, rpcs=rpcs, hf_config=hf_config, app=app)


def test_an_opted_in_chat_completion_gets_ctc_timestamps_and_releases_its_capture(server):
    response = server.post()

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-ctc-") and body["choices"][0]["message"]["content"] == "hi there"
    timestamps = body["ctc_timestamps"]
    assert [(w["word"], w["start"], w["end"], w["speaker"]) for w in timestamps["words"]] == [
        ("hi", 0.1, 0.18, "0"),
        ("there", 0.2, 0.28, "0"),
    ]
    assert timestamps["diarization"] == [{"speaker": 1, "start": 0.05, "end": 0.93}]
    assert timestamps["error"] is None
    assert server.rpcs == [ct.WORKER_PREPARE_METHOD] and ct._store == {}


def test_chat_requests_without_the_opt_in_pass_through_untouched(server):
    flag_off = {"capture_ctc_timestamps": False}
    assert "ctc_timestamps" not in server.post(opt_in=False).json()
    assert "ctc_timestamps" not in server.post(opt_in=False, mm_processor_kwargs=flag_off).json()
    assert server.post(path="/v1/completions").status_code == 404 and server.rpcs == []
    server.hf_config.ctc_timestamps = None
    assert "ctc_timestamps" not in server.post().json()
    assert server.rpcs == []


def test_the_client_request_id_names_the_engine_request(server):
    response = server.post(headers={"X-Request-Id": "client-1"})
    assert response.json()["id"] == "chatcmpl-client-1" and response.json()["ctc_timestamps"]["words"]

    response = server.post(request_id="body-1")
    assert response.json()["id"] == "chatcmpl-body-1" and response.json()["ctc_timestamps"]["words"]
    assert ct._store == {}


def test_alignment_uses_the_request_id_vllm_reports(server, caplog):
    server.script["id_prefix"] = "chat-"  # a vLLM whose request id scheme changed

    response = server.post()

    assert response.json()["id"].startswith("chat-ctc-") and response.json()["ctc_timestamps"]["words"]
    assert "not the predicted" in caplog.text
    assert server.rpcs == [ct.WORKER_PREPARE_METHOD] and ct._store == {}


def test_opted_in_requests_refuse_n_above_1(server):
    for fields in ({"n": 2}, {"stream": True, "n": 2}):
        response = server.post(**fields)

        assert response.status_code == 400
        assert "n > 1" in response.json()["error"]["message"]
    assert server.rpcs == []


def _sse(chunk):
    return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()


def _stream_chunks(response):
    events = response.content.decode().split("\n\n")
    assert events[-2:] == ["data: [DONE]", ""]
    return [json.loads(event.removeprefix("data: ")) for event in events[:-2]]


@pytest.mark.parametrize("fragment_bytes", [1, 7, 65536])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_streaming_timestamps_follow_the_full_transcript_and_preserve_usage(server, fragment_bytes, finish_reason):
    server.script.update(
        transcript="<spk:0> héllo 世界", stream_fragment_bytes=fragment_bytes, finish_reason=finish_reason
    )
    expected = server.post().json()["ctc_timestamps"]

    response = server.post(stream=True, stream_options={"include_usage": True})

    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-upstream"] == "preserved"
    chunks = _stream_chunks(response)
    assert (
        "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"])
        == server.script["transcript"]
    )
    terminal = [chunk for chunk in chunks if chunk["choices"] and chunk["choices"][0]["finish_reason"]]
    assert len(terminal) == 1 and terminal[0]["choices"][0]["finish_reason"] == finish_reason
    assert terminal[0]["choices"][0]["delta"] == {} and terminal[0]["ctc_timestamps"] == expected
    assert all("ctc_timestamps" not in chunk for chunk in chunks if chunk is not terminal[0])
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 5
    assert server.rpcs == [ct.WORKER_PREPARE_METHOD] and ct._store == {}


@pytest.mark.parametrize(
    "transcript, error", [("", None), ("untokenizable", "alignment_failed"), ("crash", "alignment_failed")]
)
def test_stream_alignment_failure_keeps_the_transcript_and_finishes(server, transcript, error):
    server.script["transcript"] = transcript

    chunks = _stream_chunks(server.post(stream=True))

    terminal = next(chunk for chunk in chunks if "ctc_timestamps" in chunk)
    assert terminal["ctc_timestamps"]["words"] == [] and terminal["ctc_timestamps"]["error"] == error
    assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == transcript
    assert ct._store == {} and ct._request_hashes == {}


def test_an_upstream_stream_error_passes_through_without_aligning(server):
    server.script["stream_error"] = {"message": "engine failed", "code": 500}

    chunks = _stream_chunks(server.post(stream=True))

    assert chunks[-1] == {"error": server.script["stream_error"]}
    assert not any("ctc_timestamps" in chunk for chunk in chunks)
    assert server.rpcs == [ct.WORKER_RELEASE_METHOD] and ct._store == {}


def test_a_stream_without_the_opt_in_is_unchanged(server):
    chunks = _stream_chunks(server.post(opt_in=False, stream=True))

    assert len(chunks) == 2 and chunks[-1]["choices"][0]["delta"]["content"] == "hi there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and server.rpcs == []


def test_streaming_uses_the_capture_on_a_later_data_parallel_core(dp_server):
    dp_server.script["owners"] = [2]
    expected = dp_server.post().json()["ctc_timestamps"]
    dp_server.engine.rpcs.clear()

    chunks = _stream_chunks(dp_server.post(stream=True))

    terminal = next(chunk for chunk in chunks if "ctc_timestamps" in chunk)
    assert terminal["ctc_timestamps"] == expected and dp_server.engine.stores() == [{}, {}, {}]
    assert dp_server.engine.rpcs == [(ct.WORKER_PREPARE_METHOD, core) for core in range(3)]


def _stream_for(server, body, request_id="chatcmpl-stream"):
    from nemo.collections.speechlm2.vllm.salm.ctc_serving import _stream_with_timestamps

    server.script["capture"](request_id)
    return _stream_with_timestamps(body, server.app.state.engine_client, request_id)


def _stream_choice(content, finish_reason=None, **fields):
    return {
        "id": "chatcmpl-stream",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason, **fields}],
    }


def test_streaming_does_not_delay_text_or_duplicate_final_logprobs_and_token_ids(server, monkeypatch):
    from nemo.collections.speechlm2.vllm.salm import ctc_serving as serving

    original = serving.align_async
    logprobs = {"content": [{"token": "there", "logprob": -0.1, "bytes": [116], "top_logprobs": []}]}

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        async def delayed(*args, **kwargs):
            started.set()
            await finish.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(serving, "align_async", delayed)

        async def body():
            yield _sse(_stream_choice("hi "))
            yield _sse(_stream_choice("there", "stop", logprobs=logprobs, token_ids=[42], stop_reason=99))
            yield b"data: [DONE]\n\n"

        async with contextlib.aclosing(_stream_for(server, body())) as stream:
            first, last_text = await anext(stream), await anext(stream)
            assert not started.is_set() and server.rpcs == []
            assert json.loads(first[6:])["choices"][0]["delta"]["content"] == "hi "
            partial = json.loads(last_text[6:])["choices"][0]
            assert partial["delta"]["content"] == "there" and partial["finish_reason"] is None
            assert partial["logprobs"] == logprobs and partial["token_ids"] == [42] and partial["stop_reason"] is None
            waiting = asyncio.create_task(anext(stream))
            await asyncio.wait_for(started.wait(), 2)
            assert not waiting.done()
            finish.set()
            terminal = json.loads((await asyncio.wait_for(waiting, 2))[6:])
            choice = terminal["choices"][0]
            assert choice["delta"] == {} and choice["logprobs"] is None and choice["token_ids"] == []
            assert choice["finish_reason"] == "stop" and choice["stop_reason"] == 99
            assert _words(terminal["ctc_timestamps"]) == ["hi", "there"]
            assert await anext(stream) == b"data: [DONE]\n\n"
        assert ct._store == {}

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["generation", "alignment"])
@pytest.mark.parametrize("cancel_scope", [False, True])
def test_a_disconnected_stream_releases_captures_and_closes_upstream(server, monkeypatch, phase, cancel_scope):
    import anyio

    from nemo.collections.speechlm2.vllm.salm import ctc_serving as serving

    async def run():
        started, closed = asyncio.Event(), asyncio.Event()

        async def blocking_align(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(serving, "align_async", blocking_align)

        async def body():
            try:
                yield _sse(_stream_choice("hi"))
                if phase == "generation":
                    started.set()
                    await asyncio.Event().wait()
                yield _sse(_stream_choice("", "stop"))
                yield b"data: [DONE]\n\n"
            finally:
                closed.set()

        async def consume():
            async for _ in _stream_for(server, body()):
                pass

        if cancel_scope:
            async with anyio.create_task_group() as group:
                group.start_soon(consume)
                await asyncio.wait_for(started.wait(), 2)
                group.cancel_scope.cancel()
        else:
            task = asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert closed.is_set() and ct._store == {} and ct._request_hashes == {}
        assert server.rpcs == [ct.WORKER_RELEASE_METHOD]

    asyncio.run(run())


def test_closing_a_stream_between_deltas_releases_its_capture(server):
    async def body():
        yield _sse(_stream_choice("hi"))
        yield _sse(_stream_choice(" there", "stop"))
        yield b"data: [DONE]\n\n"

    async def run():
        stream = _stream_for(server, body())
        await anext(stream)
        await stream.aclose()
        assert ct._store == {} and server.rpcs == [ct.WORKER_RELEASE_METHOD]

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["headers", "text", "alignment"])
def test_an_asgi_send_failure_releases_the_capture_immediately(server, monkeypatch, phase):
    from starlette.requests import ClientDisconnect
    from starlette.responses import StreamingResponse

    from nemo.collections.speechlm2.vllm.salm import ctc_serving as serving

    async def run():
        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
        server.script["capture"]("chatcmpl-stream")
        upstream_closed = False

        async def body():
            nonlocal upstream_closed
            try:
                yield _sse(_stream_choice("hi"))
                yield _sse(_stream_choice("", "stop"))
                yield b"data: [DONE]\n\n"
            finally:
                upstream_closed = True

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if phase == "headers" or (phase == "text" and message["type"] == "http.response.body"):
                raise OSError("client socket closed")

        async def cancelled_align(*args, **kwargs):
            raise asyncio.CancelledError

        if phase == "alignment":
            monkeypatch.setattr(serving, "align_async", cancelled_align)
        response = serving._TimestampStreamingResponse(
            StreamingResponse(body(), media_type="text/event-stream"),
            server.app.state.engine_client,
            "chatcmpl-stream",
        )
        with pytest.raises(asyncio.CancelledError if phase == "alignment" else ClientDisconnect):
            await response(scope, receive, send)
        assert ct._store == {} and ct._request_hashes == {} and server.rpcs == [ct.WORKER_RELEASE_METHOD]
        # An unstarted async generator has no body to finalize.
        assert upstream_closed or phase == "headers"

    asyncio.run(run())


@pytest.mark.parametrize("tail", [b"data: {broken}\n\n", b"data: {", b"", RuntimeError("upstream failed")])
def test_a_malformed_or_interrupted_stream_reports_an_error_and_releases(server, tail):
    async def body():
        yield _sse(_stream_choice("hi"))
        if isinstance(tail, Exception):
            raise tail
        yield tail

    async def run():
        events = [event async for event in _stream_for(server, body())]
        assert events[-1] == b"data: [DONE]\n\n" and "error" in json.loads(events[-2][6:])
        assert not any(b'"ctc_timestamps"' in event for event in events)
        assert ct._store == {} and server.rpcs == [ct.WORKER_RELEASE_METHOD]

    asyncio.run(run())


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_sse_comments_multiline_data_and_utf8_survive_byte_fragmentation(server, newline):
    comment = b": keepalive" + newline * 2
    event = _sse(_stream_choice("世界", "stop")).replace(b'"choices":', b'\n data: "choices":')
    event = event.replace(b"\n data:", b"\ndata:").replace(b"\n", newline)
    done = b"data: [DONE]" + newline * 2

    async def body():
        for value in comment + event + done:
            yield bytes([value])

    async def run():
        events = [event async for event in _stream_for(server, body())]
        assert events[0] == comment and events[-1] == done
        assert _words(json.loads(events[-2][6:])["ctc_timestamps"]) == ["世界"]
        assert ct._store == {}

    asyncio.run(run())


def test_concurrent_streams_align_their_own_transcripts_and_release_all_captures(server):
    async def one(index):
        request_id = f"chatcmpl-{index}"

        async def body():
            chunk = _stream_choice(f"word{index}", "stop")
            chunk["id"] = request_id
            yield _sse(chunk)
            await asyncio.sleep(0)
            yield b"data: [DONE]\n\n"

        events = [event async for event in _stream_for(server, body(), request_id)]
        result = json.loads(events[-2][6:])
        assert result["id"] == request_id and _words(result["ctc_timestamps"]) == [f"word{index}"]

    async def run():
        await asyncio.wait_for(asyncio.gather(*(one(index) for index in range(16))), 5)
        assert ct._store == {} and ct._request_hashes == {} and ct._hash_owners == {}

    asyncio.run(run())


def test_a_failed_chat_completion_releases_its_capture(server):
    server.script["status"] = 500

    response = server.post()

    assert response.status_code == 500 and "ctc_timestamps" not in response.json()
    assert server.rpcs == [ct.WORKER_RELEASE_METHOD] and ct._store == {}


def test_an_alignment_failure_releases_the_capture_and_answers_500(server, caplog):
    server.script["transcript"] = "crash"

    response = server.post()

    assert response.status_code == 500 and response.json()["error"]["message"] == "kernel failure"
    assert server.rpcs == [ct.WORKER_PREPARE_METHOD, ct.WORKER_RELEASE_METHOD] and ct._store == {}
    # The log keeps the traceback, not only the message.
    assert any(record.exc_info and str(record.exc_info[1]) == "kernel failure" for record in caplog.records)


def test_a_cancelled_chat_completion_releases_its_capture(server):
    from starlette.requests import Request

    from nemo.collections.speechlm2.vllm.salm.ctc_serving import ctc_timestamp_middleware

    body = b'{"model": "hr9a", "messages": [], "mm_processor_kwargs": {"capture_ctc_timestamps": true}}'

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def call_next(request):
        # The client goes away after the engine captured the request's audio.
        request_id = "chatcmpl-" + dict(request.scope["headers"])[b"x-request-id"].decode()
        _capture(1, 4, ["hash-cancelled"])
        ct._record_request_hashes([(f"{request_id}-0123abcd", "hash-cancelled")])
        raise asyncio.CancelledError

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [], "app": server.app}
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ctc_timestamp_middleware(Request(scope, receive), call_next))

    assert server.rpcs == [ct.WORKER_RELEASE_METHOD] and ct._store == {}


def test_a_completion_vllm_cancelled_for_a_disconnected_client_passes_through_and_releases(server, caplog):
    server.script["cancelled"] = True

    response = server.post()

    assert response.status_code == 200 and response.json() is None
    assert server.rpcs == [ct.WORKER_RELEASE_METHOD] and ct._store == {}
    assert not [record for record in caplog.records if record.levelname == "ERROR"]


def _checkpoint(directory, bundles_ctc_head=True):
    """A ``model.safetensors`` whose metadata marks a bundled CTC timestamp head when asked."""
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import CTC_TIMESTAMP_ARTIFACT_FORMAT

    path = directory / "model.safetensors"
    metadata = {"ctc_timestamp_format": CTC_TIMESTAMP_ARTIFACT_FORMAT} if bundles_ctc_head else {}
    safetensors_torch.save_file({"llm.weight": torch.zeros(1)}, str(path), metadata=metadata)
    return str(path.resolve())


def test_a_checkpoint_that_bundles_the_ctc_head_is_its_own_adapter(tmp_path):
    bundled = _checkpoint(tmp_path)

    def model_config(ctc_timestamps):
        return SimpleNamespace(hf_config=SimpleNamespace(ctc_timestamps=ctc_timestamps), model=str(tmp_path))

    # Found on its own, the head is optional: an engine that cannot use it serves without it.
    assert ct.ctc_timestamp_config(model_config(None)) == {
        "adapter_path": bundled,
        "speaker_logprob_weight": ct.DEFAULT_SPEAKER_PRIOR_WEIGHT,
        "optional": True,
    }
    assert ct.ctc_timestamp_config(model_config({"speaker_logprob_weight": 0.5}))["speaker_logprob_weight"] == 0.5
    assert ct.ctc_timestamp_config(model_config({"enabled": True}))["optional"] is False
    named = {"adapter_path": "/adapter.safetensors"}
    assert ct.ctc_timestamp_config(model_config(named)) is named
    assert ct.ctc_timestamp_config(model_config({"enabled": False})) is None
    assert ct.ctc_timestamp_config(model_config({**named, "enabled": False})) is None


def test_a_checkpoint_without_a_bundled_head_keeps_timestamps_off(tmp_path):
    _checkpoint(tmp_path, bundles_ctc_head=False)
    model_config = SimpleNamespace(hf_config=SimpleNamespace(ctc_timestamps=None), model=str(tmp_path))

    assert ct.ctc_timestamp_config(model_config) is None
    model_config.hf_config.ctc_timestamps = {"enabled": True}
    with pytest.raises(ValueError, match="bundles no CTC timestamp head"):
        ct.ctc_timestamp_config(model_config)


def _sharded_checkpoint(directory, ctc_shards):
    """A sharded checkpoint whose index maps the language model to one shard and the CTC head to ``ctc_shards``."""
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import CTC_TIMESTAMP_ARTIFACT_FORMAT

    weight_map = {"llm.weight": "model-00001-of-00001.safetensors"}
    safetensors_torch.save_file({"llm.weight": torch.zeros(1)}, str(directory / "model-00001-of-00001.safetensors"))
    for index, shard in enumerate(ctc_shards):
        name = f"ctc_timestamp.decoder.weight{index}"
        metadata = {"ctc_timestamp_format": CTC_TIMESTAMP_ARTIFACT_FORMAT}
        safetensors_torch.save_file({name: torch.zeros(1)}, str(directory / shard), metadata=metadata)
        weight_map[name] = shard
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_a_sharded_checkpoint_bundles_its_head_in_one_shard(tmp_path, caplog):
    # As the FP8 and NVFP4 exports of GA-RC1 write it: the whole head in model-ctc-bf16.safetensors.
    _sharded_checkpoint(tmp_path, ["model-ctc-bf16.safetensors"])
    model_config = SimpleNamespace(hf_config=SimpleNamespace(ctc_timestamps=None), model=str(tmp_path))

    adapter = ct.ctc_timestamp_config(model_config)["adapter_path"]
    assert adapter == str((tmp_path / "model-ctc-bf16.safetensors").resolve())

    split = tmp_path / "split"
    split.mkdir()
    _sharded_checkpoint(split, ["model-ctc-a.safetensors", "model-ctc-b.safetensors"])
    model_config.model = str(split)
    assert ct.ctc_timestamp_config(model_config) is None and "spans 2 shards" in caplog.text


def test_a_bundled_head_this_engine_cannot_use_is_skipped_but_requested_timestamps_fail(monkeypatch, caplog):
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

    monkeypatch.setattr(ct, "_registry", {})
    model = object.__new__(NeMoSpeechLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.perception = SimpleNamespace(encoder=SimpleNamespace(supports_ctc_timestamp_inputs=True))
    model._uses_pe_encoder = False
    model.encoder_chunk_size_seconds = None
    model_runner_v2 = SimpleNamespace(use_v2_model_runner=True)

    model._maybe_enable_ctc_timestamps({"adapter_path": "/model.safetensors", "optional": True}, model_runner_v2)
    assert ct.active_aligner() is None and "Serving without CTC timestamps" in caplog.text

    # A named adapter, or a bundled head with enabled: true, asked for timestamps, so startup fails.
    for requested in ({"adapter_path": "/adapter.pt"}, {"adapter_path": "/model.safetensors", "optional": False}):
        with pytest.raises(ValueError, match="Model Runner V2"):
            model._maybe_enable_ctc_timestamps(requested, model_runner_v2)


def test_a_server_told_to_skip_the_bundled_head_passes_requests_through(server, tmp_path):
    _checkpoint(tmp_path)
    server.hf_config.ctc_timestamps = {"enabled": False}
    server.app.state.engine_client.model_config.model = str(tmp_path)
    server.script["capture"] = lambda request_id: None  # an engine without timestamps keeps nothing

    response = server.post()

    assert "ctc_timestamps" not in response.json() and server.rpcs == []


def test_requests_to_an_engine_that_skipped_the_bundled_head_report_not_enabled(server, tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    server.hf_config.ctc_timestamps = None
    server.app.state.engine_client.model_config.model = str(tmp_path)
    # The engine skipped the head, for example on Model Runner V2: it registered no aligner and kept nothing.
    monkeypatch.setattr(ct, "_registry", {})
    server.script["capture"] = lambda request_id: None

    response = server.post()

    assert response.status_code == 200 and response.json()["ctc_timestamps"] == ct._empty_result("not_enabled")


def test_a_server_whose_checkpoint_bundles_the_ctc_head_attaches_timestamps(server, tmp_path):
    _checkpoint(tmp_path)
    server.hf_config.ctc_timestamps = None
    server.app.state.engine_client.model_config.model = str(tmp_path)

    response = server.post()

    assert _words(response.json()["ctc_timestamps"]) == ["hi", "there"]


class _Worker:
    """One vLLM worker process, with an engine-side store of its own; only rank 0 holds captures."""

    def __init__(self, rank):
        self.rank = rank
        self.state = {
            "_store": {},
            "_request_hashes": {},
            "_request_items": {},
            "_hash_owners": {},
            "_engine_cached": set(),
            "_external_ids": {},
            "_uncompacted": deque(),
        }

    @contextlib.contextmanager
    def active(self):
        """Run the block in this worker: the module's store is this worker's meanwhile."""
        outer = {name: getattr(ct, name) for name in [*self.state, "_holds_captures"]}
        for name, value in self.state.items():
            setattr(ct, name, value)
        ct._holds_captures = lambda: self.rank == 0
        try:
            yield
        finally:
            self.state = {name: getattr(ct, name) for name in self.state}
            for name, value in outer.items():
                setattr(ct, name, value)


class _DataParallelEngine:
    """vLLM's AsyncLLM over DPLBAsyncMPClient, its internal data-parallel load balancer, as the middleware sees it."""

    def __init__(self, hf_config, cores=3, tensor_parallel=2):
        self.model_config = SimpleNamespace(hf_config=hf_config)
        self.workers = [[_Worker(rank) for rank in range(tensor_parallel)] for _ in range(cores)]
        self.engine_core = SimpleNamespace(
            core_engines=[core.to_bytes(2, "little") for core in range(cores)], _call_utility_async=self._call_utility
        )
        self.rpcs = []

    async def _call_utility(self, method, *args, engine):
        """One core's utility call: each of its workers runs the collective RPC, replying in rank order."""
        assert method == "collective_rpc"
        worker_method, _, worker_args, _ = args
        core = int.from_bytes(engine, "little")
        self.rpcs.append((worker_method, core))
        replies = []
        for worker in self.workers[core]:
            with worker.active():
                replies.append(_WORKER_METHODS[worker_method](None, *worker_args))
        return replies

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        # As DPLBAsyncMPClient.call_utility_async: every core runs it; the first core's replies return.
        replies = await asyncio.gather(
            *(
                self._call_utility("collective_rpc", method, timeout, args, kwargs, engine=identity)
                for identity in self.engine_core.core_engines
            )
        )
        return replies[0]

    def stores(self):
        return [core[0].state["_store"] for core in self.workers]


@pytest.fixture
def dp_server(server):
    """``server`` over three data-parallel engine cores; the cores in ``script["owners"]`` capture the audio."""

    def connect(cores=3):
        server.engine = _DataParallelEngine(server.hf_config, cores=cores)
        server.app.state.engine_client = server.engine

    def capture(request_id):
        for core in server.script["owners"]:
            # Sortformer marks speaker 1 from 10 ms label frame 5 to 15.
            inputs = _inputs(1, 4)
            inputs.diarization_labels[:, 1, 5:15] = True
            with server.engine.workers[core][0].active():
                _capture(1, 4, [f"hash-{request_id}"], inputs=inputs)
                ct._record_request_hashes([(f"{request_id}-0123abcd", f"hash-{request_id}")])

    server.script.update(capture=capture, owners=[1])
    server.connect = connect
    connect()
    return server


def test_vllm_data_parallel_collective_rpc_returns_only_the_first_cores_replies():
    """The vLLM behavior _DataParallelEngine copies, which the middleware's capture-owner RPC works around."""
    core_client = pytest.importorskip("vllm.v1.engine.core_client")
    client = object.__new__(core_client.DPLBAsyncMPClient)
    client.core_engines = [core.to_bytes(2, "little") for core in range(3)]

    async def call_utility(method, *args, engine):
        return [f"core {int.from_bytes(engine, 'little')}"]

    client._call_utility_async = call_utility

    assert asyncio.run(client.collective_rpc_async(ct.WORKER_PREPARE_METHOD)) == ["core 0"]


def test_a_request_served_by_a_later_data_parallel_core_gets_its_timestamps(dp_server):
    response = dp_server.post()

    timestamps = response.json()["ctc_timestamps"]
    assert response.status_code == 200 and _words(timestamps) == ["hi", "there"] and timestamps["error"] is None
    assert dp_server.engine.rpcs == [(ct.WORKER_PREPARE_METHOD, core) for core in range(3)]
    assert dp_server.engine.stores() == [{}, {}, {}]


@pytest.mark.parametrize("transcript, error", [("", None), ("untokenizable", "alignment_failed")])
def test_a_later_core_whose_words_are_not_aligned_still_answers_with_its_diarization(dp_server, transcript, error):
    dp_server.script.update(transcript=transcript, owners=[2])

    response = dp_server.post()

    segment = [{"speaker": 1, "start": 0.05, "end": 0.15}]
    assert response.status_code == 200 and response.json()["ctc_timestamps"] == ct._empty_result(error, segment)
    assert dp_server.engine.stores() == [{}, {}, {}]


def test_a_request_no_core_captured_reports_no_capture(dp_server):
    dp_server.script["owners"] = []

    response = dp_server.post()

    assert response.status_code == 200 and response.json()["ctc_timestamps"] == ct._empty_result("no_capture")


def test_a_request_two_cores_captured_is_refused_and_released_on_both(dp_server):
    dp_server.script["owners"] = [0, 2]

    response = dp_server.post()

    assert response.status_code == 500 and "2 data-parallel engine cores" in response.json()["error"]["message"]
    assert dp_server.engine.stores() == [{}, {}, {}]


def test_a_single_engine_core_keeps_vllms_collective_rpc(dp_server):
    from nemo.collections.speechlm2.vllm.salm.ctc_serving import _capture_owner_rpc

    dp_server.connect(cores=1)
    dp_server.script["owners"] = [0]

    response = dp_server.post()

    assert _capture_owner_rpc(dp_server.engine) == dp_server.engine.collective_rpc
    assert _words(response.json()["ctc_timestamps"]) == ["hi", "there"] and dp_server.engine.stores() == [{}]


def test_a_failed_chat_completion_releases_the_capture_on_the_core_that_holds_it(dp_server):
    dp_server.script.update(status=500, owners=[2])

    response = dp_server.post()

    assert response.status_code == 500
    assert dp_server.engine.rpcs == [(ct.WORKER_RELEASE_METHOD, core) for core in range(3)]
    assert dp_server.engine.stores() == [{}, {}, {}]


def test_transcripts_without_speaker_tags_are_reported(aligner, caplog):
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    _align_one("req", "hello world", release=True)

    assert "no <spk:N> speaker tags" in caplog.text


def test_prepared_transcripts_lose_special_tokens_but_keep_speaker_tags(aligner):
    ct.register_aligner(aligner, ["<|im_end|>", "<spk:0>"])
    _capture(1, 4, ["hash-a"])
    ct._record_request_hashes([("req", "hash-a")])

    _align_one("req", "<spk:0> hi <|im_end|>", release=True)

    assert aligner.calls[-1][1] == ["<spk:0> hi"]


def test_requests_without_recorded_audio_are_not_reported_as_evicted(aligner, caplog):
    assert ct.align_finished_requests([("text-only", "<spk:0> hello")], release=True) == [
        ct._empty_result("no_capture")
    ]
    assert "No capture is recorded" in caplog.text and "evicted" not in caplog.text
