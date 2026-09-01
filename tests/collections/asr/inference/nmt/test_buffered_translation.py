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

import os
from types import SimpleNamespace

import torch

from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline, TranscribeStepOutput
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.state import StreamingState


class _Translator:
    supports_structured_context_turns = False

    def __init__(self, empty_sources=()):
        self.empty_sources = set(empty_sources)
        self.requests = []

    def translate(self, sources, prefixes, src_langs, tgt_langs, src_contexts, tgt_contexts):
        self.requests.extend(zip(sources, prefixes, src_contexts, tgt_contexts))
        return ["" if source in self.empty_sources else f"<{source}>" for source in sources]

    def get_prefixes(self, sources, translations, previous):
        return [os.path.commonprefix([old, new]) for old, new in zip(previous, translations)]


class _Pipeline(BasePipeline):
    def __init__(self, translator):
        super().__init__()
        self.nmt_model = translator
        self.nmt_enabled = True
        self.mt_source_buffer_enabled = True
        self.chunk_size = 0.48
        self.mt_max_source_units = 256
        self.mt_max_source_duration_ms = 30_000
        self.mt_history_size = 2
        self.mt_history_max_tokens = 1024
        self.mt_max_handoff_deferrals = 1
        self.mt_flush_at_stream_end = True

    def _stable_asr_view(self, state, step_output):
        return step_output.current_step_transcript

    def transcribe_step_for_frames(self, frames):
        raise NotImplementedError

    def transcribe_step_for_feature_buffers(self, fbuffers):
        raise NotImplementedError

    def get_request_generator(self):
        raise NotImplementedError

    def get_sep(self):
        return " "

    def create_state(self, options):
        state = StreamingState()
        state.options = options
        return state


class _NativePipeline(_Pipeline):
    def transcribe_step_for_frames(self, frames):
        for frame in frames:
            state = self.get_state(frame.stream_id)
            state.partial_transcript = "unfinished ending"
            state.current_step_transcript = "unfinished ending"


def _state():
    state = StreamingState()
    state.options = SimpleNamespace(enable_nmt=True, source_language="English", target_language="German")
    return state


def test_acoustic_eou_does_not_reset_mt_source():
    pipeline = _Pipeline(_Translator())
    state = _state()

    first = TranscribeStepOutput(
        stream_id=0, final_transcript="short fragment", current_step_transcript="short fragment"
    )
    pipeline._translate_step_buffered([state], [first])
    second = TranscribeStepOutput(stream_id=0, final_transcript="continues.", current_step_transcript="continues.")
    pipeline._translate_step_buffered([state], [second])

    assert first.final_translation == ""
    assert first.partial_translation == "<short fragment>"
    assert second.final_translation == "<short fragment continues.>"


def test_acoustic_eou_residue_is_translated_in_the_same_update():
    pipeline = _Pipeline(_Translator())
    state = _state()
    output = TranscribeStepOutput(
        stream_id=0,
        final_transcript="Closed.",
        partial_transcript="New temporary",
        current_step_transcript="Closed.",
    )

    pipeline._translate_step_buffered([state], [output])

    assert output.final_translation == "<Closed.>"
    assert output.partial_translation == "<New temporary>"
    assert state.mt_previous_source == "New temporary"


def test_boundary_closes_unit_and_translates_suffix_in_same_update():
    translator = _Translator()
    pipeline = _Pipeline(translator)
    state = _state()
    output = TranscribeStepOutput(
        stream_id=0,
        partial_transcript="First. Next temporary",
        current_step_transcript="First. Next",
    )

    pipeline._translate_step_buffered([state], [output])

    assert output.final_translation == "<First.>"
    assert output.partial_translation == "<Next temporary>"
    assert state.mt_context_history == [("First.", "<First.>")]
    assert state.previous_translation_info == ("<Next temporary>", "")


def test_empty_suffix_handoff_is_deferred_once():
    translator = _Translator(empty_sources={"New", "New words"})
    pipeline = _Pipeline(translator)
    state = _state()
    first = TranscribeStepOutput(
        stream_id=0,
        partial_transcript="Closed. New",
        current_step_transcript="Closed. New",
    )
    pipeline._translate_step_buffered([state], [first])

    assert first.final_translation == ""
    assert first.partial_translation == "<Closed.>"
    assert state.mt_source_buffer.active_source == "Closed. New"
    assert state.mt_handoff_deferrals == 1

    second = TranscribeStepOutput(
        stream_id=0,
        partial_transcript="Closed. New words",
        current_step_transcript="Closed. New words",
    )
    pipeline._translate_step_buffered([state], [second])

    assert second.final_translation == "<Closed.>"
    assert state.mt_handoff_deferrals == 0


def test_empty_closed_unit_handoff_preserves_previous_visible_translation():
    translator = _Translator(empty_sources={"Closed."})
    pipeline = _Pipeline(translator)
    state = _state()
    state.set_translation_info("<Earlier source>", "<Earlier")
    state.mt_previous_source = "Earlier source"
    output = TranscribeStepOutput(
        stream_id=0,
        partial_transcript="Closed. New",
        current_step_transcript="Closed. New",
    )

    pipeline._translate_step_buffered([state], [output])

    assert output.final_translation == ""
    assert output.partial_translation == "<Earlier source>"
    assert state.previous_translation_info == ("<Earlier source>", "<Earlier")
    assert state.mt_previous_source == "Earlier source"


def test_stream_end_regenerates_source_not_covered_by_deferred_translation():
    translator = _Translator(empty_sources={"New"})
    pipeline = _Pipeline(translator)
    state = _state()
    pipeline._state_pool[0] = state
    output = TranscribeStepOutput(
        stream_id=0,
        partial_transcript="Closed. New",
        current_step_transcript="Closed. New",
    )
    pipeline._translate_step_buffered([state], [output])

    flushed = pipeline.flush_translation_stream(0)

    assert flushed.final_translation == "<Closed. New>"


def test_native_last_request_flushes_unpunctuated_translation_before_state_deletion():
    pipeline = _NativePipeline(_Translator())
    options = ASRRequestOptions(
        enable_nmt=True,
        source_language="English",
        target_language="German",
    )
    request = Frame(
        samples=torch.zeros(8),
        stream_id=7,
        is_first=True,
        is_last=True,
        options=options,
    )

    output = pipeline.transcribe_step([request])[0]

    assert output.final_translation == "<unfinished ending>"
    assert output.partial_translation == ""
    assert output.current_step_translation == "<unfinished ending>"
    assert pipeline.get_state(7) is None
