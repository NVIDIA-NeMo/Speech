# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import json
import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from examples.asr.asr_cache_aware_streaming import speech_to_text_multitalker_streaming_infer as streaming_infer
from examples.asr.asr_cache_aware_streaming.speech_to_text_multitalker_streaming_infer import (
    MultitalkerTranscriptionConfig,
)
from omegaconf import OmegaConf

from nemo.collections.asr.models.configs.asr_models_config import CacheAwareStreamingConfig
from nemo.collections.asr.parts.utils.diarization_utils import (
    collect_diar_predictions,
    write_and_score_diar_predictions,
)
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
    MultiTalkerInstanceManager,
    SpeakerTaggedASR,
    append_word_and_ts_seq,
    configure_diar_streaming,
    fix_frame_time_step,
    get_multi_talker_samples_from_manifest,
    get_multitoken_words,
    get_new_sentence_dict,
    get_simulated_softmax,
    get_word_dict_content_offline,
    get_word_dict_content_online,
    set_batch_rttm_masks,
    validate_feature_frame_strides,
    write_seglst,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from tests.collections.asr.test_asr_rnnt_encoder_model_bpe import asr_model as offline_asr_model
from tests.collections.speaker_tasks.test_diar_sortformer_models import _create_sortformer_model
from tests.collections.speaker_tasks.test_diar_sortformer_models import sortformer_model as diar_model


def make_character_hypothesis(text, timestamps):
    return Hypothesis(
        score=0.0,
        y_sequence=[ord(char) for char in text],
        text=text,
        timestamp=torch.tensor(timestamps, dtype=torch.long),
        dec_state=SimpleNamespace(decoded_length=torch.tensor(max(timestamps, default=-1) + 1)),
        length=torch.tensor(len(timestamps)),
    )


def run_parallel_hypothesis_updates(updates, sent_break_sec):
    state = MultiTalkerInstanceManager.ASRState(
        max_num_of_spks=1,
        sent_break_sec=sent_break_sec,
        uppercase_first_letter=False,
    )
    state.speakers = [0]
    for text, timestamps, offset in updates:
        state.previous_hypothesis = [make_character_hypothesis(text, timestamps)]
        state.update_sessionwise_seglsts_for_parallel(offset=offset)
    return state


@pytest.fixture()
def asr_model(offline_asr_model):
    """Wrapper fixture that adds streaming_cfg to the asr_model from test_asr_rnnt_encoder_model_bpe"""
    # Add streaming_cfg to encoder for streaming tests
    streaming_cfg = CacheAwareStreamingConfig(
        valid_out_len=1,
        drop_extra_pre_encoded=7,
        chunk_size=8,
        shift_size=4,
        cache_drop_size=4,
        last_channel_cache_size=64,
        pre_encode_cache_size=0,
        last_channel_num=0,
        last_time_num=0,
    )
    offline_asr_model.encoder.streaming_cfg = streaming_cfg

    # Mock get_initial_cache_state method for MultiTalkerInstanceManager tests
    def get_initial_cache_state(batch_size=1):
        """Mock method to return initial cache state for streaming"""
        # Return dummy cache state tensors
        cache_last_channel = torch.zeros(2, batch_size, 64)
        cache_last_time = torch.zeros(2, batch_size, 64)
        cache_last_channel_len = torch.zeros(batch_size)
        return (cache_last_channel, cache_last_time, cache_last_channel_len)

    offline_asr_model.encoder.get_initial_cache_state = get_initial_cache_state

    return offline_asr_model


class TestGetNewSentenceDict:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "speaker,start_time,end_time,text,session_id",
        [
            ("speaker_0", 0.0, 1.0, "hello", None),
            ("spk1", 1.23, 4.56, "world", "session_A"),
        ],
    )
    def test_get_new_sentence_dict(self, speaker, start_time, end_time, text, session_id):
        result = get_new_sentence_dict(
            speaker=speaker, start_time=start_time, end_time=end_time, text=text, session_id=session_id
        )
        assert result["speaker"] == speaker
        assert result["start_time"] == start_time
        assert result["end_time"] == end_time
        assert result["words"] == text.lstrip()
        assert result["session_id"] == session_id


class TestFixFrameTimeStep:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "new_tokens,new_words,frame_inds_seq,expected",
        [
            # Case 1: Trim when frame_inds longer than tokens
            (["x", "y"], ["x", "y"], [1, 2, 3, 4], [3, 4]),
            # Case 2: No change when lengths match
            (["u", "v", "w"], ["u", "v", "w"], [7, 8, 9], [7, 8, 9]),
        ],
    )
    def test_fix_frame_time_step_shapes(self, new_tokens, new_words, frame_inds_seq, expected):
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(log=False))
        out = fix_frame_time_step(cfg, new_tokens, new_words, frame_inds_seq)
        assert out == expected


class TestGetSimulatedSoftmax:
    @pytest.mark.unit
    def test_invalid_dims(self):
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(min_sigmoid_val=0.0, max_num_of_spks=2))
        with pytest.raises(ValueError):
            get_simulated_softmax(cfg, torch.zeros((2, 3)))

    @pytest.mark.unit
    def test_invalid_length_vs_maxspks(self):
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(min_sigmoid_val=0.0, max_num_of_spks=4))
        with pytest.raises(ValueError):
            get_simulated_softmax(cfg, torch.zeros(3))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "vec,min_sigmoid_val,max_num_of_spks,expected_prefix",
        [
            # Clamp first element to min, normalize across all then zero out tail
            ([0.0, 0.2, 0.3, 0.5], 0.1, 3, [0.1 / 1.1, 0.2 / 1.1, 0.3 / 1.1]),
            # Zero-sum uniform case with tail zeroed
            ([0.0, 0.0, 0.0, 0.0], 0.0, 2, [0.25, 0.25]),
        ],
    )
    def test_valid_softmax_behavior(self, vec, min_sigmoid_val, max_num_of_spks, expected_prefix):
        cfg = OmegaConf.structured(
            MultitalkerTranscriptionConfig(min_sigmoid_val=min_sigmoid_val, max_num_of_spks=max_num_of_spks)
        )
        out = get_simulated_softmax(cfg, torch.tensor(vec))
        assert out.shape[0] == len(vec)
        # Tail past max_num_of_spks is zero
        if len(vec) > max_num_of_spks:
            assert torch.all(out[max_num_of_spks:] == 0)
        # First max_num_of_spks entries match expected prefix approximately
        assert torch.allclose(out[: len(expected_prefix)], torch.tensor(expected_prefix), atol=1e-5)


class TestWordDictContentOffline:
    @pytest.mark.unit
    @pytest.mark.parametrize("frame_stt,frame_end,expected_end", [(2, 5, 5), (2, 2, 3)])
    def test_get_word_dict_content_offline(self, frame_stt, frame_end, expected_end):
        # diar_pred_out with highest mean on speaker 2 within the selected frames
        T, N = 6, 3
        diar_pred_out = torch.zeros((T, N))
        diar_pred_out[2:5, 2] = 10.0
        cfg = OmegaConf.structured(
            MultitalkerTranscriptionConfig(
                left_frame_shift=0, right_frame_shift=0, max_num_of_spks=3, min_sigmoid_val=0.0
            )
        )
        word = "hello"
        res = get_word_dict_content_offline(
            cfg=cfg,
            word=word,
            word_index=0,
            diar_pred_out=diar_pred_out,
            time_stt_end_tuple=(frame_stt, frame_end),
            frame_len=0.08,
        )
        assert res["word"] == word
        assert res["speaker"] == "speaker_2"
        assert res["frame_stt"] == frame_stt
        assert res["frame_end"] == expected_end
        assert abs(res["start_time"] - frame_stt * 0.08) < 1e-6
        assert abs(res["end_time"] - expected_end * 0.08) < 1e-6


class TestWordDictContentOnline:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "token_group,frame_inds_seq,time_step_local_offset,expected_stt,expected_end",
        [
            # Single token: end = stt + 1
            (["t1"], [4, 5, 6], 1, 5, 6),
            # Multi-token falling back due to IndexError -> end = stt + 1
            (["t1", "t2", "t3"], [8], 0, 8, 9),
        ],
    )
    def test_get_word_dict_content_online(
        self, token_group, frame_inds_seq, time_step_local_offset, expected_stt, expected_end
    ):
        T, N = 12, 4
        diar_pred_out_stream = torch.zeros((T, N))
        diar_pred_out_stream[expected_stt:expected_end, 1] = 5.0  # Make speaker 1 dominant in the window
        cfg = OmegaConf.structured(
            MultitalkerTranscriptionConfig(
                left_frame_shift=0, right_frame_shift=0, max_num_of_spks=4, min_sigmoid_val=0.0
            )
        )

        res = get_word_dict_content_online(
            cfg=cfg,
            word="token",
            word_index=0,
            diar_pred_out_stream=diar_pred_out_stream,
            token_group=token_group,
            frame_inds_seq=frame_inds_seq,
            time_step_local_offset=time_step_local_offset,
            frame_len=0.08,
        )
        assert res["frame_stt"] == expected_stt
        assert res["frame_end"] == expected_end
        assert res["speaker"] == "speaker_1"


class TestGetMultitokenWords:
    @pytest.mark.unit
    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_multitoken_words_replaces_shorter_saved(self, verbose):
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(verbose=verbose))
        word_and_ts_seq = {
            "words": [{"word": "hello"}, {"word": "multi"}, {"word": "world"}],
            "buffered_words": [],
            "word_window_seq": [],
        }
        word_seq = ["hello", "multi token", "world", "new"]
        new_words = ["new"]
        out = get_multitoken_words(cfg, word_and_ts_seq, word_seq, new_words, fix_prev_words_count=2)
        # The second-to-last element should be replaced by the longer previous word
        assert out["words"][1]["word"] == "multi token"


class TestAppendWordAndTsSeq:
    @pytest.mark.unit
    def test_append_and_fifo_pop(self):
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(word_window=2))
        word_and_ts_seq = {
            "words": [{"word": "a", "speaker": "speaker_0"}, {"word": "b", "speaker": "speaker_1"}],
            "buffered_words": [{"word": "a", "speaker": "speaker_0"}, {"word": "b", "speaker": "speaker_1"}],
            "token_frame_index": [],
            "offset_count": 0,
            "status": "success",
            "sentences": None,
            "last_word_index": 0,
            "speaker_count": None,
            "transcription": None,
            "max_spk_probs": [],
            "word_window_seq": ["a", "b"],
            "speaker_count_buffer": ["speaker_0", "speaker_1"],
            "sentence_memory": {},
        }
        word_dict = {"word": "c", "speaker": "speaker_1"}
        word_idx_offset, out = append_word_and_ts_seq(cfg, 0, word_and_ts_seq, word_dict)
        assert word_idx_offset == 0
        # FIFO: buffered_words and word_window_seq should maintain length <= word_window
        assert len(out["buffered_words"]) == cfg.word_window
        assert len(out["word_window_seq"]) == cfg.word_window
        # speaker_count: unique speakers in buffer
        assert out["speaker_count"] == 2


class TestGetDiarPredOutStream:
    class Dummy:
        def __init__(self, diar_model, block_frame_length, frame_hop_length):
            self.diar_model = diar_model
            self._nframes_per_chunk = block_frame_length
            self._frame_hop_length = frame_hop_length

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "step_num,block_frame_length,frame_hop_length,is_buffer_empty,"
            "expected_stream_end,expected_block_start,expected_block_end"
        ),
        [
            (0, 3, 3, False, 3, 0, 3),
            (1, 3, 3, False, 6, 3, 6),
            (1, 4, 3, False, 6, 3, 7),
            (2, 4, 3, True, 10, 6, 10),
        ],
    )
    def test_get_diar_pred_out_stream(
        self,
        diar_model,
        step_num,
        block_frame_length,
        frame_hop_length,
        is_buffer_empty,
        expected_stream_end,
        expected_block_start,
        expected_block_end,
    ):
        B, T, N = 2, 10, 4
        mats = torch.arange(B * T * N, dtype=torch.float32).reshape(B, T, N)
        diar_model.rttms_mask_mats = mats
        dummy = self.Dummy(diar_model, block_frame_length, frame_hop_length)

        new_stream, new_chunk = SpeakerTaggedASR.get_diar_pred_out_stream(
            dummy, step_num=step_num, is_buffer_empty=is_buffer_empty
        )

        assert torch.equal(new_stream, mats[:, :expected_stream_end])
        assert torch.equal(new_chunk, mats[:, expected_block_start:expected_block_end])


class TestConfigureDiarStreaming:
    @pytest.mark.unit
    @pytest.mark.parametrize("diar_right_context", [0, 1, 2, 3, 5, 8, 13])
    def test_accepts_nonnegative_diar_right_context(self, diar_model, diar_right_context):
        cfg = MultitalkerTranscriptionConfig(diar_right_context=diar_right_context)

        configure_diar_streaming(diar_model, cfg, output_subsampling_factor=8)

        assert diar_model.sortformer_modules.chunk_right_context == diar_right_context

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "diar_right_context,error_type",
        [
            (-1, ValueError),
            (1.5, TypeError),
            ("3", TypeError),
            (None, TypeError),
            (True, TypeError),
        ],
    )
    def test_rejects_invalid_diar_right_context(self, diar_model, diar_right_context, error_type):
        cfg = MultitalkerTranscriptionConfig(diar_right_context=diar_right_context)

        with pytest.raises(error_type, match="diar_right_context"):
            configure_diar_streaming(diar_model, cfg, output_subsampling_factor=8)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "diar_chunk_len,diar_right_context",
        [(14, 0), (14, 5), (10, 13)],
    )
    def test_applies_asr_aligned_diarization_geometry(self, diar_model, diar_chunk_len, diar_right_context):
        cfg = MultitalkerTranscriptionConfig(diar_right_context=diar_right_context)

        configure_diar_streaming(
            diar_model,
            cfg,
            output_subsampling_factor=8,
            diar_chunk_len=diar_chunk_len,
        )

        assert diar_model.sortformer_modules.chunk_len == diar_chunk_len
        assert diar_model.sortformer_modules.chunk_right_context == diar_right_context

    @pytest.mark.unit
    def test_preserves_model_streaming_geometry(self, diar_model):
        expected_geometry = (
            diar_model.sortformer_modules.chunk_len,
            diar_model.sortformer_modules.chunk_left_context,
        )
        cfg = MultitalkerTranscriptionConfig()

        configure_diar_streaming(diar_model, cfg, output_subsampling_factor=8)

        assert (
            diar_model.sortformer_modules.chunk_len,
            diar_model.sortformer_modules.chunk_left_context,
        ) == expected_geometry
        assert diar_model.sortformer_modules.chunk_right_context == 0

    @pytest.mark.unit
    @pytest.mark.parametrize("spkcache_len", [None, 256])
    def test_spkcache_len_is_an_optional_model_override(self, diar_model, spkcache_len):
        original_spkcache_len = diar_model.sortformer_modules.spkcache_len
        cfg = MultitalkerTranscriptionConfig(spkcache_len=spkcache_len)

        configure_diar_streaming(diar_model, cfg, output_subsampling_factor=8)

        expected_spkcache_len = original_spkcache_len if spkcache_len is None else spkcache_len
        assert diar_model.sortformer_modules.spkcache_len == expected_spkcache_len

    @pytest.mark.unit
    @pytest.mark.parametrize("asr_output_subsampling_factor,diar_chunk_len", [(8, 14)])
    def test_aligns_high_resolution_diarizer_to_asr_factor(self, asr_output_subsampling_factor, diar_chunk_len):
        diar_model = _create_sortformer_model(
            high_resolution=True,
            output_subsampling_factor=1,
        )
        cfg = MultitalkerTranscriptionConfig()

        effective_factor = configure_diar_streaming(
            diar_model,
            cfg,
            output_subsampling_factor=asr_output_subsampling_factor,
            diar_chunk_len=diar_chunk_len,
        )

        assert effective_factor == asr_output_subsampling_factor
        assert diar_model.output_subsampling_factor == asr_output_subsampling_factor
        assert diar_model._cfg.output_subsampling_factor == asr_output_subsampling_factor

    @pytest.mark.unit
    @pytest.mark.parametrize("asr_output_subsampling_factor", [3])
    def test_rejects_diarizer_that_cannot_match_asr_factor(self, diar_model, asr_output_subsampling_factor):
        cfg = MultitalkerTranscriptionConfig(streaming_mode=False)

        with pytest.raises(ValueError, match="requires the diarization output subsampling factor"):
            configure_diar_streaming(
                diar_model,
                cfg,
                output_subsampling_factor=asr_output_subsampling_factor,
            )


class TestFeatureFrameStrides:
    @pytest.mark.unit
    @pytest.mark.parametrize("asr_stride,diar_stride", [(0.01, 0.01), (0.02, 0.02)])
    def test_accepts_equal_feature_frame_strides(self, asr_stride, diar_stride):
        asr_model = SimpleNamespace(cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=asr_stride)))
        diar_model = SimpleNamespace(_cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=diar_stride)))

        validate_feature_frame_strides(asr_model, diar_model)

    @pytest.mark.unit
    @pytest.mark.parametrize("asr_stride,diar_stride", [(0.01, 0.02)])
    def test_rejects_mismatched_feature_frame_strides(self, asr_stride, diar_stride):
        asr_model = SimpleNamespace(cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=asr_stride)))
        diar_model = SimpleNamespace(_cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=diar_stride)))

        with pytest.raises(ValueError, match="equal ASR and diarization feature-frame strides"):
            validate_feature_frame_strides(asr_model, diar_model)


class TestStreamingViewRouting:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "strategy,spk_supervision,masked_asr,diar_right_context",
        [
            ("serial", "diar", True, 3),
            ("parallel", "diar", False, 5),
            ("serial", "rttm", False, 13),
            ("parallel", "rttm", True, 13),
        ],
    )
    def test_launch_routes_separate_views_without_changing_asr(
        self, monkeypatch, strategy, spk_supervision, masked_asr, diar_right_context
    ):
        captured = {}
        asr_audio = torch.randn(1, 80, 112)
        asr_lengths = torch.tensor([112])

        class FakeBuffer:
            def iter_with_right_context(self, right_context_size):
                captured["requested_right_context_size"] = right_context_size
                if right_context_size == 0:
                    diar_audio, diar_lengths = asr_audio, asr_lengths
                else:
                    diar_audio = torch.randn(1, 80, 112 + right_context_size)
                    diar_lengths = torch.tensor([112 + right_context_size])
                captured["diar_audio"] = diar_audio
                captured["diar_lengths"] = diar_lengths
                yield asr_audio, asr_lengths, diar_audio, diar_lengths

            def is_buffer_empty(self):
                return True

        class FakeSpeakerTaggedASR:
            def __init__(self, cfg, asr_model, diar_model):
                pass

            def perform_serial_streaming_stt_spk(self, **kwargs):
                captured["perform_kwargs"] = kwargs

            def perform_parallel_streaming_stt_spk(self, **kwargs):
                captured["perform_kwargs"] = kwargs

        cfg = OmegaConf.structured(
            MultitalkerTranscriptionConfig(
                spk_supervision=spk_supervision,
                masked_asr=masked_asr,
                diar_right_context=diar_right_context,
            )
        )
        asr_model = SimpleNamespace(
            encoder=SimpleNamespace(
                streaming_cfg=SimpleNamespace(drop_extra_pre_encoded=2),
            )
        )
        diar_model = SimpleNamespace(encoder=SimpleNamespace(subsampling_factor=8))
        monkeypatch.setattr(streaming_infer, "SpeakerTaggedASR", FakeSpeakerTaggedASR)
        monkeypatch.setattr(streaming_infer, "autocast", nullcontext(), raising=False)

        launch_fn = (
            streaming_infer.launch_serial_streaming
            if strategy == "serial"
            else streaming_infer.launch_parallel_streaming
        )
        launch_fn(
            cfg=cfg,
            asr_model=asr_model,
            diar_model=diar_model,
            streaming_buffer=FakeBuffer(),
        )

        expected_right_offset = 0 if spk_supervision == "rttm" else diar_right_context * 8
        assert captured["requested_right_context_size"] == expected_right_offset
        assert captured["perform_kwargs"]["chunk_audio"] is asr_audio
        assert captured["perform_kwargs"]["chunk_lengths"] is asr_lengths
        assert captured["perform_kwargs"]["diar_chunk_audio"] is captured["diar_audio"]
        assert captured["perform_kwargs"]["diar_chunk_lengths"] is captured["diar_lengths"]
        if spk_supervision == "rttm":
            assert captured["diar_audio"] is asr_audio
            assert captured["diar_lengths"] is asr_lengths


class TestPrepareDiarChunk:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "pre_encode_cache_size,drop_extra_pre_encoded,chunk_lengths,expected_shape,"
            "expected_lengths,expected_first_frame,expected_same_tensor"
        ),
        [
            ([0, 9], 2, (121,), (1, 1, 112), (112,), 9, False),
            ([0, 9], 0, (121,), (1, 1, 121), (121,), 0, True),
            (4, 2, (121, 3), (2, 1, 117), (117, 0), 4, False),
        ],
    )
    def test_feature_stacking_removes_asr_preencode_cache(
        self,
        pre_encode_cache_size,
        drop_extra_pre_encoded,
        chunk_lengths,
        expected_shape,
        expected_lengths,
        expected_first_frame,
        expected_same_tensor,
    ):
        dummy = SimpleNamespace(
            _diar_uses_feature_stacking=True,
            asr_model=SimpleNamespace(
                encoder=SimpleNamespace(
                    streaming_cfg=SimpleNamespace(pre_encode_cache_size=pre_encode_cache_size),
                )
            ),
        )
        chunk_audio = torch.arange(len(chunk_lengths) * 121, dtype=torch.float32).reshape(len(chunk_lengths), 1, 121)

        diar_audio, diar_lengths, diar_drop = SpeakerTaggedASR._prepare_diar_chunk(
            dummy,
            chunk_audio=chunk_audio,
            chunk_lengths=torch.tensor(chunk_lengths),
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )

        assert diar_audio.shape == expected_shape
        assert diar_audio[0, 0, 0] == expected_first_frame
        assert torch.equal(diar_lengths, torch.tensor(expected_lengths))
        assert diar_drop == 0
        assert (diar_audio is chunk_audio) is expected_same_tensor

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "chunk_shape,chunk_lengths,drop_extra_pre_encoded",
        [((1, 128, 121), (121,), 2)],
    )
    def test_convolutional_subsampling_keeps_asr_cache_and_drop(
        self, chunk_shape, chunk_lengths, drop_extra_pre_encoded
    ):
        dummy = SimpleNamespace(_diar_uses_feature_stacking=False)
        chunk_audio = torch.randn(chunk_shape)
        chunk_lengths = torch.tensor(chunk_lengths)

        diar_audio, diar_lengths, diar_drop = SpeakerTaggedASR._prepare_diar_chunk(
            dummy,
            chunk_audio=chunk_audio,
            chunk_lengths=chunk_lengths,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )

        assert diar_audio is chunk_audio
        assert diar_lengths is chunk_lengths
        assert diar_drop == drop_extra_pre_encoded

    @pytest.mark.unit
    @pytest.mark.parametrize("pre_encode_kind", ["feature_stacking", "conv_subsampling"])
    @pytest.mark.parametrize("diar_right_context", [0, 1, 3, 5, 13])
    @pytest.mark.parametrize("is_first_chunk", [True, False])
    def test_actual_pre_encode_preserves_central_and_future_frames(
        self, pre_encode_kind, diar_right_context, is_first_chunk
    ):
        frontend_encoder = "transformer" if pre_encode_kind == "feature_stacking" else "conformer"
        diar_model = _create_sortformer_model(frontend_encoder=frontend_encoder).eval()
        cache_size = 0 if is_first_chunk else 9
        central_mel_frames = 105 if is_first_chunk else 112
        drop_extra_pre_encoded = 0 if is_first_chunk else 2
        input_frames = cache_size + central_mel_frames + diar_right_context * 8
        dummy = SimpleNamespace(
            _diar_uses_feature_stacking=pre_encode_kind == "feature_stacking",
            asr_model=SimpleNamespace(
                encoder=SimpleNamespace(
                    streaming_cfg=SimpleNamespace(pre_encode_cache_size=[0, 9]),
                )
            ),
        )
        chunk_audio = torch.randn(1, diar_model.encoder._feat_in, input_frames)

        diar_audio, diar_lengths, diar_drop = SpeakerTaggedASR._prepare_diar_chunk(
            dummy,
            chunk_audio=chunk_audio,
            chunk_lengths=torch.tensor([input_frames]),
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )
        pre_encoded, pre_encoded_lengths = diar_model._call_pre_encode(diar_audio.transpose(1, 2), diar_lengths)
        if diar_drop:
            pre_encoded = pre_encoded[:, diar_drop:]
            pre_encoded_lengths = pre_encoded_lengths - diar_drop

        expected_frames = 14 + diar_right_context
        assert pre_encoded.shape[1] == expected_frames
        assert pre_encoded_lengths.tolist() == [expected_frames]


class TestDiarizationStreamingRouting:
    @pytest.mark.unit
    @pytest.mark.parametrize("diar_right_context", [0, 1, 3, 5, 13])
    def test_forward_passes_wider_view_and_fixed_right_offset(self, diar_right_context):
        captured = {}
        diar_audio = torch.randn(2, 80, 112 + diar_right_context * 8)
        diar_lengths = torch.tensor([diar_audio.shape[-1], diar_audio.shape[-1] - 8])
        streaming_state = object()
        total_preds = torch.zeros(2, 0, 4)

        def forward_streaming_step(**kwargs):
            captured.update(kwargs)
            return streaming_state, total_preds

        dummy = SimpleNamespace(
            _diar_right_offset=diar_right_context * 8,
            _prepare_diar_chunk=lambda audio, lengths, drop: (audio, lengths, drop),
            diar_model=SimpleNamespace(forward_streaming_step=forward_streaming_step),
            instance_manager=SimpleNamespace(
                diar_states=SimpleNamespace(
                    streaming_state=streaming_state,
                    diar_pred_out_stream=total_preds,
                )
            ),
        )

        SpeakerTaggedASR._forward_diarization_streaming_step(
            dummy,
            diar_chunk_audio=diar_audio,
            diar_chunk_lengths=diar_lengths,
            drop_extra_pre_encoded=2,
        )

        torch.testing.assert_close(captured["processed_signal"], diar_audio.transpose(1, 2))
        assert captured["processed_signal"].data_ptr() == diar_audio.data_ptr()
        assert captured["processed_signal_length"] is diar_lengths
        assert captured["drop_extra_pre_encoded"] == 2
        assert captured["right_offset"] == diar_right_context * 8


class TestParallelStreamingCacheGating:
    @pytest.mark.unit
    def test_timestamp_offset_advances_once_for_empty_and_active_steps(self):
        frame_hop_length = 14
        frame_len_sec = 0.08
        step_duration = frame_hop_length * frame_len_sec
        diar_preds = torch.ones(1, frame_hop_length, 1)
        active_result = [(None, None, None, None)]
        asr_forward_calls = []
        seglst_offsets = []

        def conformer_stream_step(**kwargs):
            asr_forward_calls.append(kwargs)
            return (
                torch.zeros(1, 1),
                None,
                torch.zeros(1, 1, 1),
                torch.zeros(1, 1, 1),
                torch.zeros(1),
                [object()],
            )

        instance_manager = SimpleNamespace(
            diar_states=SimpleNamespace(streaming_state=object()),
            reset=lambda **kwargs: None,
            to=lambda device: None,
            update_diar_state=lambda **kwargs: None,
            get_active_speakers_info=lambda **kwargs: active_result[0],
            active_cache_last_channel=torch.zeros(1, 1, 1),
            active_cache_last_time=torch.zeros(1, 1, 1),
            active_cache_last_channel_len=torch.zeros(1),
            active_previous_hypotheses=[None],
            active_asr_pred_out_stream=[None],
            update_asr_state=lambda *args: None,
            update_seglsts=lambda offset: seglst_offsets.append(offset),
            batch_asr_states=[],
        )
        streamer = SimpleNamespace(
            _offset_chunk_start_time=0.0,
            _frame_hop_length=frame_hop_length,
            _frame_len_sec=frame_len_sec,
            _nframes_per_chunk=frame_hop_length,
            _max_num_of_spks=1,
            _single_speaker_mode=False,
            _cache_gating=True,
            _cache_gating_buffer_size=4,
            _masked_asr=True,
            _use_mask_preencode=False,
            _binary_diar_preds=True,
            n_active_speakers_per_stream=1,
            instance_manager=instance_manager,
            diar_model=SimpleNamespace(rttms_mask_mats=diar_preds),
            asr_model=SimpleNamespace(conformer_stream_step=conformer_stream_step),
            get_diar_pred_out_stream=lambda step_num, is_buffer_empty: (diar_preds, diar_preds),
            _find_active_speakers=lambda preds, n_active_speakers_per_stream: [[0]],
            mask_features=lambda chunk_audio, mask: chunk_audio,
            cfg={"generate_realtime_scripts": False},
            transcribed_speaker_texts=[],
        )
        step_kwargs = {
            "chunk_audio": torch.ones(1, 80, frame_hop_length),
            "chunk_lengths": torch.tensor([frame_hop_length]),
            "is_buffer_empty": False,
            "drop_extra_pre_encoded": 0,
        }

        SpeakerTaggedASR.perform_parallel_streaming_stt_spk(streamer, step_num=0, **step_kwargs)
        assert streamer._offset_chunk_start_time == pytest.approx(step_duration)
        assert not asr_forward_calls

        SpeakerTaggedASR.perform_parallel_streaming_stt_spk(streamer, step_num=1, **step_kwargs)
        assert streamer._offset_chunk_start_time == pytest.approx(2 * step_duration)
        assert not asr_forward_calls

        active_result[0] = (
            step_kwargs["chunk_audio"],
            step_kwargs["chunk_lengths"],
            torch.ones(1, frame_hop_length),
            torch.zeros(1, frame_hop_length, dtype=torch.bool),
        )
        SpeakerTaggedASR.perform_parallel_streaming_stt_spk(streamer, step_num=2, **step_kwargs)
        assert seglst_offsets == [pytest.approx(2 * step_duration)]
        assert streamer._offset_chunk_start_time == pytest.approx(3 * step_duration)
        assert len(asr_forward_calls) == 1

        SpeakerTaggedASR.perform_parallel_streaming_stt_spk(streamer, step_num=0, **step_kwargs)
        assert seglst_offsets[-1] == pytest.approx(0.0)
        assert streamer._offset_chunk_start_time == pytest.approx(step_duration)
        assert len(asr_forward_calls) == 2


class TestParallelWordAwareSegmentation:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "previous_text,current_text,expected_delta",
        [
            ("go go", "go go now", " now"),
            ("turn left", "turn right", "turn right"),
            ("same", "same", None),
        ],
    )
    def test_text_delta_uses_prefix_slicing_and_preserves_non_prefix_fallback(
        self, previous_text, current_text, expected_delta
    ):
        state = MultiTalkerInstanceManager.ASRState(max_num_of_spks=1)
        state._prev_history_speaker_texts[0] = previous_text

        assert state._is_new_text(spk_idx=0, text=current_text) == expected_delta

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "updates,expected_text",
        [
            (
                [
                    ("Lind", [0, 1, 2, 3], 0.0),
                    ("Linda", [0, 1, 2, 3, 30], 2.0),
                ],
                "Linda",
            ),
            (
                [
                    ("Don", [0, 1, 2], 0.0),
                    ("Don't start", [0, 1, 2, 30, 31, 32, 40, 41, 42, 43, 44], 2.0),
                ],
                "Don't start",
            ),
        ],
    )
    def test_word_continuations_never_create_segment_boundaries(self, updates, expected_text):
        state = run_parallel_hypothesis_updates(updates, sent_break_sec=0.1)

        exported_text = " ".join(segment["words"] for segment in state.seglsts)

        assert exported_text == expected_text

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "updates,expected_segments",
        [
            pytest.param(
                [
                    ("hello", [0, 1, 2, 3, 4], 0.0),
                    ("hello.", [0, 1, 2, 3, 4, 5], 10.0),
                ],
                [("hello.", 0.0, 0.4)],
                id="punctuation-after-long-gap",
            ),
            pytest.param(
                [
                    ("Don", [0, 1, 2], 0.0),
                    ("Don't start", list(range(11)), 10.0),
                ],
                [("Don't", 0.0, 0.24), ("start", 10.0, 10.64)],
                id="partial-word-and-new-words-after-long-gap",
            ),
            pytest.param(
                [
                    ("Lind", [0, 1, 2, 3], 0.0),
                    ("Linda", [0, 1, 2, 3, 4], 10.0),
                ],
                [("Linda", 0.0, 0.32)],
                id="partial-word-only-after-long-gap",
            ),
            pytest.param(
                [
                    ("Lind", [0, 1, 2, 3], 0.0),
                    ("Linda", [0, 1, 2, 3, 4], 0.32),
                ],
                [("Linda", 0.0, 0.4)],
                id="adjacent-continuation",
            ),
        ],
    )
    def test_continuation_timestamps_respect_inactive_gaps(self, updates, expected_segments):
        state = run_parallel_hypothesis_updates(updates, sent_break_sec=0.5)
        segments = state.seglsts

        assert len(segments) == len(expected_segments)
        for segment, (expected_text, expected_start, expected_end) in zip(segments, expected_segments):
            assert segment["words"] == expected_text
            assert segment["start_time"] == pytest.approx(expected_start)
            assert segment["end_time"] == pytest.approx(expected_end)

        assert " ".join(segment["words"] for segment in segments) == updates[-1][0]
        assert all(segment["words"].strip() for segment in segments)
        assert all(segment["words"].strip() not in {".", ",", "?", "!"} for segment in segments)
        assert all(segment["start_time"] <= segment["end_time"] for segment in segments)
        assert [segment["start_time"] for segment in segments] == sorted(segment["start_time"] for segment in segments)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "updates,expected_segments",
        [
            (
                [
                    ("hello", [0, 1, 2, 3, 4], 0.0),
                    ("hello world", [0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 34], 2.0),
                ],
                ("hello", "world"),
            ),
            (
                [
                    ("hello ", [0, 1, 2, 3, 4, 5], 0.0),
                    ("hello world", [0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 34], 2.0),
                ],
                ("hello", "world"),
            ),
        ],
    )
    def test_clean_word_boundary_can_create_segment(self, updates, expected_segments):
        state = run_parallel_hypothesis_updates(updates, sent_break_sec=0.5)

        assert tuple(segment["words"].strip() for segment in state.seglsts) == expected_segments

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "updates,expected_text",
        [
            (
                [
                    ("hello", [0, 1, 2, 3, 4], 0.0),
                    ("hello,", [0, 1, 2, 3, 4, 20], 1.0),
                    ("hello, world", [0, 1, 2, 3, 4, 20, 21, 30, 31, 32, 33, 34], 2.0),
                ],
                "hello, world",
            ),
            (
                [
                    ("go go", [0, 1, 2, 3, 4], 0.0),
                    ("go go now", [0, 1, 2, 3, 4, 5, 20, 21, 22], 1.0),
                ],
                "go go now",
            ),
        ],
    )
    def test_punctuation_and_repeated_prefixes_preserve_text(self, updates, expected_text):
        state = run_parallel_hypothesis_updates(updates, sent_break_sec=0.1)

        exported_text = " ".join(segment["words"] for segment in state.seglsts)

        assert exported_text == expected_text
        assert all(segment["words"] not in {",", ".", "!", "?"} for segment in state.seglsts)

    @pytest.mark.unit
    def test_word_stream_is_invariant_while_segment_layout_changes_with_threshold(self):
        updates = [
            ("Lind", [0, 1, 2, 3], 0.0),
            ("Linda", [0, 1, 2, 3, 30], 2.0),
            ("Linda next", [0, 1, 2, 3, 30, 60, 61, 62, 63, 64], 4.08),
        ]
        layouts = {}
        for sent_break_sec in [30.0, 1.0, 0.5, 0.1, 0.0]:
            state = run_parallel_hypothesis_updates(updates, sent_break_sec=sent_break_sec)
            segments = state.seglsts
            layouts[sent_break_sec] = tuple(segment["words"].strip() for segment in segments)
            normalized_words = " ".join(segment["words"] for segment in segments).split()

            assert normalized_words == ["Linda", "next"]
            assert all(segment["start_time"] <= segment["end_time"] for segment in segments)
            assert [segment["start_time"] for segment in segments] == sorted(
                segment["start_time"] for segment in segments
            )

        assert len(set(layouts.values())) >= 2
        assert layouts[30.0] == ("Linda next",)
        assert layouts[0.0] == ("Linda", "next")


class TestParallelASRStateTimestamps:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "initial_decoded_length,next_decoded_length,next_timestamp,offset,expected_start,expected_end",
        [(14, 28, 15, 1.12, 1.20, 1.28)],
    )
    def test_decoded_length_advances_when_chunk_has_no_new_text(
        self,
        initial_decoded_length,
        next_decoded_length,
        next_timestamp,
        offset,
        expected_start,
        expected_end,
    ):
        asr_state = MultiTalkerInstanceManager.ASRState(max_num_of_spks=1, uppercase_first_letter=False)
        asr_state.speakers = [0]
        asr_state.previous_hypothesis = [
            Hypothesis(
                score=0.0,
                y_sequence=[],
                text="",
                timestamp=torch.empty(0, dtype=torch.long),
                dec_state=SimpleNamespace(decoded_length=torch.tensor(initial_decoded_length)),
                length=torch.tensor(0),
            )
        ]

        asr_state.update_sessionwise_seglsts_for_parallel(offset=0.0)

        assert asr_state._prev_decoded_lengths[0] == initial_decoded_length

        asr_state.previous_hypothesis = [
            Hypothesis(
                score=0.0,
                y_sequence=[1],
                text="hello",
                timestamp=torch.tensor([next_timestamp]),
                dec_state=SimpleNamespace(decoded_length=torch.tensor(next_decoded_length)),
                length=torch.tensor(1),
            )
        ]
        asr_state.update_sessionwise_seglsts_for_parallel(offset=offset)

        assert asr_state.seglsts[0]["start_time"] == pytest.approx(expected_start)
        assert asr_state.seglsts[0]["end_time"] == pytest.approx(expected_end)


class TestWriteSeglst:
    @pytest.mark.unit
    def test_write_and_read(self, tmp_path):
        seglst = [
            {"speaker": "speaker_0", "start_time": 0.0, "end_time": 1.0, "words": "hi", "session_id": "S1"},
            {"speaker": "speaker_1", "start_time": 1.0, "end_time": 2.0, "words": "there", "session_id": "S1"},
        ]
        outpath = tmp_path / "out.json"
        write_seglst(str(outpath), seglst)
        content = outpath.read_text(encoding="utf-8")
        assert content == json.dumps(seglst, indent=2) + "\n"


class TestWriteDiarPredictionsToRttm:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "audio_filepath,prediction_shape,active_regions,feature_length,expected_lines",
        [
            (
                "/audio/session.wav",
                (1, 6, 2),
                ((1, 3, 0, 0.9), (3, 6, 1, 0.8)),
                40,
                (
                    "SPEAKER session 1 0.080 0.160 <NA> <NA> speaker_0 <NA> <NA>",
                    "SPEAKER session 1 0.240 0.160 <NA> <NA> speaker_1 <NA> <NA>",
                ),
            )
        ],
    )
    def test_writes_contiguous_segments_and_ignores_batch_padding(
        self,
        tmp_path,
        audio_filepath,
        prediction_shape,
        active_regions,
        feature_length,
        expected_lines,
    ):
        diar_preds = torch.zeros(prediction_shape)
        for start, end, speaker, score in active_regions:
            diar_preds[0, start:end, speaker] = score
        output_dir = tmp_path / "rttms"

        predictions, metadata = collect_diar_predictions(
            diar_preds=diar_preds,
            samples=[{"audio_filepath": audio_filepath}],
            feature_lengths=torch.tensor([feature_length]),
            feature_frame_length_sec=0.01,
            diar_frame_length_sec=0.08,
        )
        write_and_score_diar_predictions(
            predictions=predictions,
            samples=metadata,
            output_subsampling_factor=8,
            diar_output_rttm_dir=str(output_dir),
            diar_collar=0.0,
            diar_ignore_overlap=False,
        )

        assert (output_dir / "session.rttm").read_text(encoding="utf-8").splitlines() == list(expected_lines)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "samples,expected_recording_ids",
        [
            (
                (
                    {"audio_filepath": "/dataset/a/first.wav", "duration": 0.16, "offset": 0.0},
                    {"audio_filepath": "/dataset/b/second.wav", "duration": 0.16, "offset": 0.0},
                ),
                ("first", "second"),
            ),
            (
                (
                    {
                        "audio_filepath": "/dataset/a/session.wav",
                        "uniq_id": "session_a",
                        "duration": 0.16,
                        "offset": 0.0,
                    },
                    {
                        "audio_filepath": "/dataset/b/session.wav",
                        "uniq_id": "session_b",
                        "duration": 0.16,
                        "offset": 0.0,
                    },
                ),
                ("session_a", "session_b"),
            ),
        ],
    )
    def test_distinct_recording_ids_write_distinct_rttms(self, tmp_path, samples, expected_recording_ids):
        output_dir = tmp_path / "rttms"
        predictions = [torch.full((1, 2, 1), 0.9) for _ in samples]

        write_and_score_diar_predictions(
            predictions=predictions,
            samples=[dict(sample) for sample in samples],
            output_subsampling_factor=8,
            diar_output_rttm_dir=str(output_dir),
            diar_collar=0.0,
            diar_ignore_overlap=False,
        )

        assert sorted(path.stem for path in output_dir.glob("*.rttm")) == sorted(expected_recording_ids)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "samples,duplicate_id",
        [
            (
                (
                    {"audio_filepath": "/dataset/a/session.wav"},
                    {"audio_filepath": "/dataset/b/session.wav"},
                ),
                "session",
            ),
            (
                (
                    {"audio_filepath": "/dataset/a/first.wav", "uniq_id": "shared"},
                    {"audio_filepath": "/dataset/b/second.wav", "uniq_id": "shared"},
                ),
                "shared",
            ),
        ],
    )
    def test_duplicate_recording_ids_raise_value_error(self, tmp_path, samples, duplicate_id):
        with pytest.raises(ValueError, match=f"Duplicate recording ID '{duplicate_id}'") as error:
            write_and_score_diar_predictions(
                predictions=[torch.zeros(1, 1, 1) for _ in samples],
                samples=[dict(sample) for sample in samples],
                output_subsampling_factor=8,
                diar_output_rttm_dir=str(tmp_path / "rttms"),
                diar_collar=0.0,
                diar_ignore_overlap=False,
            )

        for sample in samples:
            assert sample["audio_filepath"] in str(error.value)


class TestGetMultiTalkerSamplesFromManifest:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "all_masks_shape,batch_start,batch_size,expected_start,expected_end",
        [((5, 4, 2), 2, 2, 2, 4)],
    )
    def test_set_batch_rttm_masks_selects_current_audio_batch(
        self, all_masks_shape, batch_start, batch_size, expected_start, expected_end
    ):
        class DummyDiarModel:
            rttms_mask_mats = None

        diar_model = DummyDiarModel()
        all_masks = torch.arange(math.prod(all_masks_shape)).reshape(all_masks_shape)

        set_batch_rttm_masks(
            diar_model=diar_model,
            rttms_mask_mats=all_masks,
            batch_start=batch_start,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )

        assert torch.equal(diar_model.rttms_mask_mats, all_masks[expected_start:expected_end])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "rttm_line,duration,feat_per_sec,max_spks,expected_first_speaker_mask",
        [
            (
                "SPEAKER sample 1 0.16 0.16 <NA> <NA> speaker_A <NA> <NA>\n",
                1.0,
                0.08,
                2,
                (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            )
        ],
    )
    def test_rttm_targets_align_with_nonoverlapping_output_frames(
        self,
        tmp_path,
        rttm_line,
        duration,
        feat_per_sec,
        max_spks,
        expected_first_speaker_mask,
    ):
        rttm_path = tmp_path / "sample.rttm"
        rttm_path.write_text(rttm_line, encoding="utf-8")
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": "sample.wav",
                    "duration": duration,
                    "rttm_filepath": str(rttm_path),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(spk_supervision="rttm"))

        _, rttm_masks = get_multi_talker_samples_from_manifest(
            cfg, str(manifest_path), feat_per_sec=feat_per_sec, max_spks=max_spks
        )

        assert torch.equal(rttm_masks[0, :, 0], torch.tensor(expected_first_speaker_mask))

    @pytest.mark.unit
    def test_missing_audio_filepath(self, tmp_path):
        mpath = tmp_path / "manifest.jsonl"
        mpath.write_text(json.dumps({}) + "\n", encoding="utf-8")
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(spk_supervision="none"))
        with pytest.raises(KeyError):
            get_multi_talker_samples_from_manifest(cfg, str(mpath), feat_per_sec=100.0, max_spks=2)

    @pytest.mark.unit
    def test_rttm_missing_file(self, tmp_path):
        mpath = tmp_path / "manifest.jsonl"
        missing_rttm = str(tmp_path / "missing.rttm")
        line = {
            "audio_filepath": "sample.wav",
            "duration": 10.0,
            "rttm_filepath": missing_rttm,
        }
        mpath.write_text(json.dumps(line) + "\n", encoding="utf-8")
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(spk_supervision="rttm"))
        with pytest.raises(FileNotFoundError):
            get_multi_talker_samples_from_manifest(cfg, str(mpath), feat_per_sec=100.0, max_spks=2)


class TestSpeakerTaggedASRInit:
    """Test the initialization of SpeakerTaggedASR class"""

    @pytest.mark.unit
    @pytest.mark.parametrize("diar_right_context", [0, 1, 3, 5, 13])
    def test_init_converts_diar_right_context_to_input_frames(
        self, asr_model, diar_model, tmp_path, diar_right_context
    ):
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        cfg = OmegaConf.structured(
            MultitalkerTranscriptionConfig(
                audio_file=str(audio_path),
                diar_right_context=diar_right_context,
                batch_size=1,
            )
        )

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        assert speaker_tagged_asr._diar_right_offset == diar_right_context * diar_model.encoder.subsampling_factor

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "valid_out_len,cache_drop_size,expected_block_frame_length",
        [(10, 4, 14)],
    )
    def test_init_separates_frame_hop_from_block_length(
        self, asr_model, diar_model, tmp_path, valid_out_len, cache_drop_size, expected_block_frame_length
    ):
        audio_path = tmp_path / "test.wav"
        audio_path.touch()
        asr_model.encoder.streaming_cfg.valid_out_len = valid_out_len
        asr_model.encoder.streaming_cfg.cache_drop_size = cache_drop_size
        cfg = OmegaConf.structured(
            MultitalkerTranscriptionConfig(
                audio_file=str(audio_path),
                att_context_size=[70, 13],
                batch_size=1,
            )
        )

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        assert speaker_tagged_asr._frame_hop_length == valid_out_len
        assert speaker_tagged_asr._nframes_per_chunk == expected_block_frame_length

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "max_num_of_spks,sent_break_sec,masked_asr,cache_gating,"
            "cache_gating_buffer_size,mask_preencode,single_speaker_mode"
        ),
        [
            (3, 30.0, True, False, 2, False, False),
            (1, 0.5, False, True, 4, True, True),
        ],
    )
    def test_init_config_values(
        self,
        asr_model,
        diar_model,
        tmp_path,
        max_num_of_spks,
        sent_break_sec,
        masked_asr,
        cache_gating,
        cache_gating_buffer_size,
        mask_preencode,
        single_speaker_mode,
    ):
        """Test initialization with config values accessed through .get()."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        cfg = MultitalkerTranscriptionConfig(
            manifest_file=None,
            audio_file=str(audio_path),
            fix_prev_words_count=5,
            update_prev_words_sentence=10,
            ignored_initial_frame_steps=2,
            max_num_of_spks=max_num_of_spks,
            att_context_size=[0, 40],
            binary_diar_preds=True,
            batch_size=1,
            sent_break_sec=sent_break_sec,
            masked_asr=masked_asr,
            cache_gating=cache_gating,
            cache_gating_buffer_size=cache_gating_buffer_size,
            mask_preencode=mask_preencode,
            single_speaker_mode=single_speaker_mode,
            generate_realtime_scripts=False,
        )
        # Convert to OmegaConf to support .get() method
        cfg = OmegaConf.structured(cfg)

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        # Verify values from .get() calls are properly set
        # pylint: disable=protected-access
        assert speaker_tagged_asr._max_num_of_spks == max_num_of_spks
        assert speaker_tagged_asr._sent_break_sec == sent_break_sec
        assert speaker_tagged_asr._cache_gating is cache_gating
        assert speaker_tagged_asr._cache_gating_buffer_size == cache_gating_buffer_size
        assert speaker_tagged_asr._masked_asr is masked_asr
        assert speaker_tagged_asr._use_mask_preencode is mask_preencode
        assert speaker_tagged_asr._single_speaker_mode is single_speaker_mode
        # pylint: enable=protected-access

    @pytest.mark.unit
    def test_init_instance_manager_creation(self, asr_model, diar_model, tmp_path):
        """Test that instance_manager is properly created during initialization"""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        cfg = MultitalkerTranscriptionConfig(
            manifest_file=None,
            audio_file=str(audio_path),
            fix_prev_words_count=5,
            update_prev_words_sentence=10,
            ignored_initial_frame_steps=2,
            max_num_of_spks=4,
            att_context_size=[0, 50],
            binary_diar_preds=True,
            batch_size=2,
            generate_realtime_scripts=False,
        )
        # Convert to OmegaConf to support .get() method
        cfg = OmegaConf.structured(cfg)

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        # Verify instance_manager is created and has correct attributes
        assert speaker_tagged_asr.instance_manager is not None
        assert isinstance(speaker_tagged_asr.instance_manager, MultiTalkerInstanceManager)
        assert speaker_tagged_asr.instance_manager.asr_model == asr_model
        assert speaker_tagged_asr.instance_manager.diar_model == diar_model
        assert speaker_tagged_asr.instance_manager.max_num_of_spks == 4
        assert speaker_tagged_asr.instance_manager.batch_size == 2


class TestSpeakerTaggedASRMethods:
    """Test various methods of the SpeakerTaggedASR class"""

    @pytest.mark.unit
    def test_get_offset_sentence(self, asr_model, diar_model, tmp_path):
        """Test _get_offset_sentence method"""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        cfg = MultitalkerTranscriptionConfig(
            manifest_file=None,
            audio_file=str(audio_path),
            fix_prev_words_count=5,
            update_prev_words_sentence=10,
            ignored_initial_frame_steps=2,
            max_num_of_spks=4,
            att_context_size=[0, 50],
            binary_diar_preds=True,
            batch_size=1,
            generate_realtime_scripts=False,
        )
        cfg = OmegaConf.structured(cfg)

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        # Create a mock session_trans_dict
        session_trans_dict = {
            'uniq_id': 'session_1',
            'words': [
                {'speaker': 'speaker_0', 'start_time': 0.0, 'end_time': 0.5, 'word': 'hello'},
                {'speaker': 'speaker_0', 'start_time': 0.5, 'end_time': 1.0, 'word': 'world'},
            ],
        }

        # pylint: disable=protected-access
        result = speaker_tagged_asr._get_offset_sentence(session_trans_dict, 0)
        # pylint: enable=protected-access

        assert result['session_id'] == 'session_1'
        assert result['speaker'] == 'speaker_0'
        assert result['start_time'] == 0.0
        assert result['end_time'] == 0.5
        assert result['words'] == 'hello '

    @pytest.mark.unit
    def test_find_active_speakers_valid(self, asr_model, diar_model, tmp_path):
        """Test _find_active_speakers with valid inputs"""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        cfg = MultitalkerTranscriptionConfig(
            manifest_file=None,
            audio_file=str(audio_path),
            fix_prev_words_count=5,
            update_prev_words_sentence=10,
            ignored_initial_frame_steps=2,
            max_num_of_spks=4,
            att_context_size=[0, 50],
            binary_diar_preds=True,
            batch_size=1,
            generate_realtime_scripts=False,
        )
        cfg = OmegaConf.structured(cfg)

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        # Create mock diar predictions: (B=2, T=10, N=4)
        diar_preds = torch.zeros(2, 10, 4)
        # First batch: speakers 0 and 2 are active (high values)
        diar_preds[0, :, 0] = 0.8
        diar_preds[0, :, 2] = 0.9
        # Second batch: speaker 1 is active
        diar_preds[1, :, 1] = 0.7

        # pylint: disable=protected-access
        result = speaker_tagged_asr._find_active_speakers(diar_preds, n_active_speakers_per_stream=2)
        # pylint: enable=protected-access

        assert len(result) == 2
        assert 0 in result[0] and 2 in result[0]
        assert 1 in result[1]

    @pytest.mark.unit
    def test_mask_features_valid(self, asr_model, diar_model, tmp_path):
        """Test mask_features with valid inputs"""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        cfg = MultitalkerTranscriptionConfig(
            manifest_file=None,
            audio_file=str(audio_path),
            fix_prev_words_count=5,
            update_prev_words_sentence=10,
            ignored_initial_frame_steps=2,
            max_num_of_spks=4,
            att_context_size=[0, 50],
            binary_diar_preds=True,
            batch_size=1,
            generate_realtime_scripts=False,
        )
        cfg = OmegaConf.structured(cfg)

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        # Create mock audio: (B=2, C=80, T=100)
        chunk_audio = torch.randn(2, 80, 100)
        # Create mask: (B=2, T=12) - will be expanded to match T=100
        mask = torch.zeros(2, 12)
        mask[0, :5] = 0.8  # First batch: first 5 frames active
        mask[1, 5:] = 0.9  # Second batch: last 7 frames active

        result = speaker_tagged_asr.mask_features(chunk_audio, mask, threshold=0.5, mask_value=-16.6355)

        assert result.shape == chunk_audio.shape
        assert result.dtype == chunk_audio.dtype
