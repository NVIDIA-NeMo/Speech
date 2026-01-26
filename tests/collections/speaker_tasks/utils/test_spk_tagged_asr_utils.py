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

import pytest
import torch
from examples.asr.asr_cache_aware_streaming.speech_to_text_multitalker_streaming_infer import (
    MultitalkerTranscriptionConfig,
)
from omegaconf import OmegaConf

from nemo.collections.asr.models.configs.asr_models_config import CacheAwareStreamingConfig
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
    MultiTalkerInstanceManager,
    SpeakerTaggedASR,
    append_word_and_ts_seq,
    fix_frame_time_step,
    get_multi_talker_samples_from_manifest,
    get_multitoken_words,
    get_new_sentence_dict,
    get_simulated_softmax,
    get_word_dict_content_offline,
    get_word_dict_content_online,
    write_seglst,
)
from tests.collections.asr.test_asr_rnnt_encoder_model_bpe import asr_model as offline_asr_model
from tests.collections.speaker_tasks.test_diar_sortformer_models import sortformer_model as diar_model


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
        def __init__(self, diar_model, nframes):
            self.diar_model = diar_model
            self._nframes_per_chunk = nframes

    @pytest.mark.unit
    @pytest.mark.parametrize("step_num,nframes", [(0, 3), (1, 3)])
    def test_get_diar_pred_out_stream(self, diar_model, step_num, nframes):
        B, T, N = 2, 10, 4
        mats = torch.arange(B * T * N, dtype=torch.float32).reshape(B, T, N)
        # Set rttms_mask_mats on the diar_model
        diar_model.rttms_mask_mats = mats
        dummy = self.Dummy(diar_model, nframes)
        new_stream, new_chunk = SpeakerTaggedASR.get_diar_pred_out_stream(dummy, step_num)

        start = step_num * nframes
        end = start + nframes
        assert new_stream.shape[1] == min(end, T)
        assert torch.equal(new_chunk, new_stream[:, start:end])


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


class TestGetMultiTalkerSamplesFromManifest:
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
    def test_init_default_config_values(self, asr_model, diar_model, tmp_path):
        """Test initialization with default config values using .get() from MultitalkerTranscriptionConfig"""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        cfg = MultitalkerTranscriptionConfig(
            manifest_file=None,
            audio_file=str(audio_path),
            fix_prev_words_count=5,
            update_prev_words_sentence=10,
            ignored_initial_frame_steps=2,
            max_num_of_spks=3,
            att_context_size=[0, 40],
            binary_diar_preds=True,
            batch_size=1,
            sent_break_sec=30.0,  # Explicit value
            masked_asr=True,  # Explicit value
            cache_gating=False,  # Explicit value
            mask_preencode=False,  # Explicit value
            single_speaker_mode=False,  # Explicit value
            generate_realtime_scripts=False,
        )
        # Convert to OmegaConf to support .get() method
        cfg = OmegaConf.structured(cfg)

        speaker_tagged_asr = SpeakerTaggedASR(cfg, asr_model, diar_model)

        # Verify values from .get() calls are properly set
        # pylint: disable=protected-access
        assert speaker_tagged_asr._max_num_of_spks == 3  # From cfg.get("max_num_of_spks", 4)
        assert speaker_tagged_asr._sent_break_sec == 30.0  # From cfg
        # cache_gating and cache_gating_buffer_size use defaults via cfg.get() since they're not in MultitalkerTranscriptionConfig
        assert speaker_tagged_asr._cache_gating is False  # Default value from cfg.get("cache_gating", False)
        assert (
            speaker_tagged_asr._cache_gating_buffer_size == 2
        )  # Default value from cfg.get("cache_gating_buffer_size", 2)
        assert speaker_tagged_asr._masked_asr is True  # From cfg
        assert speaker_tagged_asr._use_mask_preencode is False  # From cfg
        assert speaker_tagged_asr._single_speaker_mode is False  # From cfg
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
