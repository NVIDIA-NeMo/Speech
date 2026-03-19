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

import pytest
import torch

from nemo.collections.asr.models.configs.asr_models_config import CacheAwareStreamingConfig
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import MultiTalkerInstanceManager
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


class TestMultiTalkerInstanceManagerMethods:
    """Test methods of the MultiTalkerInstanceManager class"""

    @pytest.mark.unit
    def test_reset_active_speaker_buffers(self, asr_model, diar_model):
        """Test _reset_active_speaker_buffers method"""
        instance_manager = MultiTalkerInstanceManager(
            asr_model=asr_model,
            diar_model=diar_model,
            batch_size=2,
            max_num_of_spks=4,
            sent_break_sec=5.0,
        )

        # Populate some buffers first
        # pylint: disable=protected-access
        instance_manager._active_chunk_audio = [torch.randn(100)]
        instance_manager._active_chunk_lengths = [torch.tensor(100)]
        instance_manager._active_speaker_targets = [torch.randn(10)]

        # Reset the buffers
        instance_manager._reset_active_speaker_buffers()

        # Verify all buffers are empty
        assert len(instance_manager._active_chunk_audio) == 0
        assert len(instance_manager._active_chunk_lengths) == 0
        assert len(instance_manager._active_speaker_targets) == 0
        assert len(instance_manager._inactive_speaker_targets) == 0
        assert len(instance_manager._active_previous_hypotheses) == 0
        assert len(instance_manager._active_asr_pred_out_stream) == 0
        assert len(instance_manager._active_cache_last_channel) == 0
        assert len(instance_manager._active_cache_last_time) == 0
        assert len(instance_manager._active_cache_last_channel_len) == 0
        # pylint: enable=protected-access

    @pytest.mark.unit
    def test_reset_with_new_params(self, asr_model, diar_model):
        """Test reset method with new batch_size and max_num_of_spks"""
        instance_manager = MultiTalkerInstanceManager(
            asr_model=asr_model,
            diar_model=diar_model,
            batch_size=2,
            max_num_of_spks=4,
            sent_break_sec=5.0,
        )

        # Reset with new parameters
        instance_manager.reset(batch_size=3, max_num_of_spks=6)

        # Verify new parameters are applied
        assert instance_manager.batch_size == 3
        assert instance_manager.max_num_of_spks == 6
        assert len(instance_manager.batch_asr_states) == 3

    @pytest.mark.unit
    def test_add_speaker(self, asr_model, diar_model):
        """Test add_speaker method"""
        instance_manager = MultiTalkerInstanceManager(
            asr_model=asr_model,
            diar_model=diar_model,
            batch_size=2,
            max_num_of_spks=4,
            sent_break_sec=5.0,
        )
        instance_manager.reset()

        # Initially, batch 0 should have speaker [0]
        speakers_before = instance_manager.get_speakers(batch_idx=0)
        assert 0 in speakers_before

        # Add speaker 1
        instance_manager.add_speaker(batch_idx=0, speaker_id=1)

        # Verify speaker 1 is added
        speakers_after = instance_manager.get_speakers(batch_idx=0)
        assert 0 in speakers_after
        assert 1 in speakers_after

    @pytest.mark.unit
    def test_update_diar_state(self, asr_model, diar_model):
        """Test update_diar_state method"""
        instance_manager = MultiTalkerInstanceManager(
            asr_model=asr_model,
            diar_model=diar_model,
            batch_size=2,
            max_num_of_spks=4,
            sent_break_sec=5.0,
        )
        instance_manager.reset()

        # Create mock diarization data
        diar_pred_out_stream = torch.randn(2, 20, 4)
        previous_chunk_preds = torch.randn(2, 10, 4)

        # Get initial streaming state from diar_model
        diar_streaming_state = diar_model.sortformer_modules.init_streaming_state(batch_size=2)

        # Update diar state
        instance_manager.update_diar_state(
            diar_pred_out_stream=diar_pred_out_stream,
            previous_chunk_preds=previous_chunk_preds,
            diar_streaming_state=diar_streaming_state,
        )

        # Verify diar state is updated
        assert torch.equal(instance_manager.diar_states.diar_pred_out_stream, diar_pred_out_stream)
        assert torch.equal(instance_manager.diar_states.previous_chunk_preds, previous_chunk_preds)
        assert instance_manager.diar_states.streaming_state is not None

    @pytest.mark.unit
    def test_update_asr_state(self, asr_model, diar_model):
        """Test update_asr_state method"""
        instance_manager = MultiTalkerInstanceManager(
            asr_model=asr_model,
            diar_model=diar_model,
            batch_size=2,
            max_num_of_spks=4,
            sent_break_sec=5.0,
        )
        instance_manager.reset()

        # Get the initial cache state structure
        asr_state = instance_manager.batch_asr_states[0]

        # Create mock ASR cache data with correct shapes
        cache_shape = asr_state.cache_last_channel.shape
        time_shape = asr_state.cache_last_time.shape

        cache_last_channel = torch.randn(cache_shape[0], cache_shape[2])  # Remove speaker dimension
        cache_last_time = torch.randn(time_shape[0], time_shape[2])
        cache_last_channel_len = torch.tensor([10])

        # Create a simple mock hypothesis
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        previous_hypothesis = Hypothesis(score=0.0, y_sequence=[], text="test")
        previous_pred_out = torch.randn(1, 10, 128)

        # Update ASR state for batch 0, speaker 0
        instance_manager.update_asr_state(
            batch_idx=0,
            speaker_id=0,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            previous_hypotheses=previous_hypothesis,
            previous_pred_out=previous_pred_out,
        )

        # Verify the state was updated
        updated_asr_state = instance_manager.batch_asr_states[0]
        assert updated_asr_state.previous_hypothesis[0] is previous_hypothesis
        assert updated_asr_state.previous_pred_out[0] is previous_pred_out

    @pytest.mark.unit
    def test_get_active_speakers_info(self, asr_model, diar_model):
        """Test get_active_speakers_info with both empty and active speakers"""
        instance_manager = MultiTalkerInstanceManager(
            asr_model=asr_model,
            diar_model=diar_model,
            batch_size=2,
            max_num_of_spks=4,
            sent_break_sec=5.0,
        )
        instance_manager.reset()

        # Set up diar state with mock data
        previous_chunk_preds = torch.randn(2, 10, 4)
        instance_manager.diar_states.previous_chunk_preds = previous_chunk_preds

        # Test 1: No active speakers - should return None
        active_speakers_empty = [[], []]
        chunk_audio = torch.randn(2, 1600)
        chunk_lengths = torch.tensor([1600, 1600])

        result = instance_manager.get_active_speakers_info(active_speakers_empty, chunk_audio, chunk_lengths)
        assert result == (None, None, None, None)

        # Test 2: Active speakers - batch 0 has speaker 0, batch 1 has speakers 0 and 1
        active_speakers = [[0], [0, 1]]

        active_chunk_audio, active_chunk_lengths, active_speaker_targets, inactive_speaker_targets = (
            instance_manager.get_active_speakers_info(active_speakers, chunk_audio, chunk_lengths)
        )

        # Should have 3 active speakers total (1 from batch 0, 2 from batch 1)
        assert active_chunk_audio is not None
        assert active_chunk_audio.shape[0] == 3
        assert active_chunk_lengths.shape[0] == 3
        assert active_speaker_targets.shape[0] == 3
        assert inactive_speaker_targets.shape[0] == 3

    @pytest.mark.unit
    def test_update_seglsts(self, asr_model, diar_model):
        """Test update_seglsts method"""
        instance_manager = MultiTalkerInstanceManager(
            asr_model=asr_model,
            diar_model=diar_model,
            batch_size=2,
            max_num_of_spks=4,
            sent_break_sec=5.0,
        )
        instance_manager.reset()

        # Call update_seglsts (should not raise an error)
        offset = 0.0
        instance_manager.update_seglsts(offset=offset)

        # Verify seglsts are updated in each ASR state
        for asr_state in instance_manager.batch_asr_states:
            assert isinstance(asr_state.seglsts, list)
