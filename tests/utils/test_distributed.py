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

from unittest import mock

from nemo.utils.distributed import gather_objects


class TestGatherObjects:
    """Test distributed object gathering helpers."""

    @mock.patch('nemo.utils.distributed.HAVE_MEGATRON_CORE', False)
    @mock.patch('nemo.utils.distributed.dist.is_initialized', return_value=False)
    def test_returns_partial_results_if_distributed_is_not_initialized_without_megatron_core(
        self, mock_is_initialized
    ):
        """Test gather_objects returns the local results when Megatron Core is unavailable and DDP is off."""
        partial_results = ['a', 'b']

        assert gather_objects(partial_results) == partial_results
        mock_is_initialized.assert_called_once()

    @mock.patch('nemo.utils.distributed.HAVE_MEGATRON_CORE', False)
    @mock.patch('nemo.utils.distributed.dist.all_gather_object')
    @mock.patch('nemo.utils.distributed.dist.get_world_size', return_value=2)
    @mock.patch('nemo.utils.distributed.dist.get_rank', return_value=0)
    @mock.patch('nemo.utils.distributed.dist.is_initialized', return_value=True)
    def test_gathers_objects_with_torch_distributed_if_megatron_core_is_unavailable(
        self, mock_is_initialized, mock_get_rank, mock_get_world_size, mock_all_gather_object
    ):
        """Test gather_objects falls back to torch.distributed when Megatron Core is unavailable."""
        partial_results = ['a', 'b']

        def fake_all_gather_object(gathered_results, local_results):
            gathered_results[0] = local_results
            gathered_results[1] = ['c']

        mock_all_gather_object.side_effect = fake_all_gather_object

        assert gather_objects(partial_results, main_rank=0) == ['a', 'b', 'c']
        mock_is_initialized.assert_called_once()
        mock_get_rank.assert_called_once()
        mock_get_world_size.assert_called_once()
        mock_all_gather_object.assert_called_once()
