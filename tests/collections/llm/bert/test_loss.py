# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm.bert.loss import (
    BERTInBatchExclusiveHardNegativesRankingLoss,
    BERTLossReduction,
    HardNegativeRankingLoss,
    sentence_order_prediction_loss,
)


def mock_average_losses(losses):
    """Mock function to average losses without distributed operations"""
    averaged_losses = torch.stack([loss.clone().detach() for loss in losses])
    return averaged_losses.mean().repeat(len(losses))


@pytest.fixture(autouse=True)
def mock_distributed(mocker):
    # Mock parallel state functions
    mocker.patch('megatron.core.parallel_state.get_context_parallel_world_size', return_value=1)
    mocker.patch('megatron.core.parallel_state.get_data_parallel_world_size', return_value=1)
    mocker.patch('megatron.core.parallel_state.get_data_parallel_rank', return_value=0)
    mocker.patch('megatron.core.parallel_state.get_data_parallel_group', return_value=None)

    # Mock the average_losses function
    mocker.patch(
        'nemo.collections.llm.bert.loss.average_losses_across_data_parallel_group', side_effect=mock_average_losses
    )


def test_hard_negative_ranking_loss():
    loss_fn = HardNegativeRankingLoss(num_hard_negatives=2)
    batch_size = 4
    embed_dim = 128

    forward_out = torch.randn(batch_size * 4, embed_dim)
    batch = {}

    loss, stats = loss_fn.forward(batch=batch, forward_out=forward_out)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert 'avg' in stats

    # Test that averaging works as expected
    assert torch.allclose(stats['avg'], loss.detach())


def test_average_losses_mock():
    """Test to verify the mocked averaging function works as expected"""
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    averaged = mock_average_losses(losses)
    expected_mean = torch.tensor(2.0)  # (1 + 2 + 3) / 3

    assert torch.allclose(averaged, expected_mean.repeat(3))
    assert averaged.shape == torch.Size([3])  # Should return same length as input


def test_bert_in_batch_negatives_loss():
    loss_fn = BERTInBatchExclusiveHardNegativesRankingLoss(num_hard_negatives=1, global_in_batch_negatives=False)
    batch_size = 4
    embed_dim = 128

    forward_out = torch.randn(batch_size * 3, embed_dim)
    batch = {}

    loss, stats = loss_fn.forward(batch=batch, forward_out=forward_out)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert 'avg' in stats

    # Test that averaging works as expected
    assert torch.allclose(stats['avg'], loss.detach())


def test_bert_loss_with_sop():
    """Test BERTLossReduction with SOP loss enabled (lines 32-38)"""
    loss_fn = BERTLossReduction(validation_step=False, val_drop_last=True, add_sop_loss=True)
    assert loss_fn.validation_step == False
    assert loss_fn.val_drop_last == True
    assert loss_fn.add_sop_loss == True
    assert not hasattr(loss_fn, 'mlm')  # Should not create mlm when add_sop_loss is True


def test_bert_loss_forward_with_sop(mocker):
    """Test BERTLossReduction forward with SOP loss (lines 50-71)"""
    loss_fn = BERTLossReduction(add_sop_loss=True)

    # Create test inputs
    batch = {'loss_mask': torch.ones(2, 4), 'is_random': torch.tensor([0, 1])}

    forward_out = {'lm_loss': torch.randn(2, 4), 'loss_mask': torch.ones(2, 4), 'binary_logits': torch.randn(2, 2)}

    # Test with CP size = 1
    mocker.patch('megatron.core.parallel_state.get_context_parallel_world_size', return_value=1)
    loss, stats = loss_fn.forward(batch=batch, forward_out=forward_out)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(stats, dict)
    assert 'avg' in stats

    # Test with CP size > 1
    mocker.patch('megatron.core.parallel_state.get_context_parallel_world_size', return_value=2)
    with pytest.raises(NotImplementedError, match='CP is not supported for SOP loss yet'):
        loss_fn.forward(batch=batch, forward_out=forward_out)


def test_bert_in_batch_negatives_init():
    """Test BERTInBatchExclusiveHardNegativesRankingLoss initialization (lines 213-228)"""
    loss_fn = BERTInBatchExclusiveHardNegativesRankingLoss(
        validation_step=True,
        val_drop_last=False,
        num_hard_negatives=2,
        scale=30.0,
        label_smoothing=0.1,
        global_in_batch_negatives=True,
        backprop_type='global',
    )

    assert loss_fn.validation_step == True
    assert loss_fn.val_drop_last == False
    assert loss_fn.num_hard_negatives == 2
    assert loss_fn.scale == 30.0
    assert loss_fn.global_in_batch_negatives == True
    assert loss_fn.backprop_type == 'global'


def test_bert_in_batch_forward_validation():
    """Test BERTInBatchExclusiveHardNegativesRankingLoss forward validation (lines 278-298)"""
    loss_fn = BERTInBatchExclusiveHardNegativesRankingLoss(
        num_hard_negatives=1, global_in_batch_negatives=True, validation_step=True
    )

    batch_size = 4
    embed_dim = 8
    forward_out = torch.randn(batch_size * 3, embed_dim)
    batch = {}

    loss, stats = loss_fn.forward(batch=batch, forward_out=forward_out)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(stats, dict)
    assert 'avg' in stats


@pytest.mark.parametrize(
    "tensor_input,expected",
    [
        (torch.tensor([[0.5, 0.5], [0.2, 0.8]]), torch.tensor(0.0)),
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor(1.0)),
    ],
)
def test_sentence_order_prediction(tensor_input, expected):
    """Test sentence_order_prediction_loss (lines 307-314)"""
    sentence_order = torch.tensor([0, 1])
    loss = sentence_order_prediction_loss(tensor_input, sentence_order)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
