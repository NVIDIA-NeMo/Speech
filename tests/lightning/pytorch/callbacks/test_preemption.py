# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import signal
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from pytorch_lightning import Trainer

from nemo.lightning.pytorch.callbacks.preemption import PreemptionCallback


class TestPreemptionCallback:

    @pytest.fixture
    def callback(self):
        return PreemptionCallback()

    @pytest.fixture
    def mock_trainer(self):
        trainer = MagicMock(spec=Trainer)
        trainer.should_stop = False
        return trainer

    def test_init(self, callback):
        assert callback.sig == signal.SIGTERM
        assert not callback._interrupted
        assert callback._handler_context is None

    def test_custom_signal(self):
        custom_callback = PreemptionCallback(sig=signal.SIGUSR1)
        assert custom_callback.sig == signal.SIGUSR1

    @pytest.mark.parametrize("initially_supported,becomes_supported", [(False, True), (False, False), (True, True)])
    def test_on_train_batch_start_distributed_init(
        self, callback, mock_trainer, initially_supported, becomes_supported
    ):
        with (
            patch.object(PreemptionCallback, '_check_preemption_support') as mock_check,
            patch.object(callback, '_preemption_handler') as mock_handler,
        ):

            mock_check.side_effect = [initially_supported, becomes_supported]

            callback.on_train_start(mock_trainer, None)
            callback.on_train_batch_start(mock_trainer, None, None, 0)

            expected_call_count = 1 if initially_supported else (1 if becomes_supported else 0)
            assert mock_handler.call_count == expected_call_count

            if initially_supported:
                mock_handler.assert_called_once_with()
            elif becomes_supported:
                mock_handler.assert_called_once_with()
            else:
                mock_handler.assert_not_called()

    @pytest.mark.parametrize(
        "is_supported,interrupted,expected",
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_interrupted_property(self, callback, is_supported, interrupted, expected):
        with (
            patch.object(PreemptionCallback, '_check_preemption_support', return_value=is_supported),
            patch('torch.distributed.broadcast'),
            patch('torch.tensor', return_value=torch.tensor(interrupted)),
            patch('torch.cuda.is_available', return_value=True),
            patch('torch.cuda.current_device', return_value=0),
        ):
            callback._interrupted = interrupted
            assert callback.interrupted == expected

    def test_on_train_start(self, callback, mock_trainer):
        with (
            patch.object(PreemptionCallback, 'preemption_supported', new_callable=PropertyMock) as mock_supported,
            patch.object(callback, '_preemption_handler') as mock_handler,
        ):

            # Test when preemption is supported
            mock_supported.return_value = True
            callback.on_train_start(mock_trainer, None)
            mock_handler.assert_called_once()
            mock_handler.reset_mock()

            # Test when preemption is not supported
            mock_supported.return_value = False
            callback.on_train_start(mock_trainer, None)
            mock_handler.assert_not_called()

    def test_on_train_end(self, callback, mock_trainer):
        mock_context = MagicMock()
        callback._handler_context = mock_context
        callback.on_train_end(mock_trainer, None)
        mock_context.__exit__.assert_called_once_with(None, None, None)

    @pytest.mark.parametrize("interrupted", [True, False])
    def test_on_train_batch_end(self, callback, mock_trainer, interrupted):
        with patch.object(PreemptionCallback, 'interrupted', new_callable=lambda: property(lambda self: interrupted)):
            if interrupted:
                with pytest.raises(SystemExit):
                    callback.on_train_batch_end(mock_trainer, None, None, None, 0)
            else:
                callback.on_train_batch_end(mock_trainer, None, None, None, 0)
            assert mock_trainer.should_stop == interrupted
