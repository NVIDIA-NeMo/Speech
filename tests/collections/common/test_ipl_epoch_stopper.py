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
from types import SimpleNamespace

import pytest

from nemo.collections.common.callbacks.ipl_epoch_stopper import IPLEpochStopper


class TestIPLEpochStopper:
    @pytest.mark.unit
    def test_enable_stop_false_is_inert(self):
        # Docstring contract: "If False, the callback is inert."
        callback = IPLEpochStopper(enable_stop=False, stop_every_n_epochs=1)
        trainer = SimpleNamespace(should_stop=False)

        callback.on_train_epoch_end(trainer, None)

        assert trainer.should_stop is False

    @pytest.mark.unit
    def test_enable_stop_true_requests_stop_after_n_epochs(self):
        callback = IPLEpochStopper(enable_stop=True, stop_every_n_epochs=2)
        trainer = SimpleNamespace(should_stop=False)

        callback.on_train_epoch_end(trainer, None)
        assert trainer.should_stop is False

        callback.on_train_epoch_end(trainer, None)
        assert trainer.should_stop is True
