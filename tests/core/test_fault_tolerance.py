# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
import pytest

from nemo.utils.exp_manager import exp_manager

try:
    from ptl_resiliency import FaultToleranceCallback

    HAVE_FT = True
except (ImportError, ModuleNotFoundError):
    HAVE_FT = False


@pytest.mark.skipif(not HAVE_FT, reason="requires resiliency package to be installed.")
class TestFaultTolerance:

    @pytest.mark.unit
    def test_fault_tol_callback_not_created_by_default(self):
        """There should be no FT callback by default"""
        test_conf = {"create_tensorboard_logger": False, "create_checkpoint_callback": False}
        test_trainer = pl.Trainer(accelerator='cpu')
        ft_callback_found = None
        exp_manager(test_trainer, test_conf)
        for cb in test_trainer.callbacks:
            if isinstance(cb, FaultToleranceCallback):
                ft_callback_found = cb
        assert ft_callback_found is None

    @pytest.mark.unit
    def test_fault_tol_callback_created(self):
        """Verify that fault tolerance callback is created"""
        try:
            os.environ['FAULT_TOL_CFG_PATH'] = "/tmp/dummy"
            test_conf = {
                "create_tensorboard_logger": False,
                "create_checkpoint_callback": False,
                "create_fault_tolerance_callback": True,
            }
            test_trainer = pl.Trainer(accelerator='cpu')
            ft_callback_found = None
            exp_manager(test_trainer, test_conf)
            for cb in test_trainer.callbacks:
                if isinstance(cb, FaultToleranceCallback):
                    ft_callback_found = cb
            assert ft_callback_found is not None
        finally:
            del os.environ['FAULT_TOL_CFG_PATH']
