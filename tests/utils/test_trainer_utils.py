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

from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf

from nemo.utils.trainer_utils import resolve_trainer_cfg


def test_resolve_trainer_cfg_strategy():
    cfg = OmegaConf.create({"strategy": "ddp"})
    ans = resolve_trainer_cfg(cfg)
    assert isinstance(ans, dict)
    assert ans["strategy"] == "ddp"

    cfg = OmegaConf.create(
        {"strategy": {"_target_": "lightning.pytorch.strategies.DDPStrategy", "gradient_as_bucket_view": True}}
    )
    ans = resolve_trainer_cfg(cfg)
    assert isinstance(ans, dict)
    assert isinstance(ans["strategy"], DDPStrategy)
    assert "gradient_as_bucket_view" in ans["strategy"]._ddp_kwargs
    assert ans["strategy"]._ddp_kwargs["gradient_as_bucket_view"] == True
