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

from nemo import lightning as nl


class TestFabricConversion:
    def test_simple_conversion(self):
        trainer = nl.Trainer(
            devices=1,
            accelerator="cpu",
            strategy=nl.MegatronStrategy(tensor_model_parallel_size=2),
            plugins=nl.MegatronMixedPrecision(precision='16-mixed'),
        )

        fabric = trainer.to_fabric()

        assert isinstance(fabric.strategy, nl.FabricMegatronStrategy)
        assert fabric.strategy.tensor_model_parallel_size == 2
        assert isinstance(fabric._precision, nl.FabricMegatronMixedPrecision)
        assert fabric._precision.precision == '16-mixed'
