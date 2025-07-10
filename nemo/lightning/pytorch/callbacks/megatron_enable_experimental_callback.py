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

from lightning.pytorch.callbacks.callback import Callback


class MegatronEnableExperimentalCallback(Callback):
    """
    This callback is used to enable experimental features in Megatron.

    Example:
        >>> callback = MegatronEnableExperimentalCallback()
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(self):
        super().__init__()

    def setup(self, *unused_args, **unused_kwargs) -> None:
        """
        This method is called when the callback is added to the trainer.
        """
        from megatron.core.config import set_experimental_flag

        set_experimental_flag(True)
