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

from lightning.pytorch.callbacks import Callback as PTLCallback


class BaseCallback(PTLCallback):
    """Base callback ABC for NeMo lifecycle hooks (extends PTL callback).

    Implementers may override any subset of the following methods. All are
    optional no-op defaults to keep implementations lightweight.
    """

    # App lifecycle
    def on_app_start(self, *args, **kwargs) -> None:
        """Called when the application starts."""
        pass

    def on_app_end(self, *args, **kwargs) -> None:
        """Called when the application ends."""
        pass

    # Model lifecycle
    def on_model_init_start(self, *args, **kwargs) -> None:
        """Called when model initialization starts."""
        pass

    def on_model_init_end(self, *args, **kwargs) -> None:
        """Called when model initialization ends."""
        pass

    # Dataloader lifecycle
    def on_dataloader_init_start(self, *args, **kwargs) -> None:
        """Called when dataloader initialization starts."""
        pass

    def on_dataloader_init_end(self, *args, **kwargs) -> None:
        """Called when dataloader initialization ends."""
        pass

    # Optimizer lifecycle
    def on_optimizer_init_start(self, *args, **kwargs) -> None:
        """Called when optimizer initialization starts."""
        pass

    def on_optimizer_init_end(self, *args, **kwargs) -> None:
        """Called when optimizer initialization ends."""
        pass

    # Checkpoint lifecycle
    def on_load_checkpoint_start(self, *args, **kwargs) -> None:
        """Called when checkpoint loading starts."""
        pass

    def on_load_checkpoint_end(self, *args, **kwargs) -> None:
        """Called when checkpoint loading ends."""
        pass

    def on_save_checkpoint_start(self, *args, **kwargs) -> None:
        """Called when checkpoint saving starts."""
        pass

    def on_save_checkpoint_end(self, *args, **kwargs) -> None:
        """Called when checkpoint saving ends."""
        pass

    def on_save_checkpoint_success(self, *args, **kwargs) -> None:
        """Called when checkpoint saving succeeds."""
        pass

    # Configuration update
    def update_config(self, *args, **kwargs) -> None:
        """Update callback-specific configuration after initialization."""
        pass


__all__ = ["BaseCallback"]
