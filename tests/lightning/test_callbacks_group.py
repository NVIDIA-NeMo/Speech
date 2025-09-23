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

import importlib
from unittest.mock import MagicMock

from lightning.pytorch.callbacks import Callback as PTLCallback

from nemo.lightning.base_callback import BaseCallback


def _fresh_group_module():
    """Reset the CallbackGroup singleton and stub OneLoggerNeMoCallback safely.

    This avoids deleting modules from sys.modules. We import the module,
    replace the OneLoggerNeMoCallback symbol with a lightweight stub,
    and reset the internal singleton so a new instance is built.
    """
    mod = importlib.import_module('nemo.lightning.callback_group')

    class _StubOneLoggerCallback(BaseCallback):
        def __init__(self, *args, **kwargs):
            pass

        def update_config(self, *args, **kwargs):
            pass

    setattr(mod, 'OneLoggerNeMoCallback', _StubOneLoggerCallback)
    # Reset the singleton so the next get_instance() uses the stubbed class
    mod.CallbackGroup._instance = None
    return mod


def test_base_callback_noops_do_not_raise():
    """Test BaseCallback hooks are no-ops and do not raise exceptions."""
    cb = BaseCallback()

    cb.on_app_start()
    cb.on_app_end()
    cb.on_model_init_start()
    cb.on_model_init_end()
    cb.on_dataloader_init_start()
    cb.on_dataloader_init_end()
    cb.on_optimizer_init_start()
    cb.on_optimizer_init_end()
    cb.on_load_checkpoint_start()
    cb.on_load_checkpoint_end()
    cb.on_save_checkpoint_start()
    cb.on_save_checkpoint_end()
    cb.on_save_checkpoint_success()
    cb.update_config()


def test_base_callback_is_ptl_callback():
    """Test BaseCallback derives from Lightning PTL Callback."""
    assert isinstance(BaseCallback(), PTLCallback)


def test_callback_group_singleton_identity():
    """Test CallbackGroup returns the same singleton instance."""
    mod = _fresh_group_module()
    a = mod.CallbackGroup.get_instance()
    b = mod.CallbackGroup.get_instance()
    assert a is b


def test_callback_group_update_config_fanout_and_attach(monkeypatch):
    """Test update_config fans out to callbacks and attaches them to trainer."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    class _StubCallback(BaseCallback):
        def __init__(self):
            self.called = False
            self.kwargs = None

        def update_config(self, *args, **kwargs):
            self.called = True
            self.kwargs = kwargs

    stub_cb = _StubCallback()
    group._callbacks = [stub_cb]

    class Trainer:
        def __init__(self):
            self.callbacks = []

    trainer = Trainer()
    marker = object()
    group.update_config('v2', trainer, data=marker)

    assert stub_cb.called
    kwargs = stub_cb.kwargs
    assert kwargs['nemo_version'] == 'v2'
    assert kwargs['trainer'] is trainer
    assert kwargs['data'] is marker
    assert trainer.callbacks[0] is stub_cb


def test_callback_group_dynamic_dispatch_calls_when_present():
    """Test dynamic dispatch calls methods when present on callbacks."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    mock_cb = MagicMock()
    group._callbacks = [mock_cb]

    group.on_app_start()
    assert mock_cb.on_app_start.called


def test_callback_group_dynamic_dispatch_ignores_missing_methods():
    """Test dynamic dispatch ignores missing methods without raising."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    class Dummy:
        pass

    group._callbacks = [Dummy()]

    # Should not raise even if method not present
    group.on_nonexistent_method()


def test_hook_class_init_with_callbacks_wraps_and_emits(monkeypatch):
    """Test inheritance-based hook via __init_subclass__ emits start/end once (e2e-style)."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    start = MagicMock()
    end = MagicMock()

    monkeypatch.setattr(group, 'on_model_init_start', start)
    monkeypatch.setattr(group, 'on_model_init_end', end)

    class Base:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            # Mirror IOMixin: hook subclasses at definition time
            mod.hook_class_init_with_callbacks(cls, 'on_model_init_start', 'on_model_init_end')

    class Child(Base):
        def __init__(self):
            self.x = 1

    class GrandChild(Child):
        def __init__(self):
            self.y = 2
            super().__init__()

    c = Child()
    assert c.x == 1
    # Flag indicating wrapping applied on the subclass
    assert getattr(Child.__init__, '_init_wrapped_for_callbacks', False) is True

    d = GrandChild()
    assert d.x == 1
    assert d.y == 2

    assert start.call_count == 2
    assert end.call_count == 2
    # Flag indicating wrapping applied on the subclass
    assert getattr(GrandChild.__init__, '_init_wrapped_for_callbacks', False) is True


def test_hook_class_init_with_callbacks_idempotent():
    """Test inheritance-based hook is idempotent and does not re-wrap on repeated calls."""
    mod = _fresh_group_module()

    class Base:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            mod.hook_class_init_with_callbacks(cls, 'on_model_init_start', 'on_model_init_end')

    class Child(Base):
        def __init__(self):
            pass

    # Hook was applied via __init_subclass__ at class creation time
    first = Child.__init__
    # Attempt to apply again explicitly; should be a no-op
    mod.hook_class_init_with_callbacks(Child, 'on_model_init_start', 'on_model_init_end')
    second = Child.__init__
    assert first is second


def test_on_app_end_is_idempotent(monkeypatch):
    """Test on_app_end fans out only once even if called multiple times."""
    mod = _fresh_group_module()
    group = mod.CallbackGroup.get_instance()

    mock_cb = MagicMock()
    group._callbacks = [mock_cb]

    group.on_app_end()
    group.on_app_end()

    assert mock_cb.on_app_end.call_count == 1
