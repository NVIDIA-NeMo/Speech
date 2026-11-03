# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo preserves the callback defaults used before the Lightning upgrade."""

from importlib import import_module

import pytest

import nemo.utils  # noqa: F401


@pytest.mark.unit
@pytest.mark.parametrize('namespace', ['lightning.pytorch', 'pytorch_lightning'])
@pytest.mark.parametrize('flags', [{}, {'enable_progress_bar': None, 'enable_model_summary': None}])
def test_default_callbacks_are_tqdm_and_plain_summary(tmp_path, namespace, flags):
    lightning = import_module(namespace)
    callbacks = import_module(f'{namespace}.callbacks')
    trainer = lightning.Trainer(default_root_dir=tmp_path, accelerator='cpu', logger=False, **flags)
    assert type(trainer.progress_bar_callback) is callbacks.TQDMProgressBar
    assert (
        type(next(cb for cb in trainer.callbacks if isinstance(cb, callbacks.ModelSummary))) is callbacks.ModelSummary
    )


@pytest.mark.unit
@pytest.mark.parametrize('namespace', ['lightning.pytorch', 'pytorch_lightning'])
@pytest.mark.parametrize('option', ['enable_progress_bar', 'enable_model_summary', 'barebones'])
def test_disabled_callbacks_stay_disabled(tmp_path, namespace, option):
    lightning = import_module(namespace)
    callbacks = import_module(f'{namespace}.callbacks')
    flags = {option: option == 'barebones'}
    trainer = lightning.Trainer(default_root_dir=tmp_path, accelerator='cpu', logger=False, **flags)
    if option in ('enable_progress_bar', 'barebones'):
        assert trainer.progress_bar_callback is None
    if option in ('enable_model_summary', 'barebones'):
        assert not any(isinstance(cb, callbacks.ModelSummary) for cb in trainer.callbacks)


@pytest.mark.unit
@pytest.mark.parametrize('namespace', ['lightning.pytorch', 'pytorch_lightning'])
@pytest.mark.parametrize('rich', [False, True])
def test_explicit_progress_bar_keeps_old_summary_selection(tmp_path, namespace, rich):
    pytest.importorskip('rich')
    lightning = import_module(namespace)
    callbacks = import_module(f'{namespace}.callbacks')
    progress = callbacks.RichProgressBar() if rich else callbacks.TQDMProgressBar(refresh_rate=7)
    trainer = lightning.Trainer(default_root_dir=tmp_path, accelerator='cpu', logger=False, callbacks=[progress])
    assert trainer.progress_bar_callback is progress
    summary = next(cb for cb in trainer.callbacks if isinstance(cb, callbacks.ModelSummary))
    assert type(summary) is (callbacks.RichModelSummary if rich else callbacks.ModelSummary)


@pytest.mark.unit
@pytest.mark.parametrize('namespace', ['lightning.pytorch', 'pytorch_lightning'])
@pytest.mark.parametrize('rich', [False, True])
def test_explicit_summary_is_retained(tmp_path, namespace, rich):
    pytest.importorskip('rich')
    lightning = import_module(namespace)
    callbacks = import_module(f'{namespace}.callbacks')
    summary = callbacks.RichModelSummary(max_depth=3) if rich else callbacks.ModelSummary(max_depth=3)
    trainer = lightning.Trainer(default_root_dir=tmp_path, accelerator='cpu', logger=False, callbacks=[summary])
    assert [cb for cb in trainer.callbacks if isinstance(cb, callbacks.ModelSummary)] == [summary]


@pytest.mark.unit
def test_callback_patch_is_idempotent():
    from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector
    from nemo.utils.lightning_logger_patch import preserve_trainer_callback_defaults

    before = _CallbackConnector._configure_progress_bar
    preserve_trainer_callback_defaults()
    preserve_trainer_callback_defaults()
    assert _CallbackConnector._configure_progress_bar is before


@pytest.mark.unit
@pytest.mark.parametrize('namespace', ['lightning.pytorch', 'pytorch_lightning'])
def test_explicit_progress_bar_validation_is_not_bypassed(tmp_path, namespace):
    lightning = import_module(namespace)
    callbacks = import_module(f'{namespace}.callbacks')
    exceptions = import_module(f'{namespace}.utilities.exceptions')
    for flags, bars in [
        ({'enable_progress_bar': False}, [callbacks.TQDMProgressBar()]),
        ({}, [callbacks.TQDMProgressBar(), callbacks.TQDMProgressBar()]),
    ]:
        with pytest.raises(exceptions.MisconfigurationException):
            lightning.Trainer(default_root_dir=tmp_path, accelerator='cpu', logger=False, callbacks=bars, **flags)
