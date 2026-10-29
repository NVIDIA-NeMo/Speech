# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from tests.collections.tts.models.test_easy_magpietts_attention_alignment import (
    _attention_batch,
    _config,
    _make_attention_model,
    _step,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    'start,step,expected',
    [
        (0, 0, (1.0, 1.0)),
        (0, 1, (1.0, 0.9998)),
        (0, 2500, (1.0, 0.5)),
        (0, 5000, (1.0, 0.0)),
        (0, 6000, (1.0, 0.0)),
        (100, 0, (0.0, 0.0)),
        (100, 99, (0.0, 0.0)),
        (100, 100, (1.0, 1.0)),
        (100, 101, (1.0, 0.9998)),
        (100, 2600, (1.0, 0.5)),
        (100, 5100, (1.0, 0.0)),
    ],
)
def test_zero_ramp_enables_full_guidance_at_start_and_anneals_only_the_prior(start, step, expected):
    model = _schedule_holder(step, start=start, ramp=0)
    assert EasyMagpieTTSModel._attention_alignment_schedule(model) == pytest.approx(expected)


@pytest.mark.parametrize(
    'step,expected',
    [(0, (0.0, 0.0)), (10000, (0.0, 0.0)), (10500, (0.5, 0.5)), (11000, (1.0, 1.0)), (16000, (1.0, 0.0))],
)
def test_existing_positive_ramp_schedule_is_preserved(step, expected):
    model = _schedule_holder(step, start=10000, ramp=1000)
    assert EasyMagpieTTSModel._attention_alignment_schedule(model) == pytest.approx(expected)


def test_zero_ramp_respects_the_configured_prior_strength():
    model = _schedule_holder(0, start=0, ramp=0, strength=0.4)
    assert EasyMagpieTTSModel._attention_alignment_schedule(model) == pytest.approx((1.0, 0.4))


def test_real_model_guides_attention_on_its_first_step_without_enabling_the_repeated_head():
    model = _make_attention_model(
        _config(
            extra={
                # This setting controls the separate head, so it must not delay this experiment.
                'start_aligner_text_head_steps': 10000,
                'attention_alignment': {'start_step': 0, 'ramp_steps': 0},
            }
        )
    )
    with torch.no_grad(), _step(model, 0), patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run:
        output = model.process_batch(**_attention_batch(model))

    request = run.call_args.kwargs['attention_alignment']
    assert request is not None
    assert request.prior_strength == 1.0
    assert output.attention_alignment_metrics['loss_scale'] == 1.0
    assert output.attention_alignment_metrics['prior_strength'] == 1.0
    assert output.attention_alignment_loss > 0
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.aligner_loss)
    assert output.aligned_text_loss is None
    assert not hasattr(model, 'aligned_text_head')


def _schedule_holder(step, *, start, ramp, strength=1.0):
    return SimpleNamespace(
        global_step=step,
        attention_alignment_config={
            'start_step': start,
            'ramp_steps': ramp,
            'prior_anneal_steps': 5000,
            'prior_strength': strength,
        },
    )
