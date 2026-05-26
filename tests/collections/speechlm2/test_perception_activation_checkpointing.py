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

"""Unit tests for perception-module activation checkpointing.

These tests exercise:
  * ``_set_encoder_activation_checkpointing`` (the module-level helper that
    wraps ``encoder.pre_encode`` and each ``encoder.layers[i]``).
  * ``AudioPerceptionModule.set_activation_checkpointing`` and
    ``AudioTranscriptionPerceptionModule.set_activation_checkpointing`` —
    both are thin delegators, so we assert that each class exposes the method
    and that it routes through the helper.
"""

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo.collections.speechlm2.modules.perception import (
    AudioPerceptionModule,
    AudioTranscriptionPerceptionModule,
    _set_encoder_activation_checkpointing,
)


class _FakeConvSubsampling(nn.Module):
    """Stand-in for ``ConvSubsampling``/``StackingSubsampling``: non-Linear, so
    the helper should wrap it."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x, lengths):  # pragma: no cover — forward not exercised
        return x, lengths


class _FakeConformerLayer(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):  # pragma: no cover — forward not exercised
        return self.linear(x)


class _FakeConformerEncoder(nn.Module):
    """Mimics just the shape the helper inspects: ``.pre_encode`` and
    ``.layers``. Good enough to test wrapping without pulling the real
    ``ConformerEncoder``, which needs heavy deps."""

    def __init__(self, num_layers: int = 3, pre_encode: nn.Module | None = None):
        super().__init__()
        if pre_encode is None:
            pre_encode = _FakeConvSubsampling()
        self.pre_encode = pre_encode
        self.layers = nn.ModuleList([_FakeConformerLayer() for _ in range(num_layers)])


# ---------------------------------------------------------------------------
# _set_encoder_activation_checkpointing
# ---------------------------------------------------------------------------


class TestSetEncoderActivationCheckpointing:
    def test_noop_when_disabled(self):
        encoder = _FakeConformerEncoder(num_layers=3)
        original_pre_encode = encoder.pre_encode
        original_layers = [encoder.layers[i] for i in range(len(encoder.layers))]

        _set_encoder_activation_checkpointing(encoder, enabled=False)

        # Nothing should be wrapped.
        assert encoder.pre_encode is original_pre_encode
        assert not isinstance(encoder.pre_encode, CheckpointWrapper)
        for i, layer in enumerate(original_layers):
            assert encoder.layers[i] is layer
            assert not isinstance(encoder.layers[i], CheckpointWrapper)

    def test_wraps_pre_encode_and_all_layers_when_enabled(self):
        encoder = _FakeConformerEncoder(num_layers=4)

        _set_encoder_activation_checkpointing(encoder, enabled=True)

        assert isinstance(encoder.pre_encode, CheckpointWrapper)
        assert len(encoder.layers) == 4
        for layer in encoder.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_skips_linear_pre_encode(self):
        """``ConformerEncoder.forward`` dispatches on ``isinstance(pre_encode,
        nn.Linear)`` — wrapping would hide the type and route the call to the
        wrong branch. The helper must leave Linear pre_encode untouched even
        when ``enabled=True``."""
        encoder = _FakeConformerEncoder(num_layers=2, pre_encode=nn.Linear(8, 8))

        _set_encoder_activation_checkpointing(encoder, enabled=True)

        # Linear pre_encode unchanged …
        assert isinstance(encoder.pre_encode, nn.Linear)
        assert not isinstance(encoder.pre_encode, CheckpointWrapper)
        # … but layers are still wrapped.
        for layer in encoder.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_noop_when_encoder_has_no_pre_encode(self):
        """Graceful degradation: a non-Conformer encoder with ``.layers`` but
        no ``.pre_encode`` should still have its layers wrapped."""

        class EncoderNoPreEncode(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeConformerLayer() for _ in range(2)])

        encoder = EncoderNoPreEncode()
        _set_encoder_activation_checkpointing(encoder, enabled=True)
        for layer in encoder.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_noop_when_encoder_has_no_layers(self):
        """Graceful degradation: an encoder with ``.pre_encode`` but no
        ``.layers`` should still have its pre_encode wrapped."""

        class EncoderNoLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_encode = _FakeConvSubsampling()

        encoder = EncoderNoLayers()
        _set_encoder_activation_checkpointing(encoder, enabled=True)
        assert isinstance(encoder.pre_encode, CheckpointWrapper)

    def test_noop_when_encoder_has_neither(self):
        """Foreign architecture with no recognisable structure: helper is a
        no-op and does not raise."""

        class UnrelatedEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.something = nn.Linear(4, 4)

        encoder = UnrelatedEncoder()
        _set_encoder_activation_checkpointing(encoder, enabled=True)  # must not raise
        assert not isinstance(encoder.something, CheckpointWrapper)


# ---------------------------------------------------------------------------
# set_activation_checkpointing method bindings
# ---------------------------------------------------------------------------


class TestPerceptionSetActivationCheckpointingMethod:
    """Verify the method exists and routes through the helper on both
    perception classes. We don't instantiate the full module (that requires
    loading an ASR checkpoint) — we inspect the class attribute and call it
    on a lightweight fake ``self`` that exposes only the bits the method
    touches (``self.encoder``)."""

    def test_audio_perception_module_has_method(self):
        assert callable(getattr(AudioPerceptionModule, "set_activation_checkpointing", None))

    def test_audio_transcription_perception_module_has_method(self):
        assert callable(getattr(AudioTranscriptionPerceptionModule, "set_activation_checkpointing", None))

    def test_method_wraps_encoder_layers(self):
        """Invoke the unbound method against a fake ``self`` that only has
        ``self.encoder``. This is enough to verify that the method delegates
        to ``_set_encoder_activation_checkpointing`` against the encoder
        exposed via ``self.encoder``."""

        class FakeSelf:
            pass

        fake = FakeSelf()
        fake.encoder = _FakeConformerEncoder(num_layers=2)

        AudioPerceptionModule.set_activation_checkpointing(fake, enabled=True)

        assert isinstance(fake.encoder.pre_encode, CheckpointWrapper)
        for layer in fake.encoder.layers:
            assert isinstance(layer, CheckpointWrapper)

    def test_method_disabled_is_noop(self):
        class FakeSelf:
            pass

        fake = FakeSelf()
        fake.encoder = _FakeConformerEncoder(num_layers=2)
        original_pre_encode = fake.encoder.pre_encode
        original_layers = list(fake.encoder.layers)

        AudioPerceptionModule.set_activation_checkpointing(fake, enabled=False)

        assert fake.encoder.pre_encode is original_pre_encode
        for i, layer in enumerate(original_layers):
            assert fake.encoder.layers[i] is layer
