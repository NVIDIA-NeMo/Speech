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
"""CPU-only tests for the CP-helper module.

The ``cp_size > 1`` path in ``encode_audio_with_cp_distribution`` requires
a real ``torch.distributed`` process group; it's exercised by the 2-GPU
smoke. These tests cover the fallback contracts that run on every machine
(``cp_mesh is None``, ``B_aud == 0``).
"""
import pytest
import torch

from nemo.collections.speechlm2.parts.cp_helpers import encode_audio_with_cp_distribution, get_cp_mesh


def test_get_cp_mesh_none():
    assert get_cp_mesh(None) == (None, 1, 0)


class _DummyCpDim:
    """Stand-in for ``device_mesh['cp']`` whose ``.size()`` is 1 (CP inactive)."""

    def size(self):
        return 1


class _DummyDeviceMesh:
    """Minimal ``DeviceMesh``-like object exposing only the bits ``get_cp_mesh`` reads."""

    def __init__(self, cp_size: int = 1, has_cp: bool = True):
        self.mesh_dim_names = ("dp", "cp", "tp") if has_cp else ("dp", "tp")
        self._cp_size = cp_size

    def __getitem__(self, key):
        if key == "cp":

            class _Dim:
                def __init__(self, size):
                    self._size = size

                def size(self):
                    return self._size

            return _Dim(self._cp_size)
        raise KeyError(key)


def test_get_cp_mesh_cp_size_one():
    assert get_cp_mesh(_DummyDeviceMesh(cp_size=1)) == (None, 1, 0)


def test_get_cp_mesh_no_cp_dim():
    assert get_cp_mesh(_DummyDeviceMesh(has_cp=False)) == (None, 1, 0)


class _PerceptionStub:
    """Stand-in for ``self.perception``: returns a deterministic embedding per audio."""

    def __init__(self, hidden_size: int = 4):
        self.hidden_size = hidden_size

    def __call__(self, *, input_signal, input_signal_length):
        # Pretend each audio of length L produces L // 2 frames of embeddings;
        # encode the row index into the first column so we can verify ordering.
        B, T = input_signal.shape
        if B == 0:
            return torch.zeros(0, 0, self.hidden_size, dtype=torch.float32), input_signal_length
        # Frame count per row scales with audio_lens.
        out_lens = (input_signal_length // 2).clamp(min=1)
        max_out = int(out_lens.max().item())
        embs = torch.zeros(B, max_out, self.hidden_size, dtype=torch.float32)
        for i in range(B):
            embs[i, : int(out_lens[i].item()), 0] = float(i)  # marker
        return embs, out_lens


def test_encode_audio_no_cp_returns_unpadded_list():
    perception = _PerceptionStub(hidden_size=4)
    audios = torch.zeros(3, 1600, dtype=torch.float32)
    audio_lens = torch.tensor([800, 1200, 1600], dtype=torch.long)
    embs = encode_audio_with_cp_distribution(
        perception,
        audios,
        audio_lens,
        chunk_size_seconds=None,
        sampling_rate=16000,
        cp_mesh=None,
    )
    # 3 audios → 3 embedding tensors with row-specific lengths.
    assert len(embs) == 3
    expected_lens = [400, 600, 800]
    for i, e in enumerate(embs):
        assert e.shape == (expected_lens[i], 4)
        # Marker preserved.
        assert torch.all(e[:, 0] == float(i))


def test_encode_audio_empty_batch_returns_empty():
    perception = _PerceptionStub()
    audios = torch.zeros(0, 1600, dtype=torch.float32)
    audio_lens = torch.zeros(0, dtype=torch.long)
    embs = encode_audio_with_cp_distribution(
        perception,
        audios,
        audio_lens,
        chunk_size_seconds=None,
        sampling_rate=16000,
        cp_mesh=None,
    )
    assert embs == []


class _FakeCpMesh:
    def size(self):
        return 2

    def get_group(self):
        return "fake-cp-group"


class _TrainablePerceptionStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, *, input_signal, input_signal_length):
        B = input_signal.shape[0]
        embs = input_signal[:, :2].unsqueeze(-1) * self.scale
        lens = torch.full((B,), 2, dtype=input_signal_length.dtype, device=input_signal_length.device)
        return embs, lens


def test_encode_audio_cp_distribution_preserves_local_autograd(monkeypatch):
    perception = _TrainablePerceptionStub()
    audios = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    audio_lens = torch.tensor([3, 3], dtype=torch.long)

    def fake_all_gather(local_stack, group):
        assert group == "fake-cp-group"
        remote_stack = torch.zeros_like(local_stack)
        return (local_stack, remote_stack)

    def fake_lens_all_gather(gathered_lens, local_lens, group):
        assert group == "fake-cp-group"
        gathered_lens[0].copy_(local_lens)
        gathered_lens[1].fill_(2)

    monkeypatch.setattr("nemo.collections.speechlm2.parts.cp_helpers.dist.get_rank", lambda group: 0)
    monkeypatch.setattr("nemo.collections.speechlm2.parts.cp_helpers.dist.all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr("nemo.collections.speechlm2.parts.cp_helpers.dist.all_gather", fake_lens_all_gather)
    monkeypatch.setattr("nemo.collections.speechlm2.parts.cp_helpers.differentiable_all_gather", fake_all_gather)

    embs = encode_audio_with_cp_distribution(
        perception,
        audios,
        audio_lens,
        chunk_size_seconds=None,
        sampling_rate=16000,
        cp_mesh=_FakeCpMesh(),
    )

    assert embs[0].requires_grad
    embs[0].sum().backward()
    assert perception.scale.grad is not None
    assert perception.scale.grad.item() == pytest.approx(3.0)
