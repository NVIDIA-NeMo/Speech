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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.speechlm2.models.salm_automodel import _fully_shard_perception
from nemo.collections.speechlm2.modules.perception import AudioPerceptionModule, IdentityConnector
from nemo.collections.speechlm2.parts.cp_helpers import encode_audio_with_cp_distribution
from tests.collections.asr.test_parallel_expert_encoder_two_branch import build_toy_packed_pe_encoder


class _FeaturePassthrough(torch.nn.Module):
    def forward(self, input_signal, length):
        return input_signal, length


class _RepeatFeaturePreprocessor(torch.nn.Module):
    def forward(self, input_signal, length):
        return input_signal.unsqueeze(1).expand(-1, 128, -1).contiguous(), length


class _WorldCpMesh:
    def size(self):
        return dist.get_world_size()

    def get_group(self):
        return dist.group.WORLD


class _ScalePerception(torch.nn.Module):
    supports_sequence_packed_output = True

    def __init__(self, device):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([2.0], device=device))

    def forward_sequence_packed(self, *, input_signal, input_signal_length, **kwargs):
        return pack_encoder_output(input_signal.unsqueeze(-1) * self.scale, input_signal_length)


def _make_perception(device) -> AudioPerceptionModule:
    encoder = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        subsampling_factor=1,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        self_attention_model="rope",
        qk_norm=True,
        sync_max_audio_length=False,
    ).to(device)
    perception = AudioPerceptionModule.__new__(AudioPerceptionModule)
    torch.nn.Module.__init__(perception)
    perception.preprocessor = _FeaturePassthrough()
    perception._modules["encoder"] = encoder
    perception.modality_adapter = IdentityConnector()
    perception.proj = torch.nn.Linear(32, 24, device=device)
    perception.spec_augmentation = None
    perception.rote = None
    return perception.train()


def _make_pee_perception(device) -> AudioPerceptionModule:
    encoder = build_toy_packed_pe_encoder().to(device=device, dtype=torch.bfloat16)
    perception = AudioPerceptionModule.__new__(AudioPerceptionModule)
    torch.nn.Module.__init__(perception)
    perception.preprocessor = _RepeatFeaturePreprocessor()
    perception._modules['encoder'] = encoder
    perception.modality_adapter = IdentityConnector()
    perception.proj = torch.nn.Linear(encoder.d_model, 24, device=device, dtype=torch.bfloat16)
    perception.spec_augmentation = None
    perception.rote = None
    return perception.train()


def _run_fsdp2_packed_perception_test(rank: int, world_size: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

        # Exercise the production custom perception entry point. Rank 1 owns no
        # valid tokens, while rank 0 has audio; both ranks must reach identical
        # FSDP2 collectives and materialize gradients for every sharded parameter.
        perception = _fully_shard_perception(_make_perception(device), mesh)
        optimizer = torch.optim.AdamW(perception.parameters(), lr=1e-2)
        optimizer_parameters = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
        projection_parameter = perception.proj.weight
        projection_before = projection_parameter.detach().full_tensor().clone()
        if rank == 0:
            features = torch.randn(1, 8, 12, device=device, requires_grad=True)
            lengths = torch.tensor([12], device=device)
        else:
            features = torch.empty(1, 8, 0, device=device, requires_grad=True)
            lengths = torch.tensor([0], device=device)
        packed = perception.forward_sequence_packed(input_signal=features, input_signal_length=lengths)
        packed.data.float().sum().backward()

        assert features.grad is not None
        assert all(parameter.grad is not None for parameter in optimizer_parameters)
        optimizer.step()
        projection_after = projection_parameter.detach().full_tensor()
        assert not torch.equal(projection_before, projection_after)
        assert optimizer.state[projection_parameter]["step"] == 1

        # A second step gives every rank an all-empty batch, covering the zero-token
        # collective case independently of the uneven-rank step above.
        optimizer.zero_grad(set_to_none=True)
        empty_features = torch.empty(1, 8, 0, device=device, requires_grad=True)
        empty_lengths = torch.tensor([0], device=device)
        empty = perception.forward_sequence_packed(
            input_signal=empty_features,
            input_signal_length=empty_lengths,
        )
        empty.data.sum().backward()
        assert empty_features.grad is not None
        assert all(parameter.grad is not None for parameter in optimizer_parameters)

        # Exercise the CP distribution/gather path with an FSDP2-sharded custom
        # packed method. B=1 forces one CP rank to encode a dummy row.
        scaled = _fully_shard_perception(_ScalePerception(device), mesh)
        audios = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        audio_lens = torch.tensor([3], device=device)
        embeddings = encode_audio_with_cp_distribution(
            scaled,
            audios,
            audio_lens,
            chunk_size_seconds=None,
            sampling_rate=1,
            cp_mesh=_WorldCpMesh(),
            fsdp_sync_group=dist.group.WORLD,
            sequence_packed=True,
            packed_cp_gather=True,
        )
        assert len(embeddings) == 1
        torch.testing.assert_close(embeddings[0][:, 0], audios[0] * 2.0)
        embeddings[0].sum().backward()
        assert all(parameter.grad is not None for parameter in scaled.parameters())
    finally:
        dist.destroy_process_group()


def _run_fsdp2_canonical_pee_test(rank: int, world_size: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        perception = _fully_shard_perception(_make_pee_perception(device), mesh)

        frames = 32 if rank == 0 else 16
        raw_audio = torch.randn(1, frames, device=device, dtype=torch.bfloat16, requires_grad=True)
        lengths = torch.tensor([frames if rank == 0 else 0], device=device)
        packed = perception.forward_sequence_packed(input_signal=raw_audio, input_signal_length=lengths)
        packed.data.float().sum().backward()

        assert raw_audio.grad is not None and torch.isfinite(raw_audio.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in perception.parameters()
            if parameter.requires_grad
        )
        perception.zero_grad(set_to_none=True)
        empty_audio = torch.randn(1, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        empty = perception.forward_sequence_packed(
            input_signal=empty_audio,
            input_signal_length=torch.tensor([0], device=device),
        )
        empty.data.sum().backward()
        # With no valid samples, the packed feature stacker has no reason to retain
        # an autograd edge to the waveform. The distributed invariant is that every
        # trainable parameter still receives a zero gradient and reaches the same
        # FSDP collectives on every rank.
        assert all(parameter.grad is not None for parameter in perception.parameters() if parameter.requires_grad)

        perception.zero_grad(set_to_none=True)
        cp_audio = torch.randn(1, 24, device=device, dtype=torch.bfloat16)
        embeddings = encode_audio_with_cp_distribution(
            perception,
            cp_audio,
            torch.tensor([24], device=device),
            chunk_size_seconds=None,
            sampling_rate=1,
            cp_mesh=_WorldCpMesh(),
            fsdp_sync_group=dist.group.WORLD,
            sequence_packed=True,
            packed_cp_gather=True,
        )
        assert len(embeddings) == 1 and embeddings[0].shape[-1] == 24
        assert torch.isfinite(embeddings[0]).all()
        embeddings[0].float().sum().backward()
        assert all(parameter.grad is not None for parameter in perception.parameters() if parameter.requires_grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="Test requires 2 GPUs")
def test_canonical_pee_packed_perception_fsdp2_empty_rank_all_empty_and_cp(tmp_path):
    mp.spawn(
        _run_fsdp2_canonical_pee_test,
        args=(2, str(tmp_path / "canonical_pee_fsdp2_init")),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="Test requires 2 GPUs")
def test_packed_perception_fsdp2_custom_forward_empty_rank_and_cp_gather(tmp_path):
    mp.spawn(_run_fsdp2_packed_perception_test, args=(2, str(tmp_path / "fsdp2_init")), nprocs=2, join=True)
