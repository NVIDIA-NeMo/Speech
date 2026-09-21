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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.speechlm2.parts.cp_helpers import encode_audio_with_cp_distribution


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2,), 2.0))

    def forward(self, inputs):
        return inputs.unsqueeze(-1) * self.weight.mean()


class _PackedPerception(torch.nn.Module):
    supports_sequence_packed_output = True

    def __init__(self, device):
        super().__init__()
        self.core = FSDP(_Scale().to(device), device_id=device)

    def forward_sequence_packed(self, *, input_signal, input_signal_length, **kwargs):
        return pack_encoder_output(self.core(input_signal), input_signal_length)


def _run_audio_free_chunk_test(rank: int, world_size: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        perception = _PackedPerception(device)
        if rank == 0:
            audios = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], device=device)
            lengths = torch.tensor([6], dtype=torch.long, device=device)
        else:
            audios = torch.empty(0, 6, device=device)
            lengths = torch.empty(0, dtype=torch.long, device=device)

        embeddings, dummy_loss = encode_audio_with_cp_distribution(
            perception,
            audios,
            lengths,
            chunk_size_seconds=2.0,
            chunk_batch_size=2,
            sampling_rate=1,
            cp_mesh=None,
            fsdp_sync_group=dist.group.WORLD,
            return_dummy_loss=True,
            sequence_packed=True,
        )

        if rank == 0:
            assert dummy_loss is None
            assert len(embeddings) == 1
            torch.testing.assert_close(embeddings[0][:, 0], audios[0] * 2.0)
            loss = embeddings[0].sum()
        else:
            assert embeddings == []
            assert dummy_loss is not None
            loss = dummy_loss
        loss.backward()

        grad = next(perception.core.parameters()).grad
        assert grad is not None
        gathered = [torch.zeros_like(grad) for _ in range(world_size)]
        dist.all_gather(gathered, grad)
        torch.testing.assert_close(gathered[0], gathered[1])
        assert torch.isfinite(gathered[0]).all() and gathered[0].abs().sum() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="Test requires 2 GPUs")
def test_packed_chunk_microbatches_keep_audio_free_fsdp_rank_collectives_and_backward_aligned(tmp_path):
    mp.spawn(_run_audio_free_chunk_test, args=(2, str(tmp_path / "audio_free_init")), nprocs=2, join=True)
