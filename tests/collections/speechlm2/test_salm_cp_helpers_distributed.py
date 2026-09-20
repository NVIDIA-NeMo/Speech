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

from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.speechlm2.parts.cp_helpers import encode_audio_with_cp_distribution


class _WorldCpMesh:
    def size(self):
        return dist.get_world_size()

    def get_group(self):
        return dist.group.WORLD


class _PackedPerception(torch.nn.Module):
    supports_sequence_packed_output = True

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0, device='cuda'))

    def forward_sequence_packed(
        self,
        *,
        input_signal,
        input_signal_length,
        input_signal_cu_seqlens=None,
        **kwargs,
    ):
        if input_signal_cu_seqlens is not None:
            offsets = input_signal_cu_seqlens.tolist()
            input_signal = torch.nn.utils.rnn.pad_sequence(
                [input_signal[offsets[row] : offsets[row + 1]] for row in range(input_signal_length.numel())],
                batch_first=True,
            )
        padded = input_signal.unsqueeze(-1) * self.scale
        return pack_encoder_output(padded, input_signal_length)


def _run_remote_gradient_test(rank: int, world_size: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=f'file://{init_file}', rank=rank, world_size=world_size)
    try:
        cases = [
            (
                [[1.0, 2.0], [3.0, 4.0]],
                [2, 2],
                lambda outputs: outputs[1 - rank].sum(),
                [3.0, 7.0],
            ),
            (
                [[1.0, 2.0, 3.0], [4.0, 0.0, 0.0], [5.0, 6.0, 0.0]],
                [3, 1, 2],
                lambda outputs: outputs[2].sum() if rank == 0 else outputs[0].sum() + outputs[1].sum(),
                [10.0, 11.0],
            ),
            (
                [[2.0, 3.0]],
                [2],
                lambda outputs: outputs[0].sum(),
                [10.0, 0.0],
            ),
            (
                [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
                [0, 3],
                lambda outputs: sum(output.sum() for output in outputs),
                [0.0, 30.0],
            ),
            (
                [[0.0, 0.0, 0.0]],
                [0],
                lambda outputs: outputs[0].sum(),
                [0.0, 0.0],
            ),
        ]
        for audio_rows, lengths, make_loss, expected_grads in cases:
            padded_audios = torch.tensor(audio_rows, device='cuda')
            audio_lens = torch.tensor(lengths, dtype=torch.long, device='cuda')
            for pack_waveforms in (False, True):
                perception = _PackedPerception()
                if pack_waveforms:
                    audios = torch.cat([audio[:length] for audio, length in zip(padded_audios, lengths)])
                    audio_cu_seqlens = torch.cat([audio_lens.new_zeros(1), audio_lens.cumsum(dim=0)])
                else:
                    audios = padded_audios
                    audio_cu_seqlens = None
                embeddings = encode_audio_with_cp_distribution(
                    perception,
                    audios,
                    audio_lens,
                    audio_cu_seqlens=audio_cu_seqlens,
                    chunk_size_seconds=None,
                    sampling_rate=16_000,
                    cp_mesh=_WorldCpMesh(),
                    sequence_packed=True,
                    packed_cp_gather=True,
                )

                assert [embedding.shape[0] for embedding in embeddings] == lengths
                for embedding, audio, length in zip(embeddings, padded_audios, lengths):
                    torch.testing.assert_close(embedding[:, 0], audio[:length] * 2.0)
                make_loss(embeddings).backward()

                grad = perception.scale.grad.detach()
                gathered_grads = [torch.zeros_like(grad) for _ in range(world_size)]
                dist.all_gather(gathered_grads, grad)
                torch.testing.assert_close(torch.stack(gathered_grads).cpu(), torch.tensor(expected_grads))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='Test requires 2 GPUs')
def test_packed_cp_gather_handles_uneven_batches_dummies_and_remote_gradients(tmp_path):
    mp.spawn(_run_remote_gradient_test, args=(2, str(tmp_path / 'cp_init')), nprocs=2, join=True)
