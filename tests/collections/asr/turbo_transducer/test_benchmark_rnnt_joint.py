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

from tests.collections.asr.turbo_transducer import benchmark_rnnt_joint, benchmark_rnnt_joint_multi


@pytest.mark.unit
@pytest.mark.parametrize("name", ["bf16-mixed", "bfloat16-mixed"])
def test_mixed_precision_dtype_aliases(name):
    assert benchmark_rnnt_joint.get_dtype(name) == "bfloat16-mixed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "joint,fused_batch_size", [("standard", 0), ("standard", 1), ("fused_vocab", 0), ("fused", 0)]
)
def test_mixed_precision_benchmark_preserves_master_weights(monkeypatch, joint, fused_batch_size):
    calls = []

    def check_backward(gradient):
        assert not torch.is_autocast_enabled("cuda")
        return gradient

    def record_inputs(hidden, weight, bias):
        assert torch.is_autocast_enabled("cuda")
        assert hidden.dtype == torch.bfloat16
        assert weight.dtype == bias.dtype == torch.float32
        weight.register_hook(check_backward)
        calls.append(True)

    original_linear = torch.nn.functional.linear

    def record_linear(hidden, weight, bias=None):
        record_inputs(hidden, weight, bias)
        return original_linear(hidden, weight, bias)

    def fake_fused_vocab(joint_hidden, weight, bias, **kwargs):
        record_inputs(joint_hidden, weight, bias)
        anchor = joint_hidden.float().sum(-1) + weight.sum() + bias.sum()
        return anchor, anchor

    def fake_fused(encoder_output_projected, predictor_output_projected, weight, bias, **kwargs):
        hidden = encoder_output_projected.unsqueeze(2) + predictor_output_projected.unsqueeze(1)
        return fake_fused_vocab(hidden, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", record_linear)
    monkeypatch.setattr(benchmark_rnnt_joint, "rnnt_joint_fused_vocab", fake_fused_vocab)
    monkeypatch.setattr(benchmark_rnnt_joint, "rnnt_fused_joint", fake_fused)
    result = benchmark_rnnt_joint.run_benchmark(
        joint=joint,
        loss="turbo_rnnt",
        dtype="bf16-mixed",
        batch_size=2,
        max_time=4,
        max_targets=3,
        hidden_dim=32,
        vocab_size=17,
        warmup_iters=1,
        bench_iters=1,
        fused_batch_size=fused_batch_size,
    )
    assert calls
    assert result.dtype == "bfloat16-mixed"


@pytest.mark.unit
@pytest.mark.parametrize("dtype", ["bf16-mixed", "bfloat16-mixed"])
def test_multi_benchmark_passes_precision_options(monkeypatch, dtype):
    calls = []
    monkeypatch.setattr("sys.argv", ["benchmark_rnnt_joint_multi.py", "--dtype", dtype, "--mm-full-precision"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test GPU")
    monkeypatch.setattr(benchmark_rnnt_joint_multi, "VOCAB_SIZES", [17])
    monkeypatch.setattr(
        benchmark_rnnt_joint_multi,
        "EXPERIMENTS",
        [("standard", "-", "turbo_rnnt"), ("fused_vocab", "triton", "turbo_rnnt")],
    )
    monkeypatch.setattr(benchmark_rnnt_joint_multi, "run_benchmark", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(benchmark_rnnt_joint_multi, "print_results_table", lambda results: None)
    benchmark_rnnt_joint_multi.main()
    assert [call["dtype"] for call in calls] == ["bfloat16-mixed"] * 2
    assert [call["mm_full_precision"] for call in calls] == [False, True]


class _RecordingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = "sum"
        self.batch_sizes = []

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        self.batch_sizes.append(log_probs.shape[0])
        return log_probs.sum(dim=(1, 2, 3))

    def reduce(self, losses, target_lengths):
        return torch.cat(losses).sum()


@pytest.mark.unit
def test_standard_joint_uses_native_fused_batching():
    batch_size, hidden_dim, max_time, max_targets, vocab_size = 5, 4, 3, 2, 7
    joint = benchmark_rnnt_joint._create_standard_joint(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dropout_p=0.0,
        fused_batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    loss = _RecordingLoss()
    joint.set_loss(loss)
    joint.set_wer(object())
    encoder_outputs = torch.randn(batch_size, hidden_dim, max_time, requires_grad=True)
    decoder_outputs = torch.randn(batch_size, hidden_dim, max_targets + 1, requires_grad=True)
    targets = torch.randint(0, vocab_size - 1, (batch_size, max_targets))
    source_lengths = torch.full((batch_size,), max_time, dtype=torch.long)
    target_lengths = torch.full((batch_size,), max_targets, dtype=torch.long)

    loss_value, wer, wer_num, wer_denom = joint(
        encoder_outputs=encoder_outputs,
        decoder_outputs=decoder_outputs,
        encoder_lengths=source_lengths,
        transcripts=targets,
        transcript_lengths=target_lengths,
    )
    loss_value.backward()

    assert joint.fuse_loss_wer
    assert joint.fused_batch_size == 2
    assert isinstance(joint.enc, torch.nn.Identity)
    assert isinstance(joint.pred, torch.nn.Identity)
    assert loss.batch_sizes == [2, 2, 1]
    assert encoder_outputs.grad is not None
    assert decoder_outputs.grad is not None
    assert (wer, wer_num, wer_denom) == (None, None, None)


@pytest.mark.unit
def test_standard_joint_disables_fused_batching_by_default():
    joint = benchmark_rnnt_joint._create_standard_joint(
        hidden_dim=4,
        vocab_size=7,
        dropout_p=0.0,
        fused_batch_size=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert not joint.fuse_loss_wer
    assert joint.fused_batch_size is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_forward_memory_baseline_excludes_warmup_gradients(monkeypatch):
    batch_size = 2
    max_time = 4
    max_targets = 3

    def fake_fused_joint(encoder_output_projected, predictor_output_projected, weight, bias, **kwargs):
        anchor = (
            encoder_output_projected[..., 0].sum()
            + predictor_output_projected[..., 0].sum()
            + weight[:, 0].sum()
            + bias.sum()
        ) * 0.0
        output_shape = (batch_size, max_time, max_targets + 1)
        target_logprobs = torch.zeros(output_shape, device="cuda", dtype=torch.float32) + anchor
        blank_logprobs = torch.zeros(output_shape, device="cuda", dtype=torch.float32) + anchor
        return target_logprobs, blank_logprobs

    def fake_loss(target_logprobs, blank_logprobs, **kwargs):
        return (target_logprobs + blank_logprobs).sum(dim=(1, 2))

    monkeypatch.setattr(benchmark_rnnt_joint, "rnnt_fused_joint", fake_fused_joint)
    monkeypatch.setattr(benchmark_rnnt_joint, "rnnt_loss_from_logprobs_triton", fake_loss)

    result = benchmark_rnnt_joint.run_benchmark(
        joint="fused",
        loss="turbo_rnnt",
        dtype=torch.float32,
        warmup_iters=1,
        bench_iters=1,
        batch_size=batch_size,
        max_time=max_time,
        hidden_dim=64,
        vocab_size=32,
        max_targets=max_targets,
        dropout_p=0.0,
    )

    output_bytes = 2 * batch_size * max_time * (max_targets + 1) * torch.float32.itemsize
    assert result.forward_memory_gb * 1024**3 >= output_bytes
