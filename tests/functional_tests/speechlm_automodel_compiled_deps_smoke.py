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
"""Two-GPU smoke coverage for SALMAutomodel compiled Automodel dependencies."""

import importlib.util
import os
import subprocess
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig

from nemo.collections.speechlm2.models import SALMAutomodel
from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy


pytestmark = pytest.mark.integration


def _require_two_gpu_torchrun() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("SALMAutomodel compiled dependency smoke requires CUDA")
    if torch.cuda.device_count() < 2:
        pytest.skip("SALMAutomodel compiled dependency smoke requires at least 2 visible GPUs")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        pytest.skip("Run with `torchrun --nproc-per-node 2` to exercise DeepEP")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, torch.device(f"cuda:{local_rank}")


def _require_nvlink_topology() -> None:
    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("DeepEP topology check requires `nvidia-smi topo -m`")

    for line in topo.splitlines():
        columns = line.split()
        if columns[:1] == ["GPU0"] and len(columns) > 2 and columns[2].startswith("NV"):
            return

    pytest.skip("DeepEP intranode dispatch requires NVLink between the two visible GPUs")


def _require_compiled_dependencies() -> None:
    missing = [
        name
        for name in ("causal_conv1d", "deep_ep", "grouped_gemm", "mamba_ssm", "nemo_automodel", "transformer_engine")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip(f"Missing compiled Automodel dependencies: {', '.join(missing)}")


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _assert_finite_nonzero_grad(parameters, label: str) -> None:
    grads = []
    for name, parameter in parameters:
        if parameter.grad is None:
            continue
        grad = _local_tensor(parameter.grad)
        assert torch.isfinite(grad).all(), f"{label} parameter {name} has a non-finite gradient"
        grads.append(grad.detach().float().abs().sum())

    assert grads, f"{label} did not receive any gradients"
    assert torch.stack(grads).sum() > 0, f"{label} gradients are all zero"


def _init_distributed(device: torch.device) -> None:
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=device)


def _make_strategy(*, world_size: int, ep_size: int) -> AutomodelParallelStrategy:
    from nemo_automodel.components.distributed.config import FSDP2Config
    from nemo_automodel.components.moe.config import MoEParallelizerConfig

    strategy = AutomodelParallelStrategy(
        dp_size=world_size // ep_size,
        dp_replicate_size=1,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ep_size=ep_size,
        distributed_config=FSDP2Config(),
        moe_config=MoEParallelizerConfig(wrap_outer_model=False),
    )
    strategy.create_device_mesh()
    return strategy


def _tiny_nemotron_v3_config(dtype: torch.dtype) -> PretrainedConfig:
    return PretrainedConfig(
        architectures=["NemotronHForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        chunk_size=4,
        conv_kernel=3,
        head_dim=16,
        hidden_size=64,
        initializer_range=0.02,
        intermediate_size=128,
        layer_norm_epsilon=1e-5,
        layers_block_type=["attention", "mamba", "moe"],
        mamba_head_dim=16,
        mamba_hidden_act="silu",
        mamba_num_heads=4,
        mlp_bias=False,
        mlp_hidden_act="relu2",
        model_type="nemotron_h",
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        n_group=1,
        n_groups=1,
        n_routed_experts=2,
        norm_topk_prob=False,
        num_attention_heads=4,
        num_experts_per_tok=1,
        num_hidden_layers=3,
        num_key_value_heads=4,
        output_hidden_states=False,
        rescale_prenorm_residual=False,
        residual_in_fp32=False,
        routed_scaling_factor=1.0,
        ssm_state_size=8,
        time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        time_step_max=0.1,
        time_step_min=0.001,
        topk_group=1,
        torch_dtype=dtype,
        use_bias=False,
        use_cache=False,
        use_conv_bias=True,
        vocab_size=96,
    )


def _make_automodel_backend(dispatcher: str):
    from nemo_automodel.components.models.common import BackendConfig

    return BackendConfig(
        attn="te",
        linear="te",
        rms_norm="torch_fp32",
        experts="gmm",
        dispatcher=dispatcher,
        dispatcher_num_sms=20,
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=False,
        gate_precision=torch.float32,
    )


def _make_tiny_nemotron_llm(
    *, strategy: AutomodelParallelStrategy, dispatcher: str, dtype: torch.dtype, parallelize_moe: bool
):
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    llm = NemotronHForCausalLM(
        _tiny_nemotron_v3_config(dtype),
        backend=_make_automodel_backend(dispatcher),
    )
    llm.initialize_weights(buffer_device=torch.device(f"cuda:{torch.cuda.current_device()}"), dtype=dtype)
    llm.to(device=torch.device(f"cuda:{torch.cuda.current_device()}"), dtype=dtype)

    if parallelize_moe:
        from nemo_automodel.components.moe.parallelizer import parallelize_model

        parallelize_model(
            llm,
            world_mesh=strategy.device_mesh,
            moe_mesh=strategy.moe_mesh,
            dp_axis_names=("dp",),
            cp_axis_name="cp",
            tp_axis_name="tp",
            ep_axis_name="ep",
            ep_shard_axis_names=None,
            **strategy.moe_config.to_dict(),
        )

    return llm


def _make_salm_automodel_smoke_model(*, llm: torch.nn.Module) -> SALMAutomodel:
    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    model.llm = llm
    model.perception = None
    model.cfg = {}
    model._use_fsdp = False
    model._use_tp = False
    return model


def _run_salm_automodel_nemotron3_forward_backward(*, dispatcher: str, expect_deepep: bool, ep_size: int) -> None:
    _require_compiled_dependencies()
    local_rank, world_size, device = _require_two_gpu_torchrun()
    dtype = torch.bfloat16
    torch.manual_seed(1234 + local_rank)

    _init_distributed(device)
    strategy = _make_strategy(world_size=world_size, ep_size=ep_size)
    llm = _make_tiny_nemotron_llm(
        strategy=strategy,
        dispatcher=dispatcher,
        dtype=dtype,
        parallelize_moe=expect_deepep,
    )
    model = _make_salm_automodel_smoke_model(llm=llm)

    input_embeds = torch.randn(2, 8, 64, device=device, dtype=dtype, requires_grad=True)
    outputs = model(input_embeds, attention_mask=torch.ones(2, 8, dtype=torch.bool, device=device))
    logits = outputs["logits"]
    assert logits.shape == (2, 8, 96)
    assert torch.isfinite(logits).all()

    loss = logits.float().square().mean()
    loss.backward()

    assert input_embeds.grad is not None
    assert torch.isfinite(input_embeds.grad).all()
    _assert_finite_nonzero_grad(model.llm.model.layers.named_parameters(), "Nemotron3 backbone")
    _assert_finite_nonzero_grad(model.llm.lm_head.named_parameters(), "Nemotron3 LM head")

    dist.all_reduce(loss.detach())


def test_salm_automodel_nemotron3_transformer_engine_grouped_gemm_forward_backward():
    """Run a tiny real Nemotron3 SALMAutomodel forward/backward through TE and grouped-gemm."""
    _run_salm_automodel_nemotron3_forward_backward(dispatcher="torch", expect_deepep=False, ep_size=1)


def test_salm_automodel_nemotron3_deepep_grouped_gemm_forward_backward():
    """Run a tiny real Nemotron3 SALMAutomodel forward/backward through DeepEP-dispatched grouped-gemm.

    This is intentionally launched with two CUDA ranks. DeepEP falls back or is
    inactive in a single-process run. It also requires NVLink for intranode
    dispatch; PCIe-only machines skip this test to avoid a known DeepEP hang.
    """
    _require_nvlink_topology()
    _run_salm_automodel_nemotron3_forward_backward(dispatcher="deepep", expect_deepep=True, ep_size=2)
