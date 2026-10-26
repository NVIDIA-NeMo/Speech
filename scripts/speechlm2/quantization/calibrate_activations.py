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

"""Record the activation ranges for FP8 calibration of the audio encoder and the MTP draft head.

The script serves a SpeechLM checkpoint with vLLM's offline engine and the NeMo SpeechLM plugin, transcribes
a manifest of held-out audio with MTP speculative decoding on, and records the largest absolute input of:

- the audio encoder's ``attn.w_qkv``, ``attn.out_proj``, ``ffn.net.0`` and ``ffn.net.3`` Linears;
- every Linear of the MTP draft head, its routed experts' input, and the squared-ReLU activation that their
  down projections read, which the fused MoE kernel never exposes.

Calibrate the checkpoint you will serve, quantized decoder included (the output of ``assemble_checkpoint.py``):
the draft head reads the target model's hidden states. The engine runs eagerly because CUDA graphs bypass the
observers. The observers are installed after vLLM's startup, so its profiling inputs are not recorded. The draft
head runs vLLM's Triton MoE backend, which keeps the expert weights in the unpacked layout that the observers
read; other backends may repack them.

The output, ``{"encoder": {"layers.N.attn.w_qkv": range, ...}, "mtp": {"model.layers.N.eh_proj": range, ...}}``,
is the input of ``quantize_encoder_mtp_fp8.py``. Needs vLLM 0.28 and the SpeechLM plugin with
NVIDIA-NeMo/Speech#16341.

Example (inside the vLLM image with NeMo on PYTHONPATH):
    VLLM_PLUGINS=nemo_speechlm VLLM_WORKER_MULTIPROC_METHOD=spawn python calibrate_activations.py \\
        --checkpoint /models/ckpt-nvfp4-bf16heads --manifest calib.jsonl --kv-cache-dtype fp8 --output ranges.json
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path

from checkpoint_utils import check_output_path, write_json

ENCODER_PREFIX = "perception.encoder.asr_encoder."
ENCODER_TARGETS = ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")
PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)


def expert_activations(hidden_states, topk_ids, w13):
    """Yield, for each routed expert, the squared-ReLU activation that its down projection reads.

    ``w13`` holds the experts' up projections in the unpacked ``[experts, intermediate, hidden]`` layout.
    """
    if w13.dim() != 3 or w13.shape[-1] != hidden_states.shape[-1]:
        raise RuntimeError(
            f"expert weights of shape {tuple(w13.shape)} are not in the unpacked [experts, intermediate, hidden] "
            "layout; run the draft head with vLLM's Triton MoE backend"
        )
    for expert in topk_ids.unique().tolist():
        rows = (topk_ids == expert).any(dim=-1)
        yield (hidden_states[rows] @ w13[expert].t()).relu().square()


def install_observers(worker) -> int:
    """Install the activation-range observers in a vLLM worker and return how many modules they watch.

    Runs in the worker process (sent through ``LLM.collective_rpc``), so it imports what it needs itself.
    """
    import functools

    import torch
    from torch import nn
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import MoERunnerInterface
    from vllm.model_executor.layers.linear import LinearBase

    ranges: dict[str, torch.Tensor] = {}
    worker.activation_ranges = ranges

    def record(name: str, x: torch.Tensor) -> None:
        value = x.detach().abs().amax().float()
        ranges[name] = value if name not in ranges else torch.maximum(ranges[name], value)

    class ObservedLinear(nn.Linear):
        # Not a plain nn.Linear, so the encoder's packed attention path calls forward() rather than
        # slicing the weight.
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            record(self.range_name, x)
            return super().forward(x)

    watched = 0
    for name, module in worker.model_runner.model.named_modules():
        if (
            type(module) is nn.Linear
            and name.startswith(ENCODER_PREFIX + "layers.")
            and name.endswith(tuple("." + target for target in ENCODER_TARGETS))
        ):
            module.__class__ = ObservedLinear
            module.range_name = "encoder:" + name[len(ENCODER_PREFIX) :]
            watched += 1

    draft_model = getattr(getattr(worker.model_runner, "drafter", None), "model", None)
    if draft_model is None:
        raise RuntimeError("the engine has no MTP draft model; speculative decoding must be on")

    def drafting() -> bool:
        # Steps without attention metadata are vLLM's dummy runs, which carry no real data.
        return is_forward_context_available() and get_forward_context().attn_metadata is not None

    def linear_hook(name: str, module: nn.Module, args: tuple) -> None:
        if drafting():
            record("mtp:" + name, args[0])

    def moe_hook(name: str, module: nn.Module, args: tuple, kwargs: dict) -> None:
        if not drafting():
            return
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        router_logits = kwargs["router_logits"] if "router_logits" in kwargs else args[1]
        record(f"mtp:{name}.w13_input", hidden_states)
        activation = getattr(module.routed_experts.activation, "value", module.routed_experts.activation)
        if activation != "relu2_no_mul":
            raise NotImplementedError(f"{name}: only squared-ReLU experts are supported, not {activation}")
        _, topk_ids = module.router.select_experts(hidden_states, router_logits)
        for x in expert_activations(hidden_states, topk_ids, module.routed_experts.w13_weight):
            record(f"mtp:{name}.w2_input", x)

    for name, module in draft_model.named_modules():
        if isinstance(module, LinearBase):
            module.register_forward_pre_hook(functools.partial(linear_hook, name))
            watched += 1
        elif isinstance(module, MoERunnerInterface):
            module.register_forward_pre_hook(functools.partial(moe_hook, name), with_kwargs=True)
            watched += 1
    return watched


def collect_ranges(worker) -> dict[str, float]:
    """Return the recorded ranges from a vLLM worker."""
    return {name: float(value.item()) for name, value in sorted(worker.activation_ranges.items())}


def chat_request(audio_path: str, prompt: str) -> list[dict]:
    """Return one chat conversation that sends ``audio_path`` with ``prompt``."""
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    data = base64.b64encode(Path(audio_path).read_bytes()).decode()
    content = [
        {"type": "text", "text": prompt},
        {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{data}"}},
    ]
    return [{"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint to serve, from assemble_checkpoint.py"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL with an audio_filepath per line")
    parser.add_argument("--output", type=Path, required=True, help="activation ranges JSON to write")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument("--kv-cache-dtype", default="auto", help="fp8 for an NVFP4 checkpoint")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N manifest entries")
    args = parser.parse_args()

    check_output_path(args.output, args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists")
    lines = [line for line in args.manifest.read_text().splitlines() if line.strip()]
    items = [json.loads(line) for line in lines][: args.limit]

    # collective_rpc sends a Python function to the engine process, which vLLM allows only with this set.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.checkpoint),
        tokenizer=str(args.checkpoint),
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
            "moe_backend": "triton",
        },
    )
    print(f"observing {llm.collective_rpc(install_observers)[0]} modules")
    llm.chat(
        [chat_request(item["audio_filepath"], args.prompt) for item in items],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=False),
        chat_template_kwargs={"enable_thinking": False},
    )
    ranges = llm.collective_rpc(collect_ranges)[0]

    encoder = {name.removeprefix("encoder:"): value for name, value in ranges.items() if name.startswith("encoder:")}
    mtp = {name.removeprefix("mtp:"): value for name, value in ranges.items() if name.startswith("mtp:")}
    if not encoder:
        raise SystemExit("recorded no encoder ranges; is the audio encoder still BF16?")
    if not mtp:
        raise SystemExit("recorded no MTP ranges; did speculative decoding run?")
    write_json(args.output, {"encoder": encoder, "mtp": mtp})
    print(f"wrote {len(encoder)} encoder and {len(mtp)} MTP ranges from {len(items)} recordings to {args.output}")


if __name__ == "__main__":
    main()
