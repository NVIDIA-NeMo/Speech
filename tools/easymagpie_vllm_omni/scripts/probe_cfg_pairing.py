# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""GPU completion/abort/reuse smoke test for V2 paired histories or guided synthesis."""
from __future__ import annotations

import argparse
import asyncio
import copy
from pathlib import Path

from benchmark_model import _load_model_meta, build_prompt
from easymagpie_vllm_omni.cfg_pairing import CFG_PAIRING_PROBE_KEY


async def run_probe(omni, prompt, *, request_id: str, timeout: float, abort_after_output: bool = False) -> int:
    """Require public-only output, then completion or an explicit abort."""
    generator = omni.generate(prompt, request_id=request_id)
    count, codec_finished = 0, False
    try:
        async with asyncio.timeout(timeout):
            async for output in generator:
                if output.request_id != request_id:
                    raise RuntimeError(f"Unexpected physical request in public output: {output.request_id}")
                if getattr(output, "error", None):
                    raise RuntimeError(str(output.error))
                count += 1
                codec_finished |= output.stage_id == 1 and output.finished
                if abort_after_output:
                    return count
        if not count or not codec_finished:
            raise RuntimeError("Pairing probe ended without a completed codec output")
        return count
    finally:
        # Also covers timeout, disconnect-like generator close, and errors.
        try:
            await omni.abort(request_id)
        finally:
            await generator.aclose()


async def main(args) -> None:
    from vllm.plugins import load_general_plugins
    from vllm_omni import AsyncOmni

    load_general_plugins()
    meta = _load_model_meta(args.model, args.speaker)
    prompt = build_prompt(args.text, meta)
    use_cfg = getattr(args, "use_cfg", False)
    info = prompt["additional_information"]
    if use_cfg:
        info.update(use_cfg=True, cfg_scale=args.cfg_scale, temperature=args.temperature)
    else:
        info[CFG_PAIRING_PROBE_KEY] = True
    profile_name = "easymagpie_cfg.yaml" if use_cfg else "easymagpie_cfg_pairing.yaml"
    profile = Path(__file__).resolve().parents[1] / "deploy" / profile_name
    omni = AsyncOmni(model=args.model, deploy_config=str(profile), log_stats=False)
    try:
        # The first request checks normal completion, the second cancellation,
        # and the third that freed physical sequence slots can be used again.
        for index, abort in enumerate((False, True, False)):
            count = await run_probe(
                omni,
                copy.deepcopy(prompt),
                request_id=f"cfg-pairing-probe-{index}",
                timeout=args.timeout,
                abort_after_output=abort,
            )
            print(f"request={index} {'aborted' if abort else 'completed'} public_outputs={count}")
    finally:
        omni.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted V2 model directory")
    parser.add_argument("--speaker", default="eng")
    parser.add_argument("--text", default="This is a test of paired sequence cache ownership.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per request in seconds")
    parser.add_argument(
        "--use-cfg", action="store_true", help="Exercise guided synthesis with a reconverted V2 artifact"
    )
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=0.0, help="Acoustic temperature for guided synthesis")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
