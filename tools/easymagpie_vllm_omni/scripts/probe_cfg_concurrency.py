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
"""GPU smoke test for mixed CFG/conditional admission, cancellation, and reuse."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

from benchmark_model import _load_model_meta, build_prompt
from probe_cfg_pairing import run_probe


async def main(args: argparse.Namespace) -> dict[str, Any]:
    """Run the configured benchmark or lifecycle validation workflow.

    Args:
        args: Parsed command-line options.

    Returns:
        The completed validation report.
    """
    from vllm.plugins import load_general_plugins
    from vllm_omni import AsyncOmni

    load_general_plugins()
    prompt = build_prompt(args.text, _load_model_meta(args.model, args.speaker))
    results = []
    omni = AsyncOmni(model=args.model, deploy_config=str(args.deploy_config), log_stats=False)

    async def request(
        name: str,
        guided: bool,
        abort: bool = False,
    ) -> None:
        """Run one mixed-traffic request and record its lifecycle outcome.

        Args:
            name: Request or attribute name.
            guided: Whether to enable classifier-free guidance.
            abort: Whether to cancel after the requested boundary.
        """
        incoming = copy.deepcopy(prompt)
        incoming["additional_information"].update(
            use_cfg=guided, cfg_scale=args.cfg_scale, temperature=args.temperature, top_k=args.top_k
        )
        count = await run_probe(omni, incoming, request_id=name, timeout=args.timeout, abort_after_output=abort)
        result = dict(request_id=name, use_cfg=guided, aborted=abort, public_outputs=count)
        results.append(result)
        print(json.dumps(result), flush=True)

    tasks = []
    try:
        # Mix nine physical sequences; this is a lifecycle smoke test, not a
        # saturation test of the profile's 64 physical Stage-0 slots.
        # Abort one pair while unrelated public requests are still in flight.
        tasks = [
            asyncio.create_task(request(name, guided, abort))
            for name, guided, abort in (
                ("mixed-guided-0", True, False),
                ("mixed-plain-0", False, False),
                ("mixed-guided-abort", True, True),
                ("mixed-plain-1", False, False),
                ("mixed-guided-1", True, False),
                ("mixed-plain-2", False, False),
            )
        ]
        await asyncio.gather(*tasks)
        tasks = [
            asyncio.create_task(request(name, guided))
            for name, guided in (("reuse-guided", True), ("reuse-plain", False))
        ]
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        omni.shutdown()
    report = dict(
        status="passed",
        model=args.model,
        speaker=args.speaker,
        text=args.text,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        deploy_config=str(args.deploy_config),
        results=results,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line options.

    Returns:
        Validated command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--speaker", default="eng")
    parser.add_argument("--text", default="This voice is conditioned from request-time reference audio.")
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "deploy" / "easymagpie_cfg_concurrent.yaml",
    )
    parser.add_argument("--output", help="Write a JSON summary only after every check passes")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
