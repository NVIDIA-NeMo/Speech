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

"""
Voice Agent Evaluation Entry Point

Usage:
  # Run all registered scenarios
  python run_evaluation.py --user-url ws://localhost:8766 --agent-url ws://localhost:8765

  # Run specific scenarios by name
  python run_evaluation.py --scenarios fastbite --user-url ws://localhost:8765 --agent-url ws://localhost:8766

  # List available scenarios
  python run_evaluation.py --list
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

from nemo.agents.voice_agent.evaluation.runner import run_dynamic_evaluation
from nemo.agents.voice_agent.evaluation.scenarios import get_eval_scenario, list_eval_scenarios
from nemo.agents.voice_agent.evaluation.utils import LLMJudge
from nemo.agents.voice_agent.utils import FileLogger


def main():
    parser = argparse.ArgumentParser(
        description="Run voice agent evaluation with structured scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all registered scenarios
  python run_evaluation.py \\
      --user-url ws://localhost:8765 \\
      --agent-url ws://localhost:8766

  # Run specific scenarios by name(s)
  python run_evaluation.py \\
      --user-url ws://localhost:8765 \\
      --agent-url ws://localhost:8766 \\
      --scenarios fastbite simple_qa_1 simple_qa_3

  # List available scenarios
  python run_evaluation.py --list
        """,
    )
    parser.add_argument(
        "--user-url",
        default="ws://localhost:8766",
        help="WebSocket URL of user (simulated user) (default: ws://localhost:8766)",
    )
    parser.add_argument(
        "--agent-url",
        default="ws://localhost:8765",
        help="WebSocket URL of agent being tested (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--output-dir", default="./eval_results", help="Output directory for results (default: ./eval_results)"
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help="Scenario names to run (default: all registered scenarios). Use --list to see available names.",
    )
    parser.add_argument("--list", action="store_true", help="List all available scenarios and exit")
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Default duration per scenario in seconds (default: 120), only used when the scenario does not specify a duration.",
    )
    parser.add_argument("--pause", type=float, default=0.5, help="Pause between scenarios in seconds (default: 0.5)")
    parser.add_argument(
        "--output-sample-rate", type=int, default=16000, help="Output sample rate for recorded audio (default: 16000)"
    )
    parser.add_argument(
        "--judge-url", default="http://localhost:8000/v1/chat/completions", help="URL of the judge API"
    )
    parser.add_argument(
        "--judge-model",
        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        help="Model name for the judge API (default: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)",
    )
    parser.add_argument("--judge-api-key", default=None, help="API key for the LLM judge")

    args = parser.parse_args()

    # List mode
    if args.list:
        available = list_eval_scenarios()
        if not available:
            print("No scenarios registered.")
        else:
            print("Available scenarios:")
            for name in available:
                print(f"  - {name}")
        return 0

    # Resolve which scenarios to run
    scenario_names = args.scenarios if args.scenarios else list_eval_scenarios()

    if not scenario_names:
        print("No scenarios available. Register scenarios using @register_eval_scenario.", file=sys.stderr)
        return 1

    # Instantiate scenario objects
    scenarios = []
    for name in scenario_names:
        scenario = get_eval_scenario(name)
        if scenario is None:
            available = list_eval_scenarios()
            print(f"Unknown scenario: '{name}'. Available: {available}", file=sys.stderr)
            return 1
        scenarios.append(scenario)

    # Set up output directory
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"eval_{session_timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    logger = FileLogger(os.path.join(session_dir, "evaluation_log.txt"))
    logger.info(f"Running {len(scenarios)} scenario(s): {[s.name for s in scenarios]}")

    if args.judge_url and args.judge_model:
        logger.info(f"Using LLM judge: {args.judge_url} with model: {args.judge_model}")
        judge = LLMJudge(
            url=args.judge_url,
            model=args.judge_model,
            api_key=args.judge_api_key,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            seed=42,
            chat_template_kwargs={"enable_thinking": True},
            vllm_xargs={"thinking_budget": 512},
        )
    else:
        judge = None

    # Run evaluation
    try:
        asyncio.run(
            run_dynamic_evaluation(
                user_url=args.user_url,
                agent_url=args.agent_url,
                output_dir=session_dir,
                scenarios=scenarios,
                duration_per_scenario=args.duration,
                pause_between_scenarios=args.pause,
                output_sample_rate=args.output_sample_rate,
                global_timestamp=session_timestamp,
                logger=logger,
                judge=judge,
            )
        )
        return 0
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
