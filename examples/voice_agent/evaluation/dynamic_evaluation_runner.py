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

"""
Dynamic Voice Agent Evaluation Runner

Runs evaluation scenarios with dynamic system prompt updates.
Each scenario can specify different prompts for evaluator and target agents.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from nemo.agents.voice_agent.evaluation.bridge import VoiceAgentEvaluationBridge
from nemo.agents.voice_agent.utils import FileLogger


async def run_dynamic_evaluation(
    user_url: str,
    agent_url: str,
    output_dir: str,
    scenarios: list[dict],
    duration_per_scenario: int = 60,
    pause_between_scenarios: float = 0.5,
    user_output_sample_rate: int = 24000,
    agent_output_sample_rate: int = 24000,
    user_input_sample_rate: int = 16000,
    agent_input_sample_rate: int = 16000,
    output_sample_rate: int = 24000,
    global_timestamp: str = None,
    logger: FileLogger = None,
):
    """
    Run evaluation with dynamic scenario switching and latency measurement.

    Args:
        user_url: WebSocket URL of user (simulated user)
        agent_url: WebSocket URL of agent being tested
        output_dir: Output directory for results
        scenarios: List of scenarios, each with:
            - name: Scenario name
            - user_prompt: User system prompt
            - agent_prompt: Optional agent system prompt
            - duration: Optional duration override
        duration_per_scenario: Default duration per scenario (seconds)
        pause_between_scenarios: Seconds to pause between scenarios
        user_output_sample_rate: User TTS output sample rate (default: 24000)
        agent_output_sample_rate: Agent TTS output sample rate (default: 24000)
        user_input_sample_rate: User STT input sample rate (default: 16000)
        agent_input_sample_rate: Agent STT input sample rate (default: 16000)
        output_sample_rate: Output sample rate for recorded audio (default: 24000)
    """

    if not logger:
        logger = FileLogger()

    os.makedirs(output_dir, exist_ok=True)
    global_timestamp = global_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    bridge = VoiceAgentEvaluationBridge(
        user_url=user_url,
        agent_url=agent_url,
        output_dir=None,  # Will be set per scenario
        user_output_sample_rate=user_output_sample_rate,
        agent_output_sample_rate=agent_output_sample_rate,
        user_input_sample_rate=user_input_sample_rate,
        agent_input_sample_rate=agent_input_sample_rate,
        output_sample_rate=output_sample_rate,
    )

    all_results = []

    for idx, scenario in enumerate(scenarios):
        logger.info(f"{'='*80}")
        logger.info(f"Starting Scenario {idx+1}/{len(scenarios)}: {scenario['name']}")
        logger.info(f"Scenario config: {scenario}")
        logger.info(f"{'='*80}\n")

        # Create scenario-specific directory
        scenario_dir = os.path.join(output_dir, scenario['name'])
        os.makedirs(scenario_dir, exist_ok=True)

        logger.info(f"Preparing for scenario: {scenario['name']}...")
        await bridge.prepare_for_scenario(scenario, scenario_dir)
        await asyncio.sleep(pause_between_scenarios)

        # Run scenario
        duration = scenario.get("duration", duration_per_scenario)
        logger.info(f"Running scenario for {duration} seconds...")

        scenario_start = datetime.now()
        await bridge.run_scenario(duration=duration)
        scenario_end = datetime.now()

        # Collect metrics for this scenario
        metrics = bridge.get_metrics()
        metrics["scenario_name"] = scenario["name"]
        metrics["scenario_directory"] = scenario_dir
        metrics["scenario_duration"] = (scenario_end - scenario_start).total_seconds()
        all_results.append(metrics)

        # Log scenario summary
        latency_stats = metrics["latency_stats"]
        logger.info(f"{'='*80}")
        logger.info(f"Scenario '{scenario['name']}' Complete")
        logger.info(f"{'='*80}")
        logger.info(f"  Total turns: {metrics['total_turns']}")
        logger.info(f"  Duration: {metrics['scenario_duration']:.1f}s")
        logger.info(f"  Latency measurements: {latency_stats['count']}")
        if latency_stats['count'] > 0:
            logger.info(f"  Mean latency: {latency_stats['mean_ms']:.1f}ms")
            logger.info(f"  Median latency: {latency_stats['median_ms']:.1f}ms")
            logger.info(f"  P95 latency: {latency_stats['p95_ms']:.1f}ms")

    # Save detailed results
    results_file = os.path.join(output_dir, f"results_{global_timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV with latency details
    latency_csv_file = os.path.join(output_dir, f"latencies_{global_timestamp}.csv")
    with open(latency_csv_file, "w") as f:
        f.write("Scenario,User_Transcript,Agent_Transcript,Latency_ms\n")
        for result in all_results:
            scenario_name = result["scenario_name"]
            for latency in result["latencies"]:
                user_text = latency["user_transcript"].replace('"', '""')
                agent_text = latency["agent_transcript"].replace('"', '""')
                f.write(f'"{scenario_name}","{user_text}","{agent_text}",{latency["latency_ms"]:.1f}\n')

    # Save summary
    summary_file = os.path.join(output_dir, f"summary_{global_timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        total_turns = sum(r["total_turns"] for r in all_results)
        total_duration = sum(r["scenario_duration"] for r in all_results)

        f.write(f"Total Scenarios: {len(scenarios)}\n")
        f.write(f"Total Duration: {total_duration:.1f}s\n")
        f.write(f"Total Turns: {total_turns}\n\n")

        f.write("Per-Scenario Results:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            stats = result["latency_stats"]
            f.write(f"\n{result['scenario_name']}:\n")
            f.write(f"  Turns: {result['total_turns']}\n")
            f.write(f"  Duration: {result['scenario_duration']:.1f}s\n")
            if result['scenario_duration'] > 0:
                f.write(f"  Turns/min: {result['total_turns'] / (result['scenario_duration'] / 60):.1f}\n")
            f.write(f"  Latency Measurements: {stats['count']}\n")
            if stats['count'] > 0:
                f.write(f"    Mean: {stats['mean_ms']:.1f}ms\n")
                f.write(f"    Median: {stats['median_ms']:.1f}ms\n")
                f.write(f"    P95: {stats['p95_ms']:.1f}ms\n")
                f.write(f"    Min: {stats['min_ms']:.1f}ms\n")
                f.write(f"    Max: {stats['max_ms']:.1f}ms\n")

        # Overall latency statistics
        all_latencies = []
        for result in all_results:
            all_latencies.extend([l["latency_ms"] for l in result["latencies"]])

        if all_latencies:
            all_latencies.sort()
            count = len(all_latencies)
            f.write(f"\n\nOverall Latency Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Measurements: {count}\n")
            f.write(f"  Mean: {sum(all_latencies) / count:.1f}ms\n")
            f.write(f"  Median: {all_latencies[count // 2]:.1f}ms\n")
            f.write(f"  P95: {all_latencies[int(count * 0.95)]:.1f}ms\n")
            f.write(f"  Min: {all_latencies[0]:.1f}ms\n")
            f.write(f"  Max: {all_latencies[-1]:.1f}ms\n")

    logger.info(f"{'='*80}")
    logger.info(f"Evaluation Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Latencies saved to: {latency_csv_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"\nScenario directories:")
    for result in all_results:
        logger.info(f"  {result['scenario_name']}: {result['scenario_directory']}")
    logger.info(f"\nTotal: {len(scenarios)} scenarios, {total_turns} turns, {total_duration:.1f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run voice agent evaluation with dynamic scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default scenarios
  python dynamic_evaluation_runner.py \\
      --user-url ws://localhost:8765 \\
      --agent-url ws://localhost:8766

  # Run with custom scenarios file
  python dynamic_evaluation_runner.py \\
      --user-url ws://localhost:8765 \\
      --agent-url ws://localhost:8766 \\
      --scenarios-file scenarios/customer_service.json \\
      --output-dir ./results/cs_eval

  # Run with custom duration
  python dynamic_evaluation_runner.py \\
      --user-url ws://localhost:8765 \\
      --agent-url ws://localhost:8766 \\
      --duration 120 \\
      --pause 10
        """,
    )
    parser.add_argument(
        "--user-url",
        default="ws://localhost:8765",
        help="WebSocket URL of user (simulated user) (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--agent-url",
        default="ws://localhost:8766",
        help="WebSocket URL of agent being tested (default: ws://localhost:8766)",
    )
    parser.add_argument(
        "--output-dir", default="./eval_results", help="Output directory for results (default: ./eval_results)"
    )
    parser.add_argument("--scenarios-file", help="JSON file with scenarios (see scenarios/ directory for examples)")
    parser.add_argument(
        "--duration", type=int, default=120, help="Default duration per scenario in seconds (default: 120)"
    )
    parser.add_argument("--pause", type=float, default=0.5, help="Pause between scenarios in seconds (default: 0.5)")
    parser.add_argument(
        "--output-sample-rate", type=int, default=16000, help="Output sample rate for recorded audio (default: 16000)"
    )

    args = parser.parse_args()

    # Default scenarios
    scenarios = [
        # {
        #     "name": "Friendly_Conversation-Noisy",
        #     "user_prompt": """You are a friendly human user named Bob, and you are testing a voice assistant.
        #     Start by saying that "Hi I'm Bob", then ask the following questions one by one, wait for response before asking the next question:
        #     1. Tell me a joke about a cat.
        #     2. What's the capital of the United States?
        #     3. What's the result of 1+1?
        #     4. What's the color of the sky?
        #     After the agent has answered all the questions, say "Thank you for your answers, goodbye", then keep responding with empty responses "\n" if any following turns.
        #     """,
        #     "duration": 90,
        #     "noise_config": {
        #         "noise_files": "/home/heh/github/NeMo-main/examples/voice_agent/evaluation/nemo_experiments/id_494165-FX_Car_Driving.wav",
        #         "gain_db": 0.0,
        #         "max_noise_duration": 100.0,
        #         "random_offset": True,
        #     },
        # },
        {
            "name": "Friendly_Conversation-Clean",
            "user_prompt": """You are a friendly human user named Bob, and you are testing a voice assistant. You don't help the assistant to do anything, you only ask the question and wait for the response.
            Start by saying that "Hi I'm Bob", then say the following messages one by one, wait for response before saying the next message. You should strictly follow the content of the messages, don't add any other information.
            - What is the weather in San Francisco?
            - Now send scenario summary with the external tool.

            Then ask the assistant to send scenario summary using the external tool.
            """,
            "agent_prompt": """You are a helpful AI agent named Lisa. Start by greeting the user with 'Hi, I'm Lisa, your helpful AI assistant. How can I help you today?'.
            You need to answer the user's question based on your internal knowledge. 
            
            When asked to send scenario summary, you must use the `SendScenarioSummaryTool` tool, the input message to that tool should contain all the user questions and your answers one by one. 
            """,
            "duration": 90,
        },
        #         {
        #             "name": "Challenging Questions",
        #             "user_prompt": """You are a human user. You are testing a voice assistant with difficult questions.
        # Ask complex, multi-part questions and test edge cases.
        # Start with a challenging question about a technical topic.""",
        #             "duration": 60,
        #         },
        #         {
        #             "name": "Rapid Interaction",
        #             "user_prompt": """You are a human user. You are testing how well the assistant handles quick back-and-forth.
        # Ask short questions and wait for answers. Keep responses brief.
        # Start with a simple question and build from there.""",
        #             "duration": 60,
        #         },
    ]
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"eval_{session_timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    logger = FileLogger(os.path.join(session_dir, f"evaluation_log.txt"))

    # Load scenarios from file if provided
    if args.scenarios_file:
        scenarios_path = Path(args.scenarios_file)
        if not scenarios_path.exists():
            raise FileNotFoundError(f"Scenarios file not found: {args.scenarios_file}")

        with open(scenarios_path) as f:
            scenarios = json.load(f)
        logger.info(f"Loaded {len(scenarios)} scenarios from {args.scenarios_file}")
    else:
        logger.info(f"Using {len(scenarios)} default scenarios")

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
    main()
