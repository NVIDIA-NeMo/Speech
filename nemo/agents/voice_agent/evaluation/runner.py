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
Dynamic Voice Agent Evaluation Runner

Runs evaluation scenarios with dynamic system prompt updates.
Accepts structured Scenario objects instead of raw dicts.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List

from nemo.agents.voice_agent.evaluation.bridge import VoiceAgentEvaluationBridge
from nemo.agents.voice_agent.evaluation.scenarios.classes import Scenario
from nemo.agents.voice_agent.utils import FileLogger


async def run_dynamic_evaluation(
    user_url: str,
    agent_url: str,
    output_dir: str,
    scenarios: List[Scenario],
    duration_per_scenario: int = 120,
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
        scenarios: List of Scenario objects defining each evaluation scenario
        duration_per_scenario: Default duration per scenario in seconds
        pause_between_scenarios: Seconds to pause between scenarios
        user_output_sample_rate: User TTS output sample rate (default: 24000)
        agent_output_sample_rate: Agent TTS output sample rate (default: 24000)
        user_input_sample_rate: User STT input sample rate (default: 16000)
        agent_input_sample_rate: Agent STT input sample rate (default: 16000)
        output_sample_rate: Output sample rate for recorded audio (default: 24000)
        global_timestamp: Timestamp string for output file naming
        logger: FileLogger instance for logging
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
        logger.info(f"Starting Scenario {idx+1}/{len(scenarios)}: {scenario.name}")
        logger.info(f"{'='*80}\n")

        # Create scenario-specific directory
        scenario_dir = os.path.join(output_dir, scenario.name)
        os.makedirs(scenario_dir, exist_ok=True)

        # Build dict for bridge.prepare_for_scenario
        scenario_dict = {
            "name": scenario.name,
            "user_prompt": scenario.get_user_prompt(),
            "agent_prompt": scenario.get_agent_prompt(),
            "user_tools": scenario.get_user_tools(),
            "agent_tools": scenario.get_agent_tools(),
        }
        if scenario.noise_config:
            scenario_dict["noise_config"] = scenario.noise_config

        logger.info(f"Preparing for scenario: {scenario.name}...")
        await bridge.prepare_for_scenario(scenario_dict, scenario_dir)
        scenario.save(os.path.join(scenario_dir, "scenario_config"))
        await asyncio.sleep(pause_between_scenarios)

        # Run scenario
        duration = scenario.max_duration if scenario.max_duration is not None else duration_per_scenario
        logger.info(f"Running scenario for {duration} seconds...")

        scenario_start = datetime.now()
        await bridge.run_scenario(duration=duration)
        scenario_end = datetime.now()

        # Collect metrics for this scenario
        metrics = bridge.get_metrics()
        metrics["scenario_name"] = scenario.name
        metrics["scenario_directory"] = scenario_dir
        metrics["scenario_duration"] = (scenario_end - scenario_start).total_seconds()
        all_results.append(metrics)

        # Log scenario summary
        latency_stats = metrics["latency_stats"]
        logger.info(f"{'='*80}")
        logger.info(f"Scenario '{scenario.name}' Complete")
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
