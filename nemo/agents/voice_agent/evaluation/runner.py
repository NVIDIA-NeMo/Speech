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
from typing import List, Optional

from nemo.agents.voice_agent.evaluation.bridge import VoiceAgentEvaluationBridge
from nemo.agents.voice_agent.evaluation.scenarios.classes import Scenario
from nemo.agents.voice_agent.evaluation.utils import LLMJudge, check_if_task_success
from nemo.agents.voice_agent.utils import FileLogger


async def run_dynamic_evaluation(
    user_url: str,
    agent_url: str,
    output_dir: str,
    scenarios: List[Scenario],
    audio_chunk_in_seconds: float = 0.016,
    duration_per_scenario: Optional[int] = None,
    pause_between_scenarios: float = 0.5,
    user_output_sample_rate: int = 24000,
    agent_output_sample_rate: int = 24000,
    user_input_sample_rate: int = 16000,
    agent_input_sample_rate: int = 16000,
    output_sample_rate: int = 24000,
    global_timestamp: str = None,
    logger: FileLogger = None,
    judge: Optional[LLMJudge] = None,
    judge_threshold: Optional[float] = None,
):
    """
    Run evaluation with dynamic scenario switching and latency measurement.

    Args:
        user_url: WebSocket URL of user (simulated user)
        agent_url: WebSocket URL of agent being tested
        output_dir: Output directory for results
        scenarios: List of Scenario objects defining each evaluation scenario
        audio_chunk_in_seconds: Audio chunk in seconds for the audio stream (default: 0.016)
        duration_per_scenario: Maximum duration per scenario in seconds, which overrides the scenario's own max_duration if set.
        pause_between_scenarios: Seconds to pause between scenarios
        user_output_sample_rate: User TTS output sample rate (default: 24000)
        agent_output_sample_rate: Agent TTS output sample rate (default: 24000)
        user_input_sample_rate: User STT input sample rate (default: 16000)
        agent_input_sample_rate: Agent STT input sample rate (default: 16000)
        output_sample_rate: Output sample rate for recorded audio (default: 24000)
        global_timestamp: Timestamp string for output file naming
        logger: FileLogger instance for logging
        judge: LLMJudge instance for judging the scenario
        judge_threshold: Threshold for judging the scenario if binary result is desired, None for score based result
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
        audio_chunk_in_seconds=audio_chunk_in_seconds,
    )

    all_results = []
    success_results = []
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
        scenario_config_dir = os.path.join(scenario_dir, "scenario_config")
        os.makedirs(scenario_config_dir, exist_ok=True)
        scenario.save(scenario_config_dir)
        await asyncio.sleep(pause_between_scenarios)

        # Run scenario
        duration = duration_per_scenario if duration_per_scenario is not None else scenario.max_duration
        assert duration > 0, f"Duration per scenario must be greater than 0, got {duration}"
        logger.info(f"Running scenario for {duration} seconds...")

        scenario_start = datetime.now()
        await bridge.run_scenario(duration=duration)
        scenario_end = datetime.now()

        # Check if the scenario is successful
        reference_file = os.path.join(scenario_config_dir, scenario.reference_file)
        prediction_file = os.path.join(scenario_dir, bridge.final_response_file)
        if not os.path.exists(reference_file):
            logger.info(f"Reference file {reference_file} not found, skipping checking for task success...")
            is_successful = "N/A"
        elif not os.path.exists(prediction_file):
            logger.info(f"Prediction file {prediction_file} not found, setting task success to False...")
            is_successful = False
            success_results.append(False)
        elif judge is not None:
            result = judge.judge_file(
                reference=reference_file,
                prediction=prediction_file,
            )
            with open(os.path.join(scenario_dir, "judge_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            if judge_threshold is not None:
                is_successful = result["score"] >= judge_threshold
            else:
                is_successful = result["score"]
            success_results.append(is_successful)
        else:
            is_successful = check_if_task_success(
                reference=reference_file,
                prediction=prediction_file,
                ignore_capitalization=getattr(scenario, "ignore_capitalization", False),
                ignore_punctuation=getattr(scenario, "ignore_punctuation", False),
                clean_text=getattr(scenario, "clean_text", False),
            )
            success_results.append(is_successful)

        # Collect metrics for this scenario
        metrics = bridge.get_metrics()
        metrics["scenario_name"] = scenario.name
        metrics["scenario_directory"] = scenario_dir
        metrics["scenario_duration"] = (scenario_end - scenario_start).total_seconds()
        metrics["is_successful"] = is_successful

        # Save metrics to file
        metrics_file = os.path.join(scenario_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Scenario Metrics saved to: {metrics_file}")

        all_results.append(metrics)

        # Log scenario summary
        latency_stats = metrics["latency_stats"]
        logger.info(f"{'='*80}")
        logger.info(f"Scenario '{scenario.name}' Complete")
        logger.info(f"{'='*80}")
        logger.info(f"  Is successful: {metrics['is_successful']}")
        logger.info(f"  Total turns: {metrics['total_turns']}")
        logger.info(f"  Duration: {metrics['scenario_duration']:.1f}s")
        logger.info(f"  Latency measurements: {latency_stats['count']}")
        if latency_stats['count'] > 0:
            logger.info(f"  Mean latency: {latency_stats['mean_ms']:.1f}ms")
            logger.info(f"  P50 latency: {latency_stats['p50_ms']:.1f}ms")
            logger.info(f"  P95 latency: {latency_stats['p95_ms']:.1f}ms")

    # Save detailed results
    results_file = os.path.join(output_dir, "all_metrics.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV with latency details
    latency_csv_file = os.path.join(output_dir, "all_latencies.csv")
    with open(latency_csv_file, "w") as f:
        f.write("Scenario,User_Transcript,Agent_Transcript,Latency_ms\n")
        for result in all_results:
            scenario_name = result["scenario_name"]
            for latency in result["latencies"]:
                user_text = latency["user_transcript"].replace('"', '""')
                agent_text = latency["agent_transcript"].replace('"', '""')
                f.write(f'"{scenario_name}","{user_text}","{agent_text}",{latency["latency_ms"]:.1f}\n')

    # Save summary
    summary_file = os.path.join(output_dir, "all_summary.txt")
    success_rate = sum(success_results) / len(success_results) if len(success_results) > 0 else 0
    all_latencies = []
    for result in all_results:
        all_latencies.extend([lat["latency_ms"] for lat in result["latencies"]])
    all_latencies.sort()
    overall_latency_stats = {
        "count": len(all_latencies),
        "mean_ms": sum(all_latencies) / len(all_latencies) if len(all_latencies) > 0 else -1,
        "p50_ms": all_latencies[len(all_latencies) // 2] if len(all_latencies) > 0 else -1,
        "p95_ms": all_latencies[int(len(all_latencies) * 0.95)] if len(all_latencies) > 0 else -1,
        "min_ms": all_latencies[0] if len(all_latencies) > 0 else -1,
        "max_ms": all_latencies[-1] if len(all_latencies) > 0 else -1,
    }
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
            f.write(f"\n====== {result['scenario_name']} ======:\n")
            f.write(f"  Is successful: {result['is_successful']}\n")
            f.write(f"  Turns: {result['total_turns']}\n")
            f.write(f"  Duration: {result['scenario_duration']:.1f}s\n")
            if result['scenario_duration'] > 0:
                f.write(f"  Turns/min: {result['total_turns'] / (result['scenario_duration'] / 60):.1f}\n")
            f.write(f"  Latency Measurements: {stats['count']}\n")
            if stats['count'] > 0:
                f.write(f"    Mean: {stats['mean_ms']:.1f}ms\n")
                f.write(f"    P50: {stats['p50_ms']:.1f}ms\n")
                f.write(f"    P95: {stats['p95_ms']:.1f}ms\n")
                f.write(f"    Min: {stats['min_ms']:.1f}ms\n")
                f.write(f"    Max: {stats['max_ms']:.1f}ms\n")

        # Overall latency statistics
        f.write("\n\nOverall Latency Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Measurements: {overall_latency_stats['count']}\n")
        f.write(f"  Mean: {overall_latency_stats['mean_ms']:.1f}ms\n")
        f.write(f"  P50: {overall_latency_stats['p50_ms']:.1f}ms\n")
        f.write(f"  P95: {overall_latency_stats['p95_ms']:.1f}ms\n")
        f.write(f"  Min: {overall_latency_stats['min_ms']:.1f}ms\n")
        f.write(f"  Max: {overall_latency_stats['max_ms']:.1f}ms\n")

        f.write(f"\n\nOverall Success Rate: {success_rate*100:.2f}%\n")

    logger.info(f"{'='*80}")
    logger.info("Evaluation Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Overall Success Rate: {success_rate*100:.2f}%")
    logger.info(f"Overall Latency P95: {overall_latency_stats['p95_ms']:.1f}ms")
    logger.info(f"Overall Latency P50: {overall_latency_stats['p50_ms']:.1f}ms")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Latencies saved to: {latency_csv_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("\nScenario directories:")
    for result in all_results:
        logger.info(f"  {result['scenario_name']}: {result['scenario_directory']}")
    logger.info(f"\nTotal: {len(scenarios)} scenarios, {total_turns} turns, {total_duration:.1f}s")

    return all_results
