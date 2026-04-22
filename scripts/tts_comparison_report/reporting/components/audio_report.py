# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import random
import warnings

from scripts.tts_comparison_report.reporting.constants import SEED
from scripts.tts_comparison_report.reporting.models import AudioPair, BucketData, BucketStructure

_RNG = random.Random(SEED)


def _collect_audio_pairs(
    benchmark_name: str,
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
    bucket_structure: BucketStructure,
) -> list[AudioPair]:
    baseline_paths = bucket_baseline.get_benchmark_audio_paths(benchmark_name)
    candidate_paths = bucket_candidate.get_benchmark_audio_paths(benchmark_name)
    baseline_meta = bucket_baseline.get_benchmark_sample_meta(benchmark_name, bucket_structure)
    candidate_meta = bucket_candidate.get_benchmark_sample_meta(benchmark_name, bucket_structure)
    pairs = []

    if set(baseline_paths) != set(candidate_paths):
        raise ValueError(f"Audio sample sets differ for benchmark '{benchmark_name}'.")

    for name in baseline_paths:
        if name not in candidate_paths or name not in baseline_meta or name not in candidate_meta:
            raise ValueError(
                f"Missing matched sample '{name}' in audio paths or metadata for benchmark '{benchmark_name}'."
            )

        if baseline_meta[name].sample_id != candidate_meta[name].sample_id:
            raise ValueError(
                f"Sample id mismatch for '{name}' in benchmark '{benchmark_name}'. "
                "Probably you use different versions of buckets."
            )

        pair = AudioPair(
            context_path=baseline_meta[name].context_path,
            baseline_path=baseline_paths[name],
            candidate_path=candidate_paths[name],
            text=baseline_meta[name].gt_text,
        )
        pairs.append(pair)

    pairs.sort(key=lambda p: p.baseline_path.stem)

    return pairs


def prepare_audio_pairs(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData,
    bucket_structure: BucketStructure,
    used_benchmarks: list[str],
    samples_per_benchmark: int,
) -> dict[str, list[AudioPair]]:
    """Prepare audio pairs for the selected benchmarks.

    Args:
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.
        used_benchmarks: Benchmark names to include in the audio report.
        samples_per_benchmark: Maximum number of audio pairs to sample per benchmark.

    Returns:
        Mapping from benchmark name to sampled baseline/candidate audio pairs.

    Raises:
        ValueError: If benchmark audio sets or sample metadata are inconsistent.
    """
    pairs = {}

    for benchmark_name in used_benchmarks:
        benchmark_pairs = _collect_audio_pairs(benchmark_name, bucket_baseline, bucket_candidate, bucket_structure)
        sampled_pairs = _RNG.sample(benchmark_pairs, k=min(samples_per_benchmark, len(benchmark_pairs)))

        if len(sampled_pairs) < samples_per_benchmark:
            warnings.warn(
                f"\nBenchmark '{benchmark_name}' contains only {len(sampled_pairs)} available paired samples, "
                f"but {samples_per_benchmark} were requested.",
                stacklevel=2,
            )
        pairs[benchmark_name] = sampled_pairs

    return pairs
