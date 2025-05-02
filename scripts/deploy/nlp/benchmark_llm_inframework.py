# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import sys
import time
from typing import Any, Dict

import numpy as np

from nemo.deploy.nlp import NemoQueryLLMPyTorch

# Test prompts for benchmarking
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about artificial intelligence.",
    "What are the main differences between Python and Java?",
    "Describe the process of photosynthesis.",
    "What is the meaning of life?",
    "Explain the concept of blockchain technology.",
    "Write a brief summary of the novel '1984' by George Orwell.",
    "What are the key principles of machine learning?",
    "Describe the water cycle in nature.",
]


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Benchmarks Triton server running an in-framework Nemo model",
    )
    parser.add_argument("-u", "--url", default="0.0.0.0", type=str, help="url for the triton server")
    parser.add_argument("-mn", "--model_name", required=True, type=str, help="Name of the triton model")
    parser.add_argument("-n", "--num_queries", default=10, type=int, help="Number of queries to run")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="Number of queries to send in a batch")
    parser.add_argument("-mol", "--max_output_len", default=128, type=int, help="Max output token length")
    parser.add_argument("-tk", "--top_k", default=1, type=int, help="top_k")
    parser.add_argument("-tpp", "--top_p", default=0.0, type=float, help="top_p")
    parser.add_argument("-t", "--temperature", default=1.0, type=float, help="temperature")
    parser.add_argument("-it", "--init_timeout", default=60.0, type=float, help="init timeout for the triton server")
    parser.add_argument("-clp", "--compute_logprob", default=None, action='store_true', help="Returns log_probs")
    parser.add_argument(
        "-w", "--warmup", default=3, type=int, help="Number of warmup queries to run before benchmarking"
    )

    args = parser.parse_args(argv)
    return args


def run_benchmark(
    url: str,
    model_name: str,
    num_queries: int,
    batch_size: int,
    max_output_len: int = 128,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    compute_logprob: bool = None,
    init_timeout: float = 60.0,
    warmup: int = 3,
) -> Dict[str, Any]:
    """
    Run a benchmark of the LLM deployment.

    Args:
        url: URL of the Triton server
        model_name: Name of the model to query
        num_queries: Number of queries to run for benchmarking
        batch_size: Number of queries to send in a batch
        max_output_len: Maximum output length
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        temperature: Temperature for sampling
        compute_logprob: Whether to compute log probabilities
        init_timeout: Initialization timeout
        warmup: Number of warmup queries to run

    Returns:
        Dictionary containing benchmark results
    """
    nemo_query = NemoQueryLLMPyTorch(url, model_name)
    latencies = []
    outputs = []

    # Warmup phase
    print(f"Running {warmup} warmup queries...")
    for _ in range(warmup):
        nemo_query.query_llm(
            prompts=[TEST_PROMPTS[0]],  # Use first prompt for warmup
            max_length=max_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            compute_logprob=compute_logprob,
            init_timeout=init_timeout,
        )

    # Benchmark phase
    print(f"Running {num_queries} benchmark queries with batch size {batch_size}...")
    num_batches = (num_queries + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_queries)
        current_batch_size = end_idx - start_idx

        # Select prompts for this batch
        batch_prompts = []
        for i in range(current_batch_size):
            prompt_idx = (start_idx + i) % len(TEST_PROMPTS)
            batch_prompts.append(TEST_PROMPTS[prompt_idx])

        start_time = time.time()
        result = nemo_query.query_llm(
            prompts=batch_prompts,
            max_length=max_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            compute_logprob=compute_logprob,
            init_timeout=init_timeout,
        )
        end_time = time.time()

        # Calculate per-query latency
        batch_latency = end_time - start_time
        per_query_latency = batch_latency / current_batch_size

        for i in range(current_batch_size):
            latencies.append(per_query_latency)
            outputs.append(result[i] if isinstance(result, list) else result)
            print(f"Query {start_idx + i + 1}/{num_queries} completed in {per_query_latency:.2f} seconds")

    # Calculate statistics
    latencies = np.array(latencies)
    stats = {
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "std_latency": np.std(latencies),
        "queries_per_second": 1.0 / np.mean(latencies),
        "total_queries": num_queries,
        "warmup_queries": warmup,
        "batch_size": batch_size,
    }

    return stats


def print_benchmark_results(stats: Dict[str, Any]) -> None:
    """Print benchmark results in a formatted way."""
    print("\nBenchmark Results:")
    print("=" * 50)
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Warmup Queries: {stats['warmup_queries']}")
    print(f"Batch Size: {stats['batch_size']}")
    print("\nLatency Statistics (seconds):")
    print(f"Mean: {stats['mean_latency']:.3f}")
    print(f"Median: {stats['median_latency']:.3f}")
    print(f"95th Percentile: {stats['p95_latency']:.3f}")
    print(f"99th Percentile: {stats['p99_latency']:.3f}")
    print(f"Min: {stats['min_latency']:.3f}")
    print(f"Max: {stats['max_latency']:.3f}")
    print(f"Std Dev: {stats['std_latency']:.3f}")
    print(f"\nThroughput: {stats['queries_per_second']:.2f} queries/second")


def benchmark(argv):
    args = get_args(argv)

    stats = run_benchmark(
        url=args.url,
        model_name=args.model_name,
        num_queries=args.num_queries,
        batch_size=args.batch_size,
        max_output_len=args.max_output_len,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        compute_logprob=args.compute_logprob,
        init_timeout=args.init_timeout,
        warmup=args.warmup,
    )

    print_benchmark_results(stats)


if __name__ == '__main__':
    benchmark(sys.argv[1:])
