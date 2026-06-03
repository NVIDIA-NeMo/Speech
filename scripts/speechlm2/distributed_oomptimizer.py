#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
Restartable OOMptimizer for distributed speechlm2 training.

This script intentionally lives next to, rather than inside, ``oomptimizer.py``. The original OOMptimizer is a
single-process calibration tool that relies on catching CUDA OOM exceptions in the same Python process, emptying
enough state to keep going, and then continuing a binary search over synthetic batch sizes. That model is useful for
one GPU and for simple DDP-style memory estimates, but it becomes unreliable once the model is truly distributed.
With FSDP2, EP, and multi-node NCCL process groups, a single rank hitting CUDA OOM can leave other ranks blocked in
collectives, can poison the process group, and often prevents the training process from reaching Python exception
handling on every rank. In practice, trying to recover from those errors in-process is exactly the failure mode we
want to avoid.

The distributed OOMptimizer uses a different unit of recovery: a whole ``torchrun`` child job. A lightweight
supervisor process owns the search state and launches short-lived probe jobs. Each probe instantiates the real model
and optimizer from the provided config, creates synthetic batches through the model's OOMptimizer schema, runs one or
more candidate batch sizes, records the observed peak CUDA memory, and exits. If a candidate succeeds, rank 0 writes a
JSONL record with the batch size, bucket, peak allocated memory, peak reserved memory, and target memory. If a
candidate reaches the requested memory fraction, the worker records ``memory_target`` and stops probing that session.
If a candidate OOMs, hangs, crashes, or loses the distributed process group, the child process is allowed to die and
the supervisor interprets the missing or failed result as the first bad candidate.

The main control flow is:

1. The supervisor reads bucket boundaries and model config, then converts each bucket into synthetic input/output
   sequence lengths. SALMAutomodel has its own conversion because a single token bucket represents both audio
   locator/audio-equivalent tokens and text tokens; the ``--salm-audio-token-ratio`` option controls that split.
2. Buckets are processed from largest to smallest to preserve the memory-fragmentation behavior expected during real
   training. The next smaller bucket starts near the previous bucket's discovered batch size instead of starting from
   scratch.
3. For each bucket, the supervisor proposes one or more candidate batch sizes. Early probes expand quickly; later
   probes use the observed memory slope when possible, otherwise they fall back to doubling or bisection.
4. A probe session is launched with ``torchrun --max-restarts=0`` and a short process-group timeout. On a single node
   the supervisor uses ``--standalone``. On multiple nodes, one supervisor per node coordinates through a shared
   filesystem barrier and a rendezvous endpoint.
5. Probe workers run the actual model training step under the requested dtype. In distributed mode, workers reduce
   the maximum observed CUDA memory across ranks so the profile reflects the most memory-constrained rank.
6. The supervisor merges successful, memory-target, timeout, and failed-child observations into the same search state:
   ``max_ok`` tracks the largest usable batch size and ``min_err`` tracks the smallest known bad batch size. The search
   finishes when the relative gap between those bounds is below ``--threshold`` or the bounds differ by one.
7. The primary supervisor emits the same style of final ``bucket_duration_bins`` and ``bucket_batch_size`` output as
   the original tool, while preserving the per-probe logs for debugging.

The important design choice is that CUDA OOM recovery is delegated to process lifetime instead of Python exception
cleanup. A failed child can leave NCCL, CUDA allocator state, or model state in a bad condition, but that state dies
with the child process. The supervisor waits for GPU memory to be reclaimed, keeps the search state on the host, and
continues with the next candidate. This makes the tool slower than an in-process loop, but it matches the failure
semantics of FSDP2/EP training and allows us to profile batch sizes that put real distributed jobs near a target GPU
memory pressure without manually babysitting every OOM.
"""

import importlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Literal

import click
import lightning.pytorch as pl
import torch
from lhotse import compute_num_samples
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from nemo.collections.speechlm2 import SALM, SALMAutomodel, SALMWithAsrDecoder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging
from nemo.utils.trainer_utils import resolve_trainer_cfg


class ProfilingBatchGenerator:
    """
    ProfilingBatchGenerator is used to generate artificial mini-batches for model training
    and tracking the progress of batch size optimization.

    The high-level usage API is the following::

        >>> gen = ProfilingBatchGenerator(schema)
        ... finished = False
        ... while not finished:
        ...     batch = gen(input_seq_len, output_seq_len)
        ...     try:
        ...         training_step(model, batch)
        ...         oom = False
        ...     except torch.cuda.OutOfMemoryError:
        ...         oom = True
        ...     finished = gen.advance(oom)
        ... solution = gen.max_batch_size  # The solution of the search problem.
        ... gen.reset()  # Can re-use for other sequence lengths now.

    The search terminates once the difference between max working batch size and min OOM batch size
    divided by the latter is smaller than ``rel_gap_thresh`` that difference amounts to a single element.
    For example, a max working batch size is 96 and min OOM batch size is 100 indicates a gap of 0.04,
    which would terminate the search with threshold of 0.05.

    In order to generate mini-batches compatible with a given model, the generator:

    * accepts a ``schema`` argument in its constructor, and

    * accepts input/output sequence lengths in each call to generate a mini-batch.

    ``schema`` has the following structure::


        >>> {
        ...     "cls": tuple | MyBatchType,
        ...     "inputs": [
        ...         {
        ...             "type": NeuralType(...) | Literal["dummy"],
        ...             "seq_length": Literal["input", "output"],
        ...             "vocab_size": int,  # optional, required only for LabelsType
        ...             "name": str,  # optional, indicates kwarg
        ...         },
        ...         ...,
        ...     ]
        ... }

    ``cls`` indicates how we should construct the mini-batch. Typically you can just use ``tuple`` for most
    batch schemas. However, if the model expects a specific, e.g., dataclass, you can tell ``ProfilingBatchGenerator``
    to use it. The mini-batch object will be constructed using the items in ``inputs``.

    Each element of ``inputs`` specifies a NeMo NeuralType which needs to have a defined ``elements_type``.
    The supported types are ``AudioSignal``, ``LengthsType`` and ``LabelsType``.
    If "type" is not a NeuralType, we interpret that as a placeholder tensor that's not relevant but expected
    by the model/batch constructor. In addition, ``"seq_length"`` key is used to determine whether we should apply
    input or output sequence length to a given tensor.

    Optional keys:

    * ``vocab_size`` is required for ``LabelsType`` so that we can generate proper label values.

    * ``name`` is required if objects of ``cls`` have to be constructed using keyword arguments.

    A simple schema example for a model using audio/lengths tensor pair (unsupervised/self-supervised)::

        >>> {
        ...     "cls": tuple,
        ...     "inputs": [
        ...         {"type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
        ...         {"type": NeuralType(("B"), LengthsType()), "seq_length": "input"},
        ...     ]
        ... }

    """

    def __init__(
        self,
        schema: dict,
        start_batch_size: int = 32,
        rel_gap_thresh: float = 0.05,
        device: str = "cuda",
        float_dtype: torch.dtype = torch.float32,
    ):
        self.schema = schema
        self.start_batch_size = start_batch_size
        self.rel_gap_thresh = rel_gap_thresh
        self.device = device
        self.float_dtype = float_dtype
        self.reset()

    def __call__(self, input_seq_length: int, output_seq_length: int):
        B = self._current
        select_seq_length = {"input": input_seq_length, "output": output_seq_length}
        batch = []
        names = []
        for item in self.schema["inputs"]:
            nt = item["type"]
            if isinstance(nt, str) and nt == "constant":
                if isinstance(val := item["value"], str) and val == "batch":
                    tnsr = torch.tensor([B], dtype=torch.long, device=self.device)
                else:
                    tnsr = torch.tensor([val], dtype=torch.long, device=self.device)
            elif not isinstance(nt, NeuralType):  # placeholder
                tnsr = torch.tensor([])
            elif isinstance(nt.elements_type, AudioSignal):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.randn(B, seq_length, dtype=self.float_dtype, device=self.device)
            elif isinstance(nt.elements_type, LengthsType):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.ones(B, dtype=torch.long, device=self.device) * seq_length
            elif isinstance(nt.elements_type, MaskType):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.ones(B, seq_length, device=self.device, dtype=torch.bool)
            elif isinstance(nt.elements_type, LabelsType):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.randint(0, item["vocab_size"], size=(B, seq_length), device=self.device)
                for token_id in item.get("excluded_token_ids", []):
                    tnsr.masked_fill_(tnsr == token_id, 0)
                for position, token_id in item.get("forced_token_ids", {}).items():
                    position = int(position)
                    if position < 0:
                        position += seq_length
                    if 0 <= position < seq_length:
                        tnsr[:, position] = token_id
            else:
                raise RuntimeError("Unexpected item in oomptimizer schema: {item}")
            batch.append(tnsr)
            names.append(item.get("name"))
        args = [elem for name, elem in zip(names, batch) if name is None]
        kwargs = {name: elem for name, elem in zip(names, batch) if name is not None}
        if not kwargs and self.schema["cls"] == tuple:
            return tuple(args)
        return self.schema["cls"](*args, **kwargs)

    @property
    def max_batch_size(self) -> int | None:
        """
        Return the solution of the batch size search problem.
        It will keep returning None until the search is done.
        """
        if (
            self._max_ok is not None
            and self._min_err is not None
            and (self.current_rel_gap <= self.rel_gap_thresh or self._min_err - self._max_ok <= 1)
        ):
            return self._max_ok
        return None

    @property
    def current_rel_gap(self) -> float | None:
        """
        Return the current gap between the largest batch that works and the smallest batch that triggers OOM.
        The gap is defined as the batch size difference divided by the larger element.
        E.g., if the best found batch size is 95 and the smallest that triggers OOM is 100, the gap is 0.05.
        """
        if self._min_err is None or self._max_ok is None:
            return None
        return (self._min_err - self._max_ok) / self._min_err

    def reset(self):
        """Reset the generator to prepare it for a new search."""
        self._current = self.start_batch_size
        self._max_ok = None  # max batch size that works
        self._min_err = None  # min batch size that doesn't work

    def advance(self, oom: bool) -> bool:
        """
        Adjusts the current batch size based on the outcome.
        Returns a bool indicating whether the calibration is complete.
        """
        if self.max_batch_size is not None:
            return True

        if oom:
            # Training step failed with OOM.
            # Update the minimum known batch size that causes an error.
            self._min_err = min(float("inf") if self._min_err is None else self._min_err, self._current)
            # Training step failed on OOM
            if self._max_ok is None:
                # We haven't found a batch size that works yet, keep going 2x down.
                self._current = round(self._current / 2)
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)
        else:
            # Training step successful.
            # Update the maximum known batch size that works.
            self._max_ok = max(-1 if self._max_ok is None else self._max_ok, self._current)
            if self._min_err is None:
                # We haven't found a batch size that causes an error yet, keep going 2x higher
                self._current *= 2
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)

        return False


class FloatList(click.Option):
    """Support passing bucket duration bins as [1.1,2.5,5.6,...]"""

    name = "list[float]"

    def type_cast_value(self, ctx, value):
        if isinstance(value, list) and all(isinstance(v, float) for v in value):
            return value
        try:
            import ast

            ans = ast.literal_eval(value)
            if isinstance(ans[0], list):
                ans = [tuple(item) for item in ans]
            return ans
        except ValueError:
            raise click.BadParameter(value)


def _parse_int_list(value: str) -> list[int]:
    if value.startswith("["):
        import ast

        parsed = ast.literal_eval(value)
        return [int(item) for item in parsed]
    return [int(item) for item in value.split(",") if item]


def _is_2d_bucketing(buckets) -> bool:
    return all(
        isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, Number) for v in item)
        for item in buckets
    )


def _count_visible_devices() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 1
    return max(1, len([line for line in result.stdout.splitlines() if line.strip()]))


def _trainer_devices_to_int(devices) -> int:
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str):
        if devices.isdigit():
            return int(devices)
        if devices in ("auto", "-1"):
            return _count_visible_devices()
    return 1


def _query_gpu_memory_mib() -> list[tuple[int, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    memory = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        used, total = line.split(",")
        memory.append((int(used.strip()), int(total.strip())))
    return memory


def _wait_for_gpu_memory_reclaim(
    timeout_seconds: float, tolerance_mb: int, poll_interval_seconds: float = 2.0
) -> None:
    if timeout_seconds <= 0:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            memory = _query_gpu_memory_mib()
        except (OSError, subprocess.CalledProcessError):
            return
        if memory and max(used for used, _ in memory) <= tolerance_mb:
            return
        time.sleep(poll_interval_seconds)


def _wait_for_path(path: Path, timeout_seconds: float, description: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {description}: {path}")


def _shared_file_barrier(
    log_dir: Path,
    name: str,
    node_rank: int,
    nnodes: int,
    timeout_seconds: float = 300.0,
) -> None:
    if nnodes <= 1:
        return
    barrier_dir = log_dir / ".supervisor_barriers" / name
    barrier_dir.mkdir(parents=True, exist_ok=True)
    marker = barrier_dir / f"rank_{node_rank}.ready"
    marker.write_text(str(os.getpid()))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if len(list(barrier_dir.glob("rank_*.ready"))) >= nnodes:
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out in OOMptimizer supervisor barrier {name}.")


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w") as f:
        json.dump(data, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _get_distributed_supervisor_seq_lens(
    module_name: str,
    cfg,
    buckets,
    ratio: float,
    salm_audio_token_ratio: float,
) -> list[tuple[int, int]]:
    is_2d_bucketing = _is_2d_bucketing(buckets)
    if module_name.endswith("SALMAutomodel"):
        sampling_rate = OmegaConf.select(cfg, "data.train_ds.sample_rate", default=16000)
        token_equivalent_duration = OmegaConf.select(cfg, "data.train_ds.token_equivalent_duration", default=0.08)

        def salm_automodel_lens(bucket):
            audio_tokens = max(1, int(math.ceil(salm_audio_token_ratio * bucket)))
            text_tokens = max(2, int(math.ceil((1.0 - salm_audio_token_ratio) * bucket)))
            audio_len = int(math.ceil(audio_tokens * token_equivalent_duration * sampling_rate))
            return audio_len, text_tokens

        return [salm_automodel_lens(bucket) for bucket in buckets]
    if module_name.endswith("SALM") or module_name.endswith("SALMWithAsrDecoder"):
        return [(bucket, bucket) for bucket in buckets]
    if is_2d_bucketing:
        return [(int(input_len), int(output_len)) for input_len, output_len in buckets]
    return [(compute_num_samples(bucket, sampling_rate=16000), int(math.ceil(ratio * bucket))) for bucket in buckets]


def _read_probe_records(path: Path) -> list[dict]:
    records = []
    seen = set()
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                key = json.dumps(record, sort_keys=True)
                if key not in seen:
                    records.append(record)
                    seen.add(key)
    return records


def _append_probe_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _first_unreported_candidate(candidates: list[int], records: list[dict]) -> int | None:
    reported = {int(record["batch_size"]) for record in records}
    for candidate in candidates:
        if candidate not in reported:
            return candidate
    return None


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _terminate_process_group(proc: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _predict_batch_for_target(ok_points: list[tuple[int, int]], target_memory: float) -> int | None:
    points_by_batch = {}
    for batch_size, peak_allocated in ok_points:
        if peak_allocated > 0:
            points_by_batch[int(batch_size)] = int(peak_allocated)
    points = sorted(points_by_batch.items())
    if len(points) < 2:
        return None
    b1, p1 = points[-2]
    b2, p2 = points[-1]
    if b2 <= b1 or p2 <= p1:
        return None
    slope = (p2 - p1) / (b2 - b1)
    intercept = p2 - slope * b2
    predicted = math.floor((target_memory - intercept) / slope)
    if predicted <= b2:
        return None
    return int(min(predicted, max(b2 + 1, b2 * 2)))


def _make_probe_plan(
    current: int,
    min_err: int | None,
    ok_points: list[tuple[int, int]],
    target_memory: float,
) -> list[int]:
    current = max(1, int(current))
    if min_err is not None:
        plan = [min(current, max(1, min_err - 1))]
        if ok_points:
            while len(plan) < 3 and plan[-1] < min_err - 1:
                next_candidate = plan[-1] + max(1, round((min_err - plan[-1]) / 2))
                next_candidate = min(next_candidate, min_err - 1)
                if next_candidate in plan:
                    break
                plan.append(next_candidate)
        return plan

    plan = [current]
    if len(ok_points) < 2:
        while len(plan) < 3:
            plan.append(plan[-1] * 2)
    else:
        predicted = _predict_batch_for_target(ok_points, target_memory)
        plan.append(predicted if predicted is not None else plan[-1] * 2)

    deduped = []
    for item in plan:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _next_batch_size(
    max_ok: int | None,
    min_err: int | None,
    ok_points: list[tuple[int, int]],
    target_memory: float,
) -> int:
    if max_ok is None:
        assert min_err is not None
        return max(1, min_err // 2)
    if min_err is not None:
        return max_ok + max(1, round((min_err - max_ok) / 2))
    predicted = _predict_batch_for_target(ok_points, target_memory)
    return predicted if predicted is not None else max(1, max_ok * 2)


def _search_finished(max_ok: int | None, min_err: int | None, threshold: float) -> bool:
    if max_ok is None or min_err is None:
        return False
    return (min_err - max_ok) / min_err <= threshold or min_err - max_ok <= 1


def _torchrun_launcher() -> list[str]:
    torchrun = Path(sys.executable).with_name("torchrun")
    if torchrun.exists() and os.access(torchrun, os.X_OK):
        return [str(torchrun)]
    return [sys.executable, "-m", "torch.distributed.run"]


def _is_torchrun_worker() -> bool:
    return bool(
        os.environ.get("TORCHELASTIC_RUN_ID")
        or ("LOCAL_RANK" in os.environ and "RANK" in os.environ and "GROUP_RANK" in os.environ)
    )


def _clean_torchrun_launcher_env(env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    # Supervisors may be launched by srun with one task per node. If these rank variables leak into the torchrun
    # workers, Lightning prefers SLURMEnvironment over TorchElastic and sees only the supervisor task world.
    for name in (
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
        "SLURM_GTIDS",
        "SLURM_STEP_TASKS_PER_NODE",
    ):
        env.pop(name, None)
    for name in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    return env


def _run_probe_session(
    *,
    module_name: str,
    config_path: str,
    bucket,
    seq_len_in: int,
    seq_len_out: int,
    batch_sizes: list[int],
    nproc_per_node: int,
    nnodes: int,
    node_rank: int,
    rdzv_endpoint: str | None,
    rdzv_id: str,
    memory_fraction: float,
    dtype: str,
    ddp: bool,
    salm_audio_token_ratio: float,
    distributed_timeout_seconds: float,
    probe_timeout_seconds: float,
    log_dir: Path,
    probe_index: int,
) -> tuple[list[dict], int | None, Path, int | None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_bucket = str(bucket).replace("/", "_").replace("[", "").replace("]", "").replace(",", "_")
    result_path = log_dir / f"probe_{probe_index:04d}_bucket_{safe_bucket}_bs_{batch_sizes[0]}.jsonl"
    log_suffix = "" if nnodes <= 1 else f"_node{node_rank}"
    log_path = log_dir / f"probe_{probe_index:04d}_bucket_{safe_bucket}_bs_{batch_sizes[0]}{log_suffix}.log"
    outcome_path = log_dir / f"probe_{probe_index:04d}_bucket_{safe_bucket}_bs_{batch_sizes[0]}_outcome.json"
    if node_rank == 0:
        result_path.unlink(missing_ok=True)
        outcome_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)

    cmd = [
        *_torchrun_launcher(),
        f"--nnodes={nnodes}",
        f"--nproc-per-node={nproc_per_node}",
    ]
    if nnodes <= 1:
        cmd.append("--standalone")
    else:
        if not rdzv_endpoint:
            raise click.ClickException("--rdzv-endpoint is required when supervisor nnodes > 1.")
        cmd.extend(
            [
                f"--node-rank={node_rank}",
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint={rdzv_endpoint}",
                f"--rdzv-id={rdzv_id}",
            ]
        )
    cmd.extend(
        [
            "--max-restarts=0",
            "--monitor-interval=1",
            str(Path(__file__).resolve()),
            "--module-name",
            module_name,
            "--config-path",
            config_path,
            "--memory-fraction",
            str(memory_fraction),
            "--dtype",
            dtype,
            "--salm-audio-token-ratio",
            str(salm_audio_token_ratio),
            "--distributed-timeout-seconds",
            str(distributed_timeout_seconds),
            "--probe-batch-sizes",
            ",".join(str(item) for item in batch_sizes),
            "--probe-seq-len-in",
            str(seq_len_in),
            "--probe-seq-len-out",
            str(seq_len_out),
            "--probe-result-path",
            str(result_path),
            "--probe-bucket",
            str(bucket),
            "--no-distributed-supervisor",
        ]
    )
    cmd.append("--ddp" if ddp else "--no-ddp")

    env = _clean_torchrun_launcher_env(os.environ)
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _shared_file_barrier(log_dir, f"probe_{probe_index:04d}_start", node_rank, nnodes)

    with log_path.open("w") as log_f:
        log_f.write(f"COMMAND: {' '.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid,
        )
        timed_out = False
        try:
            returncode = proc.wait(timeout=probe_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc)
            returncode = proc.returncode
            log_f.write(f"\nOOMPTIMIZER_PROBE_TIMEOUT after {probe_timeout_seconds}s\n")
            log_f.flush()

    if nnodes <= 1 or node_rank == 0:
        records = _read_probe_records(result_path)
        failed_candidate = None
        if timed_out or returncode != 0:
            if records and records[-1].get("status") not in ("ok", "memory_target"):
                failed_candidate = None
            else:
                failed_candidate = _first_unreported_candidate(batch_sizes, records)
                if failed_candidate is None and records:
                    failed_candidate = int(records[-1]["batch_size"])
        outcome = {
            "records": records,
            "failed_candidate": failed_candidate,
            "log_path": str(log_path),
            "returncode": returncode,
        }
        if nnodes > 1:
            _write_json_atomic(outcome_path, outcome)
    else:
        _wait_for_path(outcome_path, probe_timeout_seconds + 60.0, "multi-node probe outcome")
        outcome = _read_json(outcome_path)
        records = outcome["records"]
        failed_candidate = outcome["failed_candidate"]
        log_path = Path(outcome["log_path"])
        returncode = outcome["returncode"]

    _shared_file_barrier(log_dir, f"probe_{probe_index:04d}_done", node_rank, nnodes)
    return records, failed_candidate, log_path, returncode


def _run_distributed_supervisor(
    *,
    pretrained_name: str | None,
    module_name: str | None,
    config_path: str | None,
    buckets,
    threshold: float,
    start_batch_size: int,
    ratio: float,
    memory_fraction: float,
    dtype: str,
    ddp: bool,
    salm_audio_token_ratio: float,
    distributed_timeout_seconds: float,
    nproc_per_node: int | None,
    supervisor_nnodes: int | None,
    supervisor_node_rank: int | None,
    rdzv_endpoint: str | None,
    probe_log_dir: str | None,
    probe_timeout_seconds: float,
    probe_memory_reclaim_timeout_seconds: float,
    probe_memory_tolerance_mb: int,
) -> None:
    assert pretrained_name is None, "--pretrained-name is not supported yet for Duplex S2S"
    assert config_path is not None, "--module-name requires --config-path to be specified as well."
    assert module_name is not None, "--config-path requires --module-name to be specified as well."

    cfg = OmegaConf.load(config_path)
    requested_devices = _trainer_devices_to_int(OmegaConf.select(cfg, "trainer.devices", default=1))
    nproc_per_node = int(nproc_per_node or requested_devices)
    if nproc_per_node <= 1:
        raise click.ClickException("Distributed supervisor requires nproc_per_node > 1.")
    supervisor_nnodes = int(
        supervisor_nnodes or os.environ.get("OOMPTIMIZER_SUPERVISOR_NNODES") or os.environ.get("SLURM_NNODES") or 1
    )
    supervisor_node_rank = int(
        supervisor_node_rank
        if supervisor_node_rank is not None
        else os.environ.get("OOMPTIMIZER_SUPERVISOR_NODE_RANK")
        or os.environ.get("SLURM_NODEID")
        or os.environ.get("SLURM_PROCID")
        or 0
    )
    rdzv_endpoint = rdzv_endpoint or os.environ.get("OOMPTIMIZER_RDZV_ENDPOINT")
    if supervisor_nnodes > 1 and not rdzv_endpoint:
        master_addr = os.environ.get("MASTER_ADDR") or os.environ.get("SLURM_MASTER_NODE")
        master_port = os.environ.get("MASTER_PORT") or os.environ.get("OOMPTIMIZER_RDZV_PORT") or "29500"
        if master_addr:
            rdzv_endpoint = f"{master_addr}:{master_port}"
    if supervisor_nnodes > 1 and not rdzv_endpoint:
        raise click.ClickException(
            "Multi-node distributed supervisor requires --rdzv-endpoint or OOMPTIMIZER_RDZV_ENDPOINT."
        )
    if not 0 <= supervisor_node_rank < supervisor_nnodes:
        raise click.ClickException(
            f"Supervisor node rank must be in [0, {supervisor_nnodes}); got {supervisor_node_rank}."
        )
    is_primary_supervisor = supervisor_node_rank == 0

    max_seq_lens = _get_distributed_supervisor_seq_lens(module_name, cfg, buckets, ratio, salm_audio_token_ratio)
    gpu_memory = _query_gpu_memory_mib()
    if not gpu_memory:
        raise click.ClickException("Could not query GPU memory via nvidia-smi.")
    target_memory = memory_fraction * min(total for _, total in gpu_memory) * 1024 * 1024

    log_dir = (
        Path(probe_log_dir)
        if probe_log_dir
        else Path(config_path).with_suffix("").parent / (Path(config_path).stem + "_oomptimizer_probes")
    )
    if is_primary_supervisor:
        click.echo("Starting restartable distributed profiling.")
        click.echo(f"Probe logs: {log_dir}")
        click.echo(
            f"Using nnodes={supervisor_nnodes}, nproc_per_node={nproc_per_node}; "
            f"target allocated memory={target_memory / (1024 ** 3):.2f}GiB"
        )

    profile = {}
    next_start = max(1, start_batch_size)
    probe_index = 0
    for bucket, (seq_len_in, seq_len_out) in reversed(list(zip(buckets, max_seq_lens))):
        if is_primary_supervisor:
            click.echo(f"The current sequence lengths are: input={seq_len_in} output={seq_len_out}.")
        max_ok = None
        min_err = None
        ok_points = []
        current = next_start
        last_log = None

        while not _search_finished(max_ok, min_err, threshold):
            plan = _make_probe_plan(current, min_err, ok_points, target_memory)
            if is_primary_supervisor:
                click.echo(
                    f"\tProbe plan for bucket={bucket}: {plan} "
                    f"(max_ok={max_ok}, min_err={min_err}, ok_points={len(ok_points)})"
                )
            records, failed_candidate, log_path, returncode = _run_probe_session(
                module_name=module_name,
                config_path=config_path,
                bucket=bucket,
                seq_len_in=seq_len_in,
                seq_len_out=seq_len_out,
                batch_sizes=plan,
                nproc_per_node=nproc_per_node,
                nnodes=supervisor_nnodes,
                node_rank=supervisor_node_rank,
                rdzv_endpoint=rdzv_endpoint,
                rdzv_id=f"{os.environ.get('SLURM_JOB_ID', os.getpid())}_{probe_index:04d}",
                memory_fraction=memory_fraction,
                dtype=dtype,
                ddp=ddp,
                salm_audio_token_ratio=salm_audio_token_ratio,
                distributed_timeout_seconds=distributed_timeout_seconds,
                probe_timeout_seconds=probe_timeout_seconds,
                log_dir=log_dir,
                probe_index=probe_index,
            )
            probe_index += 1
            last_log = log_path
            for record in records:
                batch_size = int(record["batch_size"])
                peak_allocated = int(record.get("peak_allocated", 0))
                status = record["status"]
                if status == "ok":
                    max_ok = max(batch_size, -1 if max_ok is None else max_ok)
                    ok_points.append((batch_size, peak_allocated))
                    if is_primary_supervisor:
                        click.echo(
                            f"\tOK batch={batch_size}; peak={peak_allocated / (1024 ** 3):.2f}GiB "
                            f"({peak_allocated / target_memory:.1%} of target)"
                        )
                elif status == "memory_target":
                    max_ok = max(batch_size, -1 if max_ok is None else max_ok)
                    min_err = min(batch_size + 1, int(1e18) if min_err is None else min_err)
                    ok_points.append((batch_size, peak_allocated))
                    if is_primary_supervisor:
                        click.echo(
                            f"\tMEMORY TARGET batch={batch_size}; peak={peak_allocated / (1024 ** 3):.2f}GiB "
                            f"({peak_allocated / target_memory:.1%} of target)"
                        )
                else:
                    min_err = min(batch_size, int(1e18) if min_err is None else min_err)
                    if is_primary_supervisor:
                        click.echo(f"\tFAILED batch={batch_size}; status={status}")
            if failed_candidate is not None:
                min_err = min(failed_candidate, int(1e18) if min_err is None else min_err)
                if is_primary_supervisor:
                    click.echo(f"\tFAILED batch={failed_candidate}; child_returncode={returncode}; log={log_path}")
                _wait_for_gpu_memory_reclaim(
                    probe_memory_reclaim_timeout_seconds,
                    probe_memory_tolerance_mb,
                )

            if max_ok is None and min_err is not None and min_err <= 1:
                if is_primary_supervisor:
                    click.secho(
                        f"\tBatch size 1 failed for bucket={bucket}; recording max_batch_size=0 and continuing.",
                        fg="yellow",
                    )
                max_ok = 0
            if not _search_finished(max_ok, min_err, threshold):
                current = _next_batch_size(max_ok, min_err, ok_points, target_memory)

        if is_primary_supervisor:
            click.secho(
                f"=> Optimal setting for bucket={bucket} (input={seq_len_in} output={seq_len_out}) "
                f"is max_batch_size={max_ok}",
                fg="green",
            )
        profile[(bucket, seq_len_in, seq_len_out)] = max_ok
        next_start = max(max_ok + 1, int(math.ceil(max_ok * 1.5)))

    if is_primary_supervisor:
        _emit_profile(profile, buckets, memory_fraction, ddp, dtype)


def _emit_profile(profile: dict, buckets, memory_fraction: float, ddp: bool, dtype: str) -> None:
    profile = dict(reversed(list(profile.items())))
    click.echo("The 1st stage profile is:")
    for (bucket, seq_len_in, seq_len_out), bs in profile.items():
        click.echo(f"Bucket={bucket} (input={seq_len_in} output={seq_len_out}) => max_batch_size={bs}")

    if _is_2d_bucketing(buckets):
        final_profile = [["[" + ",".join(map(str, b)) + "]", bs] for (b, _, __), bs in profile.items()]
    else:
        click.echo("Bucket merging stage...")
        final_profile = []
        for idx, ((bucket, seq_len_in, seq_len_out), bs) in enumerate(profile.items()):
            if idx == 0:
                final_profile.append([bucket, bs])
                continue
            if bs == final_profile[-1][1]:
                click.echo(f"Merging bucket {idx} with bucket {idx-1} due to identical batch sizes.")
                final_profile[-1][0] = bucket
                continue
            final_profile.append([bucket, bs])

    click.secho(f"The profile was created with the following settings:")
    click.secho(f"* using {memory_fraction:.1%} of available GPU RAM.")
    click.secho(f"* {'' if ddp else 'not '}simulating DDP memory overhead.")
    click.secho(f"* using AMP with dtype={dtype}.")
    click.secho("The final profile is:", bold=True)
    click.secho("\tbucket_duration_bins=[" + ",".join(str(seqlen) for seqlen, bs in final_profile) + "]", bold=True)
    click.secho("\tbucket_batch_size=[" + ",".join(str(bs) for seqlen, bs in final_profile) + "]", bold=True)


def _is_oom_like(error: RuntimeError) -> bool:
    error_msg = str(error)
    return (
        "cuFFT error: CUFFT_INTERNAL_ERROR" in error_msg
        or "CUDA out of memory" in error_msg
        or "CUDACachingAllocator" in error_msg
        or "NCCL" in error_msg
    )


@click.command(context_settings={'show_default': True})
@click.option(
    "-n",
    "--pretrained-name",
    type=str,
    default=None,
    help="Name of a pretrained model to use, e.g. 'nvidia/canary-1b'.",
)
@click.option(
    "-m",
    "--module-name",
    type=str,
    default=None,
    help="Full path to NeMo's module corresponding to CONFIG_PATH, e.g. 'nemo.collections.asr.models.EncDecMultiTaskModel'.",
)
@click.option(
    "-c", "--config-path", type=str, default=None, help="Path to the training configuration file for MODULE_NAME."
)
@click.option(
    "-b",
    "--buckets",
    cls=FloatList,
    default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    help="List of upper-bound bucket bins (i.e. first bucket is [0.0 - item0), second bucket is [item0 - item1), etc.). "
    "We also support a nested list for 2D bucketing, e.g. [[2.0, 10],[2.0,20],[4.5,15],[4.5,30],...], "
    "where each item is a pair of (max_input_seq_len, max_output_seq_len) for a given bucket.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.05,
    help="Search stopping criterion in range [0, 1], lower is more precise. Interpret as the uncerainty gap, i.e. (min_oom_batch_size - max_ok_batch_size) / min_oom_batch_size.",
)
@click.option("-s", "--start-batch-size", type=int, default=32, help="Initial batch size to start the search from.")
@click.option(
    "-r",
    "--ratio",
    type=float,
    default=12,  # conservative estimate towards longer transcripts
    help="The output_sequence_length to input_sequence_length ratio for the purpose of determing the maximum output sequence lengths. "
    "The interpretation depends on input and output modalities. Examples: for audio->text it's tokens per second. "
    "For text->audio it's seconds per token. For audio->audio it's output seconds per input second. "
    "For text->text it's output tokens per input token. "
    "In general larger ratio means longer output sequences and increased memory consumption. "
    "The default value is set adequately for automatic speech recognition. "
    "This argument is ignored when 2D buckets are provided to --buckets option.",
)
@click.option(
    "-f",
    "--memory-fraction",
    type=float,
    default=0.9,
    help="Limits the use of CUDA memory for this process to MEMORY_FRACTION of the total device memory. "
    "By default we force 5% memory to be unused to account for non-training-loop related CUDA memory usage"
    "in actual training scripts.",
)
@click.option(
    "-y",
    "--dtype",
    default="bfloat16",
    help="Float precision to use for computation (used together with autocast).",
)
@click.option(
    "--ddp/--no-ddp",
    type=bool,
    default=True,
    help="Whether we should simulate DDP GPU RAM usage. Stores an extra copy of the model in GPU memory. Enabled by default.",
)
@click.option(
    "--salm-audio-token-ratio",
    type=float,
    default=0.75,
    help="For SALMAutomodel 1D token buckets, fraction of the bucket represented by audio-equivalent tokens.",
)
@click.option(
    "--distributed-timeout-seconds",
    type=float,
    default=15.0,
    help="Process-group timeout used for distributed profiling so collective failures surface quickly.",
)
@click.option(
    "--distributed-supervisor/--no-distributed-supervisor",
    type=bool,
    default=True,
    help="Use restartable torchrun child probes for multi-GPU configs instead of in-process OOM recovery.",
)
@click.option(
    "--nproc-per-node",
    type=int,
    default=None,
    help="Number of local workers used by the distributed supervisor. Defaults to trainer.devices.",
)
@click.option(
    "--supervisor-nnodes",
    type=int,
    default=None,
    help="Number of nodes coordinated by the distributed supervisor. Defaults to OOMPTIMIZER_SUPERVISOR_NNODES or SLURM_NNODES.",
)
@click.option(
    "--supervisor-node-rank",
    type=int,
    default=None,
    help="Node rank for the distributed supervisor. Defaults to OOMPTIMIZER_SUPERVISOR_NODE_RANK, SLURM_NODEID, or SLURM_PROCID.",
)
@click.option(
    "--rdzv-endpoint",
    type=str,
    default=None,
    help="Torchrun rendezvous endpoint for multi-node supervisor probes. Defaults to OOMPTIMIZER_RDZV_ENDPOINT.",
)
@click.option(
    "--probe-log-dir",
    type=str,
    default=None,
    help="Directory where distributed supervisor probe logs and JSONL results are written.",
)
@click.option(
    "--probe-timeout-seconds",
    type=float,
    default=900.0,
    help="Wall-clock timeout for one distributed probe session.",
)
@click.option(
    "--probe-memory-reclaim-timeout-seconds",
    type=float,
    default=60.0,
    help="How long the supervisor waits for GPU memory to be reclaimed after a failed child probe.",
)
@click.option(
    "--probe-memory-tolerance-mb",
    type=int,
    default=1024,
    help="GPU memory threshold used by the supervisor reclaim wait.",
)
@click.option("--probe-batch-sizes", type=str, default=None, hidden=True)
@click.option("--probe-seq-len-in", type=int, default=None, hidden=True)
@click.option("--probe-seq-len-out", type=int, default=None, hidden=True)
@click.option("--probe-result-path", type=str, default=None, hidden=True)
@click.option("--probe-bucket", type=str, default=None, hidden=True)
def oomptimizer(
    pretrained_name: str | None,
    module_name: str | None,
    config_path: str | None,
    buckets: list[float],
    threshold: float,
    start_batch_size: int,
    ratio: float,
    memory_fraction: float,
    dtype: str,
    ddp: bool,
    salm_audio_token_ratio: float,
    distributed_timeout_seconds: float,
    distributed_supervisor: bool,
    nproc_per_node: int | None,
    supervisor_nnodes: int | None,
    supervisor_node_rank: int | None,
    rdzv_endpoint: str | None,
    probe_log_dir: str | None,
    probe_timeout_seconds: float,
    probe_memory_reclaim_timeout_seconds: float,
    probe_memory_tolerance_mb: int,
    probe_batch_sizes: str | None,
    probe_seq_len_in: int | None,
    probe_seq_len_out: int | None,
    probe_result_path: str | None,
    probe_bucket: str | None,
):
    """
    OOMptimizer finds the optimal batch sizes for training your model with bucketing dataloading.
    It performs a search over batch sizes until it converges by measuring the GPU memory usage for
    a model's training step and optimizer update.

    \b
    There are two main usage patterns: for using a pretrained model or an untrained model configuration.
    The latter is more flexible but requires the user to provide two separate arguments. Examples:
    * python oomptimizer.py --pretrained-name nvidia/canary-1b
    * python oomptimizer.py --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
        --config-path examples/asr/conf/speech_multitask/fast-conformer_aed.yaml

    Dynamic bucketing is notoriously difficult to tune as you risk running into CUDA OOM many steps into the training.
    In order to simplify finding the optimal settings, OOMptimizer scans each bucket to find the maximum possible
    batch size that doesn't trigger a CUDA OOM.

    \b
    The suggested workflow is the following:
    1) Run scripts/speech_recognition/estimate_duration_bins.py to get the duration distribution of your data.
        (consider running estimate_duration_bins_2d.py for models with a strong dependency on output sequence length
        such as attention-encoder-decoder models).
    2) Run OOMptimizer to find the optimal batch sizes for your specific model, optimizer, and GPU.
    3) Use these optimal settings in your actual training script and enjoy optimal GPU utilization OOM-free.

    In the unlikely event that OOMptimizer bucket batch sizes are still leading to OOMs,
    please try a lower setting of the MEMORY_FRACTION option, e.g. 0.75 (75% of GPU memory).
    This may be required in very complex setups where there are additional GPU RAM loads that can't be anticipated
    through the combination of training_step and optimizer update.
    """
    assert pretrained_name is None, "--pretrained-name is not supported yet for Duplex S2S"
    if all(opt is None for opt in (pretrained_name, module_name, config_path)):
        click.secho(
            "You need to provide either PRETRAINED_NAME or the pair of MODULE_NAME and CONFIG_PATH.", fg="yellow"
        )
        sys.exit(1)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if probe_batch_sizes is None and distributed_supervisor and not _is_torchrun_worker() and config_path is not None:
        cfg_for_supervisor = OmegaConf.load(config_path)
        requested_devices = _trainer_devices_to_int(OmegaConf.select(cfg_for_supervisor, "trainer.devices", default=1))
        requested_devices = int(nproc_per_node or requested_devices)
        if requested_devices > 1:
            _run_distributed_supervisor(
                pretrained_name=pretrained_name,
                module_name=module_name,
                config_path=config_path,
                buckets=buckets,
                threshold=threshold,
                start_batch_size=start_batch_size,
                ratio=ratio,
                memory_fraction=memory_fraction,
                dtype=dtype,
                ddp=ddp,
                salm_audio_token_ratio=salm_audio_token_ratio,
                distributed_timeout_seconds=distributed_timeout_seconds,
                nproc_per_node=nproc_per_node,
                supervisor_nnodes=supervisor_nnodes,
                supervisor_node_rank=supervisor_node_rank,
                rdzv_endpoint=rdzv_endpoint,
                probe_log_dir=probe_log_dir,
                probe_timeout_seconds=probe_timeout_seconds,
                probe_memory_reclaim_timeout_seconds=probe_memory_reclaim_timeout_seconds,
                probe_memory_tolerance_mb=probe_memory_tolerance_mb,
            )
            return
    logging.setLevel(logging.CRITICAL)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = getattr(torch, dtype)
    # Distributed profiling stops on allocated memory. Leave extra reservation headroom for FSDP all-gathers and
    # allocator cache so the artificial cap does not reject candidates before the target is reached.
    memory_cap = memory_fraction if not distributed else min(0.99, memory_fraction + 0.10)
    torch.cuda.set_per_process_memory_fraction(memory_cap, device)

    if distributed:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=distributed_timeout_seconds))
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True

    assert config_path is not None, "--module-name requires --config-path to be specified as well."
    assert module_name is not None, "--config-path requires --module-name to be specified as well."
    cfg = OmegaConf.load(config_path)
    cfg_sampling_rate = OmegaConf.select(cfg, "data.train_ds.sample_rate", default=16000)
    cfg_token_equivalent_duration = OmegaConf.select(cfg, "data.train_ds.token_equivalent_duration", default=0.08)
    namespace, name = module_name.rsplit('.', maxsplit=1)
    model_cls = getattr(importlib.import_module(namespace), name)
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    if not distributed:
        trainer_cfg = {**trainer_cfg, "devices": 1, "num_nodes": 1}
        trainer_cfg.pop("strategy", None)
    trainer = pl.Trainer(
        **{
            **trainer_cfg,
            "max_steps": 1,
            "max_epochs": 1,
            "limit_val_batches": 0.0,
            "val_check_interval": 0.0,
        }
    )
    with trainer.init_module():
        model = model_cls(OmegaConf.to_container(cfg.model, resolve=True))
    model = model.to(device)

    if isinstance(model, (SALM, SALMWithAsrDecoder)):
        model.prepare_inputs = partial(_override_prepare_inputs, model)

    if not hasattr(model, "oomptimizer_schema"):
        click.secho(
            f"We read model of type {type(model)} which doesn't seem to support OOMptimizer "
            f"(we could not find the property .oomptimizer_schema).",
            fg="red",
        )
        sys.exit(1)

    schema = model.oomptimizer_schema

    is_2d_bucketing = all(
        isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, Number) for v in item)
        for item in buckets
    )
    # Determine modality for input and output.
    modalities = [
        (
            "text"
            if any(
                isinstance(item["type"], NeuralType)
                and isinstance(item["type"].elements_type, LabelsType)
                and item["seq_length"] == direction
                for item in schema["inputs"]
                if item["type"] != "dummy"
            )
            else "audio"
        )
        for direction in ("input", "output")
    ]

    def get_max_seq_lens(buckets):

        def _determine_lens_for_bucket(bin):
            if isinstance(model, (SALM, SALMWithAsrDecoder)):
                return bin, bin  # Note: only 1D bucketing, only counted in tokens
            elif isinstance(model, SALMAutomodel):
                audio_tokens = max(1, int(math.ceil(salm_audio_token_ratio * bin)))
                text_tokens = max(2, int(math.ceil((1.0 - salm_audio_token_ratio) * bin)))
                audio_len = int(math.ceil(audio_tokens * cfg_token_equivalent_duration * cfg_sampling_rate))
                return audio_len, text_tokens
            elif is_2d_bucketing:
                input_len, output_len = bin
            else:
                input_len = bin
                output_len = math.ceil(ratio * input_len)
            sampling_rate = getattr(
                model, "sample_rate", 16000
            )  # TODO: may need to extend schema for broader model coverage
            match modalities:
                case "audio", "audio":
                    return (
                        compute_num_samples(input_len, sampling_rate=sampling_rate),
                        compute_num_samples(output_len, sampling_rate=sampling_rate),
                    )
                case "audio", "text":
                    return (compute_num_samples(input_len, sampling_rate=sampling_rate), output_len)
                case "text", "audio":
                    return (
                        input_len,
                        compute_num_samples(output_len, sampling_rate=sampling_rate),
                    )
                case "text", "text":
                    return input_len, output_len
                case _:
                    raise RuntimeError(f"Unexpected modality combination: {_}")

        return [_determine_lens_for_bucket(bin) for bin in buckets]

    click.echo("Starting profiling.")
    max_seq_lens = get_max_seq_lens(buckets)
    target_memory = memory_fraction * torch.cuda.get_device_properties(device).total_memory
    profile_by_memory = distributed
    gen = ProfilingBatchGenerator(
        schema=schema, start_batch_size=start_batch_size, rel_gap_thresh=threshold, device=device, float_dtype=dtype
    )
    profile = {}

    class _GenDataset(IterableDataset):
        def __iter__(self):
            gen.reset()
            gen._current = 1
            yield gen(*get_max_seq_lens([33])[0])
            gen.reset()

        def __len__(self):
            return 1

    # initialize everything PTL needs
    trainer.fit(model, DataLoader(_GenDataset(), batch_size=None))
    model = model.to(device)
    optimizer = model.configure_optimizers()["optimizer"]
    model.log = lambda *args, **kwargs: None  # no logging
    if probe_batch_sizes is not None:
        if probe_seq_len_in is None or probe_seq_len_out is None or probe_result_path is None:
            raise click.ClickException("--probe-batch-sizes requires probe sequence lengths and result path.")
        _run_probe_batch_sizes(
            gen=gen,
            model=model,
            optimizer=optimizer,
            seq_len_in=probe_seq_len_in,
            seq_len_out=probe_seq_len_out,
            batch_sizes=_parse_int_list(probe_batch_sizes),
            result_path=Path(probe_result_path),
            target_memory=target_memory,
            bucket=probe_bucket,
            distributed=distributed,
            device=device,
        )
        return

    # Iterate buckets from the largest to the smallest sequences. This usually ends up creating
    # a tiny bit smaller batches, likely due to worse memory fragmentation.
    with torch.autocast("cuda", dtype=None, enabled=False):
        for bucket, (seq_len_in, seq_len_out) in reversed(list(zip(buckets, max_seq_lens))):
            click.echo(f"The current sequence lengths are: input={seq_len_in} output={seq_len_out}.")
            gen.reset()
            batch_idx = 0

            def step():
                click.echo(
                    f"\t[BEGIN step] [CUDA RAM CURRENT: {torch.cuda.memory_allocated() / (1024 * 1024):.1f}MB] [CUDA RAM MAX: {torch.cuda.max_memory_allocated() / (1024*1024):.1f}MB]"
                )
                batch = gen(seq_len_in, seq_len_out)

                oom = False
                peak_allocated = 0
                status = "OK"
                try:
                    click.echo(f"\tCurrent gap: {gen.current_rel_gap}... ", nl=False)
                    optimizer.zero_grad()
                    out = model.training_step(batch, batch_idx)
                    out['loss'].sum().backward()
                    optimizer.step()
                    peak_allocated = torch.cuda.max_memory_allocated()
                except torch.cuda.OutOfMemoryError as e:
                    oom = True
                    status = "OOM!"
                except RuntimeError as e:
                    error_msg = str(e)
                    oom_like = (
                        "cuFFT error: CUFFT_INTERNAL_ERROR" in error_msg
                        or "CUDA out of memory" in error_msg
                        or "CUDACachingAllocator" in error_msg
                    )
                    if not oom_like:
                        raise
                    oom = True
                    status = "OOM!"
                else:
                    status = "OK!"
                finally:
                    if distributed:
                        oom_t = torch.tensor([int(oom)], dtype=torch.int32, device=device)
                        try:
                            torch.distributed.all_reduce(oom_t, op=torch.distributed.ReduceOp.MAX)
                            oom = bool(oom_t.item())
                        except RuntimeError:
                            oom = True
                    if not oom and profile_by_memory:
                        peak_t = torch.tensor([peak_allocated], dtype=torch.float64, device=device)
                        torch.distributed.all_reduce(peak_t, op=torch.distributed.ReduceOp.MAX)
                        peak_allocated = int(peak_t.item())
                        if peak_allocated >= target_memory:
                            oom = True
                            status = f"MEMORY TARGET ({peak_allocated / (1024 * 1024):.1f}MB)!"
                    elif oom:
                        status = "OOM!"
                    click.secho(status, fg="yellow" if oom else "green")
                    click.echo(
                        f"\t[END step] [CUDA RAM CURRENT: {torch.cuda.memory_allocated() / (1024 * 1024):.1f}MB] [CUDA RAM MAX: {torch.cuda.max_memory_allocated() / (1024*1024):.1f}MB]"
                    )
                    del batch
                    if oom:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    # Note: We could call empty_cache() to free up some more memory on the GPU,
                    #       but we have found out empirically that this causes a mismatched condition
                    #       between OOMptimizer and the actual training. During training, there is some
                    #       degree of memory fragmentation and it's better to simulate that in OOMptimizer.
                    # torch.cuda.memory.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                return oom

            oom = step()
            while not (finished := gen.advance(oom)):
                click.echo("\t" + "=" * 80)
                oom = step()

            click.secho(
                f"=> Optimal setting for bucket={bucket} (input={seq_len_in} output={seq_len_out}) is max_batch_size={gen.max_batch_size}",
                fg="green",
            )
            profile[(bucket, seq_len_in, seq_len_out)] = gen.max_batch_size
            gen.start_batch_size = gen.max_batch_size * 2

    # Reverse the profile to be ascendingly sorted again.
    profile = dict(reversed(list(profile.items())))

    click.echo("The 1st stage profile is:")
    for (bucket, seq_len_in, seq_len_out), bs in profile.items():
        click.echo(f"Bucket={bucket} (input={seq_len_in} output={seq_len_out}) => max_batch_size={bs}")

    if is_2d_bucketing:
        # 2D bucketing doesn't support bucket merging.
        final_profile = [["[" + ",".join(map(str, b)) + "]", bs] for (b, _, __), bs in profile.items()]
    else:
        click.echo("Bucket merging stage...")
        final_profile = []
        for idx, ((bucket, seq_len_in, seq_len_out), bs) in enumerate(profile.items()):
            if idx == 0:
                final_profile.append([bucket, bs])
                continue
            if bs == final_profile[-1][1]:
                click.echo(f"Merging bucket {idx} with bucket {idx-1} due to identical batch sizes.")
                final_profile[-1][0] = bucket
                continue
            final_profile.append([bucket, bs])

    click.secho(f"The profile was created with the following settings:")
    click.secho(f"* using {memory_fraction:.1%} of available GPU RAM.")
    click.secho(f"* {'' if ddp else 'not '}simulating DDP memory overhead.")
    click.secho(f"* using AMP with dtype={dtype}.")
    click.secho("The final profile is:", bold=True)
    click.secho("\tbucket_duration_bins=[" + ",".join(str(seqlen) for seqlen, bs in final_profile) + "]", bold=True)
    click.secho("\tbucket_batch_size=[" + ",".join(str(bs) for seqlen, bs in final_profile) + "]", bold=True)


def _run_probe_batch_sizes(
    *,
    gen: ProfilingBatchGenerator,
    model: pl.LightningModule,
    optimizer: torch.optim.Optimizer,
    seq_len_in: int,
    seq_len_out: int,
    batch_sizes: list[int],
    result_path: Path,
    target_memory: float,
    bucket: str | None,
    distributed: bool,
    device: torch.device,
) -> None:
    global_rank = int(os.environ.get("RANK", "0"))
    batch_idx = 0

    with torch.autocast("cuda", dtype=None, enabled=False):
        for batch_size in batch_sizes:
            click.echo(
                f"OOMPTIMIZER_PROBE bucket={bucket} batch_size={batch_size} "
                f"input={seq_len_in} output={seq_len_out}"
            )
            gen.reset()
            gen._current = batch_size
            torch.cuda.reset_peak_memory_stats()
            batch = None
            try:
                optimizer.zero_grad()
                batch = gen(seq_len_in, seq_len_out)
                out = model.training_step(batch, batch_idx)
                out['loss'].sum().backward()
                optimizer.step()
                torch.cuda.synchronize(device)
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
            except torch.cuda.OutOfMemoryError as e:
                click.echo(f"OOMPTIMIZER_PROBE_OOM batch_size={batch_size}: {e}")
                if global_rank == 0:
                    _append_probe_record(
                        result_path,
                        {
                            "batch_size": batch_size,
                            "bucket": bucket,
                            "status": "oom",
                            "message": str(e),
                        },
                    )
                os._exit(42)
            except RuntimeError as e:
                if not _is_oom_like(e):
                    raise
                click.echo(f"OOMPTIMIZER_PROBE_OOM_LIKE batch_size={batch_size}: {e}")
                if global_rank == 0:
                    _append_probe_record(
                        result_path,
                        {
                            "batch_size": batch_size,
                            "bucket": bucket,
                            "status": "oom",
                            "message": str(e),
                        },
                    )
                os._exit(43)
            finally:
                if batch is not None:
                    del batch

            if distributed:
                try:
                    peak_t = torch.tensor([peak_allocated, peak_reserved], dtype=torch.float64, device=device)
                    torch.distributed.all_reduce(peak_t, op=torch.distributed.ReduceOp.MAX)
                    peak_allocated = int(peak_t[0].item())
                    peak_reserved = int(peak_t[1].item())
                except RuntimeError as e:
                    click.echo(f"OOMPTIMIZER_PROBE_COLLECTIVE_FAILED batch_size={batch_size}: {e}")
                    os._exit(44)

            status = "memory_target" if peak_allocated >= target_memory else "ok"
            if global_rank == 0:
                _append_probe_record(
                    result_path,
                    {
                        "batch_size": batch_size,
                        "bucket": bucket,
                        "status": status,
                        "peak_allocated": peak_allocated,
                        "peak_reserved": peak_reserved,
                        "target_memory": target_memory,
                    },
                )
            click.echo(
                f"OOMPTIMIZER_PROBE_RESULT batch_size={batch_size} status={status} "
                f"peak_allocated={peak_allocated / (1024 ** 3):.2f}GiB"
            )
            if status == "memory_target":
                break


def _override_prepare_inputs(self, batch: dict) -> dict:
    ratio = 0.8
    input_embs = self.embed_tokens(batch["input_ids"][:, :-1])
    target_ids = batch["input_ids"][:, 1:]
    attention_mask = torch.ones_like(target_ids, dtype=torch.bool)

    B, T = input_embs.shape[:2]
    audio_emb_len = int(input_embs.shape[1] * ratio)
    n_samples = int(audio_emb_len * self.token_equivalent_duration * self.sampling_rate)
    audio = torch.randn(B, n_samples, device=input_embs.device, dtype=torch.float32)
    audio_lens = torch.tensor([n_samples] * B, device=input_embs.device)
    audio_embs, _ = self.perception(input_signal=audio, input_signal_length=audio_lens)
    input_embs[:, : audio_embs.shape[1]] = audio_embs

    return {
        "input_embeds": input_embs,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
    }


if __name__ == "__main__":
    oomptimizer()
