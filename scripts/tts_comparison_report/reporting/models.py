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
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Self

from scripts.tts_comparison_report.reporting.constants import TQDM_NCOLS
from scripts.tts_comparison_report.reporting.storage import BaseStorage
from tqdm import tqdm


_REQUIRED_SAMPLE_ID_KEYS: list[str] = [
    "pred_audio_filepath",
    "gt_text",
    "gt_audio_filepath",
    "context_audio_filepath",
]


@dataclass
class BucketStructure:
    """Paths and naming conventions used to locate artifacts inside an evaluation bucket."""

    eval_output_subdir: str = "results"
    metrics_suffix: str = "_metrics_0.json"
    metrics_filewise_suffix: str = "_filewise_metrics_0.json"
    context_audio_dir: str = "audio/repeat_0"
    context_audio_prefix: str = "context_audio_"
    generated_audio_dir: str = "audio/repeat_0"
    generated_audio_prefix: str = "predicted_audio_"


def _map_generated_to_context_name(
    generated_name: str,
    generated_prefix: str,
    context_prefix: str,
) -> str:
    suffix = generated_name.split(generated_prefix)[-1]
    return f"{context_prefix}{suffix}"


@dataclass(frozen=True)
class BenchmarkSampleMeta:
    """Metadata describing one generated sample within a benchmark."""

    name: str
    gt_text: str
    context_path: Path
    sample_id: str

    @staticmethod
    def _validate(item: dict[str, Any]) -> None:
        for key in _REQUIRED_SAMPLE_ID_KEYS:
            if key not in item:
                raise ValueError(f"Missing required key '{key}' in filewise metrics item.")

    @staticmethod
    def _get_sample_id(item: dict[str, Any]) -> str:
        parts = [item["gt_audio_filepath"], item["context_audio_filepath"]]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        item: dict[str, Any],
        context_audio_paths: dict[str, Path],
        bucket_structure: BucketStructure,
    ) -> Self:
        """Create sample metadata from one filewise metrics item.

        Args:
            item: One entry from the filewise metrics JSON.
            context_audio_paths: Mapping from context audio file name to its path.
            bucket_structure: Bucket naming and path conventions.

        Returns:
            Sample metadata extracted from the given filewise metrics item.

        Raises:
            ValueError: If required keys are missing from the item.
            KeyError: If the corresponding context audio file is not found.
        """
        cls._validate(item)

        name = Path(item["pred_audio_filepath"]).stem

        key = _map_generated_to_context_name(
            generated_name=name,
            generated_prefix=bucket_structure.generated_audio_prefix,
            context_prefix=bucket_structure.context_audio_prefix,
        )
        obj = cls(
            name=name,
            gt_text=item["gt_text"],
            context_path=context_audio_paths[key],
            sample_id=cls._get_sample_id(item),
        )
        return obj


def _collect_audio_paths(
    root: Path,
    prefix: str,
    audio_paths: dict[str, Path],
    storage: BaseStorage,
) -> None:
    if not storage.exists(root):
        raise FileNotFoundError(f"Missing audio directory: '{root}'.")

    for p in storage.iter_dir(root):
        if not p.stem.startswith(prefix) or p.suffix != ".wav":
            continue
        audio_paths[p.stem] = p


def _validate_audio_pairs(
    context_audio_paths: dict[str, Path],
    generated_audio_paths: dict[str, Path],
    bucket_structure: BucketStructure,
) -> None:
    for name in generated_audio_paths:
        key = _map_generated_to_context_name(
            generated_name=name,
            generated_prefix=bucket_structure.generated_audio_prefix,
            context_prefix=bucket_structure.context_audio_prefix,
        )
        if key not in context_audio_paths:
            raise ValueError(f"Missing context audio: '{key}'.")


@dataclass
class BenchmarkData:
    """Artifacts and loaded data associated with one evaluation benchmark."""

    name: str
    metrics_path: Optional[Path] = None
    filewise_metrics_path: Optional[Path] = None
    generated_audio_paths: dict[str, Path] = field(default_factory=dict)
    context_audio_paths: dict[str, Path] = field(default_factory=dict)

    metrics: Optional[dict[str, float]] = None
    filewise_metrics: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_storage(
        cls,
        benchmark_name: str,
        benchmark_path: Path,
        bucket_structure: BucketStructure,
        check_audio: bool,
        storage: BaseStorage,
    ) -> Self:
        """Create benchmark data by discovering benchmark artifacts in storage.

        Args:
            benchmark_name: Name of the benchmark.
            benchmark_path: Path to the benchmark directory inside the evaluation bucket.
            bucket_structure: Bucket naming and path conventions.
            check_audio: Whether generated audio files should also be discovered.
            storage: Storage backend used to access local or remote files.

        Returns:
            Benchmark data initialized with discovered artifact paths.

        Raises:
            FileNotFoundError: If required metrics files are missing, audio directories
                are missing, or expected audio files cannot be found.
            ValueError: If generated audio files do not have matching context audio files.
        """
        obj = cls(name=benchmark_name)

        path = benchmark_path / f"{benchmark_name}{bucket_structure.metrics_suffix}"
        if not storage.exists(path):
            raise FileNotFoundError(f"Missing metrics file: '{path}'.")
        obj.metrics_path = path

        path = benchmark_path / f"{benchmark_name}{bucket_structure.metrics_filewise_suffix}"
        if not storage.exists(path):
            raise FileNotFoundError(f"Missing filewise metrics file: '{path}'.")
        obj.filewise_metrics_path = path

        if check_audio:
            _collect_audio_paths(
                root=benchmark_path / bucket_structure.context_audio_dir,
                prefix=bucket_structure.context_audio_prefix,
                audio_paths=obj.context_audio_paths,
                storage=storage,
            )
            if not obj.context_audio_paths:
                raise FileNotFoundError(
                    f"No context audio files were found in '{benchmark_path / bucket_structure.context_audio_dir}'. "
                    "The bucket structure likely differs from the one specified in 'BucketStructure'."
                )
            _collect_audio_paths(
                root=benchmark_path / bucket_structure.generated_audio_dir,
                prefix=bucket_structure.generated_audio_prefix,
                audio_paths=obj.generated_audio_paths,
                storage=storage,
            )
            if not obj.generated_audio_paths:
                raise FileNotFoundError(
                    f"No generated audio files were found in '{benchmark_path / bucket_structure.generated_audio_dir}'. "
                    "The bucket structure likely differs from the one specified in 'BucketStructure'."
                )
            _validate_audio_pairs(
                context_audio_paths=obj.context_audio_paths,
                generated_audio_paths=obj.generated_audio_paths,
                bucket_structure=bucket_structure,
            )
        return obj

    def load_metrics(self, storage: BaseStorage) -> None:
        """Load aggregated benchmark metrics from storage.

        Args:
            storage: Storage instance used to read the metrics file.

        Raises:
            TypeError: If the metrics file does not contain a JSON object.
        """
        if self.metrics_path is None:
            return

        data = storage.read_json(self.metrics_path)

        if not isinstance(data, dict):
            raise TypeError(f"Metrics file must contain a JSON object: '{self.metrics_path}'.")

        self.metrics = data

    def load_filewise_metrics(self, storage: BaseStorage) -> None:
        """Load filewise benchmark metrics from storage.

        Args:
            storage: Storage instance used to read the filewise metrics file.

        Raises:
            TypeError: If the filewise metrics file does not contain a JSON array.
        """
        if self.filewise_metrics_path is None:
            return

        data = storage.read_json(self.filewise_metrics_path)

        if not isinstance(data, list):
            raise TypeError(f"Filewise metrics file must contain a JSON array: '{self.filewise_metrics_path}'.")

        self.filewise_metrics = data


def _validate_numeric_metric_value(
    value: Any,
    metric_name: str,
    context: str,
) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Metric '{metric_name}' in {context} must be numeric, "
            f"but got value {value!r} of type {type(value).__name__}."
        )
    return float(value)


@dataclass
class BucketData:
    """Evaluation bucket metadata and loaded metric data."""

    name: str
    path: Path
    configuration_str: Optional[str] = None
    benchmarks: dict[str, BenchmarkData] = field(default_factory=dict)

    @classmethod
    def from_storage(
        cls,
        bucket_name: str,
        bucket_path: Path,
        bucket_structure: BucketStructure,
        benchmark_names: tuple[str, ...],
        check_audio: bool,
        storage: BaseStorage,
    ) -> Self:
        """Create bucket data by discovering benchmark artifacts in storage.

        Args:
            bucket_name: Display name of the bucket, typically the name of the model it belongs to.
            bucket_path: Path to the bucket root directory.
            bucket_structure: Bucket naming and path conventions.
            benchmark_names: Benchmark names expected in the bucket.
            check_audio: Whether generated audio files should also be discovered.
            storage: Storage instance used to access local or remote files.

        Returns:
            Bucket data initialized with discovered benchmark artifacts.

        Raises:
            FileNotFoundError: If the expected results directory is missing.
        """
        obj = cls(name=bucket_name, path=bucket_path)
        results_path = bucket_path / bucket_structure.eval_output_subdir

        if not storage.exists(results_path):
            raise FileNotFoundError(f"Missing results directory: '{results_path}'.")

        for benchmark_path in storage.iter_dir(results_path, only_dirs=True):
            if len(obj.benchmarks) == len(benchmark_names):
                break

            dir_name = benchmark_path.name
            name = next((n for n in benchmark_names if dir_name == n or dir_name.endswith(f"_{n}")), None)

            if name is None:
                continue

            obj.benchmarks[name] = BenchmarkData.from_storage(
                benchmark_name=name,
                benchmark_path=benchmark_path,
                bucket_structure=bucket_structure,
                check_audio=check_audio,
                storage=storage,
            )
            if obj.configuration_str is None:
                suffix = f"_{name}"
                obj.configuration_str = dir_name[: -len(suffix)]

        return obj

    def load_metrics(
        self,
        storage: BaseStorage,
        show_pbar: bool = False,
    ) -> None:
        """Load aggregated and filewise metrics for all discovered benchmarks.

        Args:
            storage: Storage instance used to read metrics files.
            show_pbar: Whether to display a progress bar while loading metrics.
        """
        pbar = tqdm(total=len(self.benchmarks), ncols=TQDM_NCOLS) if show_pbar else None

        for benchmark_data in self.benchmarks.values():
            benchmark_data.load_metrics(storage)
            benchmark_data.load_filewise_metrics(storage)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    def get_metric_avg_value(
        self,
        metric_name: str,
        benchmark_name: str,
    ) -> Optional[float]:
        """Return the aggregated value of a metric for one benchmark.

        Args:
            metric_name: Name of the metric to retrieve.
            benchmark_name: Name of the benchmark.

        Returns:
            Aggregated metric value, or `None` if the metric is not present.

        Raises:
            ValueError: If the benchmark is unknown or metrics are not loaded.
            TypeError: If the metric value is not numeric.
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: '{benchmark_name}'.")

        metrics = self.benchmarks[benchmark_name].metrics

        if metrics is None:
            raise ValueError(f"Metrics not loaded for benchmark: '{benchmark_name}'.")

        if metric_name not in metrics:
            return None

        value = _validate_numeric_metric_value(
            value=metrics[metric_name],
            metric_name=metric_name,
            context=f"averaged metrics for benchmark '{benchmark_name}'",
        )
        return value

    def _get_metric_stats(
        self,
        metric_name: str,
        benchmark_name: str,
    ) -> list[float]:
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: '{benchmark_name}'.")

        items = self.benchmarks[benchmark_name].filewise_metrics

        if items is None or not items:
            raise ValueError(f"Filewise metrics not loaded for benchmark: '{benchmark_name}'.")

        output = []
        validation_context = f"filewise metrics for benchmark '{benchmark_name}'"

        for item in items:
            if metric_name not in item:
                continue

            value = _validate_numeric_metric_value(
                value=item[metric_name],
                metric_name=metric_name,
                context=validation_context,
            )
            output.append(value)

        if not output:
            raise ValueError(f"Unknown or empty metric '{metric_name}' for benchmark '{benchmark_name}'.")

        return output

    def _aggregate_metric_stats(self, metric_name: str) -> list[float]:
        output = []

        for benchmark_name in self.benchmarks:
            output.extend(self._get_metric_stats(metric_name, benchmark_name))

        if not output:
            raise ValueError(f"Unknown or empty aggregated metric '{metric_name}'.")

        return output

    def get_metric_samples(
        self,
        metric_name: str,
        benchmark_name: Optional[str] = None,
    ) -> list[float]:
        """Return filewise samples for a metric from one or all benchmarks.

        Args:
            metric_name: Name of the metric to retrieve.
            benchmark_name: Benchmark name. If omitted, samples are aggregated
                across all benchmarks.

        Returns:
            List of numeric metric samples.

        Raises:
            ValueError: If the benchmark is unknown, filewise metrics are not loaded,
                or the metric is missing.
            TypeError: If any metric value is not numeric.
        """
        if benchmark_name is None:
            return self._aggregate_metric_stats(metric_name)
        return self._get_metric_stats(metric_name, benchmark_name)

    def get_benchmark_audio_paths(self, benchmark_name: str) -> dict[str, Path]:
        """Return generated audio file paths for a benchmark.

        Args:
            benchmark_name: Name of the benchmark.

        Returns:
            Mapping from sample name to generated audio path.

        Raises:
            ValueError: If the benchmark is unknown or audio paths are not loaded.
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: '{benchmark_name}'.")

        paths = self.benchmarks[benchmark_name].generated_audio_paths

        if not paths:
            raise ValueError(f"Generated audio paths not loaded for benchmark: '{benchmark_name}'.")

        return paths

    def get_benchmark_sample_meta(
        self,
        benchmark_name: str,
        bucket_structure: BucketStructure,
    ) -> dict[str, BenchmarkSampleMeta]:
        """Return sample metadata for a benchmark derived from filewise metrics.

        Args:
            benchmark_name: Name of the benchmark.
            bucket_structure: Bucket naming and path conventions used to resolve
                matching context audio files.

        Returns:
            Mapping from sample name to benchmark sample metadata.

        Raises:
            ValueError: If the benchmark is unknown, filewise metrics are not loaded,
                or context audio paths are not loaded.
            KeyError: If a matching context audio file cannot be found for a sample.
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: '{benchmark_name}'.")

        items = self.benchmarks[benchmark_name].filewise_metrics

        if not items:
            raise ValueError(f"Filewise metrics not loaded for benchmark: '{benchmark_name}'.")

        paths = self.benchmarks[benchmark_name].context_audio_paths

        if not paths:
            raise ValueError(f"Context audio paths not loaded for benchmark: '{benchmark_name}'.")

        output = {}

        for item in items:
            meta = BenchmarkSampleMeta.create(
                item=item,
                context_audio_paths=paths,
                bucket_structure=bucket_structure,
            )
            output[meta.name] = meta

        return output


@dataclass(frozen=True)
class TaskInfo:
    """Task identifiers and derived Jira link information used in reports."""

    task_id: str
    jira_id: str
    jira_url: str


@dataclass(frozen=True)
class ExpirationInfo:
    """Formatted expiration metadata used in reports and S3 artifact paths."""

    timestamp: int
    path_str: str
    user_str: str


class Winner(str, Enum):
    """Possible outcomes of a statistical comparison between baseline and candidate."""

    baseline = "baseline"
    candidate = "candidate"
    tie = "tie"


@dataclass(frozen=True)
class StatTestResult:
    """Result of a statistical comparison for a single metric."""

    metric_name: str
    winner: Winner
    alternative: str
    p_value: float


@dataclass(frozen=True)
class StatTestAnalysisInfo:
    """Summary information used to describe statistical test outcomes in reports."""

    winner: Optional[str]
    advantages: Optional[str]


@dataclass(frozen=True)
class EvalResult:
    """Evaluation results for one report section, including tables, analysis, and plot."""

    metrics_table_row: list[str | float]
    stat_test_table_row: list[str | float]
    stat_tests_analysis_info: StatTestAnalysisInfo
    box_plots: BytesIO


@dataclass(frozen=True)
class ModelConfiguration:
    """Configuration strings associated with the baseline and candidate models."""

    baseline: str
    candidate: str


@dataclass(frozen=True)
class EvalArtifacts:
    """Prepared evaluation results, configuration, and comparison metadata used to render reports."""

    configuration: ModelConfiguration
    summary: EvalResult
    benchmarks: dict[str, EvalResult]
    is_self_comparison: bool


@dataclass(frozen=True)
class UploadedBoxPlotsInfo:
    """S3 URLs of uploaded summary and benchmark-level box plot images."""

    summary_url: str
    benchmark_urls: dict[str, str]


@dataclass(frozen=True)
class AudioPair:
    """Matched context, baseline, and candidate audio files for one sample."""

    context_path: Path
    baseline_path: Path
    candidate_path: Path
    text: str


@dataclass(frozen=True)
class UploadedAudioPairInfo:
    """Uploaded context, baseline, and candidate audio URLs for one sample."""

    context_url: str
    baseline_url: str
    candidate_url: str
    text: str
