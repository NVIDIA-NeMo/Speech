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
from io import BytesIO
from logging import Logger
from pathlib import Path
from typing import Optional

from scripts.tts_comparison_report.reporting.components import (
    BoxPlotsConfig,
    prepare_audio_pairs,
    prepare_eval_artifacts,
)
from scripts.tts_comparison_report.reporting.constants import (
    S3_AUDIO_DIR,
    S3_IMAGES_DIR,
    S3_LINK_EXPIRES_IN,
    TQDM_NCOLS,
)
from scripts.tts_comparison_report.reporting.helpers import generate_s3_prefix, make_expiration_info, make_task_info
from scripts.tts_comparison_report.reporting.models import (
    AudioPair,
    BucketData,
    BucketStructure,
    EvalArtifacts,
    ExpirationInfo,
    TaskInfo,
    UploadedAudioPairInfo,
    UploadedBoxPlotsInfo,
)
from scripts.tts_comparison_report.reporting.renderer import Renderer, TemplateName
from scripts.tts_comparison_report.reporting.s3_client import S3Client
from scripts.tts_comparison_report.reporting.storage import BaseStorage
from tqdm import tqdm


class Orchestrator:
    """Coordinate loading, processing, rendering, and uploading of comparison reports."""

    def __init__(
        self,
        bucket_structure: BucketStructure,
        storage: BaseStorage,
        s3_client: S3Client,
        renderer: Renderer,
        logger: Optional[Logger] = None,
    ) -> None:
        self.bucket_structure = bucket_structure
        self.storage = storage
        self.s3_client = s3_client
        self.renderer = renderer
        self.logger = logger

        self.show_pbar = True if logger is not None else False

    def _log_info(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    def _load_buckets(
        self,
        baseline_name: str,
        candidate_name: str,
        baseline_path: Path,
        candidate_path: Path,
        benchmark_names: tuple[str, ...],
        check_audio: bool,
    ) -> tuple[BucketData, BucketData]:
        self._log_info(f"\nLoading metadata for {baseline_name}...")
        bucket_baseline = BucketData.from_storage(
            bucket_name=baseline_name,
            bucket_path=baseline_path,
            bucket_structure=self.bucket_structure,
            benchmark_names=benchmark_names,
            check_audio=check_audio,
            storage=self.storage,
        )
        self._log_info(f"Loading metadata for {candidate_name}...")
        bucket_candidate = BucketData.from_storage(
            bucket_name=candidate_name,
            bucket_path=candidate_path,
            bucket_structure=self.bucket_structure,
            benchmark_names=benchmark_names,
            check_audio=check_audio,
            storage=self.storage,
        )

        baseline_set = set(bucket_baseline.benchmarks.keys())
        candidate_set = set(bucket_candidate.benchmarks.keys())

        if baseline_set != candidate_set:
            raise ValueError(f"Benchmark sets differ: '{baseline_set}' vs '{candidate_set}'.")

        self._log_info(f"\nLoading metric data for {baseline_name}:")
        bucket_baseline.load_metrics(storage=self.storage, show_pbar=self.show_pbar)

        self._log_info(f"\nLoading metric data for {candidate_name}:")
        bucket_candidate.load_metrics(storage=self.storage, show_pbar=self.show_pbar)

        return bucket_baseline, bucket_candidate

    def _upload_audio_file(
        self,
        path: Path,
        key: str,
    ) -> str:
        with self.storage.open_file(path) as f:
            url = self.s3_client.upload_fileobj(
                fileobj=f,
                key=key,
                expires_in=S3_LINK_EXPIRES_IN,
                content_type="audio/wav",
            )
        return url

    def _upload_png_image(
        self,
        image: BytesIO,
        key: str,
    ) -> str:
        url = self.s3_client.upload_bytes(
            data=image.getvalue(),
            key=key,
            expires_in=S3_LINK_EXPIRES_IN,
            content_type="image/png",
        )
        return url

    def _upload_audio(
        self,
        used_benchmarks: list[str],
        audio_pairs: dict[str, list[AudioPair]],
        s3_prefix: str,
    ) -> dict[str, list[UploadedAudioPairInfo]]:
        total = sum(len(v) for v in audio_pairs.values())
        pbar = tqdm(total=total, ncols=TQDM_NCOLS) if self.show_pbar else None
        uploaded_info = {}

        for benchmark_name in used_benchmarks:
            benchmark_info = []

            for i, pair in enumerate(audio_pairs[benchmark_name]):
                context_url = self._upload_audio_file(
                    path=pair.context_path,
                    key=f"{s3_prefix}/{S3_AUDIO_DIR}/context_{benchmark_name}_{i}.wav",
                )
                baseline_url = self._upload_audio_file(
                    path=pair.baseline_path,
                    key=f"{s3_prefix}/{S3_AUDIO_DIR}/baseline_{benchmark_name}_{i}.wav",
                )
                candidate_url = self._upload_audio_file(
                    path=pair.candidate_path,
                    key=f"{s3_prefix}/{S3_AUDIO_DIR}/candidate_{benchmark_name}_{i}.wav",
                )
                pair_info = UploadedAudioPairInfo(
                    context_url=context_url,
                    baseline_url=baseline_url,
                    candidate_url=candidate_url,
                    text=pair.text,
                )
                benchmark_info.append(pair_info)

                if pbar:
                    pbar.update(1)

            uploaded_info[benchmark_name] = benchmark_info

        if pbar:
            pbar.close()

        return uploaded_info

    def _upload_boxplots(
        self,
        eval_artifacts: EvalArtifacts,
        s3_prefix: str,
    ) -> UploadedBoxPlotsInfo:
        name_prefix = "box_plot"
        pbar = tqdm(total=len(eval_artifacts.benchmarks) + 1, ncols=TQDM_NCOLS) if self.show_pbar else None

        summary_url = self._upload_png_image(
            image=eval_artifacts.summary.box_plots,
            key=f"{s3_prefix}/{S3_IMAGES_DIR}/{name_prefix}_summary.png",
        )
        if pbar:
            pbar.update(1)

        benchmark_urls = {}

        for benchmark_name, benchmark_result in eval_artifacts.benchmarks.items():
            benchmark_urls[benchmark_name] = self._upload_png_image(
                image=benchmark_result.box_plots,
                key=f"{s3_prefix}/{S3_IMAGES_DIR}/{name_prefix}_{benchmark_name}.png",
            )
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return UploadedBoxPlotsInfo(
            summary_url=summary_url,
            benchmark_urls=benchmark_urls,
        )

    def _upload_report(
        self,
        report: str,
        s3_prefix: str,
        report_name: str,
    ) -> str:
        report_url = self.s3_client.upload_bytes(
            data=report.encode("utf-8"),
            key=f"{s3_prefix}/{report_name}.html",
            expires_in=S3_LINK_EXPIRES_IN,
            content_type="text/html; charset=utf-8",
        )
        return report_url

    def _render_audio_report(
        self,
        baseline_name: str,
        candidate_name: str,
        used_benchmarks: list[str],
        uploaded_audio_info: dict[str, list[UploadedAudioPairInfo]],
        task_info: TaskInfo,
        expiration_info: ExpirationInfo,
    ) -> str:
        expiration_comment = f"This report will expire at {expiration_info.user_str}"

        header_block = self.renderer.render(
            name=TemplateName.audio_report_header,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            expiration_comment=expiration_comment,
        )
        benchmark_blocks, benchmark_section_info = [], []

        for benchmark_name in used_benchmarks:
            pair_blocks = []

            for pair in uploaded_audio_info[benchmark_name]:
                block = self.renderer.render(
                    name=TemplateName.audio_report_pair,
                    context_url=pair.context_url,
                    baseline_url=pair.baseline_url,
                    candidate_url=pair.candidate_url,
                    text=pair.text,
                )
                pair_blocks.append(block)

            block = self.renderer.render(
                name=TemplateName.audio_report_block,
                title=benchmark_name,
                section_id=benchmark_name,
                baseline_name=baseline_name,
                candidate_name=candidate_name,
                pair_blocks=pair_blocks,
            )
            benchmark_blocks.append(block)
            benchmark_section_info.append((benchmark_name, benchmark_name))

        report = self.renderer.render(
            name=TemplateName.audio_report,
            jira_id=task_info.jira_id,
            jira_url=task_info.jira_url,
            header_block=header_block,
            benchmark_blocks=benchmark_blocks,
            benchmark_section_info=benchmark_section_info,
        )

        return report

    def _render_eval_report(
        self,
        baseline_name: str,
        candidate_name: str,
        eval_artifacts: EvalArtifacts,
        uploaded_box_plots_info: UploadedBoxPlotsInfo,
        task_info: TaskInfo,
        expiration_info: ExpirationInfo,
        audio_report_url: Optional[str],
    ) -> str:
        expiration_comment = f"This report will expire at {expiration_info.user_str}"

        configuration_block = self.renderer.render(
            name=TemplateName.eval_report_configuration,
            baseline_name=baseline_name,
            baseline_configuration=eval_artifacts.configuration.baseline,
            candidate_name=candidate_name,
            candidate_configuration=eval_artifacts.configuration.candidate,
        )
        header_block = self.renderer.render(
            name=TemplateName.eval_report_header,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            expiration_comment=expiration_comment,
        )
        metrics_table = self.renderer.render(
            name=TemplateName.eval_report_table,
            title="Metrics (macro-average across benchmarks)",
            headers=["Metric", baseline_name, candidate_name],
            rows=eval_artifacts.summary.metrics_table_row,
        )
        stat_tests_table = self.renderer.render(
            name=TemplateName.eval_report_table,
            title="Statistical Tests (pooled filewise across benchmarks)",
            headers=["Metric", "Winner", "Alternative", "p-value"],
            rows=eval_artifacts.summary.stat_test_table_row,
        )
        stat_tests_analysis = self.renderer.render(
            name=TemplateName.eval_report_stat_analysis,
            winner=eval_artifacts.summary.stat_tests_analysis_info.winner,
            advantages=eval_artifacts.summary.stat_tests_analysis_info.advantages,
        )
        image_block = self.renderer.render(
            name=TemplateName.eval_report_image,
            image_url=uploaded_box_plots_info.summary_url,
        )
        summary_block = self.renderer.render(
            name=TemplateName.eval_report_block,
            is_summary=True,
            metrics_table=metrics_table,
            stat_tests_table=stat_tests_table,
            stat_tests_analysis=stat_tests_analysis,
            image_block=image_block,
        )
        benchmark_blocks, benchmark_section_info = [], []

        for benchmark_name in sorted(eval_artifacts.benchmarks.keys()):
            metrics_table = self.renderer.render(
                name=TemplateName.eval_report_table,
                title="Metrics",
                headers=["Metric", baseline_name, candidate_name],
                rows=eval_artifacts.benchmarks[benchmark_name].metrics_table_row,
            )
            stat_tests_table = self.renderer.render(
                name=TemplateName.eval_report_table,
                title="Statistical Tests",
                headers=["Metric", "Winner", "Alternative", "p-value"],
                rows=eval_artifacts.benchmarks[benchmark_name].stat_test_table_row,
            )
            stat_tests_analysis = self.renderer.render(
                name=TemplateName.eval_report_stat_analysis,
                winner=eval_artifacts.benchmarks[benchmark_name].stat_tests_analysis_info.winner,
                advantages=eval_artifacts.benchmarks[benchmark_name].stat_tests_analysis_info.advantages,
            )
            image_block = self.renderer.render(
                name=TemplateName.eval_report_image,
                image_url=uploaded_box_plots_info.benchmark_urls[benchmark_name],
            )
            block = self.renderer.render(
                name=TemplateName.eval_report_block,
                is_summary=False,
                title=benchmark_name,
                section_id=benchmark_name,
                metrics_table=metrics_table,
                stat_tests_table=stat_tests_table,
                stat_tests_analysis=stat_tests_analysis,
                image_block=image_block,
            )
            benchmark_blocks.append(block)
            benchmark_section_info.append((benchmark_name, benchmark_name))

        report = self.renderer.render(
            name=TemplateName.eval_report,
            is_self_comparison=eval_artifacts.is_self_comparison,
            jira_id=task_info.jira_id,
            jira_url=task_info.jira_url,
            audio_report_url=audio_report_url,
            configuration_block=configuration_block,
            header_block=header_block,
            summary_block=summary_block,
            benchmark_blocks=benchmark_blocks,
            benchmark_section_info=benchmark_section_info,
        )
        return report

    def run(
        self,
        baseline_name: str,
        candidate_name: str,
        baseline_path: Path,
        candidate_path: Path,
        benchmarks: list[str],
        generate_audio_report: bool,
        audio_report_benchmarks: Optional[list[str]],
        samples_per_benchmark: int,
        task_id: str,
    ) -> tuple[str, Optional[str]]:
        """Generate evaluation reports, upload report artifacts to S3, and return report URLs.

        This method performs the full end-to-end comparison workflow:
        it loads evaluation buckets, prepares summary and benchmark-level artifacts,
        uploads plots and optional audio samples to S3, renders the final HTML
        reports, uploads them, and returns their presigned URLs.

        Args:
            baseline_name: Name of the baseline model used in the reports.
            candidate_name: Name of the candidate model used in the reports.
            baseline_path: Path to the baseline evaluation bucket root.
            candidate_path: Path to the candidate evaluation bucket root.
            benchmarks: Benchmark names to include in the evaluation report.
            generate_audio_report: Whether to generate the audio comparison report.
            audio_report_benchmarks: Benchmark names to include in the audio report.
            samples_per_benchmark: Number of audio pairs to sample per benchmark.
            task_id: Task identifier used for report metadata and Jira linking.

        Returns:
            Tuple containing the evaluation report URL and the optional audio report URL.

        Raises:
            ValueError: If input configuration is inconsistent, required benchmarks
                are missing, or report generation inputs are invalid.
            FileNotFoundError: If required bucket artifacts are missing from storage.
            TypeError: If loaded metric files have unexpected types.
        """
        benchmark_names = tuple(sorted(benchmarks, key=len, reverse=True))

        audio_report: Optional[str] = None
        audio_report_url: Optional[str] = None

        bucket_baseline, bucket_candidate = self._load_buckets(
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            benchmark_names=benchmark_names,
            check_audio=generate_audio_report,
        )

        task_info = make_task_info(task_id)
        expiration_info = make_expiration_info(S3_LINK_EXPIRES_IN)
        s3_prefix = generate_s3_prefix(baseline_path, candidate_path, task_info, expiration_info)
        box_plots_cfg = BoxPlotsConfig()

        self._log_info("\nPreparing evaluation artifacts...")
        eval_artifacts = prepare_eval_artifacts(
            bucket_baseline=bucket_baseline,
            bucket_candidate=bucket_candidate,
            box_plots_cfg=box_plots_cfg,
        )
        self._log_info("\nUploading images to S3:")
        uploaded_box_plots_info = self._upload_boxplots(
            eval_artifacts=eval_artifacts,
            s3_prefix=s3_prefix,
        )

        if generate_audio_report:
            if audio_report_benchmarks is None:
                raise ValueError("Audio report benchmarks must be provided when audio report is enabled.")

            audio_pairs = prepare_audio_pairs(
                bucket_baseline=bucket_baseline,
                bucket_candidate=bucket_candidate,
                bucket_structure=self.bucket_structure,
                used_benchmarks=audio_report_benchmarks,
                samples_per_benchmark=samples_per_benchmark,
            )
            self._log_info("\nUploading audio files to S3:")
            uploaded_audio_info = self._upload_audio(
                used_benchmarks=audio_report_benchmarks,
                audio_pairs=audio_pairs,
                s3_prefix=s3_prefix,
            )
            self._log_info("\nPreparing audio report...")
            audio_report = self._render_audio_report(
                baseline_name=baseline_name,
                candidate_name=candidate_name,
                used_benchmarks=audio_report_benchmarks,
                uploaded_audio_info=uploaded_audio_info,
                task_info=task_info,
                expiration_info=expiration_info,
            )
            audio_report_url = self._upload_report(
                report=audio_report,
                s3_prefix=s3_prefix,
                report_name="audio_report",
            )

        self._log_info("\nPreparing evaluation report...")
        eval_report = self._render_eval_report(
            baseline_name=bucket_baseline.name,
            candidate_name=bucket_candidate.name,
            eval_artifacts=eval_artifacts,
            uploaded_box_plots_info=uploaded_box_plots_info,
            task_info=task_info,
            expiration_info=expiration_info,
            audio_report_url=audio_report_url,
        )
        eval_report_url = self._upload_report(
            report=eval_report,
            s3_prefix=s3_prefix,
            report_name="eval_report",
        )
        self._log_info(f"\nUploaded artifacts to bucket '{self.s3_client.cfg.bucket}' with prefix '{s3_prefix}'.")

        return eval_report_url, audio_report_url
