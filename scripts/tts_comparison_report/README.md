# TTS Comparison Report

This tool generates HTML comparison reports for TTS evaluation buckets and uploads them to S3.

The `generate_report` script compares two evaluation buckets produced by `magpietts_inference` and generates:
1. an HTML evaluation report with aggregated and per-benchmark metrics;
2. an optional HTML audio comparison report that shows, for each sampled utterance, the context audio
prompt (audio-context benchmarks), the target recording, and the baseline and candidate samples side by side.

Both reports are uploaded to S3-compatible object storage and returned as presigned URLs, 
which can be opened directly in a browser. If the audio report is enabled, its link is 
also embedded into the evaluation report.

The generated reports are designed to make model comparison faster, easier to share, and easier to review.

## Supported workflows

The script supports:

- local evaluation buckets;
- remote evaluation buckets accessed over SSH/SFTP;
- upload of generated reports and audio assets to S3-compatible object storage.

## Terminology

A **bucket** in this tool means the root directory of one evaluation run.

Evaluation artifacts are expected to be located either:
- directly inside the experiment root; or
- inside the subdirectory given by `--results_subdir` (default: `results`).

Typical layouts:

**Local generation**
```text
experiment_root/
├── benchmark_1
├── benchmark_2
└── benchmark_3
```

**Cluster / Slurm generation**
```text
experiment_root/
├── logs_dir
└── results_subdir
    ├── benchmark_1
    ├── benchmark_2
    └── benchmark_3
```

In both cases, `--baseline_path` and `--candidate_path` should point to the experiment root. 
If evaluation artifacts are stored directly inside the experiment root, set `--results_subdir` 
to an empty string. If evaluation artifacts are stored under a dedicated subdirectory, 
set `--results_subdir` to that subdirectory name.

## Installation

It is recommended to run the script inside the NeMo Docker container, starting from version `25.11`, 
since it already includes all required dependencies.

From the repository root, update the local NeMo package with:
```bash
pip install -e ./ --no-deps
```
The `--no-deps` flag is used because the required dependencies are already available 
in the recommended NeMo Docker environment.

If you use a different environment, install the required dependencies from `requirements.txt`:
```bash
pip install -r scripts/tts_comparison_report/requirements.txt
```

## Environment variables

Before running the script, make sure that the required environment variables are set.

The following variables are required for uploading reports and related assets 
to S3-compatible object storage:
- `S3_ACCESS_KEY_ID` - S3 access key,
- `S3_SECRET_ACCESS_KEY` - S3 secret key.

Example:
```bash
export S3_ACCESS_KEY_ID='your_s3_key_id' S3_SECRET_ACCESS_KEY='your_s3_secret'
```

If the evaluation buckets are stored on a remote machine, also set:
- `REMOTE_PASSWORD` - password used for SSH authentication.

Example:
```bash
export REMOTE_PASSWORD='your_ssh_password'
```

## Usage examples

To generate and upload only the evaluation report from local buckets, run:

```bash
python scripts/tts_comparison_report/generate_report.py \
  --baseline_name "Model A" \
  --baseline_path /workspace/NeMo/exp/buckets/baseline \
  --candidate_name "Model B" \
  --candidate_path /workspace/NeMo/exp/buckets/candidate \
  --s3_endpoint https://your-s3-endpoint \
  --s3_bucket your_bucket_name \
  --s3_region us-west-2 \
  --task_id NEMOTTS-2007
```

If the evaluation artifacts are stored in a non-default results subdirectory, 
use `--results_subdir`.

To generate both the evaluation report and the audio comparison report, use `--audio_report`.
You can also use `--audio_report_benchmarks` and `--samples_per_benchmark` 
to control which benchmarks are included in the audio report and how many 
samples are selected for each benchmark.

```bash
python scripts/tts_comparison_report/generate_report.py \
  --baseline_name "Model A" \
  --baseline_path /workspace/NeMo/exp/buckets/baseline \
  --candidate_name "Model B" \
  --candidate_path /workspace/NeMo/exp/buckets/candidate \
  --s3_endpoint https://your-s3-endpoint \
  --s3_bucket your_bucket_name \
  --s3_region us-west-2 \
  --task_id NEMOTTS-2007 \
  --audio_report \
  --audio_report_benchmarks libritts,riva_en \
  --samples_per_benchmark 20
```

If the buckets are located on a remote machine, specify `--remote_hostname`
and `--remote_username`:

```bash
python scripts/tts_comparison_report/generate_report.py \
  --baseline_name "Model A" \
  --baseline_path /mnt/exps/baseline \
  --candidate_name "Model B" \
  --candidate_path /mnt/exps/candidate \
  --s3_endpoint https://your-s3-endpoint \
  --s3_bucket your_bucket_name \
  --s3_region us-west-2 \
  --task_id NEMOTTS-2007 \
  --audio_report \
  --audio_report_benchmarks libritts,riva_en \
  --samples_per_benchmark 20 \
  --remote_hostname your_remote_host \
  --remote_username your_user
```

You can also restrict the evaluation report to a selected set of benchmarks 
by using `--benchmarks`:

```bash
python scripts/tts_comparison_report/generate_report.py \
  --baseline_name "Model A" \
  --baseline_path /workspace/NeMo/exp/buckets/baseline \
  --candidate_name "Model B" \
  --candidate_path /workspace/NeMo/exp/buckets/candidate \
  --benchmarks libritts,riva_en,riva_en_hard_sentences \
  --s3_endpoint https://your-s3-endpoint \
  --s3_bucket your_bucket_name \
  --s3_region us-west-2 \
  --task_id NEMOTTS-2007 \
  --audio_report \
  --audio_report_benchmarks libritts,riva_en \
  --samples_per_benchmark 20
```

## Notes

- `magpietts_inference` supports several repetitions, but this script compares
only artifacts from repetition `0`.
- `--results_subdir` is not the experiment root. It is the subdirectory inside 
the experiment root that contains evaluation outputs such as metrics and generated audio. 
If evaluation artifacts are stored directly inside the experiment root, `--results_subdir` 
should be set to an empty string.
- Both generated reports are HTML reports uploaded to S3-compatible object storage.
- If the audio report is enabled, the evaluation report includes a link to the audio report.
- Audio files referenced by the audio report are uploaded separately and linked through presigned URLs.
- Box plot images are also uploaded to S3 and embedded into the evaluation report via presigned URLs.
- Presigned S3 links expire. Both generated reports include the expiration time directly in the HTML page. 
The default expiration time is one year.
- The expiration time is also included as a suffix in the uploaded artifacts
directory name, using the format `%Y-%m-%dT%H-%M-%SZ`, so uploaded reports
can be filtered and deleted later if needed.
- Both generated reports include a clickable Jira link derived from `--task_id`. 
If no task ID is specified, the link points to the Jira project page.
- Metrics that compare generated audio with the context audio (`SSIM (pred vs context)`) are
reported only for benchmarks generated with audio context. In the summary section they are
macro-averaged and pooled across audio-context benchmarks only, and they are omitted from the
sections of text-context benchmarks (names ending with `_ct_text`).
- `SSIM (pred vs GT)` (speaker similarity against the ground-truth recording) does not depend on the
context type, so it is reported for text-context benchmarks as well. Every benchmark has
ground-truth audio, so a NaN value, aggregated or per file, fails the report: it indicates a broken
evaluation or manifest. Filewise metrics written before `pred_gt_ssim` was saved per file lack the key;
such benchmarks keep the metrics-table row (from the aggregated value) but skip the statistical test and
the box plot with a warning, and the pooled statistical test of the summary section includes the metric only
when every benchmark carries it in both buckets. Re-run the evaluation to include them.
- The audio report shows the target recording (the ground-truth utterance) for every sample next to the
generated audio; audio-context benchmarks also show the context prompt. Text-context benchmarks can be
listed in `--audio_report_benchmarks`; their sections have no context column.
- The audio report expects `predicted_audio_<idx>.wav`, `target_audio_<idx>.wav` and, for audio-context
benchmarks, `context_audio_<idx>.wav` under `audio/repeat_0` of each benchmark directory, as written by
`magpietts_inference`. When `--audio_report` is enabled these files are required for every benchmark listed in
`--benchmarks`, not only for those in `--audio_report_benchmarks`. Every sample scored in the filewise metrics
must have its generated file, and both buckets must contain the same samples; a mismatch fails the report with
the samples named.

## Maintenance

### Updating benchmarks

Supported benchmarks are declared in `BENCHMARK_META` in `reporting/constants.py`. Each entry maps
a benchmark name to a `BenchmarkMeta` holding its language code and the context type used when
the benchmark was generated:
- `ContextType.audio` (default) - speaker identity was conditioned on a context audio prompt;
- `ContextType.text` - speaker identity was conditioned on a text description; no context audio exists,
so the audio report sections of these benchmarks have no context column.

To add or remove a benchmark, update `BENCHMARK_META`; `SUPPORTED_BENCHMARK_NAMES` is derived from it.
Declare text-context benchmarks explicitly, otherwise the report fails because their context
speaker-similarity values are NaN.

### Updating metrics

By design, metrics are divided into two groups:
- **standard metrics**, used for reporting aggregated values in the evaluation report;
- **distribution metrics**, used for statistical tests and box plot visualization.

The metric specifications are defined in `reporting/metrics/specs.py`.

To add or remove a metric, update the metric registries in `reporting/metrics/registry.py`:
- `MetricsRegistry` - for standard aggregated metrics;
- `DistributionMetricsRegistry` - for metrics used in statistical tests and visualizations.

Two spec attributes control when a metric is reported:
- `optional` - the metric key may be absent. For `MetricsRegistry` entries (the four `eou_*_rate` metrics)
an absent or NaN aggregated value skips the row. For `DistributionMetricsRegistry` entries (`pred_gt_ssim`, in
artifacts written before it was saved per file) a key absent from every sample skips the statistical test and
the box plot with a warning, and the pooled statistical test of the summary section includes the metric only
when every benchmark carries it in both buckets. Metrics without `optional` fail the report when the aggregated
value is missing or NaN or when no sample carries the key. For every distribution metric, optional or not, a key
present in some samples only, or any NaN value, fails the report.
- `context_type` - the metric is reported only for benchmarks generated with this context type.
`ssim_pred_context_avg` and `pred_context_ssim` are restricted to `ContextType.audio`: they are
omitted from text-context benchmark sections, and the summary table and pooled statistical tests
use audio-context benchmarks only. For audio-context benchmarks these metrics are required, so
missing or NaN values fail report generation instead of being silently dropped.

### Modifying bucket structure

If the bucket structure changes, update the `BucketStructure` class, 
which defines how report artifacts are located inside an evaluation bucket.
See `reporting/models.py`.
