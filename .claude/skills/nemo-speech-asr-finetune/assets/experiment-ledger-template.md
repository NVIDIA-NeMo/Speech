# ASR Fine-Tuning Experiment Ledger

## Goal

- User goal:
- Target checkpoint:
- Architecture:
- Success metric:
- Optional guardrail metrics:

## Data

- Train manifests or input config:
- Validation manifest:
- Test manifest:
- Data sources and weights:
- Tarred/non-tarred:
- Manifest sharding:
- Transcript style policy:
- Style transform artifact:
- Original checkpoint output style:
- Style mismatch decision:

## Preflight

- Disk space:
- GPUs:
- Container/image:
- Manifest validation:
- Duration distribution:
- Text/token distribution:
- Duration/token filters:
- Examples/hours filtered:

## Lhotse And OOMptimizer

- Lhotse train:
- Lhotse validation:
- Bucketing mode:
- Duration bins:
- Bucket batch sizes:
- Static batch size:
- `bucket_buffer_size`:
- `shuffle_buffer_size`:
- `seed`:
- `shard_seed`:
- OOMptimizer settings:
- Training pilot utilization:
- CPU memory notes:

## Training

- Init checkpoint:
- Script/config:
- Precision:
- `sync_batchnorm`:
- `max_steps`:
- `limit_train_batches`:
- `val_check_interval`:
- LR:
- Scheduler:
- Warmup:
- Min LR:
- Save top K:
- Command/log path:

## Evaluation

| Model | Artifact | Prediction Manifest | Raw WER | Default WER | CER | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| baseline |  |  |  |  |  |  |
| final `.nemo` |  |  |  |  |  |  |
| best single |  |  |  |  |  |  |
| averaged |  |  |  |  |  |  |

Default WER uses capitalization and punctuation removal unless the user requested a different metric.

## Error Analysis

- Raw vs default WER gap:
- Worst sources/domains:
- Worst categories:
- Label/audio defects:
- Decoding findings:

## Decision

- Keep artifact:
- Drop artifacts:
- Next intervention:
- Reason:
- If validation/test influenced data or weights, blind holdout plan:
