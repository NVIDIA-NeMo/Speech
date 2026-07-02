---
name: migrate-to-resumable-dataloader
description: This skill should be used when the user asks to "migrate to the resumable dataloader", "switch to indexed Lhotse", "adopt the indexed + resumable pipeline", "make my training resumable", "set up StatefulDataLoader for NeMo/Lhotse", "use AIStore GetBatch", or "convert this YAML to the resumable path". Walks a NeMo training YAML and optional launcher, data blend, and runtime context through the indexed + resumable Lhotse migration; lints interacting fields; auto-patches safe YAML changes; emits a migration report, pre-flight checklist, and index-build command. Static analysis only; never launches training.
argument-hint: '<config.yaml> [launcher.py] [blend.yaml] [runtime-notes]'
---

# Migrate a NeMo training YAML to indexed + resumable Lhotse

Use this skill to port a NeMo training config from streaming/replay-style Lhotse
loading to indexed access plus `torchdata.StatefulDataLoader` checkpoint/restore.
The migration is fragile because YAML flags, launcher seed policy, index paths,
storage backend, and resume topology all interact.

## Core concepts

- Indexed sources need `.idx` sidecars for random access into JSONL, tar, and
  supported Shar-style data. Build these once per blend/source set.
- `use_stateful_dataloader: true` lets Lightning checkpoint the dataloader
  iterator state, but only if seeds, worker counts, and distributed topology are
  stable across chunks.
- Training configs must use `force_map_dataset: false` so indexed sources
  partition across data-parallel ranks and workers without map-style sampler
  overhead. Treat `force_map_dataset: true` for training as not launch-ready
  unless the user explicitly approves a temporary exception; every source in the
  training iteration graph must be indexed and partition-compatible before
  launch.
- Remote audio on AIStore/S3 generally needs `USE_AIS_GET_BATCH=true` so audio
  fetches are deferred to sample time instead of constructing eager tar readers
  for every shard.

## Inputs

| input | required | source | purpose |
|---|---|---|---|
| Training YAML | yes | argument or `--config=` | Inspect `data.train_ds`, `data.validation_ds`, `trainer`, `exp_manager`, and any model fields that affect resume. |
| Launcher script | no | argument or auto-detect from project conventions | Check per-chunk seed policy, resume topology invariance, Python path setup, AIStore env vars, and optional index staging. |
| Data-blend YAML | no | resolved from `data.train_ds.input_cfg` when possible | Check indexability: compressed paths, non-seekable paths, unsupported `extra_fields`, `slice_length`, and mixed indexed/non-indexed chains. |
| Runtime context | no | argument, config file, or user-provided notes | Detect storage backend, AIStore endpoint availability, container constraints, and index mirror destination. |

## Outputs

Every output lands in `migrate-resumable/<config-stem>/` in the current repo:

| output | purpose |
|---|---|
| `migration-report.md` | Findings, rationale, patched fields, and unresolved blockers. |
| `<config-stem>-resumable.yaml` | Patched training config when safe automatic edits are possible. |
| `<blend-stem>-resumable.yaml` | Patched blend, only when a blend was inspected and safe changes are possible. |
| `pre-flight-checklist.md` | User-run steps before submitting training. |
| `build-indexes-cmd.sh` | One-shot index-build command using the project wrapper when available, otherwise the generic NeMo/Lhotse index builder. |

## Workflow

### 1. Discover and parse inputs

1. Resolve the training YAML path and read it with OmegaConf or a
   comment-preserving YAML parser.
2. Resolve any referenced blend YAMLs from `data.*.input_cfg`. Prefer project
   conventions when obvious, but fall back to paths relative to the config.
3. If a launcher path is supplied, read it. Otherwise inspect likely project
   launchers (`train.py`, `pretrain.py`, shell wrappers, or raw `torchrun` /
   `python` commands) and pick the closest match.
4. If runtime context is supplied, read it for container image, environment
   variables, filesystem mounts, worker counts, and AIStore endpoint settings.
5. Detect remote storage from source paths (`s3://`, `ais://`, `http(s)://`) and
   local filesystem storage from ordinary absolute or relative paths.

### 2. Run lint pipeline

Run every relevant check in:

- `references/option-reference.md`
- `references/conflict-matrix.md`
- `references/failure-modes.md`
- `references/aistore-vs-non-aistore.md` when remote storage is present

Each finding should include severity, field/path, current value, recommended
value, and a short rationale.

Severities:

- **fatal**: automatic patching is not possible; user must preprocess data or
  change the source layout.
- **error**: automatic patching is safe and should be applied.
- **warning**: context-dependent; emit a report item and optional YAML comment.
- **note**: informational; no patch.

### 3. Emit patched YAML and blend

Apply safe `error`-severity patches. Preserve comments when possible with
`ruamel.yaml`; otherwise serialize with OmegaConf/YAML and rely on the report for
rationale. For blend edits, never silently drop data: leave an explicit report
entry and comment for every excluded or rewritten source.

### 4. Generate `migration-report.md`

Use `templates/migration-report.md`. Include:

1. Summary of storage workflow, counts by severity, and readiness.
2. Inputs inspected.
3. Findings table.
4. Walkthrough for train data, validation data, trainer/exp manager, launcher,
   and storage backend.
5. Data-blend audit.
6. Verification and pre-flight steps.

### 5. Generate `pre-flight-checklist.md`

Use `templates/pre-flight-checklist.md` when present. Required steps:

- Build `.idx` sidecars for every training/validation/test blend involved.
- Verify `indexes_root` points at the same stable mirror used by the runtime, or
  that explicit node-local index staging populates it before training starts.
- If AIStore is in play: verify `aistore` SDK availability, `AIS_ENDPOINT`, and
  whether `USE_AIS_GET_BATCH` or `USE_AIS_INDIVIDUAL_GETS` is required.
- Verify one invariant seed across resumable chunks.
- Verify `num_workers`, `world_size`, and relevant distributed topology do not
  change across resume boundaries.
- Recommend a small smoke ladder: single-node single chunk, single-node resume,
  then full topology.

### 6. Generate `build-indexes-cmd.sh`

Prefer a project-provided wrapper when one is clearly present. Otherwise emit a
generic command using:

```bash
python <NeMo>/scripts/dataloading/build_indexes.py \
    --indexes-root <shared-index-mirror> \
    --workers <N> \
    <blend>.yaml [<validation-blend>.yaml ...]
```

If running through a managed runtime or container wrapper, include comments for required
container image, mounts, environment variables, worker count, and any CPU/GPU
container-hook workaround the project requires.

### 7. Print final summary to chat

Keep the final chat response under 10 lines: output directory, finding counts,
report path, and the next command the user should run.

## Knowledge base

- `references/option-reference.md`: field-by-field reference for YAML and
  launcher settings.
- `references/failure-modes.md`: known failure signatures, triggers, and fixes.
- `references/conflict-matrix.md`: incompatible option pairs.
- `references/best-practices.md`: priority-ordered checklist.
- `references/aistore-vs-non-aistore.md`: storage workflow selection.
- `templates/migration-report.md`: report template.
- `templates/pre-flight-checklist.md`: checklist template, when present.
- `scripts/analyze.py`: optional static-analysis helper, when present.

## Constraints

- Prefer static analysis. Do not launch training, build indexes, prefetch data, or
  modify external runtime state unless the user explicitly asks.
- Cross-check recommendations against the actual NeMo/Lhotse code in the user's
  checkout when paths are available. Relevant areas are common Lhotse dataloader
  config, indexed adapters, `lhotse.indexing`, AIStore batch loading, and NeMo
  dataloader construction.
- Treat project wrappers as optional conveniences, not as part of the generic
  migration contract.
- When evidence is missing, say so. Do not encode project-specific run history
  or local experiment names as general guidance.
