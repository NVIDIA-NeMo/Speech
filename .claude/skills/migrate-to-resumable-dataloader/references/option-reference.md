# Option reference - indexed + resumable Lhotse migration

Field-by-field reference for YAML and launcher settings that interact with
indexed access, `StatefulDataLoader`, distributed topology, and storage backend.
Line numbers in local code may drift; verify against the checkout in front of
you when producing a report.

## `data.train_ds`

| field | required value | purpose | see also |
|---|---|---|---|
| `indexed` | `true` | Routes supported sources through indexed adapters such as `IndexedJsonlReader` and indexed NeMo-tar readers. Without it, streaming/replay behavior remains active. | `nemo.collections.common.data.lhotse.dataloader`, `lhotse.indexing` |
| `use_stateful_dataloader` | `true` | Uses `torchdata.StatefulDataLoader` so dataloader iterator state can be saved in Lightning checkpoints. | NeMo Lhotse dataloader config |
| `force_map_dataset` | `false` for training | Enforces iterable partitioning across data-parallel ranks and workers. Map-style training has too much sampler/manifest overhead; if a source cannot yet be indexed, report the migration as not launch-ready unless the user explicitly approves a temporary exception. | failure-modes §§18-22, conflict-matrix |
| `indexes_root` | stable filesystem mirror, or node-local path populated before startup | Tells indexed readers where to find `.idx` sidecars. Prefer a persistent shared mirror. Use `/tmp/idx` only when the launcher stages indexes there before training. | failure-modes §16 |
| `seed` | fixed integer, invariant across chunks | Lightning reseeds Python/NumPy/Torch at chunk start. Rotating this across resumable chunks breaks model-level bit-exactness even when sampler state restores correctly. | failure-modes §11 |
| `shard_seed` | fixed integer, not `"randomized"` | Controls sampler/multiplexer RNG. Randomized shard seeds can diverge across resume and are invalid for multi-shard iterable partitioning. | conflict-matrix |
| `num_workers` | invariant between save and restore | `StatefulDataLoader` and iterable partition state depend on worker topology. | failure-modes §14, §21 |
| `concurrent_bucketing` | `false` for resumable training | Background bucketing producers can advance source iterators outside the checkpointed main-thread state. | failure-modes §17 |
| `force_iterable_dataset` | unset or compatible with `force_map_dataset: false` | Do not enable mutually exclusive dataset modes. The training target is iterable partitioning through `force_map_dataset: false`. | conflict-matrix |
| `force_finite` | unset/false for training | Training usually needs infinite or epoch-controlled iteration; finite mode is normally for validation. | validation section |
| `extra_fields` on indexed NeMo entries | unset | Indexed NeMo adapters cannot preserve arbitrary runtime field rewrites. Preprocess manifests instead. | failure-modes §2 |
| `slice_length` on indexed entries | unset | Slicing rewrites cut/audio access and has no stable index unless preprocessed. | failure-modes §2 |
| compressed `.jsonl.gz` / `.tar.gz` paths | reject for indexed sidecars | Indexing requires seekable uncompressed JSONL/tar inputs. Re-export or unpack first. | failure-modes §1 |
| `pipe:` paths | reject | Pipe commands are not seekable. Materialize data first. | `lhotse.indexing` |

## Training iterable partition (`force_map_dataset: false`)

This is the required training mode for efficient indexed/resumable runs. Do not
ship a migrated training config in map-style mode. If an indexing blocker
prevents iterable partitioning, mark the migration not launch-ready unless the
user explicitly approves a temporary exception.

| concern | requirement | purpose |
|---|---|---|
| Worker partition signal | Set only by NeMo/Lhotse worker init path | Prevents map-style mode from accidentally partitioning under `torchrun` environment variables. |
| All sources indexed | required | Non-indexed sources do not partition and will be duplicated across ranks/workers. |
| Multiplexer seed | fixed integer | All shards must pick the same source at each multiplexing step to preserve global weighted distribution. |
| Resume topology | invariant `(world_size, num_workers)` | Saved iterator state validates topology on restore. |

## `data.validation_ds`

| field | required value | purpose |
|---|---|---|
| `indexed` | `true` when validation sources need indexed access | Uses the same sidecar/index readers as training. |
| `force_map_dataset` | `true` | Validation should be finite and deterministic; map-style access is simpler. |
| `force_finite` | `true` | Prevents infinite validation loops when the training blend is infinite. |
| `use_stateful_dataloader` | usually `false` | Validation is normally run to completion and not resumed mid-loop. |
| `indexes_root` | same mirror as training unless intentionally separate | Validation readers need the same sidecars. |
| `seed` / `shard_seed` | fixed integers | Keeps validation deterministic. |

## Lightning / trainer settings

| field | recommendation | purpose |
|---|---|---|
| `resume_if_exists` or equivalent | enabled for resumable chains | Ensures later chunks restore checkpointed model, optimizer, scheduler, and dataloader state. |
| `resume_ignore_no_checkpoint` or equivalent | enabled for first chunk when supported | Allows chunk 1 to start without an existing checkpoint. |
| Checkpoint cadence | frequent step- or time-based saves | External termination may bypass graceful preemption callbacks. Avoid losing an entire chunk. |
| `save_top_k` / pruning policy | do not prune required resume checkpoints | Resume needs recent checkpoints and dataloader metadata. |
| `max_time_per_run` / walltime guard | comfortably below runtime walltime | Internal graceful-stop callbacks need teardown time. |
| `devices`, `num_nodes`, distributed topology | invariant across resume | Dataloader state is topology-sensitive. To scale differently, restart without dataloader state. |
| `max_steps` | stable across chain | Later chunks continue global step accounting. |

## Launcher contract

| concern | requirement | purpose |
|---|---|---|
| Per-chunk seed | invariant for all chunks in a resumable chain | Prevents model-level RNG divergence across resumes. |
| Index mirror availability | `.idx` sidecars exist before training starts | Indexed readers fail or fall back to slow behavior when sidecars are missing. |
| Optional index staging | YAML `indexes_root` matches the staged destination | Node-local paths such as `/tmp/idx` must be populated in every chunk. |
| `num_workers`, `world_size` | unchanged between save and restore | Required by stateful dataloading and iterable partitioning. |
| Python path / package selection | loads the NeMo and Lhotse versions with indexed/resumable support | Avoids accidentally using stock packages without the required code. |
| Container/runtime hooks | compatible with available CPU/GPU runtime | CPU-only index builds may need different container settings than GPU training. |

## AIStore environment

| env var | required when | purpose |
|---|---|---|
| `AIS_ENDPOINT` | any `s3://` / `ais://` source is read through AIStore | Points Lhotse/AIS clients at the proxy. |
| `USE_AIS_GET_BATCH` | remote tar/audio sources should be fetched lazily by batch | Avoids eager tar-reader construction for every remote shard. |
| `USE_AIS_INDIVIDUAL_GETS` | batch endpoint is unavailable or returns empty content | Falls back to per-object reads. Slower but useful for backend-specific failures. |
| `aistore` SDK | AIStore backend in builder/training container | Required by Lhotse AIStore access paths. |

## Index building

| concern | recommendation | purpose |
|---|---|---|
| Source format | uncompressed, seekable JSONL/tar or supported Shar cuts | Sidecar offsets must map to stable byte positions. |
| Workers | tune for memory and storage backend | Large manifests/tars plus many workers can OOM. Reduce workers or split blends. |
| Mirror destination | persistent shared filesystem when available | Reuse sidecars across runs and avoid per-launch rebuilds. |
| Remote sources | verify credentials/backend before building | Indexing remote data exercises storage credentials and byte-range access. |
| Reusability | build once per source path set | Existing sidecars can be reused while source contents and paths are unchanged. |
