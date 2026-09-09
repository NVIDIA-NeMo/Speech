# Conflict matrix - indexed + resumable Lhotse

Table format: `A | B | conflict | severity | resolution`.

Severities:

- **fatal**: automatic patching is impossible; data must be preprocessed or the
  launcher/storage setup must change.
- **error**: automatic patching is usually safe.
- **warning**: context-dependent; report clearly.
- **note**: informational.

| A | B | conflict | severity | resolution |
|---|---|---|---|---|
| `data.train_ds.indexed: true` | `extra_fields:` on indexed NeMo entries | Indexed adapters cannot preserve arbitrary runtime field rewrites. | fatal | Preprocess the manifest to materialize fields, then drop `extra_fields`. |
| `data.train_ds.indexed: true` | `slice_length:` on indexed entries | Slicing changes cut/audio access and has no stable sidecar unless preprocessed. | fatal | Re-shard or preprocess offline, then drop `slice_length`. |
| `data.train_ds.indexed: true` | compressed JSONL/Shar cuts or compressed tar paths | Compressed streams do not provide stable seekable offsets for sidecars. | fatal | Re-export uncompressed or materialize seekable sources. |
| `data.train_ds.indexed: true` | `pipe:` paths | Pipes are not seekable. | fatal | Materialize upstream data to files or a seekable backend. |
| `data.train_ds.force_map_dataset: true` | resumable training launch | Map-style training keeps too much sampler/manifest work on the main process. | error | Set `data.train_ds.force_map_dataset: false` after making every training source indexed and partition-compatible. |
| `force_map_dataset: true` | `force_iterable_dataset: true` | Dataset mode selection is contradictory. | error | Keep one mode. For training, use `force_map_dataset: false`; for validation/test, keep map-style unless intentionally testing iterable behavior. |
| `use_stateful_dataloader: true` | per-chunk seed rotation | Model-level RNG diverges across resumed chunks. | error | Pin one seed for the whole chain in YAML and launcher. |
| `use_stateful_dataloader: true` | `num_workers` changes between chunks | Saved dataloader state is incompatible. | error | Keep worker count invariant or restart without dataloader state. |
| `use_stateful_dataloader: true` | `world_size` / rank topology changes | Saved iterator and sampler state are topology-sensitive. | error | Keep topology invariant or restart without dataloader state. |
| `force_map_dataset: false` | any non-indexed source in the chain | Non-indexed sources do not partition and are duplicated across ranks/workers. | fatal | Convert all sources to indexed access or split/remove the non-indexed source. Do not switch to map-style training to bypass this unless the user explicitly approves a temporary exception. |
| `force_map_dataset: false` | multiplexer seed is `"randomized"` | Shards may choose different sources at the same step. | error | Use a fixed integer seed. |
| `force_finite: true` | training dataset | Can cap infinite training mixtures unexpectedly. | error | Use finite mode for validation/test only unless intentionally bounded. |
| Checkpoint cadence absent | external preemption / walltime kill | Chunk progress can be lost without mid-chunk saves. | warning | Add frequent step- or time-based checkpoints. |
| Node-local `indexes_root` | no prefetch/staging before startup | `.idx` files are missing at runtime. | error | Point to a persistent mirror or stage indexes before every chunk. |
| AIStore batch mode | objects unavailable through batch endpoint | Batch loader may return empty content or fail collation. | warning | Verify object availability, replicate data, or set `USE_AIS_INDIVIDUAL_GETS=true`. |
| Container lacks AIStore SDK | AIStore source paths | Remote reads may fall back to the wrong backend or fail. | error | Install a compatible `aistore` SDK in build/training containers. |
| CPU-only index build | GPU container hook requires GPU runtime | Container startup can fail before index build begins. | warning | Use CPU-safe container settings or bypass GPU hooks. |
