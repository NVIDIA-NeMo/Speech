# AIStore vs filesystem workflows

Indexed + resumable Lhotse can read audio/tar sources from a local filesystem or
from AIStore-compatible URLs. Manifests/cuts may be on disk in either workflow.
Choose the workflow from source path schemes, not from where the process runs.

## Detection

| signal | workflow |
|---|---|
| `tarred_audio_filepaths: s3://...`, `ais://...`, or `http(s)://...` | AIStore/remote workflow |
| `tarred_audio_filepaths: /path/...` or relative filesystem path | filesystem workflow |
| mixed local and remote paths | remote workflow, because it has the stricter requirements |

`AIS_ENDPOINT` in the environment is necessary for AIStore access, but it is not
sufficient evidence that the blend uses AIStore.

## Remote AIStore workflow

Required setup:

- `aistore` SDK installed in the build/training container.
- `AIS_ENDPOINT` exported into the process that reads remote sources.
- `USE_AIS_GET_BATCH=true` when remote tar/audio should be fetched lazily by
  minibatch instead of opening every shard eagerly.

Optional setup:

- `USE_AIS_INDIVIDUAL_GETS=true` to bypass the batch endpoint and fetch each
  object individually. This is slower but useful when the batch endpoint is
  unavailable or returns empty content for some objects.

Index building:

- The index builder reads remote tar files through AIStore byte-range capable
  paths and writes `.idx` sidecars to the configured index mirror.
- A successful index build proves byte-range access worked for the indexed
  source paths. It does not prove the batch endpoint will later serve every
  object successfully.

Runtime data access:

1. Keep manifests/cuts on a local/shared filesystem when random access would be
   inefficient from remote storage.
2. Point `data.*.indexes_root` at a persistent index mirror by default.
3. Use node-local index staging only when direct mirror reads are too slow or
   metadata-heavy; make the YAML path match the staged destination.
4. Use manifest prefetch only as a fallback for remote manifest paths that
   cannot be cached persistently.

## Filesystem-only workflow

Required setup:

- All audio/tar paths resolve through the local filesystem visible in the
  container/process.
- AIStore env vars are unset or ignored when no remote paths are present.
- `USE_AIS_GET_BATCH=false` unless a mixed remote source requires it.

Index building:

- The index builder reads local files directly.
- Filesystem throughput and metadata behavior determine the best worker count.

Runtime data access:

1. Keep manifests/cuts on a local/shared filesystem.
2. Point `data.*.indexes_root` at a persistent index mirror.
3. Stage indexes to node-local SSD only when needed and only with matching YAML
   paths.

## Common gotchas

- Do not infer workflow from runtime labels alone; inspect the source paths.
- Verify filesystem mounts inside the runtime/container, not only in the host shell.
- Reusing an index mirror requires identical source path strings and unchanged
  source contents.
- AIStore individual GETs and batch GETs can exercise different backend paths;
  test the exact access mode used by training.
