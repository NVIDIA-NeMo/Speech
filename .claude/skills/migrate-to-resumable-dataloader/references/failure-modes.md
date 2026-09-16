# Failure-mode catalog

Failure signatures, triggers, and fixes for indexed + resumable Lhotse
migrations. These are generic patterns; verify exact file names and line numbers
against the user's checkout before citing them in a report.

## §1 - Compressed JSONL, Shar cuts, or tar paths

**Signature**: index build raises a `ValueError` saying the source requires
uncompressed JSONL or tar data but received a compressed path such as
`*.jsonl.gz` or `*.tar.gz`.

**Trigger**: an indexed source points at compressed cuts, manifests, or tar
files. Sidecar offsets require stable byte positions in seekable files.

**Fix**: re-export or materialize the source in an uncompressed seekable format.
For Shar-style data, export cuts as plain JSONL when sidecar indexing is needed.

## §2 - `extra_fields` or `slice_length` on indexed NeMo entries

**Signature**: an indexed NeMo iterator raises that `extra_fields` is not
supported, or data order diverges after slicing.

**Trigger**: the source applies runtime field injection or slicing while also
requesting indexed access.

**Fix**: preprocess the manifest offline so the indexed source already contains
all required fields and shard/slice layout. Drop `extra_fields` and
`slice_length` from the indexed YAML entry.

## §3 - Remote object reader is not seekable

**Signature**: `io.UnsupportedOperation: seek` or `tell` on first read of a
remote URL source.

**Trigger**: the code path uses a backend reader that does not implement the
seek/tell operations required by indexing.

**Fix**: ensure the remote-storage SDK is installed and that Lhotse routes the
path through the intended seekable/range-capable backend. For AIStore, verify
`aistore` is installed and `AIS_ENDPOINT` is set.

## §4 - Stdlib filesystem operations on URLs

**Signature**: `FileNotFoundError` from `open("s3://...")` or
`os.path.getsize("s3://...")`.

**Trigger**: a URL path reaches code that assumes local filesystem semantics.

**Fix**: route URL paths through the storage-aware reader and load index metadata
from the `.idx` file rather than local `os.path` calls.

## §5 - Too many memory maps for large shard counts

**Signature**: `OSError: [Errno 12] Cannot allocate memory` or system
`vm.max_map_count` exhaustion during startup.

**Trigger**: one memory map per `.idx` file across a very large number of shards.

**Fix**: load sidecars into resident arrays or otherwise reduce mmap count. The
sidecars are usually small enough that resident arrays are acceptable.

## §6 - Line-delimited JSON with `.json` extension rejected

**Signature**: index validation rejects a line-delimited JSON manifest with a
`.json` suffix.

**Trigger**: extension filtering assumes only `.jsonl` is valid, while some NeMo
manifests use `.json` for one-record-per-line JSON.

**Fix**: accept both `.jsonl` and line-delimited `.json` when the contents are
newline-separated records.

## §7 - Process pool OOM during index build

**Signature**: `concurrent.futures.process.BrokenProcessPool` after partial
index-build progress.

**Trigger**: too many workers parse large manifests or tar headers concurrently,
exceeding available process memory.

**Fix**: reduce worker count, split the blend/source list across multiple index
runs, or increase available memory.

## §8 - GPU container hook runs during CPU-only index build

**Signature**: container startup fails before Python runs, often around a GPU
runtime hook such as `nvidia-container-cli`.

**Trigger**: a CPU-only index build uses a container/runtime setup that assumes
GPU devices are present.

**Fix**: use CPU-safe container settings for index builds, or bypass/disable GPU
hooks when the runtime has no GPU access.

## §9 - AIStore SDK response shape changed

**Signature**: an AttributeError on fields returned by the AIStore batch API,
often in an error or empty-content path.

**Trigger**: code assumes one SDK response schema while the installed SDK returns
another.

**Fix**: normalize SDK response attributes at the boundary and use that helper at
all consumer sites. Avoid raw direct field access in error-handling code.

## §10 - `shard_seed: "randomized"` with stateful dataloading

**Signature**: usually silent. Resume is not bit-exact even though the dataloader
snapshot appears to restore.

**Trigger**: randomized shard/sampler seed is re-derived at chunk startup while
stateful sampler data is loaded from checkpoint.

**Fix**: pin `shard_seed` to a fixed integer, typically matching the top-level
training seed.

## §11 - Per-chunk seed rotation in launcher

**Signature**: silent model-level divergence across chunk boundaries. Data-order
state may restore, but dropout, augmentation, and other model/global RNG draws do
not match a continuous run.

**Trigger**: the launcher chooses a different seed for each resumable chunk.

**Fix**: use one invariant seed for the entire resumable chain. If the launcher
computes seeds from run index, override that behavior for indexed + stateful
runs.

## §12 - No mid-chunk checkpoint trigger

**Signature**: only epoch-boundary checkpoints exist; progress after the last
boundary is lost when a chunk is preempted or reaches walltime.

**Trigger**: checkpoint config relies only on long epoch boundaries or sparse
validation events.

**Fix**: add an appropriate step-based or time-based checkpoint trigger and keep
resume-required checkpoints from being pruned prematurely.

## §13 - Internal time guard does not catch external termination

**Signature**: the runtime sends SIGTERM/SIGKILL and no final checkpoint is
written.

**Trigger**: external cancellation, node failure, preemption, or walltime signal
bypasses the framework's graceful preemption callback.

**Fix**: leave a walltime buffer for graceful stops and rely on frequent
mid-chunk checkpoints as the primary mitigation.

## §14 - Worker or world-size mismatch on resume

**Signature**: `StatefulDataLoader` or indexed iterator state raises a mismatch
error during `load_state_dict`, or restored data order is invalid.

**Trigger**: chunk restores with different `num_workers`, world size, or
rank/worker topology than the chunk that saved the checkpoint.

**Fix**: keep topology invariant for a resumable chain. To change topology,
restart from model weights without restoring dataloader state.

## §15 - AIStore batch endpoint returns empty content

**Signature**: batch collation receives empty content for one or more requested
objects, often followed by a downstream `NoneType` or collation error.

**Trigger**: object is not available through the batch endpoint, credentials are
wrong, or batch and individual-object paths exercise different backend state.

**Fix**: verify object availability through the exact access mode used by
training. As a workaround, set `USE_AIS_INDIVIDUAL_GETS=true` and investigate
backend replication/permission issues separately.

## §16 - `indexes_root` points at missing node-local storage

**Signature**: `FileNotFoundError` or `.idx file not found` from an indexed
reader at startup.

**Trigger**: YAML points at a node-local path such as `/tmp/idx`, but the launcher
does not stage sidecars there before every chunk; or the staging destination does
not match YAML.

**Fix**: use a persistent shared mirror by default. If staging to node-local SSD,
ensure the preamble runs before training in every chunk and the YAML path matches
that destination exactly.

## §17 - Concurrent bucketing breaks bit-exact resume

**Signature**: silent data-order divergence across resume boundaries.

**Trigger**: a background bucketing producer advances the source iterator outside
the checkpointed main-thread state.

**Fix**: set `concurrent_bucketing: false` for resumable training so only the
checkpointed path advances the iterator.

## §18 - Iterable mode partitions when partition signal is missing or wrong

**Signature**: silent under-sampling or over-partitioning under distributed
environment variables.

**Trigger**: indexed iterators read rank/world environment directly instead of
using a dataloader-worker partition signal.

**Fix**: ensure partitioning is activated only by the intended worker init path.
Map-style mode should see the trivial `(0, 1)` partition.

## §19 - Iterable mode with non-indexed source in the chain

**Signature**: non-indexed sources appear on every rank/worker while indexed
sources are partitioned.

**Trigger**: `force_map_dataset: false` with a chain that mixes indexed and
non-indexed iterators.

**Fix**: convert every source in the iterable chain to indexed access, or split
or remove the non-indexed sources before launching training. Do not switch to
map-style training to bypass this unless the user explicitly approves a
temporary exception with the expected overhead.

## §20 - Iterable mode with randomized multiplexer seed

**Signature**: loud `ValueError` from the multiplexer, or silent source-weight
drift if no guard exists.

**Trigger**: each shard draws a different multiplexer RNG state and chooses a
different source at the same logical step.

**Fix**: pin multiplexer seed, usually through the top-level `shard_seed`.

## §21 - Iterable resume topology mismatch

**Signature**: indexed range or chain state reports `shard_id` / `num_shards` /
`world_size` mismatch on restore.

**Trigger**: a checkpoint saved under one distributed-worker topology is restored
under another.

**Fix**: keep `(world_size, num_workers)` invariant. To scale differently,
restart without dataloader state.

## §22 - Training left in map-style mode

**Signature**: long startup or step-time overhead from repeated sampler/manifest
work, especially at larger world sizes.

**Trigger**: migrated training YAML keeps `data.train_ds.force_map_dataset: true`
instead of enforcing iterable partitioning.

**Fix**: set `data.train_ds.force_map_dataset: false` and make every source in
the training iteration graph indexed and partition-compatible. If a source cannot
yet be indexed, mark the migration not launch-ready unless the user explicitly
approves a temporary map-style exception with the specific blocker and expected
overhead.

## §23 - Build/prefetch tool imports stock Lhotse/NeMo

**Signature**: `ModuleNotFoundError`, missing `lhotse.indexing`, or import errors
for indexed/resumable symbols.

**Trigger**: build-index or prefetch command does not place the modified NeMo and
Lhotse checkouts before stock packages on `PYTHONPATH`.

**Fix**: set `PYTHONPATH` or install the correct packages so helper scripts and
training use the same indexed/resumable implementation.

## §23 - Distributed backend errors hide an earlier Python exception

**Signature**: NCCL/watchdog/collective timeout or launcher-level distributed
failure appears after one rank already logged a Python traceback.

**Trigger**: one rank fails during data loading or collation; other ranks block
in distributed work until the backend times out.

**Fix**: inspect logs before the distributed timeout and identify the first
Python exception. Treat later backend chatter as a cascade unless it is the first
error in time.
