---
name: debug-training-logs
description: Debug distributed training failures (NeMo, Megatron, PyTorch) from worker stderr logs and optional AIStore daemon logs. Finds root cause across NCCL timeouts, data loading errors, and storage failures.
disable-model-invocation: true
allowed-tools: Bash Read Grep Glob Agent
argument-hint: <path-to-logs-dir> [ais-logs-dir]
---

# Distributed Training Log Debugger

You are debugging a distributed training job failure. The user will provide one or two directories:
- **Worker logs** (stderr from SLURM/torchrun): `$ARGUMENTS[0]` (required)
- **AIStore daemon logs** (proxy/target tarballs or extracted dirs): `$ARGUMENTS[1]` (optional)

Your goal: find the **root cause**, not just the symptom. There are often cascading failures — one root cause triggers many downstream errors. Trace backwards from the final crash to the original trigger.

## CRITICAL: Verification discipline

**You MUST double-check every conclusion before presenting it to the user.** Distributed training failures have cascading effects that make it easy to mistake a symptom for the root cause. Follow these rules:

1. **Check ALL ranks, not a sample.** Do not look at 5 ranks and assume the other 123 are the same. Use `sort -u` on the full output to find outliers. A single outlier rank can be the entire root cause.
2. **Verify NCCL state for EVERY rank.** Extract `last enqueued work` and `last completed work` for all ranks and check for ANY rank that differs. A rank with `enqueued == completed` (no pending ops) is fundamentally different from ranks with `enqueued > completed` (ops pending) — it means that rank never entered the stuck collective.
3. **Distinguish "First PG on this rank to signal dumping" from "Observed flight recorder dump signal from another rank".** The first is the initiator (its own watchdog fired). The second was notified (it may not even be in the collective). These are DIFFERENT failure modes.
4. **Before stating a root cause, re-verify it against the raw data.** Re-read the actual log lines. Do not rely on earlier summaries you wrote — they may have been wrong.
5. **If your analysis changes during the investigation, explicitly state what was wrong before and why.** Do not silently shift conclusions.

## Phase 0: Obtain logs

If the user has not provided worker logs, ask them to download and provide the SLURM error logs. The logs are typically on the compute nodes or in a shared filesystem. Suggest:

```bash
# From the user's machine, SCP the SLURM error logs:
scp <cluster>:/path/to/slurm/logs/error-<JOBID>-*.out ./training_logs/

# Or if the logs are on a shared filesystem accessible from a login node:
mkdir -p ./training_logs
cp /path/to/slurm/error-<JOBID>-*.out ./training_logs/
```

The user needs to provide ALL per-node error files (e.g., `error-JOBID-0.out` through `error-JOBID-N.out`) — not just one node. Without all files you cannot identify which specific rank caused the failure.

## Phase 1: Triage worker logs

### 1.1 Understand the job
Read the first 80 lines of a few log files to determine:
- Framework (NeMo, Megatron, PyTorch Lightning, DeepSpeed, etc.)
- Scale (GPU count, node count, ranks per node)
- What the job is doing (training, fine-tuning, inference)
- Whether it resumed from a checkpoint

### 1.2 Find the fatal errors
Search ALL log files in parallel for these patterns (priority order):

**Tier 1 — Process killers:**
```
NCCL.*timeout|Watchdog caught collective operation timeout
taking the entire process down
SIGTERM|SIGKILL|SIGABRT
CUDA error|CUDA out of memory|OOM
```

**Tier 2 — Training loop crashes:**
```
RuntimeError|Exception.*Error
AISBatchLoaderError|StopIteration
Traceback \(most recent call last\)
```

**Tier 3 — Data loading / IO:**
```
Connection reset|Connection broken|Connection refused
retrying [0-9]+/[0-9]+
timed out|deadline exceeded
broken pipe
```

### 1.3 Identify the initiator and the straggler
For NCCL timeouts, you MUST check ALL ranks — do not sample. Extract the full NCCL state for every rank:

```bash
# Get watchdog state for all ranks that fired their own watchdog:
grep "failure detected by watchdog" error-*.out | grep -o "Rank [0-9]*.*last enqueued work: [0-9]*, last completed work: [0-9]*" | sort -u

# Get ranks that were NOTIFIED (did not fire their own watchdog):
grep "Observed flight recorder dump signal from another rank" error-*.out | grep -o "Rank [0-9]*"

# Count total unique ranks found vs expected:
grep "failure detected by watchdog" error-*.out | grep -o "Rank [0-9]*" | sort -t' ' -k2 -n -u | wc -l
```

**The rank that "Observed" instead of "detected" is likely the straggler** — it had no pending NCCL operations because it never entered the collective. Check its `Last enqueued NCCL work` — if it equals `last completed NCCL work`, the rank was stuck OUTSIDE NCCL (in the training loop, data loading, etc.), not inside a collective.

For NCCL BROADCAST/ALLREDUCE timeouts, calculate when the collective started:
`start_time = timeout_time - timeout_ms` (usually 1800000ms = 30min)

### 1.4 Count and classify
- Count `Connection reset` across all files (total and per-file)
- Check retry patterns: are retries always 1/N (recovering) or do they escalate to N/N (exhausted)?
- Count unique error types and affected ranks
- Look for `AISBatchLoaderError` or similar batch loader errors — these indicate AIStore returned fewer objects than requested

### 1.5 Build the timeline
Determine the **causal chain**: which error happened first, which are consequences.
The pattern is usually:
```
Data loading error (root cause)
  -> Some ranks crash out of training loop
  -> Crashed ranks can't participate in NCCL collective
  -> NCCL collective hangs for timeout period (usually 30min)
  -> Watchdog kills all remaining ranks
```

### 1.6 Check NeMo-specific synchronization points

NeMo has per-step collective operations that can cause rank desynchronization:

- **PreemptionCallback** (`nemo/utils/callbacks/preemption.py`): Calls `torch.distributed.broadcast(interrupted, 0)` at the end of EVERY training step via `on_train_batch_end`. If one rank's training step takes >30 min longer than others, this broadcast times out.
- **NeMoModelCheckpoint** (`nemo/utils/callbacks/nemo_model_checkpoint.py`): Multiple `trainer.strategy.broadcast()` calls during checkpoint save/load.
- **DDP gradient all-reduce**: Automatic per-step synchronization during backward pass.
- **`broadcast_buffers`** (DDP default=True): Broadcasts model buffers (e.g., batch norm stats) from rank 0 at each forward pass.

If rank 0 is ahead of other ranks in NCCL SeqNum, check whether the gap matches the number of per-step collectives (PreemptionCallback broadcast + DDP allreduce + broadcast_buffers = ~3 ops per step).

## Phase 1.7: Request AIStore logs if not provided

If the analysis points to storage I/O issues (connection resets, data loading errors, timeouts) and the user did NOT provide AIS daemon logs, suggest downloading them. First check if the `ais` CLI is available (`which ais`). If it is, guide the user through:

1. **Set the cluster endpoint** (ask the user for the endpoint URL):
   ```bash
   export AIS_ENDPOINT=https://<ais-cluster-endpoint>:<port>
   ```

2. **Set the auth token** (ask the user for the token value):
   ```bash
   export AIS_AUTHN_TOKEN=<token>
   ```

3. **Handle TLS** — skip cert verification or point to the CA cert:
   ```bash
   # Option A: skip verification
   ais config cli set cluster.skip_verify_crt=true
   # Option B: set CA cert
   export AIS_SERVER_CRT=/path/to/ca.crt
   ```

4. **Download all cluster logs** to the current log directory:
   ```bash
   ais log get cluster <path-to-worker-logs-dir>/ais_logs
   ```
   This downloads TAR.GZ archives from all proxy and target nodes.

5. Extract and analyze per Phase 2 below.

If the `ais` CLI is not installed, it can be built from the AIStore repository (`cd cmd/cli && go install .`) or downloaded as a binary. Alternatively, ask the user to download the logs manually from the AIS cluster.

## Phase 2: Debug AIStore logs (if provided)

### 2.1 Extract and orient
If tarballs (`.tar.gz`), extract them:
```bash
mkdir -p extracted && cd extracted
for f in ../*.tar.gz; do
  name=$(basename "$f" .tar.gz)
  mkdir -p "$name" && tar xzf "$f" -C "$name"
done
```

**AIS log file naming and time ranges:**

AIS daemon logs follow the naming convention:
```
aistarget.ais-target-N.INFO.MMDD-HHMMSS.1    # target logs
aisproxy.ais-proxy-N.INFO.MMDD-HHMMSS.1      # proxy logs
```

A single daemon (target or proxy) may have **multiple log files** — each file covers a specific time range:
- The `MMDD-HHMMSS` in the filename is the **start time** of that file
- The file covers from its start time **until the start time of the next file** for the same daemon
- If there is no next file, it covers until the daemon stopped or the logs were collected
- A new file is created when the daemon restarts (crash, upgrade, maintenance cycle)

**To find the correct file for a failure window:**
1. List all files for each daemon, sorted by the timestamp in the filename
2. For each daemon, find the file whose start time is BEFORE the failure window AND whose next file's start time is AFTER the failure window (or there is no next file)
3. A daemon may have been restarted DURING the failure window — check if any file starts within the window (this indicates a restart, which is itself significant)
4. Log lines within a file only have TIME (HH:MM:SS), not dates. If a file spans midnight, the same time (e.g., "21:30") may appear twice — once for each day. Use surrounding context (stats counter values, known events) to disambiguate which day a log line belongs to.

**Always check ALL files for a daemon, not just the latest.** The latest file may only cover minutes after a restart, while the failure evidence is in an earlier file.

**Timezone verification:**
AIS daemon logs and worker (SLURM/torchrun) logs may be on different machines in different timezones. NEVER assume they match. To verify:
1. Look for AIS periodic timestamp markers: `common:NNN DD Mon YY HH:MM UTC =============` — these explicitly state the timezone (typically UTC).
2. Look for absolute timestamps in worker logs: NeMo uses `YYYY-MM-DD HH:MM:SS`, SLURM uses `YYYY-MM-DDThh:mm:ss` — but neither includes timezone by default.
3. **Cross-reference a known event** visible in both log sources. The best anchor is the job death: find the SLURM `CANCELLED` timestamp in worker logs, then find the exact moment `get.n` stops incrementing in AIS stats. If they align, the timezones match. If they're offset by a round number of hours, one system is in a different timezone.
4. If timezones don't match, apply the offset consistently before correlating events.

### 2.2 Check target stats during failure window
AIStore targets emit periodic stats lines (every ~3 min). Extract key counters around the failure time:

**Critical counters to track:**
- `err.get.n` — GET errors (should be stable; spikes indicate problems)
- `err.getbatch.n` — batch GET errors
- `err.http.write.n` — broken HTTP responses to clients
- `err.put.n`, `err.head.n`, `err.lst.n` — other operation errors
- `get.n` vs `err.get.n` — compute error rate
- `getbatch.n`, `getbatch.obj.n` — batch operation counts

Compare values between consecutive stats lines to find the **delta** (new errors in that interval).

### 2.3 Search for error-level messages
```
grep "^E " <logfile>          # Error messages
grep "^W " <logfile>          # Warning messages
```

**Key error patterns in AIStore:**
- `x-get-batch.*out-of-bounds index` — batch GET lost objects during inter-target streaming
- `shared-dm.*terminated.*broken pipe` — inter-target data mover stream failure
- `shared-dm: xid.*not found, dropping recv` — objects dropped because batch job already aborted
- `resource pressure: load=critical` — target under disk/memory/CPU pressure
- `lcache.*hk.*dsk=critical` — disk at critical level, housekeeping skipped
- `gc:.*free mem` / `oom:` — memory pressure / forced GC

### 2.4 Check proxy logs
The proxy orchestrates batch GETs. Check proxy stats for:
- `err.get.n` — should be near zero; high = proxy-level routing failures
- `err.http.write.n` — proxy dropping client connections

If proxy errors are stable but target errors are spiking, the problem is target-side (disk, memory, inter-target networking).

### 2.5 Correlate the `x-get-batch` failure mechanism
The typical AIStore batch GET failure chain:
```
1. Target under resource pressure (dsk=critical, mem=low)
2. Inter-target shared-dm streams break (broken pipe)
3. x-get-batch gets out-of-bounds index (recv'd len=0)
4. Batch job aborted, subsequent objects dropped (xid not found)
5. Client receives fewer objects than requested
6. Lhotse/client batch loader raises error (iterator exhausted prematurely)
```

## Phase 3: Synthesize

### 3.1 Write the report
Structure your output as:

**Job Details** — framework, scale, start time, data source

**Timeline Table** — chronological events with timestamps, source file:line

**Root Cause Chain** — numbered chain from trigger to final crash, with arrows

**Key Files** — which log files contain the critical evidence

**Recommendations** — actionable fixes (storage-side, client-side, training config)

### 3.2 Classify the failure
Common root cause categories:
- **Storage I/O**: disk pressure, broken pipes, connection resets, batch object loss
- **Network**: NCCL timeout without data errors, NIC failures, switch issues
- **GPU/CUDA**: OOM, ECC errors, CUDA assertions
- **Data**: corrupt/missing data files, manifest mismatches, schema errors
- **Software**: version mismatches, config errors, OOM in Python process
- **Infrastructure**: node failure, preemption, SLURM timeout
- **Data loading stall**: single rank stuck in data loading (no timeout on read), blocking all other ranks at the next collective

## Reference: AIStore error counter meanings

| Counter | Meaning |
|---|---|
| `err.get.n` | Individual object GET failures on this target |
| `err.getbatch.n` | Batch GET operation failures |
| `err.http.write.n` | Failed HTTP writes to client (connection dropped) |
| `err.append.n` | Append operation failures |
| `err.put.n` | PUT operation failures |
| `err.head.n` | HEAD (metadata) operation failures |
| `err.lst.n` | List operation failures |
| `err.kalive.n` | Keep-alive failures |
| `getbatch.n` | Total batch GET operations |
| `getbatch.obj.n` | Total objects served via batch GET |
| `getbatch.throttle.ns` | Time spent throttling batch GETs |

## Reference: NCCL timeout anatomy

- `Watchdog caught collective operation timeout` — the NCCL watchdog detected a stuck collective
- `SeqNum=N, OpType=BROADCAST/ALLREDUCE` — which collective and sequence number
- `last enqueued work: N, last completed work: M` — work M completed, work M+1 is stuck
- `Timeout(ms)=1800000` — 30-minute timeout (default)
- `First PG on this rank to signal dumping` — THIS rank initiated the cascade
- `Observed flight recorder dump signal from another rank` — this rank is reacting to another's timeout
- `To avoid data inconsistency, we are taking the entire process down` — watchdog kills the process

### Distinguishing data loading stalls from GPU/NCCL hangs

The `last enqueued work` vs `last completed work` state is critical for determining whether the hang is in the CPU (data loading, training loop) or in the GPU (NCCL communication):

- **If `enqueued == completed` (no pending ops)**: This rank has NO NCCL work in-flight. It is stuck in the CPU-side training loop (data loading, audio decoding, batch prep) and never entered the collective. **This is the straggler — the rank that caused the hang.**
- **If `enqueued == completed + 1`**: The rank has submitted exactly one operation that hasn't completed. It entered the collective but can't complete because the straggler rank hasn't joined.
- **If `enqueued > completed + 1`**: Multiple operations are queued — the CPU has progressed past the stuck point and submitted additional operations asynchronously (e.g., DDP gradient allreduce via hooks). Still waiting for the straggler.
- **If rank 0 has higher `enqueued`/`completed` than others**: Rank 0 (often the broadcast root) completed its side of a collective (send) but receivers can't complete (receive) because the straggler hasn't joined.

**The key pattern for a data loading stall:**
- 1 rank: `enqueued == completed`, `active collectives: 0`, "Observed flight recorder dump signal from another rank" — **this is the straggler**
- N-1 ranks: `enqueued > completed`, "failure detected by watchdog" — these are waiting for the straggler
- Rank 0: may be further ahead if it's the broadcast root (can complete send without receivers)

**The key pattern for a GPU fabric issue:**
- ALL ranks: `enqueued > completed` (all entered the collective), all show "First PG on this rank to signal dumping" — no straggler, the collective itself is broken

Compare `enqueued` counts across ALL ranks. Even ONE outlier changes the entire diagnosis.

## Reference: NeMo/Lhotse data loading pitfalls

Common data loading issues that cause single-rank stalls:

- **No timeout on Lhotse URL audio reads**: `AudioSource._prepare_for_reading()` calls `f.read()` with no Lhotse-level timeout. A stalled download blocks the DataLoader worker indefinitely.
- **`fault_tolerant=True` silently drops failed audio**: Failed audio files are skipped, reducing effective batch size per rank. Different ranks may have different failure rates depending on their shard assignments.
- **`.m4a` files lose extension in BytesIO**: When audio is downloaded from URLs and wrapped in BytesIO, the file extension is lost. Lhotse's `CompositeAudioBackend` cannot use the fast-path for m4a (TorchaudioFFMPEGBackend) and falls through to expensive cascading backend trials.
- **Connection reset on idle keep-alive**: AIStore closes idle HTTP connections after 30s (`DfltMaxIdleTimeout`). The Python SDK's urllib3 pool doesn't match this timeout, causing `Connection reset by peer` on stale pooled connections. These are caught and retried (always succeed on 1st retry) — they are noise, NOT a root cause.
- **Each rank gets disjoint data shards**: Lhotse splits shards via `src[rank::world_size]`. One rank may get shards with more corrupt files, larger audio, or slower storage targets.
