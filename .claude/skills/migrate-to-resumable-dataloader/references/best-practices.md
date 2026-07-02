# Best practices - indexed + resumable Lhotse migration

Prioritized checklist for migrating a NeMo config to indexed access and
checkpointable dataloading.

## Tier 1 - non-negotiable

1. **Pin `seed` and `shard_seed` to fixed integers.** The sampler and model RNG
   must resume from a stable state. Avoid `"randomized"` for resumable chains.

2. **Use one seed across every chunk of a resumable chain.** Lightning reseeds
   global RNGs at chunk startup. Rotating the seed breaks bit-exact resume even
   when dataloader state restores correctly.

3. **Keep `num_workers` and distributed topology invariant.** Changing worker
   count, world size, or rank/worker assignment invalidates stateful dataloader
   snapshots and iterable partition state.

4. **Build `.idx` sidecars once per stable source path set.** Reuse a persistent
   index mirror across experiments. Rebuild only when source contents or path
   strings change.

5. **Disable concurrent bucketing for resumable training.** Background producer
   threads can advance iterators outside the checkpointed main-thread state.

## Tier 2 - strongly recommended

6. **Run a bit-exact dataloader resume check before sweeping.** Take a few
   batches, save dataloader state, take a few more as ground truth, restore in a
   fresh process, and compare the restored batches.

7. **Enforce `force_map_dataset: false` for training.** Map-style training has
   too much sampler/manifest overhead. Before launch, confirm every training
   source is indexed, multiplexer seeds are fixed, and topology is stable; if a
   source cannot be indexed, report it as a migration blocker instead of
   silently keeping map-style training.

8. **Use frequent checkpoint triggers.** External termination may not execute a
   graceful preemption callback. Step- or time-based saves reduce lost progress.

9. **Smoke test in stages.** Run single-node single-chunk, then single-node
   multi-chunk resume, then the intended full topology.

10. **Keep `.idx` files on a persistent filesystem by default.** Stage to
    node-local SSD only when direct filesystem reads are proven problematic, and
    ensure the YAML `indexes_root` matches the staged destination.

11. **Use AIStore batch fetching deliberately.** For remote tar/audio sources,
    `USE_AIS_GET_BATCH=true` avoids eager remote tar-reader construction. If the
    batch endpoint fails for a dataset, use `USE_AIS_INDIVIDUAL_GETS=true` as a
    slower fallback while investigating storage availability.

## Tier 3 - operational hygiene

12. **Tune index-build workers to memory and storage backend.** Many workers can
    OOM on large manifests or remote tar headers. Reduce workers or split the
    blend when needed.

13. **Keep optional prefetch steps explicit.** Manifest prefetch, index staging,
    and model-cache preambles should be visible in the launcher and documented in
    the report.

14. **Use CPU-safe container settings for CPU-only index builds.** Some container
    runtimes expect GPU hooks by default; bypass or disable them when the index
    build runs without GPU access.

## What not to do

- Do not trust `meta.pt` key presence alone as proof of bit-exact resume.
- Do not combine incompatible Lightning checkpoint triggers.
- Do not point `indexes_root` at a node-local path unless the launcher populates
  it before every chunk.
- Do not launch iterable training until every source in the chain has been
  audited and made partition-compatible.
- Do not use map-style training to bypass indexing blockers; mark the migration
  not launch-ready unless the user explicitly approves a temporary exception
  with the blocker and expected overhead.
- Do not set `LHOTSE_USE_WORKER_PARTITION` manually; it is an internal signal set
  by the dataloader worker initialization path.
