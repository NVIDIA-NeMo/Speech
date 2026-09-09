# Migration report - `<config-stem>`

- **Generated**: <YYYY-MM-DD HH:MM>
- **Source YAML**: `<path/to/source-config.yaml>`
- **Patched YAML**: `<path/to/source-config-resumable.yaml>`
- **Source blend** (if inspected): `<path/to/blend.yaml>`
- **Patched blend** (if emitted): `<path/to/blend-resumable.yaml>`
- **Launcher** (if inspected): `<path/to/launcher>` (or "skipped - no launcher provided")
- **Storage workflow**: <filesystem-only | AIStore/remote | mixed | unknown>

## Summary

<One paragraph: what changed, severity counts, whether the patched YAML is ready
to launch, and what manual work remains.>

## Findings

### Fatal (must fix; auto-patching not possible)

- _none_ - OR -
- **`<field-or-path>`** (`<file>:<line>`): <explanation>
  - **Current**: `<value>`
  - **Recommended**: `<value>` (or "manual rewrite")
  - **Why fatal**: <reason auto-patch is unsafe or impossible>
  - **References**: <reference file/section>

### Errors (auto-patched; review the diff)

- **`data.train_ds.indexed`** (`<file>:<line>`): <description>
  - **Was**: `false` -> **now**: `true`
  - **Why**: <rationale>
  - **References**: <reference file/section>

### Warnings (review manually)

- **`<field-or-path>`** (`<file>:<line>`): <description>
  - **Current**: `<value>`
  - **Recommended**: `<value>`
  - **Why**: <rationale>
  - **References**: <reference file/section>

### Notes (informational)

- **`<field-or-path>`** (`<file>:<line>`): <description>

## Dedup Mode

<One paragraph: confirm training uses `force_map_dataset: false`; if not, mark
the migration not launch-ready and list the blocker or explicit user exception.>

- **Training target**: `force_map_dataset: false`. This enforces iterable
  partitioning and avoids map-style sampler/manifest overhead.
- **Validation/test target**: `force_map_dataset: true` unless intentionally
  testing iterable behavior; finite deterministic validation is simpler in
  map-style mode.
- **Blocker/exception**: if training still uses `force_map_dataset: true`, mark
  the migration not launch-ready unless the user explicitly approved an
  exception; list the unindexed source or runtime blocker, expected overhead, and
  work needed to move back to iterable training.

For training iterable mode, list:

- Sources confirmed indexed: <list>
- Multiplexer seeds confirmed integer: <list>
- World-size / num-workers commitment: `<W>` x `<NW>` for the full chain

## Data Blend Audit

<List unindexable entries such as compressed manifests/tars, `pipe:` paths,
unsupported `extra_fields`, `slice_length`, or mixed indexed/non-indexed chains.>

| entry | reason | upstream fix |
|---|---|---|
| `<source>` | compressed cuts/manifests | re-export as uncompressed seekable files |
| `<source>` | unsupported `extra_fields` | preprocess fields into the manifest |

## Launcher Review

<If launcher was inspected, list findings. Otherwise write "skipped".>

- **Per-chunk seed rotation**: <not detected | detected at file:line; must pin one seed>
- **Index access wired**: <persistent mirror | node-local staging | missing>
- **AIStore batch audio fetch**: <needed and enabled | not needed | missing>
- **Topology invariance**: <verified | not verifiable | violated>
- **Python path/package selection**: <verified | not verifiable | missing>

## Storage Workflow

<One paragraph: filesystem-only vs AIStore/remote workflow, whether manifests and
indexes are local/shared filesystem paths, and whether any prefetch/staging is
required.>

## Patched Output Diff

### `<config>.yaml` -> `<config>-resumable.yaml`

```diff
-  data.train_ds:
-    indexed: false
-    use_stateful_dataloader: false
-    shard_seed: "randomized"
+  data.train_ds:
+    indexed: true
+    use_stateful_dataloader: true
+    force_map_dataset: false
+    indexes_root: /shared/fs/.../indexes_mirror
+    shard_seed: 42
```

_(full diff inline)_

### `<blend>.yaml` -> `<blend>-resumable.yaml`

```diff
-  - type: lhotse_shar
-    shar_path:
-      cuts: s3://bucket/path/cuts.0.jsonl.gz
+  # Source excluded: compressed Shar cuts cannot be indexed.
+  # Re-export with uncompressed cuts or convert to another seekable format.
```

_(full diff inline)_

## Pre-flight Checklist

1. Build indexes via the generated `build-indexes-cmd.sh`.
2. Run a bit-exact dataloader resume check on the migrated config.
3. Confirm storage SDKs and environment variables required by the selected
   workflow.
4. Confirm `indexes_root` exists and is populated from every node/container that
   will train.
5. Run single-node single-chunk, single-node resume, then full-topology smoke.
6. Submit the real run.

## References

- `references/option-reference.md`
- `references/conflict-matrix.md`
- `references/failure-modes.md`
- `references/best-practices.md`
- `references/aistore-vs-non-aistore.md`
