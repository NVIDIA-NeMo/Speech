# CLAUDE.md / AGENTS.md

This file provides guidance when working with code in this repository.

## Project Overview

NeMo Speech — toolkit for training/deploying speech models (ASR, TTS, Speech LLM). Active collections: `asr`, `tts`, `audio`, `speechlm2`, `common`. No Megatron / Megatron Core / Transformer Engine — parallelism is PyTorch-native (DDP, FSDP2, TP/SP via DTensor).

## Build & Install

See the canonical installation guide — [`docs/source/starthere/install.rst`](docs/source/starthere/install.rst) (published at https://docs.nvidia.com/nemo/speech/nightly/) — for the uv, pip (bring-your-own Python/PyTorch/CUDA), Docker, and optional `compiled` (SpeechLM2/Automodel) install paths.

Dev quickstart matching the current CI/container baseline: `uv sync --locked --python 3.13 --extra all --extra cu13 --group test`. Use `cu12` for CUDA 12.x or omit the CUDA extra on macOS. `test` and `docs` are dependency groups, not extras.

## Code Style & Pre-commit

- **Line length: 119** (not default 88) — consistent across black, isort, flake8
- Black with `skip_string_normalization = true`
- isort with `profile = black`
- Jupyter Notebooks are excluded from automatic black reformatting (see `extend-exclude`), but can be still reformatted when passed directly. Do not reformat notebooks outside your changes.
- **Helper placement**: keep public APIs and top-level classes/functions near the top of a file; place private
  helpers and utilities at the bottom of the file unless a local module convention requires otherwise.

Set up the repository's checks once; the installed hook then checks staged files automatically on every commit:

```bash
uv tool install pre-commit
pre-commit install
```

For an immediate check, stage the intended files and run `pre-commit run`. After committing, verify the complete branch with `pre-commit run --from-ref origin/main --to-ref HEAD`. Use `pre-commit run --all-files` only when changing shared formatting configuration or when a full-repository check is needed. Hooks may modify files; review and stage those fixes, then rerun until clean.

## Testing

```bash
pytest tests/collections/asr -m "not pleasefixme" -v     # ASR tests, skip broken
pytest tests/collections/tts -m unit -v                  # TTS unit tests
pytest -k "test_name" tests/                             # Single test by name
```

Markers: `unit`, `integration`, `system`, `pleasefixme` (broken — skip), `skipduringci`.

## CI & PRs

- NVIDIA developers: feature branches off `main`; community: fork-based workflow
- Trusted PRs trigger CI automatically through copy-pr-bot. For an untrusted PR, a maintainer must comment `/ok to test <head-sha>`; repeat after a new push if the PR remains untrusted.
- E2E nightly tests: only when really needed. Add **"Run e2e nightly"** before CI starts; labels are read by the pre-flight job.
- `skip-linting` / `skip-docs` labels bypass those checks
- Formatting CI is check-only and does not fix the branch. Run pre-commit locally before pushing.
- CI: GitHub Actions in `.github/workflows/`

Every commit must carry a Developer Certificate of Origin sign-off. Create commits with `git commit -s`; when amending, use `git commit --amend --no-edit -s`. Before pushing, inspect every branch commit with `git log --format='%h %s%n%(trailers:key=Signed-off-by)' origin/main..HEAD` and repair any missing sign-off. The `-s` sign-off trailer is distinct from a cryptographic `-S` signature.

## PR Verification & Review

Before committing or requesting review:

1. Review the full diff against the target branch and run `git diff --check`.
2. Run pre-commit and the smallest relevant test set. Bug fixes require a regression test that fails before the fix; behavior changes require unit tests covering the new behavior and important edge cases.
3. Check whether public APIs, configuration, CLI behavior, examples, or user workflows changed. Update the relevant documentation in the same PR, or state why no documentation change is needed.
4. Record the exact checks run and any intentionally skipped checks in the PR description.

When reviewing a PR, explicitly assess whether unit test coverage is appropriate for the changed behavior and whether affected documentation is accurate and complete. Treat unjustified gaps in either area as actionable review findings.

## Documentation

Sphinx-based docs live in `docs/source/`. Build with:

```bash
uv sync --locked --group docs                        # one-time setup (matches CI)
uv run make -C docs clean html                       # full rebuild
uv run make -C docs html                             # incremental rebuild
```

Output goes to `docs/build/html/`. Open `docs/build/html/index.html` to preview locally.

Other useful targets: `make -C docs linkcheck` (verify external links), `make -C docs doctest` (run embedded doctests).

## Training & Inference

Entry-point scripts live under `examples/<collection>/`.

All scripts follow the same Hydra pattern — a `@hydra_runner` decorator points to a YAML config in a nearby `conf/` directory:

```python
@hydra_runner(config_path="conf", config_name="fast-conformer_transducer_bpe")
def main(cfg):
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)
```

Override any config value from the CLI with Hydra syntax: `python script.py model.optim.lr=1e-4 trainer.max_epochs=50`. Browse configs with `ls examples/<collection>/conf/` to see which models and variants are supported.

## Handy Scripts

Utility scripts live under `scripts/`. Key subdirectories: `speech_recognition/`, `speechlm2/`, `speaker_tasks/`, `tokenizers/`, `dataset_processing/`, `asr_language_modeling/`. Browse with `ls scripts/`.

Four frequently used data/training helpers:

- **`scripts/speech_recognition/estimate_duration_bins.py`** — estimate Lhotse dynamic-bucketing duration bins from a manifest or YAML input config. Usage: `python scripts/speech_recognition/estimate_duration_bins.py <input> -b 30 -n 100000`
- **`scripts/speech_recognition/oomptimizer.py`** — find the largest batch size per bucket that fits in GPU memory. Usage: `python scripts/speech_recognition/oomptimizer.py --pretrained-name nvidia/canary-1b` or point to a config with `--config-path`.
- **`scripts/speech_recognition/estimate_data_weights.py`** — compute per-dataset sampling weights from YAML input configs, with optional temperature re-weighting. Usage: `python scripts/speech_recognition/estimate_data_weights.py input.yaml output.yaml -t 0.5`
- **`scripts/speech_recognition/convert_to_tarred_audio_dataset.py`** — shard audio+manifest into tar files. Usage: `python scripts/speech_recognition/convert_to_tarred_audio_dataset.py --manifest_path=m.json --target_dir=./tar --num_shards=512 --max_duration=60.0`

## Architecture

- **Hydra + OmegaConf** for all config management (YAML configs)
- **PyTorch Lightning** for training orchestration
- **Lhotse** (>=1.32.2) for audio data loading
- Collections are semi-isolated domains sharing `nemo.core` and `nemo.collections.common`

## Subdirectory Instructions

Module-specific instructions can be added as `CLAUDE.md` or `AGENTS.md` files in subdirectories.

## Issue Reproduction

When fixing a bug, always:
1. First reproduce the issue with a minimal test case
2. Add the reproduction as a unit test
3. Then fix the issue
4. Verify the test passes

## Forbidden Operations

- Never push directly to `main`
- Never modify `.github/workflows/` without explicit instruction
- Never delete test files without explicit instruction
