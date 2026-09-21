---
name: verify
description: Verify a NeMo Speech change or pull request before commit, push, or review by checking the complete diff, formatting, tests, documentation, and DCO sign-offs. Use when asked to verify changes, prepare a commit or PR, or review a PR's validation and coverage.
---

Verify the current change proportionally to its risk. Do not report success for checks that were not run.

1. **Establish the complete scope.** Inspect committed, staged, unstaged, and untracked changes:

   ```bash
   git status --short
   git diff --name-only origin/main...HEAD
   git diff --name-only
   git diff --cached --name-only
   ```

   Replace `origin/main` if the PR targets another branch. Read the full diff and any applicable nested `AGENTS.md` or `CLAUDE.md` files.

2. **Review tests and documentation.** Before running commands, determine whether the change is adequately covered:

   - Require a regression test for a bug fix and focused unit tests for new or changed behavior, including important edge cases. If no test is appropriate, require a concrete explanation.
   - Check whether changes to public APIs, configuration, CLI behavior, examples, or user workflows are reflected in the relevant documentation. If no documentation update is needed, record why.

3. **Run repository checks.** For staged changes, run:

   ```bash
   pre-commit run
   ```

   For committed branch changes, run `pre-commit run --from-ref origin/main --to-ref HEAD`. If pre-commit is unavailable, use `uvx pre-commit` for the same command. Review and stage hook fixes, then rerun until clean. Also run `git diff --check`.

4. **Run the smallest relevant tests.** Prefer focused test files or test names while iterating, then expand to the affected collection when practical:

   - `nemo/collections/asr/` → `uv run pytest tests/collections/asr -m "not pleasefixme"`
   - `nemo/collections/tts/` → `uv run pytest tests/collections/tts -m "not pleasefixme"`
   - `nemo/collections/audio/` → `uv run pytest tests/collections/audio -m "not pleasefixme"`
   - `nemo/collections/speechlm2/` → `uv run pytest tests/collections/speechlm2 -m "not pleasefixme"`
   - `nemo/collections/common/` → `uv run pytest tests/collections/common -m "not pleasefixme"`
   - `nemo/core/` → `uv run pytest tests/core -m "not pleasefixme"`

   Add `--with_downloads` only for tests marked as requiring model downloads. Use `--cpu` when supported and a GPU is unavailable. If documentation changed, install its dependencies with `uv sync --locked --group docs` and build it with `uv run make -C docs html`.

5. **Verify commit sign-offs.** Every commit in the PR must have a DCO `Signed-off-by` trailer:

   ```bash
   git log --format='%h %s%n%(trailers:key=Signed-off-by)' origin/main..HEAD
   ```

   Create commits with `git commit -s`. Add a missing trailer to your own latest commit with `git commit --amend --no-edit -s`; do not rewrite other contributors' commits without authorization.

6. **Report results.** Summarize the reviewed diff, test-coverage and documentation conclusions, exact commands and outcomes, commit sign-off status, and any skipped checks with reasons. For failures, include the relevant error and a concrete next step.
