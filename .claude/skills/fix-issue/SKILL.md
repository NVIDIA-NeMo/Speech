---
name: fix-issue
description: Fix a GitHub issue in NeMo Speech (NVIDIA-NeMo/NeMo). Read the issue, reproduce the bug with a failing test, implement the fix, and verify tests pass. Only opens a PR if the user explicitly asks for it.
---

# fix-issue

Fix a GitHub issue in NeMo Speech from first principles: understand the bug, prove it with a test, and fix the root cause.

The key discipline is **reproduce first, fix second**. Writing the failing test before touching the source code forces a precise understanding of what broken actually means, which keeps the fix focused and correct.

## Inputs

The issue number is the primary input. It may come from:
- An explicit argument: `/fix-issue 1234`
- The conversation context: "fix issue #1234"
- A GitHub issue URL pasted into the chat

If no issue number is clear, and the description provided is not good enough, ask for more details before proceeding.

## Before starting

Read the issue description carefully. Identify:
- What is the reported failure (error message, wrong output, crash)?
- Which collections in `nemo/collections` and components are affected?
- Are there any repro steps already in the issue? If so, run them first.
- Is there a linked PR or related issue that gives more context?

## Workflow

1. Read the issue: `gh issue view <ISSUE_NUMBER> --repo NVIDIA-NeMo/NeMo`
2. Understand the bug — identify the relevant code
3. Write a minimal reproduction test in `tests/` that demonstrates the failure
4. Run the test to confirm it fails: `pytest <your_test_file> -v`
5. Implement the fix in the source code
6. Run the test again to confirm it now passes
7. Run the broader test suite for the affected collection to check for regressions:
   ```bash
   pytest tests/collections/<collection>/ -m "not pleasefixme" -v --timeout=120
   ```
8. Tell the user the fix is ready and what was changed

## Opening a PR

Only create a branch and open a PR if the user explicitly asks for it. When they do:

```bash
git checkout -b fix/<ISSUE_NUMBER>-<short-description>
git add <changed files>
git commit -s -m "Fix <short-description> (closes #<ISSUE_NUMBER>)"
git push origin fix/<ISSUE_NUMBER>-<short-description>
gh pr create --repo NVIDIA-NeMo/NeMo \
  --title "Fix <short-description>" \
  --body "$(cat <<'EOF'
# What does this PR do ?
<one-line overview>

# Changelog
<line-by-line high-level changes>

# Usage
<usage example if behavior changed>

Fixes #<ISSUE_NUMBER>
EOF
)"
```

Notes:
- Add specific files by name — do not use `git add -A` or `git add .`.

## Rules

- **Reproduce first.** Do not attempt a fix without a failing test that demonstrates the bug.
- **Do NOT open a PR unless the user asks.** The default is local-only: fix the code, verify it works, report back.
- **Do NOT post comments on the GitHub issue.** Communicate with the user directly.
- **Do NOT reformat files outside your changes.** The `Isort and Black Formatting` action handles formatting automatically.
- **Never push to `main` directly.**
- **Never attempt to merge the PR yourself.**
