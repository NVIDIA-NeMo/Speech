---
name: babysit-pr
description: Get a pull request to green CI. Diagnose and fix CI failures, push fixes, re-trigger CI via the "Run CICD" label, and repeat until all checks pass. Does not post comments — this is a local developer tool.
---

# babysit-pr

Get a PR's CI to green. Nothing else — no review comments, no PR comments, no status summaries. This skill is run locally by developers in their own sandboxes.

## Inputs

The PR number is the primary input. It may come from:
- An explicit argument: `/babysit-pr 567`
- The conversation context: "check on PR #567"
- A GitHub PR URL pasted into the chat

If no PR number is clear, ask for it before proceeding.

## Workflow

### Step 1 — Get the full picture

```bash
gh pr view <PR_NUMBER> --repo NVIDIA-NeMo/NeMo
gh pr checks <PR_NUMBER> --repo NVIDIA-NeMo/NeMo
gh pr diff <PR_NUMBER> --repo NVIDIA-NeMo/NeMo
```

Determine the current state:

| State                        | What to do                           |
|------------------------------|--------------------------------------|
| CI failing                   | Diagnose and fix (Step 2)            |
| Merge conflicts              | Resolve (Step 3)                     |
| Formatting check failing     | Wait — do NOT fix (see below)        |
| CI green                     | Done, nothing to do                  |
| CI pending                   | Wait for it to finish, then reassess |

### Checks to ignore

The **"Isort and Black Formatting"** workflow (`reformat_with_isort_and_black` job) auto-pushes formatting fixes. If that check is failing or pending:
- Do NOT fix formatting yourself — the auto-formatter will push a commit shortly.
- Wait for it to complete and for the new commit to land before assessing other failures, since the formatting push changes the HEAD SHA and may re-trigger CI.

### Step 2 — Fix CI failures

Check out the PR branch and inspect the failure logs:

```bash
gh pr checkout <PR_NUMBER> --repo NVIDIA-NeMo/NeMo
gh run list --repo NVIDIA-NeMo/NeMo --branch <branch-name>
gh run view <RUN_ID> --repo NVIDIA-NeMo/NeMo --log-failed
```

Before attempting a fix, check `git log` for recent commits. If you see a previous fix attempt that addressed the same failure and it is still failing, **stop and tell the user** — the issue needs human attention. Do not keep retrying the same fix.

Otherwise, identify the root cause, fix the code, and push:

```bash
git add <changed files>
git commit -s -m "<brief summary of fix>"
git push
```

### Step 3 — Re-trigger CI

After pushing a fix, add the "Run CICD" label to re-trigger the CI pipeline:

```bash
gh pr edit <PR_NUMBER> --repo NVIDIA-NeMo/NeMo --add-label "Run CICD"
```

The "CICD NeMo" workflow is triggered by this label and removes it automatically when done.

Then wait for CI to complete and reassess. Go back to Step 1.

### Step 4 — Resolve merge conflicts

If the PR branch has fallen behind main and has conflicts:

```bash
git fetch origin main
git rebase origin/main
# Resolve conflicts — keep the PR's intent, adopt refactors from main
git add <resolved files>
git rebase --continue
git push --force-with-lease
```

After rebasing, go back to Step 3 to re-trigger CI.

## Rules

- **Do NOT post comments on the PR.** This is a local tool — communicate with the user directly.
- **Do NOT address review comments.** The goal is green CI, not review resolution.
- **Do NOT fix formatting.** The `Isort and Black Formatting` action handles that automatically.
- **Do NOT retry a failed fix.** If your previous commit didn't resolve the failure, tell the user.
- **Do NOT create new PRs.** Push to the existing branch only.
- **Do NOT attempt to merge the PR or push to main.**

## What done looks like

CI is green (or the only remaining failures are pre-existing / flaky and you've told the user about them). That's it.
