> [!IMPORTANT]  
> The `Update branch` button must only be pressed in very rare occassions.
> An outdated branch is never blocking the merge of a PR.
> Please reach out to the automation team before pressing that button.

# What does this PR do ?

Add a one line overview of what this PR aims to accomplish.

**Collection**: [Note which collection this PR will affect]

# Changelog

- Add specific line by line info of high level changes in this PR.

# Usage

- You can potentially add a usage example below

```python
# Add a code snippet demonstrating how to use this 
```

# GitHub Actions CI

The Jenkins CI system has been replaced by GitHub Actions self-hosted runners.

Trusted PRs run automatically through copy-pr-bot. For an untrusted PR, a maintainer can trigger CI by commenting
`/ok to test <head-sha>`; repeat this after a new push if the PR remains untrusted.

# Before your PR is "Ready for review"

**Pre checks**:

- [ ] Make sure you read and followed [Contributor guidelines](https://github.com/NVIDIA-NeMo/Speech/blob/main/CONTRIBUTING.md)
- [ ] Did you write any new necessary tests?
- [ ] Did you add or update any necessary documentation?
- [ ] Does the PR affect components that are optional to install? (Ex: Numba, Pynini, Apex etc)
  - [ ] Reviewer: Does the PR have correct import guards for all optional libraries?
  
**PR Type**:

- [ ] New Feature
- [ ] Bugfix
- [ ] Documentation

If you haven't finished some of the above items you can still open "Draft" PR.

## Who can review?

Anyone in the community is free to review the PR once the checks have passed.
[Contributor guidelines](https://github.com/NVIDIA-NeMo/Speech/blob/main/CONTRIBUTING.md) contains specific people who can review PRs to various areas.

# Additional Information

- Related to # (issue)
