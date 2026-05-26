---
name: verify
description: Run style checks and tests on changed files to verify code quality before committing.
---

Run verification on the current changes:

1. **Find changed files**:
   ```bash
   git diff --name-only HEAD
   ```
   Also include any files you've been editing in this session.

2. **Style check** on each changed Python file or its parent directory:
   ```bash
   python setup.py style --scope <path>
   ```
   If issues are found, fix them with `--fix` and report what changed.

3. **Run relevant tests** based on which collection was modified:
   - `nemo/collections/asr/` → `pytest tests/collections/asr --download -m "not pleasefixme" -v --timeout=300`
   - `nemo/collections/tts/` → `pytest tests/collections/tts --download -m "not pleasefixme" -v --timeout=300`
   - `nemo/collections/audio/` → `pytest tests/collections/audio --download -m "not pleasefixme" -v --timeout=300`
   - `nemo/collections/speechlm2/` → `pytest tests/collections/speechlm2 -m "not pleasefixme" -v --timeout=300`
   - `nemo/collections/common/` → `pytest tests/collections/common -m "not pleasefixme" -v --timeout=300`
   - `nemo/core/` → `pytest tests/core -m "not pleasefixme" -v --timeout=300`

4. **Report results**: summarize passes and failures. For failures, show relevant error output and suggest fixes.
