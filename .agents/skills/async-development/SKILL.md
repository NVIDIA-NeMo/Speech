---
name: async-development
description: Resume, refresh context, or continue the NeMo async CFG implementation under work/async_implementation using its current state, checkpoints, and acceptance evidence. Use for this project's handoffs and execution, not unrelated NeMo development.
---

# Async CFG development

Resolve paths below from the NeMo repository root.

0. Run commands inside the devcontainer with `env -u PYTHONPATH` (see WORKFLOW.md, Environment).
1. Read `work/async_implementation/STATUS.md` for the current state and evidence pointers.
2. Read `work/async_implementation/WORKFLOW.md` for the resume and update procedure.
3. Follow only the evidence references needed for the user's request.

For a context refresh, report status and next steps without launching workloads.
For a continuation, perform eligible work within existing authorization. A recorded
GPU stop requires recovery confirmation after that stop before any device query
or workload; invoking this skill does not confirm recovery.

Keep changing facts in STATUS.md and detailed evidence in work/. Do not embed
checkpoint numbers, capture names, or acceptance totals in this entry point.
If the work directory is unavailable, report the missing state rather than
reconstructing it from memory or starting a new evaluation.
