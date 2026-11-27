---
name: cfg-preemption
description: Resume, refresh context, or continue the NeMo async CFG serving work (EasyMagpie / Nemotron TTS V2 in vLLM-Omni), currently adding preemption to async CFG, using work/async_preemption state, plans and evidence. Use for this project's handoffs and execution, not unrelated NeMo development.
---

# CFG preemption

Resolve paths below from the NeMo repository root.

0. Run commands inside the devcontainer with `env -u PYTHONPATH` (see WORKFLOW.md, Environment).
1. Read `work/async_preemption/STATUS.md` for the current state and evidence pointers.
2. Read `work/async_preemption/WORKFLOW.md` for the resume and update procedure.
3. Read `work/async_preemption/CONTEXT.md` when background is needed: how CFG pairing and async
   scheduling work, and why preemption is currently rejected.
4. Follow only the evidence references needed for the user's request. The previous task's notes
   are a read-only archive in `work/nemotts-114-dev_notes/`;
   its `CONTEXT.md` is the entry point.

For a context refresh, report status and next steps without launching workloads.
For a continuation, perform eligible work within existing authorization. A recorded
GPU stop requires recovery confirmation after that stop before any device query
or workload; invoking this skill does not confirm recovery.

Keep changing facts in STATUS.md and detailed evidence in work/. Do not embed
checkpoint numbers, capture names, or acceptance totals in this entry point.
If the work directory is unavailable, report the missing state rather than
reconstructing it from memory or starting a new evaluation.
