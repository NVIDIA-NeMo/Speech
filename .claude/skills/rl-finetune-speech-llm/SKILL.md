---
name: rl-finetune-speech-llm
description: Adapt Nemotron Transcribe to customer audio with NeMo-RL GSPO. Use for reward-based speech fine-tuning, assessing rollout signal, selecting informative training recordings, choosing a bounded training run, or diagnosing RL that improves reward but not task quality. Covers train/dev/test inputs and prompts, transcript and task-specific rewards, stochastic mining, the native SpeechLM update surface, training, HF export, and matched evaluation. Use finetune-speech-llm for supervised fine-tuning instead.
---

# Reinforcement learning for Nemotron Transcribe

Turn the model's better sampled answers into a better model. Start with a small sample of customer audio, measure
whether useful alternatives exist, then train on recordings that expose those alternatives. Success is a better
held-out task score with acceptable retention—not a higher training reward.

The default seed is [`nvidia/Nemotron-3.5-Transcribe-30B-A4B`](https://huggingface.co/nvidia/Nemotron-3.5-Transcribe-30B-A4B).
Use its full Hugging Face snapshot, including the tokenizer, speech encoder, projection and auxiliary heads.
This skill uses NeMo-Speech, NeMo-RL and NeMo-Automodel.

## References

Paths beginning with `examples/` are relative to the Speech checkout. `$RL` means the separate NeMo-RL checkout.

| When | Read |
|---|---|
| Preparing rows, prompts and rollout selection | [Data and signal](references/data-and-signal.md) |
| Choosing a reward or a small-task preset | [Tasks and presets](references/tasks-and-presets.md) |
| Installing, launching, exporting and evaluating | [Execution](references/execution.md) |
| Running a small Mandarin–English experiment | [ASCEND end-to-end example](references/ascend-example.md) |
| Understanding the underlying GSPO recipe | [NeMo-RL SpeechLM guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/gspo-speechlm.md) |
| Reusing dataset preparation and task scoring | `examples/speechlm2/finetuning/README.md`, `task_metrics.py` in that directory |
| Choosing SFT when RL has no useful signal | `../finetune-speech-llm/SKILL.md` |

## 1. Define the experiment

Collect what is missing; use information the customer already supplied. Write a short run plan and continue within
the agreed compute budget. Do not require a separate approval round for every stage.

| Input | Record |
|---|---|
| Seed | HF repository, immutable revision and downloaded snapshot |
| Training data | Labeled audio, source/version, usable split and any language/domain balance |
| Development data | Data for prompt, hyperparameter and checkpoint decisions |
| Evaluation data | Held-out final measurement; never used for mining or checkpoint selection |
| Task and prompts | Desired output, user instruction(s), reference conventions and language information allowed at inference |
| Objective | Primary metric, target improvement, protected tasks/classes and acceptable degradation |
| Resources | GPU type/count, time/step limit, storage, scheduler and existing compatible image/environment |

Ask for a retention dataset when preserving another capability matters. If none is available, state that retention
will remain unmeasured. If splits are missing, create them from training material before sampling; separate speakers
or recordings where possible. Check audio identity and overlapping recording segments across splits, not just filenames.

Keep `plan.md`, `progress.md` and a compact experiment ledger in the customer work directory. Record source commits,
resolved configs, data membership, prompt text, metric code and output paths. Record failures as well as successes.

## 2. Establish a useful baseline

1. Compare a few references with greedy seed outputs. Resolve wrong language, script, label vocabulary, number style
   or prompt/reference mismatch before interpreting the error rate.
2. If no prompt is established, compare a few plausible prompts **on development data**. Keep the system message
   empty and put the user instruction before audio, as required by the native ASR processor. For a focused task,
   prefer one prompt; use prompt variation when robustness to several deployment instructions is itself a goal.
3. Freeze the chosen prompt, scoring/normalization, decoding settings and selection criterion. Measure the seed on
   development and retention data. A seed test decode may be cached, but do not inspect it to tune the experiment.
4. For label tasks, also establish a development-fitted no-training comparator when aliases, label priors or output
   parsing explain many errors. Apply the same allowed post-processing to compared models. Report strict and parsed
   results separately. An RL gain over a poorly prompted seed is not necessarily a gain over this comparator.

Report error-rate reduction as `(seed_error - candidate_error) / seed_error`; it is not a relative increase in
accuracy. For nearly perfect baselines, report error counts and uncertainty rather than an unstable percentage.

## 3. Find useful training signal

Read [Data and signal](references/data-and-signal.md). Reuse compatible cached rollouts first. Otherwise sample a
small, shuffled, stratified subset of the **training split** and decode a greedy answer plus eight stochastic answers
per recording. Use the intended training prompt and reward, temperature 0.8, and sufficient output/context limits.

Summarize by task/language/domain:

- valid completed responses, truncation/failure rate and cost;
- groups with different rewards, their spread, and groups containing an answer better than greedy;
- all-correct, all-wrong and mixed groups for binary tasks;
- representative actual errors: substitutions, omissions, repetitions, empty outputs, label confusions or code switches.

Extend uncertain, useful strata to 16 samples; use 32–64 only when the expected extra yield justifies it. Eight mining
samples and 64 online training samples serve different purposes. An all-equal eight-sample group is weak evidence,
not proof that all future responses will be identical.

Select useful contrast without demanding a perfect answer or a catastrophic alternative. Keep source/language and
protected-task coverage; avoid filling the dataset solely with the largest reward gaps. Small differences created
only by normalization quirks are not useful task signal. The oracle best-of-samples score is an opportunity estimate,
not an expected trained-model score.

To improve task/domain X while preserving Y, include representative **training data from both** in the RL mix.
Keep development and evaluation examples out of that mix. Mine useful reward variation in each domain and track
their actual retained groups; merely evaluating Y, or adding Y examples whose rewards are all equal, supplies no
retention learning signal. Adjust the mixture using development results for both X and Y.

If meaningful contrasts are scarce, inspect reward correctness, references and prompts, then sample a different
training stratum. If the model still cannot produce useful answers, explain the evidence and recommend SFT or new
data. Do not spend a long RL run trying to manufacture a capability absent from its samples.

## 4. Choose the smallest informative run

Read [Tasks and presets](references/tasks-and-presets.md). Start from the public native SpeechLM GSPO config and
change data, task reward, prompt, LR/step budget and evaluation cadence deliberately. Preserve the tested optimizer,
update surface, precision, synchronous rollout/refit, dropout behavior and exporter for the first comparison.

- Default starting LR: `1e-6`, five-update warmup, then constant. Small tasks: 20 updates first; at most 100 in the
  initial comparison. Export/evaluate every 10–20 updates. These are starting settings, not a promised improvement.
- Preserve 64 online responses per recording and 20 retained groups per update initially. The recipe offers 40
  groups, then takes the first 20 with nonconstant rewards; it does **not** rank and take the “20 best” groups.
- At 20 retained groups, 100 updates mean 2,000 group presentations and 128,000 response presentations, with possible
  repeated audio. Decoding discarded/refill groups costs extra. A selected 500–1,000-recording pool can be sufficient
  for a small study; count actual distinct contributors rather than demanding a large dataset up front.
- Preserve the recipe's explicit 269 trainable tensors / 1,074,318,400 scalars for this model. CTC, MTP, routers,
  experts and the output head remain frozen. Keep trainable storage FP32; serving storage is a separate contract.
- Four eight-GPU nodes are the reference training layout. More nodes do not automatically change batch size.
  Changing the 20×64 batch is a separate experiment that also changes sample exposure and may require an LR comparison.

Size context/output limits from the actual audio and task. A short label may need only 16 output tokens; a transcript
may need thousands. Never copy a classification cap into ASR or treat a tokenizer-only prompt length as audio-expanded
context length. Keep the data loader's epoch allowance large enough to reach the step budget, and stop on the step limit.

## 5. Train and use early results

Follow [Execution](references/execution.md). Record the fully resolved config; the YAML inherits defaults. Start
the bounded run in a fresh output directory. A restart from that directory is a resume, not a new seed comparison.

Use the first updates to inspect the actual run while training proceeds:

- Correct seed and update surface loaded; CTC is frozen; rewards, log probabilities, advantages and loss are finite.
- Correct audio/prompt grouping, nonzero loss masks and aligned **retained** rewards. Use `filtered_rewards` where
  available: raw offered-batch `rewards` may not align with retained row IDs.
- Retained groups and distinct contributing recordings by task/class, cumulative repeats, refill cost and throughput.
  Homogeneous “retention anchors” receive zero advantage; their presence in the manifest does not protect a capability.
- Checkpoint exports decode with the same prompt, scorer and serving settings as the seed. Retention/class recall
  can deteriorate while aggregate reward improves.

For retained group size G, the tested estimator subtracts the other G−1 rewards' mean, then divides by the full
group's sample standard deviation plus epsilon. Sequence-level clipped importance ratios weight these advantages.
Do not substitute a different estimator, reward clipping or KL penalty while debugging data selection.

If refills dominate, inspect which strata still have contrast and refresh the training-only selection. If reward
improves but dev quality does not, inspect paired outputs and the scoring contract before adding steps. If protected
classes or retention exceed the agreed tolerance, keep the earlier acceptable checkpoint and diagnose the change.
Do not silently extend the budget or change the metric to make a run look successful.

## 6. Select, evaluate and hand over

Choose the checkpoint on the predeclared development metric subject to retention/class constraints; include the seed
as a valid outcome. Export that checkpoint canonically, then evaluate it and the seed on the untouched evaluation
split under the same conditions. If the evaluation set has already influenced development, label it a reused benchmark.

Report overall and relevant slice metrics, error counts and paired changes. ASR needs insertion/deletion/substitution,
empty-output and repetition diagnostics; code switching needs monolingual and mixed slices; classification needs
macro recall, per-class recall, confusion and invalid-output rates. Use paired uncertainty estimates when the gain
is small. Frozen CTC weights do not establish timestamp retention when the speech encoder was updated.

Deliver a short result table plus exact commands, resolved config, selected data/rollout provenance, native checkpoint
and HF export paths, prompt/metric definitions, compute consumed, and the next justified action. Clearly distinguish
measured gains, unsuccessful trials and settings that have not been tested on the customer's stack.
