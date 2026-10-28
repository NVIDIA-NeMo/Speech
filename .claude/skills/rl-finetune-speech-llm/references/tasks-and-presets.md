# Tasks, rewards and starting presets

Use the customer's metric to define useful behavior. These settings narrow the first experiment; they are not a
catalog of guaranteed wins. The tested full-scale recipe and small studies used a fixed native update surface,
64 responses/group, 20 retained groups/update, temperature 0.8 and LR 1e-6 with five-update warmup.

## Task routing

| Task | Reward and evaluation | Prompt and important slices |
|---|---|---|
| Word-based ASR | Built-in fractional `1-WER`; also score with the customer's fixed benchmark normalizer | One instruction matching reference style; language, accent, duration, insertion/deletion and empty output |
| Mandarin ASR | Built-in CER path needs the optional public scoring API; otherwise a tested CER adapter | Preserve expected script; use the same character/space policy for mining and training |
| Mandarin/English code switching | ASCEND extension: `1-MER` with Chinese characters and English words | One verbatim, language-preserving instruction; Mandarin-only, English-only and mixed speech |
| CommonLanguage / language ID | Adapter: binary exact canonical-label correctness | Fixed full label vocabulary, one-label-only response; strict accuracy, macro/per-class recall and invalid labels |
| Speaker-attributed transcription | Define speaker/token normalization and permutation behavior explicitly | Dedicated speaker-tag prompt; score speaker attribution as well as words; plain WER is not cpWER |
| Other audio tasks | Explicit verifiable task reward; existing ASR WER is not a general task reward | Match desired output and reference schema; use SFT if meaningful alternatives are absent |

Avoid requesting punctuation/capitalization when references do not carry it. For a focused transcription task a
starting prompt is “Transcribe the audio.” For code switching, “Transcribe the speech verbatim in the languages spoken.”
is a candidate, not a measured winner. Choose it on development data against the customer's established prompt.
Do not supply the answer language or class in a per-row prompt when inference must discover it.

## Small task adapters

The public ASR environment dispatches to `nemo_rl/environments/asr_reward.py`. The general recipe supports WER and
Mandarin CER. The [ASCEND example](ascend-example.md) adds a small `reward_metric="mer"` branch and its matching
Speech tokenizer; use both coordinated revisions. `"accuracy_strict"` still requires a task adapter. Implement only
the missing dispatch branch, with pure CPU scoring and focused tests, in the customer's pinned RL source branch.
Keep the native data processor, grouping, estimator, optimizer and exporter unchanged. Package the adapter on all
reward-worker nodes; do not inject runtime monkey-patches.

- **MER:** use the example's `nemo.collections.speechlm2.parts.metrics.mer.mer_tokens`, tested against the existing
  `examples/speechlm2/finetuning/task_metrics.py:mer_tokens` convention (or the
  customer's established benchmark tokenizer). Apply edit distance to its token lists and return `1 - edits/ref_units`.
  Test Chinese-only, English-only, mixed boundaries, punctuation, numerals and empty references. Do not use whitespace
  WER for Chinese or CER for English words. Record the explicit empty-reference rule.
- **Labels:** validate references against the fixed vocabulary. Score one canonical label as 1 and wrong, multiple,
  empty or invalid labels as 0. Define whitespace/case handling once; never reward a substring mention of the right
  answer inside a list. Test aliases and regional labels. Mining, training and strict evaluation use the same function.
- Use row `task`/`reward_metric` to dispatch; the native processor already forwards these fields. New metadata needs
  an explicit processor change. Do not assume registering an arbitrary environment name works through
  `run_vlm_grpo.py`: inspect its `setup_response_data(..., is_vlm=True)` environment routing first.

Measure the adapter on stored responses before new decoding; compare it with the task evaluator on the same examples.
Any intentionally different evaluation parser/normalizer must be named and reported separately.

## Presets

All rows below inherit the native four-node/eight-GPU-per-node config and its full explicit trainable-name list.
Offline mining counts are separate from the **64** online training responses.

| Starting experiment | Train recordings | Updates / development checks | What it establishes |
|---|---:|---|---|
| Focused ASR adaptation | 500–1,000 informative recordings, plus meaningful protected-domain coverage | 20 first; compare up to 100 if dev improves; check every 10–20 | Whether the selected contrast transfers to greedy transcription |
| ASCEND-style code switching | About 500 train recordings spanning monolingual and mixed speech | 20, check at 10 and 20 | A small MER trial; inspect English retention separately |
| CommonLanguage-style labels | 500–1,000 selected recordings, covering the full vocabulary as far as useful signal permits | Check every 10, at most 100 initially | Whether RL beats the optimized no-training comparator without class collapse |
| Broad ASR continuation | A larger selected pool sized for the task mixture and expected reuse | Up to 100–400, check every 20 | Sustained multi-domain improvement; extend only with development evidence and budget |

The code-switching pilot motivated a short first run: modest overall gains were concentrated in mixed speech, while
English-only performance could regress. Label experiments improved raw accuracy but much less over a development-fit
no-training comparator; some classes had no positive samples and Chinese regional confusions were substantial.
Neither study establishes a general 20% relative improvement expectation. These observations motivate the slices and
stopping decisions above, not a new objective or universal quota.

Start with `1e-6`. A `5e-7` comparison is a reasonable **proposed** lower-LR option when retention or drift limits the
first run; it is not a validated winner. Compare from the same seed/data with a fresh optimizer/output directory and
the same update budget. Do not change LR, prompt, reward and batch simultaneously if the purpose is diagnosis.

In binary tasks, homogeneous groups cannot teach the missing class. More epochs or more all-wrong examples do not
fix that. Look for training strata with genuine positive samples, reconsider an unidentifiable label distinction,
or recommend SFT. Never silently merge classes or score a different label space to manufacture improvement.
