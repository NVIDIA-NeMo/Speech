# Setup, training, export and evaluation

## Public sources and environment

Use the public [Speech](https://github.com/NVIDIA-NeMo/Speech), [RL](https://github.com/NVIDIA-NeMo/RL) and
[Automodel](https://github.com/NVIDIA-NeMo/Automodel) repositories. Set `$SPEECH`, `$RL` and `$AUTOMODEL` to their
checkouts and record commit IDs. Inspect current implementations rather than relying on old experiment branches.

Required capabilities:

- Speech native RL checkpoint export: `examples/speechlm2/prepare_rl_export_config.py`, `to_hf.py`,
  `nemo/collections/speechlm2/models/salm_automodel.py`.
- RL native SpeechLM policy, ASR processor/environment and config
  `examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml`.
- Automodel `FreezeConfig` exact parameter selection used by Speech.

If a current release lacks these, check the public contribution state:
[Speech #16387](https://github.com/NVIDIA-NeMo/Speech/pull/16387),
[RL #4748](https://github.com/NVIDIA-NeMo/RL/pull/4748),
[Automodel #4377](https://github.com/NVIDIA-NeMo/Automodel/pull/4377).
Prefer merged code; otherwise explicitly pin the reviewed PR revision. Do not say an unmerged feature exists in main.
There is no dependency on Skills #1539. WER works without Skills; the optional CER scorer is described in
[Data and signal](data-and-signal.md).

Use the selected RL revision's [installation guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/about/installation.md)
and [SpeechLM guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/gspo-speechlm.md) for compatible Python,
PyTorch/CUDA, Automodel and vLLM versions. Follow their native-policy and generation-worker environment separation.
Do not blindly copy a whole historical environment or the single-GPU SFT skill's package pins into RL. Reuse a known
compatible image when available; avoid rebuilding native dependencies unnecessarily. A v0.7.0 image requires its
matching source/dependency set and is not proof that arbitrary current mainlines work inside it.

The standalone WER path needs Speech/RL/Automodel and their declared dependencies, not a Skills launcher installation.
For adapters, install their CPU scoring dependencies in the reward-worker environment as well. Record both policy and
generation interpreter paths/versions. The prepared policy Python is called `$SPEECHLM_GSPO_PYTHON` below; the serving
Python is `$SERVING_PYTHON`. Do not globally force every Ray actor into the policy environment.

Download an immutable full HF snapshot with the customer's existing authenticated Hub access. Preserve all shards,
config, tokenizer and generation assets. Never print credentials or embed them in a job/config. Set:

```bash
export SPEECHLM_GSPO_MODEL=/shared/models/nemotron-transcribe-snapshot
export SPEECHLM_GSPO_DATA=/shared/experiment/selected-train.parquet
export SPEECHLM_GSPO_OUTPUT_DIR=/shared/experiment/train-r1
```

These are customer-chosen placeholders. Ensure the same paths are visible on all workers. Stage code/manifests once;
do not copy shared audio or the model through every submission directory.

## Configure and launch

Use NeMo-RL's documented [cluster launch](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/cluster.md) to provision
the requested nodes and Ray. Run the driver below **once** in that allocation, not once per rank. Scheduler and
container commands depend on the customer's infrastructure; do not invent a universal `sbatch` or Skills command.

```bash
cd "$RL"
"$SPEECHLM_GSPO_PYTHON" examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml \
  grpo.max_num_steps=20 grpo.max_num_epochs=20 \
  checkpointing.save_period=10 checkpointing.keep_top_k=3
```

This is a bounded 20-update starting run. The higher epoch allowance permits repeated use of a small dataset; it does
not override the step limit or guarantee sufficient nonconstant groups. Validate resolved keys against the selected
revision. Longer runs need an explicit step budget and an appropriate checkpoint-retention count.

Keep these parts of the reference config unless the experiment specifically studies them:

| Setting | Reference behavior |
|---|---|
| Algorithm | Sequence-level GSPO; leave-one-out mean, full-group sample SD; no reference-policy KL |
| Batch | 20 admitted groups × 64 responses = 1,280 training responses; initial offer 40 groups |
| Admission | FIFO nonconstant-reward groups with bounded refills; never “largest spread first” |
| Optimizer | AdamW, LR 1e-6, betas 0.9/0.999, eps 1e-8, weight decay 0.01; five-update linear warmup then constant |
| Native policy | Explicit trainable list; selected tensors FP32; deterministic perception; CTC/MTP frozen |
| Parallelism | 4 nodes × 8 GPUs; policy EP8, generation TP8/EP8; microbatch 1 |
| Generation | Temperature 0.8, synchronous refit, speculation off |
| Context | Global total 62,464; output ceiling 17,278, further limited by row/remaining context |
| Other | Sequence packing off; router activation-checkpointing setting preserved |

Preserve the inherited dropout/masking/optimizer defaults too. Print the fully resolved config and compare intended
changes. Increasing nodes does not automatically increase the 1,280-response batch. If changing groups/responses,
update dependent batch sizes consistently and count effective exposure anew.

`grpo.val_period=0` in this recipe: checkpoint saving does **not** run development evaluation. Schedule explicit export
and dev decode when complete numeric checkpoints appear. Never read partially written weights. Inspect the selected
revision's refill and epoch-tail behavior when a run stops early; do not assume a configured refill threshold equals
the exact number of generation calls. Record actual completed optimizer steps and all generated/retained groups.

## Export the actual trained state

Use the full seed to prepare the native export config, then the supported distributed exporter. Example for step 20:

```bash
export MODEL_CONFIG=/shared/experiment/export-config.json
export EXPORT_OUTPUT=/shared/experiment/hf-step20
"$SPEECHLM_GSPO_PYTHON" "$SPEECH/examples/speechlm2/prepare_rl_export_config.py" \
  --seed "$SPEECHLM_GSPO_MODEL" --output "$MODEL_CONFIG"
"$SPEECHLM_GSPO_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$SPEECH/examples/speechlm2/to_hf.py" \
  class_path=nemo.collections.speechlm2.models.SALMAutomodel \
  checkpoint_format=nemo_rl \
  ckpt_path="$SPEECHLM_GSPO_OUTPUT_DIR/checkpoints/step_20/policy/weights/model" \
  ckpt_config="$MODEL_CONFIG" output_dir="$EXPORT_OUTPUT"
```

Run this inside an appropriate allocated export environment. Keep the native checkpoint for resume and the full HF
directory for serving. Verify the loaded step, tensor coverage, tokenizer assets and CTC/MTP preservation. Learned
weights must come from this checkpoint, not the seed. Canonical serving dtypes may round small updates; do not replace
the exporter with an ad hoc cast or claim every FP32 training delta survives BF16 serving.

Resume only from the same run's complete native checkpoint, with optimizer, scheduler and data cursor. Starting
from an HF export creates a new optimizer history; it does not resume the previous training state.

## Matched evaluation without a Skills PR

Reuse `examples/speechlm2/finetuning/vllm_task_eval.py` and `task_metrics.py` for supported tasks. Its JSON-lines
evaluation rows use `audio_filepath`, `text`, and `context`; use `context = instruction + "\n<|audio|>"` to preserve
the native training rendering exactly. The reward reference is `expected_answer` in training and `text` here.
Set `VLLM_PLUGINS=nemo_speechlm`. Disable the evaluator's default MTP with `--spec-k 0` for the matched initial run.

```bash
"$SERVING_PYTHON" "$SPEECH/examples/speechlm2/finetuning/vllm_task_eval.py" \
  --model "$SPEECHLM_GSPO_MODEL" --manifest /shared/experiment/dev.jsonl \
  --out /shared/experiment/seed-dev.jsonl --task asr \
  --max-model-len 62464 --max-tokens 17278 --spec-k 0
```

Repeat with `--model "$EXPORT_OUTPUT"` and a distinct output path. This helper currently instantiates a single-GPU
engine; it has **no tensor-parallel CLI argument**. If the seed/context does not fit, make a small job-local driver
using the same Speech plugin, prompt and scorer with `vllm.LLM(tensor_parallel_size=8, ...)`, as in the mining pattern.
Keep its greedy sampling, EOS and context rules identical for seed and candidate. Do not silently shorten inputs.
`--task mer` and `--task cls --labels <label-file>` cover the tutorial metrics; label parsing may differ from strict
reward, so report both. Generic `--task asr` is word scoring, not a Mandarin CER switch or a SOT cpWER evaluator.

The helper stores predictions but not every finish reason. When cap/empty/stop diagnostics matter, preserve those
fields in the job-local driver before scoring. Check exact row coverage and aggregate error counts; never average
utterance WER as if it were corpus WER. For language-balanced objectives, report the equal-language mean separately.
Use the customer's canonical evaluator when the tutorial scorer does not implement the required metric.
