# SALM fine-tuning toolkit

Tools for adapting a SALM checkpoint (a speech encoder, a projection, and an LLM backbone, exported by
`SALMAutomodel`) to a new task on a single GPU, and evaluating it with vLLM. They power the
[SALM fine-tuning tutorials](../../../tutorials/speechlm2/finetuning/README.md): sixteen tasks spanning ASR, speech
translation, spoken QA, intent, emotion, sounds, music, language ID, pronunciation scoring, anti-spoofing and
stuttering detection.

**Start here.**
- **Run a recipe end to end:** the [tutorials](../../../tutorials/speechlm2/finetuning/README.md).
- **Understand the concepts, judgment calls and pitfalls:** [docs/fine-tuning-guide.md](docs/fine-tuning-guide.md).
- **See expected gains, costs and data requirements:** [docs/results.md](docs/results.md).
- **Build training data when none exists:** [docs/medical-asr-synthetic-data.md](docs/medical-asr-synthetic-data.md)
  and [`sdg/`](sdg/README.md).

## The workflow

| Step | Script | What it does |
|---|---|---|
| data | `prepare_benchmarks.py` | Hugging Face dataset → 16 kHz FLAC + NeMo manifests with a per-row `context` prompt (16 benchmarks, LibriSpeech, and medical/accented-English sets) |
| | `set_context.py` | write a prompt (or a `{field}` template) into each manifest row's `context` |
| | `shuffle_manifest.py` | shuffle a training manifest on disk (Lhotse's shuffle buffer can't mix a large label-sorted file) |
| | `subset_manifest.py` | draw a random or class-stratified subset (data-efficiency studies) |
| | `check_contamination.py` | drop training rows that equal or overlap a test reference; report vocabulary overlap |
| config | `make_ft_config.py` | derive the training YAML from the checkpoint's own `config.json` (freeze policy, LoRA, optimizer, batching) |
| | `epoch_batches.py` | count the real batches per epoch, so `limit_train_batches` stays below it and validation runs |
| train | `run_training.py` | coverage preflight (`check_checkpoint_coverage.py`), 20-step smoke test, full run with a validation watchdog, checkpoint pruning, GPU lock |
| export | `export_checkpoint.py` | convert to HF format, check the LoRA adapters are non-zero, merge LoRA (`merge_lora_checkpoint.py`) → a directory vLLM can serve |
| | `average_checkpoints.py` | uniform weight average of merged checkpoints |
| evaluate | `vllm_task_eval.py` | vLLM decoding with per-row prompts, speculative decoding where the checkpoint supports it, label log-odds, output-length cap |
| | `task_metrics.py` | WER, MER, BLEU/chrF, SQuAD-v2 F1, label accuracy, Pearson r, EER, multi-label F1; re-scores saved hypotheses offline |
| | `prompt_search.py` | automatic prompt search on a dev set (an LLM proposes prompts from the model's errors) |
| | `vllm_label_scores.py`, `label_probe.py` | exact per-label log-probabilities and a linear probe on them |
| | `make_tta.py`, `rover_combine.py` | speed-perturbed test-time augmentation and ROVER hypothesis voting |

## Environments

- **Training:** runs in the NeMo environment.
- **Evaluation:** `vllm_task_eval.py`, `vllm_label_scores.py` and the `sdg/` text-generation scripts need a separate
  environment with vLLM 0.28.0 and this repository installed. vLLM pins its own PyTorch, so keep the two
  environments apart.
- **Outputs:** everything is written under `$SALM_FT_WORK`.

## A minimal run

```bash
FT=examples/speechlm2/finetuning
python $FT/prepare_benchmarks.py slurp
python $FT/make_ft_config.py --checkpoint $SALM_CHECKPOINT --prompt manifest --out slu.yaml --exp-dir exp/slu \
    --train-manifest $SALM_FT_WORK/manifests/slurp/train_ft.json --val-manifest $SALM_FT_WORK/manifests/slurp/devel_ft.json \
    --train-encoder --train-proj --lr 1e-4 --max-steps 500 --batch-tokens 96000 --limit-train-batches 125
python $FT/run_training.py --config slu.yaml --preflight --smoke --keep-steps 250 500
python $FT/export_checkpoint.py --exp-dir exp/slu --step 500 --base-checkpoint $SALM_CHECKPOINT --out export/slu_s500 --delete-ckpt
python $FT/vllm_task_eval.py --model export/slu_s500 --manifest $SALM_FT_WORK/manifests/slurp/test_ft.json \
    --out slu_test.jsonl --task cls --max-tokens 16 --labels $SALM_FT_WORK/manifests/slurp/labels.txt
```

Every script documents its options with `--help`.
