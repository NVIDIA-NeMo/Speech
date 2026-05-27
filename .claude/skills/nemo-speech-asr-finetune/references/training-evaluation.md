# Stage 4: Training, Checkpoint Averaging, And Evaluation

## Optimizer And Trainer

Use `trainer.max_steps`, not `trainer.max_epochs`. Use a cosine LR schedule with the same `max_steps` as the trainer.
Start with `model.optim.lr=1e-4`, then tune. Set warmup to 1-2% of `trainer.max_steps`.

```bash
trainer.max_steps=50000 \
+trainer.limit_train_batches=1000 \
trainer.val_check_interval=1000 \
model.optim.lr=1e-4 \
model.optim.sched.name=CosineAnnealing \
model.optim.sched.max_steps=50000 \
model.optim.sched.warmup_steps=500 \
model.optim.sched.min_lr=5e-6
```

Prefer `trainer.precision=bf16-true` for memory savings and larger batch sizes, especially for datasets over 1k
hours. If it diverges or has stability issues, fall back to a more stable precision mode. Prefer `bf16-true` over
`bf16-mixed` as the first bfloat16 option.

Monitor `val_wer` for all validation runs and checkpoint selection:

```bash
exp_manager.checkpoint_callback_params.monitor=val_wer \
exp_manager.checkpoint_callback_params.mode=min \
exp_manager.checkpoint_callback_params.save_top_k=5 \
exp_manager.checkpoint_callback_params.always_save_nemo=true
```

## Checkpoint Averaging

At the end of a run, optionally average the N best checkpoints saved by `save_top_k=N`. Then evaluate the averaged
model and keep it only if it beats the best individual checkpoint on validation/test.

The simple `.nemo` averaging utility averages all non-`-last.ckpt` checkpoints in the same folder as the `.nemo` file,
so control N with `save_top_k`:

```bash
python scripts/checkpoint_averaging/checkpoint_averaging.py \
  /exp/asr-ft/checkpoints/best.nemo
```

This produces `*-averaged.nemo`. If model class loading fails, use the script's `--class_path` or `--import_fname_list`
options. Some checkpoint averaging scripts are marked deprecated in the repo; verify they still work in the current
checkout before relying on them.

Evaluate both:

```bash
python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/val.json \
  output_filename=/exp/asr-ft/val_best.json \
  batch_size=32 \
  amp=false \
  compute_dtype=bfloat16

python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best-averaged.nemo \
  dataset_manifest=/data/val.json \
  output_filename=/exp/asr-ft/val_avg.json \
  batch_size=32 \
  amp=false \
  compute_dtype=bfloat16
```

## Evaluation

Do not use AMP for inference/evaluation. Use `compute_dtype=bfloat16` and `amp=false`.

```bash
python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/test.json \
  output_filename=/exp/asr-ft/test_predictions.json \
  batch_size=32 \
  amp=false \
  compute_dtype=bfloat16 \
  matmul_precision=high \
  use_cer=False
```

For RNNT/TDT evaluation, enable CUDA graphs when supported:

```bash
python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/test.json \
  output_filename=/exp/asr-ft/test_predictions.json \
  batch_size=32 \
  amp=false \
  compute_dtype=bfloat16 \
  matmul_precision=high \
  rnnt_decoding.strategy=greedy_batch \
  rnnt_decoding.greedy.use_cuda_graph_decoder=true
```

For a hybrid model, compare decoder choices:

```bash
python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/test.json \
  decoder_type=ctc \
  output_filename=/exp/asr-ft/test_predictions_ctc.json \
  amp=false \
  compute_dtype=bfloat16

python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/test.json \
  decoder_type=rnnt \
  output_filename=/exp/asr-ft/test_predictions_rnnt.json \
  amp=false \
  compute_dtype=bfloat16 \
  rnnt_decoding.strategy=greedy_batch \
  rnnt_decoding.greedy.use_cuda_graph_decoder=true
```

If predictions already exist:

```bash
python examples/asr/speech_to_text_eval.py \
  dataset_manifest=/exp/asr-ft/test_predictions.json \
  only_score_manifest=True \
  use_cer=False
```

Use `examples/asr/transcribe_speech.py` for direct transcription and streaming or chunked inference scripts for
streaming models.
