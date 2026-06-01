# Stage 4: Training, Checkpoint Averaging, And Evaluation

## Optimizer And Trainer

Use `trainer.max_steps`, not `trainer.max_epochs`. Use a cosine LR schedule with the same `max_steps` as the trainer.
Choose the initial LR by data size and risk:

- For large or mixed fine-tuning runs, start with `model.optim.lr=1e-4`, then tune.
- For small domain adaptation, especially below roughly 20 hours of high-value target-domain audio, start around
  `3e-5` and watch early validation closely.
- For later refinement phases or unstable/diverging runs, try `1e-5` or lower.

Set warmup to 1-2% of `trainer.max_steps`.

```bash
+init_from_pretrained_model=<hf-or-ngc-name> \
trainer.max_steps=50000 \
+trainer.limit_train_batches=1000 \
trainer.val_check_interval=1000 \
model.optim.lr=1e-4 \
model.optim.sched.name=CosineAnnealing \
++model.optim.sched.max_steps=50000 \
model.optim.sched.warmup_steps=500 \
model.optim.sched.min_lr=5e-6
```

`examples/asr/speech_to_text_finetune.py` supports `init_from_pretrained_model`, but some fine-tune YAMLs do not
declare that key. Use `+init_from_pretrained_model=...` when Hydra says the key is not in struct. Use `+` or `++` for
other trainer/model keys that are valid for the script but absent from the selected YAML; do not remove the plus just
because a similar key exists in another config.

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

## Multi-GPU Launch Troubleshooting

For routine single-node runs, start with the normal Lightning launch (`trainer.devices=<num_gpus>`). If a container run
hangs immediately after NCCL registration or one rank restores the model on the wrong GPU before DDP takes ownership,
use `torchrun --nproc_per_node=<num_gpus>` with the normal script entry point. Keep `trainer.devices=<num_gpus>` and
set `trainer.use_distributed_sampler=false` for Lhotse.

Container `torchrun` template:

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONPATH=/workspace/NeMo \
  -v /path/to/NeMo:/workspace/NeMo \
  -v /data:/data \
  nemo-speech:<tag> bash -lc '
    cd /workspace/NeMo &&
    torchrun --nproc_per_node=2 examples/asr/speech_to_text_finetune.py \
      +init_from_pretrained_model=nvidia/parakeet-tdt-0.6b-v3 \
      model.train_ds.manifest_filepath=/data/train.json \
      model.validation_ds.manifest_filepath=/data/val.json \
      ++model.train_ds.use_lhotse=true \
      ++model.validation_ds.use_lhotse=true \
      +trainer.use_distributed_sampler=false \
      trainer.devices=2 \
      trainer.max_steps=10000
  '
```

Keep `trainer.num_nodes` for real multi-node jobs only. For single-node container fine-tuning, avoid manual
one-visible-GPU-per-process launch patterns unless the user explicitly asks to debug that environment.

On systems with NCCL P2P or CUDA memory allocator issues, retry with conservative distributed environment settings
after confirming the job is stuck in communication setup:

```bash
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
```

Keep `trainer.sync_batchnorm` the same as it was set in the original model's config unless the user explicitly asks to
change it.

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

With PyTorch 2.6+, the deprecated averaging utility may fail because `torch.load` defaults to `weights_only=True`.
Only for checkpoints you trust, rerun with:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python scripts/checkpoint_averaging/checkpoint_averaging.py \
  /exp/asr-ft/checkpoints/best.nemo
```

Do not assume the latest `.nemo` is the best validation checkpoint. `always_save_nemo=true` can overwrite the `.nemo`
artifact on later saves. Check the `.ckpt` filenames and validation logs, then evaluate the exported `.nemo`, the best
checkpoint-derived artifact, and any averaged model before deciding what to keep.

Decision rule: evaluate the final exported `.nemo`, the best individual validation checkpoint-derived artifact, and the
averaged artifact with the same standalone command and scoring contract. Keep the artifact with the best standalone
default WER. If averaging is worse, record it and drop the averaged artifact.

Always run the same standalone evaluation command for the baseline model, the final exported `.nemo`, the best
checkpoint-derived artifact, and any averaged model. In-training `val_wer` is useful for checkpoint selection, but it is
not the final number to report. Standalone WER is the fair comparison because it holds decoding options, text
processing, precision, and output scoring constant.

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

Do not use AMP for inference/evaluation. Use `compute_dtype=bfloat16` and `amp=false`. Report standalone
`speech_to_text_eval.py` results for every model variant being compared; do not report only trainer logs.

Use `evaluation-style-contract.md` for scoring. By default, report WER with capitalization and punctuation removed via
score-only evaluation of the saved prediction manifest. Report raw WER separately only when the user requests it or it
helps diagnose transcript-style mismatch.

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
  text_processing.do_lowercase=true \
  text_processing.rm_punctuation=true \
  use_cer=False
```

Use `examples/asr/transcribe_speech.py` for direct offline transcription and streaming or chunked inference scripts for
streaming models.
