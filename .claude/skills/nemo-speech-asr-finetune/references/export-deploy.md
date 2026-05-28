# Stage 6: Export And Deployment Handoff

Use this stage only after standalone evaluation identifies the checkpoint to keep. Export artifacts are downstream of
model selection; do not export the latest checkpoint just because it was saved last.

## Before Export

Record:

- Final `.nemo` path and source checkpoint.
- Baseline, final, and optional averaged standalone WER/CER.
- Tokenizer path/type, sample rate, decoding strategy, and any prompts or language/task tags.
- Training config overrides that affect inference, such as tokenizer replacement or hybrid decoder choice.
- License/data restrictions and whether the artifact may be shared externally.

Keep the original `.nemo`, evaluation manifest, prediction manifest, and exact evaluation command with the export.

## NeMo Artifact

For most handoffs, the `.nemo` file is the primary artifact. Validate it after training or checkpoint averaging:

```bash
python examples/asr/speech_to_text_eval.py \
  model_path=/exp/asr-ft/checkpoints/best.nemo \
  dataset_manifest=/data/test.json \
  output_filename=/exp/asr-ft/test_best.json \
  batch_size=32 \
  amp=false \
  compute_dtype=bfloat16 \
  matmul_precision=high
```

If using a hybrid model, evaluate the chosen export head explicitly with `decoder_type=ctc` or `decoder_type=rnnt`.

## ONNX Or TorchScript

Check current repo examples before giving architecture-specific export commands. RNNT/TDT export usually creates
separate encoder and decoder/joint artifacts, while CTC export is simpler.

For RNNT/TDT parity checks, start from the repo helper:

```bash
python examples/asr/export/transducer/infer_transducer_onnx.py \
  --nemo_model=/exp/asr-ft/checkpoints/best.nemo \
  --export \
  --dataset_manifest=/data/test.json \
  --batch_size=32 \
  --max_symbold_per_step=10
```

For direct export experiments, use the model's `export()` API in the same container and then verify predictions against
the `.nemo` model on a small manifest before handing the artifact to serving:

```python
from nemo.collections.asr.models import ASRModel

model = ASRModel.restore_from("/exp/asr-ft/checkpoints/best.nemo")
model.export("/exp/asr-ft/export/model.onnx")
```

If export fails, check whether the architecture supports the requested export format and whether the model needs an
export-specific config such as cache-aware streaming settings.

## Riva

Riva packaging is outside the core NeMo training scripts and can change independently. Treat it as a separate validation
target:

- Confirm the selected architecture and tokenizer are supported by the current Riva toolchain.
- Package from the evaluated `.nemo` artifact, not an unverified checkpoint.
- Preserve decoding settings, sample rate, vocabulary/tokenizer files, and language metadata.
- Run a post-package smoke test on the same small manifest used for export parity checks.
- Run the full standalone evaluation path against the deployed service before reporting production WER.

If the model is a hybrid RNNT/CTC checkpoint, decide which head is being served and evaluate that same head before
packaging.

## Hugging Face Or Model Registry

For a Hub or internal registry handoff, include enough metadata for a user to reproduce evaluation:

- `.nemo` artifact or supported converted format.
- Model card with base checkpoint, fine-tuning data summary, intended language/domain, limitations, and license.
- Standalone evaluation command and WER/CER table.
- Tokenizer details and whether the tokenizer was replaced or aggregated.
- Example inference command with `amp=false compute_dtype=bfloat16` for evaluation.

Do not publish private-domain data examples, secrets, or manifests with absolute internal paths.

## Deployment Checklist

- Artifact was selected by standalone eval.
- Exported artifact matches `.nemo` predictions on a smoke-test manifest.
- Domain WER and general guardrail WER are recorded.
- Runtime decoding settings match evaluation settings.
- Latency, memory, and batch-size targets were tested on the serving hardware.
- Rollback artifact is available.
