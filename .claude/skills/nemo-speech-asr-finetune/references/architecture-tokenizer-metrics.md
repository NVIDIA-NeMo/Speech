# Stage 3: Architecture, Tokenizer, And AED Metrics

## Architecture Detection

Inspect the model config before choosing scripts and overrides:

```python
from nemo.collections.asr.models import ASRModel

cfg = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", return_config=True)
print(cfg.target)
print(cfg.get("decoder", None))
print(cfg.get("joint", None))
print(cfg.get("loss", None))
print(cfg.get("decoding", None))
```

Classify:

- CTC: `EncDecCTC*`, `ConvASRDecoder`, no RNNT-style `joint`.
- RNNT: `EncDecRNNT*` with `decoder` and `joint`.
- TDT: RNNT-family config with `loss.loss_name: tdt`, `decoding.model_type: tdt`, durations, or extra duration
  outputs.
- Hybrid RNNT/CTC or TDT/CTC: `EncDecHybridRNNTCTC*`, `aux_ctc`, `ctc_decoder`.
- AED/Canary: `EncDecMultiTaskModel`, Transformer decoder, `prompt_format`.

Use `examples/asr/speech_to_text_finetune.py` for compatible-architecture fine-tuning. For architecture-specific
recipes:

- CTC: `examples/asr/asr_ctc/speech_to_text_ctc_bpe.py`
- RNNT: `examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py`
- Hybrid RNNT/CTC or TDT/CTC: `examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py`
- AED/Canary: `examples/asr/speech_multitask/speech_to_text_aed.py`

Reference configs to inspect before writing overrides:

- CTC: `examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml`
- RNNT: `examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml`
- TDT: `examples/asr/conf/conformer/tdt/conformer_tdt_bpe.yaml`
- Hybrid TDT/CTC: `examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe.yaml`
- AED/Canary: `examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`

## Tokenizer Decisions

Keep the pretrained tokenizer when language/script, casing, punctuation, and symbols match. Replace or extend it when
the target language/script is new, important symbols are missing, normalization changes substantially, the run is
multilingual/code-switching, or Canary/AED prompt/special tokens change.

Train from training text only:

```bash
python scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest=/data/train.json \
  --data_root=/data/tokenizers/my_tokenizer \
  --vocab_size=1024 \
  --tokenizer=spe \
  --spe_type=unigram \
  --log
```

Replace in generic fine-tuning:

```bash
model.tokenizer.update_tokenizer=true \
model.tokenizer.dir=/data/tokenizers/my_tokenizer/tokenizer_spe_unigram_v1024 \
model.tokenizer.type=bpe
```

Changing tokenizer size usually reinitializes decoder-side parameters such as CTC projection or RNNT/TDT
decoder/joint pieces. Use conservative LR and validate early.

Aggregate tokenizer:

```yaml
tokenizer:
  type: agg
  langs:
    en:
      dir: /data/tokenizers/en/tokenizer_spe_unigram_v1024
      type: bpe
    es:
      dir: /data/tokenizers/es/tokenizer_spe_unigram_v1024
      type: bpe
```

Standard aggregate ASR configs expect a manifest language field such as `lang`. AED/Canary configs use their own
prompt and language fields; follow `examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`.

## Architecture-Specific Knobs

CTC:

- Main tokenizer-sensitive module is the decoder projection.
- Useful when alignments or non-autoregressive decoding matter.
- Long transcripts can violate CTC length constraints; subword tokenization helps reduce target length.

RNNT:

- Disable fused loss/WER for this skill: `model.joint.fuse_loss_wer=false`.
- Do not tune `fused_batch_size`; use Lhotse bucketing plus OOMptimizer-generated `bucket_batch_size`.
- `model.compute_eval_loss=false` is common when validation samples are long and WER is the main metric.
- Use CUDA graphs for inference/evaluation when supported.

TDT:

- Preserve `loss.loss_name=tdt`, duration settings, extra outputs, and `decoding.model_type=tdt`.
- Disable fused loss/WER and use Lhotse bucketing plus OOMptimizer.
- Use CUDA graphs for inference/evaluation when supported.

Hybrid:

- Check `model.aux_ctc.ctc_loss_weight`; reference configs often use `0.3`.
- Evaluate both decoder paths when relevant with `decoder_type=ctc` and `decoder_type=rnnt`.

AED/Canary:

- Use `examples/asr/speech_multitask/speech_to_text_aed.py`.
- Preserve `prompt_format` and expected manifest fields.
- Prefer 2D Lhotse buckets plus OOMptimizer.

## AED/Canary Multitask Metrics

`EncDecMultiTaskModel` reads `model.multitask_metrics_cfg` and constructs `MultiTaskMetric`
(`nemo/collections/asr/metrics/multitask.py`). Metric constraints are evaluated against each Lhotse cut's `custom`
dict, including manifest fields and `input_cfg.tags`.

Reference config:

```yaml
model:
  multitask_metrics_cfg:
    log_predictions: true
    metrics:
      wer:
        _target_: nemo.collections.asr.metrics.WER
        constraint: ".source_lang==.target_lang"
      bleu:
        _target_: nemo.collections.asr.metrics.BLEU
        constraint: ".source_lang!=.target_lang"
        bleu_tokenizer: 13a
        check_cuts_for_bleu_tokenizers: false
```

Use constraints to route ASR samples to WER and translation samples to BLEU. Add dataset/task/domain metadata through
manifest fields or `input_cfg.tags`, then reference it in constraints such as `.domain==target` or
`.task==asr and .source_lang==.target_lang`.

Current implementation supports only one instance of each metric class in a single `multitask_metrics_cfg`. For
multiple WER slices by language/domain, prefer separate validation manifests/dataloaders or extend metric aggregation
rather than defining duplicate WER metrics.

For AED validation data, set `use_lhotse: true`, `use_bucketing: false`, static `batch_size`, `text_field: "text"`,
and `lang_field: "target_lang"`.
