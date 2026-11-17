# Nemotron TTS v2 attention L1 experiment

Branch `vklimkov/nemotronttsv2_l1` adds a hardcoded attention weight penalty to
`NemotronTTSModel.training_step`:

```text
training loss = data loss + 0.1 * sum(abs(attention projection weights))
```

The sum covers trainable Q/K/V/output projection weights in every Nemotron-H
attention layer of the main decoder and acoustic codes predictor, including the
FlashAttention2 implementation. Shared weights are counted once. Biases,
normalization, embeddings, Mamba and MLP/MoE weights are excluded. The reduction
uses FP32 and is not normalized by parameter count.

The existing `model_type=v2` submission enables this behavior automatically on
this branch; no Hydra override is needed. AdamW weight decay is unchanged.
Validation reports the original data losses, without L1. Inference and checkpoint
keys are unchanged.

Training logs expose `train/data_loss`, the weighted contribution
`train/attention_l1_loss`, and their sum as `train/loss`. Compare generation
stability and validation quality against the baseline; the unnormalized penalty
scales with the number of trainable attention weights.
