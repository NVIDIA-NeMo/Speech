# Nemotron TTS v2 attention regularization experiment

Branch `vklimkov/nemotronttsv2_l1` combines two hardcoded training penalties:

```text
training loss = data loss + 0.1 * sum(abs(Q/K projection weights))
                          + 0.1 * mean(normalized attention entropy)
```

Both terms cover the main decoder and acoustic codes predictor, including
FlashAttention2. The existing `model_type=v2` submission enables them automatically
on this branch; no Hydra override is needed. AdamW weight decay is unchanged.
Validation and inference do not compute either penalty; checkpoint keys are
unchanged. `process_batch.loss` remains the data loss; `training_step` adds both
penalties.

## Q/K weight L1

L1 covers trainable query/key projection weights. Shared weights are counted once.
Value/output projections, biases, normalization, embeddings, Mamba, MLP/MoE and
frozen weights are excluded. The sum accumulates in FP32 and is not normalized by
parameter count. Smaller projection entries are a weight-regularization bias;
robustness to autoregressive errors must be measured empirically.

## Attention entropy

For one query and head, attention probabilities are
`p = softmax(Q @ K.T / sqrt(head_dim))`, before attention dropout. The entropy is
`H(p) = -sum(p * log(p))`, using natural logarithms and treating `0 * log(0)` as zero.
The row penalty is `H(p) / log(N)`, where `N` is the number of valid causal keys.
It is zero for one-hot attention and one for uniform attention. Adding a positive
coefficient encourages focused attention. It does not select the correct key or
impose a monotonic alignment.

Padding keys and queries, future keys, and rows with fewer than two available keys
are excluded. We average over eligible queries and heads within each attention
layer, then average equally over attention-layer invocations. Every eligible
query and key is used; there is no sampling. The helper assumes full-sequence,
causal attention with right padding, as used by this v2 training path.

Projected Q/K tensors are captured with temporary hooks, leaving the fused
SDPA/FlashAttention forward untouched. Entropy is computed after the relevant
forward call, outside decoder checkpoint regions. FP32 score/softmax calculations
use 32-query chunks and activation recomputation during backward, avoiding saved
full quadratic attention matrices. This adds quadratic compute; throughput and
peak memory on the ORD training job have not been benchmarked. Hooks are removed
on exit, including exceptions, and no hooks are installed for validation.

For four available keys:

| Attention probabilities | Entropy (nats) | Normalized entropy | Weighted contribution |
| --- | ---: | ---: | ---: |
| `[0.25, 0.25, 0.25, 0.25]` | 1.3863 | 1.0000 | 0.1000 |
| `[0.50, 0.50, 0, 0]` | 0.6931 | 0.5000 | 0.0500 |
| `[0.70, 0.10, 0.10, 0.10]` | 0.9404 | 0.6784 | 0.0678 |
| `[0.97, 0.01, 0.01, 0.01]` | 0.1677 | 0.1210 | 0.0121 |
| `[1, 0, 0, 0]` | 0.0000 | 0.0000 | 0.0000 |

## Logging and interpretation

Training logs expose `train/data_loss`, weighted `train/attention_l1_loss`,
unweighted normalized `train/attention_entropy`, weighted
`train/attention_entropy_loss`, and their sum as `train/loss`.

L1 can shrink attention-score magnitudes while entropy favors concentrated
probabilities, so the objectives may compete. The L1 sum scales with parameter
count; normalized entropy contributes at most approximately 0.1. Equal
coefficients do not imply comparable contributions. Inspect the separate logs
and compare generated-speech stability and validation quality against a baseline.
