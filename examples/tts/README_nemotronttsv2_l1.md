# Nemotron TTS v2 attention regularization experiment

Branch `vklimkov/nemotronttsv2_l1` enables Q/K L1 by default, with optional
attention entropy regularization:

```text
training loss = data loss + attention_l1_weight * sum(abs(Q/K projection weights))
                          + optional 0.1 * mean(normalized attention entropy)
```

Both terms cover the main decoder and acoustic codes predictor, including
FlashAttention2. With the existing `model_type=v2` submission, L1 defaults to
`attention_l1_weight=0.1` and entropy defaults to `enable_attention_entropy=false`.
When entropy is disabled, no Q/K hooks are installed, no activations are retained
for entropy, and no entropy scores, reductions, backward graph or logs are computed.
AdamW weight decay is unchanged.
Validation and inference do not compute either penalty; checkpoint keys are
unchanged. `process_batch.loss` remains the data loss; `training_step` adds both
enabled penalties.

Hydra overrides (the `+` adds fields absent from the base YAML):

```text
+model.attention_l1_weight=0.0005       # smaller L1 coefficient
+model.enable_attention_entropy=true  # opt into entropy; omitted means disabled
```

Set `attention_l1_weight=0` to skip the L1 traversal/reduction as well. There is
no delayed start or ramp schedule; enabled penalties apply from the first step.

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
and their sum as `train/loss`. Only when entropy is enabled, they additionally
expose unweighted normalized `train/attention_entropy` and weighted
`train/attention_entropy_loss`, which is included in the total loss.

L1 can shrink attention-score magnitudes while entropy favors concentrated
probabilities, so the objectives may compete. The L1 sum scales with parameter
count; normalized entropy contributes at most approximately 0.1. Equal
coefficients do not imply comparable contributions. At fixed model weights,
changing L1 weight from 0.1 to 0.005 divides the logged L1 contribution by 20;
0.002 divides it by 50 and 0.0005 divides it by 200. These ratios do not predict
how the weights will evolve, nor the contribution at a fresh initialization.
Inspect the separate logs
and compare generated-speech stability and validation quality against a baseline.
