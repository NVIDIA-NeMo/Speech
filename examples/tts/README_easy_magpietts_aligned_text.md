# Normalized aligned-text head for EasyMagpie / Nemotron-TTS

This optional experiment learns a Glow-TTS-style alignment teacher and a separate
autoregressive BPE output channel. It preserves the existing raw/IPA conditioning
channel and uses only `normalized_text` for the teacher and the new head.

```bash
++model.use_aligned_text_head=true \
++model.start_aligner_text_head_steps=10000
```

The existing aligner recipe can use these two overrides unchanged. Defaults are
declared in `conf/magpietts/easy_magpietts_lhotse_multiturn.yaml`. Disabling the
feature creates no additional modules and preserves the original training path.

## Training

The loader retokenizes normalized transcripts separately from the source text.
It removes nonspoken boundary punctuation by default, retaining internal
apostrophes/hyphens, numeric punctuation and symbols. It does not perform text
normalization itself. A missing, empty or unusable `normalized_text` skips the
auxiliary supervision and increments `aligner_missing_normalized_text`; raw text
and IPA are never substituted. The ordinary TTS/phoneme losses still apply.

Each complete output turn is packed independently using the existing supervision
start/end boundaries. User turns, padding, cropped turns,
overlaps and deliberately mismatched transcripts do not teach alignment. The
teacher receives pure content BPE tokens and no BOS/EOS positions. MAS requires
at least one acoustic frame per token; turns with fewer frames are skipped.

The acoustic observation is reconstructed from original codec IDs using the
frozen codec's quantizer. Frame stacking concatenates feature vectors, retaining
the decoder's frame rate without a learned compression layer. Small uniform
jitter gives quantized features a continuous observation model. Existing CAS/text
embeddings are detached before entering the tiny text prior. The invertible flow
is trained by acoustic negative log likelihood, including the Jacobian; there is
no posterior encoder or VITS KL objective.

The aligner trains from step zero. Before `start_aligner_text_head_steps`, the new
head predicts normalized tokens followed by right padding within each turn. At
the threshold it predicts BPE tokens repeated according to the current MAS path.
The threshold uses Lightning's global optimizer step, including checkpoint resume.
The frozen codec and shared embeddings do not receive the aligner's gradients.

When `aligned_text_feedback=true` (default), the new input channel contains BOS
at each supervised turn start, then the previous output label. Its delay matches
the audio channel. User regions contribute zero. The source-text channel remains
available. Audio, phoneme, aligned-text and acoustic-likelihood losses train
together; text-only batches supply zero gradients for the new parameters so
mixed distributed batches remain valid.

The default teacher has 64 hidden channels, two text-convolution layers and four
flow blocks (about 210k parameters with 80 acoustic/1536 text dimensions). The
factorized output head uses `aligned_text_head_dim=128`; the vocabulary projection
is evaluated in checkpointed chunks of `aligned_text_head_chunk_size=128` valid
frames. It never retains conversation-length vocabulary logits. This compact head
is independent of the old `disable_lm_text_head` setting. Feedback reuses CAS and
embeds each distinct token once per call.

Useful controls:

- `aligned_text_loss_weight` and `aligner_loss_weight`: default 1.0 each.
- `aligned_text_aligner`: hidden width, flow/text-layer counts, feature jitter,
  and `mas_backend` (`auto`, `triton`, `cpu`).
- `aligned_text_max_turns_per_batch=32` and
  `aligned_text_max_frames_per_batch=8192`: padding-aware teacher batch limits.
  Oversized individual turns raise explicitly; increase the limit rather than
  truncating speech or transcripts.
- `aligned_text_feedback_scale=1.0`: feedback embedding scale.
- `aligned_text_strip_punctuation=true`: optional target punctuation cleanup.

Monitor `aligned_text_loss`, `aligner_nll`, `aligner_valid_turn_fraction`,
`aligner_one_frame_token_fraction`, `aligner_missing_normalized_text` and
`aligned_text_curriculum_active` under the training/validation metric prefix.
Inspect valid-turn coverage before interpreting the losses.

## MAS and profiling

`parts/utils/monotonic_alignment.py` keeps float32 dynamic programming and compact
frame-to-position indices on the GPU. Triton stores one-byte traceback decisions;
it does not copy the score matrix to CPU or create a dense float hard alignment.
Unsupported CUDA shapes and CPU runs reuse NeMo's existing Numba MAS. The default
GPU path supports padded text widths up to 4096; explicit `triton` rejects larger
widths, while `auto` falls back to CPU. Invalid paths return -1 indices.

Only the selected-path likelihood retains gradients. Dense search scores are
computed without autograd, then selected Gaussian terms are recomputed, avoiding
retaining a full score matrix for backward across turn packs.

```bash
python scripts/tts/profile_glow_tts_aligner.py \
  --device cuda --cpu-threads 8 --output /absolute/path/profile.json
```

The profiler checks CPU/GPU parity and measures MAS, full teacher
forward/backward, memory, and separate-turn versus whole-conversation lattices.
Compilation is warmed up; CPU transfers are included. Synthetic teacher timings
exclude codec decoding, CAS, the text head and the TTS backbone.

## Inference and portability

The head predicts one normalized BPE/PAD/EOS label per decoder audio step, and its
previous prediction is fed back alongside the previous audio token. No aligner
or MAS runs at inference and no extra autoregressive pass is added. Text EOS does
not terminate audio generation. Raw streaming text and CFG remain supported.

Streaming state retains the labels. `streaming_finalize` returns optional
`aligned_text_tokens` / `aligned_text_tokens_lens`; `infer_batch` returns
`predicted_aligned_text_tokens` / `predicted_aligned_text_tokens_lens`. These are
trimmed to actual generated audio, excluding user-prefill frames and codec
minimum-length padding. One label spans
`codec_model_samples_per_frame * frame_stacking_factor / sample_rate` seconds
(80 ms for 25 FPS codec with stacking factor 2).

These predictions estimate progress; they are not an independent ASR check of
generated sound. Word boundaries require validation. Adjacent equal BPE IDs also
cannot reveal their intervening position boundary without an additional advance
signal. No hard stay/advance decoding or attention constraint is imposed here.

For another branch, cherry-pick the self-contained feature commit. Reusable code
lives in `modules/glow_tts_aligner.py`, `modules/aligned_text_head.py` and
`parts/utils/monotonic_alignment.py`. The integration touches only the model and
inference base, the two target loaders, the multiturn YAML, tests and this guide.
The cluster recipe is not required by the modules and is not copied into the
repository. Older checkpoints initialize the added modules through the existing
partial-pretrained-loading path; enabled checkpoints save the teacher and head
for exact restoration.
