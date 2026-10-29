# Normalized aligned-text head for EasyMagpie / Nemotron-TTS

This optional experiment learns a Glow-TTS-style alignment teacher and a separate
autoregressive BPE output channel. It preserves the existing raw/IPA conditioning
channel and prefers `normalized_text` for the teacher and the new head, falling
back to the original `text` when normalization metadata is unavailable.

Set `model.aligned_text_use_normalized_text=false` to select original raw `text`
for both the teacher and all text-head targets, including warmup and aligned
targets. This option defaults to `true` and leaves existing experiments unchanged.
With `false`, unusable raw text skips the auxiliary task instead of falling back
to normalized text. The normalization-availability metric still counts metadata
availability, independently of which source is selected. Punctuation cleanup
remains separate: set `model.aligned_text_strip_punctuation=false` to retain the
raw punctuation too. Neither option changes the decoder's conditioning stream.

```bash
++model.use_aligned_text_head=true \
++model.start_aligner_text_head_steps=10000
```

The existing aligner recipe can use these two overrides unchanged. Defaults are
declared in `conf/magpietts/easy_magpietts_lhotse_multiturn.yaml`. Disabling the
feature creates no additional modules and preserves the original training path.

To initialize only the teacher from an existing checkpoint, set the top-level
`pretrained_model_text_speech_aligner=/path/to/checkpoint.nemo`. It loads only
`text_aligner` after full-model pretrained initialization; the repeated text head
and shared embeddings keep their current weights. Configure
`model.aligned_text_aligner` to match the source teacher exactly. This optional
parameter defaults to `null` and does not alter the curriculum switch. See the
[attention alignment initialization notes](README_easy_magpietts_attention_alignment.md)
for supported formats, compatibility requirements and resume behavior.

## Training

By default, the loader selects `normalized_text` when it is a nonempty string, otherwise
the original raw `text`. It retokenizes this selected transcript separately
from the conditioning channel, using exactly the same transcript for the aligner,
right-padded warmup labels, and repeated aligned-text labels.
It removes nonspoken boundary punctuation by default, retaining internal
apostrophes/hyphens, numeric punctuation and symbols. It does not perform text
normalization itself. Missing, null, nonstring, empty, or whitespace-only
normalization metadata falls back to original text. A provided nonempty normalized
transcript remains authoritative, even if punctuation cleanup makes it empty.
Unusable selected text skips auxiliary supervision; IPA and text-conditioned
decoder states are never substituted. `aligner_missing_normalized_text` counts
unavailable normalization metadata even when raw-text fallback is successfully
supervised. It does not count skipped turns. Complex normalization cases should
provide `normalized_text`. The ordinary TTS/phoneme losses still apply.

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

The default `text_encoder_type=relative_transformer` prior projects those CAS
features to width 128, then applies a small `RelativePositionTransformer`: two
layers, two heads, a 512-wide convolutional feed-forward network (kernel three,
ReLU), dropout 0.1, and residual connections with post-layer normalization. It
uses the [VITS relative-attention design](https://github.com/jaywalnut310/vits/blob/main/attentions.py):
learned relative key and value embeddings, shared across heads, for offsets
-4 through +4. This window controls relative embeddings; content attention is
bidirectional across the entire turn. No absolute positional embedding is added.
The final projection predicts Gaussian latent means, with variance fixed to one.
There is no duplicate vocabulary embedding or duration predictor. The encoder
receives only the selected transcript's features; decoder states do not enter it.

`modules/relative_position_transformer.py` adapts the MIT-licensed VITS attention
implementation and reuses NeMo's masked convolution layers. It does not require
the Coqui TTS package. The same attention/FFN structure is used at a smaller
capacity than VITS's usual six-layer, width-192 encoder. Masks are applied between
convolutions and after residual normalization, including fully invalid turn rows.

Optional speaker conditioning is disabled by default. Enable it with:

```bash
++model.aligned_text_aligner.use_speaker_conditioning=true
```

The frozen codec quantizer decodes the original reference audio codes into
continuous features. A small learnable encoder with two masked residual
convolution layers (width 64) applies channelwise attentive statistics pooling:
learned attention weights produce a weighted mean and standard deviation, then
a projection yields a normalized 64-dimensional vector. This follows the ASP
construction in `audio_codec_modules.ResNetSpeakerEncoder`, using masked NeMo
convolutions and no waveform/mel frontend, large ResNet, batch normalization, or
external speaker model. `speaker_hidden_channels`, `speaker_embedding_dim`, and
`speaker_num_layers` control its size.

The reference vector conditions the gated affine-coupling networks of the ST-TTS
flow; the latent means depend only on the selected transcript. The acoustic NLL
trains the ASP encoder and conditioning projections. Only codec observations are
detached. References are encoded once per conversation and gathered using each
packed turn's original sample index, so sorting and microbatching preserve the
association. The loader currently supplies one output-speaker reference per
conversation, shared by its output turns; it does not provide separate reference
clips for different output-speaker identities within that conversation.

This path uses reference codec IDs before decoder-context vocabulary conversion,
special tokens, CFG dropout, and context-embedding shuffling. If loader-level
reference shuffling replaces a reference with another sample's voice, the batch
marks that reference unavailable for the aligner. Missing or empty references
produce zero conditioning with finite gradients. The aligner never substitutes
target audio when a reference is missing; the dataset's existing reference
selection may itself choose an output turn when no separate clip is supplied.
The pooled vector is learned for alignment without a speaker-classification
objective, so speaker-only disentanglement is not guaranteed.

With the option disabled, no speaker parameters are created and old checkpoint
layouts are preserved. Speaker conditioning requires `flow_type=sttts`. Enabling
new modules from an older TTS checkpoint uses partial-pretrained initialization;
do not resume an incompatible optimizer state. The aligner and ASP encoder run
only during training and teacher-forced validation, not autoregressive inference.

The aligner trains from step zero. Before `start_aligner_text_head_steps`, the new
head predicts selected-transcript tokens followed by right padding within each turn. At
the threshold it predicts BPE tokens repeated according to the current MAS path.
The threshold uses Lightning's global optimizer step, including checkpoint resume.
The frozen codec and shared embeddings do not receive the aligner's gradients.

When `aligned_text_feedback=true` (default), the new input channel contains BOS
at each supervised turn start, then the previous output label. Its delay matches
the audio channel. User regions contribute zero. The source-text channel remains
available. Audio, phoneme, aligned-text and acoustic-likelihood losses train
together; text-only batches supply zero gradients for the new parameters so
mixed distributed batches remain valid.

The default `flow_type=sttts` teacher follows the flow capacity in
[ST-TTS section 3.1.4](https://www.isca-archive.org/interspeech_2021/kim21n_interspeech.pdf#page=3):
six flow blocks, 128 hidden channels, and four residual convolution layers with
kernel size five in each affine coupling network. Each block uses ActNorm,
learned invertible channel mixing, and a gated tanh/sigmoid residual-and-skip
coupling stack, following the paper's cited Glow-TTS implementation. The final
coupling projection is zero-initialized, and no extra time squeezing is applied.
ActNorm starts as identity, without data-dependent initialization or distributed
initialization collectives. The reference coupling log scale is unconstrained;
`max_log_scale` can optionally bound it.

This follows the published flow capacity, not the entire ST-TTS model. The text
prior uses the small VITS-style relative-position Transformer described above.
Frozen quantized codec features with jitter replace ST-TTS's mel observations.
Consequently the same kernel covers more seconds at 12.5 FPS, and boundary accuracy still requires
evaluation. The Gaussian prior variance remains fixed to one.

The earlier compact flow is available as `flow_type=legacy` with its original
parameter names. Unversioned saved aligner dictionaries are interpreted as
legacy profiles (64 hidden channels/four flows by default); new runs use the
explicit `sttts` profile in the YAML. Resuming an old compact aligner keeps its
architecture. Saved configurations without `text_encoder_type` retain the earlier
CNN prior, including checkpoints that already use the ST-TTS flow. Explicitly
select `text_encoder_type=cnn` to keep that prior in a new experiment. New YAML
configurations explicitly select `relative_transformer` and save all its settings.
To start the new text encoder or reference flow from an older TTS checkpoint, use
the existing partial-pretrained initialization path with the new YAML, rather
than resuming an old optimizer state into a different architecture.

The factorized output head uses `aligned_text_head_dim=128`; the vocabulary projection
is evaluated in checkpointed chunks of `aligned_text_head_chunk_size=128` valid
frames. It never retains conversation-length vocabulary logits. This compact head
is independent of the old `disable_lm_text_head` setting. Feedback reuses CAS and
embeds each distinct token once per call.

Useful controls:

- `aligned_text_loss_weight` and `aligner_loss_weight`: default 1.0 each.
- `aligned_text_aligner`: versioned flow type, hidden width, flow/text-layer
  counts, `coupling_num_layers`, `coupling_kernel_size`, `flow_n_split`, feature
  jitter, and `mas_backend` (`auto`, `triton`, `cpu`).
- Text prior controls inside `aligned_text_aligner`: `text_encoder_type`,
  `text_num_heads`, `text_filter_channels` (default four times the hidden width),
  `text_kernel_size`, `text_dropout`, and `text_rel_attn_window_size`.
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
It defaults to the ST-TTS flow with the relative-position text encoder. Compare
the previous ST-TTS teacher using `--text-encoder-type cnn`, or the original
compact teacher using `--flow-type legacy --hidden-channels 64 --num-flows 4`
together with `--text-encoder-type cnn`. Reports separate text-encoder and flow
parameter counts, as well as their combined forward/backward cost.
Pass `--use-speaker-conditioning` to include ASP reference encoding and its
backward pass; `--speaker-reference-frames` defaults to 75 stacked codec frames.

## Inference and portability

The head predicts one selected-transcript BPE/PAD/EOS label per decoder audio step, and its
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

For another branch, cherry-pick the feature commit, the ST-TTS flow update,
the relative-position text-encoder update, the optional ASP conditioning update,
and the normalized-first/raw-text fallback update, or apply the exported patch series.
Reusable code lives in `modules/glow_tts_aligner.py`, `modules/aligned_text_head.py`,
`modules/relative_position_transformer.py`, `modules/codec_speaker_encoder.py`,
and `parts/utils/monotonic_alignment.py`.
The integration touches only the model and
inference base, the two target loaders, the multiturn YAML, tests and this guide.
The cluster recipe is not required by the modules and is not copied into the
repository. Older checkpoints initialize the added modules through the existing
partial-pretrained-loading path; enabled checkpoints save the teacher and head
for exact restoration.
