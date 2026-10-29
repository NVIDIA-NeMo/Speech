# Flow-guided self-attention for EasyMagpie / Nemotron-TTS

This opt-in experiment uses the existing small Glow-TTS/ST-TTS flow as a
training-time teacher for selected heads in the Nemotron-H decoder. It adapts
the attention-prior mechanism in [T5-TTS, section 3.3](https://www.isca-archive.org/interspeech_2024/neekhara24_interspeech.pdf),
replacing its attention CTC objective with cross-entropy against a softened,
detached acoustic alignment. It is not an exact reproduction of T5-TTS:
our model uses streaming self-attention, rather than text cross-attention, and
the prior follows the flow teacher rather than a length-based diagonal.

## Enable a separate experiment

```bash
++model.use_attention_alignment=true \
++model.use_aligned_text_head=false \
++model.aligned_text_feedback=false
```

The feature defaults to false. The repeated-text-head experiment remains
unchanged and is mutually exclusive with this mode. No repeated BPE prediction
head or additional text-feedback input is created for attention alignment.
The initial implementation requires `decoder_type=nemotron_h` and
`nemotron_h_config.rswa_window=null`. Ordinary cached synthesis remains
unchanged; teacher requests are rejected during cached decoding.

Start from the original good TTS checkpoint using partial pretrained
initialization. To reuse a trained aligner from another checkpoint, set the
top-level `pretrained_model_text_speech_aligner` parameter:

```bash
++pretrained_model=/path/to/base_tts.nemo \
++pretrained_model_text_speech_aligner=/path/to/trained_aligner.nemo \
++model.use_attention_alignment=true \
++model.use_aligned_text_head=false \
++model.aligned_text_feedback=false
```

The aligner-only load runs after `pretrained_model` and NeMo's `init_from_*`
initialization. It restores only `text_aligner`: the flow, small text encoder,
and optional speaker encoder. Decoder, shared text embeddings, repeated text
head, optimizer and source configuration are not loaded by this parameter.
Full `.nemo` archives, flat PyTorch state dictionaries, Lightning `state_dict`
checkpoints and bare aligner state dictionaries are supported.
PyTorch/Lightning files must be trusted; their metadata uses ordinary pickle
deserialization. `.nemo` tensor weights use restricted loading. The receiving
`model.aligned_text_aligner` configuration must match all teacher keys and tensor
shapes, including speaker conditioning; mismatches fail before weights are
copied. The source must use the same text tokenizer and acoustic representation;
these semantics cannot be verified from tensor shapes. A separately trained
teacher also depends on compatible shared text embeddings, which are retained
from the base TTS checkpoint.

The parameter defaults to `null` and never enables an alignment mode implicitly.
`use_attention_alignment` remains `false` by default. A pretrained teacher does
not change the attention schedule; set `model.attention_alignment.start_step=0`
explicitly if guidance should begin without the usual warmup.

Do not resume an optimizer from a checkpoint with an incompatible module layout.
Trainer checkpoint resume restores the complete training state after these
initializations, overriding them. Completed attention-alignment checkpoints
retain their complete configuration and teacher weights for restoration.

## What is supervised

The loader gathers the **actual conditioning BPE IDs and their original text
timeline positions** for each complete output turn. It does not retokenize a
normalized or punctuation-stripped string. BOS/EOS, PAD, user turns, cropped or
overlapping turns and deliberately replaced text are excluded. Turns whose
conditioning channel lacks its EOS or contains non-BPE controls are ineligible.
The existing MAS path requires at least one acoustic frame per BPE token.

The existing frozen codec quantizer reconstructs acoustic embeddings. The flow
and its small relative-position text encoder learn their acoustic NLL from
detached features. Optional codec-reference ASP conditioning uses the existing
`aligned_text_aligner` settings and remains disabled by default. Compact turn
packing, frame stacking and optimized MAS reuse the original implementation.

The teacher returns token **positions**, rather than token IDs, on each turn's
acoustic grid. The model restores them to the full conversation, then adds the
actual context prefix and text/audio delays. A frame is eligible for attention
guidance only if its aligned conditioning key is already causally available.
Soft target bands are clipped to visible text keys and normalized.

For selected attention heads, cross-entropy supervises their original attention
probabilities over all visible decoder keys. This also encourages attention
mass on text; a separately normalized text-only slice cannot conceal a head
that ignores text. Teacher indices and target bands are detached. This is
different from predicting repeated BPE IDs from the final decoder state.

The optional forward prior multiplies the selected-head probabilities by the
soft teacher band and normalizes the result before computing their value sum.
All other heads and unselected query positions keep ordinary attention. A
prior strength of zero leaves decoder outputs unchanged while retaining the
attention loss. No new attention projection weights are introduced.

Attention computation is chunked for the selected heads and checkpointed
during training. It does not retain a conversation-square probability matrix.
The existing FlashAttention path is preserved for ordinary attention.

## Schedule and losses

```yaml
attention_alignment:
  loss_weight: 0.1
  start_step: 10000
  ramp_steps: 1000
  prior_anneal_steps: 5000
  prior_strength: 1.0
  band_width: 1.0
  heads: [0]
  layers: middle
  look_back_tokens: 0
  look_ahead_tokens: 0
  query_chunk_size: 128
```

These are optimizer-global-step settings, including checkpoint resume:

- Before 10,000 steps, only the flow's auxiliary NLL trains; it does not change
  decoder inputs or send its gradients into the TTS backbone.
- From 10,000 to 11,000, both attention-loss weight and forward-prior strength
  ramp from zero to their configured values.
- From 11,000 to 16,000, the forward prior fades to zero. Attention supervision
  remains active after the prior disappears.

`layers` selects **attention-layer ordinals**, counting only attention blocks in
the hybrid stack. `middle` (default) selects ordinal `count // 2`: with three
attention layers, only the second receives the attention loss and forward prior.
The first and last use ordinary attention. Gradients can still flow through
them from downstream losses. For even counts, `middle` selects the later middle
layer. Use `[1]` to select the second attention layer explicitly, `[0, 2]` for
first and third, or `null` to recover guidance across all attention layers.
These indices are independent of absolute decoder block depths. Selected
ordinals and block depths are logged during model initialization.

`heads` selects head indices within each selected attention layer; `[0]` starts
with one head in the middle layer. `null` selects every head in the selected
layers. `band_width` is the Gaussian standard
deviation in BPE-position units. `query_chunk_size` bounds recomputation chunks.

To tolerate local text-normalization reordering, widen the soft target around
each MAS center, for example:

```bash
++model.attention_alignment.look_back_tokens=2 \
++model.attention_alignment.look_ahead_tokens=2
```

These counts are BPE token positions within the current turn. The loss target
and forward prior have equal unnormalized weight inside this tolerance interval,
then Gaussian falloff outside it, controlled by `band_width`. Attention may move
backward or forward inside the interval; no attention pointer is forced to advance
monotonically. The flow teacher's MAS path remains monotonic, so this relaxes local
attention guidance and does not solve arbitrary long-range reordering. Future
text positions are included only after their keys become causally available;
padding, other turns and unavailable future tokens remain excluded from targets.
Both counts default to zero, preserving the original Gaussian target. The global
feature flags `use_attention_alignment` and `use_aligned_text_head` both default
to `false`; these settings never enable a feature or an inference-time mask.

The attention teacher always consumes the exact conditioning BPE IDs. To use
raw conditioning text even when normalized metadata is available, set:

```bash
++model.train_ds.load_normalized_text_percent=0.0 \
++model.validation_ds.load_normalized_text_percent=0.0
```

The separate repeated-head experiment also supports
`model.aligned_text_use_normalized_text=false` for raw teacher/head transcripts.
That flag leaves conditioning unchanged and does not replace attention key
metadata. The two experimental modes remain mutually exclusive. Punctuation
cleanup in the repeated-head mode is controlled independently by
`model.aligned_text_strip_punctuation`; attention guidance always retains the
actual conditioning tokens.

The total training objective adds `aligner_loss_weight * aligner_nll` and the
scheduled `attention_alignment.loss_weight * attention_alignment_loss` to the
existing speech and phoneme losses. There is no CTC loss in this experiment.

W&B/Lightning metrics under `train/` and `val/` include:

- `aligner_nll`
- `attention_alignment_loss`
- `attention_alignment_loss_scale`
- `attention_alignment_prior_strength`
- `attention_alignment_valid_turn_fraction`
- `attention_alignment_causal_frame_fraction`
- `attention_alignment_supervised_turns`
- `attention_alignment_invalid_spans`, `attention_alignment_short_turns`, and
  `attention_alignment_empty_turns`

Coverage metrics must be inspected alongside losses. Unsupervised batches have
zero auxiliary contribution; zero loss alone is not evidence of a useful path.
The attention loss uses soft targets, so its minimum includes their entropy;
it is not directly comparable to the repeated-BPE head's cross-entropy.

### Validation alignment figures

With an alignment mode and a W&B logger enabled, validation logs paired figures
under `val/alignment/turn_0`, `val/alignment/turn_1`, and so on. Each image compares
the flow teacher's hard MAS path with the actual guided head's attention over
the same eval turn, using shared BPE-position and acoustic-time axes. The caption
identifies the attention-layer ordinal, absolute decoder block and head.

```yaml
alignment_visualization:
  enabled: true
  max_examples: 2
  max_audio_frames: 1024
  max_text_tokens: 256
```

The sample limit applies across the complete validation epoch on rank zero.
Sanity checks, other ranks, training batches and runs without W&B do not capture
attention. Both alignment features remain disabled by default; diagnostics do
not activate either feature. Set `alignment_visualization.enabled=false` to skip
these images in an alignment experiment.

The attention panel shows **raw probabilities before applying the training
prior**, normalized over all visible decoder keys. This reveals whether the
head actually attends to text rather than just displaying the imposed prior.
Warmup capture does not activate the alignment loss or prior. The first selected
guided layer and first selected head are plotted; the default is the middle
attention layer, head zero. In repeated-head-only mode, only MAS is available.

Long-turn attention is a clearly labeled leading excerpt, capped by the settings
above. Gray regions are uncaptured or have no valid alignment. Captured text keys
excluded by causal or padding masks have zero probability and appear dark.
Rendering uses a bounded raster on the full turn's axes; it never constructs or
logs a conversation-square attention matrix. These are teacher-forced validation
diagnostics, not alignments recovered from autoregressively generated speech.

## Inference and evaluation

The flow, teacher prior and supervision are not run during synthesis. By default,
the decoder uses its ordinary cached attention, with no additional text-feedback
channel or moving window. Training guidance and online inference guidance have
independent, disabled-by-default switches.

For an explicitly enabled online experiment, the controller adapts the moving
window in NeMo's [Magpie-TTS inference code](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/tts/models/magpietts.py).
This is distinct from the original T5-TTS paper's static training prior, which is
annealed away before inference.

```yaml
attention_alignment_inference:
  enabled: false
  layer: middle
  head: 0
  look_back_tokens: 1
  look_ahead_tokens: 5
  prior_epsilon: 0.1
```

Enable it with `model.attention_alignment_inference.enabled=true` on a model
whose selected head has learned useful alignment. The single `layer` is an
attention-layer ordinal or `middle`; `head` is the single head used both for
tracking and guidance. It uses the existing decoder weights and no flow,
teacher transcript or new prediction head. It is supported for native
Nemotron-H full-attention cached evaluation, with one decoder query per step.
It is not wired into external inference engines such as vLLM.

The controller tracks actual observed BPE key positions within the current turn.
It softly downweights text keys outside the backward/forward window by
`prior_epsilon`; the value must be positive, so this never hard-blocks a token.
Other heads, layers, CFG unconditional rows, non-BPE timeline positions and
reference context retain ordinary attention. A BPE-carrying KV position in this
decoder-only model can also contain audio/phoneme information; the prior weights
that combined position, not an isolated text component. Look-ahead includes only
text already
received by the stream. The progress anchor is nondecreasing, while attention
can read backward within the window and can still reach outside it with the
smaller weight. There is no forced advancement after a duration threshold:
raw text normalization can require many audio frames for a single source BPE.

The optional controller is available as
`state.attention_alignment_inference_state`; `last_attended_tokens` contains the
latest current-turn BPE-position estimates. New BOS or a reset turn counter
clears current-turn tracking. Multiturn callers can explicitly invoke
`model.reset_attention_alignment_turn(state, batch_indices=...)` together with
their generation-state reset; this preserves the KV cache. Text EOS alone does
not clear tracking while the remaining speech for that turn is generated.
User/profile prefill is excluded from response-text registration.

Teacher-forced validation can use the scheduled training prior while ordinary
autoregressive generation never uses that teacher prior. The optional moving
window is a separate inference heuristic. Compare teacher-forced and generated
metrics separately during ramp/fade, and evaluate WER with the online controller
both disabled and enabled after training. Its attended token positions are
alignment estimates, not verified word boundaries or evidence that a word was
actually pronounced.

Evaluate generated WER and the original phoneme/codebook losses against the
same source checkpoint, with matching data and synthesis settings. Low teacher
NLL or a monotonic attention plot alone cannot establish correct boundaries.
Raw-text normalization can reorder spoken content or contain silent symbols;
the current MAS still assigns every retained BPE at least one frame. The soft
band tolerates approximate boundaries but does not solve these limitations.

## Portability

The new reusable module is `nemo/collections/tts/modules/attention_alignment.py`.
Integration adds an optional request and auxiliary output to Nemotron-H,
independent metadata in the two loaders, model loss/schedule wiring, config
defaults and focused tests. Existing decoder/flow parameter names are retained.
Cherry-pick the feature commit onto a branch containing the original alignment
modules, or apply its exported patch after those prerequisites.
