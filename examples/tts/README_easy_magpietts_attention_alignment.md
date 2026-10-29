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
  query_chunk_size: 128
```

These are optimizer-global-step settings, including checkpoint resume:

- Before 10,000 steps, only the flow's auxiliary NLL trains; it does not change
  decoder inputs or send its gradients into the TTS backbone.
- From 10,000 to 11,000, both attention-loss weight and forward-prior strength
  ramp from zero to their configured values.
- From 11,000 to 16,000, the forward prior fades to zero. Attention supervision
  remains active after the prior disappears.

`heads` selects head indices within each attention layer; `[0]` starts with one
head per layer. `null` selects every head. `band_width` is the Gaussian standard
deviation in BPE-position units. `query_chunk_size` bounds recomputation chunks.

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

## Inference and evaluation

The flow, teacher prior and supervision are not run during synthesis. The
decoder uses its learned attention without an additional text-feedback channel.
No hard monotonic inference mask or attention-derived timestamps are introduced
by this experiment. Those require a separately evaluated online controller.
Teacher-forced validation can use the scheduled prior while autoregressive
generation never does. Compare these separately during the ramp/fade, and check
generated speech again after the prior has fully disappeared.

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
