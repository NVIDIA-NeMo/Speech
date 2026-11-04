# Nemotron-TTS V2 alignment and streaming timestamps

## Training alignment

`model.use_attention_alignment` defaults to `false`. Enabling it jointly trains
a small Glow-TTS flow aligner and supervises a selected main-backbone attention
head using its MAS path. It does not add a repeated text head or text feedback.
The aligner always sees the same conditioning BPE tokens as the TTS backbone.
If a dataset supplies normalized conditioning text, that selected input is used;
otherwise raw input text is used directly.

The aligner receives detached input text embeddings and codec features. Its
small VITS relative-position Transformer text encoder and normalizing flow are
trainable, while the existing TTS input embeddings are not updated through the
aligner loss. Alignment search batches individual turns and supports automatic
MAS backend selection. Speaker conditioning through a small codec-latent ASP
encoder remains optional and defaults to disabled.

For example, add these overrides to the usual V2 training command:

```bash
++model.use_attention_alignment=true \
++model.aligner_loss_weight=1.0 \
++model.attention_alignment.layers=middle \
++model.attention_alignment.heads=[0] \
++model.attention_alignment.loss_weight=0.1
```

The default schedule trains the aligner from step zero, starts attention
supervision after 10,000 steps, and ramps it over 1,000 steps. A temporary
training-only soft prior is annealed away over 5,000 steps; the alignment loss
continues afterward. `look_back_tokens` and `look_ahead_tokens` can widen the
training target/prior around the teacher path to tolerate local text-order
differences. They never apply an inference attention window.

An already-trained aligner can initialize only the aligner weights:

```bash
++pretrained_model_text_speech_aligner="$ALIGNER_CHECKPOINT" \
++model.attention_alignment.start_step=0 \
++model.attention_alignment.ramp_steps=0
```

The same pretrained option can be placed inside the model configuration. This
is independent of the normal TTS `pretrained_model`; loading the aligner keeps
the other model weights unchanged. Ordinary V2 recipes without alignment
enabled retain their existing training and inference behavior.

Training logs include `aligner_nll`, `attention_alignment_loss`, valid-turn and
causal-frame fractions, and the fraction of one-frame text tokens. The latter
metrics help distinguish a falling likelihood loss from useful text coverage.

## Streaming timestamps

An attention-aligned V2 checkpoint can expose input BPE timestamps while retaining
ordinary speech generation. Timestamp recording is disabled by default and is
independent of the training aligner. Inference does not run MAS or the flow.
External timestamp export requires a V2 checkpoint; requesting it for V1 raises
an explicit error instead of changing legacy inference.

For the external inference and evaluation script, add:

```bash
python examples/tts/magpietts_inference.py \
  --model_type easy_magpie \
  --nemo_files "$CHECKPOINT" \
  --codecmodel_path "$CODEC" \
  --datasets_json_path "$DATASETS" \
  --out_dir "$OUTPUT" \
  --save_timestamps
```

Keep the other sampling and CFG arguments used by the experiment. This option
records attention without applying an attention prior or changing decoding.
The selected layer defaults to the middle full-attention layer of the **main
backbone**, excluding additional acoustic-code predictor attention layers.

Each waveform receives a neighboring `predicted_audio_<id>.timestamps.json`.
Records identify each input occurrence by `turn_index`, `token_index`, and
`token_id`, with `start` and `end` in seconds relative to the saved waveform.
They include `confidence`, `attention_confidence`, and `forced`:

- Ordinary confidence is the duration-weighted full-key attention probability.
  It is not a calibrated probability that a boundary is correct.
- Approximate forward estimates have `forced=true` and `confidence=0.0`.
  Their observed attention score remains in `attention_confidence`.
- Tokens with no estimate retain null times and confidence. Punctuation and
  repeated token IDs retain their original source positions.

Single-turn export writes `timestamps_manifest.jsonl` with `timestamps_filepath`.
Multiturn export includes the same key in its existing turn-level evaluation
manifest and crops/rebases timestamps to each saved turn. Distributed multiturn
merging preserves the files and rewrites their audio references. Filewise
evaluation metrics include `timestamps_filepath` when a sidecar exists.

Native inference pipelines can consume timestamps during generation:

```python
model.eval()
model.configure_timestamps(enabled=True)
state = model.streaming_init(...)

while not state.finished.all():
    state, audio_codes, phonemes = model.streaming_step(state, next_text_token)
    events = model.drain_timestamp_events(state)
    # Deliver audio and process the timestamp events during this same step.

output = model.streaming_finalize(state)
timestamps = output.timestamps
```

Events contain `start`, `end`, `event`, `emitted_at`, `confidence`, `forced`,
`batch_index`, `turn_index`, and `span_id`. Starts and updates are available
before utterance finalization; consumers should apply subsequent updates or
retractions to the same positional identity. No future attention frames are
buffered and there is no timestamp lookahead option.

For another conversation turn, call `model.reset_timestamp_turn(state)` before
the turn's text warmup. Real text tokens consumed during warmup are registered;
silent user/context prefill receives no audio timestamp intervals.
Custom streaming pipelines that supply response text through
`streaming_step(..., prefill_like_step=True)` should also pass
`timestamp_text_warmup=True` for those response-text warmup steps. Leave it
disabled for user/profile prefill so that user text does not enter the agent's
timestamp sequence. The external multiturn runner handles this distinction.

The model's `timestamp_config` configures recording independently of generation:

```yaml
timestamp_config:
  enabled: false
  layer: middle
  head: 0
  method: monotonic
  force_progress: true
  progress_threshold: 0.5
  skip_penalty: 6.0
  confirmation_frames: 2
  min_confidence: 0.0
  min_text_mass: 0.0
  min_margin: 0.0
```

The observer and its per-stream state live in
`nemo/collections/tts/modules/nemotron_tts/v2/timestamps.py`. The reusable
standard-library tracker is `nemo/collections/tts/modules/attention_timestamps.py`.
This path retains only the latest attention row and aggregate token spans,
without storing full attention matrices for the utterance.
