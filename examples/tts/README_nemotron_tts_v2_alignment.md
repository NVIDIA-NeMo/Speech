# Nemotron-TTS V2 alignment and streaming timestamps

## Training

Enable a small, jointly trained ST-TTS flow aligner to supervise selected
main-backbone attention heads with its monotonic alignment search (MAS) path.
Its VITS relative-position Transformer learns from detached conditioning-text
embeddings and codec latents. The aligner uses the same BPE input as the backbone.
The flow uses 80 acoustic channels by default. Codec latents with another width
pass through a shared, learnable orthogonal projection before the flow and
codec-reference speaker encoder. Matching widths use an identity mapping.
Speaker conditioning through attentive statistics pooling defaults to enabled;
set `model.aligned_text_aligner.use_speaker_conditioning=false` to disable it.

Add these overrides to the usual V2 training command:

```bash
++model.use_attention_alignment=true \
++model.attention_alignment.layers=middle \
++model.attention_alignment.heads=[0] \
++model.attention_alignment.loss_weight=0.1
```

The middle layer counts full-attention layers in the main backbone. Additional
acoustic-code predictor layers are separate. By default, the aligner starts at
step zero; attention supervision starts at 10,000 steps and ramps over 1,000.
A temporary training prior anneals away over 5,000 steps. `look_back_tokens` and
`look_ahead_tokens` widen the target band for local text-order differences.

To initialize from an existing ST-TTS/relative-Transformer aligner archive:

```bash
++pretrained_model_text_speech_aligner="$ALIGNER_CHECKPOINT" \
++model.attention_alignment.start_step=0 \
++model.attention_alignment.ramp_steps=0
```

This loads only aligner weights and keeps the current architecture and speaker
conditioning configuration. It reuses V2's partial initialization helper: matching
tensors are loaded, overlapping rows are copied when trailing dimensions match,
and missing or incompatible weights retain their current initialization. The
loader logs partial copies and skipped weights. A new codec projection is trained
when the source checkpoint has none. A checkpoint without aligner weights is
rejected. Keep the default warmup when adapting a pretrained aligner to a new
codec. The text encoder, flow, projection, and speaker encoder remain trainable.
Logs include `aligner_nll`, `attention_alignment_loss`, valid-turn/causal-frame
fractions, and the fraction of tokens assigned one frame.

## Training precision

V2 supports `bf16-true` and `16-true` training. The codec, quantizer conversion,
resampling, evaluation models, and acoustic aligner remain in FP32; the backbone
and prediction heads use the requested dtype. Token losses accumulate in FP32.

With the default AdamW optimizer, enable its fused implementation and use an
epsilon representable in FP16:

```bash
trainer.precision=bf16-true \
++model.optim.fused=true \
++model.optim.eps=1e-4
```

Change `trainer.precision` to `16-true` for FP16. AdamW moments follow parameter
precision in these modes. Existing mixed-precision recipes and optimizer defaults
are unchanged.

## Evaluation text selection

Set `load_normalized_text_percent` per dataset in the JSON passed to
`--datasets_json_path`, for both single-turn and multiturn evaluation:

```json
{
  "en_raw": {
    "manifest_path": "/path/to/manifest.json",
    "audio_dir": "/path/to/audio",
    "language": "en",
    "load_normalized_text_percent": 0.0
  }
}
```

Use `0.0` for raw `text`, `1.0` for `normalized_text`, or a probability between
them to sample the input per example/turn. Missing normalized text falls back to
raw text, including missing entries in multiturn lists. Omitting the key defaults
to `1.0` (normalized text) for both single-turn and multiturn evaluation.
Saved evaluation manifests record the selected input separately from the
normalized CER/WER reference. Use distinct dataset names for raw and normalized
runs to keep their output directories separate.

## Streaming timestamps

Recording is opt-in and observes one attention head during ordinary generation.
External V2 inference/evaluation accepts `--save_timestamps`; each waveform gets
an adjacent `.timestamps.json`, referenced by `timestamps_filepath` in evaluation
manifests. Native consumers can drain updates after every decoding step:

```python
model.eval()
model.configure_timestamps(enabled=True)
state = model.streaming_init(...)

while not state.finished.all():
    state, audio_codes, phonemes = model.streaming_step(state, next_text_token)
    frames = model.drain_timestamp_frames(state)
    events = model.drain_timestamp_events(state)
    # Deliver audio and consume alignment updates before the next step.

output = model.streaming_finalize(state)
final_events = model.drain_timestamp_events(state)
all_frames = output.timestamps[0]["frames"]
```

Frame records identify `batch_index`, `turn_index`, `frame_index`, `token_index`,
`token_id`, and the backbone `text_key_position`. They contain the audio interval
`[start, end)`, `emitted_at`, instantaneous full-key attention confidence, forced
status, and the covered codec-frame indices. A 25 FPS codec with stacking factor
two produces one shared alignment observation per 80 ms audio step, using current
and past attention. Partial final steps are clipped before publishing records.

Token records retain their observed spans. Missing estimates have null times;
forced approximations have `confidence=0`, with raw scores retained in
`attention_confidence`. Scores are uncalibrated estimates. Saved turn crops rebase
times and codec indices while preserving source indices and actual availability.

The frame and token-event queues drain independently and accept `batch_indices`.
Final output retains frame history by default. A consumer storing its own frames
can set `retain_frame_history=False`. Each stream keeps the configuration selected
at `streaming_init`; `get_timestamp_turn_indices(state)` reads turn metadata cheaply.

```yaml
timestamp_config:
  enabled: false
  retain_frame_history: true
  layer: middle
  head: 0
  method: monotonic
  confirmation_frames: 2
  force_progress: true
  progress_threshold: 0.5
  skip_penalty: 6.0
  min_confidence: 0.0
  min_text_mass: 0.0
  min_margin: 0.0
```

For a new response, call `reset_timestamp_turn(state)` before text warmup.
Response-text warmup through `streaming_step(..., prefill_like_step=True)` also
needs `timestamp_text_warmup=True`. User/profile prefill remains outside the
response token sequence. The external multiturn runner handles these transitions.

`tts.modules.aligner` contains the flow aligner, relative-position Transformer,
codec speaker encoder, MAS, and training/checkpoint helpers.
`tts.modules.attention_alignment` contains attention supervision and streaming
estimation, including `build_token_timestamps` for waveform crops and token labels
and `save_token_timestamps` for adjacent JSON files. Both features default to disabled.
