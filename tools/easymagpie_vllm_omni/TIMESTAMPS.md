# V2 streaming attention timestamps

This opt-in path follows NeMo's `AttentionTimestampTracker`: it observes head
zero in the middle **full-attention layer of the main backbone**, applies the
same causal monotonic decoder, and preserves missing/forced/provisional states.
The acoustic predictor's separate attention layers do not participate.

Use a V2 checkpoint trained with the corresponding attention-alignment loss.
The observer does not modify attention outputs, weights or the acoustic sampler.
Without the opt-in, no attention observations are collected or emitted.
The original model forward and CUDA-graph settings are preserved. Request
bookkeeping and posterior extraction run in the output hook after graph replay.

## Export and enable

Use the regular `scripts/convert_to_vllm.py` export command. V2 exports now
include a SHA256-checked copy of NeMo's standalone timestamp tracker and its
settings. Serving still does not import the NeMo package.

Add this field to **stage 0** of the regular deployment YAML:

```yaml
hf_overrides:
  easymagpie_timestamps: true
```

Keep any existing `hf_overrides` fields. This version supports a known speaker,
CFG off, tensor/pipeline parallel size one, and prefix caching disabled. Text can
arrive incrementally for a single spoken turn.

## Complete-text request, streaming response

```bash
curl -N http://127.0.0.1:8091/v1/audio/speech/alignments \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, how are you?","voice":"eng","response_format":"pcm",
       "extra_params":{"temperature":0.7,"top_k":80}}'
```

The response is `application/x-ndjson`:

- `speech.start` identifies the stream.
- `alignment.delta` publishes observed frame records, native span events and
  changed `token_timestamps` / `character_timestamps` dictionaries.
- `audio.chunk` carries base64 PCM16, its sample rate, and its sample offset.
- `alignment.done` contains complete token/character dictionaries, word spans
  and the frame trace. Input tokens lacking evidence keep null boundaries.
- `speech.done` closes the response. A `type=error` event means the request failed.

The regular audio endpoint's default byte response is unchanged.

## Incremental WebSocket

Use the existing `/v1/audio/speech/stream` protocol with
`word_timestamps=true` in session configuration. Binary messages remain PCM;
JSON `alignment.delta` messages arrive during generation, followed by
`alignment.done`. Ordinary text may be split across messages. When using
pronunciation-control markers with character timestamps, send each complete
marked span in one `input.text` message. Raw `input.tokens` always retain token
identity; character offsets remain null if the decoded text cannot reproduce
the supplied tokenization losslessly.
The streaming adapter records its initial context length so vLLM-Omni 0.24's
synthetic one-token resume prompts retain the autoregressive decode state.
Runtimes with native terminal-stage tracking retain that lifecycle so input
completion waits for downstream audio; the legacy immediate-close bridge is
only installed on older runtimes.

## Dictionary and clock conventions

`token_timestamps` is keyed by **input-token occurrence index**, not by token ID:
repeated IDs must not overwrite each other. Each record includes `token_id`,
`decoded`, `frame_step`, `frame_end_exclusive`, `time_seconds`, `start`, `end`,
confidence and status. Frame steps count stacked decoder audio observations;
`codec_frame_start/end` in the trace refer to unstacked codec frames. Times are
relative to the first generated audio sample; context and speech-delay warmup
are excluded.

`character_timestamps` uses Unicode character indices and records the character,
its contributing token indices/IDs and its audio interval. Byte-level tokens
that share a Unicode character are retained. **Characters inherit their token
intervals**; this does not estimate independent boundaries inside a BPE token.
Word spans group these character intervals for players and subtitles.

The codec configuration determines resolution. For example, a 12.5 Hz V2 codec
with frame stacking one yields one 80 ms frame per decoder step. These are
attention estimates, not measured ground-truth phonetic boundaries. Forced approximations have zero confidence;
missing evidence has null times. Network arrival time differs from the audio
timeline; use the supplied times to synchronize playback.
The native progress fallback may subdivide one observed frame among crossed
tokens. Those subframe boundaries are forced estimates with zero confidence,
not additional temporal precision. A diffuse attention head can make the
causal decoder stall and then jump; matching NeMo's tracker does not establish
ground-truth word timing. `tracker_settings` records the exact native policy.

For diagnostics, set `extra_params.timestamp_debug=true` on the aligned HTTP
request. Each delta then includes its raw `attention_observation`, allowing an
independent replay through NeMo's tracker without regenerating audio. This is
off by default because posterior arrays increase response size.

## Offline AsyncOmni clients

Stage 0 exposes compact raw observation tensors under `alignment`. Use
`unpack_alignment_packets(output)` and `StreamingAlignmentDecoder.update(...)`
to produce the same online dictionaries. Initialize the decoder with the
converted model path, `EasyMagpieTextTokenizer`, and the supplied text; call
`finalize(audio_duration_seconds)` after consuming audio. Use DELTA output mode
to drain observations instead of retransmitting accumulated history.

The export bundles the exact native path decoder; the GPU worker retains only
the selected head's key history and streams text-key probabilities normalized
over **all** causal keys. It does not materialize full multi-head attention maps
or run an ASR/forced-alignment model.
