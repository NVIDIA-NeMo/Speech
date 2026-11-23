# Nemotron-TTS V2 word-boundary pilot

Enable an auxiliary frame classifier with a frozen Nemotron 3.5 Transcribe
acoustic encoder and CTC timestamp adapter as its teacher:

```bash
python examples/tts/easy_magpietts.py \
  --config-path=conf/magpietts \
  --config-name=easy_magpietts_lhotse_multiturn \
  ++model.model_type=v2 \
  ++model.use_timestamp_head=true \
  ++model.load_cached_codes_if_available=false \
  ++model.timestamp_head.aligner.snapshot_path=/path/to/local/huggingface/snapshot \
  ++model.timestamp_head.loss_weight=0.1
```

The snapshot must contain `config.json` and `model.safetensors` from
`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`. Only its mel frontend,
ASR encoder, and CTC adapter are loaded. The language model and diarization
model are not needed for aligning individual ground-truth speech turns.
The teacher stays frozen in evaluation mode and is excluded from TTS
checkpoints and optimizers. Its checkpoint precision is preserved when the
TTS model uses BF16-true; preprocessing and the CTC adapter use FP32.

The teacher force-aligns each reference transcript to its ground-truth audio.
Multiturn examples are split into independent agent turns and the resulting
labels are restored to the original audio timeline, including fractional turn
offsets. The dataset preserves each complete lexical reference at the same
raw/normalized-text selection used for conditioning, before inline phonemization
or token-channel truncation. Failed teacher alignments do not contribute
timestamp supervision. Singleturn target waveforms are loaded even when cached
codec codes are available; `audio_16khz` can be speaker-reference audio and is
not used as a target waveform.

The output labels are `OUTSIDE=0`, `START=1`, `CONTINUE=2`, `END=3`, and
`SINGLE=4`. Each label describes one acoustic prediction frame: 80 ms with
the 12.5 Hz codec and a frame stacking factor of one. A word occupying one
frame receives `SINGLE`. Silence between words receives `OUTSIDE`. Word
boundaries are rounded to the nearest frame boundary, so adjacent words
share one quantized boundary rather than losing both boundary labels.
Short words that round to zero frames receive `SINGLE` in their midpoint
frame. Bins claimed by multiple words are counted and excluded, without
shifting neighboring words. Ordinary boundaries round by at most 40 ms;
representing a short word as one full frame can extend its interval further.
Cropping does not manufacture new word boundaries.

The small head uses LayerNorm, a 128-unit projection, GELU, and a five-class
projection. Masked FP32 cross entropy is added to the existing training loss.
`train/timestamp_loss`, `val/timestamp_loss`, supervised-frame accuracy,
valid/invalid turn counts, collision/collapsed-word counts, and supervised-frame counts
are logged. This is an experimental training target; successful forced
alignment does not establish the accuracy of a newly trained causal head.
Per-class counts and recall help distinguish boundary learning from predicting
only the most frequent class.

The training teacher retains raw CTC emission spans instead of extending word
ends to the next word's onset. These spans preserve gaps but can be brief,
especially for words represented by one CTC token. They are pseudo-labels,
not measured phonetic boundaries. Inspect the listening report before a long
training run. `timestamp_head.aligner.alignment.refine_word_timestamps=true`
selects the ASR utility's usual boundary refinement instead; the utility's
default behavior for existing ASR callers is unchanged.

`use_timestamp_head` defaults to `false`. The head can coexist with the
existing aligned BPE head and attention loss. Its predictions do not affect
audio sampling or feed tokens back into the decoder.

Limit the teacher's padded acoustic batch with
`timestamp_head.aligner.max_batch_size` (default 8) and
`max_batch_duration` (default 120 seconds). A single turn longer than that
budget retains its full context. `measure_timings=true` synchronizes CUDA
to measure preprocessing, encoder, CTC preparation, and forced alignment;
leave it disabled for normal training.

## Inspect reference-word alignments

Prepare a JSONL manifest with `id`, `audio_filepath`, `text`, and `language`
(`en`, `pt`, or `es`, for example), then run:

```bash
python examples/tts/inspect_nemotron_timestamp_aligner.py \
  --manifest /path/to/samples.jsonl \
  --model-path /path/to/local/huggingface/snapshot \
  --batch-size 8 \
  --output-json /path/to/alignments.json \
  --output-html /path/to/alignments.html
```

The standalone HTML embeds the audio. Clicking a word plays exactly its
aligned interval; full recordings, interval tables, and teacher diagnostics
are also available. An existing result JSON can be rendered again with
`--alignments-json` instead of running the teacher again.

## Passive inference output

For checkpoints trained with the head enabled, `infer_batch()` returns
`predicted_timestamp_classes` and `predicted_timestamp_classes_lens` on the
generated audio frame grid. Streaming state accumulates these classes and
`streaming_finalize()` returns `timestamp_classes` and
`timestamp_classes_lens`. These are predicted boundary classes; word-to-text
assignment and timestamp accuracy must be evaluated separately.
