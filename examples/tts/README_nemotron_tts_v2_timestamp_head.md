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
offsets. The original cut's sample-based frame indices preserve the text/audio
packing clock across resampling. The dataset preserves each complete lexical reference at the same
raw/normalized-text selection used for conditioning, before inline phonemization
or token-channel truncation. Failed teacher alignments do not contribute
timestamp supervision. Singleturn target waveforms are loaded even when cached
codec codes are available; `audio_16khz` can be speaker-reference audio and is
not used as a target waveform.

The output labels are `NONEND=0` and `END=1`. Each describes one acoustic
prediction frame: 80 ms with the 12.5 Hz codec and a frame stacking factor
of one. A single logit is trained with masked FP32 binary cross entropy.
An `END` marks the boundary shared by the end of one word and the next word's
raw CTC onset. These boundaries round to the nearest frame boundary; an `END`
on zero-based frame `f` marks time `(f + 1) * frame_seconds`, at that frame's end.
The first word begins implicitly at zero. The final aligned endpoint is compared
with the actual speech EOS position. Every eligible speech EOS receives an `END`
target, including when the aligned endpoint is missing or shifted. Interior word
targets remain unchanged. The existing audio-delay slice selects the hidden
state, so the timestamp channel receives the speech delay exactly once.
Multiple word ends in one bin remain a positive binary
target and are counted separately as collisions. A binary head cannot represent
how many words share that frame. Padding and failed teacher turns are excluded
from interior word supervision. Cropping does not invent a word end; an actual
speech EOS remains supervised even when the teacher alignment fails.

Teacher intervals cover the full turn: the first word starts at zero, each
word ends at the following CTC onset, and the final word ends at the exact turn
duration. Leading silence belongs to the first word, pauses to the preceding
word, and trailing silence to the last word. Original CTC emission bounds remain
available in `ctc_start` and `ctc_end` for inspection. These are word occupancy
intervals, rather than measured phonetic offsets.

The small head uses LayerNorm, a 128-unit projection, GELU, and one output.
`timestamp_head.pos_weight` controls BCE positive weighting (default 1.0).
`train/timestamp_loss`, `val/timestamp_loss`, supervised-frame accuracy,
valid/invalid turn counts, boundary collisions, and supervised-frame counts
are logged. This is an experimental training target; successful forced
alignment does not establish the accuracy of a newly trained causal head.
Per-class counts and recall help distinguish boundary learning from predicting
only the most frequent class. Training metrics share one distributed reduction.
`timestamp_missing_end_markers` counts turns whose terminal control marker was
overwritten or truncated during packing; their interior word targets remain usable.
`train/timestamp_eos_alignment_missing_count` counts eligible speech EOS positions
without a matching aligned terminal. `train/timestamp_eos_alignment_missing_percent`
reports their percentage using globally reduced missing and total counts.
`train/timestamp_eos_alignment_extra_count` counts aligned terminals away from EOS.
These diagnostics measure target corrections, separately from the head's EOS recall.

`use_timestamp_head` defaults to `false`. The head can coexist with the
existing aligned BPE head and attention loss. Its predictions do not affect
audio sampling or feed tokens back into the decoder.

Limit the teacher's padded acoustic batch with
`timestamp_head.aligner.max_batch_size` (default 8) and
`max_batch_duration` (default 120 seconds). A single turn longer than that
budget retains its full context. `measure_timings=true` synchronizes CUDA
to measure preprocessing, encoder, CTC preparation, and forced alignment;
leave it disabled for normal training.

The teacher batches languages with the same alignment-unit policy and allows
up to `ctc_max_batch_elements=134217728` vocabulary values per decoder call
(512 MiB in FP32). Set a smaller budget to reduce temporary memory.
`use_cuda_alignment=true` uses batched Triton Viterbi search when available,
including the native anchor bands for long turns. CPU execution and unsupported
records retain the existing search. Set `use_cuda_alignment=false`,
`group_languages_by_policy=false`, and `ctc_max_batch_elements=16777216`
to reproduce the original teacher execution path.

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
generated audio decoder grid, including the terminal EOS step when generated.
Streaming state accumulates these classes and
`streaming_finalize()` returns `timestamp_classes` and
`timestamp_classes_lens`. These are predicted boundary classes; word-to-text
assignment and timestamp accuracy must be evaluated separately.

Begin the first word at time zero. Each predicted `END` closes the current word
and opens the next word at the same timestamp. The terminal `END` accompanies
speech EOS and closes the final word at the actual audio duration. This
allows consumers to receive word-end updates during streaming without waiting
for a separate start event or assigning silence its own class.
