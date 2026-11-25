# Nemotron-TTS V2 timestamp heads

## CTC and teacher distillation

Enable the CTC head independently of the binary word-end head:

```bash
++model.use_ctc_timestamp_head=true \
++model.use_timestamp_head=false \
++model.ctc_timestamp_head.aligner.snapshot_path=/path/to/local/huggingface/snapshot \
++model.ctc_timestamp_head.loss_weight=0.1 \
++model.ctc_timestamp_head.ctc_weight=1.0 \
++model.ctc_timestamp_head.distillation_weight=1.0
```

`use_ctc_timestamp_head` defaults to `false`. The head uses the frozen ASR
aligner's dedicated SentencePiece tokenizer and blank ID, independently of the
TTS input tokenizer. The tokenizer is packaged with the TTS checkpoint so
inference can decode its predictions without loading the ASR teacher.
Vocabulary size, blank ID, and CTC frame interval are derived from the teacher
artifact and saved in the TTS configuration. By default, `ctc_timestamp_head.aligner`
inherits `timestamp_head.aligner`; it can also be configured directly for a
CTC-only experiment.

Each audio hidden state produces two ordered CTC distributions. With the
12.5 Hz codec and frame stacking factor one, these correspond to successive
40 ms subframes, matching the teacher's 25 Hz CTC output. The teacher's acoustic
encoder operates at 12.5 Hz and its CTC adapter upsamples by two. Both student
predictions are available during the same TTS step; this does not add a future
TTS step or guarantee 40 ms timestamp accuracy.

The objective is
`loss_weight * (ctc_weight * CTC + distillation_weight * KL)`.
The default outer weight is 0.1, with both component weights set to 1.0.
In a fresh-head A100 smoke test on three ground-truth speech samples, the
unweighted objective was approximately 41.8–51.4, adding 4.18–5.14 to the
training loss at this outer weight. This small initialization check is a
starting-point calibration, not a converged-loss target or a quality result.
CTC uses the selected reference transcript encoded with the teacher tokenizer.
Distillation matches the frozen teacher's framewise probabilities, including
blank. The teacher retains its highest-probability nonblank classes
(`distillation_topk`, default 32), blank, and the remaining probability mass as
one other category. KL is exact for this coarsened categorical distribution;
it does not distinguish individual classes in that remaining category.

The student computes projection chunks (`chunk_size`, default 128 audio frames)
and checkpoints their intermediate activations during training. Transcript CTC
uses the unique target classes, blank, and a log-sum-exp other category. This
preserves the full-vocabulary CTC objective and gradients while limiting retained
logit memory. Loss arithmetic uses FP32 when model parameters use BF16-true.

Ground-truth agent turns are processed independently and restored to the
original audio timeline. Padding, user turns, and speech EOS are excluded from
lexical CTC predictions. The existing speech-delay slice is applied once.
The CTC-only teacher path uses its encoder and CTC adapter without forced
alignment search. When both heads are enabled with matching teacher settings,
the teacher pass can provide both kinds of supervision.

During streaming, call `model.drain_ctc_timestamp_frames(state)` after each
`streaming_step()` to retrieve the new subframe token predictions and their
probabilities. `streaming_finalize()` exposes `ctc_timestamp_tokens`,
`ctc_timestamp_probabilities`, and `ctc_timestamp_lens`; batch inference exposes
the corresponding fields. These are lexical CTC predictions. Recovering word
timestamps requires matching their decoded token sequence against the available
input transcript; counting nonblank predictions does not provide recovery after
an insertion or omission. The head is an auxiliary output and does not change
audio sampling.

### CTC validation diagnostics

With `use_ctc_timestamp_head=true`, validation reports character error rates
against the exact lexical text selected for TTS, including the chosen raw or
normalized form. Greedy decoding collapses repeated CTC labels before removing
blank; padding and invalid frames do not join separate occurrences or turns.
Both reference and prediction use the existing TTS CER normalization (case,
punctuation, and whitespace), without expanding written numbers into words.

- `val/ctc_timestamp_cer`: student CTC predictions on teacher-forced GT audio.
- `val/ctc_timestamp_teacher_cer`: the teacher's CTC predictions on the same
  retained frame grid.
- `val/ctc_timestamp_generated_cer`: the generated CTC stream when
  `run_val_inference=true` and one exact input reference is available per sample.

CER is the sum of character insertions, deletions, and substitutions divided
by the total reference characters across batches and distributed ranks. It
can exceed one. Reference-character and evaluated-turn/sample counts are logged
alongside the rates; a rate is omitted when its reference-character count is zero.
Generated CER skips packed multiturn samples whose output cannot be paired
unambiguously with one reference. These metrics evaluate the CTC head; generated
speech ASR CER remains a separate metric.

The first validation batch on rank zero logs up to three
`CTC_Timestamp/Alignment_*` figures and a `CTC_Timestamp/Transcripts` table.
The figures compare teacher and student CTC emissions on the same GT speech,
including blank and prediction confidence. Repeated vocabulary IDs share a
token-identity row; the figure does not assign separate word occurrences.
PNG figures and full frame-level JSON records are also saved under
`val_audios/ctc_timestamps_step_<step>/sample_<sample>_turn_<turn>.*`.

Validation capture reuses the teacher and student projections already computed
for their losses. It retains small token/probability observations rather than
full-vocabulary logits and adds no ASR pass. Training does not capture or decode
these diagnostics.

## Binary word-end head

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
Encoder and CTC attention use PyTorch SDPA to avoid compilation for each new
audio shape. The CUDA CTC search reuses kernels across frame and text lengths.

The teacher force-aligns each reference transcript to its ground-truth audio.
The batch layout selects single-turn or multiturn supervision, so multiturn
training can use single-turn validation datasets. Multiturn examples are split
into independent agent turns and the resulting
labels are restored to the original audio timeline, including fractional turn
offsets. The original cut's sample-based frame indices preserve the text/audio
packing clock across resampling. The dataset preserves each complete lexical reference at the same
raw/normalized-text selection used for conditioning, before inline phonemization
or token-channel truncation. Failed teacher alignments do not contribute
interior timestamp supervision. Turns with fewer than two mel frames are skipped
before normalization. Singleturn target waveforms are loaded even when cached
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
An optional local count penalty adds a squared difference between the sum of
END probabilities and binary END targets in each fully supervised sliding
window. Enable it with:

```bash
++model.timestamp_head.local_count_loss.enabled=true \
++model.timestamp_head.local_count_loss.weight=2.0 \
++model.timestamp_head.local_count_loss.window_size=5
```

It defaults to disabled. Five frames cover 400 ms at 12.5 Hz; weight 2.0 is
a strong initial coefficient to validate, not a tuned optimum. The timestamp
objective is `BCE + weight * local_count_loss`; `timestamp_head.loss_weight`
scales that complete objective. Invalid frames break windows, and speech EOS
can close a window but cannot occur inside one, keeping turns independent.
Sequences shorter than the window retain BCE supervision. Genuine adjacent
ENDs remain valid, and target counts use binary frame labels after collisions.
The penalty is computed in FP32 and adds no inference state or lookahead.
It constrains expected counts, not the number of thresholded END events.
When enabled, `timestamp_bce_loss`, `timestamp_local_count_loss`, and
`timestamp_local_count_loss_weighted` are logged under both `train/` and `val/`.

To supervise word-count progress throughout a turn, enable the cumulative
trajectory loss instead:

```bash
++model.timestamp_head.local_count_loss.enabled=false \
++model.timestamp_head.cumulative_count_loss.enabled=true \
++model.timestamp_head.cumulative_count_loss.weight=32.0 \
++model.timestamp_head.cumulative_count_loss.delta=1.0
```

It defaults to disabled. The head accumulates `sigmoid(logit) - binary_target`
on the existing audio prediction grid and applies Huber loss to the cumulative
error at every supervised frame. Counts start at zero, include speech EOS,
and reset after EOS and across invalid regions. A missed or extra END therefore
affects the remaining trajectory until another error compensates it; a short
timing shift affects only the intervening frames. Genuine adjacent ENDs remain
valid. Each frame's Huber contribution is divided by the length of its valid
region before the FP32 supervised-frame mean. This controls the growth of
cumulative gradients on long turns while preserving raw boundary counts and
the minimum of the objective. Empty supervision receives a differentiable zero.
The loss adds no inference state or lookahead.

Weight 32.0 is a strong starting coefficient for this length-normalized loss,
with Huber delta 1.0 measured in boundary-count units. It requires validation
on generated speech. The complete objective is
`BCE + local_weight * local_count_loss + cumulative_weight * cumulative_count_loss`
for enabled terms, scaled by `timestamp_head.loss_weight`. Both count losses
can be enabled independently. `timestamp_cumulative_count_loss` and
`timestamp_cumulative_count_loss_weighted` are logged under `train/` and `val/`,
along with raw BCE. This constrains expected counts; measure thresholded END
counts and timing on generated audio to establish its effect on duplicates.

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
to restore the original batching and CPU alignment settings.

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

## Validation diagnostics

With W&B enabled, the first validation batch on rank zero logs up to three
valid speech turns:

- `Timestamp/Alignment_*`: ASR word occupancy, the head's cumulative word
  position, and `P(END)` with reference and predicted boundaries.
- `Timestamp/GT_reference_*`: ground-truth speech with clicks at ASR word ends.
- `Timestamp/GT_predicted_*`: the same speech with teacher-forced head clicks.
- `Timestamp/Generated_predicted_*`: generated speech with its own head clicks,
  when `run_val_inference=true`.

Plots, marked WAV copies, and frame-level JSON diagnostics are also saved under
`val_audios/timestamps_step_<step>/`. The JSON includes reference words,
probabilities, supervision masks, and predicted end times. Generated recordings
use their own audio duration and predicted boundary classes.

Clicks are short tapered tones, including a clipped tone at the waveform end
when an END is actually predicted there. The displayed word position follows
the raw boundary count; missing and extra ends remain visible. Word identity
is inferred from reference order. Plots show at most 800 frames and 80 word
positions; audio and JSON retain the complete turn. The existing alignment is
reused, and all capture and rendering is limited to validation examples.

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
