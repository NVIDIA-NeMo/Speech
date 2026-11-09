## Full spoken_low with V1.1 and CFG

`ord/spoken_low_cfg.py` runs all 1,836 saved m4a-kl-sft-2 iteration-6000 replies
through the V1.1 FT101k checkpoint with monotonic attention loss, hard-repeat data
and single-turn TN. The recipe is `configs/spoken_low_v11_cfg.json`.
It uses CFG 2.5, audio temperature 0.7, top-k 80, four concurrent independent
requests, the Mg_a_00759 reference voice, and native GPU Parakeet-TDT 1.1B.

From a clean, committed Speech checkout on ORD, with the existing runtime and
converted model under your evaluation root:

```bash
python tools/speech_roundtrip/ord/spoken_low_cfg.py prepare \
  --eval-root /absolute/path/to/Evaluations --run-dir /absolute/path/to/new_run
python tools/speech_roundtrip/ord/spoken_low_cfg.py submit \
  --run-dir /absolute/path/to/new_run --account nemotron_speech_chat \
  --partition polar4,interactive
```

The first Slurm job uses one GPU and the NeMo Speech 26.07.00 container. A
three-reply diagnostic first verifies nonzero guidance-logit changes; a fresh
production sampler then generates the complete dataset with tracing disabled.
Long answers use lossless, independently synthesized chunks of at most 600
characters. Both language-model and codec capacities are 4,096 to accommodate
the 2,048-token audio bound. Bad, short or valid silent speech remains measured.
Original source failures remain in the denominator. Nothing is resampled to
improve its score, and no new upstream language-model replies are generated.

The GPU job then stops TTS and transcribes every waveform with Parakeet using
batch size four and `use_lhotse=False`. Raw ASR and English-normalized `pred_text`
are both saved. A dependent CPU job grades Content/Delivery and the existing
1–5 semantic-preservation rubric with Claude Haiku 5.5. The existing ORD
`with_ord_key.py` loader reads your key into the child environment; no credential
is stored in the configuration or outputs. The source-text baseline is historical
and is not regraded by this launcher. Delivery measures transcript wording.

`plan.json`, `config.yaml`, `control/jobs.json`, `control/deploy.yaml`, runtime
receipts, source/model hashes, audio, transcripts and judge caches provide the
run provenance. TTS is in `tts_run/`, ASR/judge outputs in `Parakeet-TDT/`, and
the final metrics and percentage scoring 5 are in `summary.json`.
Verify both Slurm jobs are `COMPLETED 0:0` and `evaluation_complete.txt` exists.
If a job fails, inspect its logs and run its saved batch script again only after
the original job has ended. Matching completed TTS/ASR/grade caches are reused.
Do not run concurrent writers or edit the pinned checkout during a run.

For another checkpoint, dataset selection or sampling condition, copy the JSON
recipe and pass `--recipe`. The converted model needs its conversion receipt;
its original `.nemo` path is read from that receipt. Runtime paths are resolved
from `--eval-root`, keeping local credentials and run outputs outside Git.


# Speech roundtrip evaluation

Separate evaluation module for **saved answer text → TTS → ASR → verbal judging**.
The original language-model inference and its reports are read-only inputs. New datasets are configuration entries,
not new synthesis or scoring implementations.

Start with a **three-reply, single-turn preview**, one typical-length completed reply from each open VoiceBench
set, and review the saved audio and exact texts. Then prepare a separate full run with all 1,836 replies.
Preview metrics are explicitly labeled and must not be presented as full-benchmark results.

## Pinned ORD profile

`configs/ord_rc23_spoken_low.yaml` points to:

- m4a-kl-sft-2 iteration 6000 `spoken_low` outputs from the completed replication.
- Nemotron-TTS-RC-2.3 Exact-KL/GRPO step 390, using EasyMagpie in NeMo Speech commit
  `f6f61caaab4b13c94c787e7c4b75ab24418a79ed`.
- Reference `Nemotron-VoiceChat/Mg_a_00759.wav`, SHA256
  `b3785260a939849961fa52a006145d6c9d0f97679ddc217705752489f012a201`.
- TTS: `nvcr.io/nvidia/nemo-speech:26.07.00`, checkpoint conversion in its NeMo environment, then an
  **isolated** serving environment containing `vllm==0.24.0` and `vllm-omni==0.24.0`.
  Conversion uses a system-package overlay with Speech's exact `lhotse==2.0.0a6` requirement while retaining
  the image's Torch 2.12; serving separately uses Torch 2.11 required by vLLM 0.24.
- ASR: `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`, BF16, a separate environment with vLLM 0.32.0 and the
  same pinned Speech checkout. TTS and ASR dependencies must not be installed into one environment.
- Judge: the original, unchanged `judge_verbal_rollouts.py` and `judge_verbal_open_v1.yaml`, Claude Haiku 5.5,
  omitted temperature, 2,048 output tokens, 16 workers, initial rate 5. The prior evaluation's Python environment
  has OpenAI 3.20 and its native `httpx2`; the profile explicitly maps the pinned judge's HTTP transport import.

The ASR prompt requests **verbatim transcription**, fillers and repetitions included. No original question or
answer is supplied to ASR. Thinking is disabled, decoding temperature is zero, and truncated ASR responses are
never accepted as transcripts. Raw ASR responses are retained; no automatic repetition cleanup, inverse text normalization,
paraphrasing, or correction is applied by this module.
The ASR setup receipt pins Hugging Face revision `c17667641642c4265ff2b7037dd101b7d4567634`; the runner exports
`SPEECH_ROUNDTRIP_ASR_REVISION` so sidecars and cache fingerprints record that revision.

Replies longer than five minutes use their existing TTS audio chunks as independent ASR inputs (configurable
with `asr.max_single_request_audio_s`, default 300). Before transcription, the module verifies that concatenating
their PCM payloads reconstructs **every frame** of the finalized reply exactly. Raw segment responses are retained
and joined with one space, without punctuation cleanup or text hints. This bounds long-form ASR requests for the
unusually long incomplete source replies. It preserves their inclusion in the benchmark and their full audio.

ASR output tokens are bounded by `min(16384, max(512, ceil(32 * audio_seconds) + 256))`. If a request reaches that
limit, the partial hypothesis is saved for audit and excluded from the transcript. Recovery recursively splits
the audio at a low-energy boundary near its midpoint; subsequent requests for that reply target at most 15
seconds. All frame ranges are contiguous with no overlap or missing samples. Successful segment responses are
checkpointed individually under `asr_chunks/`, so an interrupted reply can resume without redoing them. The
derived WAV ranges are under `asr_audio_chunks/`; synthesis WAVs remain unchanged. An unresolved output limit on
a minimal segment stops evaluation explicitly. A successful, empty ASR response is retained as an empty measured
transcript rather than fabricated or dropped.

A successful `stop` response with null content is also an empty transcript when it has no reasoning, refusal or
tool output. Those auxiliary fields are never promoted into transcript text. Other unclean responses are saved
and recovered through the same audio-subdivision path. The ORD batch helper permits five missing-only client
attempts with 60-second backoff while the server remains healthy; successful reply and segment caches survive.

The first full ASR attempt exposed repetition loops on six replies. Before resuming, all 1,028 successful
transcripts were checked: every successful response finished below the new duration-aware bound. Those saved
responses are reused unchanged. `asr_recovery_receipt.json` records their checksums and the recovery protocol.

## Stages

Install this package with `python -m pip install -e /path/to/speech_roundtrip`, or set
`PYTHONPATH=/path/to/speech_roundtrip/src`. Core CLI/adapters require Python 3.10+ and PyYAML; GPU dependencies
belong to their respective runtime environments.

```bash
# Immutable input manifest; validates full source row counts even when selecting a preview.
python -m speech_roundtrip prepare --config configs/ord_rc23_spoken_low.yaml \
  --run-dir /absolute/run/preview --preview-per-dataset 1

# Each HTTP speech request is independent: known reference voice, no user-audio history or prior turns.
# Start the matching EasyMagpie server before running this stage.
SPEECH_ROUNDTRIP_TTS_URL=http://127.0.0.1:8091 \
  python -m speech_roundtrip synthesize --run-dir /absolute/run/preview

# Start the ASR vLLM server in its separate environment before running this stage.
SPEECH_ROUNDTRIP_ASR_URL=http://127.0.0.1:8092 \
  python -m speech_roundtrip transcribe --run-dir /absolute/run/preview

# Use the original judge's environment with an existing OPENAI_API_KEY exported in memory.
python -m speech_roundtrip judge --run-dir /absolute/run/preview
python -m speech_roundtrip report --run-dir /absolute/run/preview
python -m speech_roundtrip status --run-dir /absolute/run/preview
```

`judge` grades **both selected original text and recovered speech transcripts in the same fresh pass**. It does
not reuse grades from the earlier experiment. This supports a paired comparison without mixing judge deployments
or caches. Successful grades are reused only inside this run; missing grades are retried without regrading
successful rows. Content, delivery, and overall use unweighted dataset means, matching the original protocol.
Median words pools all selected responses. Grading includes every selected row, including original source failures.
The original `candidate_status` is preserved for provenance; stage completion is tracked separately.

## Semantic preservation judge

The separate semantic judge uses `src/speech_roundtrip/prompts/semantic_preservation.txt`, revised at the user's
request to **always return an integer from 1 to 5**. It has no verdict or check-status fields and never returns a
null score. The provided original is retained in `docs/semantic_preservation_original.txt` for provenance.
The revised prompt SHA256 is `5bbdbfb93a554011eb6e9a157517a75ffd73f681006ab08904916550d1458a3a`.

```bash
python -m speech_roundtrip semantic-judge --run-dir /absolute/run/full
python -m speech_roundtrip semantic-report --run-dir /absolute/run/full
```

By default this uses the configured verbal judge model, Claude Haiku 5.5 in the ORD profile. Each user message
contains exactly `source_text` and `asr_transcript`, both complete strings without truncation. It receives no
original question, language labels, prior grades, generator identity, or normalization hints. It measures whether
the speech transcript preserves the source meaning; faithfully transmitting a preexisting factual mistake can
receive 5 even when the separate Content judge gives the original answer a low score.

The output retains literal evidence, uncertainty notes, wording notes, and observability limits.
Uncertainty alone does not lower the grade; if no definite semantic loss can be established, the revised prompt
requests 5 with any material limitations disclosed. Scores below 5 require concrete input evidence, and score 4
permits only `minor_nuance_changed` errors. Every response must state `audio_assessed=false` and
`error_attribution=UNDETERMINED_FROM_TEXT`.

The driver validates the exact JSON schema, integer scores, score/error consistency and literal evidence
substrings. Invalid or failed API responses are retried without modifying the supplied pair. It never substitutes
a placeholder score. A final report requires a valid score for every row. Successful grades are cached only for
the exact model, prompt, text pair and generation settings; use `--output-dir` for a different judge variant.

Artifacts are under `semantic_judge/`: a prompt snapshot, input pairs, immutable identity, per-record cache with
raw outputs/attempt history, and `reports/{summary.json,report.txt,grades.jsonl}`. Reports retain the ordinal 1-5
scale with per-set means, a three-set mean, pooled median and counts at each grade. They do not convert the
scores to percentages. The full ORD runner sequences semantic grading after Content/Delivery grading to keep
the Hub request load bounded.

The original Content/Delivery judge sends up to 3,000 characters of the question and 8,000 characters of the answer, and supplies
the word count of the complete answer. This preserves the original evaluation protocol; the full source text,
speech, and recovered transcript remain available even for replies exceeding the judge's character limit.

**Delivery is transcript-based spoken wording/readiness, not acoustic prosody or voice quality.** Retain the WAVs
for listening checks. ASR spelling and punctuation can also change word counts and delivery flags.

For a full run after reviewing the preview, choose a new run directory and omit `--preview-per-dataset`.
This includes all input rows; it does not drop incomplete original language-model outputs. Empty source answers
are represented by explicitly labeled short silence and an empty transcript, and still receive a grade.

## Semantic audio LLM: direct audio, 1–3

`semantic-audio-llm` implements the supplied `judge.py` as a separate stage named **semantic audio LLM**.
The user confirmed this judge's native **1–3** scale: 3 = correct, 2 = minor issues, 1 = semantic or identity
errors, missing content, unintelligibility, or repetition loops. The exact supplied rubric is packaged in
`src/speech_roundtrip/prompts/semantic_audio_llm.txt`; the original script and its hashes are preserved in `docs/`.
The default model is `gcp/google/gemini-3.8-flash` on the existing inference gateway.

```bash
# Only finalized TTS outputs are required. Nemotron ASR outputs are not read.
python -m speech_roundtrip semantic-audio-llm --run-dir /absolute/run/full
python -m speech_roundtrip semantic-audio-llm-report --run-dir /absolute/run/full

# A separate, explicitly labeled subset for validation:
python -m speech_roundtrip semantic-audio-llm --run-dir /absolute/run/full \
  --output-dir /absolute/run/full/audio_llm_preview --limit 3
```

Each request contains the exact reference text and the original WAV as `input_audio`, matching the supplied
script. It supplies no prior ASR transcript, question, generator identity or existing judge score. The model returns
`heard`, integer `score` (1, 2 or 3), and `issues`. Temperature is zero; the initial output budget is 1,200 tokens
and retries can use 4,000. The ORD convenience wrapper is `ord/semantic_audio_llm.sh` (also installed under the
runtime scripts directory); it uses the existing ORD key loader without storing a credential.

By default, requests are bounded to 12 MiB of WAV data and 300 seconds. Larger replies use their saved TTS
chunks, with each original text chunk paired to its exact WAV. Before any API calls, the runner verifies that the
text concatenation is unchanged and that the chunk PCM concatenation covers every frame of the finalized full
WAV. A long reply's score is the **minimum** of its chunk scores, matching the rubric's rule that any major error
determines the overall score. Segment grades and literal heard text remain available. No reference or waveform
is truncated; an oversized individual TTS chunk raises an explicit error.

If the model exhausts the output budget while transcribing a long repetition run, resume with
`--recover-length`. This adds an explicitly recorded **output-format** recovery instruction after the native
retry policy has failed. For source replies already known to hit that limit, remaining unscored chunks use the
same recovery. The original rubric, reference and complete WAV stay intact; `heard` can use a few literal
repetitions followed by `[repetition continues]`, with the continuation described in `issues`. Such results carry
`heard_is_excerpt: true` and must not be used as complete ASR transcripts. A locked recovery-policy snapshot,
the triggering failures and successful-cache preservation checksums are saved. Existing valid grades and raw
responses are never rewritten. Other malformed or filtered responses are not converted into a grade.

Artifacts are in `semantic_audio_llm/`: immutable prompt/input snapshots and manifest, per-segment request
history in `segments/`, per-reply results in `cache/`, and `judge.json`, `judge.utterances.jsonl`,
`judge.cache.jsonl`, and `reports/`. Reports include the pooled mean (as in the supplied script), the unweighted
mean of dataset means, per-set distributions and worst cases. Every selected row, including incomplete original
language-model answers, must receive a valid grade before a complete report is written.

Compared with the supplied standalone script, this adapter validates audio and source checksums, binds caches
to model/base URL/rubric/generation settings/text/audio/chunk plan, preserves raw responses, rejects truncated
outputs, and requires complete JSON rather than inventing a grade from an isolated `score:` string. Successful
grades are reused on resume; a valid low score is never retried to obtain a better score. The existing 1–5
text-pair semantic judge remains a separate metric.

For a future dataset, use the existing generic JSONL adapter and add the block in
`configs/semantic_audio_llm.yaml` to its new run configuration. Existing finalized runs can use the CLI defaults
without changing their original manifests. Change the model, limits or rubric only in a new `--output-dir`.

## Parakeet-TDT ASR variants

Keep a new ASR evaluation in its own run directory. `prepare-asr-variant` copies the immutable input and TTS
metadata from a finalized run, preserving all rows and referencing the same original WAVs. It creates no new
speech. Its default label is **Parakeet-TDT**.

```bash
python -m speech_roundtrip prepare-asr-variant \
  --source-run /absolute/path/original_tts_run --run-dir /absolute/path/Parakeet-TDT \
  --checkpoint /absolute/path/parakeet-tdt-1.1b.nemo
python -m speech_roundtrip transcribe --run-dir /absolute/path/Parakeet-TDT
python -m speech_roundtrip judge-asr-variant --run-dir /absolute/path/Parakeet-TDT
python -m speech_roundtrip semantic-judge --run-dir /absolute/path/Parakeet-TDT
```

Parakeet runs on CUDA through `ASRModel.restore_from(...).to('cuda').eval()`, loading once and reusing the model
with `TranscribeConfig(batch_size=4, use_lhotse=False)`. Checkpoint decoding and dtype are retained. Both
`raw_asr_text` and `pred_text` are saved; `pred_text` uses `get_text_processor('en').process_text_for_wer` and is
the transcript supplied to Content/Delivery and the separate **1–5** text semantic judge. Normalization runs
on the complete joined transcript. The unchanged original question is supplied only to the verbal judge.
The separate 1–3 audio LLM stage is independent of ASR and is not rerun for an ASR replacement.

Replies over 180 seconds use the already generated TTS chunks, verifying exact PCM reconstruction and full
frame coverage. Native transcription batches are ordered by duration to limit padding, while results map back
to their immutable reply IDs. No text or question is supplied to ASR. Every segment is committed atomically;
resume reuses only checksummed, matching outputs. Checkpoint SHA256, decoding configuration, runtime and
postprocessor source hashes are recorded. Missing or malformed hypotheses stop the run without placeholder
transcripts or changed denominators.

`judge-asr-variant` makes a fresh Haiku two-axis pass over the new transcripts, using the requested evaluation
label and a separate cache. It does not regrade the original source-text baseline. The 1–5 judge retains the
existing score-only prompt and complete-pair validation. `report-asr-variant` and `semantic-report` recompute
reports from saved grades. Content/Delivery use the unweighted mean across datasets; median words is pooled.
Parakeet's lowercase, unpunctuated wording and the requested normalization are part of this evaluation's
transcript representation and are recorded alongside both forms of text.

## New datasets

A generic JSONL needs one object per line:

```json
{"id":"example-1","question":"Original user question","text":"Saved answer to synthesize."}
```

```yaml
datasets:
  - name: my_dataset
    format: jsonl
    path: /absolute/path/answers.jsonl
    expected_rows: 100
    # Optional field mapping:
    fields: {id: sample_id, question: prompt, text: response}
```

Use `format: audio_evals` for the original harness schema. Dataset names and identifiers are validated; duplicate
identifiers, malformed JSON, missing fields, changed run configuration, and unexpected row counts fail explicitly.
Adding a dataset changes the run identity, so use a new run directory.

## Artifacts and resume guarantees

Each run contains `run_manifest.json`, `inputs.jsonl`, exact `texts/<key>.txt`, generated `audio/<key>.wav`, TTS and
ASR JSON sidecars, staged judge inputs, a new judge cache, and final `reports/summary.json` / `report.txt`.
Identifiers map deterministically to filenames. Source files, reference voice, converted-model receipt, text,
generation settings, and generated audio are fingerprinted or checksummed.

Audio is mono PCM16 at 22,050 Hz. The HTTP client requests raw PCM and writes a complete WAV atomically. The
codec runs in FP32 as recommended upstream. Long text is split at sentence/whitespace boundaries and concatenated
without inserted silence; **all original characters are preserved**. These are independent synthesis chunks of
one answer, not conversational turns. The preview replies fit in one chunk each.

Use the runtime's `configs/easymagpie_ord.yaml` deployment file. It copies the pinned upstream preset and raises
Stage 1 `max_model_len` and `max_num_batched_tokens` from 512 to 4,096. The native streaming codec retains causal frame history; the
512-entry input buffer can overflow when a longer utterance resumes. The larger buffer accommodates the unchanged
2,048-token generation bound. With chunked prefill disabled for atomic codec chunks, vLLM requires the scheduling
budget to be at least the model-length limit. Codec precision, streaming chunk sizes, LM sampling and the reference voice stay
at the reviewed settings. `tts_engine_capacity.json` records both deployment hashes and the capacity change.

Outputs are committed per reply. Resume skips only matching completed artifacts with valid audio checksums.
Corruption, truncated WAVs, failed upstream stages, and missing judge grades stop processing instead of changing
the evaluation denominator. A failed/uncommitted reply can be retried without regenerating completed replies.
ASR and status checks validate the WAV payload and SHA256 without recomputing per-sample RMS; synthesis already
records that measurement. This keeps audio integrity checks from becoming a Python CPU bottleneck during GPU ASR.
Do not run two writers against the same run directory.
The optional `tts.concurrency` and `asr.concurrency` settings bound HTTP requests in flight (default one, maximum
32). They batch independent single-turn replies within the same GPU server; they do not create extra GPU jobs
or share conversational state. The full ORD profile uses eight requests per stage on one GPU, within the server's
configured capacity. Concurrency is fingerprinted with other stage settings.

## ORD orchestration

After the pinned runtime is installed, a new prepared run can use the reusable launcher:

```bash
ROOT=/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/users/ecasanova/Nemotron-VoiceChat/Evaluations
RUN="$ROOT/speech_roundtrip_runs/my_dataset_v1"
export PYTHONPATH="$ROOT/speech_roundtrip/src"
"$ROOT/spoken_low_replication_20261105/code/audio-evals/.venv/bin/python" -m speech_roundtrip prepare \
  --config /absolute/path/my_dataset.yaml --run-dir "$RUN"
bash "$ROOT/speech_roundtrip_runtime/scripts/submit_roundtrip.sh" "$RUN"
```

Use a fresh run directory for a new dataset/configuration. This launcher checks the pinned Speech revision,
records both job IDs under the run, runs TTS and ASR sequentially on one GPU, then runs both judges. Outputs and
completion markers belong to that run. It refuses duplicate initial submissions. For a failed run, inspect the
saved job IDs and logs before resubmitting only the required stage; valid reply/segment caches are reusable.
The `*_run.sbatch` scripts accept `SPEECH_ROUNDTRIP_RUN`; the date-specific `*_full.sbatch` scripts reproduce the
original experiment. The reusable wrappers were syntax-checked; their server settings and stage commands are
copied from the validated ORD run. Changing model checkpoints requires a corresponding runtime conversion.

Slurm scripts, image cache, and conversion receipts are under the sibling `speech_roundtrip_runtime/` directory.
The current scripts are `scripts/bootstrap.sbatch`, `scripts/tts_preview.sbatch`, and their inside-container helper.
Submit using `sbatch --parsable`; verify `COMPLETED` / exit `0:0` and the stage artifacts. All GPU work runs through
Slurm. Container commands mount `/lustre:/lustre` and use `--no-container-mount-home`.

Container assembly must set `ENROOT_TEMP_PATH` to compute-node local scratch: overlay assembly on Lustre fails.
Downloaded layer caches and the final SquashFS remain in the authorized workspace. See the
[upstream Enroot filesystem issue](https://github.com/NVIDIA/enroot/issues/30).
Serving also binds a short node-local directory at `/rt` and sets `TMPDIR=/rt`; Unix-domain IPC socket paths
cannot exceed 107 characters. Persistent model outputs and compiler caches remain under Evaluations.

ASR runners must `cd` to the pinned `Speech` checkout before launching Python. The container's default
`/workspace` directory contains another NeMo checkout that otherwise shadows the intended source in Python
inspection subprocesses. The runner checks the imported source path. Keep vLLM's model-info cache separate for
this checkout (`cache/vllm_asr_f6f61caa`): metadata created while the bundled checkout was imported can incorrectly
report that ReplaySSM is unsupported, even after fixing the Python path. No upstream source patch is required.

On the ORD A100, use `--mamba-backend triton` and omit `--use-replayssm` and speculative decoding. The model card's
FlashInfer ReplaySSM/MTP recipe compiles a checkpointing kernel containing Hopper-only `griddepcontrol`
instructions and fails on this A100. This is a serving optimization choice: the ASR checkpoint remains BF16 in
vLLM 0.32. Set `FLASHINFER_WORKSPACE_BASE` explicitly under the runtime cache directory so any remaining
FlashInfer sampling builds persist between container jobs; its default workspace ignores `XDG_CACHE_HOME`.

The full ORD profile is `configs/ord_rc23_spoken_low_full.yaml`. Its stage scripts are `tts_full.sbatch` and
`asr_full.sbatch`; submit the second with an `afterok` dependency on the first. The full run preserves all 636
AlpacaEval, 200 CommonEval, and 1,000 WildVoice replies, including five incomplete source replies. There are 2,328
lossless synthesis chunks across these 1,836 independent replies. A run's `control/finish_after_asr.sh` waits for
successful Slurm completion before invoking the paired judge. Verify the stage markers and the actual Slurm
exit codes, not only a server startup or an output-file count.

No credentials belong in this configuration, source files, or reports. The Hub judge can use the existing
NVIDIA_API_KEY in ORD bashrc through the prior run's `scripts/with_ord_key.py`; it sets OPENAI_API_KEY only in the
child process environment. Local TTS and ASR inference require no Hub key.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests cover preservation of source failures, duplicate IDs/counts, immutable resume identities, lossless chunking,
real HTTP stage execution against a local mock server, WAV validation, resume behavior, and prevention of original
answer/question leakage into ASR. They also verify exact long-audio frame coverage, transcript concatenation and
rejection of altered synthesis chunks. Real-model validation is separate: listen to the single-turn preview before a
full batch, then transcribe and grade a small sample before scaling.

Upstream references: [EasyMagpie serving and conversion](https://github.com/NVIDIA-NeMo/Speech/blob/main/tools/easymagpie_vllm_omni/README.md),
[Nemotron 3.5 Transcribe vLLM setup and prompts](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B).
