# Stage 2: Data, Lhotse, Bucketing, And Blends

Use Lhotse by default for training and validation. For validation, prefer Lhotse with static `batch_size` and no
bucketing; dynamic validation batches can rarely starve DDP ranks.

## Manifest Preparation

Standard ASR JSONL:

```json
{"audio_filepath": "/data/audio/sample.wav", "text": "transcript text", "duration": 3.42}
```

Use separate train, validation, and test manifests. Prefer 16 kHz mono audio unless the model card/config says
otherwise.

## Transcript Style Preflight

Do this before sharding, tokenizer changes, OOMptimizer, or fine-tuning. Treat mixed transcript style as an experiment
setup failure.

For every train, validation, and test manifest, and for each source inside a blend, audit a representative sample of
reference transcripts. At minimum, report:

- Fraction with uppercase letters.
- Fraction with punctuation or symbols, including sentence punctuation, quotes, dashes, and brackets.
- Number, abbreviation, currency, unit, and special-symbol conventions.
- Unicode normalization and script/language consistency.
- A few examples before and after the intended text transform.

The target style must be consistent within and across all fine-tuning sources, and the validation/test style must match
the metric being optimized. Do not rely on blend weights to average away style conflicts. If one source is cased and
punctuated and another is lowercase/no-punctuation, fix the manifests or split the task before launching training.

Also check the original checkpoint's prediction style when applicable:

- First read the model card/config/docs for punctuation, capitalization, ITN, language, and prompt behavior.
- If the checkpoint can already transcribe the target language or a close proxy, run a small baseline transcription on
  representative fine-tuning audio and compute the same style rates on predictions.
- If the checkpoint is being moved to a new language/script and baseline predictions are unreliable, do not infer style
  from bad transcripts; record the checkpoint style as documented or unknown.

If all fine-tuning data is internally consistent but differs from the original checkpoint's prediction style, that can
be acceptable. Flag it to the user because the model may shift output style and raw WER may move even when recognition
improves. In autonomous runs, consider adapting the fine-tuning transcripts to the checkpoint's output style when that
style is required by deployment or evaluation. If that requires adding missing punctuation/case/ITN labels, treat the
new labels as generated data: validate them and keep the raw manifests.

Choose one target style and materialize new manifests before training:

- Normalized ASR style: lowercase if appropriate for the language, remove punctuation/symbols that are not part of the
  target transcript, normalize whitespace and Unicode, and score with the same normalization.
- Cased/punctuated/ITN style: use data that already has reliable labels in that style, or restore missing style labels
  with a validated post-processing pipeline. Do not mix unstyled labels into this target style without relabeling.
- Prompted AED/Canary style: use task/language/punctuation prompts only when the selected script and metrics support
  separating those behaviors; otherwise still enforce one transcript style per fine-tuning objective.

Preserve raw manifests. Write derived manifests with a clear suffix such as `_style_normalized.json`, record the exact
transform, and include the style audit in the run ledger.

Shard training manifests even when not tarred. For resume-heavy fine-tuning with an unsharded non-tarred manifest,
each restart can begin iterating from the start again. Split into shards like `manifest_0.json` ... `manifest_N.json`
with roughly 200 utterances per shard. This is less important for one-shot runs that will not be interrupted.

Strongly prefer tarred data for slow filesystems and object stores. If data is small or there is abundant fast local
SSD, non-tarred audio is fine, but still shard the manifest. Do not use tarred datasets for validation/test unless the
target script and docs explicitly support the behavior you need.

Before training, inspect duration and token-per-second distributions. Set `min_duration`, `max_duration`, `min_tps`,
and `max_tps` so they do not silently filter out a large or important part of the fine-tuning set.

If OOMptimizer cannot fit batch size 1 for an extreme duration bucket, inspect the duration tail before changing the
model or precision. When only a tiny outlier fraction is affected, set `max_duration` to cap that tail, record the
number and hours filtered, and rerun OOMptimizer with the capped final bucket. Do not silently cap substantial or
domain-critical long-form data.

When duration-tail capping is used, include the chosen cap as the final duration bucket passed to OOMptimizer. This
keeps the last bucket aligned with the longest sample that training will actually admit.

Run the first OOMptimizer pass with its default CLI settings. Do not lower `--memory-fraction`, disable DDP simulation,
change dtype, or adjust the search threshold unless there is a concrete reason. The default profile is intended to be
aggressive enough for high GPU utilization while reserving non-training-loop memory. If the resulting real training run
OOMs, then rerun OOMptimizer with a lower `--memory-fraction` and document why.

Validate the generated profile with a short multi-GPU pilot and sample GPU utilization during training steps, not just
startup, checkpointing, or validation. OOMptimizer sizes worst-case synthetic batches at the bucket upper bounds; real
training memory can be noticeably lower when the sampler draws shorter cuts or smaller buckets. Focus on peak training
memory and sustained SM utilization over enough steps to exercise several buckets.

Sample utilization while training is inside the train loop:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader,nounits
```

Do not judge utilization from checkpoint save, validation, export, dataloader prefill, or process shutdown windows.
Record peak memory and sustained SM utilization in the run ledger. If memory remains much lower than expected, first
confirm the run is using the OOMptimizer `bucket_batch_size` profile and that conflicting batch settings are null.

For non-tarred data, Lhotse may tokenize samples during sampling when `pretokenize=true`. This enables token-per-second
filtering and 2D token bucketing, but tokenization happens in the main training process and can slow training with large
tokenizers. If the run does not use token-per-second filters or 2D bucketing, consider `pretokenize=false`.

## Batch Mode Compatibility

When changing Lhotse batch settings, explicitly null conflicting options. `dataloader.py` accepts these batch sizing
modes:

- Static batch size: set `batch_size=<N>`, `use_bucketing=false`, and set `batch_duration=null`,
  `quadratic_duration=null`, `bucket_duration_bins=null`, `bucket_batch_size=null`, `batch_tokens=null`.
- OOMptimizer profile: set `bucket_duration_bins=[...]` and `bucket_batch_size=[...]`, and set `batch_size=null`,
  `batch_duration=null`, `quadratic_duration=null`, `batch_tokens=null`.
- Heuristic dynamic duration batching: set `batch_duration=<seconds>`, optionally `quadratic_duration=<seconds>`, and
  set `bucket_batch_size=null`, `batch_tokens=null`. Prefer OOMptimizer instead.

`bucket_batch_size` requires `bucket_duration_bins`. Setting either auto-enables `use_bucketing=true`, but set it
explicitly for clarity.

## Training Lhotse Defaults

Use integer `val_check_interval`, `limit_train_batches` as pseudo-epoch length, and `max_steps` as the real duration.
Do not use `trainer.max_epochs`.

```bash
++model.train_ds.use_lhotse=true \
++model.train_ds.use_bucketing=true \
++model.train_ds.batch_size=null \
++model.train_ds.batch_duration=null \
++model.train_ds.quadratic_duration=null \
++model.train_ds.bucket_duration_bins='[...]' \
++model.train_ds.bucket_batch_size='[...]' \
++model.train_ds.bucket_buffer_size=10000 \
++model.train_ds.shuffle_buffer_size=10 \
++trainer.use_distributed_sampler=false \
+trainer.limit_train_batches=1000 \
trainer.val_check_interval=1000 \
trainer.max_steps=<steps>
```

Validation should use Lhotse but static batches:

```bash
++model.validation_ds.use_lhotse=true \
++model.validation_ds.use_bucketing=false \
model.validation_ds.batch_size=8 \
++model.validation_ds.batch_duration=null \
++model.validation_ds.quadratic_duration=null \
++model.validation_ds.bucket_duration_bins=null \
++model.validation_ds.bucket_batch_size=null \
model.validation_ds.shuffle=false
```

Use Hydra `++` for dataloader keys that are not declared in the selected YAML. This commonly applies to Lhotse-specific
validation keys such as `use_lhotse`, `use_bucketing`, `batch_duration`, `bucket_duration_bins`, and sometimes
`min_duration`/`max_duration`.

## Bucketing Policy

- For CTC/RNNT/TDT, prefer 1D duration bucketing.
- For AED/Canary, prefer 2D bucketing over duration and token count.
- Strongly prefer OOMptimizer-generated `bucket_batch_size` over manually tuning `batch_duration` and
  `quadratic_duration`.
- If all utterances are fixed length, disable bucketing and use static `batch_size`.
- For very small fine-tuning datasets around 100 hours, especially on one GPU, consider disabling bucketing and using
  fully random static `batch_size`.

1D bins and OOMptimizer:

```bash
python scripts/speech_recognition/estimate_duration_bins.py -b 30 /data/train_input_cfg.yaml
python scripts/speech_recognition/oomptimizer.py \
  --pretrained-name nvidia/parakeet-tdt-0.6b-v2 \
  --buckets '[2.0,3.1,5.6,8.4,12.0,18.0,30.0]'
```

If `max_duration` was capped to 30.0, make sure 30.0 is present as the final bucket.

2D AED/Canary bins and OOMptimizer:

```bash
python scripts/speech_recognition/estimate_duration_bins_2d.py \
  --prompt-format canary \
  --prompt "[{'role':'user','slots':{'source_lang':'en','target_lang':'en','pnc':'yes'}}]" \
  --tokenizer /data/tokenizers/spl_tokens/tokenizer.model /data/tokenizers/en/tokenizer.model \
  --langs spl_tokens en \
  --buckets 30 \
  --sub-buckets 2 \
  /data/train_input_cfg.yaml

python scripts/speech_recognition/oomptimizer.py \
  --config-path examples/asr/conf/speech_multitask/fast-conformer_aed.yaml \
  --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
  --buckets '[[3.9,30],[3.9,48],[5.0,37]]'
```

Nested `bucket_duration_bins` automatically activate 2D bucketing.

## Buffers, Memory, And RNG

- `bucket_buffer_size=10000` is a good default.
- With bucketing enabled, set `shuffle_buffer_size=10`; it is added on top of the bucket buffer.
- With bucketing disabled, set `shuffle_buffer_size=1000` to `10000`, but do not rely on it as the only source of
  randomness. Shard the data and blend datasets when applicable.
- With tarred data, larger buffers and more workers directly increase CPU memory pressure. Segfaults, CPU OOM, or
  unexplained dataloader errors often indicate CPU memory pressure.
- `seed` controls base dataloading RNGs.
- `shard_seed` controls sharded/tarred randomization. Default `shard_seed="trng"` gives different non-reproducible
  orders per run, rank, and worker, but is simpler to manage. Use `shard_seed="randomized"` with a managed `seed` when deterministic
  dataloading is needed. For 100% determinism, disable `concurrent_bucketing=false` at the cost of longer tarred data bucket prefill.

## Data Blends

Use Lhotse `input_cfg` for mixed datasets rather than concatenating manifests blindly.

Before running helper scripts such as duration-bin estimation, OOMptimizer, or data-weight estimation, validate the
input YAML shape they expect in the current checkout. Some scripts consume a list-form input config directly, while
training configs may wrap the same list under `input_cfg:`. Run a tiny dry run or inspect the script help before a long
OOMptimizer pass.

For a large generic dataset plus a smaller domain dataset, either manually upweight the domain data or estimate weights
then apply temperature reweighting. Lower temperature oversamples smaller datasets; `1.0` is neutral.

For small-domain adaptation, target-domain real data should usually dominate synthetic or heavily augmented data.
Synthetic data can help coverage, but it can also dilute the real-domain signal. Start conservatively, e.g. keep
synthetic sources below roughly 30% total weight, evaluate ablations, and only increase synthetic weight when standalone
domain WER improves. This applies especially when the synthetic data is generated from TTS, noisy augmentation, or
generic prompts rather than real target-domain audio.

```bash
python scripts/speech_recognition/estimate_data_weights.py \
  generic.yaml domain.yaml blended.yaml \
  --temperature 0.5 \
  --strategy num_hours
```

Example:

```yaml
input_cfg:
  - type: nemo
    manifest_filepath: /data/generic/manifest__OP_0..512_CL_.json
    weight: 0.7
    tags:
      domain: generic
  - type: nemo
    manifest_filepath: /data/domain/manifest__OP_0..128_CL_.json
    weight: 0.3
    tags:
      domain: target
```

For tarred data, use `type: nemo_tarred` with matching `manifest_filepath` and `tarred_audio_filepath`. Avoid mixing
tarred and non-tarred inputs in one Lhotse multi-source setup unless the current dataloader docs say the selected mode
supports it.
