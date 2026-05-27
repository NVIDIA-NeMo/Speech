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

Shard training manifests even when not tarred. For resume-heavy fine-tuning with an unsharded non-tarred manifest,
each restart can begin iterating from the start again. Split into shards like `manifest_0.json` ... `manifest_N.json`
with roughly 200 utterances per shard. This is less important for one-shot runs that will not be interrupted.

Strongly prefer tarred data for slow filesystems and object stores. If data is small or there is abundant fast local
SSD, non-tarred audio is fine, but still shard the manifest. Do not use tarred datasets for validation/test unless the
target script and docs explicitly support the behavior you need.

Before training, inspect duration and token-per-second distributions. Set `min_duration`, `max_duration`, `min_tps`,
and `max_tps` so they do not silently filter out a large or important part of the fine-tuning set.

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

For a large generic dataset plus a smaller domain dataset, either manually upweight the domain data or estimate weights
then apply temperature reweighting. Lower temperature oversamples smaller datasets; `1.0` is neutral.

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
