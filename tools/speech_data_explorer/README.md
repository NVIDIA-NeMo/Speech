Speech Data Explorer
--------------------

[Dash](https://plotly.com/dash/)-based tool for interactive exploration of ASR/TTS datasets.

Features:
- dataset's statistics (alphabet, vocabulary, duration-based histograms)
- navigation across dataset (sorting, filtering)
- inspection of individual utterances (waveform, spectrogram, audio player)
- errors' analysis (Word Error Rate, Character Error Rate, Word Match Rate, Mean Word Accuracy, diff)
- comparison of two ASR models using interactive word-level accuracy plot
- read manifests and audio directly from S3-compatible storage (including AIStore)
- support for tarred audio datasets with efficient byte-range reads via DALI index files

## Quick Start

Install the requirements:
```
pip install -r requirements.txt
```

Run with a local manifest:
```
python data_explorer.py path_to_manifest.json
```

## S3 / AIStore Support

Speech Data Explorer can read manifests and audio files directly from S3-compatible
object storage, including NVIDIA AIStore (AIS).

### Using an S3 config file

```
python data_explorer.py s3://bucket/manifest.json --s3cfg ~/.s3cfg[default]
```

### Using AIStore with environment variables

```
export AIS_ENDPOINT=http://ais-gateway:8080
export AIS_AUTHN_TOKEN=your_token
python data_explorer.py s3://bucket/manifest.json --s3cfg AIS
```

### Sharded paths (`_OP_/_CL_` syntax)

Manifests and tar files are often split into numbered shards. Instead of listing
every shard explicitly, use the `_OP_start..end_CL_` range pattern. The tool
expands it into individual paths automatically:

```
s3://bucket/manifest__OP_0..255_CL_.json
→  s3://bucket/manifest_0.json
   s3://bucket/manifest_1.json
   ...
   s3://bucket/manifest_255.json
```

Multiple ranges in a single path produce a **cartesian product** — useful when
shards are spread across several buckets or directories:

```
s3://store_OP_1..2_CL_/audio__OP_0..1_CL_.tar
→  s3://store1/audio_0.tar
   s3://store1/audio_1.tar
   s3://store2/audio_0.tar
   s3://store2/audio_1.tar
```

### Tarred audio

When audio is stored in tar archives locally or on S3, use `--tar-base-path` to
point to the tar files. DALI index files are used automatically (if available at
`<tar_dir>/dali_index/`) for fast byte-range lookups:

```
python data_explorer.py /data/manifests/manifest.json \
    --tar-base-path /data/tarred/audio.tar
```

```
python data_explorer.py s3://bucket/manifests/manifest__OP_0..255_CL_.json \
    --tar-base-path s3://bucket/tarred/audio__OP_0..255_CL_.tar \
    --s3cfg ~/.s3cfg[default]
```

You can also specify a custom DALI index location:
```
python data_explorer.py s3://bucket/manifest.json \
    --tar-base-path s3://bucket/tarred/audio__OP_0..255_CL_.tar \
    --dali-index-base s3://bucket/tarred/dali_index/ \
    --s3cfg ~/.s3cfg[default]
```

## Comparing Two ASR Models

### Single manifest with two prediction fields

If your manifest contains two `pred_text_*` fields (e.g. `pred_text_contextnet`
and `pred_text_conformer`):

```
python data_explorer.py path_to_manifest.json \
    -nc pred_text_contextnet pred_text_conformer
```

### Two separate manifests

You can also pass two separate manifests (order-invariant). Each manifest must
contain a plain `pred_text` field, and `-nc` names the models:

```
python data_explorer.py manifest_model_A.json manifest_model_B.json \
    -nc pred_text_model_A pred_text_model_B
```

## Manifest Format

JSON manifest file should contain the following fields:
- `audio_filepath` — path to audio file (local path, or filename inside a tar archive when using `--tar-base-path`)
- `duration` — duration of the audio file in seconds
- `text` — reference transcript

Errors' analysis requires `pred_text` (ASR transcript) for all utterances.

Any additional field will be parsed and displayed in the Samples tab.

## Additional Options

| Flag | Description |
|------|-------------|
| `--vocab` | Vocabulary file to highlight OOV words |
| `--port` | Serving port (default: 8050) |
| `--estimate-audio-metrics` / `-a` | Estimate audio metrics |
| `--base-path` | Base path for relative audio paths in the manifest |
| `--tar-base-path` | Local or S3 path to tarred audio files (supports sharded `_OP_..._CL_` patterns) |
| `--dali-index-base` | Local or S3 path to DALI index directory for fast tar lookups |
| `--s3cfg` / `-s3c` | S3 config file and section, or `AIS` for AIStore env vars |
| `--force` / `-f` | Tolerate manifest entries with missing required fields |
| `-nc` / `--names_compared` | Two field names for model comparison |
| `--show_statistics` / `-shst` | Field name to show statistics for |
| `--debug` / `-d` | Enable debug mode |

![Speech Data Explorer](screenshot.png)
