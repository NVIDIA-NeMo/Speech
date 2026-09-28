# Train an ASR model on Nebius Serverless Jobs

This tutorial trains a small character-level Conformer CTC model on
[AN4](http://www.speech.cs.cmu.edu/databases/an4/) using one GPU. A
[Nebius Serverless Job](https://docs.nebius.com/serverless/jobs/manage) runs the
container, saves a `.nemo` artifact to Object Storage, and releases the job's
compute resources when execution ends. No cluster or multi-node setup is needed.

The default **50 optimizer steps** exercise the pipeline, not speech-recognition
quality. The model starts from random weights. Expect poor transcriptions and
high word error rate (WER). The AN4 test split is also used for smoke validation;
this is not an independent model-selection benchmark. Training a useful model
requires a suitable dataset, schedule, and separate validation/test splits.

## Prerequisites

- An authenticated [Nebius CLI](https://docs.nebius.com/cli/install) with
  `nebius ai job run` (commands checked with **0.12.279**). Use
  `nebius profile create` to configure your project. The run interface is beta.
- The [Jobs permissions](https://docs.nebius.com/serverless/jobs/manage#prerequisites),
  GPU quota and capacity for `gpu-h100-sxm` / `1gpu-16vcpu-200gb`, and a project
  subnet with outbound access to NVIDIA NGC and the public AN4 download on S3.
  If your project has multiple subnets, pass `--subnet-id YOUR_SUBNET_ID`.
- An Object Storage bucket in your project and region. Configure an AWS CLI
  profile named `nebius` using the
  [Object Storage instructions](https://docs.nebius.com/object-storage/interfaces/aws-cli)
  for downloading results. These S3 access keys are distinct from Nebius CLI
  credentials. Keep credentials outside this directory and source control.
- Acceptance of the [NeMo Speech container terms](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-speech)
  and the AN4 dataset's terms. The example uses public speech data only.

[`nebius.yaml`](nebius.yaml) pins the Linux amd64 manifest of the upstream
[supported NeMo Speech 26.07.00 image](../../../docs/source/starthere/install.rst).
It supplies NeMo, PyTorch, Lightning and audio dependencies without a runtime
`pip install`. It needs a host NVIDIA driver compatible with its CUDA runtime;
verify this on your selected platform before using the recipe for longer runs.
The 250 GiB container disk holds the image and temporary dataset/model files.

**Validation status:** CPU dataset preparation, unit tests and CLI packaging have
been checked. Training, GPU/driver compatibility, remote storage persistence and
reload on Serverless have not yet been validated end to end. No quality or run
time result is claimed.

## Dataset and model configuration

From a checkout of this repository:

```bash
cd tutorials/cloud/nebius
```

[`train.py`](train.py) downloads the approximately 61 MiB AN4 SPHERE archive inside
the job and verifies its fixed SHA-256 before extraction. It converts audio to
mono 16 kHz PCM WAV with SoundFile, following the transcript and path conventions
in `nemo.utils.notebook_utils.download_an4`. This avoids a system SoX dependency.
It writes 948 training and 130 test records as JSON Lines, for example:

```json
{"audio_filepath": "/tmp/nemo-an4-.../an4/wav/an4_clstk/fash/cen4-fash-b.wav", "duration": 2.0, "text": "example transcript"}
```

The line above illustrates the schema, not a measured recording. Paths and
actual durations are generated from the extracted audio. Manifests reference
container-local absolute paths; no workstation paths are sent to training.

[`conformer.yaml`](conformer.yaml) adapts NeMo's
[Conformer CTC character configuration](../../../examples/asr/conf/conformer/conformer_ctc_char.yaml)
to four encoder layers, dimension 128, four attention heads, batch size eight
and AdamW at `0.001`. The script fills in manifest paths, seeds training with 42,
and uses one GPU at full precision. A seed does not guarantee bitwise-identical
GPU results. Data-loader workers are disabled to keep this example small.

## Preview and submit

Check the context **before every submission**. Only `train.py` and
`conformer.yaml` should be listed; dependencies and data are fetched by the
container. The CLI limits compressed source context to 64 KiB. `.nebiusignore`
excludes documentation, tests and common local files; it is not a general
secret scanner. Keep unrelated files out of this directory.

```bash
nebius ai job run train.py --show-context
```

Set the bucket **name** and your region, then launch one bounded run:

```bash
export OUTPUT_BUCKET='YOUR_EXISTING_BUCKET_NAME'
export NEBIUS_REGION='YOUR_PROJECT_REGION'
export JOB_NAME="nemo-an4-$(date -u +%Y%m%d-%H%M%S)"
nebius ai job run train.py \
  --name "$JOB_NAME" \
  --output "s3://$OUTPUT_BUCKET" \
  --timeout 1h \
  -- --max-steps 50
```

The command streams logs and waits for a terminal state. Save the job ID printed
to stderr. The script exits after the bounded training, reload and evaluation;
`--max-steps` accepts 1–1000. The provider timeout is one hour, the CLI's minimum,
and bounds the job if it stalls. Startup and teardown contribute to cost.
Do not increase the cap until the smoke run passes.

For a separate terminal, substitute the returned ID:

```bash
export JOB_ID='YOUR_JOB_ID'
nebius ai job get "$JOB_ID" --format json
nebius ai job logs "$JOB_ID" --follow
```

A successful run must reach `COMPLETED`. CLI exit code zero indicates completion;
nonzero codes indicate failure, infrastructure errors, cancellation, or local
input errors. Consult `nebius ai job run --help` for the complete mapping.
**Ctrl+C stops following logs but leaves the job running.** Cancel explicitly:

```bash
nebius ai job cancel "$JOB_ID"
nebius ai job get "$JOB_ID" --format json
```

If the submission response is lost, find the unique name with
`nebius ai job list` before retrying to avoid launching a duplicate job.

## Retrieve and verify the model

The CLI mounts the bucket's `runs/JOB_NAME/` prefix and sets `NEBIUS_OUTPUT_DIR`.
The script serializes on the local container disk, reloads `model.nemo`, evaluates
all 130 AN4 test utterances, and transcribes one test recording. Only after these
steps succeed does it copy the closed files to the output mount under `asr/`:

- `model.nemo`: model configuration and weights, loadable with `restore_from`.
- `metrics.json`: test loss and WER from the reloaded model.
- `sample.wav` and `transcription.json`: test recording, reference and prediction.
- `config.yaml` and `run.json`: model configuration, seed, actual optimizer steps
  and dataset checksum. Dataset paths in the saved config are job-local.
- `COMPLETE.json`: SHA-256 hashes, written after the other files are copied.

The `.nemo` file is for model reload/inference, not a Lightning optimizer-state
checkpoint for exact training resumption. No partial checkpoint is promised on
cancellation or timeout. A completion marker alone does not prove that the
storage mount flushed every object; download and verify the hashes:

```bash
aws --profile nebius --endpoint-url "https://storage.$NEBIUS_REGION.nebius.cloud" \
  s3 cp "s3://$OUTPUT_BUCKET/runs/$JOB_NAME/asr/" "outputs/$JOB_NAME/" --recursive
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path('outputs') / os.environ['JOB_NAME']
hashes = json.loads((root / 'COMPLETE.json').read_text())
for name, expected in hashes.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
print('Verified:', ', '.join(hashes))
print((root / 'metrics.json').read_text())
print((root / 'transcription.json').read_text())
PY
```

To independently reload the downloaded model, run this in the same pinned image
on a CUDA-capable machine, with the downloaded directory mounted at `/results`:

```python
from nemo.collections.asr.models import EncDecCTCModel

model = EncDecCTCModel.restore_from('/results/model.nemo')
model.eval()
print(model.transcribe(['/results/sample.wav'], batch_size=1, return_hypotheses=False))
```

For example, from this tutorial directory (Docker with NVIDIA Container Toolkit):

```bash
export NEMO_IMAGE='nvcr.io/nvidia/nemo-speech@sha256:8b5b7e616aba612d26ab1d2d2b654aec0336d7e3c5f10d06c0a14a9a6e44cb89'
docker run --rm --gpus all \
  -v "$PWD/outputs/$JOB_NAME:/results:ro" "$NEMO_IMAGE" \
  python -c "from nemo.collections.asr.models import EncDecCTCModel; m = EncDecCTCModel.restore_from('/results/model.nemo'); m.eval(); print(m.transcribe(['/results/sample.wav'], batch_size=1, return_hypotheses=False))"
```

## Failures and cleanup

For quota, capacity, registry or startup failures, inspect job status and the
[Jobs troubleshooting guide](https://docs.nebius.com/serverless/jobs/failure).
For dataset checksum, download, CUDA, training, evaluation or output-copy errors,
the Python exception causes a nonzero container exit. A failed run can leave
partial output files: require both `COMPLETED` and successful hash verification.
Use a new job name for retries; existing output files are never overwritten by
the script. Do not treat missing artifacts as successful training.

Cancel an active job, wait for terminal status and save any needed logs/results.
Then remove its Job record:

```bash
nebius ai job delete "$JOB_ID"
```

Job-managed compute is separate from retained Object Storage. Verify resource
cleanup in the console, especially after a timeout or infrastructure failure.
Delete only this run's output prefix after confirming your downloaded copy:

```bash
aws --profile nebius --endpoint-url "https://storage.$NEBIUS_REGION.nebius.cloud" \
  s3 rm "s3://$OUTPUT_BUCKET/runs/$JOB_NAME/" --recursive
```

The bucket remains. Review the variables before deletion; do not delete a shared
bucket or another run's prefix.

## Local checks without a GPU

The following tests do not submit jobs or import NeMo. Use Python 3.11 or newer:

```bash
python3 -m venv /tmp/nemo-tutorial-check
/tmp/nemo-tutorial-check/bin/pip install soundfile==0.13.1
/tmp/nemo-tutorial-check/bin/python -m unittest discover -s . -p test_train.py -v
nebius ai job run train.py --show-context
```

Live acceptance additionally requires one 50-step run on the selected GPU,
`COMPLETED` status, artifact download and hash verification, reloaded-model
metrics and transcription, and verification of compute cleanup. CPU tests do
not establish those results.
