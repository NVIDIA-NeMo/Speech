# Stage 1: Setup

## What the user needs

| Item | Notes |
| --- | --- |
| Checkpoint (`.nemo`) | The one the user has. NemotronTTS customers: the release, which ships **without a speaker encoder**. TODO: download link after the release. |
| Audio codec | The audio codec released with the checkpoint (`.nemo`). TODO: download link after the release. |
| Phoneme tokenizer | The IPA phoneme tokenizer released with the checkpoint (`.json`), built on espeak-ng 1.52 phonemes. TODO: download link after the release. |
| Text tokenizer | `nemotron_nano_30b` = HF `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (tokenizer files only; downloaded automatically). |
| NeMo Speech | Recommended: a version that includes commit `d2a01114d` (per-module learning rates, `model.optim_param_group_rules`, PR #16426; below); without it the task file's one-learning-rate alternative applies. |
| Environment | The NeMo Speech container (below) with the NeMo checkout installed in it, plus espeak-ng 1.52.0 for data preparation. |
| GPU | Default: 1 GPU (80 GB, `model.train_ds.dataset.tts_data.batch_duration=100` on the command line) per run for fine-tuning data under ~50 h. More GPUs: optional. GPUs with less memory are untested. |
| CPU, host memory | 8–16 CPU cores and 64 GB RAM per job (training uses ~25 GB, data and evaluation steps less, export ~40 GB). |

## Environment

Default to the NeMo Speech container `nvcr.io/nvidia/nemo-speech:26.07.00` (NGC); installation alternatives:
`docs/source/starthere/install.rst`. Use the `docker run` line below, not the one in the repository `README.md`: that
one mounts the checkout over `/workspace`, which hides the container's own files. Use local execution only when the
user asks for it.

The container ships its own NeMo in `/workspace/nemo` and starts in `/workspace`, where Python imports that copy.
Mount the checkout with per-module learning-rate support elsewhere, install it, and in every job leave `/workspace`
before running anything (`cd /workspace/NeMo`, or a separate work directory with full script paths such as
`/workspace/NeMo/examples/...`); otherwise Python silently imports the container's copy:

```bash
docker run --rm -it --gpus 1 --ipc=host \
  -v /path/to/NeMo:/workspace/NeMo -v /data:/data -v /path/to/cache:/cache \
  -e HF_HOME=/cache/hf -e NEMO_CACHE_DIR=/cache/nemo -e TORCH_HOME=/cache/torch \
  nvcr.io/nvidia/nemo-speech:26.07.00 bash
cd /workspace/NeMo
uv pip install --python /opt/venv/bin/python -e '.[tts]'   # the checkout's dependencies, e.g. its lhotse version
python -c "import nemo; print(nemo.__file__)"               # must print /workspace/NeMo/nemo/__init__.py
```

The install is required, not optional: the container's packages can be older than the checkout needs (for example
`lhotse`), and imports then fail. It takes seconds when repeated and rewrites `nemo_toolkit.egg-info/` in the checkout
(harmless). Check `nemo.__file__` in the same shell that runs the job (a `cd` inside a piped or
sub-shell command does not carry over). The codec-code script writes `lightning_logs/` into the current directory; run
it from a work directory, or delete that folder from the checkout afterwards.

espeak-ng 1.52.0 (data preparation only; the container has none and distribution packages are other versions). Build
it from source with the container's CMake:

```bash
git clone --depth 1 --branch 1.52.0 https://github.com/espeak-ng/espeak-ng.git /tmp/espeak-ng
# install prefix: any writable directory (here /opt/espeak-ng-1.52)
cmake -S /tmp/espeak-ng -B /tmp/espeak-ng/build -DCMAKE_INSTALL_PREFIX=/opt/espeak-ng-1.52 \
  -DUSE_ASYNC=OFF -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_KLATT=OFF -DUSE_SPEECHPLAYER=OFF
cmake --build /tmp/espeak-ng/build -j 8 && cmake --install /tmp/espeak-ng/build
export PATH=/opt/espeak-ng-1.52/bin:$PATH
espeak-ng --version                                          # must report 1.52.0
```

Build in a short path such as `/tmp`: from a long source path the espeak-ng data compiler truncates file names and
fails with "Bad vowel file". The install prefix must also give a short data path: with a long one (e.g. ~160
characters for `<prefix>/share/espeak-ng-data`) the build succeeds but every `espeak-ng` call crashes (segmentation
fault). With a prefix on deep mounted storage, link its `share` folder to a short path in each job and point espeak-ng
to it: `ln -sfn <prefix>/share /tmp/esd && export ESPEAK_DATA_PATH=/tmp/esd`. Check with
`echo test | espeak-ng -v en-us --ipa -q`.

UTMOSv2 is not in the container; without it the evaluator reports UTMOS as NaN, which no check here uses.

On a cluster, run the same image and the same commands through the cluster's container runtime. If each job starts
a fresh container (nothing outside mounted storage persists), repeat the install at the start of every job (and the
espeak-ng build, ~1 min, in data-preparation jobs, the only ones that need it), and keep caches on mounted storage
(below).

On a single machine with Docker:
- host prerequisites: the user in the `docker` group and the NVIDIA container toolkit (`docker run --rm --gpus all
  ubuntu nvidia-smi` must work);
- the container runs as root (`--user` does not work because the install writes to `/opt/venv`), so files it writes to
  mounted directories, the checkout included (`nemo_toolkit.egg-info`, `__pycache__`), are root-owned: give them back
  at the end of every job, failed ones too, symlinks included (`chown -h`);
- long steps (training ~1 h, evaluations) must survive a dropped SSH session and the agent's command time limits: run
  each step in its own detached container and follow its log;
- install espeak-ng to a prefix on a mounted directory (e.g. `/work/opt/espeak-ng-1.52`) so it is built once, and
  extend `PATH` / `LD_LIBRARY_PATH` in each job; do not replace them (`-e PATH=...` drops the image's own entries);
- `--gpus 1` does not pick a free GPU: on a machine with several GPUs give each job its own (`--gpus '"device=<i>"'`, with the inner double quotes: some Docker versions reject the unquoted `--gpus device=<i>`).

A launcher that does all of this, one call per step (`<step>.sh` is the step's commands, run from `/work`; the third
argument is the GPU index, or `cpu` for steps that need no GPU):

```bash
#!/bin/bash
# job.sh <name> <step>.sh [<gpu index> | cpu]   ->  follow with: tail -f <work>/logs/<name>.log
set -euo pipefail
NAME=$1; STEP=$2; DEV=${3:-0}; U=$(id -u); G=$(id -g)
GPUFLAG=(); [ "$DEV" = cpu ] || GPUFLAG=(--gpus "\"device=$DEV\"")
NEMO=/path/to/NeMo; WORK=/path/to/work; DATA=/path/to/data
mkdir -p "$WORK/logs"
docker run -d --rm --name "$NAME" ${GPUFLAG[@]+"${GPUFLAG[@]}"} --ipc=host \
  -v "$NEMO":/workspace/NeMo -v "$WORK":/work -v "$DATA":/data:ro \
  -e HF_HOME=/work/cache/hf -e NEMO_CACHE_DIR=/work/cache/nemo -e TORCH_HOME=/work/cache/torch \
  nvcr.io/nvidia/nemo-speech:26.07.00 bash -c "
    trap 'find /work /workspace/NeMo -user 0 -exec chown -h $U:$G {} + 2>/dev/null' EXIT
    exec > /work/logs/$NAME.log 2>&1
    export PATH=/work/opt/espeak-ng-1.52/bin:\$PATH LD_LIBRARY_PATH=/work/opt/espeak-ng-1.52/lib:\$LD_LIBRARY_PATH
    cd /workspace/NeMo && uv pip install --python /opt/venv/bin/python -e '.[tts]' || exit 1
    cd /work && python -c 'import nemo; print(nemo.__file__)' | grep -qx /workspace/NeMo/nemo/__init__.py || exit 1
    bash /work/$STEP; rc=\$?; echo JOB_EXIT=\$rc; exit \$rc
  "
```

On a SLURM cluster with a container plugin (here pyxis), the same step as a batch job; CPU-only steps (data
preparation, export) can drop the GPU request:

```bash
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 --mem=96G -t 02:00:00
#SBATCH -J <name> -o <work>/logs/%x-%j.out
srun --no-container-mount-home --container-image=<nemo-speech image or .sqsh> \
  --container-mounts=/path/to/NeMo:/workspace/NeMo,/path/to/work:/work,/path/to/data:/data \
  bash -c 'export HF_HOME=/work/cache/hf NEMO_CACHE_DIR=/work/cache/nemo TORCH_HOME=/work/cache/torch
    cd /workspace/NeMo && uv pip install --python /opt/venv/bin/python -e ".[tts]" || exit 1
    cd /work && python -c "import nemo; print(nemo.__file__)" | grep -qx /workspace/NeMo/nemo/__init__.py || exit 1
    bash /work/<step>.sh'
```

Container runtimes that run as root (pyxis included, depending on the cluster) also leave root-owned files in the
mounted directories: check with `find <work> -user 0` after the first job and, if any, give them back at the end of
every job as in the Docker launcher.

Jobs that start at the same time share the checkout and the caches:
- every container still needs its own install, but the editable install also writes into the shared checkout
  (`nemo_toolkit.egg-info`): parallel installs can fail or leave it half-written. Run the install under a lock on
  mounted storage (`flock /work/install.lock uv pip install ...`), or start the jobs a few minutes apart;
- the first download of a model into a shared cache can collide: download every model once (one job, or the step
  that needs it) before starting jobs in parallel.
## Per-module learning rates (default)

Tasks that train modules with different learning rates (new voice: new speaker encoder vs the rest) use
`model.optim_param_group_rules` (`ModelPT`, NeMo commit `d2a01114d`, PR #16426): a rule selects parameters by name
pattern (`param_names`, e.g. `[speaker_encoder.*]`) and sets their options (`lr`). NeMo versions
without this support silently ignore the key, so every parameter trains at `model.optim.lr`; with the new-voices task's values
that is the task file's one-learning-rate alternative (5e-6 for the whole model): tell the user and record it in the
ledger.

Verify after the first training start; the optimizer log must show two groups:

```
Optimizer config = AdamW (
    Parameter Group 0
        ...
        lr: 5e-06
        ...
    Parameter Group 1
        ...
        lr: 5e-05
```

Only `Parameter Group 0` means the NeMo version lacks the support or the override did not reach the model (expected
only with the one-learning-rate alternative).

## Fine-tuning config

`examples/tts/conf/magpietts/easy_magpietts_finetuning.yaml` (`references/new-voices-supported-language/training.md`). Set the codec and
phoneme-tokenizer paths on the command line. TODO: the config is added with the model release.

## Model downloads and caches

By default, models download automatically on first use (Hugging Face and NGC). Keep the caches on persistent storage
so they are downloaded once: `HF_HOME`, `NEMO_CACHE_DIR` and `TORCH_HOME` (inside a container the default `~/.cache` is
lost when the container exits).

Downloaded automatically, by step:

| Step | Models |
| --- | --- |
| Codec restore (codes, training, inference) | `microsoft/wavlm-base-plus` (loaded by the codec's discriminator) |
| Training and inference | the text tokenizer `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (tokenizer files only, not the model weights) |
| Evaluation and the optional data check (ASR) | per language, `references/new-voices-supported-language/evaluation.md` (e.g. `nvidia/parakeet-tdt-1.1b`, `openai/whisper-large-v3`) |
| Reference clips and evaluation (speaker similarity) | `titanet_large`; the evaluator also loads `titanet_small` |
| Evaluation, English only (cut-off check) | `facebook/wav2vec2-base-960h` |
| Training with `model.run_val_inference=true` only | `nvidia/parakeet-ctc-0.6b` (or `openai/whisper-large-v3` with `model.use_multilingual_asr=true`), `titanet_large` |

**Machines without internet:** three things need it: the install (`uv pip install` downloads packages), the
espeak-ng build (clones from GitHub) and the model downloads. On a machine with internet: run the install once in the
container with the checkout mounted at `/workspace/NeMo` and save that container as an image (`docker commit`, then
`docker save` / `docker load`); build espeak-ng into a directory you copy over; run each step once into the cache
directories (or pre-download the models above) and copy them. On the offline machine: use the saved image, keep
mounting the checkout at `/workspace/NeMo`, drop the install line from the jobs and set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Preflight

- Inside the container: `nemo.__file__` points to the mounted checkout; `espeak-ng --version` reports 1.52.0.
- `nvidia-smi`: one free GPU; disk: ~5 GB per kept checkpoint (≈ 3 kept per run) + data + generated audio.
- No separate load of the release is needed: the restored-tensor count is printed in the first minutes of training
  (`references/new-voices-supported-language/training.md`). Where GPU jobs wait in a queue, first run the training command on CPU with Hydra's
  `--cfg job` appended: it prints the composed config without training and catches mistyped overrides.
- The evaluator runs on a GPU only.
