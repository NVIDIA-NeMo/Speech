# Stage 1: Setup And Checkpoint Selection

Default to the NeMo release container. Use local execution only when the user explicitly asks for it.

Check `README.md`, `docker/Dockerfile.speech`, and the NGC catalog before quoting container tags. The container avoids
most local CUDA, torch, audio backend, `sox`, `ffmpeg`, and `libsndfile` issues.

When running repo scripts from a mounted source checkout inside a container, ensure Python imports the mounted checkout,
not an unrelated package already installed in the image:

```bash
export PYTHONPATH=/workspace/NeMo
python -c "import nemo.collections.asr as asr; print(asr.__file__)"
```

If the source checkout must be installed in the container, prefer `pip install -e '.[asr]'`.

For Hugging Face models or datasets that require authentication, verify access before launching long jobs. Containers
often do not include `huggingface-cli`; mounting a populated `HF_HOME` or token/cache directory is usually enough:

```bash
docker run -e HF_HOME=/tmp/hf_home -v /host/hf_home:/tmp/hf_home ...
```

For explicit local setup, verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import nemo.collections.asr as asr; print(asr.__file__)"
nvidia-smi
```

For a source checkout local install, prefer:

```bash
pip install -e '.[asr]'
```

If CUDA, torch, audio backend, or dependency issues appear, steer back to the container.

## Checkpoint Candidates

Use the user's constraints first: language, streaming vs offline, latency, memory budget, punctuation/capitalization,
translation, speaker overlap, CTC alignments, and deployment target.

Good candidates from local docs:

- `nvidia/parakeet-tdt-0.6b-v3` or `nvidia/parakeet-tdt-0.6b-v2`: general Parakeet/TDT fine-tuning. Check the
  model card for exact language and punctuation support.
- `nvidia/parakeet-tdt_ctc-110m`: compact hybrid candidate.
- `nvidia/parakeet-ctc-0.6b` or `nvidia/parakeet-ctc-1.1b`: CTC when simple non-autoregressive decoding, alignments,
  or vocabulary replacement matter.
- `nvidia/nemotron-speech-streaming-en-0.6b`: real-time English streaming.
- `nvidia/multitalker-parakeet-streaming-0.6b-v1`: streaming multi-speaker input.
- `nvidia/canary-1b-v2`: AED/Canary for multilingual ASR, speech translation, and prompt-controlled behavior.
- `nvidia/canary-1b-flash` or `nvidia/canary-180m-flash`: faster Canary-family models.

Always check current `README.md`, `docs/source/asr/featured_models.rst`, `docs/source/asr/asr_checkpoints.rst`, and
the model card or HF collection before finalizing the checkpoint list.
