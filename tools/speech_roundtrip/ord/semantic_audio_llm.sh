#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/users/ecasanova/Nemotron-VoiceChat/Evaluations
if [[ $# -lt 1 ]]; then printf 'Usage: %s RUN_DIR [semantic-audio-llm options]\n' "$0" >&2; exit 2; fi
RUN=$(realpath "$1")
ROOT_REAL=$(realpath "$ROOT")
case "$RUN" in "$ROOT_REAL"/*) ;; *) printf 'Run must be inside the authorized Evaluations directory\n' >&2; exit 2 ;; esac
shift
test -f "$RUN/run_manifest.json"
export PYTHONPATH="$ROOT/speech_roundtrip/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OLD="$ROOT/spoken_low_replication_20261105"
exec python3 "$OLD/scripts/with_ord_key.py" "$OLD/code/audio-evals/.venv/bin/python" \
  -m speech_roundtrip semantic-audio-llm --run-dir "$RUN" "$@"
