# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

checkpoint=${1:-/home/TestData/asr/stt_en_conformer_ctc_small-1.6.0.nemo}
url=https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.6.0/files/stt_en_conformer_ctc_small.nemo

is_checkpoint() {
    local members
    members=$(tar -tf "$1" 2>/dev/null) || return 1
    grep -Eq '^(\./)?model_config\.yaml$' <<< "$members" &&
        grep -Eq '^(\./)?model_weights\.ckpt$' <<< "$members"
}

if [[ -s "$checkpoint" ]] && is_checkpoint "$checkpoint"; then
    echo "Reusing ASR adapter checkpoint: $checkpoint"
    exit 0
fi

mkdir -p "$(dirname "$checkpoint")"
temporary=$(mktemp "$checkpoint.tmp.XXXXXX")
trap 'rm -f "$temporary"' EXIT

echo "Pre-staging ASR adapter checkpoint: $checkpoint"
curl --fail --location --silent --show-error \
    --connect-timeout 15 --max-time 120 --retry 2 --retry-delay 2 \
    --output "$temporary" "$url"
if ! is_checkpoint "$temporary"; then
    echo "Downloaded ASR adapter checkpoint is not a valid NeMo archive" >&2
    exit 1
fi
mv -fT -- "$temporary" "$checkpoint"
