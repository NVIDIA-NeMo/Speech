# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -x
# scripts/speechlm2/estimate_token_bins.py expects a YAML input_cfg file rather than a
# raw NeMo manifest, so write a minimal one that wraps the existing an4 test manifest.
INPUT_CFG=$(mktemp --suffix=.yaml)
trap 'rm -f "$INPUT_CFG"' EXIT
cat > "$INPUT_CFG" <<EOF
- type: lhotse_as_conversation
  manifest_filepath: /home/TestData/an4_dataset/an4_train_lang.json
EOF
# Conversation-style inputs require a prompt formatter to populate input_ids/answer_ids
# (NeMoMultimodalConversation has no plain ``tokenize()`` path) and an audio-locator-tag
# so AudioTurns get a non-null message slot.
# 1D buckets [SALM multimodal sampling]
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speechlm2/estimate_token_bins.py \
  "$INPUT_CFG" \
  --tokenizer /home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
  --prompt-format plain \
  --audio-locator-tag '<|audioplaceholder|>' \
  --buckets 5
# 2D buckets [SALM multimodal sampling, input_tokens x output_tokens]
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speechlm2/estimate_token_bins.py \
  "$INPUT_CFG" \
  --tokenizer /home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4/tokenizer.model \
  --prompt-format plain \
  --audio-locator-tag '<|audioplaceholder|>' \
  --buckets 5 \
  --sub-buckets 2
