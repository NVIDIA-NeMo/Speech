# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
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
set -e

# Fine-tune from Hugging Face `datasets` input. CI runs with HF_HUB_OFFLINE=1 and its image has no FFmpeg, which
# `datasets` needs to decode `Audio` columns. Build a small local Parquet dataset from AN4 that stores the raw samples.
HF_DATA_DIR=/tmp/an4_hf_parquet
# Keep the datasets cache out of the shared, mounted HF_HOME.
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
python - "$HF_DATA_DIR" <<'PY'
import json
import os
import sys

import datasets
import soundfile

os.makedirs(sys.argv[1], exist_ok=True)
for split, manifest in [("train", "an4_train.json"), ("validation", "an4_val.json")]:
    rows = {"audio": [], "text": []}
    with open(f"/home/TestData/an4_dataset/{manifest}") as f:
        for _, line in zip(range(16), f):
            entry = json.loads(line)
            samples, sampling_rate = soundfile.read(entry["audio_filepath"], dtype="float32")
            rows["audio"].append({"array": samples.tolist(), "sampling_rate": sampling_rate})
            rows["text"].append(entry["text"])
    datasets.Dataset.from_dict(rows).to_parquet(os.path.join(sys.argv[1], f"{split}.parquet"))
PY

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/speech_to_text_finetune.py \
    --config-path="conf/asr_finetune" --config-name="speech_to_text_hf_finetune" \
    ~model.train_ds.hf_data_cfg \
    model.train_ds.num_workers=1 \
    model.train_ds.batch_size=2 model.validation_ds.batch_size=2 \
    model.train_ds.streaming=true \
    +model.train_ds.hf_data_cfg.path="parquet" \
    +model.train_ds.hf_data_cfg.data_files="${HF_DATA_DIR}/train.parquet" \
    +model.train_ds.hf_data_cfg.split="train" \
    +model.train_ds.hf_data_cfg.streaming=true \
    ~model.validation_ds.hf_data_cfg \
    model.validation_ds.streaming=true \
    +model.validation_ds.hf_data_cfg.path="parquet" \
    +model.validation_ds.hf_data_cfg.data_files="${HF_DATA_DIR}/validation.parquet" \
    +model.validation_ds.hf_data_cfg.split="train" \
    +model.validation_ds.hf_data_cfg.streaming=true \
    ~model.test_ds \
    init_from_nemo_model=/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo \
    model.tokenizer.update_tokenizer=False \
    model.optim.sched.warmup_steps=0 \
    +model.optim.sched.max_steps=3 \
    trainer.max_epochs=null \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_finetuning_results
