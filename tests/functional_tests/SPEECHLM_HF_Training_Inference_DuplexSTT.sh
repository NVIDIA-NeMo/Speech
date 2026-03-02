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

# Run training for DuplexSTTModel (speech-to-text model)
torchrun --nproc-per-node 1 --no-python \
  coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/speechlm2/stt_duplex_train.py \
      model.pretrained_llm=/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1 \
      model.pretrained_asr=/home/TestData/speechlm/pretrained_models/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo \
      data.train_ds.input_cfg.0.shar_path=/home/TestData/speechlm/lhotse/speechlm2/train_micro \
      data.validation_ds.datasets.val_set_0.shar_path=/home/TestData/speechlm/lhotse/speechlm2/train_micro \
      trainer.devices=1 \
      trainer.max_steps=10

# Convert to HF format
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  examples/speechlm2/to_hf.py \
    class_path=nemo.collections.speechlm2.models.DuplexSTTModel \
    ckpt_path=s2s_stt_results/checkpoints/step\=10-last.ckpt \
    ckpt_config=s2s_stt_results/exp_config.yaml \
    output_dir=test_speechlm2_stt_hf_model

# Test inference on the converted HF model
torchrun --nproc-per-node 1 --no-python \
  coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/speechlm2/stt_duplex_infer.py \
      model.pretrained_llm=/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1 \
      model.pretrained_asr=/home/TestData/speechlm/pretrained_models/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo \
      data.validation_ds.datasets.val_set_0.shar_path=/home/TestData/speechlm/lhotse/speechlm2/train_micro \
      trainer.devices=1 \
      exp_manager.explicit_log_dir=s2s_stt_inference_results \
      ++model.pretrained_s2s_model=test_speechlm2_stt_hf_model
