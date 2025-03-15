# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/nlp/machine_translation/megatron_nmt_training.py \
    trainer.devices=2 \
    trainer.accelerator=gpu \
    trainer.log_every_n_steps=1 \
    trainer.val_check_interval=10 \
    +trainer.limit_val_batches=2 \
    trainer.accumulate_grad_batches=1 \
    trainer.max_steps=10 \
    trainer.precision=16 \
    trainer.gradient_clip_val=1.0 \
    exp_manager.exp_dir=examples/nlp/machine_translation/megatron_nmt_results \
    model.tensor_model_parallel_size=2 \
    model.seq_length=128 \
    model.encoder.num_layers=4 \
    model.encoder.hidden_size=64 \
    model.encoder.num_attention_heads=8 \
    model.encoder.activation="swiglu" \
    model.encoder.masked_softmax_fusion=False \
    model.encoder.bias_activation_fusion=False \
    model.encoder.activations_checkpoint_method="block" \
    model.encoder.activations_checkpoint_num_layers=1 \
    model.decoder.num_layers=2 \
    model.decoder.hidden_size=64 \
    model.decoder.num_attention_heads=8 \
    model.decoder.activation="swiglu" \
    model.decoder.masked_softmax_fusion=False \
    model.decoder.bias_activation_fusion=False \
    model.decoder.activations_checkpoint_method="block" \
    model.decoder.activations_checkpoint_num_layers=1 \
    model.micro_batch_size=2 \
    model.global_batch_size=4 \
    model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    model.train_ds.num_workers=1 \
    model.validation_ds.num_workers=1 \
    ~model.test_ds \
    model.train_ds.dataset_type=text_memmap \
    model.encoder_tokenizer.library=sentencepiece \
    model.encoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
    model.decoder_tokenizer.library=sentencepiece \
    model.decoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model
# Change val_check_interval to 1 for resume as the len(dataloder) is 1 due to max_steps being the same as that of training and Lightning 2.0 raises an error
# if val_check_interval > len(dataloder: https://github.com/Lightning-AI/lightning/blob/2.0.6/src/lightning/pytorch/loops/fit_loop.py#L259 at the beginning of fit_loop.run()
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/nlp/machine_translation/megatron_nmt_training.py \
    trainer.devices=2 \
    trainer.accelerator=gpu \
    trainer.log_every_n_steps=1 \
    trainer.val_check_interval=1 \
    +trainer.limit_val_batches=2 \
    trainer.accumulate_grad_batches=1 \
    trainer.max_steps=10 \
    trainer.precision=16 \
    trainer.gradient_clip_val=1.0 \
    exp_manager.exp_dir=examples/nlp/machine_translation/megatron_nmt_results \
    model.tensor_model_parallel_size=2 \
    model.seq_length=128 \
    model.encoder.num_layers=4 \
    model.encoder.hidden_size=64 \
    model.encoder.num_attention_heads=8 \
    model.encoder.activation="swiglu" \
    model.encoder.masked_softmax_fusion=False \
    model.encoder.bias_activation_fusion=False \
    model.encoder.activations_checkpoint_method="block" \
    model.encoder.activations_checkpoint_num_layers=1 \
    model.decoder.num_layers=2 \
    model.decoder.hidden_size=64 \
    model.decoder.num_attention_heads=8 \
    model.decoder.activation="swiglu" \
    model.decoder.masked_softmax_fusion=False \
    model.decoder.bias_activation_fusion=False \
    model.decoder.activations_checkpoint_method="block" \
    model.decoder.activations_checkpoint_num_layers=1 \
    model.micro_batch_size=2 \
    model.global_batch_size=4 \
    model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    model.train_ds.num_workers=1 \
    model.validation_ds.num_workers=1 \
    ~model.test_ds \
    model.train_ds.dataset_type=text_memmap \
    model.encoder_tokenizer.library=sentencepiece \
    model.encoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
    model.decoder_tokenizer.library=sentencepiece \
    model.decoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model
