# Copyright (c) 2026, NVIDIA CORPORATION.
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
HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/magpietts.py \
    --config-name magpietts \
    name="MagpieTTS-OnlineCFGDistillation" \
    +mode="online_cfg_distillation_train" \
    +init_from_nemo_model="/home/TestData/tts/2602_MagpieTTS/feb26_Magpie-TTS-ML-V1--val_cer_gt\=0.3258-step\=1000.nemo" \
    +model.teacher_model_path="/home/TestData/tts/2602_MagpieTTS/feb26_Magpie-TTS-ML-V1--val_cer_gt\=0.3258-step\=1000.nemo" \
    model.codecmodel_path="/home/TestData/tts/21fps_causal_codecmodel.nemo" \
    model.prior_scaling_factor=null \
    model.local_transformer_type="autoregressive" \
    model.local_transformer_n_layers=1 \
    model.local_transformer_n_heads=1 \
    model.local_transformer_hidden_dim=256 \
    +train_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_train_context_v1.json" \
    +train_ds_meta.an4.audio_dir="/" \
    +train_ds_meta.an4.tokenizer_names="[english_phoneme]" \
    +train_ds_meta.an4.feature_dir=null \
    +val_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_val_context_v1.json" \
    +val_ds_meta.an4.audio_dir="/" \
    +val_ds_meta.an4.tokenizer_names="[english_phoneme]" \
    +val_ds_meta.an4.feature_dir=null \
    +model.text_tokenizers.spanish_phoneme._target_=nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer \
    +model.text_tokenizers.spanish_phoneme.locale=es-ES \
    +model.text_tokenizers.spanish_phoneme.punct=true \
    +model.text_tokenizers.spanish_phoneme.apostrophe=true \
    +model.text_tokenizers.spanish_phoneme.pad_with_space=true \
    +model.text_tokenizers.spanish_phoneme.g2p._target_=nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p \
    +model.text_tokenizers.spanish_phoneme.g2p.locale=es-ES \
    +model.text_tokenizers.spanish_phoneme.g2p.phoneme_dict="scripts/tts_dataset_files/es_ES/es_ES_nv230301.dict" \
    +model.text_tokenizers.spanish_phoneme.g2p.phoneme_probability=0.8 \
    +model.text_tokenizers.spanish_phoneme.g2p.use_chars=true \
    +model.text_tokenizers.spanish_phoneme.g2p.use_stresses=true \
    +model.text_tokenizers.german_phoneme._target_=nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer \
    +model.text_tokenizers.german_phoneme.locale=de-DE \
    +model.text_tokenizers.german_phoneme.punct=true \
    +model.text_tokenizers.german_phoneme.apostrophe=true \
    +model.text_tokenizers.german_phoneme.pad_with_space=true \
    +model.text_tokenizers.german_phoneme.g2p._target_=nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p \
    +model.text_tokenizers.german_phoneme.g2p.locale=de-DE \
    +model.text_tokenizers.german_phoneme.g2p.phoneme_dict="scripts/tts_dataset_files/de/de_nv230119.dict" \
    +model.text_tokenizers.german_phoneme.g2p.phoneme_probability=0.8 \
    +model.text_tokenizers.german_phoneme.g2p.use_chars=true \
    +model.text_tokenizers.german_phoneme.g2p.use_stresses=true \
    +model.text_tokenizers.german_phoneme.g2p.grapheme_case="mixed" \
    +model.text_tokenizers.german_phoneme.g2p.grapheme_prefix="'#'" \
    +model.text_tokenizers.mandarin_phoneme._target_=nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.ChinesePhonemesTokenizer \
    +model.text_tokenizers.mandarin_phoneme.punct=true \
    +model.text_tokenizers.mandarin_phoneme.apostrophe=true \
    +model.text_tokenizers.mandarin_phoneme.pad_with_space=true \
    +model.text_tokenizers.mandarin_phoneme.g2p._target_=nemo.collections.tts.g2p.models.zh_cn_pinyin.ChineseG2p \
    +model.text_tokenizers.mandarin_phoneme.g2p.phoneme_dict="scripts/tts_dataset_files/zh/36finals/ipa_dict_nv23.05.txt" \
    +model.text_tokenizers.mandarin_phoneme.g2p.word_segmenter="jieba" \
    +model.text_tokenizers.mandarin_phoneme.g2p.phoneme_prefix="" \
    +model.text_tokenizers.mandarin_phoneme.g2p.phoneme_case="lower" \
    +model.text_tokenizers.mandarin_phoneme.g2p.tone_prefix="'#'" \
    +model.text_tokenizers.mandarin_phoneme.g2p.ascii_letter_prefix="" \
    +model.text_tokenizers.mandarin_phoneme.g2p.ascii_letter_case="upper" \
    +model.text_tokenizers.japanese_phoneme._target_=nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.JapanesePhonemeTokenizer \
    +model.text_tokenizers.japanese_phoneme.punct=true \
    +model.text_tokenizers.japanese_phoneme.apostrophe=false \
    +model.text_tokenizers.japanese_phoneme.pad_with_space=true \
    +model.text_tokenizers.japanese_phoneme.g2p._target_=nemo.collections.tts.g2p.models.ja_jp_ipa.JapaneseKatakanaAccentG2p \
    +model.text_tokenizers.japanese_phoneme.g2p.ascii_letter_prefix="" \
    +model.text_tokenizers.japanese_phoneme.g2p.ascii_letter_case="upper" \
    +model.text_tokenizers.french_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.french_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.hindi_chartokenizer._target_=nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.HindiCharsTokenizer \
    +model.text_tokenizers.hindi_chartokenizer.charset_version=1 \
    +model.text_tokenizers.hindi_chartokenizer.punct=true \
    +model.text_tokenizers.hindi_chartokenizer.punct_version=1 \
    +model.text_tokenizers.hindi_chartokenizer.apostrophe=true \
    +model.text_tokenizers.hindi_chartokenizer.pad_with_space=true \
    +model.text_tokenizers.italian_phoneme._target_=AutoTokenizer \
    +model.text_tokenizers.italian_phoneme.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.vietnamese_phoneme._target_=AutoTokenizer \
    +model.text_tokenizers.vietnamese_phoneme.pretrained_model="google/byt5-small" \
    trainer.devices="[0]" \
    max_epochs=1 \
    batch_size=4 \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    trainer.strategy=auto \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.num_workers=0 \
    ~trainer.check_val_every_n_epoch
