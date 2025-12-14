### Offline Preference Alignment (DPO/RPO)

Code: `nemo/collections/tts/models/magpietts_preference_optimization.py`

Preference Alignment (DPO/RPO) involves the following steps
1) Create a list of text-context pairs for which we will generate preference data.
2) For each text-context pair generate multiple audios from a base TTS checkpoint and calculate metrics (CER/SSIM) for each generation.
3) Create chosen-rejected pairs from the generated audio.
4) Finetune the base TTS checkpoint on the chosen-rejected pairs.

#### 1. Create text-context pairs
We pair a list of challenging texts with context audios from our speech datasets. We add a similar number of regular transcripts our datasets such as LibriTTS paired with random context audios. We also include examples with text contexts. There are other options for generating text-context pairs.

```
python scripts/magpietts/dpo/create_text_contextpairs.py \
    --challenging_texts /Data/DPOPairsInputData/challenging_texts_nemollm.txt \
    --regular_texts_for_audiocontext /Data/DPOPairsInputData/regular_texts_for_audiocontext.txt \
    --regular_texts_for_textcontext /Data/DPOPairsInputData/regular_texts_for_textcontext.txt \
    --audio_contexts /Data/DPOPairsInputData/audio_context_list.json \
    --text_contexts /Data/DPOPairsInputData/text_context_list.txt \
    --output_manifest /Data/DPOPairsInputData/text_context_pairs_v2.json \
    --nsamples_perpair 6 ;
```
Each pair is repeated `nsamples_perpair` times which specifies how many samples we want to generate for each pair. The output manifest serves as the input for the next step.

We can also explore other options for these text-context pairs as well depending on the task.

#### 2. Generate audios for each text-context pair

Next, we can generate audios from a base TTS checkpoint using the following command. We pass the `audio_dir` as "/" since our text context pairs contains absolute paths. Model config arguments should be modified accordingly to match the base checkpoint architecture. We can run the below command on cluster to generate audios across multiple nodes. This command saves the generated audios along with the metrics for each generation in the `exp_dir`. Each generated audio file is accompanied with a `.json` file that has the CER/SSIM metrics.


```
python examples/tts/magpietts.py \
--config-name=magpietts_po_inference \
mode=test \
batch_size=64 \
+init_from_ptl_ckpt=<PATH_TO_MAGPIE_CKPT> \
exp_manager.exp_dir=<PO_EXP_DIR> \
+test_ds_meta.textcontextpairs.manifest_path=<OUTPUT_MANIFEST_FROM_STEP_1> \
+test_ds_meta.textcontextpairs.audio_dir="/" \
+test_ds_meta.textcontextpairs.feature_dir="/" \
model.model_type="decoder_context_tts"  # Change this as needed \
model.context_duration_min=5.0 \
model.context_duration_max=5.0 \
model.use_text_conditioning_encoder=true \
model.codecmodel_path=<PATH_TO_CODEC_MODEL> \
model.alignment_loss_scale=0.002 \
model.prior_scaling_factor=null \
model.load_cached_codes_if_available=false
```
#### 3. Create chosen-rejected pairs from the generations

Next, we go through the generated audio directory and create chosen-rejected pairs.

```
python scripts/magpietts/dpo/create_preference_pairs.py \
--input_manifest <OUTPUT_MANIFEST_FROM_STEP_1> \
--generated_audio_dir <PO_EXP_DIR>/MagpieTTS-PO-Infer/version_0/audio \
--group_size 6 \
--cer_threshold 0.01 \
--val_size 256 ;
```

`cer_threshold=0.01` means that filter out pairs in which the chosen CER > 0.01.

This command should save train and val manifests for DPO finetuning in the base directory of the generated_audio_dir, that is, `<PO_EXP_DIR>/MagpieTTS-PO-Infer/version_0/manifests/`

#### 4. DPO Finetuning Command

Finally, we perform DPO finetuning using the following command:

```
python examples/tts/magpietts.py \
batch_size=4 \
+init_from_ptl_ckpt=<PATH_TO_MAGPIE_CKPT> \
+mode="dpo_train" \
max_epochs=10 \
exp_manager.exp_dir=<PO_EXP_DIR> \
exp_manager.checkpoint_callback_params.always_save_nemo=false \
model.train_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
model.validation_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
+train_ds_meta.dpopreftrain.manifest_path="<PO_EXP_DIR>/MagpieTTS-PO-Infer/version_0/manifests/" \
+train_ds_meta.dpopreftrain.audio_dir="/" \
+train_ds_meta.dpopreftrain.feature_dir="/" \
+val_ds_meta.dpoprefval.manifest_path="<PO_EXP_DIR>/MagpieTTS-PO-Infer/version_0/manifests/dpo_val_manifest.json" \
+val_ds_meta.dpoprefval.audio_dir="/" \
+val_ds_meta.dpoprefval.feature_dir="/" \
+model.dpo_beta=0.01 \
+model.dpo_sft_loss_weight=0.0 \
model.model_type="decoder_context_tts"  # Change this as needed \
model.context_duration_min=5.0 \
model.context_duration_max=5.0 \
model.use_text_conditioning_encoder=true \
model.codecmodel_path=<PATH_TO_CODEC_MODEL> \
model.alignment_loss_scale=0.001 \
model.prior_scaling_factor=null \
trainer.val_check_interval=200 \
trainer.log_every_n_steps=10 \
model.optim.lr=2e-7 \
~model.optim.sched
```

Note the following overrides in the above command:

```
+mode="dpo_train" \
model.train_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
model.validation_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
```

Again, our manifest contain absolute paths so we specify `audio_dir="/"` .

### Online Preference Optimization (GRPO)

For online preference optmization, process is much simpler.

1) Create a list of text-context pairs for which we will generate preference data (just one pair for a text-context not repeated).
We'll use the same process as above, just set `nsamples_perpair 1` in the command.
```
python scripts/magpietts/dpo/create_text_contextpairs.py \
    --challenging_texts /Data/DPOPairsInputData/challenging_texts_nemollm.txt \
    --regular_texts_for_audiocontext /Data/DPOPairsInputData/regular_texts_for_audiocontext.txt \
    --regular_texts_for_textcontext /Data/DPOPairsInputData/regular_texts_for_textcontext.txt \
    --audio_contexts /Data/DPOPairsInputData/audio_context_list.json \
    --text_contexts /Data/DPOPairsInputData/text_context_list.txt \
    --output_manifest /Data/DPOPairsInputData/text_context_pairs_v2.json \
    --nsamples_perpair 1 ;
```

2. Train using GRPO

To train with GRPO, we use a similar training command as the base model training with a few modifications.

1. We start from a pretrained checkpoint supplied using `+init_from_ptl_ckpt`
2. We add `+mode="onlinepo_train"` to specify preference optimization based training.
3. Use a small batch size (bs=2) since we generate `num_generations_per_item` samples per item in the batch and the effective batch size becomes `bs*num_generations_per_item`
4. The manifest should contain absolute audio paths and the `audio_dir` is specified as "/" in the `train_ds_meta` command.
5. Use the same model specific overrides as the base model (eg. x-attn heads, is_causal, num_layers, local transformer etc).
6. Set dropout probs to 0 for all modules - This is especially important if we are not using reference free mode. KL divergence loss becomes very spiky and unstable. Set prob to 0 by `model.decoder.p_dropout=0.0`.
7. Dont use attention prior or CTC loss during GRPO.
8. Add the following GRPO specific arguments in the training command.

```
+model.grpo_beta=0.0 \ # Coeffecient for KL loss (if not using reference free mode)
+model.num_generations_per_item=12 \ # 12 samples generated for each item and we compute reward for each
+model.reference_free=true \ # Reference free means we dont use KL loss term. Only optimize for rewards
+model.inference_cfg_prob=0.0 \ # fraction of generations generated using CFG. Can set > 0.0 if we want to optimize for both CFG and non CFG modes of generation
+model.use_local_transformer_prob=0.5 \ # fraction of generations generated using Local Transformer. Set it between 0.0 and 1.0 to improve both LT outputs and non LT outputs for models with an LT
+model.inference_cfg_scale=2.5 \ # CFG scale for samples generated using CFG
+model.cer_reward_weight=0.33 \ # weightage of CER reward in the overall reward
+model.ssim_reward_weight=0.33 \ # weightage of SSIM reward in the overall reward
+model.pesq_reward_weight=0.33 \ # weightage of PESQ reward in the overall reward
+model.use_pesq=true \ # set this is true is using pesq reward
+model.reward_asr_model="whisper" \ # Use whisper only for multilingual settings, dont specify for English
model.cfg_unconditional_prob=0.0 \ # Set this to 0, we dont want want to drop out unconditional input
+model.inference_topk=2016 \ # Top-K - Not yet sure if we should use topk=80 or not. top_k 2016 just disable top_k in a way.
+model.inference_temperature=0.8 \ # Slightly higher temperature for more variety of generations in preference optimization
+model.use_kv_cache_during_online_po=true \ # Use KV caching while generating samples for GRPO
+model.loss_type="grpo" \ # can be grpo or dr_grpo. grpo works better in my experiments.
+model.scale_rewards=true \ # Whether to divide advantages by std deviation or not (set true for GRPO and false for DR_GRPO)
+model.max_decoder_steps=430 \ # Max steps for generation
```

9. We also want to validate more frequently during GRPO since each step takes longer. So we add the following args.
```
~trainer.check_val_every_n_epoch \
+trainer.val_check_interval=50 \
```

10. We use a lower learning rate and save the best checkpoints based on lowest CER on our validation set using:
```
model.optim.lr=1e-7 \
~model.optim.sched \
exp_manager.checkpoint_callback_params.monitor="val_cer_gt" \
exp_manager.checkpoint_callback_params.mode="min" \
```

11. Specify precision and gradient clipping as necessary
```
trainer.precision=32 \
+trainer.gradient_clip_val=2.5 \
```


Below is a sample training command for multilingual GRPO:

```
python examples/tts/magpietts.py \
--config-name=magpietts_multilingual_v1  #TODO(blisc) after updating yamls\
batch_size=2 \
+init_from_ptl_ckpt=<PATH_TO_MAGPIE_CKPT> \
+mode="onlinepo_train" \
+model.text_tokenizers.chartokenizer._target_=AutoTokenizer  # Change this as needed \
+model.text_tokenizers.chartokenizer.pretrained_model="google/byt5-small" # Change this as needed \
max_epochs=20 \
exp_manager.exp_dir=<GRPO_EXP_DIR> \
+exp_manager.version=0 \
exp_manager.checkpoint_callback_params.always_save_nemo=false \
+train_ds_meta.dpopreftrain.manifest_path=<TRAIN_MANIFEST_FROM_STEP_1> \
+train_ds_meta.dpopreftrain.audio_dir="/" \
+train_ds_meta.dpopreftrain.feature_dir="/" \
+train_ds_meta.dpopreftrain.tokenizer_names="[chartokenizer]"  #Change this as needed \
+val_ds_meta.dpoprefval.manifest_path=<VAL_MANIFEST_FROM_STEP_1> \
+val_ds_meta.dpoprefval.audio_dir="/" \
+val_ds_meta.dpoprefval.feature_dir="/" \
+val_ds_meta.dpoprefval.tokenizer_names="[chartokenizer]" #Change this as needed \
+model.grpo_beta=0.0 \
+model.num_generations_per_item=12 \
+model.reference_free=true \
+model.inference_cfg_prob=0.0 \
+model.inference_cfg_scale=2.5 \
+model.cer_reward_weight=0.33 \
+model.ssim_reward_weight=0.33 \
+model.pesq_reward_weight=0.33 \
+model.use_pesq=true \
+model.reward_asr_model="whisper" \
model.cfg_unconditional_prob=0.0 \
+model.inference_topk=2016 \
+model.inference_temperature=0.8 \
+model.use_kv_cache_during_online_po=true \
+model.loss_type="grpo" \
+model.scale_rewards=true \
+model.max_decoder_steps=430 \
model.model_type="decoder_context_tts" #Change this as needed \
model.context_duration_min=5.0 \
model.context_duration_max=5.0 \
model.decoder.p_dropout=0.0 \
model.encoder.p_dropout=0.0 \
model.local_transformer_type="autoregressive" #Change this as needed \
model.local_transformer_n_layers=1 #Change this as needed \
model.local_transformer_n_heads=1 #Change this as needed \
model.local_transformer_hidden_dim=256 #Change this as needed \
model.use_text_conditioning_encoder=true #Change this as needed \
model.codecmodel_path=<PATH_TO_CODEC_MODEL> \
model.alignment_loss_scale=0.0 \
model.prior_scaling_factor=null \
~trainer.check_val_every_n_epoch \
+trainer.val_check_interval=50 \
trainer.log_every_n_steps=10 \
model.optim.lr=1e-7 \
~model.optim.sched \
exp_manager.checkpoint_callback_params.monitor="val_cer_gt" \
exp_manager.checkpoint_callback_params.mode="min" \
trainer.precision=32 \
+trainer.gradient_clip_val=2.5 \
```

