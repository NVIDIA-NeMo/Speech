# Nemotron-TTS V2 online post-training

Set `mode=onlinepo_train` and `model.model_type=v2` in the EasyMagpie entrypoint to
train `NemotronTTSModelOnlinePO`. Reuse the architecture and tokenizer configuration
from the V2 checkpoint, and set `pretrained_model` to its `.nemo` archive. Resume a
post-training run through the existing experiment-manager checkpoint mechanism.

The V2 model reuses the existing GRPO/DR-GRPO rewards and chunked teacher-forcing
trainer. Policy likelihoods come from the autoregressive acoustic-code predictor.
Teacher forcing retains sampled terminal audio actions and the generated phoneme
trajectory. Sampled phonemes receive an additional policy loss controlled by
`model.aux_phoneme_po_loss_weight` (default `0.2`).

For initial runs, use these settings together with your V2 configuration:

```yaml
mode: onlinepo_train
model:
  model_type: v2
  loss_type: grpo
  reference_free: true
  n_generations_per_item: 16
  batch_size_for_chunked_tf: 1
  inference_cfg_prob: 0.0
  inference_use_cfg_in_val: false
  inference_temperature: 0.7
  inference_topk: 80
  inference_phoneme_sampling_method: sample
  phoneme_confidence_unk_threshold: 0.0
  feature_masking: null
  cer_reward_weight: 0.7
  ssim_reward_weight: 0.2
  use_utmos: true
  utmos_reward_weight: 0.1
trainer:
  precision: bf16-true
  gradient_clip_val: 0.0  # Manual optimization uses model.max_grad_norm.
```

Use a streaming inference mode with single-turn prompt batches, disable text/phoneme
corruption, and disable cached
codec codes when their codec differs from the policy. Keep
`model.use_attention_alignment=true` when restoring a checkpoint with an aligner;
its weights are retained and frozen during policy optimization. The parallel audio
head and reward networks are also excluded from the policy optimizer. Rollout
sampling and policy log probabilities use FP32 with matching temperature and top-k.
With `bf16-true`, trainable policy parameters use BF16 while codec, resampling,
aligner, and reward models remain in FP32. Exported inference archives use the
ordinary V2 model class. Small learning rates can quantize BF16 parameter updates;
`bf16-mixed` retains FP32 policy parameters when needed.
The GRPO prompt loader uses single-turn samples while retaining the checkpoint's
`model.use_multiturn_dataset` setting and vocabulary.

To mix in hard prompts, set `model.train_ds.dataset.challenging_texts_path` to a
UTF-8 file with one prompt per nonempty line and
`model.train_ds.dataset.challenging_text_replacement_prob` to the batch probability
(for example, `0.15`). One eligible English TTS sample is replaced when a prompt fits
the available token budget. Its reference audio remains available for speaker
similarity; stored IPA is skipped, so `model.gt_phoneme_input_prob` must be `0`.

CER/WER rewards and V2 validation prefer each sample's `normalized_text`, falling
back to its raw transcript. This reference is independent of
`load_normalized_text_percent`, which controls TTS input selection. Per-cut values
from input-config tags override the dataset default. A hard-prompt
replacement supplies its own reference. Ground-truth audio is used for speaker
similarity, rather than transcribed again to obtain the CER reference.

When `reference_free=false`, supply `init_from_ptl_ckpt` with a matching V2 Lightning
checkpoint for the frozen reference policy. Reward ASR, speaker verification, and
UTMOS dependencies follow the existing EasyMagpie online post-training setup.
