# NeMo SpeechLM Examples

This folder contains examples of using NeMo for training and fine-tuning speech language models. The current supported models are those that concatenate audio features with text embeddings and pass them through a GPT decoder, such as:
- SALM (https://arxiv.org/abs/2310.09424)
- VoiceTextBlender (https://arxiv.org/abs/2410.17485)
- Qwen-Audio (https://arxiv.org/abs/2311.07919)

Please run the scripts in the latest NeMo framework container.

## Data Preparation

There are two types of data format that is supported, one is **single turn question answering**, and the other one is **multi-turn multi-modal conversation**.

Below are examples of the jsonl manifest used in NeMo, note that you need to make sure each line in the manifest file is a valid dictionary in json format, but here we format them in multiple lines for better visualization.

### Single Turn Question Answering
You'll need to prepare data in the NeMo manifest format (jsonl files), where each line is a python dictionary with some keys, for example:
```
{
    "audio_filepath": "path/to/audio.wav",
    "offset": 0.0, # offset of the audio to load in seconds
    "duration": 10.0 , # duration of the audio in seconds, can set to `null` to load the whole audio
    "context": "what is the transcription of the audio?", # text prompt for the audio, 
    "answer": "the transcription of the audio", 
}
```

For better dataloading efficiency, you can tar indivisual audio files into tar files by following the script in `scripts/speech_recognition/convert_to_tarred_audio_dataset.py`

### Multi-turn Multi-modal Conversation

For multi-turn multi-modal conversation, each line in the jsonl manifest should be like:
```
{
    "id": "convo_1", 
    "conversations": 
        [
            {"from": "User", "value": "Can you help summarize the following?", "type": "text"}, 
            {"from": "User", "value": "123.wav", "type": "audio", "duration": 5.73}, 
            {"from": "Assistant", "value": "I'm glad to assist you with your request. Here's a summary:", "type": "text"}, 
            {"from": "Assistant", "value": "Once upon a time..there was a racoon..end of story...", "type": "text"}, 
            {"from": "User", "value": "Can you further shorten it?", "type": "text"}, 
            {"from": "Assistant", "value": "Of course!", "type": "text"}
        ]
}
```
Here, each conversation is a list of turns, where each turn is a dictionary with:
- `value` key: the content of the turn, either text string or path to audio files.
- `from` key: the speaker of the turn, either "User" or "Assistant".
- `type` key: the type of the turn, either "text" or "audio".
- `duration` key: the duration of the audio file in seconds, only needed for audio type turns.

Similarly you can tar them by using the script in `scripts/speech_llm/export_conversations_to_tar.py`.


### Creating Input Config for Lhotse dataloader
You can create an input config yaml file (e.g., `input_cfg.yaml`) of mixed formats like:
```
- input_cfg:
  - manifest_filepath: /path/to/multi-modal/manifest.json
    type: multimodal_conversation
    audio_locator: "<audio_locator>"  # a special string to indicate the audio positions in the combined prompt, can use arbitrary special string but need to make sure it doesn't appear in your regular context
  - manifest_filepath: /path/to/single-turn/manifest.json
    tags:
      # you can specify the default context to use if context field isn't found in the manifest
      default_context: Transcribe the audio into English text without punctuation and capitalization
    type: nemo
  - manifest_filepath: /path/to/single-turn/sharded_manfiest/manifest__OP_1..128_CL_.json
    tarred_audio_filepath: /path/to/single-turn/audio__OP_1..128_CL_.tar
    tags:
      # you can specify the default context to use if context field isn't found in the manifest
      default_context: Transcribe the audio into English text without punctuation and capitalization
    # only tarred single-turn data needs the `nemo_tarred` type, while `multimodal_conversation` is used for tarred multi-modal conversation data
    type: nemo_tarred
  type: group
```

To learn more about the dataloader configuration, please refer to the [documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#enabling-lhotse-via-configuration). 



## Training

An example training script using lhotse dataloader is:
```bash
CONFIG_PATH="<NeMo Root>/examples/speechlm/conf/salm"
CONFIG_NAME=salm_llama3.2-1b_fc_linear_peft

export WANDB_API_KEY="xxxx" && \
export CUDA_VISIBLE_DEVICES="0,1" && \
export HF_TOKEN="xxxxxxx" && \
export HF_HOME="/home/xxx/.huggingface/" && \
export HF_HUB_CACHE="/tmp/hf_cache" && \
export NEMO_MODELS_CACHE="/tmp/megatron_dist_ckpts" && \  # where to store the base LLM's distributed checkpoints
python speech_to_text_llm_train.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    data.common.add_boa_eoa=true \
    data.train_ds.manifest_filepath=null \  # use input_cfg instead
    data.validation_ds.manifest_filepath=null \
    ++data.train_ds.input_cfg=$INPUT_CFG \
    ++data.validation_ds.input_cfg=$INPUT_CFG \
    data.train_ds.num_workers=$NUM_WORKERS \
    data.validation_ds.num_workers=$NUM_WORKERS \
    data.common.global_batch_size=$GLOBAL_BATCH \
    data.common.micro_batch_size=$MICRO_BATCH \
    data.common.prompt_format='llama3' \
    strategy.tensor_model_parallel_size=$TP \
    strategy.context_parallel_size=$CP \
    ++data.train_ds.batch_size=$MICRO_BATCH \
    ++data.train_ds.defer_setup=true \
    ++data.train_ds.use_lhotse=true \
    ++data.train_ds.is_tarred=false \
    ++data.train_ds.use_bucketing=false \
    ++data.train_ds.batch_duration=null \
    ++data.train_ds.quadratic_duration=null \
    ++data.train_ds.bucket_duration_bins=null \
    ++data.train_ds.shuffle=false \
    ++data.train_ds.shuffle_buffer_size=10000 \
    ++data.train_ds.seed=10 \
    ++data.train_ds.shard_seed="randomized" \
    ++data.train_ds.force_iterable_dataset=true \
    ++data.validation_ds.batch_size=$MICRO_BATCH \
    ++data.validation_ds.defer_setup=true \
    ++data.validation_ds.use_lhotse=true \
    ++data.validation_ds.is_tarred=false \
    ++data.validation_ds.use_bucketing=false \
    ++data.validation_ds.batch_duration=null \
    ++data.validation_ds.quadratic_duration=null \
    ++data.validation_ds.bucket_duration_bins=null \
    ++data.validation_ds.shuffle_buffer_size=10000 \
    ++data.validation_ds.seed=10 \
    ++data.validation_ds.shard_seed="randomized" \
    ++data.validation_ds.shuffle=false \
    ++data.validation_ds.metric.name='loss' \ # set to `loss` to only calculate validation loss w/o LLM decoding for faster validation
    ++model.data.validation_ds.force_finite=true  \
    ++model.data.validation_ds.force_map_dataset=true  \
    ++trainer.use_distributed_sampler=false \
    ++trainer.limit_train_batches=2000 \
    trainer.val_check_interval=2000 \ # set to same value as limit_train_batches
    trainer.devices=-1 \
    trainer.max_steps=1000000 \
    trainer.accumulate_grad_batches=$GRAD_ACCUMULATION \
    name="${CONFIG_NAME}_run1" \
    strategy.ckpt_async_save=false \
    max_time_per_run="00:00:30:00"  # set to automatically stop the job after 30 minutes
```

## Inference

For running inference, we use the same script as for validation (which has groundtruth answer), but need to set a dummy groundtruth answer for doing inference. An example of inference/evaluation script is:
```bash
CONFIG_PATH="<NeMo Root>/examples/speechlm/conf/salm"
CONFIG_NAME=salm_llama3.2-1b_fc_linear_peft

export WANDB_API_KEY="xxxx" && \
export CUDA_VISIBLE_DEVICES="0,1" && \
export HF_TOKEN="xxxxxxx" && \
export HF_HOME="/home/xxx/.huggingface/" && \
export HF_HUB_CACHE="/tmp/hf_cache" && \
export NEMO_MODELS_CACHE="/tmp/megatron_dist_ckpts" && \
python speech_to_text_llm_validate.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    ~data.train_ds \ # remove training config
    data.common.add_boa_eoa=true \
    data.common.global_batch_size=$GLOBAL_BATCH \
    data.common.micro_batch_size=$MICRO_BATCH \
    data.common.prompt_format='llama3' \  # set to the same value as training
    data.validation_ds.metric.name='bleu' \  # set to `bleu` for enabling LLM decoding into text for evaluation
    data.validation_ds.manifest_filepath=null \
    ++data.validation_ds.input_cfg=$INPUT_CFG \
    data.validation_ds.num_workers=$NUM_WORKERS \
    ++data.validation_ds.batch_size=$MICRO_BATCH \
    ++data.validation_ds.defer_setup=true \
    ++data.validation_ds.use_lhotse=true \
    ++data.validation_ds.use_bucketing=false \
    ++data.validation_ds.batch_duration=null \
    ++data.validation_ds.quadratic_duration=null \
    ++data.validation_ds.bucket_duration_bins=null \
    ++data.validation_ds.shuffle=false \
    ++model.data.validation_ds.force_finite=true  \
    ++model.data.validation_ds.force_map_dataset=true  \
    ++trainer.use_distributed_sampler=false \
    ++resume.resume_from_path=$CKPT_PATH \  # path to the checkpoint to load
    ++data.validation_ds.write_predictions_to_file=true \
    ++data.validation_ds.output_dir=$OUTPUT_DIR \ # directory to save the predictions
    name="${CONFIG_NAME}_run1_eval" \  
    trainer.devices=1 \
    data.common.tokens_to_generate=256 \
    ++model.inference_config.tokens_to_generate=256 \
    ++model.inference_config.temperature=1.0 \
    ++model.inference_config.top_k=50 \
    ++model.inference_config.top_p=0.95 \
    ++model.inference_config.greedy=false \  # set to `true` to use greedy decoding instead of sampling
    ++model.inference_config.repetition_penalty=1.0 \
    ~logger.wandb  # remove wandb logger
```

## Notes
- If you want to drop PEFT, simply add `~model.peft` to the command line arguments.
- If you want to freeze/finetune each of the model's components, you can set `model.freeze_language_model`, `model.freeze_speech_model` and `model.freeze_modality_adapter` to `true` or `false` in the command line arguments.
- If you want to use other LLM models that are not in the example config, you can look for them in `nemo/collections/llm/gpt/model` and set the correspondng `model.llm._target_`, `model.llm.config._target_`, then look for their pretrained weights on Huggingface and set `model.llm.pretrained_model` to the corresponding model name.
- If you want to use Whisper encoder, please note that the current implementation in SpeechLM uses the native Whisper model on Huggingface, which pads or trims audios to a fixed 30s duration.

