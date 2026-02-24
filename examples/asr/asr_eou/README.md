# Finetuning streming ASR model for integrated end-of-utterance (EOU) detection

This tutorial shows how to finetune a streaming ASR model (e.g., [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)) for integrated EOU detection (e.g., [nvidia/parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)). 

We use [Nemotron-Speech-Streaming-En-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) as an example of pretrained ASR model.

## Steps

1. Model preparation
2. Dataset preparation
3. Model training
4. Model evaluation

## 1. Model preparation

### 1.1. Download pretrained model

Download the [Nemotron-Speech-Streaming-En-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) model from HuggingFace via:
```bash
wget https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/resolve/main/nemotron-speech-streaming-en-0.6b.nemo
```

### 1.2. Add special tokens to tokenizer

By default, we use `<EOU>` and `<EOB>` for "end-of-utterance" and "end-of-backchannel" respectively, and add these two special tokens to the tokenizer of the pretrained model:
```bash
python <NeMo Root>/scripts/asr_eou/tokenizers/add_special_tokens_to_sentencepiece.py \
    --input_file /path/to/nemotron-speech-streaming-en-0.6b.nemo \
    --output_dir /path/to/asr_eou_tokenizer_dir
```
The output directory `/path/to/asr_eou_tokenizer_dir` will contain the updated tokenizer to be used when updateding the model config.

The special tokens are added to the end of the original vocabulary. For example, if the original vocabulary size is 1024, the new vocabulary size will be 1026, and the special tokens will be added at the indices 1024 and 1025.

### 1.3. Update model config for ASR-EOU model

We can extract the model config from the downloaded .nemo file by:
```bash
tar -xvf /path/to/nemotron-speech-streaming-en-0.6b.nemo -C /path/to/asr_model_dir
```
The output file `/path/to/asr_model_dir/model_config.yaml` is the model config to be updated for finetuning the ASR model into an ASR-EOU model.

In the model config file, we need to change the tokenizer to use the new tokenizer with special tokens:
```yaml
tokenizer:
  dir: /path/to/asr_eou_tokenizer_dir
  type: bpe
```

We also need to add some additional  configurations to the model section to specify how we want to initialize the weights for the special tokens:
```yaml
model:
  token_init_method: "constant"  # choices=['min', 'max', 'mean', 'constant']
  token_init_weight_value: null  # only applicable when token_init_method='constant'
  token_init_bias_value: -1000.0  # only applicable when token_init_method='constant'
```

You may also need to change the optimization and loss parameters to suit your use cases. We empirically find that setting `fastemit_lambda` to `3e-2` is a good start.

```yaml
loss:
    loss_name: "default"
    warprnnt_numba_kwargs:
      # FastEmit regularization: https://arxiv.org/abs/2010.11148
      # You may enable FastEmit to increase the accuracy and reduce the latency of the model for streaming
      # You may set it to lower values like 1e-3 for models with larger right context
      fastemit_lambda: 3e-2 
```

We also need to change the training, validation and test data paths in the model config file based on how we prepare the EOU labeled dataset illustrated in the next section.

For a full example of the model config file, please refer to `examples/asr/conf/asr_eou/fastconformer_transducer_bpe_streaming_xlarge.yaml`.


## 2. Dataset preparation

When finetuning the ASR model for EOU detection, we need to prepare the dataset in a specific format. But more importantly, we need to make sure the dataset used for finetuning meets the criteria that each sample should contain a single utterance, otherwise the model's EOU prediction accuracy will be degraded. In the case of using a small EOU dataset, we can blend the EOU dataset with the some normal ASR dataset which does not necessarily contain EOU labels, such that ASR WER is not significantly degraded. For lowest possible EOU latency, we recommend dropping the punctuations from the transcriptions to simplify the text processing.


### 2.1 Mainifest format

We expect the input data manifest to be JSONL format, with each line containing the following fields:
```json
{
    "audio_filepath": "/path/to/audio.wav",
    "text": "The text of the audio.", # transcript of the utterance
    "offset": 0.0,  # offset of the audio, in seconds
    "duration": 3.0,  # duration of the audio, in seconds
    "sou_time": 0.2,  # start of utterance time, in seconds
    "eou_time": 1.5,  # end of utterance time, in seconds
    "is_backchannel": false  # [optional] whether the utterance is a backchannel phrase
  }
```

Your original input manifest should contain the fields `audio_filepath`, `text`, `offset` and `duration`, while the fields `sou_time`, `eou_time` and `is_backchannel` can be obtained by following steps.

### 2.2 Getting timestamps for end-of-utterance (EOU)

We recommend using forced alignment to get the timestamps for EOU. One way to do this is to use the [Nemo Forced Aligner](https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner) tool.

```bash
python <NeMo Root>/tools/nemo_forced_aligner/align_eou.py \
    pretrained_name="nvidia/parakeet-ctc-0.6b" \
    manifest_filepath=/path/to/asr_manifest.jsonl \
    output_manifest_filepath=/path/to/asr_eou_manifest.jsonl
```
The output manifest will contain the fields `audio_filepath`, `text`, `offset`, `duration`, `sou_time` and `eou_time`.


### 2.3 (Optional) Add end-of-backchannel (EOB) labels to dataset

Backchannel phrases refer to those phrases that are not part of the main conversation, but are used to acknowledge or respond to the speaker. For example, "uh-huh", "yeah", "right", "okay", "thanks", "sorry", etc. We can also train the model to detect backchannel phrases by adding end-of-backchannel (EOB) labels to the dataset, so that the cascaded system can leverage the EOU and EOB predictions to better understand the conversation. However, we can also treat EOB as a special case of EOU, and match the predicted EOU phrases with a list of predefined backchannel phrases to prediction EOB, which is more flexible in handling different backchannel phrases.

If you want to add end-of-backchannel (EOB) labels to training, you can use the following script to add the `is_backchannel` field to the manifest:

```bash
python <NeMo Root>/scripts/asr_eou/add_eob_labels.py \
    input_manifest=/path/to/asr_manifest.jsonl \
    output_manifest=/path/to/asr_eou_eob_manifest.jsonl
```

The `add_eob_labels.py` file has a list of predefined backchannel phrases, and you can edit it to add more backchannel phrases if needed. An easy way to figure out backchannel phrases is to find the most frequent one, two or three words utterances in the dataset, and manually check if they are backchannel phrases.

2.4 Creating tarred datasets for large-scale training.

For more efficient training, you can create tarred datasets for the ASR and EOU datasets by using `scripts/speech_recognition/convert_to_tarred_audio_dataset.py` script.

### 2.5 Creating input data config for blending ASR and EOU data

Please refer to the [documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#lhotse-dataloading) for more details on how to specify the dataset configuration in the model config file.

An example of the `train_input_config.yaml` file is shown below, where we use 0.1 weight for the ASR dataset and 0.9 weight for the EOU dataset.
```yaml
- input_cfg:
  - corpus: Librispeech
    language: en
    type: nemo
    manifest_filepath: /data/LibriSpeech/train_other_500.jsonl  # this is a normal ASR dataset
  tags:
    taskname: asr
  type: group
  weight: 0.1
- input_cfg:
  - corpus: LibriTTS
    language: en
    type: nemo
    manifest_filepath: /data/LibriTTS/train_clean_360_eou.jsonl  # this is a EOU manifest after adding sou_time and eou_time fields
  tags:
    taskname: eou
  type: group
  weight: 0.9
```


### 2.6 Creating evaluation dataset

We can create evaluation dataset by padding the audio signal with non-speech frames and/or adding noise to the clean audios.

Example usage with multiple manifests matching a pattern:
```bash
python <NeMo Root>/scripts/asr_eou/generate_noisy_eval_data.py \
    output_dir=/path/to/output/dir \
    data.manifest_filepath=/path/to/manifest/dir/ \
    data.pattern="*.json" \
    data.seed=42 \
    data.noise.manifest_path /path/to/noise_manifest.json
```

You can modify the yaml config to specify the augmentation parameters in `scripts/asr_eou/conf/data.yaml`.


### 2.7 Configuring dataset in model config

Now we can update the model config to use the prepared training and evaluation data config.

```yaml
model:
  train_ds:
    input_cfg: /path/to/train_input_config.yaml
    manifest_filepath: null
    tarred_audio_filepaths: null
    ignore_eob_label: true  # ignore backchannel and treat them the same as EOU

    random_padding:
      prob: 0.9
      min_post_pad_duration: 3.0 # minimum duration of post-padding silence in seconds
      min_pre_pad_duration: 0.0 # minimum duration of pre-padding silence in seconds
      max_pad_duration: 6.0  # maximum duration of pre/post padding in seconds
      max_total_duration: 40.0  # maximum total duration of the padded audio in seconds
      pad_distribution: 'uniform'  # distribution of padding duration, 'uniform' or 'normal'
      normal_mean: 0.5  # mean of normal distribution used when pad_distribution='normal'
      normal_std: 2.0  # standard deviation of normal distribution  used when pad_distribution='normal'
      
    augmentor:
      white_noise:
        prob: 0.9
        min_level: -90
        max_level: -46
      gain:
        prob: 0.2
        min_gain_dbfs: -10.0
        max_gain_dbfs: 10.0
      noise:
        prob: 0.9
        manifest_path: /path/to/noise_manifest.json
        min_snr_db: 0
        max_snr_db: 20
        max_gain_db: 300.0

  validation_ds:
    input_cfg: null
    manifest_filepath: ["/path/to/eval_manifest1.json", "/path/to/eval_manifest2.json", ...]
    tarred_audio_filepaths: null
    ignore_eob_label: true  # ignore backchannel and treat them the same as EOU

```

For a full example of the model config file, please refer to `examples/asr/conf/asr_eou/fastconformer_transducer_bpe_streaming_xlarge.yaml`.


## 3. Model training

To start the training, you can run the following command:
```bash
#!/bin/bash

TRAIN_INPUT_CFG=/path/to/train_input_config.yaml
VAL_MANIFEST=/path/to/val_manifest.json
NOISE_MANIFEST=/path/to/noise_manifest.json

PRETRAINED_NEMO=/path/to/nemotron-speech-streaming-en-0.6b.nemo 

BATCH_SIZE=16
NUM_WORKERS=8
LIMIT_TRAIN_BATCHES=1000
VAL_CHECK_INTERVAL=1000
MAX_STEPS=1000000

EXP_NAME=nemotron_speech_streaming_en_0.6b_eou
SCRIPT=${NEMO_PATH}/examples/asr/asr_eou/speech_to_text_rnnt_eou_train.py
CONFIG_PATH=${NEMO_PATH}/examples/asr/conf/asr_eou
CONFIG_NAME=fastconformer_transducer_bpe_streaming_xlarge

CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++init_from_nemo_model=$PRETRAINED_NEMO \
    model.encoder.att_context_size="[70,1]" \
    model.train_ds.input_cfg=$TRAIN_INPUT_CFG \
    model.train_ds.augmentor.noise.manifest_path=$NOISE_MANIFEST \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.train_ds.batch_size=$BATCH_SIZE \
    model.train_ds.num_workers=$NUM_WORKERS \
    model.validation_ds.batch_size=$BATCH_SIZE \
    model.validation_ds.num_workers=$NUM_WORKERS \
    ~model.test_ds \
    trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
    trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    trainer.max_steps=$MAX_STEPS \
    exp_manager.name=$EXP_NAME
```

For lowest EOU latency, we set `att_context_size` to `[70,1]` in the model config file, which means the model lookahead is 1 frame (80ms), and the input chunk size is thus 2 frames (160ms).


## 4. Model evaluation

After training, we can evaluate the model on the evaluation dataset by running the following command:
```bash
TEST_MANIFEST="[/path/to/your/test_manifest.json,/path/to/your/test_manifest2.json,...]"
TEST_NAME="[test_name1,test_name2,...]"
TEST_BATCH=32
NUM_WORKERS=8

SAVE_PRED_TO_FILE=/path/to/predictions.json  # optional, if you want to save the predictions to a file, will slow down the evaluation speed. Set to `null` to disable.
PRETRAINED_NEMO=/path/to/EOU/model.nemo
CONFIG_NAME=fastconformer_transducer_bpe_streaming_xlarge

python speech_to_text_eou_eval.py \
    --config-name $CONFIG_NAME \
    ++save_pred_to_file=$SAVE_PRED_TO_FILE \
    ++init_from_nemo_model=$PRETRAINED_NEMO \
    ~model.train_ds \
    ~model.validation_ds \
    ++model.test_ds.defer_setup=true \
    ++model.test_ds.sample_rate=16000 \
    ++model.test_ds.manifest_filepath=$TEST_MANIFEST \
    ++model.test_ds.name=$TEST_NAME \
    ++model.test_ds.batch_size=$TEST_BATCH \
    ++model.test_ds.num_workers=$NUM_WORKERS \
    ++model.test_ds.drop_last=false \
    ++model.test_ds.force_finite=true \
    ++model.test_ds.shuffle=false \
    ++model.test_ds.pin_memory=true \
    exp_manager.create_wandb_logger=false
```

The script will show the WER metrics along with EOU metrics like latency, early cutoff rate, miss detection rate, etc.


## 5. Model deployment with voice agent

Please refer to the [NeMo Voice Agent](https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/voice_agent/README.md) example for more details on how to deploy the ASR-EOU model with voice agent.

