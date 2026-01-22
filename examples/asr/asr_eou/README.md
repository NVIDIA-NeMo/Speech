# Finetuning streming ASR model for integrated end-of-utterance (EOU) detection

This tutorial shows how to finetune a streaming ASR model (e.g., [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)) for integrated EOU detection (e.g., [nvidia/parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)).

## Steps

1. Prepare model
2. Prepare dataset
3. Train model
4. Evaluate model

## 1. Prepare model

1.1. Download pretrained model

1.2. Add special tokens to tokenizer

1.3. Update model config for ASR-EOU model


## 2. Prepare dataset

2.1 Mainifest format

2.2 Getting timestamps for end-of-utterance (EOU)

2.3 (Optional) Add end-of-backchannel (EOB) labels to dataset

2.4 Creating tarred datasets for large-scale training

2.5 Creating input data config for blending ASR and EOU data

2.6 Creating evaluation dataset



## 3. Train model


## 4. Evaluate model


