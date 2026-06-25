# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


NEMO_DEFAULT_MODEL_CARD_TEMPLATE = """---
{card_data}
---

# {model_name}

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-PUT-YOUR-ARCHITECTURE-HERE-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-PUT-YOUR-MODEL-SIZE-HERE-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-PUT-YOUR-LANGUAGE-HERE-lightgrey#model-badge)](#datasets)

**Put a short model description here.**

See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/index.html) for complete architecture details.


## NVIDIA NeMo: Training

To train, fine-tune, or experiment with the model, install the PyTorch build for your platform first, then install [NVIDIA NeMo](https://docs.nvidia.com/nemo/speech/nightly/starthere/install.html) with the extras you need.
```
pip install 'nemo-toolkit[all]'
```

## How to Use this Model

The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

**NOTE**: Please update the model class below to match the class of the model being uploaded.

```python
import nemo.core import ModelPT
model = ModelPT.from_pretrained("{repo_id}")
```

### NOTE

    Add some information about how to use the model here. An example is provided for ASR inference below.

    ### Transcribing using Python
    First, let's get a sample
    ```
    wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
    ```
    Then simply do:
    ```
    asr_model.transcribe(['2086-149220-0033.wav'])
    ```

    ### Transcribing many audio files

    ```shell
    python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
     pretrained_name="{repo_id}" \
     audio_dir=""
    ```

### Input

**Add some information about what are the inputs to this model**

### Output

**Add some information about what are the outputs of this model**

## Model Architecture

**Add information here discussing architectural details of the model or any comments to users about the model.**

## Training

**Add information here about how the model was trained. It should be as detailed as possible, potentially including the the link to the script used to train as well as the base config used to train the model. If extraneous scripts are used to prepare the components of the model, please include them here.**

### NOTE

    An example is provided below for ASR

    The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py) and this [base config](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml).

    The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA-NeMo/Speech/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).


### Datasets

**Try to provide as detailed a list of datasets as possible. If possible, provide links to the datasets on HF by adding it to the manifest section at the top of the README (marked by ---).**

### NOTE

    An example for the manifest section is provided below for ASR datasets

    datasets:
    - librispeech_asr
    - fisher_corpus
    - Switchboard-1
    - WSJ-0
    - WSJ-1
    - National-Singapore-Corpus-Part-1
    - National-Singapore-Corpus-Part-6
    - vctk
    - voxpopuli
    - europarl
    - multilingual_librispeech
    - mozilla-foundation/common_voice_8_0
    - MLCommons/peoples_speech

    The corresponding text in this section for those datasets is stated below -

    The model was trained on 64K hours of English speech collected and prepared by NVIDIA NeMo and Suno teams.

    The training dataset consists of private subset with 40K hours of English speech plus 24K hours from the following public datasets:

    - Librispeech 960 hours of English speech
    - Fisher Corpus
    - Switchboard-1 Dataset
    - WSJ-0 and WSJ-1
    - National Speech Corpus (Part 1, Part 6)
    - VCTK
    - VoxPopuli (EN)
    - Europarl-ASR (EN)
    - Multilingual Librispeech (MLS EN) - 2,000 hour subset
    - Mozilla Common Voice (v7.0)
    - People's Speech  - 12,000 hour subset


## Performance

**Add information here about the performance of the model. Discuss what is the metric that is being used to evaluate the model and if there are external links explaning the custom metric, please link to it.

### NOTE

    An example is provided below for ASR metrics list that can be added to the top of the README

    model-index:
    - name: PUT_MODEL_NAME
      results:
      - task:
          name: Automatic Speech Recognition
          type: automatic-speech-recognition
        dataset:
          name: AMI (Meetings test)
          type: edinburghcstr/ami
          config: ihm
          split: test
          args:
            language: en
        metrics:
        - name: Test WER
          type: wer
          value: 17.10
      - task:
          name: Automatic Speech Recognition
          type: automatic-speech-recognition
        dataset:
          name: Earnings-22
          type: revdotcom/earnings22
          split: test
          args:
            language: en
        metrics:
        - name: Test WER
          type: wer
          value: 14.11

Provide any caveats about the results presented in the top of the discussion so that nuance is not lost.

It should ideally be in a tabular format (you can use the following website to make your tables in markdown format - https://www.tablesgenerator.com/markdown_tables)**

## Limitations

**Discuss any practical limitations to the model when being used in real world cases. They can also be legal disclaimers, or discussion regarding the safety of the model (particularly in the case of LLMs).**


### Note

    An example is provided below

    Since this model was trained on publicly available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on. The model might also perform worse for accented speech.


## License

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## References

**Provide appropriate references in the markdown link format below. Please order them numerically.**

[1] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA-NeMo/Speech)
"""
