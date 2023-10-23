# Speaker Diarization

Documentation section for speaker related tasks can be found at:
 - [Speaker Diarization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html)
 - [Speaker Identification and Verification](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html)


## Features of NeMo Speaker Diarization
- Provides pretrained speaker embedding extractor models and VAD models.
- Does not need to be tuned on dev-set while showing the better performance than AHC+PLDA method in general.
- Estimates the number of speakers in the given session.
- Provides example script for asr transcription with speaker labels. 

## Supported Pretrained Speaker Embedding Extractor models
- [titanet_large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large)
- [ecapa_tdnn](https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn)
- [speakerverification_speakernet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speakerverification_speakernet)

## Supported Pretrained VAD models
- [vad_multilingual_marblenet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet)
- [vad_marblenet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet)
- [vad_telephony_marblenet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_telephony_marblenet)

## Supported ASR models
QuartzNet, CitriNet and Conformer-CTC models are supported. 
Recommended models on NGC:
- [stt_en_quartznet15x5](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_quartznet15x5)
- [stt_en_conformer_ctc_large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_large)
- [stt_en_citrinet_1024](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_citrinet_1024)

## Performance

#### Clustering Diarizer 
Diarization Error Rate (DER) table of `titanet_large.nemo` model on well known evaluation datasets. 

|         Evaluation Condition           |   AMI(Lapel)   | AMI(MixHeadset)     |   CH109  | NIST SRE 2000 |  
|:--------------------------------------:|:--------------:|:-------------------:|:--------:|:-------------:|
|       Domain Configuration             |   Meeting      |        Meeting      |Telephonic| Telephonic    |
|  Oracle VAD <br> Known # of Speakers   |     1.28       |         1.07        |  0.56    |     5.62      |
|  Oracle VAD <br> Unknown # of Speakers |     1.28       |         1.4         |  0.88    |     4.33      | 

* All models were tested using the domain specific `.yaml` files which can be found in `conf/inference/` folder.
* The above result is based on the oracle Voice Activity Detection (VAD) result.
* This result is based on [titanet_large.nemo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large) model.

#### Neural Diarizer 
Multi-scale Diarization Decoder (MSDD) model [Multi-scale Diarization decoder](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/model.html)
Diarization Error Rate (DER) table of [diar_msdd_telephonic.nemo](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_msdd_telephonic) model on telephonic speech datasets.

|                                 CH109| Forgiving                       | Fair                             | Full                            |
|-------------------------------------:|---------------------------------|----------------------------------|---------------------------------|
|              (collar, ignore_overlap)|  (0.25, True)                   |  (0.25, True)                    |   (0.0, False)                  |
|                          False Alarm | -                               | 0.62%                            | 1.80%                           |
|                                 Miss | -                               | 2.47%                            | 5.96%                           |
|                            Confusion | -                               | 0.43%                            | 2.10%                           |
|                                  DER | **0.58%**                       | **3.52%**                        | **9.86%**                       |


|                             CALLHOME | Forgiving                       | Fair                             | Full                            |
|-------------------------------------:|---------------------------------|----------------------------------|---------------------------------|
|              (collar, ignore_overlap)|  (0.25, True)                   |  (0.25, True)                    |   (0.0, False)                  |
|                          False Alarm | -                               | 1.05%                            | 2.24%                           |
|                                 Miss | -                               | 7.62%                            | 11.09%                          |
|                            Confusion | -                               | 4.06%                            | 6.03%                           |
|                                  DER | **4.15%**                       | **12.73%**                       | **19.37%**                      |

* Evaluation setting: Oracle VAD <br> Unknown number of speakers (max. 8)
* Clustering parameter: `max_rp_threshold=0.15` 
* All models were tested using the domain specific `.yaml` files which can be found in `conf/inference/` folder.
* The above result is based on the oracle Voice Activity Detection (VAD) result.

## Run Speaker Diarization on Your Audio Files

#### Example script for clustering diarizer: with system-VAD
```bash
  python clustering_diarizer/offline_diar_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_output' \
    diarizer.speaker_embeddings.parameters.save_embeddings=False \
    diarizer.vad.model_path=<pretrained model name or path to .nemo> \
    diarizer.speaker_embeddings.model_path=<pretrained speaker embedding model name or path to .nemo> 
```

#### Example script for neural diarizer: with system-VAD
```bash
  python neural_diarizer/multiscale_diar_decoder_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_output' \
    diarizer.speaker_embeddings.parameters.save_embeddings=False \
    diarizer.vad.model_path=<pretrained model name or path to .nemo> \
    diarizer.speaker_embeddings.model_path=<pretrained speaker embedding model name or path to .nemo> \
    diarizer.msdd_model.model_path=<pretrained MSDD model name or path .nemo> \
```

If you have oracle VAD files and groundtruth RTTM files for evaluation:
Provide rttm files in the input manifest file and enable oracle_vad as shown below. 
```bash
...
    diarizer.oracle_vad=True \
...
```

#### Arguments
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To run speaker diarization on your audio recordings, you need to prepare the following file.

- **`diarizer.manifest_filepath`: <manifest file>** Path to manifest file 

Example: `manifest.json`

```bash
{"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, label: "infer", "text": "-", "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath"="/path/to/uem/filepath"}
```
Mandatory fields are `audio_filepath`, `offset`, `duration`, `label:"infer"` and `text: <ground truth or "-" >`  , and the rest are optional keys which can be passed based on the type of evaluation 

Some of important options in config file: 

- **`diarizer.vad.model_path`: voice activity detection model name or path to the model**

Specify the name of VAD model, then the script will download the model from NGC. Currently, we have 'vad_multilingual_marblenet', 'vad_marblenet' and  'vad_telephony_marblenet' as options for VAD models.

`diarizer.vad.model_path='vad_multilingual_marblenet'`


Instead, you can also download the model from [vad_multilingual_marblenet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet), [vad_marblenet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_marblenet) and [vad_telephony_marblenet](https://ngc.nvidia.com/catalog/models/nvidia:nemo:vad_telephony_marblenet) and specify the full path name to the model as below.

`diarizer.vad.model_path='path/to/vad_multilingual_marblenet.nemo'`

- **`diarizer.speaker_embeddings.model_path`: speaker embedding model name**

Specify the name of speaker embedding model, then the script will download the model from NGC. Currently, we support 'titanet_large', 'ecapa_tdnn' and 'speakerverification_speakernet'.

`diarizer.speaker_embeddings.model_path='titanet_large'`

You could also download *.nemo files from [this link](https://ngc.nvidia.com/catalog/models?orderBy=scoreDESC&pageNumber=0&query=SpeakerNet&quickFilter=&filters=) and specify the full path name to the speaker embedding model file (`*.nemo`).

`diarizer.speaker_embeddings.model_path='path/to/titanet_large.nemo'` 
 

- **`diarizer.speaker_embeddings.parameters.multiscale_weights`: multiscale diarization**

Multiscale diarization system employs multiple scales at the same time to obtain a finer temporal resolution. To use multiscale feature, at least two scales and scale weights should be provided. The scales should be provided in descending order, from the longest scale to the base scale (the shortest). If multiple scales are provided, multiscale_weights must be provided in list format. The following example shows how multiscale parameters are specified and the recommended parameters.

- **`diarizer.msdd_model.model_path`: neural diarizer (multiscale diarization decoder) name**

If you want to use a neural diarizer model (e.g., MSDD model), specify the name of the neural diarizer model, then the script will download the model from NGC. Currently, we support 'diar_msdd_telephonic'.

Note that you should not specify a scale setting that does not match with the MSDD model you are using. For example, `diar_msdd_telephonic` model is based on 5 scales as in the configs in model configs.

`diarizer.speaker_embeddings.model_path='diar_msdd_telephonic'

You could also download [diar_msdd_telephonic](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_msdd_telephonic)
and specify the full path name to the speaker embedding model file (`*.nemo`).

`diarizer.msdd_model.model_path='path/to/diar_msdd_telephonic.nemo'` 
 

#### Example script: single-scale and multiscale

Single-scale setting:
```bash
  python offline_diar_infer.py \
     ... <other parameters> ...
     parameters.window_length_in_sec=1.5 \
     parameters.shift_length_in_sec=0.75 \
     parameters.multiscale_weights=null \
```

Multiscale setting (base scale - window_length 0.5 s and shift_length 0.25):

```bash
  python offline_diar_infer.py \
     ... <other parameters> ...
     parameters.window_length_in_sec=[1.5,1.0,0.5] \
     parameters.shift_length_in_sec=[0.75,0.5,0.25] \
     parameters.multiscale_weights=[0.33,0.33,0.33] \
```
 
<br/>

## Run Speech Recognition with Speaker Diarization

Using the script `offline_diar_with_asr_infer.py`, you can transcribe your audio recording with speaker labels as shown below:

```
[00:03.34 - 00:04.46] speaker_0: back from the gym oh good how's it going 
[00:04.46 - 00:09.96] speaker_1: oh pretty well it was really crowded today yeah i kind of assumed everyone would be at the shore uhhuh
[00:12.10 - 00:13.97] speaker_0: well it's the middle of the week or whatever so
```

Currently, NeMo offline diarization inference supports QuartzNet English model and ConformerCTC ASR models (e.g.,`QuartzNet15x5Base-En`, `stt_en_conformer_ctc_large`). 

#### Example script

```bash
python offline_diar_with_asr_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_asr_output' \
    diarizer.speaker_embeddings.model_path=<pretrained model name or path to .nemo> \
    diarizer.asr.model_path=<pretrained model name or path to .nemo> \
    diarizer.speaker_embeddings.parameters.save_embeddings=False \
    diarizer.asr.parameters.asr_based_vad=True
```
If you have reference rttm files or oracle number of speaker information, you can provide those file paths and number of speakers in the manifest file path and pass `diarizer.clustering.parameters.oracle_num_speakers=True` as shown in the following example.

```bash
python offline_diar_with_asr_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_asr_output' \
    diarizer.speaker_embeddings.model_path=<pretrained model name or path to .nemo> \
    diarizer.asr.model_path=<pretrained model name or path to .nemo> \
    diarizer.speaker_embeddings.parameters.save_embeddings=False \
    diarizer.asr.parameters.asr_based_vad=True \
    diarizer.clustering.parameters.oracle_num_speakers=True
```

#### Output folders

The above script will create a folder named `./demo_asr_output/`.
For example, in `./demo_asr_output/`, you can check the results as below.

```bash
./asr_with_diar
├── pred_rttms
    └── my_audio1.json
    └── my_audio1.txt
    └── my_audio1.rttm
    └── my_audio1_gecko.json
│
└── speaker_outputs
    └── oracle_vad_manifest.json
    └── subsegments_scale2_cluster.label
    └── subsegments_scale0.json
    └── subsegments_scale1.json
    └── subsegments_scale2.json
... 
```

`my_audio1.json` file contains word-by-word json output with speaker label and time stamps. We also provide a json output file for [gecko](https://gong-io.github.io/gecko/) tool, where you can visualize the diarization result along with the ASR output.

Example: `./demo_asr_output/pred_rttms/my_audio1.json`
```bash
{
    "status": "Success",
    "session_id": "my_audio1",
    "transcription": "back from the gym oh good ...",
    "speaker_count": 2,
    "words": [
        {
            "word": "back",
            "start_time": 0.44,
            "end_time": 0.56,
            "speaker_label": "speaker_0"
        },
...
        {
            "word": "oh",
            "start_time": 1.74,
            "end_time": 1.88,
            "speaker_label": "speaker_1"
        },
        {
            "word": "good",
            "start_time": 2.08,
            "end_time": 3.28,
            "speaker_label": "speaker_1"
        },
```

`*.txt` files in `pred_rttms` folder contain transcriptions with speaker labels and corresponding time.

Example: `./demo_asr_output/pred_rttms/my_audio1.txt`
```
[00:03.34 - 00:04.46] speaker_0: back from the gym oh good how's it going
[00:04.46 - 00:09.96] speaker_1: pretty well it was really crowded today yeah i kind of assumed everylonewould be at the shore uhhuh
[00:12.10 - 00:13.97] speaker_0: well it's the middle of the week or whatever so
[00:13.97 - 00:15.78] speaker_1: but it's the fourth of july mm
[00:16.90 - 00:21.80] speaker_0: so yeah people still work tomorrow do you have to work tomorrow did you drive off yesterday
```
 
In `speaker_outputs` folder we have three kinds of files as follows:
 
 - `oracle_vad_manifest.json` file contains oracle VAD labels that are extracted from RTTM files.
 - `subsegments_scale<scale_index>.json` is a manifest file for subsegments, which includes segment-by-segment start and end time with original wav file path. In multi-scale mode, this file is generated for each `<scale_index>`.
 - `subsegments_scale<scale_index>_cluster.label` file contains the estimated cluster labels for each segment. This file is only generated for the base scale index in multi-scale diarization mode.
 
 
### Optional Features for Speech Recognition with Speaker Diarization
 
#### Beam Search Decoder

Beam-search decoder can be applied to CTC based ASR models. To use this feature, [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) should be installed. [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) supports word timestamp generation and can be applied to speaker diarization. pyctcdecode also requires [KenLM](https://github.com/kpu/kenlm) and KenLM is recommended to be installed using PyPI. Install pyctcdecode in your environment with the following commands: 
```
pip install pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip
```
You should provide a trained KenLM language model to use pyctcdecode. Binary or `.arpa` format can be provided to hydra configuration as below.

```bash
  python offline_diar_with_asr_infer.py \
    ... <other parameters> ...
    diarizer.asr.ctc_decoder_parameters.pretrained_language_model="/path/to/kenlm_language_model.binary"
```
You can download publicly available language models (`.arpa` files) at [KALDI Tedlium Language Models](https://kaldi-asr.org/models/m5). Download [4-gram Big ARPA](https://kaldi-asr.org/models/5/4gram_big.arpa.gz) and provide the model path.
 
The following CTC decoder parameters can be modified to optimize the performance.      
`diarizer.asr.ctc_decoder_parameters.beam_width` (default: 32)      
`diarizer.asr.ctc_decoder_parameters.alpha` (default: 0.5)       
`diarizer.asr.ctc_decoder_parameters.beta` (default: 2.5)       
 
#### Realign Words with a Language Model (Experimental)

Diarization result with ASR transcript can be enhanced by applying a language model. To use this feature, python package [arpa](https://pypi.org/project/arpa/) should be installed.
```
pip install arpa
```
`diarizer.asr.realigning_lm_parameters.logprob_diff_threshold` can be modified to optimize the diarization performance (default value is 1.2). The lower the threshold, the more changes are expected to be seen in the output transcript.   

`arpa` package also uses KenLM language models as in pyctcdecode. You can download publicly available [4-gram Big ARPA](https://kaldi-asr.org/models/5/4gram_big.arpa.gz) model and provide the model path to hydra configuration as follows.
 

```bash
python offline_diar_with_asr_infer.py \
    ... <other parameters> ...
    diarizer.asr.realigning_lm_parameters.logprob_diff_threshold=1.2 \
    diarizer.asr.realigning_lm_parameters.arpa_language_model="/path/to/4gram_big.arpa"\
```
