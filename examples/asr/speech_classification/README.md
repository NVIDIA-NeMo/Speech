# Speech Classification

This directory contains example scripts to train speech classification and voice activity detection models. There are two types of VAD models: Frame-VAD and Segment-VAD.

## Frame-VAD

The frame-level VAD model predicts for each frame of the audio whether it has speech or not. For example, with the default config file (`../conf/marblenet/marblenet_3x2x64_20ms.yaml`), the model provides a probability for each frame of 20ms length.

### Training
```sh
python speech_to_label.py \
    --config-path=<path to directory of configs, e.g. "../conf/marblenet">
    --config-name=<name of config without .yaml, e.g. "marblenet_3x2x64_20ms"> \
    model.train_ds.manifest_filepath="[<path to train manifest1>,<path to train manifest2>]" \
    model.validation_ds.manifest_filepath=["<path to val manifest1>","<path to val manifest2>"] \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    strategy="ddp" \
    trainer.max_epochs=100
```

The input manifest must be a manifest json file, where each line is a Python dictionary. The fields ["audio_filepath", "offset", "duration",  "label"] are required. An example of a manifest file is:
```
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000,  "label": "0 1 0 0 1"}
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 10000,  "label": "0 0 0 1 1 1 1 0 0"}
```
For example, if you have a 1s audio file, you'll need to have 50 frame labels in the manifest entry like "0 0 0 0 1 1 0 1 .... 0 1".
However, shorter label strings are also supported for smaller file sizes. For example, you can prepare the `label` in 40ms frame, and the model will properly repeat the label for each 20ms frame.


### Inference
python frame_vad_infer.py \
    --config-path="../conf/vad" --config-name="frame_vad_infer_postprocess" \
    dataset=<Path of manifest file containing evaluation data. Audio files should have unique names>

The manifest json file should have the following format (each line is a Python dictionary):
```
{"audio_filepath": "/path/to/audio_file1.wav", "offset": 0, "duration": 10000}  
{"audio_filepath": "/path/to/audio_file2.wav", "offset": 0, "duration": 10000}  
```

#### Evaluation
If you want to evaluate tne model's AUROC and DER performance, you need to set `evaluate: True` in config yaml (e.g., `../conf/vad/frame_vad_infer_postprocess.yaml`), and also provide groundtruth in label strings:
```
{"audio_filepath": "/path/to/audio_file1.wav", "offset": 0, "duration": 10000, "label": "0 1 0 0 0 1 1 1 0"}
```
or RTTM files:
```
{"audio_filepath": "/path/to/audio_file1.wav", "offset": 0, "duration": 10000, "rttm_filepath": "/path/to/rttm_file1.rttm"}
```


## Segment-VAD

Segment-level VAD predicts a single label for each segment of audio (e.g., 0.63s by default).

### Training
```sh
python speech_to_label.py \
    --config-path=<path to dir of configs, e.g. "../conf/marblenet"> \
    --config-name=<name of config without .yaml, e.g., "marblenet_3x2x64"> \
    model.train_ds.manifest_filepath="[<path to train manifest1>,<path to train manifest2>]" \
    model.validation_ds.manifest_filepath=["<path to val manifest1>","<path to val manifest2>"] \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    strategy="ddp" \
    trainer.max_epochs=100
```

The input manifest must be a manifest json file, where each line is a Python dictionary. The fields ["audio_filepath", "offset", "duration",  "label"] are required. An example of a manifest file is:
```
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 0.63,  "label": "0"}
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 0.63,  "label": "1"}
```


### Inference
```sh
python vad_infer.py \
    --config-path="../conf/vad" \
    --config-name="vad_inference_postprocessing.yaml"
    dataset=<Path of json file of evaluation data. Audio files should have unique names>
```
The manifest json file should have the following format (each line is a Python dictionary):
```
{"audio_filepath": "/path/to/audio_file1.wav", "offset": 0, "duration": 10000}  
{"audio_filepath": "/path/to/audio_file2.wav", "offset": 0, "duration": 10000}  
```
