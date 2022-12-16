# NeMo ASR+VAD Inference

This example provides the ASR+VAD inference pipeline, with the option to perform only ASR or VAD alone.

## Input

There are two types of input
-  A manifest passed to `manifest_filepath`, 
-  A directory containing audios passed to `audio_dir` and also specify `audio_type` (default to `wav`).

The input manifest must be a manifest json file, where each line is a Python dictionary. The fields ["audio_filepath", "offset", "duration",  "text"] are required. An example of a manifest file is:
```json
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000,  "text": "a b c d e"}
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 10000,  "text": "f g h i j"}
```

## Output
Output will be a folder storing the VAD predictions and/or a manifest containing the audio transcriptions. Some temporary data will also be stored.


## Usage

To run the code with ASR+VAD default settings:

```bash
python speech_to_text_with_vad.py \
    manifest_filepath=/PATH/TO/MANIFEST.json \
    vad_model=vad_multilingual_marblenet \
    asr_model=stt_en_conformer_ctc_large \
    vad_config=../conf/vad/vad_inference_postprocess.yaml
```

To use only ASR and disable VAD, set `vad_model=None` and `use_rttm=False`.

To use only VAD, set `asr_model=None` and specify both `vad_model` and `vad_config`.

To enable profiling, set `profiling=True`, but this will significantly slow down the program.

To use or disable feature masking, set `use_rttm` to `True` or `False`.

To normalize feature before masking, set `normalize=pre_norm`, 
and set `normalize=post_norm` for masking before normalization.

To use a specific value for feature masking, set `feat_mask_val` to the desired value. 
Default is `feat_mask_val=None`, where -16.530 (zero log mel-spectrogram value) will be used for `post_norm` and 0 (same as SpecAugment) will be used for `pre_norm`.

See more options in the `InferenceConfig` class.
