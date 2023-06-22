# NeMo ASR+VAD Inference

This example provides the ASR+VAD inference pipeline, with the option to perform only ASR or VAD alone.

## Input

There are two types of input
-  A manifest passed to `manifest_filepath`, 
-  A directory containing audios passed to `audio_dir` and also specify `audio_type` (default to `wav`).

The input manifest must be a manifest json file, where each line is a Python dictionary. The fields ["audio_filepath", "offset", "duration"] are required. An example of a manifest file is:
```json
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000}
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 10000}
```

If you want to calculate WER, provide `text` in manifest as groundtruth. An example of a manifest file is:
```json
{"audio_filepath": "/path/to/audio_file1", "offset": 0, "duration": 10000, "text": "hello world"}
{"audio_filepath": "/path/to/audio_file2", "offset": 0, "duration": 10000, "text": "hello world"}
```

## Output
Output will be a folder storing the VAD predictions and/or a manifest containing the audio transcriptions. Some temporary data will also be stored.


## Usage

To run the code with ASR+VAD default settings:

```bash
python speech_to_text_with_vad.py \
    manifest_filepath=/PATH/TO/MANIFEST.json \
    vad_model=vad_multilingual_frame_marblenet \
    asr_model=stt_en_conformer_ctc_large \
    vad_config=../conf/vad/frame_vad_infer_postprocess.yaml
```

- To use only ASR and disable VAD, set `vad_model=None` and `use_rttm=False`.

- To use only VAD, set `asr_model=None` and specify both `vad_model` and `vad_config`.

- To enable profiling, set `profiling=True`, but this will significantly slow down the program.

### Using RTTM to handle non-speech audio segments
- To use or disable RTTM usage, set `use_rttm` to `True` or `False`. There are two options to use RTTM files, as specified by the parameter `rttm_mode`, which must be one of `mask` or `drop`. For `mask`, the RTTM file will be used to mask the non-speech features. For `drop`, the RTTM file will be used to drop the non-speech features.

- It's recommended that for `rttm_mode='drop'`, use larger `pad_onset` and `pad_offset` to avoid dropping speech features.

- To use a specific value for feature masking, set `feat_mask_val` to the desired value. 
Default is `feat_mask_val=None`, where -16.530 (zero log mel-spectrogram value) will be used for `post_norm` and 0 (same as SpecAugment) will be used for `pre_norm`.

- To normalize feature before masking, set `normalize=pre_norm`, and set `normalize=post_norm` for masking before normalization.

### Frame-VAD and Segment-VAD
- By default, `speech_to_text_with_vad.py` and `vad_config=../conf/vad/frame_vad_infer_postprocess.yaml` will use a frame-VAD model, which generates a speech/non-speech prediction for each audio frame of 20ms. 
- To use segment-VAD, use `speech_to_text_with_vad.py vad_type='segment' vad_config=../conf/vad/vad_inference_postprocessing.yaml` instead. In segment-VAD, the audio is split into segments and VAD is performed on each segment. The segments are then stitched together to form the final output. The segment size and stride can be specified by `window_length_in_sec` and `shift_length_in_sec` in the VAD config (e.g., `../conf/vad/vad_inference_postprocessing.yaml`) respectively. The default values are 0.63 seconds and 0.08 seconds respectively.

### More options
- See more options in the `InferenceConfig` data class.
