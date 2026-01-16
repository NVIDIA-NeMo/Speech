# Universal Streaming Inference

The `asr_streaming_infer.py` script enables streaming inference for both buffered (CTC/RNNT/TDT) and cache-aware (CTC/RNNT) ASR models. It supports processing a single audio file, a directory of audio files, or a manifest file.

Beyond streaming ASR, the script also supports:

* **Inverse Text Normalization (ITN)**
* **End-of-Utterance (EoU) Detection**
* **Streaming Speech Translation (requires vLLM installation)**
* **Word-level and Segment-level Output**

All related configurations can be found in the `../conf/asr_streaming_inference/` directory.