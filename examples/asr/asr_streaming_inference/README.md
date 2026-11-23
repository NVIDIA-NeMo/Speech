# Universal Streaming Inference

The `asr_streaming_infer.py` script enables streaming inference for both buffered (CTC/RNNT/TDT) and cache-aware (CTC/RNNT) ASR models. It supports processing a single audio file, a directory of audio files, or a manifest file.

Beyond streaming ASR, the script also supports:

* **Inverse Text Normalization (ITN)**
* **End-of-Utterance (EoU) Detection**
* **Streaming Speech Translation (requires vLLM installation)**
* **Word-level and Segment-level Output**

All related configurations can be found in the `../conf/asr_streaming_inference/` directory.

The EoU detection of the pipelines is based on the silence predicted by the ASR model (`endpointing` config). End-of-utterance models with `<EOU>`/`<EOB>` tokens (e.g. `parakeet_realtime_eou_120m-v1`) and `decoding.restart_after_tokens` are not supported, and the pipelines raise an error for them: such models stop emitting tokens after the first `<EOU>` unless decoding restarts after it. Use `transcribe()` with `decoding.restart_after_tokens` or `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py` with `rnnt_decoding.restart_after_tokens='["<EOU>","<EOB>"]'` instead (see [ASR-EOU inference on audio with multiple utterances](../asr_eou/README.md#41-inference-on-audio-with-multiple-utterances)).
