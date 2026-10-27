# Serve waveform audio with TensorRT and Triton

This example runs log-mel preprocessing, the encoder, prediction network and joint in TensorRT.
NeMo's cached TDT loop handles decoding with CUDA graphs; SentencePiece converts tokens to text.
PyTorch remains a runtime dependency for tensors and decoding control.
Use PyTorch 2.14 or newer for full CUDA graphs; partial graphs are a compatibility fallback for older versions.

Build the [preprocessing and encoder plans](../../../../docs/source/asr/inference.rst) from the same checkpoint, then:

```bash
python examples/asr/export/build_trt_tdt_decoder.py \
    --model model.nemo --output decoder --max-batch 16
python examples/asr/export/transducer/package_trt_asr.py \
    --model model.nemo --preprocessor preprocessor.plan --encoder encoder.plan \
    --decoder decoder --output model_repository/asr
tritonserver --model-repository=model_repository
```

The Python backend needs this checkout, TensorRT and a compatible Python stub.
See [Triton's environment guide](https://github.com/triton-inference-server/python_backend#managing-python-runtime-and-libraries).
Build plans for the deployment GPU and TensorRT version; packaging checks their shape limits on that GPU.
Validate recognition before serving.

Send mono FP32 `AUDIO [B, samples]` at the checkpoint's sample rate and INT64 `AUDIO_LENGTH [B, 1]`.
Pad requests to matching widths for dynamic batching; lengths exclude padding.
Outputs are `TEXT [B, 1]` plus the [decoder's token outputs](README_tdt_serving.md#inputs-and-outputs).
One model instance executes requests serially.
Warm up intended shapes before measuring.

This example supports offline SentencePiece TDT models.
It does not provide streaming state or beam search.
