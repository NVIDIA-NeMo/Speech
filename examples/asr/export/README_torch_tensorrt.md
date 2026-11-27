# Build an encoder with Torch-TensorRT

[build_torch_trt_encoder.py](build_torch_trt_encoder.py) traces a NeMo Transformer encoder with `torch.export` and compiles it into a standalone TensorRT plan, without ONNX.

Use compatible PyTorch, Torch-TensorRT and TensorRT versions.
Relative attention and encoders without subsampling need [dynamic gather](https://github.com/pytorch/TensorRT/pull/4704) and [dynamic clone](https://github.com/pytorch/TensorRT/pull/4705) support, absent in Torch-TensorRT 2.14.
Validated with matching PyTorch/Torch-TensorRT 2.15.0.dev20261005+cu132 builds and TensorRT 11.3.0.99.

```bash
python examples/asr/export/build_torch_trt_encoder.py \
    --model model.nemo --engine encoder.plan
```

The builder uses the encoder's SDPA export path, FP32 and disabled TF32.
Unsupported operations fail conversion.
Defaults cover batches 1–16 and 16–4000 padded feature frames; see `--help` for options.
With feature stacking, keep the minimum padded frame count at least twice the stacking factor; true lengths may be smaller or zero.

Running the plan requires TensorRT, without Torch-TensorRT.
Its inputs are `audio_signal` and `length`; outputs are `outputs` and `encoded_lengths`.
Preprocessing, decoding and tokenization remain separate; streaming caches are unsupported.
Build for the target GPU/runtime and compare conversion routes with matching weights, precision and shape limits.
