"""
TRT direct runner for EasyMagpieTTS local transformer.

The local transformer is a 3-layer causal AR transformer (hidden_dim=1536) that
generates one codebook token per step.  We export it once to ONNX (dynamic T),
then compile to a TRT fp16 engine with trtexec / build_lt_engine.py.

Interface is intentionally minimal so it can be dropped into
_run_local_transformer as lt_backend="trt_direct".

IMPORTANT — device isolation:
  The TRT-LLM backbone engine (cuda:1) loads libnvinfer_plugin_tensorrt_llm.so
  with RTLD_GLOBAL, which modifies the global Myelin kernel context.  Running a
  plain vanilla TRT engine on the *same* device after the backbone has executed
  causes Myelin error 700 (cudaErrorIllegalAddress).
  To avoid this conflict the LT engine must be built for and run on cuda:0.
  Inputs are transferred to cuda:0 before execution and the result is transferred
  back to the original device.  The transfer cost is negligible: the LT input is
  at most 1×16×1536×fp16 = 48 KB per call, and LT is called 16 times per step.

Build steps:
  1. python /tmp/export_lt_onnx.py        (exports lt.onnx)
  2. python /tmp/build_lt_engine.py        (builds lt.engine on cuda:0)
"""
from __future__ import annotations

import torch
import tensorrt as trt


class TRTLocalTransformerRunner:
    """
    Execute the local-transformer TRT fp16 engine.

    Inputs
    ------
    x      : (B, T, H) — any device / dtype; cast to fp16, moved to run_device
    x_mask : (B, T)    — same

    Output
    ------
    output : (B, T, H) — on the same device/dtype as input ``x``

    Parameters
    ----------
    engine_path : path to the serialised TRT engine (.engine file)
    device      : CUDA device on which the inputs and outputs live (e.g. cuda:1)
    run_device  : CUDA device on which the TRT engine was built and will run.
                  Defaults to ``device``.  Set to a different GPU (e.g. cuda:0)
                  to isolate the engine from TRT-LLM's Myelin context on the
                  main TTS GPU.
    """

    HIDDEN_SIZE = 1536

    def __init__(
        self,
        engine_path: str,
        device: torch.device,
        run_device: torch.device | None = None,
    ):
        self.device = torch.device(device)
        self.run_device = torch.device(run_device) if run_device is not None else self.device

        # Ensure the TRT runtime is initialised on the correct CUDA device.
        with torch.cuda.device(self.run_device):
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as f:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            assert self._engine is not None, f"Failed to load TRT LT engine from {engine_path}"
            self._context = self._engine.create_execution_context()

        self._stream = torch.cuda.Stream(device=self.run_device)

    def __call__(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        assert H == self.HIDDEN_SIZE, f"Expected hidden_size={self.HIDDEN_SIZE}, got {H}"

        input_dtype   = x.dtype
        input_device  = x.device

        # Move to run_device (may be same device → zero-copy if dtype also matches)
        x_fp16    = x.to(device=self.run_device, dtype=torch.float16).contiguous()
        mask_fp16 = x_mask.to(device=self.run_device, dtype=torch.float16).contiguous()
        out       = torch.empty(B, T, H, dtype=torch.float16, device=self.run_device)

        ctx = self._context
        ctx.set_input_shape("x",      (B, T, H))
        ctx.set_input_shape("x_mask", (B, T))
        ctx.set_tensor_address("x",      x_fp16.data_ptr())
        ctx.set_tensor_address("x_mask", mask_fp16.data_ptr())
        ctx.set_tensor_address("output", out.data_ptr())

        with torch.cuda.device(self.run_device):
            ok = ctx.execute_async_v3(self._stream.cuda_stream)
        assert ok, "TRT LT engine execution failed"
        self._stream.synchronize()

        # Move result back to the original device/dtype
        return out.to(device=input_device, dtype=input_dtype)
