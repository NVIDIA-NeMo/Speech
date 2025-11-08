# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import os
import tempfile
import threading
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch

from nemo.utils.export_utils import add_casts_around_norms, replace_for_export
from nemo.utils.import_utils import safe_import

polygraphy, polygraphy_imported = safe_import("polygraphy")
if polygraphy_imported:
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import (
        CreateConfig,
        Profile,
        engine_bytes_from_network,
        engine_from_bytes,
        network_from_onnx_path,
    )

trt, trt_imported = safe_import("tensorrt")
torch_tensorrt, _ = safe_import("torch_tensorrt")
cudart, _ = safe_import("cuda.cudart")

lock_sm = threading.Lock()


def trt_to_torch_dtype_dict():
    """
    Map of TRT dtype -> Torch dtype
    """
    return {
        trt.int32: torch.int32,
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.float16,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }


def get_profile_shapes(input_shape: Sequence[int], dynamic_batchsize: Sequence[int] | None):
    """
    Given a sample input shape, calculate min/opt/max shapes according to dynamic_batchsize.
    """

    def scale_batch_size(input_shape: Sequence[int], scale_num: int):
        scale_shape = [*input_shape]
        scale_shape[0] = scale_num
        return scale_shape

    # Use the dynamic batchsize range to generate the min, opt and max model input shape
    if dynamic_batchsize:
        min_input_shape = scale_batch_size(input_shape, dynamic_batchsize[0])
        opt_input_shape = scale_batch_size(input_shape, dynamic_batchsize[1])
        max_input_shape = scale_batch_size(input_shape, dynamic_batchsize[2])
    else:
        min_input_shape = opt_input_shape = max_input_shape = input_shape
    return min_input_shape, opt_input_shape, max_input_shape


def get_dynamic_axes(profiles):
    """
    This method calculates dynamic_axes to use in onnx.export().
    Args:
       profiles: [[min,opt,max],...] list of profile dimensions
    """
    dynamic_axes: dict[str, list[int]] = {}
    if not profiles:
        return dynamic_axes
    for profile in profiles:
        for key in profile:
            axes = []
            vals = profile[key]
            for i in range(len(vals[0])):
                if vals[0][i] != vals[2][i]:
                    axes.append(i)
            if len(axes) > 0:
                dynamic_axes[key] = axes
    return dynamic_axes


def cuassert(cuda_ret):
    """
    Error reporting method for CUDA calls.
    Args:
     cuda_ret: CUDA return code.
    """
    err = cuda_ret[0]
    if err != 0:
        raise RuntimeError(f"CUDA ERROR: {err}")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class ShapeError(Exception):
    """
    Exception class to report errors from setting TRT plan input shapes
    """

    pass


class TRTEngine:
    """
    An auxiliary class to implement running of TRT optimized engines

    """

    def __init__(self, plan_path, logger=None):
        """
        Loads serialized engine, creates execution context and activates it
        Args:
          plan_path: path to serialized TRT engine.
          logger: optional logger object
        """
        self.plan_path = plan_path
        self.logger = logger or getLogger("trt_compile")
        self.logger.info(f"Loading TensorRT engine: {self.plan_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.plan_path))
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        self.dtypes = []
        self.cur_profile = 0
        self.input_table = {}
        dtype_dict = trt_to_torch_dtype_dict()
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_names.append(binding)
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(binding)
                dtype = dtype_dict[self.engine.get_tensor_dtype(binding)]
                self.dtypes.append(dtype)
        self.logger.info(
            f"Loaded TensorRT engine: {self.plan_path}.\nInputs: {self.input_names}\nOutputs: {self.output_names}"
        )

    def allocate_buffers(self, device):
        """
        Allocates outputs to run TRT engine
        Args:
            device: GPU device to allocate memory on
        """
        ctx = self.context

        for i, binding in enumerate(self.output_names):
            shape = list(ctx.get_tensor_shape(binding))
            if binding not in self.tensors or list(self.tensors[binding].shape) != shape:
                t = torch.empty(shape, dtype=self.dtypes[i], device=device).contiguous()
                self.tensors[binding] = t
                ctx.set_tensor_address(binding, t.data_ptr())

    def set_inputs(self, feed_dict, stream):
        """
        Sets input bindings for TRT engine according to feed_dict
        Args:
           feed_dict: a dictionary [str->Tensor]
           stream: CUDA stream to use
        """
        e = self.engine
        ctx = self.context

        last_profile = self.cur_profile

        def try_set_inputs():
            for binding in self.input_names:
                t = feed_dict.get(self.input_table[binding], None)
                if t is not None:
                    t = t.contiguous()
                    shape = t.shape
                    ctx.set_input_shape(binding, shape)
                    ctx.set_tensor_address(binding, t.data_ptr())

        while True:
            try:
                try_set_inputs()
                break
            except ShapeError:
                next_profile = (self.cur_profile + 1) % e.num_optimization_profiles
                if next_profile == last_profile:
                    raise
                self.cur_profile = next_profile
                ctx.set_optimization_profile_async(self.cur_profile, stream)
            except Exception:
                raise
        left = ctx.infer_shapes()
        assert len(left) == 0

    def infer(self, stream, use_cuda_graph=False):
        """
        Runs TRT engine.
        Args:
            stream: CUDA stream to run on
            use_cuda_graph: use CUDA graph. Note: requires all inputs to be the same GPU memory between calls.
        """
        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                cuassert(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                cuassert(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                cuassert(
                    cudart.cudaStreamBeginCapture(
                        stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
                    )
                )
                self.context.execute_async_v3(stream)
                graph = cuassert(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = cuassert(cudart.cudaGraphInstantiate(graph, 0))
                self.logger.info("CUDA Graph captured!")
        else:
            noerror = self.context.execute_async_v3(stream)
            cuassert(cudart.cudaStreamSynchronize(stream))
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


def make_tensor(d):
    """
    Creates a new tensor from d, returns d if d is already a tensor
    """
    return d if isinstance(d, torch.Tensor) else torch.tensor(d).cuda()


def unroll_input(input_names, input_example):
    """
    Simulates list/tuple unrolling during ONNX export
    """
    unrolled_input = {}
    for name in input_names:
        val = input_example[name]
        if val is not None:
            if isinstance(val, list) or isinstance(val, tuple):
                for i in range(len(val)):
                    unrolled_input[f"{name}_{i}"] = make_tensor(val[i])
            else:
                unrolled_input[name] = make_tensor(val)
    return unrolled_input


def parse_groups(
    ret: List[torch.Tensor], output_lists: List[List[int]]
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]:
    """
    Implements parsing of 'output_lists' arg of trt_compile().

    Args:
      ret: plain list of Tensors

      output_lists: list of output group sizes: to form some Lists/Tuples out of 'ret' List, this will be a list
                    of group dimensions, like [[], [5], [-1]] for returning Tensor, list of 5 items and dynamic list.
        Format: [[group_n] | [], ...]
          [] or group_n == 0 : next output from ret is a scalar
          group_n > 0  :       next output from ret is a list of group_n length
          group_n == -1:       next output is a dynamic list. This entry can be at any
                               position in output_lists, but can appear only once.
    Returns:
       Tuple of Union[torch.Tensor, List[torch.Tensor]], according to the grouping in output_lists

    """
    groups: Tuple[Union[torch.Tensor, List[torch.Tensor]], ...] = tuple()
    cur = 0
    for i in range(len(output_lists)):
        gl = output_lists[i]
        assert len(gl) == 0 or len(gl) == 1
        if len(gl) == 0 or gl[0] == 0:
            groups = (*groups, ret[cur])
            cur = cur + 1
        elif gl[0] > 0:
            groups = (*groups, ret[cur : cur + gl[0]])
            cur = cur + gl[0]
        elif gl[0] == -1:
            rev_groups: Tuple[Union[torch.Tensor, List[torch.Tensor]], ...] = tuple()
            rcur = len(ret)
            for rl in range(len(output_lists) - 1, i, -1):
                rgl = output_lists[rl]
                assert len(rgl) == 0 or len(rgl) == 1
                if len(rgl) == 0 or rgl[0] == 0:
                    rcur = rcur - 1
                    rev_groups = (*rev_groups, ret[rcur])
                elif rgl[0] > 0:
                    rcur = rcur - rgl[0]
                    rev_groups = (*rev_groups, ret[rcur : rcur + rgl[0]])
                else:
                    raise ValueError("Two -1 lists in output")
            groups = (*groups, ret[cur:rcur], *rev_groups[::-1])
            break
    return groups


class TrtCompiler:
    """
    This class implements:
      - TRT lazy persistent export
      - Running TRT with optional fallback to Torch
        (for TRT engines with limited profiles)
    """

    def __init__(
        self,
        model,
        plan_path,
        precision="fp16",
        method="onnx",
        input_names=None,
        output_names=None,
        output_lists=None,
        export_args=None,
        build_args=None,
        input_profiles=None,
        dynamic_batchsize=None,
        use_cuda_graph=False,
        timestamp=None,
        fallback=False,
        forward_override=None,
        logger=None,
    ):
        """
        Initialization method:
         Tries to load persistent serialized TRT engine
         Saves its arguments for lazy TRT build on first forward() call
        Args:
            model: Model to "wrap".
            plan_path : Path where to save persistent serialized TRT engine.
            precision: TRT builder precision o engine model. Should be 'fp32'|'tf32'|'fp16'|'bf16'.
            method: One of 'onnx'|'torch_trt'.
                    Default is 'onnx' (torch.onnx.export()->TRT). This is the most stable and efficient option.
                    'torch_trt' may not work for some nets. Also AMP must be turned off for it to work.
            input_names: Optional list of input names. If None, will be read from the function signature.
            output_names: Optional list of output names. Note: If not None, patched forward() will return a dictionary.
            output_lists: Optional list of output group sizes: when forward() returns Lists/Tuples, this will be a list
                          of their dimensions, like [[], [5], [-1]] for Tensor, list of 5 items and dynamic list.
            export_args: Optional args to pass to export method. See onnx.export() and Torch-TensorRT docs for details.
            build_args: Optional args to pass to TRT builder. See polygraphy.Config for details.
            input_profiles: Optional list of profiles for TRT builder and ONNX export.
                            Each profile is a map of the form : {"input id" : [min_shape, opt_shape, max_shape], ...}.
            dynamic_batchsize: A sequence with three elements to define the input batch size range for the model to be
                               converted. Should be a sequence like [MIN_BATCH, OPT_BATCH, MAX_BATCH].
            [note]: If neither input_profiles nor dynamic_batchsize specified, static shapes will be used.
            use_cuda_graph: Use CUDA Graph for inference. Note: inputs have to be the same GPU memory between calls!
            timestamp: Optional timestamp to rebuild TRT engine (e.g. if config file changes).
            fallback: Allow to fall back to Pytorch when TRT inference fails (e.g, shapes exceed max profile).
        """

        method_vals = ["onnx", "torch_trt"]
        if method not in method_vals:
            raise ValueError(f"trt_compile(): 'method' should be one of {method_vals}, got: {method}.")
        precision_vals = ["fp32", "tf32", "fp16", "bf16"]
        if precision not in precision_vals:
            raise ValueError(f"trt_compile(): 'precision' should be one of {precision_vals}, got: {precision}.")

        self.plan_path = plan_path
        self.precision = precision
        self.method = method
        self.return_dict = output_names is not None
        self.output_names = output_names or []
        self.output_lists = output_lists or []
        self.profiles = input_profiles or []
        self.dynamic_batchsize = dynamic_batchsize
        self.export_args = export_args or {}
        self.build_args = build_args or {}
        self.engine: TRTEngine | None = None
        self.use_cuda_graph = use_cuda_graph
        self.fallback = fallback
        self.disabled = False

        self.logger = logger or getLogger("trt_compile")
        self.argspec = inspect.getfullargspec(model.forward)
        # Normally we read input_names from forward() but can be overridden
        if input_names is None:
            input_names = self.argspec.args[1:]
        self.defaults = {}
        if self.argspec.defaults is not None:
            for i in range(len(self.argspec.defaults)):
                d = self.argspec.defaults[-i - 1]
                if d is not None:
                    d = make_tensor(d)
                    self.defaults[self.argspec.args[-i - 1]] = d

        self.input_names = input_names
        self.old_forward = model.forward

        # Force engine rebuild if older than the timestamp
        if timestamp is not None and os.path.exists(self.plan_path) and os.path.getmtime(self.plan_path) < timestamp:
            os.remove(self.plan_path)

    def _inputs_to_dict(self, input_example):
        trt_inputs = {}
        for i, inp in enumerate(input_example):
            input_name = self.input_names[i]
            trt_inputs[input_name] = inp
        return trt_inputs

    def _load_engine(self):
        """
        Loads TRT plan from disk and activates its execution context.
        """
        try:
            self.engine = TRTEngine(self.plan_path, self.logger)
            # Make sure we have names correct
            input_table = {}
            for name in self.engine.input_names:
                if name.startswith("__") and name not in self.input_names:
                    orig_name = name[2:]
                else:
                    orig_name = name
                input_table[name] = orig_name
            self.engine.input_table = input_table
            self.logger.info(f"Engine loaded, inputs:{self.engine.input_table}")
        except Exception as e:
            self.logger.info(f"Exception while loading the engine:\n{e}")

    def forward(self, model, argv, kwargs):
        """
        Main forward method:
         Builds TRT engine if not available yet.
         Tries to run TRT engine
         If exception thrown and self.callback==True: falls back to original Pytorch

        Args: Passing through whatever args wrapped module's forward() has
        Returns: Passing through wrapped module's forward() return value(s)

        """
        args = self.defaults
        args.update(kwargs)
        if len(argv) > 0:
            args.update(self._inputs_to_dict(argv))

        if self.engine is None and not self.disabled:
            # Restore original forward for export
            new_forward = model.forward
            model.forward = self.old_forward
            try:
                self._load_engine()
                if self.engine is None:
                    build_args = args.copy()
                    with torch.no_grad():
                        self._build_and_save(model, build_args)
                        # This will reassign input_names from the engine
                    self._load_engine()
                    assert self.engine is not None
            except Exception as e:
                if self.fallback:
                    self.logger.info(f"Failed to build engine: {e}")
                    self.disabled = True
                else:
                    raise e
            if not self.disabled and not self.fallback:
                # Delete all parameters
                for param in model.parameters():
                    del param
                # Call empty_cache to release GPU memory
                torch.cuda.empty_cache()
            # restore TRT hook
            model.forward = new_forward
        # Run the engine
        try:
            if self.engine is not None:
                # forward_trt is not thread safe as we do not use per-thread execution contexts
                with lock_sm:
                    device = torch.cuda.current_device()
                    stream = torch.cuda.Stream(device=device)
                    self.engine.set_inputs(unroll_input(self.input_names, args), stream.cuda_stream)
                    self.engine.allocate_buffers(device=device)
                    # Need this to synchronize with Torch stream
                    stream.wait_stream(torch.cuda.current_stream())
                    ret = self.engine.infer(stream.cuda_stream, use_cuda_graph=self.use_cuda_graph)
                    # if output_names is not None, return dictionary
                    if not self.return_dict:
                        ret = list(ret.values())
                        if self.output_lists:
                            ret = parse_groups(ret, self.output_lists)
                        elif len(ret) == 1:
                            ret = ret[0]
                    return ret
        except Exception as e:
            if self.fallback:
                self.logger.info(f"Exception: {e}\nFalling back to Pytorch ...")
            else:
                raise e
        return self.old_forward(*argv, **kwargs)

    def _onnx_to_trt(self, onnx_path):
        """
        Builds TRT engine from ONNX file at onnx_path and saves to self.plan_path
        """

        profiles = []
        for profile in self.profiles:
            p = Profile()
            for id, val in profile.items():
                p.add(id, min=val[0], opt=val[1], max=val[2])
            profiles.append(p)

        build_args = self.build_args.copy()
        build_args["tf32"] = self.precision != "fp32"
        if self.precision == "fp16":
            build_args["fp16"] = True
        elif self.precision == "bf16":
            build_args["bf16"] = True

        self.logger.info(f"Building TensorRT engine for {onnx_path}: {self.plan_path}")
        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        return engine_bytes_from_network(network, config=CreateConfig(profiles=profiles, **build_args))

    def _build_and_save(self, model, input_example):
        """
        If TRT engine is not ready, exports model to ONNX,
        builds TRT engine and saves serialized TRT engine to the disk.
        Args:
             input_example: passed to onnx.export()
        """

        if self.engine is not None:
            return

        export_args = self.export_args
        engine_bytes = None

        add_casts_around_norms(model)
        replace_for_export(model)

        if self.method == "torch_trt":
            enabled_precisions = [torch.float32]
            if self.precision == "fp16":
                enabled_precisions.append(torch.float16)
            elif self.precision == "bf16":
                enabled_precisions.append(torch.bfloat16)
            inputs = list(input_example.values())

            def get_torch_trt_input(input_shape, dynamic_batchsize):
                min_input_shape, opt_input_shape, max_input_shape = get_profile_shapes(input_shape, dynamic_batchsize)
                return torch_tensorrt.Input(
                    min_shape=min_input_shape, opt_shape=opt_input_shape, max_shape=max_input_shape
                )

            tt_inputs = [get_torch_trt_input(i.shape, self.dynamic_batchsize) for i in inputs]
            engine_bytes = torch_tensorrt.convert_method_to_trt_engine(
                model,
                "forward",
                arg_inputs=tt_inputs,
                enabled_precisions=enabled_precisions,
                **export_args,
            )
        else:
            dbs = self.dynamic_batchsize
            if dbs:
                if len(self.profiles) > 0:
                    raise ValueError("ERROR: Both dynamic_batchsize and input_profiles set for TrtCompiler!")
                if len(dbs) != 3:
                    raise ValueError("dynamic_batchsize has to have len ==3 ")
                profile = {}
                for id, val in input_example.items():

                    def add_profile(id, val):
                        sh = val.shape
                        if len(sh) > 0:
                            sh = sh[1:]
                            profile[id] = [[dbs[0], *sh], [dbs[1], *sh], [dbs[2], *sh]]

                    if isinstance(val, list) or isinstance(val, tuple):
                        for i in range(len(val)):
                            add_profile(f"{id}_{i}", val[i])
                    elif isinstance(val, torch.Tensor):
                        add_profile(id, val)
                self.profiles = [profile]

            self.dynamic_axes = get_dynamic_axes(self.profiles)

            if len(self.dynamic_axes) > 0:
                export_args.update({"dynamic_axes": self.dynamic_axes})

            # Use temporary directory for easy cleanup in case of external weights
            with tempfile.TemporaryDirectory() as tmpdir:
                if export_args.get("dynamo", False):
                    input_names = None
                else:
                    input_names = list(unroll_input(self.input_names, input_example).keys())
                onnx_path = str(Path(tmpdir) / "model.onnx")
                self.logger.info(
                    f"Exporting to {onnx_path}:\n"
                    + f"output_names={self.output_names}\ninput_names={self.input_names}\nexport args: {export_args}"
                )
                torch.onnx.export(
                    model,
                    (input_example,),
                    onnx_path,
                    input_names=input_names,
                    output_names=self.output_names,
                    **export_args,
                )
                if polygraphy_imported:
                    from polygraphy.backend.onnx.loader import fold_constants, onnx_from_path, save_onnx

                    onnx_model = fold_constants(onnx_from_path(onnx_path), size_threshold=16 * 1000 * 1000)
                    save_onnx(onnx_model, onnx_path)
                self.logger.info("Export to ONNX successful.")
                engine_bytes = self._onnx_to_trt(onnx_path)
        if engine_bytes:
            open(self.plan_path, "wb").write(engine_bytes)


def trt_forward(self, *argv, **kwargs):
    """
    Patch function to replace original model's forward() with.
    Redirects to TrtCompiler.forward()
    """
    return self._trt_compiler.forward(self, argv, kwargs)


def trt_compile(
    model: torch.nn.Module,
    base_path: str,
    args: Dict[str, Any] | None = None,
    submodule: Union[str, List[str]] | None = None,
    logger: Any | None = None,
) -> torch.nn.Module:
    """
    Instruments model or submodule(s) with TrtCompiler and replaces its forward() with TRT hook.
    Note: TRT 10.3 is recommended for best performance. Some nets may even fail to work with TRT 8.x
    Args:
      model: module to patch with TrtCompiler object.
      base_path: TRT plan(s) saved to f"{base_path}[.{submodule}].plan" path.
                 dirname(base_path) must exist, base_path does not have to.
                 If base_path does point to existing file (e.g. associated checkpoint),
                 that file becomes a dependency - its mtime is added to args["timestamp"].
      args: Optional dict : unpacked and passed to TrtCompiler() - see TrtCompiler above for details.
      submodule: Optional hierarchical id(s) of submodule to patch, e.g. ['image_decoder.decoder']
                  If None, TrtCompiler patch is applied to the whole model.
                  Otherwise, submodule (or list of) is being patched.
      logger: Optional logger for diagnostics.
    Returns:
      Always returns same model passed in as argument. This is for ease of use in configs.
    """

    default_args: Dict[str, Any] = {
        "method": "onnx",
        "precision": "fp16",
        "build_args": {"builder_optimization_level": 5, "precision_constraints": "obey"},
    }

    default_args.update(args or {})
    args = default_args

    if trt_imported and polygraphy_imported and torch.cuda.is_available():
        # if "path" filename point to existing file (e.g. checkpoint)
        # it's also treated as dependency
        if os.path.exists(base_path):
            timestamp = int(os.path.getmtime(base_path))
            if "timestamp" in args:
                timestamp = max(int(args["timestamp"]), timestamp)
            args["timestamp"] = timestamp

        def wrap(model, path):
            if not hasattr(model, "_trt_compiler"):
                model.orig_forward = model.forward
                wrapper = TrtCompiler(model, path + ".plan", logger=logger, **args)
                model._trt_compiler = wrapper
                model.forward = MethodType(trt_forward, model)

        def find_sub(parent, submodule):
            idx = submodule.find(".")
            # if there is "." in name, call recursively
            if idx != -1:
                parent_name = submodule[:idx]
                parent = getattr(parent, parent_name)
                submodule = submodule[idx + 1 :]
                return find_sub(parent, submodule)
            return parent, submodule

        if submodule is not None:
            if isinstance(submodule, str):
                submodule = [submodule]
            for s in submodule:
                parent, sub = find_sub(model, s)
                wrap(getattr(parent, sub), base_path + "." + s)
        else:
            wrap(model, base_path)
    else:
        logger = logger or getLogger("trt_compile")
        logger.warning("TensorRT and/or polygraphy packages are not available! trt_compile() has no effect.")

    return model
