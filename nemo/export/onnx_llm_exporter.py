# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import tensorrt as trt
import torch
import wrapt
from transformers import AutoModel, AutoTokenizer

from nemo.deploy import ITritonDeployable
from nemo.export.utils import get_example_inputs, get_model_device_type, is_nemo2_checkpoint, validate_fp8_network
from nemo.utils import logging


@wrapt.decorator
def noop_decorator(func):
    """No op decorator"""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch
except Exception:
    logging.warning("PyTriton is not available.")
    use_pytriton = False


use_onnxruntime = True
try:
    import onnxruntime
except Exception:
    logging.warning("onnxruntime is not available.")
    use_onnxruntime = False


# pylint: disable=line-too-long
class OnnxLLMExporter(ITritonDeployable):
    """
    Exports models to ONNX and run fast inference.

    Example:
        from nemo.export.onnx_llm_exporter import OnnxLLMExporter

        onnx_llm_exporter = OnnxLLMExporter(
            onnx_model_dir="/path/for/onnx_model/files",
            model_name_or_path="/path/for/model/files",
        )

        onnx_llm_exporter.export(
            input_names=["input_ids", "attention_mask", "dimensions"],
            output_names=["embeddings"],
        )

        output = onnx_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
        print("output: ", output)
    """

    def __init__(
        self,
        onnx_model_dir: str,
        model: Optional[torch.nn.Module] = None,
        tokenizer=None,
        model_name_or_path: str = None,
        load_runtime: bool = True,
    ):
        """
        Initializes the ONNX Exporter.

        Args:
            onnx_model_dir (str): path for storing the ONNX model files.
            model (Optional[torch.nn.Module]): torch model.
            tokenizer (HF or NeMo tokenizer): tokenizer class.
            model_name_or_path (str): a path for ckpt or HF model ID
            load_runtime (bool): load ONNX runtime if there is any exported model available in
                                 the onnx_model_dir folder.
        """
        self.onnx_model_dir = onnx_model_dir
        self.model_name_or_path = model_name_or_path
        self.onnx_model_path = str(Path(onnx_model_dir) / "model.onnx")
        self.model = model
        self.tokenizer = tokenizer
        self.model_input_names = None
        self.model_output_names = None
        self.onnx_runtime_session = None
        self.calibration_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quant_max_batch_size = None

        if self.model_name_or_path is not None:
            if model is not None:
                raise ValueError("A model was also passed but it will be overridden.")

            if Path(self.model_name_or_path).is_dir():
                if is_nemo2_checkpoint(self.model_name_or_path):
                    raise NotImplementedError("NeMo 2.0 checkpoint will be supported later.")
                else:
                    self._load_hf_model()

        if load_runtime:
            self._load_runtime()

    def _load_runtime(self):
        if use_onnxruntime:
            if Path(self.onnx_model_path).exists():
                self.onnx_runtime_session = onnxruntime.InferenceSession(self.onnx_model_path)
                self.model_input_names = [input.name for input in self.onnx_runtime_session.get_inputs()]
                self.model_output_names = [output.name for output in self.onnx_runtime_session.get_outputs()]
                self.tokenizer = AutoTokenizer.from_pretrained(
                    Path(self.onnx_model_dir) / "tokenizer", trust_remote_code=True
                )

    def _load_hf_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

    def export(
        self,
        input_names: list,
        output_names: list,
        example_inputs: dict = None,
        opset: int = 20,
        dynamic_axes_input: Optional[dict] = None,
        dynamic_axes_output: Optional[dict] = None,
        export_dtype: str = "fp32",
        verbose: bool = False,
    ):
        """
        Performs ONNX conversion from a PyTorch model.

        Args:
            input_names (list): input parameter names of the model that ONNX will export will use.
            output_names (list): output parameter names of the model that ONNX will export will use.
            example_inputs (dict): example input for the model to build the engine.
            opset (int): ONNX opset version. Default is 20.
            dynamic_axes_input (dict): Variable length axes for the input.
            dynamic_axes_output (dict): Variable length axes for the output.
            export_dtype (str): Export dtype, fp16 or fp32.
            verbose (bool): Enable verbose or not.
        """

        self._export_to_onnx(
            input_names=input_names,
            example_inputs=example_inputs,
            output_names=output_names,
            opset=opset,
            dynamic_axes_input=dynamic_axes_input,
            dynamic_axes_output=dynamic_axes_output,
            export_dtype=export_dtype,
            verbose=verbose,
        )
        self._load_runtime()

    def _export_to_onnx(
        self,
        input_names: list,
        output_names: list,
        example_inputs: dict = None,
        opset: int = 20,
        dynamic_axes_input: Optional[dict] = None,
        dynamic_axes_output: Optional[dict] = None,
        export_dtype: Union[torch.dtype, str] = "fp32",
        verbose: bool = False,
    ):

        if example_inputs is None:
            example_inputs = get_example_inputs(self.tokenizer)

        if "dimensions" in input_names:
            example_inputs["dimensions"] = torch.tensor([1] * example_inputs["input_ids"].shape[0])

        if isinstance(export_dtype, str):
            export_dtype = {"fp16": torch.float16, "fp32": torch.float32}[export_dtype]

        self.model.to(export_dtype)

        Path(self.onnx_model_dir).mkdir(parents=True, exist_ok=True)

        with torch.autocast(device_type=get_model_device_type(self.model), dtype=export_dtype):
            torch.onnx.export(
                model=self.model,
                args=(example_inputs,),
                f=self.onnx_model_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={**dynamic_axes_input, **dynamic_axes_output},
                verbose=verbose,
                opset_version=opset,
            )
        logging.info(f"Successfully exported PyTorch model to ONNX model {self.onnx_model_path}")

        existing_directory_path = Path(self.onnx_model_dir) / "tokenizer"
        existing_directory_path.mkdir(exist_ok=True)
        self.tokenizer.save_pretrained(existing_directory_path)

    def export_onnx_to_trt(
        self,
        trt_model_dir: str,
        profiles=None,
        override_layernorm_precision_to_fp32: bool = False,
        override_layers_to_fp32: List = None,
        trt_dtype: str = "fp16",
        profiling_verbosity: str = "layer_names_only",
        trt_builder_flags: List[trt.BuilderFlag] = None,
    ) -> None:
        """Performs TensorRT conversion from an ONNX model.

        Args:
            trt_model_dir: path to store the TensorRT model.
            profiles: TensorRT profiles.
            override_layernorm_precision_to_fp32 (bool): whether to convert layers to fp32 or not.
            override_layers_to_fp32 (List): Layer names to be converted to fp32.
            trt_dtype (str): "fp16" or "fp32".
            profiling_verbosity (str): Profiling verbosity. Default is "layer_names_only".
            trt_builder_flags (List[trt.BuilderFlag]): TRT specific flags.
        """
        logging.info(f"Building TRT engine from ONNX model ({self.onnx_model_path})")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, trt_logger)

        # we use parse_from_file() instead of parse() because it can be used for both single
        # file models as well as externally stored models (required when model >2GiB)
        if not parser.parse_from_file(self.onnx_model_path):
            logging.warning("ONNX model could not be parsed")
            for error in range(parser.num_errors):
                logging.error(parser.get_error(error))
            return

        if profiles:
            for profile in profiles:
                optimization_profile = builder.create_optimization_profile()

                for i in range(network.num_inputs):
                    in_tensor = network.get_input(i)
                    optimization_profile.set_shape(
                        in_tensor.name,
                        min=profile[in_tensor.name][0],
                        opt=profile[in_tensor.name][1],
                        max=profile[in_tensor.name][2],
                    )

                config.add_optimization_profile(optimization_profile)

        if trt_dtype == "fp16":
            logging.info("Setting Build Flag FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        elif trt_dtype == "fp8":
            # With FP8 export we want to also enable FP16 layers as a fallback instead of FP32
            logging.info("Setting Build Flag FP8 and FP16")
            config.set_flag(trt.BuilderFlag.FP8)
            config.set_flag(trt.BuilderFlag.FP16)
            validate_fp8_network(network)

        # patch network
        if override_layernorm_precision_to_fp32:
            logging.info("Overriding TensorRT network LayerNorm precision to float32.")
            self._override_layernorm_precision_to_fp32(network)

        if override_layers_to_fp32:
            logging.info("Overriding some layers to float32.")
            self._override_layers_to_fp32(network, override_layers_to_fp32)

        try:
            config.profiling_verbosity = {
                "detailed": trt.ProfilingVerbosity.DETAILED,
                "layer_names_only": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                "none": trt.ProfilingVerbosity.NONE,
            }[profiling_verbosity]
        except KeyError:
            error_msg = "Unknown profiling verbosity value."
            raise ValueError(error_msg)
        logging.info(f"Setting Profiling Verbosity to {config.profiling_verbosity}")

        if trt_builder_flags is not None:
            for flag in trt_builder_flags:
                config.set_flag(flag)

        engine_string = builder.build_serialized_network(network, config)
        if engine_string is None:
            raise Exception("Failed to serialize the TensorRT Engine. Please check the " "TensorRT logs for details")

        trt_model_path = Path(trt_model_dir)
        trt_model_path.mkdir(parents=True, exist_ok=True)
        trt_model_path = trt_model_path / "model.plan"
        trt_model_path.write_bytes(engine_string)
        logging.info(f"Successfully exported ONNX model ({self.onnx_model_path}) " f"to TRT engine ({trt_model_path})")

    def _override_layer_precision_to_fp32(self, layer: trt.ILayer) -> None:
        layer.precision = trt.float32
        layer.set_output_type(0, trt.float32)

    def _override_layers_to_fp32(self, network: trt.INetworkDefinition, fp32_layer_patterns: list[str]) -> None:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            if any(layer_name.startswith(pattern) for pattern in fp32_layer_patterns) and layer.precision in {
                trt.float32,
                trt.float16,
            }:
                if layer.type in {trt.LayerType.CAST}:
                    logging.info(f"Skipping overriding {layer.type} layer {i} {layer_name} dtype")
                    continue
                if any(
                    layer.get_input(input_idx).dtype in {trt.float32, trt.float16}
                    for input_idx in range(layer.num_inputs)
                ):
                    # Note: Assigning to layer.precision (even the same value) sets precision_is_set=True,
                    # which prevents TensorRT from changing this layer's precision
                    layer.precision = trt.float32
                    logging.info(f"Setting layer {i} {layer_name} (type: {layer.type}) precision to FP32")
                for j in range(layer.num_outputs):
                    if layer.get_output_type(j) in {trt.float32, trt.float16}:
                        layer.set_output_type(j, trt.float32)
                        logging.info(f"Setting layer {i} {layer_name} (type: {layer.type}) output type {j} to FP32")

    def _override_layernorm_precision_to_fp32(self, network: trt.INetworkDefinition) -> None:
        """Set the precision of LayerNorm subgraphs to FP32 to preserve accuracy.

        - https://nvbugs/4478448 (Mistral)
        - https://nvbugs/3802112 (T5)

        Args:
            network: tensorrt.INetworkDefinition
        """
        # Logic originally from OSS T5 HF export script:
        # https://gitlab-master.nvidia.com/TensorRT/Public/oss/-/blob/77495ec/demo/HuggingFace/T5/export.py
        pow_ops = {}
        for layer_index, layer in enumerate(network):
            if layer.type == trt.LayerType.IDENTITY:
                all_fp32 = all(
                    [
                        layer.output_type_is_set(o) and layer.get_output_type(o) == trt.float32
                        for o in range(layer.num_outputs)
                    ]
                )
                if all_fp32:
                    if layer.get_input(0).dtype == trt.float32:
                        layer.precision = trt.float32

            if layer.type == trt.LayerType.ELEMENTWISE:
                layer.__class__ = getattr(trt, "IElementWiseLayer")
                if layer.op == trt.ElementWiseOperation.POW:
                    pow_ops[layer] = layer_index
                    self._override_layer_precision_to_fp32(layer)

        for _, index in pow_ops.items():
            # Iterate from few layers before pow to include residual add and cast op.
            # Iterate till 10 layers after pow op to include all
            # operations included in layer norm.
            START_OFFSET = 4
            END_OFFSET = 12
            for i in range(index - START_OFFSET, index + END_OFFSET):
                layer = network.get_layer(i)
                if layer.type == trt.LayerType.REDUCE:
                    self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.ELEMENTWISE:
                    layer.__class__ = getattr(trt, "IElementWiseLayer")
                    if layer.op == trt.ElementWiseOperation.SUM:
                        self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.UNARY:
                    layer.__class__ = getattr(trt, "IUnaryLayer")
                    if layer.op == trt.UnaryOperation.SQRT:
                        self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.ELEMENTWISE:
                    layer.__class__ = getattr(trt, "IElementWiseLayer")
                    if layer.op == trt.ElementWiseOperation.DIV:
                        self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.ELEMENTWISE:
                    layer.__class__ = getattr(trt, "IElementWiseLayer")
                    if layer.op == trt.ElementWiseOperation.PROD:
                        self._override_layer_precision_to_fp32(layer)

    def forward(self, inputs: Union[List, Dict], dimensions: Optional[List] = None):
        """Run inference for a given input.

        Args:
            inputs (Union[List, Dict]): Input for the model. If list, it should be a list of strings.
                If dict, it should be a dictionary with keys as the model input names.
            dimensions (Optional[List]): The dimensions parameter of the model. Required if the model
                was exported to accept dimensions parameter and inputs is given as a list of strings.

        Returns:
            np.ndarray: Model output.
        """

        if self.onnx_runtime_session is None:
            warnings.warn("ONNX Runtime is not available. Please install the onnxruntime-gpu and try again.")
            return None

        if isinstance(inputs, List):
            if "dimensions" in self.model_input_names and dimensions is None:
                raise ValueError("Dimensions should be provided for list input.")
            inputs = dict(self.tokenizer(inputs))
            inputs["dimensions"] = dimensions

        output = self.onnx_runtime_session.run(self.model_output_names, inputs)
        return output[0]

    @property
    def get_model(self):
        """Returns the model"""

        return self.model

    @property
    def get_tokenizer(self):
        """Returns the tokenizer"""

        return self.tokenizer

    @property
    def get_model_input_names(self):
        """Returns the model input names"""

        return self.model_input_names

    @property
    def get_triton_input(self):
        """Get triton input"""

        raise NotImplementedError("This function will be implemented later.")

    @property
    def get_triton_output(self):
        """Get triton output"""

        raise NotImplementedError("This function will be implemented later.")

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        """PyTriton inference function"""

        raise NotImplementedError("This function will be implemented later.")
