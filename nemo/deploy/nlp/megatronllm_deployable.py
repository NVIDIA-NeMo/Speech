# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from enum import IntEnum, auto
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed
import wrapt
from megatron.core.inference.common_inference_params import CommonInferenceParams
from pytorch_lightning.trainer.trainer import Trainer

import nemo.lightning as nl
from nemo.collections.llm import inference
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.text_generation_utils import (
    OutputType,
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import NEMO2, cast_output, nemo_checkpoint_version, str_ndarray2list

try:
    from megatron.core.dist_checkpointing.validation import StrictHandling

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError) as e:

    HAVE_MEGATRON_CORE = False
    IMPORT_ERROR = e


@wrapt.decorator
def noop_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False

LOGGER = logging.getLogger("NeMo")


def GetTensorShape(pyvalue):
    """
    utility function to get Triton Tensor shape from a python value
    assume that lists are shape -1 and all others are scalars with shape 1
    """
    return (-1 if type(pyvalue) == list else 1,)


def GetNumpyDtype(pyvalue):
    """
    utility function to get numpy dtype of a python value
    e.g. bool -> np.bool_
    """
    '''
    manually defining the mapping of python type -> numpy type for now
    is there a better way to do it?  tried np.array(pyvalue).dtype, but that doesn't seem to work
    '''
    py_to_numpy_mapping = {str: bytes, bool: np.bool_, float: np.single, int: np.int_}
    python_type = type(pyvalue)
    # for lists, return the type of the internal elements
    if python_type == list:
        python_type = type(pyvalue[0])
    numpy_type = py_to_numpy_mapping[python_type]
    return numpy_type


class ServerSync(IntEnum):
    """Enum for synchronization messages using torch.distributed"""

    WAIT = auto()
    SIGNAL = auto()

    def to_long_tensor(self):
        return torch.tensor([self], dtype=torch.long, device='cuda')


class MegatronLLMDeploy:

    @staticmethod
    def get_deployable(
        nemo_checkpoint_filepath: str = None,
        num_devices: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
    ):

        if nemo_checkpoint_version(nemo_checkpoint_filepath) == NEMO2:
            return MegatronLLMDeployableNemo2(
                nemo_checkpoint_filepath=nemo_checkpoint_filepath,
                num_devices=num_devices,
                num_nodes=num_nodes,
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                context_parallel_size=context_parallel_size,
            )
        else:
            return MegatronLLMDeployable(
                nemo_checkpoint_filepath=nemo_checkpoint_filepath,
                num_devices=num_devices,
                num_nodes=num_nodes,
            )


class MegatronLLMDeployableNemo2(ITritonDeployable):
    """Triton inference server compatible deploy class for a .nemo model file"""

    def __init__(
        self,
        nemo_checkpoint_filepath: str = None,
        num_devices: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        params_dtype: torch.dtype = torch.bfloat16,
        inference_batch_times_seqlen_threshold: int = 1000,
    ):
        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath

        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=False,
            setup_optimizers=False,
            store_optimizer_states=False,
        )

        trainer = nl.Trainer(
            accelerator="gpu",
            devices=num_devices,
            num_nodes=num_nodes,
            strategy=strategy,
            plugins=nl.MegatronMixedPrecision(
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                autocast_enabled=False,
                grad_reduce_in_fp32=False,
            ),
        )

        self.inference_wrapped_model, self.mcore_tokenizer = inference.setup_model_and_tokenizer(
            path=Path(nemo_checkpoint_filepath),
            trainer=trainer,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
        )

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="compute_logprob", shape=(-1,), dtype=np.bool_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        return (
            Tensor(name="sentences", shape=(-1,), dtype=bytes),
            Tensor(name="log_probs", shape=(-1,), dtype=np.single),
        )

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        output_infer = {}
        try:
            prompts = str_ndarray2list(inputs.pop("prompts"))
            max_batch_size = inputs.pop("max_batch_size")[0][0] if "max_batch_size" in inputs else 32
            random_seed = inputs.pop("random_seed")[0][0] if "random_seed" in inputs else None
            temperature = inputs.pop("temperature")[0][0] if "temperature" in inputs else 1.0
            top_k = inputs.pop("top_k")[0][0] if "top_k" in inputs else 1
            top_p = inputs.pop("top_p")[0][0] if "top_k" in inputs else 0.0
            num_tokens_to_generate = inputs.pop("max_length")[0][0] if "max_length" in inputs else 256
            log_probs = inputs.pop("compute_logprob")[0][0] if "compute_logprob" in inputs else False
            text_only = True

            inference_params = CommonInferenceParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_tokens_to_generate=num_tokens_to_generate,
                return_log_probs=log_probs,
            )

            results = inference.generate(
                model=self.inference_wrapped_model,
                tokenizer=self.mcore_tokenizer,
                prompts=prompts,
                max_batch_size=max_batch_size,
                random_seed=random_seed,
                inference_params=inference_params,
            )

            output_texts = [r.generated_text if text_only else r for r in results]
            output_infer = {"sentences": cast_output(output_texts, np.bytes_)}
            if log_probs:
                output_log_probs = []
                for r in results:
                    lp = r.generated_log_probs.cpu().detach().numpy()
                    if len(lp) == 0:
                        output_log_probs.append([0])
                    else:
                        output_log_probs.append(lp)
                output_infer["log_probs"] = np.array(output_log_probs)
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_infer["sentences"] = cast_output([err_msg], np.bytes_)

        return output_infer


class MegatronLLMDeployable(ITritonDeployable):
    """Triton inference server compatible deploy class for a .nemo model file"""

    def __init__(
        self,
        nemo_checkpoint_filepath: str = None,
        num_devices: int = 1,
        num_nodes: int = 1,
        existing_model: MegatronGPTModel = None,
    ):
        if not HAVE_MEGATRON_CORE:
            raise IMPORT_ERROR
        if nemo_checkpoint_filepath is None and existing_model is None:
            raise ValueError(
                "MegatronLLMDeployable requires either a .nemo checkpoint filepath or an existing MegatronGPTModel, but both provided were None"
            )
        if num_devices > 1:
            LOGGER.warning(
                "Creating a MegatronLLMDeployable with num_devices>1 will assume running with a PyTorch Lightning DDP-variant strategy, which will run the main script once per device. Make sure any user code is compatible with multiple executions!"
            )

        # if both existing_model and nemo_checkpoint_filepath are provided, existing_model will take precedence
        if existing_model is not None:
            self.model = existing_model
        else:
            self._load_from_nemo_checkpoint(nemo_checkpoint_filepath, num_devices, num_nodes)

        self.model.eval()

    def _load_from_nemo_checkpoint(self, nemo_checkpoint_filepath: str, num_devices: int, num_nodes: int):
        if Path(nemo_checkpoint_filepath).exists():
            trainer = Trainer(
                strategy=NLPDDPStrategy(),
                devices=num_devices,
                num_nodes=num_nodes,
            )

            custom_config = MegatronGPTModel.restore_from(
                nemo_checkpoint_filepath, trainer=trainer, return_config=True
            )
            # transformer_engine should always be true according to EricH, but GPT-2B model will fail if it is enabled
            if not custom_config.transformer_engine:
                LOGGER.warning(
                    "MegatronLLMDeployable expects model config transformer_engine=True, but this model has it =False. "
                    "Overriding it to =True, but this may break certain checkpoints converted on older Nemo versions. "
                    "If your model breaks, please try re-converting the checkpoint on the current Nemo version."
                )
            custom_config.transformer_engine = True
            # using multi-gpu for tensor parallelism directly for now, could do pipeline parallel instead or a combination
            custom_config.tensor_model_parallel_size = num_devices
            # had to override these to make Nemotron3-22B work, see sample_sequence_batch() in text_generation_utils.py
            custom_config.activations_checkpoint_granularity = None
            custom_config.activations_checkpoint_method = None
            # Models trained with TE < 1.10 and loaded with TE >= 1.10 require
            # special handling on loading checkpoint due to structural updates
            custom_config.dist_ckpt_load_strictness = StrictHandling.LOG_ALL.value
            if custom_config.get("fp8", False):
                # Need to disable FP8 for in-framework inference due to shape constraints imposed by TE,
                # see https://github.com/NVIDIA/TransformerEngine/blob/v1.10/transformer_engine/pytorch/utils.py#L229
                LOGGER.warning("Disabling FP8 inference due to shape constraints imposed by Transformer Engine.")
                custom_config.fp8 = False

            self.model = MegatronGPTModel.restore_from(
                nemo_checkpoint_filepath, trainer=trainer, override_config_path=custom_config
            )

    _INPUT_PARAMETER_FIELDS = {
        "prompts": (-1, bytes, False),
    }

    '''
    there is no get_default equivalent for OutputType like there is for SamplingParameters and LengthParameters
    but we still want to generate output using a real OutputType TypedDict for static type checking
    '''
    _BLANK_OUTPUTTYPE: OutputType = {
        'sentences': [""],
        'tokens': [[""]],
        'logprob': [[0.0]],
        'full_logprob': [[0.0]],
        'token_ids': [[0]],
        'offsets': [[0]],
    }

    @property
    def get_triton_input(self):
        input_parameters = tuple(
            Tensor(name=name, shape=(shape,), dtype=dtype, optional=optional)
            for name, (shape, dtype, optional) in self._INPUT_PARAMETER_FIELDS.items()
        )
        default_sampling_params: SamplingParam = get_default_sampling_params()
        sampling_parameters = tuple(
            Tensor(
                name=parameter_name,
                shape=GetTensorShape(parameter_value),
                dtype=GetNumpyDtype(parameter_value),
                optional=True,
            )
            for parameter_name, parameter_value in default_sampling_params.items()
        )
        default_length_params: LengthParam = get_default_length_params()
        length_parameters = tuple(
            Tensor(
                name=parameter_name,
                shape=GetTensorShape(parameter_value),
                dtype=GetNumpyDtype(parameter_value),
                optional=True,
            )
            for parameter_name, parameter_value in default_length_params.items()
        )

        inputs = input_parameters + sampling_parameters + length_parameters
        return inputs

    @property
    def get_triton_output(self):
        # outputs are defined by the fields of OutputType
        outputs = [
            Tensor(
                name=parameter_name,
                shape=GetTensorShape(parameter_value),
                dtype=GetNumpyDtype(parameter_value[0]),
            )
            for parameter_name, parameter_value in MegatronLLMDeployable._BLANK_OUTPUTTYPE.items()
        ]
        return outputs

    @staticmethod
    def _sampling_params_from_triton_inputs(**inputs: np.ndarray):
        """Extract SamplingParam fields from triton input dict"""
        sampling_params: SamplingParam = get_default_sampling_params()
        for sampling_param_field in sampling_params.keys():
            if sampling_param_field in inputs:
                sampling_params[sampling_param_field] = inputs.pop(sampling_param_field)[0][0]
        return sampling_params

    @staticmethod
    def _length_params_from_triton_inputs(**inputs: np.ndarray):
        """Extract LengthParam fields from triton input dict"""
        length_params: LengthParam = get_default_length_params()
        for length_param_field in length_params.keys():
            if length_param_field in inputs:
                length_params[length_param_field] = inputs.pop(length_param_field)[0][0]
        return length_params

    def generate(self, inputs: List[str], length_params: LengthParam, sampling_params: SamplingParam):
        if torch.distributed.is_initialized():
            distributed_rank = torch.distributed.get_rank()
            if distributed_rank != 0:
                raise ValueError(
                    f"Triton inference function should not be called on a thread with torch.distributed rank != 0, but this thread is rank {distributed_rank}"
                )
            signal_value = ServerSync.SIGNAL.to_long_tensor()
            torch.distributed.broadcast(signal_value, 0)

        return self.model.generate(inputs=inputs, length_params=length_params, sampling_params=sampling_params)

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        """Triton server inference function that actually runs the model"""
        input_strings = str_ndarray2list(inputs.pop("prompts"))
        sampling_params = self._sampling_params_from_triton_inputs(**inputs)
        length_params = self._length_params_from_triton_inputs(**inputs)

        model_output = self.generate(input_strings, length_params, sampling_params)
        '''
            model_output['sentences'] will be a list of strings (one per prompt)
            other fields will either be a list of lists (tokens, for example)
            or a list of pytorch Tensor
        '''

        triton_output = {}
        _OUTPUT_FILLER_VALUES = {
            'tokens': "",
            'logprob': 0.0,
            'full_logprob': 0.0,
            'token_ids': -1,
            'offsets': -1,
        }
        for model_output_field, value in model_output.items():

            if model_output_field != 'sentences' and value is not None:
                # find length of longest non-sentence output item
                field_longest_output_item = 0
                for item in value:
                    field_longest_output_item = max(field_longest_output_item, len(item))
                # then pad shorter items to match this length
                for index, item in enumerate(value):
                    num_pad_values = field_longest_output_item - len(item)
                    if num_pad_values > 0:
                        pad_value = _OUTPUT_FILLER_VALUES[model_output_field]
                        if isinstance(item, torch.Tensor):
                            pad_tensor = torch.full(
                                (num_pad_values, item.size(1)) if item.dim() > 1 else (num_pad_values,),
                                pad_value,
                                dtype=item.dtype,
                                device='cuda',
                            )
                            padded_item = torch.cat((item, pad_tensor))
                            value[index] = padded_item
                        else:
                            pad_list = [pad_value] * num_pad_values
                            padded_item = item + pad_list
                            value[index] = padded_item

            field_dtype = GetNumpyDtype(MegatronLLMDeployable._BLANK_OUTPUTTYPE[model_output_field][0])
            if value is None:
                # triton does not allow for optional output parameters, so need to populate them if they don't exist
                triton_output[model_output_field] = np.full(
                    # 'sentences' should always have a valid value, so use that for the output shape
                    np.shape(model_output['sentences']),
                    MegatronLLMDeployable._BLANK_OUTPUTTYPE[model_output_field][0],
                    dtype=field_dtype,
                )
            elif field_dtype == bytes:
                # strings are cast to bytes
                triton_output[model_output_field] = cast_output(value, field_dtype)
            elif isinstance(value[0], torch.Tensor):
                if value[0].dtype == torch.bfloat16:
                    # numpy currently does not support bfloat16, so need to manually convert it
                    triton_output[model_output_field] = np.array([tensor.cpu().float().numpy() for tensor in value])
                else:
                    triton_output[model_output_field] = np.array([tensor.cpu().numpy() for tensor in value])
            else:
                # non-strings are output as-is (in numpy format)
                triton_output[model_output_field] = np.array(value)
        return triton_output
