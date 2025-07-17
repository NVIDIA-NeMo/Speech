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

import gc
import json
import logging
import os
import pickle
import shutil
import tempfile
import warnings
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import safetensors
import tensorrt_llm
import torch
import torch.nn.functional as F
import wrapt
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm._utils import numpy_to_torch
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.build import build as build_trtllm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (
    BaichuanForCausalLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertModel,
    BloomForCausalLM,
    ChatGLMForCausalLM,
    CogVLMForCausalLM,
    CohereForCausalLM,
    DbrxForCausalLM,
    DeciLMForCausalLM,
    DecoderModel,
    DeepseekForCausalLM,
    DeepseekV2ForCausalLM,
    DiT,
    EagleForCausalLM,
    EncoderModel,
    FalconForCausalLM,
    GemmaForCausalLM,
    GPTForCausalLM,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    GrokForCausalLM,
    LLaMAForCausalLM,
    MambaForCausalLM,
    MedusaForCausalLm,
    MLLaMAForCausalLM,
    MPTForCausalLM,
    OPTForCausalLM,
    Phi3ForCausalLM,
    PhiForCausalLM,
    QWenForCausalLM,
    RecurrentGemmaForCausalLM,
    ReDrafterForCausalLM,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaModel,
    WhisperEncoder,
)
from tensorrt_llm.plugin import PluginConfig
from transformers import PreTrainedTokenizerBase

from nemo.deploy import ITritonDeployable
from nemo.export.tarutils import TarPath
from nemo.export.trt_llm.converter.model_converter import determine_quantization_settings, model_to_trtllm_ckpt
from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import dist_model_to_trt_llm_ckpt, get_layer_prefix
from nemo.export.trt_llm.converter.utils import init_model_parallel_from_nemo
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import (
    build_tokenizer,
    get_model_type,
    get_tokenizer,
    get_weights_dtype,
    load_nemo_model,
)
from nemo.export.trt_llm.qnemo import qnemo_to_tensorrt_llm
from nemo.export.trt_llm.qnemo.tokenizer_utils import TOKENIZER_CONFIG_FILE, get_nmt_tokenizer
from nemo.export.trt_llm.qnemo.utils import is_qnemo_checkpoint
from nemo.export.trt_llm.tensorrt_llm_build import build_and_save_engine
from nemo.export.trt_llm.tensorrt_llm_run import (
    generate,
    generate_streaming,
    load,
    load_distributed,
    refit,
    unload_engine,
)
from nemo.export.trt_llm.utils import is_rank
from nemo.export.utils import is_nemo_tarfile, prepare_directory_for_export, torch_dtype_from_precision
from nemo.export.utils.constants import TRTLLM_ENGINE_DIR

use_deploy = True
try:
    from nemo.deploy.utils import cast_output, str_ndarray2list
except Exception:
    use_deploy = False

LOGGER = logging.getLogger("NeMo")


@wrapt.decorator
def noop_decorator(func):
    """No op decorator"""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch, first_value
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False


# pylint: disable=line-too-long
class TensorRTLLM(ITritonDeployable):
    """
    Exports nemo and huggingface checkpoints to TensorRT-LLM and run fast inference.

    Example:
        from nemo.export.tensorrt_llm import TensorRTLLM

        trt_llm_exporter = TensorRTLLM(model_dir="/path/for/model/files")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="llama",
            tensor_parallelism_size=1,
        )

        output = trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
        print("output: ", output)

    """

    def __init__(
        self,
        model_dir: str,
        lora_ckpt_list: List[str] = None,
        load_model: bool = True,
        use_python_runtime: bool = True,
        enable_chunked_context: bool = None,
        max_tokens_in_paged_kv_cache: int = None,
        multi_block_mode: bool = False,
    ):
        """
        Args:
            model_dir (str): path for storing the TensorRT-LLM model files.
            lora_ckpt_list (List[str]): lora checkpoint paths.
            load_model (bool): load TensorRT-LLM model if the engine files exist in the model_dir.
            use_python_runtime (bool): whether to use python or c++ runtime.
            multi_block_mode (bool): enable faster decoding in multihead attention. Required for long context. Only available when using c++ runtime
        """

        if use_python_runtime:
            if enable_chunked_context is not None or max_tokens_in_paged_kv_cache is not None:
                raise Exception(
                    "enable_chunked_context and max_tokens_in_paged_kv_cache options "
                    "work only with the TensorRT-LLM C++ runtime. Please set "
                    "use_python_runtime=False to use these options."
                )

        self.model_dir = model_dir
        self.engine_dir = os.path.join(model_dir, TRTLLM_ENGINE_DIR)
        self.lora_ckpt_list = lora_ckpt_list
        self.use_python_runtime = use_python_runtime
        self.enable_chunked_context = enable_chunked_context if enable_chunked_context is not None else False
        self.max_tokens_in_paged_kv_cache = max_tokens_in_paged_kv_cache
        self.multi_block_mode = multi_block_mode
        self.model = None
        self.tokenizer = None
        self.config = None
        self.ptuning_tables = []
        self.p_table = None
        self.task_vocab_size = 0
        self.task_vtoken_counts = []
        self.task_ids = {}

        if load_model:
            self._load()

    def export(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        pipeline_parallelism_size: int = 1,
        gpus_per_node: Optional[int] = None,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_batch_size: int = 8,
        max_prompt_embedding_table_size: Optional[int] = None,
        use_parallel_embedding: bool = False,
        use_embedding_sharing: bool = False,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        paged_context_fmha: bool = False,
        dtype: Optional[str] = None,
        load_model: bool = True,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        max_lora_rank: int = 64,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 512,
        multiple_profiles: bool = False,
        gpt_attention_plugin: str = "auto",
        gemm_plugin: str = "auto",
        use_mcore_path: bool = True,
        reduce_fusion: bool = True,
        fp8_quantized: Optional[bool] = None,
        fp8_kvcache: Optional[bool] = None,
        gather_context_logits: Optional[bool] = False,
        gather_generation_logits: Optional[bool] = False,
        build_rank: Optional[int] = 0,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            nemo_checkpoint_path (str): path for the nemo checkpoint.
            model_type (Optional[str]): type of the model (optional for NeMo 2.0 and quantized checkpoints).
            delete_existing_files (bool): if True, deletes all the files in model_dir.
            tensor_parallelism_size (int): tensor parallelism.
            pipeline_parallelism_size (int): pipeline parallelism.
            gpus_per_node (int): number of gpus per node.
            max_input_len (int): max input length.
            max_output_len (int): max output length.
            max_batch_size (int): max batch size.
            max_prompt_embedding_table_size (int): max prompt embedding size.
            use_parallel_embedding (bool): whether to use parallel embedding feature of TRT-LLM or not
            use_embedding_sharing (bool):
            paged_kv_cache (bool): if True, uses kv cache feature of the TensorRT-LLM.
            paged_context_fmha (bool): whether to use paged context fmha feature of TRT-LLM or not
            remove_input_padding (bool): enables removing input padding or not.
            dtype (Optional[str]): Floating point type for model weights (supports 'bfloat16', 'float16' or 'float32').
                If None, try to autodetect the type from model config.
            load_model (bool): load TensorRT-LLM model after the export.
            use_lora_plugin (str): use dynamic lora or not.
            lora_target_modules (List[str]): list of the target lora modules.
            max_lora_rank (int): maximum lora rank.
            max_num_tokens (int):
            opt_num_tokens (int):
            max_seq_len (int): the maximum sequence length of a single request.
            multiple_profiles: (bool): enables multiple profiles feature of TRT-LLM. Default = False
            gpt_attention_plugin (str): enable the gpt attention plugin. Default = "auto"
            gemm_plugin (str): enable the gpt plugin. Default = "auto"
            use_mcore_path (bool) : Use the more recent mcore path for export
            reduce_fusion (bool): enables fusing extra kernels after custom TRT-LLM allReduce
            fp8_quantized (Optional[bool]): enables exporting to FP8 TRT-LLM checkpoints. If not set, autodetects the type.
            fp8_kvcache (Optional[bool]): enables FP8 KV-cache quantization. If not set, autodetects the type.
            gather_context_logits (Optional[bool]): if True, enables gather_context_logits while building trtllm engine. Default: False
            gather_generation_logits (Optional[bool]): if True, enables gather_generation_logits while building trtllm engine. Default: False
            build_rank (Optional[int]): rank to export the model on. If None, builds on all ranks.
        """
        if not use_mcore_path:
            warnings.warn(
                "Exporting models using the local codebase with use_mcore_path=False is deprecated."
                " Please install megatron-core and set use_mcore_path to True.",
                stacklevel=2,
            )

        gpus_per_node = tensor_parallelism_size if gpus_per_node is None else gpus_per_node
        prepare_directory_for_export(
            self.model_dir, delete_existing_files=delete_existing_files, subdir=TRTLLM_ENGINE_DIR
        )

        if max_prompt_embedding_table_size is None:
            max_prompt_embedding_table_size = 0

        self.model = None

        if max_output_len is not None:
            warnings.warn(
                "Parameter max_output_len is deprecated and will be removed.", DeprecationWarning, stacklevel=2
            )
            max_output_len = max_output_len if max_output_len is not None else 256

            if max_seq_len is None:
                max_seq_len = max_input_len + max_output_len
            else:
                warnings.warn(
                    f"Parameter max_output_len will be overwritten by max_seq_len={max_seq_len}.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        max_seq_len = max_seq_len if max_seq_len is not None else 512

        if max_batch_size < 4:
            warnings.warn(
                "TensorRT LLM may hit a runtime issue with batch size is smaller than 4 on some models."
                " Force set to 4",
                stacklevel=2,
            )
            max_batch_size = 4

        is_export_rank = is_rank(build_rank)

        if is_export_rank:
            tmp_dir = tempfile.TemporaryDirectory()
            nemo_export_dir = Path(tmp_dir.name)

            if is_qnemo_checkpoint(nemo_checkpoint_path):
                if os.path.isdir(nemo_checkpoint_path):
                    nemo_export_dir = nemo_checkpoint_path
                else:
                    raise ValueError("Checkpoint path must be a directory")

                if os.path.exists(os.path.join(nemo_checkpoint_path, TOKENIZER_CONFIG_FILE)):
                    # Instantiate tokenizer for a legacy "Nemo 1" quantized checkpoint from a tokenizer config.
                    # Note that using the config is deprecated and it will be removed in future releases.
                    LOGGER.warning("Detected legacy tokenizer_config.yaml, using it to build tokenizer.")
                    self.tokenizer = get_nmt_tokenizer(nemo_checkpoint_path)
                else:
                    self.tokenizer = get_tokenizer(nemo_checkpoint_path)

                model_config = None

                qnemo_to_tensorrt_llm(
                    nemo_checkpoint_path=nemo_checkpoint_path,
                    engine_dir=self.engine_dir,
                    max_input_len=max_input_len,
                    max_seq_len=max_seq_len,
                    max_batch_size=max_batch_size,
                    max_prompt_embedding_table_size=max_prompt_embedding_table_size,
                    tensor_parallel_size=tensor_parallelism_size,
                    pipeline_parallel_size=pipeline_parallelism_size,
                    use_parallel_embedding=use_parallel_embedding,
                    paged_kv_cache=paged_kv_cache,
                    paged_context_fmha=paged_context_fmha,
                    remove_input_padding=remove_input_padding,
                    use_lora_plugin=use_lora_plugin,
                    lora_target_modules=lora_target_modules,
                    max_lora_rank=max_lora_rank,
                    max_num_tokens=max_num_tokens,
                    opt_num_tokens=opt_num_tokens,
                    multiple_profiles=multiple_profiles,
                    reduce_fusion=reduce_fusion,
                )
            else:
                if model_type is None:
                    # For NeMo 2.0 models we can get model_type from the model class name
                    model_type = get_model_type(nemo_checkpoint_path)

                if model_type is None:
                    raise ValueError(
                        "Parameter model_type needs to be provided and cannot be inferred from the checkpoint. "
                        "Please specify it explicitely."
                    )

                if model_type not in self.get_supported_models_list:
                    raise ValueError(
                        f"Model {model_type} is not currently a supported model type. "
                        f"Supported model types are: {self.get_supported_models_list}."
                    )

                if dtype is None:
                    dtype = get_weights_dtype(nemo_checkpoint_path)

                if dtype is None:
                    raise ValueError(
                        "Parameter dtype needs to be provided and cannot be inferred from the checkpoint. "
                        "Please specify it explicitely."
                    )

                model, model_config, self.tokenizer = load_nemo_model(
                    nemo_checkpoint_path, nemo_export_dir, use_mcore_path
                )
                if use_mcore_path:
                    from megatron.core.export.data_type import DataType
                    from megatron.core.export.export_config import ExportConfig
                    from megatron.core.export.model_type import ModelType
                    from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import (
                        DEFAULT_CONVERSION_DICT,
                    )
                    from megatron.core.export.trtllm.trtllm_helper import TRTLLMHelper
                    from tensorrt_llm.layers import MoeConfig

                    share_embeddings_and_output_weights = model_config.get(
                        "share_embeddings_and_output_weights", False
                    )
                    fp8_quantized, fp8_kvcache = determine_quantization_settings(
                        model_config, fp8_quantized, fp8_kvcache
                    )

                    # We build the transformer config using the nemo model config.
                    transformer_config = self.get_transformer_config(model_config)
                    input_model_type = getattr(ModelType, model_type)

                    # MCore export supports some default conversion dictionaries
                    mcore_model_conversion_dict = DEFAULT_CONVERSION_DICT

                    # All Mcore conversion dicts start with "decoder.layers.4.blah.blah" , while nemo models start with "model.decoder.layers.4.blahblah". so we append model. to the keys
                    nemo_model_conversion_dict = {
                        f'model.{key}': value for key, value in mcore_model_conversion_dict.items()
                    } | {  # Mapping for NeMo 2.0
                        f'module.{key}': value for key, value in mcore_model_conversion_dict.items()
                    }

                    # TODO: Workaround: Gemma uses gated activation, while mcore does not handle openai-gelu
                    # as a gated function. Remove once !11614 is merged.
                    activation = model_config.get('activation', "gelu")
                    if activation == "openai-gelu" and input_model_type.name == 'gemma':
                        activation = "geglu"

                    trtllm_helper = TRTLLMHelper(
                        transformer_config=transformer_config,
                        model_type=input_model_type,
                        trtllm_conversion_dict=nemo_model_conversion_dict,
                        position_embedding_type=model_config.get('position_embedding_type'),
                        max_position_embeddings=model_config.get('max_position_embeddings'),
                        rotary_percentage=model_config.get('rotary_percentage', 1.0),
                        rotary_base=model_config.get('rotary_base', 10000),
                        moe_tp_mode=model_config.get('moe_tp_mode', 2),
                        multi_query_mode=model_config.get("multi_query_mode", False),
                        activation=activation,
                        seq_len_interpolation_factor=model_config.get("seq_len_interpolation_factor"),
                        moe_renorm_mode=model_config.get(
                            'moe_renorm_mode', MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
                        ),
                        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                    )

                    input_dtype = getattr(DataType, dtype)
                    export_config = ExportConfig(
                        tensor_parallelism_size,
                        pipeline_parallelism_size,
                        use_parallel_embedding,
                        share_embeddings_and_output_weights,
                    )

                    trtllm_model_weights_list, trtllm_model_config_list = (
                        trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                            model_state_dict=model,
                            export_config=export_config,
                            dtype=input_dtype,
                            state_dict_split_by_layer_numbers=False,
                            fp8_quantized=fp8_quantized,
                            fp8_kvcache=fp8_kvcache,
                        )
                    )

                    for trtllm_model_weights, trtllm_model_config in zip(
                        trtllm_model_weights_list, trtllm_model_config_list
                    ):
                        trtllm_helper.build_and_save_engine(
                            max_input_len=max_input_len,
                            max_output_len=max_output_len,
                            max_batch_size=max_batch_size,
                            engine_dir=self.engine_dir,
                            trtllm_model_weights=trtllm_model_weights,
                            trtllm_model_config=trtllm_model_config,
                            lora_ckpt_list=self.lora_ckpt_list,
                            use_lora_plugin=use_lora_plugin,
                            max_lora_rank=max_lora_rank,
                            lora_target_modules=lora_target_modules,
                            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
                            paged_kv_cache=paged_kv_cache,
                            remove_input_padding=remove_input_padding,
                            paged_context_fmha=paged_context_fmha,
                            use_refit=False,
                            max_num_tokens=max_num_tokens,
                            max_seq_len=max_seq_len,
                            opt_num_tokens=opt_num_tokens,
                            max_beam_width=1,
                            tokens_per_block=128,
                            multiple_profiles=multiple_profiles,
                            gpt_attention_plugin=gpt_attention_plugin,
                            gemm_plugin=gemm_plugin,
                        )
                else:
                    if model_type == "gpt" or model_type == "starcoder":
                        model_type = "gptnext"

                    if model_type == "mixtral":
                        model_type = "llama"

                    trtllm_model_weights_list, trtllm_model_config_list = model_to_trtllm_ckpt(
                        model=model,
                        nemo_model_config=model_config,
                        nemo_export_dir=nemo_export_dir,
                        decoder_type=model_type,
                        dtype=dtype,
                        tensor_parallel_size=tensor_parallelism_size,
                        pipeline_parallel_size=pipeline_parallelism_size,
                        gpus_per_node=gpus_per_node,
                        use_parallel_embedding=use_parallel_embedding,
                        use_embedding_sharing=use_embedding_sharing,
                        fp8_quantized=fp8_quantized,
                        fp8_kvcache=fp8_kvcache,
                    )

                    for trtllm_model_weights, trtllm_model_config in zip(
                        trtllm_model_weights_list, trtllm_model_config_list
                    ):
                        build_and_save_engine(
                            max_input_len=max_input_len,
                            max_output_len=max_output_len,
                            max_batch_size=max_batch_size,
                            model_config=trtllm_model_config,
                            model_weights=trtllm_model_weights,
                            model_dir=self.engine_dir,
                            model_type=model_type,
                            lora_ckpt_list=self.lora_ckpt_list,
                            use_lora_plugin=use_lora_plugin,
                            max_lora_rank=max_lora_rank,
                            lora_target_modules=lora_target_modules,
                            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
                            paged_kv_cache=paged_kv_cache,
                            remove_input_padding=remove_input_padding,
                            paged_context_fmha=paged_context_fmha,
                            max_num_tokens=max_num_tokens,
                            opt_num_tokens=opt_num_tokens,
                            max_seq_len=max_seq_len,
                            multiple_profiles=multiple_profiles,
                            gpt_attention_plugin=gpt_attention_plugin,
                            gemm_plugin=gemm_plugin,
                            gather_context_logits=gather_context_logits,
                            gather_generation_logits=gather_generation_logits,
                        )

            tokenizer_path = os.path.join(nemo_export_dir, "tokenizer.model")
            tokenizer_path_nemo2 = os.path.join(nemo_export_dir, "nemo_context")
            vocab_path = os.path.join(nemo_export_dir, "vocab.json")
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                self.tokenizer.save_pretrained(self.model_dir)
            elif os.path.exists(tokenizer_path):
                shutil.copy(tokenizer_path, self.model_dir)
            elif os.path.exists(tokenizer_path_nemo2):
                # Copy HF tokenizer files to root model directory
                for path in glob(os.path.join(tokenizer_path_nemo2, "nemo_tokenizer", "*.json")):
                    shutil.copy(path, self.model_dir)
                # Copy SentencePiece tokenizer.model
                for path in glob(os.path.join(tokenizer_path_nemo2, "*.model")):
                    shutil.copy(path, os.path.join(self.model_dir, "tokenizer.model"))
            elif os.path.exists(vocab_path):
                shutil.copy(vocab_path, os.path.join(self.model_dir, "vocab.json"))

            nemo_model_config = os.path.join(nemo_export_dir, "model_config.yaml")
            if os.path.exists(nemo_model_config):
                shutil.copy(nemo_model_config, self.model_dir)

            tmp_dir.cleanup()

        if is_export_rank and model_config is not None:
            self._export_to_nim_format(model_config, model_type)

        if tensorrt_llm.mpi_world_size() > 1:
            tensorrt_llm.mpi_barrier()

        if is_export_rank and load_model:
            self._load()

    def export_hf_model(
        self,
        hf_model_path: str,
        max_batch_size: int = 8,
        tensor_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: int = 256,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        dtype: Optional[str] = None,
        max_seq_len: Optional[int] = 512,
        gemm_plugin: str = "auto",
        remove_input_padding: bool = True,
        paged_context_fmha: bool = False,
        paged_kv_cache: bool = True,
        tokens_per_block: int = 128,
        multiple_profiles: bool = False,
        reduce_fusion: bool = False,
        max_beam_width: int = 1,
        use_refit: bool = False,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
    ):
        """
        Export a Hugging Face model checkpoint to TensorRT-LLM format.

        Args:
            hf_model_path (str): Path to the Hugging Face model directory
            max_batch_size (int, optional): Maximum batch size for inference. Defaults to 8.
            tensor_parallelism_size (int, optional): Size of tensor parallelism. Defaults to 1.
            max_input_len (int, optional): Maximum input sequence length. Defaults to 256.
            max_output_len (int, optional): Maximum output sequence length. Defaults to 256.
            max_num_tokens (int, optional): Maximum number of tokens. Defaults to None.
            opt_num_tokens (int, optional): Optimal number of tokens. Defaults to None.
            dtype (str, optional): Data type for model weights. If None, inferred from model config.
            max_seq_len (int, optional): Maximum total sequence length. Defaults to 512.
            gemm_plugin (str, optional): GEMM plugin type. Defaults to "auto".
            remove_input_padding (bool, optional): Whether to remove input padding. Defaults to True.
            paged_context_fmha (bool, optional): Whether to use paged context FMHA. Defaults to False.
            paged_kv_cache (bool, optional): Whether to use paged KV cache. Defaults to True.
            tokens_per_block (int, optional): Number of tokens per block for paged KV cache. Defaults to 128.
            multiple_profiles (bool, optional): Whether to use multiple TensorRT profiles. Defaults to False.
            reduce_fusion (bool, optional): Whether to reduce operator fusion. Defaults to False.
            max_beam_width (int, optional): Maximum beam width for beam search. Defaults to 1.
            use_refit (bool, optional): Whether to use TensorRT refitting. Defaults to False.
            model_type (str, optional): Type of the model architecture. Defaults to None.
            delete_existing_files (bool, optional): Whether to delete existing files in export dir. Defaults to True.

        Raises:
            ValueError: If model_type is not supported or dtype cannot be determined
        """
        LOGGER.info("Starting HF export to TRT-LLM")
        if model_type not in self.get_supported_hf_model_mapping:
            raise ValueError(
                f"Model {model_type} is not currently a supported model type. "
                f"Supported model types are: {self.get_supported_hf_model_mapping.keys()}."
            )

        if dtype is None:
            dtype = self.get_hf_model_dtype(hf_model_path)
            if dtype is None:
                raise ValueError("No dtype found in hf model config. Please specify a dtype.")

        prepare_directory_for_export(
            self.model_dir, delete_existing_files=delete_existing_files, subdir=TRTLLM_ENGINE_DIR
        )

        if max_batch_size < 4:
            print("TensorRT-LLM may hit runtime issue with batch size is smaller than 4. Force set to 4")
            max_batch_size = 4

        plugin_config = PluginConfig()
        plugin_config.gemm_plugin = gemm_plugin
        if paged_kv_cache:
            plugin_config.enable_paged_kv_cache(tokens_per_block=tokens_per_block)
        else:
            plugin_config.paged_kv_cache = False
        plugin_config.remove_input_padding = remove_input_padding
        plugin_config.use_paged_context_fmha = paged_context_fmha
        plugin_config.multiple_profiles = multiple_profiles
        plugin_config.reduce_fusion = reduce_fusion
        max_seq_len = max_input_len + max_output_len
        max_num_tokens, opt_num_tokens = check_max_num_tokens(
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_beam_width=max_beam_width,
            remove_input_padding=remove_input_padding,
            enable_context_fmha=plugin_config.context_fmha,
            tokens_per_block=tokens_per_block,
            multiple_profiles=multiple_profiles,
        )
        build_dict = {
            'max_input_len': max_input_len,
            'max_output_len': max_output_len,
            'max_batch_size': max_batch_size,
            'max_beam_width': max_beam_width,
            'max_seq_len': max_seq_len,
            'max_num_tokens': max_num_tokens,
            'opt_num_tokens': opt_num_tokens,
            'strongly_typed': False,
            'builder_opt': None,
            'multiple_profiles': multiple_profiles,
            'use_refit': use_refit,
        }
        build_config = BuildConfig.from_dict(build_dict, plugin_config=plugin_config)
        for rank in range(tensor_parallelism_size):
            LOGGER.info(f"Iterating over rank:{rank}")
            mapping = Mapping(world_size=tensor_parallelism_size, rank=rank, tp_size=tensor_parallelism_size)
            trtllm_model_class = self.get_supported_hf_model_mapping[model_type]
            model = trtllm_model_class.from_hugging_face(
                hf_model_path,
                dtype,
                mapping=mapping,
            )
            engine = build_trtllm(model, build_config)
            engine.save(self.engine_dir)
        # Copy HF tokenizer files to root model directory
        for path in glob(os.path.join(hf_model_path, "*.json")):
            shutil.copy(path, self.model_dir)
        # Copy sentencepiece model to model directory
        for path in glob(os.path.join(hf_model_path, "*.model")):
            shutil.copy(path, self.model_dir)
        LOGGER.info(f"Generarated TRT-LLM checkpoint at dir:{self.model_dir}")
        LOGGER.info(f"Loading the TRT-LLM checkpoint:{self.model_dir}")
        self._load()

    def get_hf_model_dtype(self, model_dir: str) -> Optional[str]:
        """
        Read the config file from a Hugging Face model directory and identify the model's data type.

        Args:
            model_dir (str): Path to the Hugging Face model directory

        Returns:
            Optional[str]: The model's data type if found in config, None otherwise
        """
        config_path = Path(model_dir) / 'config.json'

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Check for dtype in different possible locations in the config
                if 'torch_dtype' in config:
                    return config['torch_dtype']
                elif 'dtype' in config:
                    return config['dtype']
                elif 'pretrained_config' in config and 'dtype' in config['pretrained_config']:
                    return config['pretrained_config']['dtype']

                # If no explicit dtype found, check for other indicators
                if 'fp16' in config and config['fp16']:
                    return 'float16'
                elif 'bf16' in config and config['bf16']:
                    return 'bfloat16'

            return None
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file at {config_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading config file: {str(e)}")

    def _export_to_nim_format(self, model_config: Dict[str, Any], model_type: str):
        """
        Exports the model configuration to a specific format required by NIM.
        This method performs the following steps:

        1. Copies the generation_config.json (if present) from the nemo_context directory to the root model directory.
        2. Creates a dummy Hugging Face configuration file based on the provided model configuration and type.

        Args:
            model_config (dict): A dictionary containing the model configuration parameters.
            model_type (str): The type of the model (e.g., "llama").
        """

        generation_config_path = os.path.join(self.model_dir, "nemo_context", "artifacts", "generation_config.json")
        if os.path.isfile(generation_config_path):
            shutil.copy(generation_config_path, self.model_dir)

        # Fields "architectures" and "model_type" are required by HF but not relevant for NIM
        seq_len_interpolation_factor = model_config.get("seq_len_interpolation_factor")
        hf_config = {
            "max_position_embeddings": model_config.get("encoder_seq_length"),
            "architectures": ["LLaMAForCausalLM"],
            "rope_scaling": (
                None
                if seq_len_interpolation_factor is None
                else {
                    "factor": seq_len_interpolation_factor,
                    "rope_type": "default",
                }
            ),
            "model_type": model_type,
        }
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)
            f.write("\n")

    def get_transformer_config(self, nemo_model_config):
        """Given nemo model config get transformer config"""
        from megatron.core.transformer.transformer_config import TransformerConfig

        normalization = nemo_model_config.get('normalization', 'layernorm')
        transformer_config_normalization = 'LayerNorm'
        layernorm_zero_centered_gamma = nemo_model_config.get('layernorm_zero_centered_gamma', False)
        if normalization == 'layernorm1p':
            layernorm_zero_centered_gamma = True
        elif normalization == 'rmsnorm':
            transformer_config_normalization = 'RMSNorm'

        num_moe_experts = nemo_model_config.get('num_moe_experts', 0)
        conf = TransformerConfig(
            num_layers=nemo_model_config.get('num_layers'),
            moe_router_topk=nemo_model_config.get('moe_router_topk', 0),
            num_attention_heads=nemo_model_config.get('num_attention_heads'),
            num_query_groups=nemo_model_config.get('num_query_groups', nemo_model_config['num_attention_heads']),
            kv_channels=nemo_model_config.get("kv_channels", None),
            hidden_size=nemo_model_config.get('hidden_size'),
            ffn_hidden_size=nemo_model_config.get('ffn_hidden_size'),
            layernorm_epsilon=nemo_model_config.get('layernorm_epsilon'),
            add_bias_linear=nemo_model_config.get('bias'),
            num_moe_experts=num_moe_experts if num_moe_experts > 0 else None,
            normalization=transformer_config_normalization,
            layernorm_zero_centered_gamma=layernorm_zero_centered_gamma,
            gated_linear_unit=nemo_model_config.get('gated_linear_unit', False),
        )
        return conf

    def convert_to_safe_tensors(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        pipeline_parallelism_size: int = 1,
        gpus_per_node: int = None,
        use_parallel_embedding: bool = False,
        use_embedding_sharing: bool = False,
        dtype: str = "bfloat16",
    ):
        """Convert to safe tensor"""
        gpus_per_node = tensor_parallelism_size if gpus_per_node is None else gpus_per_node

        if Path(self.model_dir).exists():
            if delete_existing_files and len(os.listdir(self.model_dir)) > 0:
                for files in os.listdir(self.model_dir):
                    path = os.path.join(self.model_dir, files)
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        os.remove(path)

                if len(os.listdir(self.model_dir)) > 0:
                    raise Exception("Couldn't delete all files.")
            elif len(os.listdir(self.model_dir)) > 0:
                raise Exception("There are files in this folder. Try setting delete_existing_files=True.")
        else:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        if model_type == "gpt" or model_type == "starcoder":
            model_type = "gptnext"

        if model_type == "mixtral":
            model_type = "llama"

        if tensorrt_llm.mpi_rank() == 0:
            tmp_dir = tempfile.TemporaryDirectory()
            nemo_export_dir = Path(tmp_dir.name)

            model, model_config, self.tokenizer = load_nemo_model(nemo_checkpoint_path, nemo_export_dir)
            weights_dicts, model_configs = model_to_trtllm_ckpt(
                model=model,
                nemo_model_config=model_config,
                nemo_export_dir=nemo_export_dir,
                decoder_type=model_type,
                dtype=dtype,
                tensor_parallel_size=tensor_parallelism_size,
                pipeline_parallel_size=pipeline_parallelism_size,
                gpus_per_node=gpus_per_node,
                use_parallel_embedding=use_parallel_embedding,
                use_embedding_sharing=use_embedding_sharing,
            )

            for weight_dict, model_config in zip(weights_dicts, model_configs):
                rank = model_config.mapping.tp_rank
                for k, v in weight_dict.items():
                    if isinstance(v, np.ndarray):
                        weight_dict[k] = numpy_to_torch(v)
                    else:
                        weight_dict[k] = v

                safetensors.torch.save_file(weight_dict, os.path.join(self.model_dir, f'rank{rank}.safetensors'))
            model_configs[0].to_json_file(os.path.join(self.model_dir, 'config.json'))

            tokenizer_path = os.path.join(nemo_export_dir, "tokenizer.model")
            if os.path.exists(tokenizer_path):
                shutil.copy(tokenizer_path, self.model_dir)
            else:
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(self.model_dir)

            nemo_model_config = os.path.join(nemo_export_dir, "model_config.yaml")
            if os.path.exists(nemo_model_config):
                shutil.copy(nemo_model_config, self.model_dir)

            tmp_dir.cleanup()

        if tensorrt_llm.mpi_world_size() > 1:
            tensorrt_llm.mpi_barrier()

    def gather_and_reshard_model(self, model_config, model, storage_dtype):
        """
        Accumulate all vp model chunks together, and reshard model (i.e) gather all pp ranks
        if required and return the final model state dict
        """

        def _get_layer_index(split_key):
            for index, key in enumerate(split_key):
                if key == "layers":
                    return index + 1
            raise ValueError(f"Unknown layer name format: {split_key}")

        def rename_layer_num(param_name, layer_num):
            split_key = param_name.split(".")
            layer_index = int(_get_layer_index(split_key))
            split_key[layer_index] = str(layer_num)
            return ".".join(split_key)

        def get_layer_num(param_name):
            split_key = param_name.split(".")
            layer_index = int(_get_layer_index(split_key))
            return int(split_key[layer_index])

        from megatron.core import parallel_state

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
        pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        if not vp_size:
            vp_size = 1

        inference_tp_size = self.tp_size
        inference_pp_size = self.pp_size
        reshard_model = False
        if inference_tp_size != tp_size or inference_pp_size != pp_size:
            LOGGER.info("Training/Generation model parallelism resharding enabled")
            if inference_pp_size == 1 and pp_size > 1 and inference_tp_size == tp_size:
                reshard_model = True
            else:
                raise NotImplementedError(
                    "NeMo currently only supports PP>1 -> PP=1 resharding, other types of resharding will come in future releases."
                )

        num_layers = model_config["num_layers"]
        layers_per_pp = num_layers // pp_size
        layers_per_chunk = layers_per_pp // vp_size

        tl_params = {}
        model_level_params = {}
        if vp_size > 1:  # consolidate params across model chunks
            for idx, model_chunk in enumerate(model):
                for key, val in model_chunk.state_dict().items():
                    # TODO: currently fp8 is not supported
                    if torch.is_tensor(val) and '_extra_state' not in key:
                        if 'layers' in key:
                            key2 = rename_layer_num(key, get_layer_num(key) + idx * pp_size * layers_per_chunk)
                            tl_params[key2] = val
                        else:
                            model_level_params[key] = val
        else:
            for key, val in model.state_dict().items():
                # TODO: currently fp8 is not supported
                if torch.is_tensor(val) and '_extra_state' not in key:
                    if 'decoder.layers' in key:
                        tl_params[key] = val
                    else:
                        model_level_params[key] = val

        if vp_size > 1 or reshard_model:
            # gather layers across pp ranks
            gathered_params = {}
            for key, val in tl_params.items():
                weight_list = [torch.zeros_like(val) for _ in range(pp_size)]
                torch.distributed.all_gather(weight_list, val, group=pp_group)
                for idx in range(pp_size):
                    layer_num = get_layer_num(key) + idx * layers_per_chunk
                    key2 = rename_layer_num(key, layer_num)
                    if not reshard_model:  # Save only layers of 1 single PP stage
                        layers_start = layers_per_pp * pp_rank
                        layers_end = layers_per_pp * (pp_rank + 1) - 1
                        if layer_num >= layers_start and layer_num <= layers_end:
                            key2 = rename_layer_num(key, layer_num % layers_per_pp)
                            gathered_params[key2] = weight_list[idx]
                    else:
                        gathered_params[key2] = weight_list[idx]
            tl_params = gathered_params

        model_state_dict = model_level_params
        model_state_dict.update(tl_params)

        def get_tensor_if_available(key, pp_src_idx, group):
            tensor = model_state_dict.get(key)
            if tensor is not None:
                tensor_shape = [tensor.shape]
            else:
                tensor_shape = [None]

            torch.distributed.broadcast_object_list(tensor_shape, pp_src_idx, group=group)

            if tensor_shape[0] is None:
                return None
            if torch.distributed.get_rank() != pp_src_idx:
                tensor = torch.empty(tensor_shape[0], dtype=storage_dtype).cuda()

            torch.distributed.broadcast(tensor.contiguous(), pp_src_idx, group=pp_group)
            return tensor

        if reshard_model:
            key = 'decoder.final_layernorm.weight'
            tensor = get_tensor_if_available(key, pp_last_rank, pp_group)
            if tensor is not None:
                model_state_dict[key] = tensor

            key = 'decoder.final_layernorm.bias'
            tensor = get_tensor_if_available(key, pp_last_rank, pp_group)
            if tensor is not None:
                model_state_dict[key] = tensor

            key = 'embedding.word_embeddings.weight'
            tensor = get_tensor_if_available(key, pp_first_rank, pp_group)
            if tensor is not None:
                model_state_dict[key] = tensor

            key = 'output_layer.weight'
            tensor = get_tensor_if_available(key, pp_last_rank, pp_group)
            if tensor is not None:
                model_state_dict[key] = tensor

        return model_state_dict

    def get_input_dtype(self, storage_dtype):
        """
        Return mcore export dtype given torch dtype
        """
        from megatron.core.export.data_type import DataType

        if storage_dtype == torch.bfloat16:
            return DataType.bfloat16
        elif storage_dtype == torch.float32:
            return DataType.float32
        elif storage_dtype == torch.float16:
            return DataType.float16

    @staticmethod
    def get_nemo_to_trtllm_conversion_dict(model_state_dict):
        """MCore export supports some default conversion dictionaries
        All Mcore conversion dicts start with "decoder.layers.4.blah.blah" , while nemo models sometimes start with "model.decoder.layers.4.blahblah". so we append model prefix. to the keys
        """
        from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import DEFAULT_CONVERSION_DICT

        model_prefix, _ = get_layer_prefix(layer_names=model_state_dict.keys(), is_mcore=True)

        nemo_model_conversion_dict = {}
        for key, value in DEFAULT_CONVERSION_DICT.items():
            if model_prefix:
                nemo_model_conversion_dict[f'{model_prefix}{key}'] = value
            else:
                nemo_model_conversion_dict[key] = value
        return nemo_model_conversion_dict

    def build(
        self,
        model,
        model_config,
        model_type,
        gpus_per_node,
        tokenizer,
        max_input_len: int = 1024,
        max_output_len: int = 1024,
        max_batch_size: int = 4,
        use_refit: bool = True,
        reshard_model: bool = False,
        use_mcore_path: bool = True,
    ):
        """
        Convert a model parallel nemo model to TensorRT-LLM.
        """
        assert tensorrt_llm.mpi_rank() == torch.distributed.get_rank()
        self.use_refit, self.model_type, self.gpus_per_node = use_refit, model_type, gpus_per_node
        self.mp_rank, self.dp_rank, self.tp_size, self.pp_size, self.dp_size = init_model_parallel_from_nemo(
            reshard_model
        )
        self.tokenizer = build_tokenizer(tokenizer)

        if self.dp_size > 1:
            self.model_dir = os.path.join(self.model_dir, f"dp_rank{self.dp_rank}")

        if use_mcore_path:
            from megatron.core.export.model_type import ModelType
            from megatron.core.export.trtllm.trtllm_helper import TRTLLMHelper
            from tensorrt_llm.layers import MoeConfig

            storage_dtype = torch_dtype_from_precision(model_config.precision)
            model_state_dict = self.gather_and_reshard_model(model_config, model, storage_dtype)
            # We build the transformer config using the nemo model config.
            transformer_config = self.get_transformer_config(model_config)
            input_model_type = getattr(ModelType, model_type)

            nemo_model_conversion_dict = self.get_nemo_to_trtllm_conversion_dict(model_state_dict)
            self.trtllm_helper = TRTLLMHelper(
                transformer_config=transformer_config,
                model_type=input_model_type,
                trtllm_conversion_dict=nemo_model_conversion_dict,
                position_embedding_type=model_config.get('position_embedding_type'),
                max_position_embeddings=model_config.get('max_position_embeddings'),
                rotary_percentage=model_config.get('rotary_percentage', 1.0),
                rotary_base=model_config.get('rotary_base', 10000),
                moe_tp_mode=model_config.get('moe_tp_mode', 2),
                multi_query_mode=model_config.get("multi_query_mode", False),
                activation=model_config.get('activation', "gelu"),
                seq_len_interpolation_factor=model_config.get("seq_len_interpolation_factor"),
                moe_renorm_mode=model_config.get(
                    'moe_renorm_mode', MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
                ),
                share_embeddings_and_output_weights=model_config.get("share_embeddings_and_output_weights", False),
            )

            input_dtype = self.get_input_dtype(storage_dtype)

            trtllm_model_weights_list, trtllm_model_config_list = (
                self.trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                    model_state_dict=model_state_dict,
                    dtype=input_dtype,
                    state_dict_split_by_layer_numbers=True,
                    on_device_distributed_conversion=True,
                    vocab_size=self.tokenizer.vocab_size,
                    gpus_per_node=gpus_per_node,
                )
            )
            trtllm_model_config = trtllm_model_config_list[0]
            trtllm_model_weights = trtllm_model_weights_list[0]

            if reshard_model:
                assert self.pp_size == 1, 'Reshard is true, but pp size is not one'
                # MCORE Export will use parallel_state to determine pp .
                # Since we reshard to pp = 1, we need to modify the config and mapping
                world_size = self.tp_size * self.pp_size
                trtllm_model_config.pp_size = self.pp_size
                trtllm_model_config.world_size = world_size
                trtllm_model_config.mapping = tensorrt_llm.Mapping(
                    world_size=world_size,
                    rank=self.mp_rank,
                    tp_size=self.tp_size,
                    pp_size=self.pp_size,
                )

            engine = self.trtllm_helper.build_and_save_engine(
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_seq_len=max_input_len + max_output_len,
                max_batch_size=max_batch_size,
                trtllm_model_config=trtllm_model_config,
                trtllm_model_weights=trtllm_model_weights,
                engine_dir=self.model_dir,
                use_refit=use_refit,
            )
        else:
            weights, model_config = model_to_trtllm_ckpt(
                model=model,
                nemo_model_config=model_config,
                nemo_export_dir=self.model_dir,
                decoder_type=model_type,
                tensor_parallel_size=self.tp_size,
                pipeline_parallel_size=self.pp_size,
                gpus_per_node=gpus_per_node,
                use_parallel_embedding=True,
                use_distributed_convert=True,
                model_parallel_rank=self.mp_rank,
                vocab_size=self.tokenizer.vocab_size,
            )

            engine = build_and_save_engine(
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_seq_len=max_input_len + max_output_len,
                max_batch_size=max_batch_size,
                model_config=model_config[0],
                model_weights=weights[0],
                model_dir=self.model_dir,
                model_type=model_type,
                use_refit=use_refit,
            )

        torch.distributed.barrier()

        cfg_path = Path(os.path.join(self.model_dir, f'config_{torch.distributed.get_rank()}.json'))
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(engine.config.to_dict(), f, indent=4)

        load_distributed(self.model_dir, self.mp_rank, gpus_per_node)

    def refit(self, model, model_config, use_mcore_path=True):
        """
        Refits an TensorRT engine using an instantiated nemo model.
        This function should only be used after calling build()
        """
        weights_dict = None
        if use_mcore_path:
            storage_dtype = torch_dtype_from_precision(model_config.precision)

            model_state_dict = self.gather_and_reshard_model(model_config, model, storage_dtype)

            nemo_model_conversion_dict = self.get_nemo_to_trtllm_conversion_dict(model_state_dict)
            self.trtllm_helper.weights_converter.convert(
                model_state_dict=model_state_dict,
                tokenizer_vocab_size=self.tokenizer.vocab_size,
                trtllm_conversion_dict=nemo_model_conversion_dict,
            )
            weights_dict = self.trtllm_helper.weights_converter.trtllm_model_weights

        else:
            weights_dict = dist_model_to_trt_llm_ckpt(
                model=model,
                nemo_model_config=model_config,
                inference_tp_size=self.tp_size,
                inference_pp_size=self.pp_size,
                tokenizer_vocab_size=self.tokenizer.vocab_size,
            )
        load_distributed(self.model_dir, self.mp_rank, self.gpus_per_node)
        gc.collect()
        torch.cuda.empty_cache()
        refit(weights_dict)

    def forward(
        self,
        input_texts: List[str],
        max_output_len: int = 64,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_words_list: List[str] = None,
        bad_words_list: List[str] = None,
        no_repeat_ngram_size: int = None,
        task_ids: List[str] = None,
        lora_uids: List[str] = None,
        prompt_embeddings_table=None,
        prompt_embeddings_checkpoint_path: str = None,
        streaming: bool = False,
        output_log_probs: bool = False,
        output_context_logits: bool = False,
        output_generation_logits: bool = False,
        **sampling_kwargs,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            input_texts (List(str)): list of sentences.
            max_output_len (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            task_ids (List(str)): list of the task ids for the prompt tables.
            prompt_embeddings_table (List(float)): prompt embeddings table.
            prompt_embeddings_checkpoint_path (str): path for the nemo checkpoint for the prompt embedding table.
            output_generation_logits (bool): if True returns generation_logits in the outout of generate method.
            sampling_kwargs: Additional kwargs to set in the SamplingConfig.
        """

        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported to TensorRT-LLM and "
                "then it should be loaded first to run inference."
            )
        else:
            if prompt_embeddings_table is not None or prompt_embeddings_checkpoint_path is not None:
                prompt_table = self._get_prompt_embedding_table(
                    prompt_embeddings_table, prompt_embeddings_checkpoint_path
                )
                tv_size = prompt_table.size(dim=0)
                task_vtoken_counts = [tv_size]
            elif len(self.ptuning_tables) > 0:
                prompt_table = self.p_table
                tv_size = self.task_vocab_size
                task_vtoken_counts = self.task_vtoken_counts
            else:
                prompt_table = None
                tv_size = None
                task_vtoken_counts = None

            if task_ids is None:
                assert prompt_table is None, "There is a prompt embedding table and task_ids cannot be None"
                input_task_ids = None
            else:
                if prompt_table is None:
                    input_task_ids = None
                else:
                    if len(task_ids) > 1:
                        assert len(task_ids) == len(input_texts), (
                            "Either len of the task_ids has to be 1 or" "it needs to match with len of input_texts."
                        )

                    if len(task_ids) == 1:
                        assert task_ids[0] in self.task_ids.keys(), "Task: {0} doesn't exist in the task list.".format(
                            task_ids[0]
                        )
                        input_task_ids = [self.task_ids[task_ids[0]] for i in range(len(input_texts))]
                    else:
                        input_task_ids = []
                        for i in range(len(input_texts)):
                            assert (
                                task_ids[i] in self.task_ids.keys()
                            ), "Task: {0} doesn't exist in the task list.".format(task_ids[i])
                            input_task_ids.append(self.task_ids[task_ids[i]])
            if not streaming:
                if torch.distributed.is_initialized() or tensorrt_llm.mpi_world_size() > 1:
                    multiprocessed_env = True
                else:
                    multiprocessed_env = False

                return generate(
                    input_texts=input_texts,
                    max_output_len=max_output_len,
                    host_context=self.model,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    prompt_table=prompt_table,
                    task_vocab_size=tv_size,
                    task_vtoken_counts=task_vtoken_counts,
                    task_ids=input_task_ids,
                    lora_uids=lora_uids,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    output_log_probs=output_log_probs,
                    multiprocessed_env=multiprocessed_env,
                    output_context_logits=output_context_logits,
                    output_generation_logits=output_generation_logits,
                    **sampling_kwargs,
                )
            else:
                return generate_streaming(
                    input_texts=input_texts,
                    max_output_len=max_output_len,
                    host_context=self.model,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    prompt_table=prompt_table,
                    task_vocab_size=tv_size,
                    task_vtoken_counts=task_vtoken_counts,
                    task_ids=input_task_ids,
                    lora_uids=lora_uids,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    **sampling_kwargs,
                )

    def add_prompt_table(self, task_name: str, prompt_embeddings_checkpoint_path: str):
        """Add prompt table"""
        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported to TensorRT-LLM and "
                "then it should be loaded first to run inference."
            )

        for pt in self.ptuning_tables:
            if pt["task_name"] == task_name:
                raise Exception("Task name: {0} has already added. Please pass a unique task name.".format(task_name))

        prompt_table = self._get_prompt_embedding_table(
            prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path
        )

        self.ptuning_tables.append({"table": prompt_table, "task_name": task_name})
        with open(os.path.join(self.model_dir, 'prompt_tables.pkl'), 'wb') as f:
            pickle.dump(self.ptuning_tables, f)

        self._prep_ptuning_table()

    def remove_prompt_table(self, task_name: str):
        """Remove prompt table"""
        if self.ptuning_tables is not None:
            for i in range(len(self.ptuning_tables)):
                if self.ptuning_tables[i]["task_name"] == task_name:
                    self.ptuning_tables.pop(i)
                    with open(os.path.join(self.model_dir, 'prompt_tables.pkl'), 'wb') as f:
                        pickle.dump(self.ptuning_tables, f)
                    return
            self._prep_ptuning_table()

    def _pad_logits(self, logits_tensor):
        """
        Pads the logits tensor with 0's on the right
        """
        padding_len = max([logit_tensor.shape[0] for logit_tensor in logits_tensor])
        for i, tensor in enumerate(logits_tensor):
            tensor_len = tensor.shape[0]
            if tensor_len < padding_len:
                padding_diff = padding_len - tensor_len
                # padding_diff num of rows of zeros are added at the bottom
                logits_tensor[i] = F.pad(tensor, (0, 0, 0, padding_diff), mode='constant', value=0)
        return logits_tensor

    @property
    def get_supported_models_list(self):
        """Supported model list"""
        # gpt and gptnext are the same. Keeping the gptnext due to backward compatibility.
        return ["gpt", "gptnext", "llama", "falcon", "starcoder", "mixtral", "gemma"]

    @property
    def get_supported_hf_model_mapping(self):
        """Supported HF Model Mapping"""
        HF_MODEL_CLASS_MAP = {
            'GPT2LMHeadModel': GPTForCausalLM,
            'GPT2LMHeadCustomModel': GPTForCausalLM,
            'GPTBigCodeForCausalLM': GPTForCausalLM,
            'Starcoder2ForCausalLM': GPTForCausalLM,
            'JAISLMHeadModel': GPTForCausalLM,
            'GPTForCausalLM': GPTForCausalLM,
            'NemotronForCausalLM': GPTForCausalLM,
            'OPTForCausalLM': OPTForCausalLM,
            'BloomForCausalLM': BloomForCausalLM,
            'RWForCausalLM': FalconForCausalLM,
            'FalconForCausalLM': FalconForCausalLM,
            'PhiForCausalLM': PhiForCausalLM,
            'Phi3ForCausalLM': Phi3ForCausalLM,
            'Phi3VForCausalLM': Phi3ForCausalLM,
            'Phi3SmallForCausalLM': Phi3ForCausalLM,
            'PhiMoEForCausalLM': Phi3ForCausalLM,
            'MambaForCausalLM': MambaForCausalLM,
            'GPTNeoXForCausalLM': GPTNeoXForCausalLM,
            'GPTJForCausalLM': GPTJForCausalLM,
            'MptForCausalLM': MPTForCausalLM,
            'MPTForCausalLM': MPTForCausalLM,
            'GLMModel': ChatGLMForCausalLM,
            'ChatGLMModel': ChatGLMForCausalLM,
            'ChatGLMForCausalLM': ChatGLMForCausalLM,
            'ChatGLMForConditionalGeneration': ChatGLMForCausalLM,
            'LlamaForCausalLM': LLaMAForCausalLM,
            'LlavaLlamaModel': LLaMAForCausalLM,
            'ExaoneForCausalLM': LLaMAForCausalLM,
            'MistralForCausalLM': LLaMAForCausalLM,
            'MixtralForCausalLM': LLaMAForCausalLM,
            'ArcticForCausalLM': LLaMAForCausalLM,
            'Grok1ModelForCausalLM': GrokForCausalLM,
            'InternLMForCausalLM': LLaMAForCausalLM,
            'InternLM2ForCausalLM': LLaMAForCausalLM,
            'InternLMXComposer2ForCausalLM': LLaMAForCausalLM,
            'GraniteForCausalLM': LLaMAForCausalLM,
            'GraniteMoeForCausalLM': LLaMAForCausalLM,
            'MedusaForCausalLM': MedusaForCausalLm,
            'MedusaLlamaForCausalLM': MedusaForCausalLm,
            'ReDrafterForCausalLM': ReDrafterForCausalLM,
            'BaichuanForCausalLM': BaichuanForCausalLM,
            'BaiChuanForCausalLM': BaichuanForCausalLM,
            'SkyworkForCausalLM': LLaMAForCausalLM,
            'GEMMA': GemmaForCausalLM,
            'GEMMA2': GemmaForCausalLM,
            'QWenLMHeadModel': QWenForCausalLM,
            'QWenForCausalLM': QWenForCausalLM,
            'Qwen2ForCausalLM': QWenForCausalLM,
            'Qwen2MoeForCausalLM': QWenForCausalLM,
            'Qwen2ForSequenceClassification': QWenForCausalLM,
            'Qwen2VLForConditionalGeneration': QWenForCausalLM,
            'Qwen2VLModel': QWenForCausalLM,
            'WhisperEncoder': WhisperEncoder,
            'EncoderModel': EncoderModel,
            'DecoderModel': DecoderModel,
            'DbrxForCausalLM': DbrxForCausalLM,
            'RecurrentGemmaForCausalLM': RecurrentGemmaForCausalLM,
            'CogVLMForCausalLM': CogVLMForCausalLM,
            'DiT': DiT,
            'DeepseekForCausalLM': DeepseekForCausalLM,
            'DeciLMForCausalLM': DeciLMForCausalLM,
            'DeepseekV2ForCausalLM': DeepseekV2ForCausalLM,
            'EagleForCausalLM': EagleForCausalLM,
            'CohereForCausalLM': CohereForCausalLM,
            'MLLaMAModel': MLLaMAForCausalLM,
            'MllamaForConditionalGeneration': MLLaMAForCausalLM,
            'BertForQuestionAnswering': BertForQuestionAnswering,
            'BertForSequenceClassification': BertForSequenceClassification,
            'BertModel': BertModel,
            'RobertaModel': RobertaModel,
            'RobertaForQuestionAnswering': RobertaForQuestionAnswering,
            'RobertaForSequenceClassification': RobertaForSequenceClassification,
        }
        return HF_MODEL_CLASS_MAP

    @property
    def get_hidden_size(self):
        """Get hidden size"""
        if self.config is None:
            return None
        else:
            return self.config["pretrained_config"]["hidden_size"]

    @property
    def get_triton_input(self):
        """Get triton input"""
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_len", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="stop_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="bad_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="no_repeat_ngram_size", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="task_id", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="lora_uids", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="output_context_logits", shape=(-1,), dtype=np.bool_, optional=False),
            Tensor(name="output_generation_logits", shape=(-1,), dtype=np.bool_, optional=False),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (
            Tensor(name="outputs", shape=(-1,), dtype=bytes),
            Tensor(name="generation_logits", shape=(-1,), dtype=np.single),
            Tensor(name="context_logits", shape=(-1,), dtype=np.single),
        )
        return outputs

    @batch
    @first_value(
        "max_output_len",
        "top_k",
        "top_p",
        "temperature",
        "random_seed",
        "no_repeat_ngram_size",
        "output_generation_logits",
        "output_context_logits",
    )
    def triton_infer_fn(self, **inputs: np.ndarray):
        """Triton infer function for streaming"""
        output_dict = {}
        context_logits_available = False
        generation_logits_available = False
        prompts = str_ndarray2list(inputs.pop("prompts"))
        infer_input = {"input_texts": prompts}
        try:
            if "max_output_len" in inputs:
                infer_input["max_output_len"] = inputs.pop("max_output_len")
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")
            if "random_seed" in inputs:
                infer_input["random_seed"] = inputs.pop("random_seed")
            if "stop_words_list" in inputs:
                stop_words_list = str_ndarray2list(inputs.pop("stop_words_list"))
                infer_input["stop_words_list"] = [[stop_word] for stop_word in stop_words_list]
            if "bad_words_list" in inputs:
                bad_words_list = str_ndarray2list(inputs.pop("bad_words_list"))
                infer_input["bad_words_list"] = [[bad_word] for bad_word in bad_words_list]
            if "no_repeat_ngram_size" in inputs:
                infer_input["no_repeat_ngram_size"] = inputs.pop("no_repeat_ngram_size")
            if "task_id" in inputs:
                task_id = np.char.decode(inputs.pop("task_id").astype("bytes"), encoding="utf-8")
                infer_input["task_ids"] = task_id[0]
            if "lora_uids" in inputs:
                lora_uids = np.char.decode(inputs.pop("lora_uids").astype("bytes"), encoding="utf-8")
                infer_input["lora_uids"] = lora_uids[0].tolist()
            if "output_generation_logits" in inputs:
                generation_logits_available = inputs["output_generation_logits"]
                infer_input["output_generation_logits"] = inputs.pop("output_generation_logits")
            if "output_context_logits" in inputs:
                context_logits_available = inputs["output_context_logits"]
                infer_input["output_context_logits"] = inputs.pop("output_context_logits")

            if generation_logits_available:
                # generation_logits is a 4d torch tensor of dim [BS,1,#generated_tokens,vocab_size]
                output_texts, generation_logits = self.forward(**infer_input)
                # convert generation_logits to numpy array. Note: from my understanding since generation_logits is
                # returned as a torch tensor it won't have varying number of tokens across multiple sequences,
                # likely due to TRTLLM taking care of padding hence no addtnl padding is needed.
                output_dict["generation_logits"] = np.array(
                    [generation_logit.cpu().numpy() for generation_logit in generation_logits]
                )

            elif context_logits_available:
                output_texts, context_logits = self.forward(**infer_input)
                # context_logits is a list of tensors shaped [#tokens, vocab_size] and the len of the list  is BS
                # In case of batched inputs (i.e multiple prompts sent as a list) context_logits returned can have
                # different seq_len. Following code pads them as it can otherwise error while converting to numpy array
                context_logits = self._pad_logits(context_logits)
                # Convert context_Logits to numpy array of shape [bS, 1, padding_len, vocab_size],.
                context_logits = np.array([logit_tensor.unsqueeze(0).cpu().numpy() for logit_tensor in context_logits])
                output_dict["context_logits"] = context_logits
            else:
                output_texts = self.forward(**infer_input)
            output_dict["outputs"] = cast_output(output_texts, np.bytes_)
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_dict["outputs"] = cast_output([err_msg] * len(prompts), np.bytes_)

        return output_dict

    @batch
    @first_value("max_output_len", "top_k", "top_p", "temperature", "random_seed", "no_repeat_ngram_size")
    def triton_infer_fn_streaming(self, **inputs: np.ndarray):
        """Triton infer function for streaming"""
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_output_len" in inputs:
                infer_input["max_output_len"] = inputs.pop("max_output_len")
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")
            if "random_seed" in inputs:
                infer_input["random_seed"] = inputs.pop("random_seed")
            if "stop_words_list" in inputs:
                stop_words_list = str_ndarray2list(inputs.pop("stop_words_list"))
                infer_input["stop_words_list"] = [[stop_word] for stop_word in stop_words_list]
            if "bad_words_list" in inputs:
                bad_words_list = str_ndarray2list(inputs.pop("bad_words_list"))
                infer_input["bad_words_list"] = [[bad_word] for bad_word in bad_words_list]
            if "no_repeat_ngram_size" in inputs:
                infer_input["no_repeat_ngram_size"] = inputs.pop("no_repeat_ngram_size")
            if "task_id" in inputs:
                task_id = np.char.decode(inputs.pop("task_id").astype("bytes"), encoding="utf-8")
                infer_input["task_ids"] = task_id[0]
            if "lora_uids" in inputs:
                lora_uids = np.char.decode(inputs.pop("lora_uids").astype("bytes"), encoding="utf-8")
                infer_input["lora_uids"] = lora_uids[0].tolist()

            partial_outputs = self.forward(**infer_input, streaming=True)
            # On each request to this generator, run the model for one step and return a dict
            # with full outputs generated until this step.
            for output_texts in partial_outputs:
                yield {"outputs": cast_output(output_texts, np.bytes_)}
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)
            return {"outputs": output}

    def _prep_ptuning_table(self):
        self.task_vocab_size = 0
        for pt in self.ptuning_tables:
            if self.task_vocab_size < pt["table"].size(dim=0):
                self.task_vocab_size = pt["table"].size(dim=0)

        # pad tasks to longest task embedding table, remember the original task vtoken counts
        vtokens_embeddings = []
        self.task_vtoken_counts = []
        self.task_ids = {}
        tid = 0
        for i, ptuning_table in enumerate(self.ptuning_tables):
            original_table = ptuning_table["table"]
            vtoken_count = original_table.size(dim=0)
            padded_table = torch.zeros((self.task_vocab_size, self.get_hidden_size), dtype=original_table.dtype)
            padded_table[:vtoken_count, :] = original_table
            vtokens_embeddings.append(padded_table)
            self.task_ids[ptuning_table["task_name"]] = tid
            self.task_vtoken_counts.append(vtoken_count)
            tid = tid + 1

        if len(vtokens_embeddings) > 0:
            self.p_table = torch.stack(vtokens_embeddings, dim=0).view(-1, self.get_hidden_size)

            max_prompt_embedding_table_size = self.config['build_config']['max_prompt_embedding_table_size']
            actual_prompt_table_size = self.p_table.shape[0]

            if actual_prompt_table_size > max_prompt_embedding_table_size:
                raise Exception(
                    f"The size of the combined prompt embedding table ({actual_prompt_table_size}) is greater than max_prompt_embedding_table_size ({max_prompt_embedding_table_size})."
                )
        else:
            self.p_table = None

    def _load_prompt_tables(self):
        if self.model_dir is not None:
            pt_path = Path(os.path.join(self.model_dir, 'prompt_tables.pkl'))
            if pt_path.exists():
                with open(pt_path, 'rb') as f:
                    self.ptuning_tables = pickle.load(f)
                self._prep_ptuning_table()
            else:
                self.ptuning_tables = []

    def _get_prompt_embedding_table_ckpt(self, prompt_embeddings_checkpoint_path):
        with TarPath(prompt_embeddings_checkpoint_path) as checkpoint_archive:
            mw_path = checkpoint_archive / "model_weights.ckpt"
            if not mw_path.exists():
                mw_path = checkpoint_archive / "mp_rank_00/model_weights.ckpt"
                if not mw_path.exists():
                    raise FileNotFoundError(
                        "File: {0} could not be found in the nemo checkpoint. "
                        "Please check the nemo checkpoint format for the prompt "
                        "embedding table.".format(mw_path)
                    )

            with mw_path.open('rb') as mw_file:
                weights = torch.load(mw_file)

            weights_found = True
            if "model.embedding.adapter_layer.ptuning_adapter.inference_table" in weights:
                weights = weights["model.embedding.adapter_layer.ptuning_adapter.inference_table"]
            elif (
                "model.language_model.adapter_layer.ptuning_adapter.inference_table.prompt_table.taskname.prompt_embeddings.weight"
                in weights
            ):
                weights = weights[
                    "model.language_model.adapter_layer.ptuning_adapter.inference_table.prompt_table.taskname.prompt_embeddings.weight"
                ]
            elif 'prompt_table' in weights:
                if "prompt_table.taskname.prompt_embeddings.weight" in weights['prompt_table']:
                    weights = weights['prompt_table']["prompt_table.taskname.prompt_embeddings.weight"]
                else:
                    weights_found = False
            else:
                weights_found = False

            if not weights_found:
                raise Exception(
                    "Could not find the embedding table in the {0}. Please check the nemo file format".format(
                        prompt_embeddings_checkpoint_path
                    )
                )

            return weights.cpu().detach()

    def _get_prompt_embedding_table(
        self,
        prompt_embeddings_table=None,
        prompt_embeddings_checkpoint_path=None,
    ):
        if prompt_embeddings_table is not None and prompt_embeddings_checkpoint_path is not None:
            LOGGER.warning(
                "prompt_embeddings_table will be used and "
                "prompt_embeddings_checkpoint_path will be "
                "ignored for ptuning."
            )
            p_tuning = "use_table"
        elif prompt_embeddings_table is not None:
            p_tuning = "use_table"
        elif prompt_embeddings_checkpoint_path is not None:
            p_tuning = "use_checkpoint"
        else:
            return None, None

        if p_tuning == "use_table":
            if not isinstance(prompt_embeddings_table, np.ndarray):
                raise TypeError("Only numpy array is allowed for the prompt embeddings table.")

            if len(prompt_embeddings_table.shape) != 2:
                raise Exception("A two dimensional prompt embeddings table for a single task is only supported.")

            prompt_embeddings_table = torch.from_numpy(prompt_embeddings_table)
        elif p_tuning == "use_checkpoint":
            if not is_nemo_tarfile(prompt_embeddings_checkpoint_path):
                raise TypeError(prompt_embeddings_checkpoint_path + " is not a nemo file.")
            prompt_embeddings_table = self._get_prompt_embedding_table_ckpt(prompt_embeddings_checkpoint_path)

        dtype = self.config['pretrained_config']['dtype']
        prompt_embeddings_table = prompt_embeddings_table.to(
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)
        ).cuda()

        if prompt_embeddings_table.size(dim=1) != self.config["pretrained_config"]["hidden_size"]:
            raise Exception(
                "Hidden dimension of the model is {0} and does not match with the dimension of the prompt table.".format(
                    self.config["pretrained_config"]["hidden_size"]
                )
            )

        return prompt_embeddings_table

    def _load_config_file(self):
        config_path = Path(self.engine_dir) / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"File: {config_path} could not be found.")

    def _load(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.ptuning_tables = []

        if Path(self.model_dir).exists():
            folders = os.listdir(self.model_dir)
            if len(folders) > 0:
                try:
                    self._load_config_file()
                    self.tokenizer = get_tokenizer(self.model_dir)
                    self.model = load(
                        tokenizer=self.tokenizer,
                        engine_dir=self.engine_dir,
                        lora_ckpt_list=self.lora_ckpt_list,
                        use_python_runtime=self.use_python_runtime,
                        enable_chunked_context=self.enable_chunked_context,
                        max_tokens_in_paged_kv_cache=self.max_tokens_in_paged_kv_cache,
                        multi_block_mode=self.multi_block_mode,
                    )
                    self._load_prompt_tables()
                except Exception as error:
                    raise RuntimeError(
                        "Files in the TensorRT-LLM folder are corrupted and the model needs to be exported again."
                    ) from error

    def unload_engine(self):
        """Unload engine"""
        unload_engine()
