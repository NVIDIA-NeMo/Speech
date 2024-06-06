# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import tensorrt_llm
import torch
import wrapt

from nemo.deploy import ITritonDeployable
from nemo.export.tarutils import TarPath, unpack_tarball
from nemo.export.trt_llm.converter.model_converter import model_to_trtllm_ckpt
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import get_tokenzier, is_nemo_file, load_nemo_model
from nemo.export.trt_llm.qnemo import qnemo_to_tensorrt_llm
from nemo.export.trt_llm.qnemo.tokenizer_utils import get_nmt_tokenizer
from nemo.export.trt_llm.tensorrt_llm_build import build_and_save_engine
from nemo.export.trt_llm.tensorrt_llm_run import generate, generate_streaming, load

use_deploy = True
try:
    from nemo.deploy.utils import cast_output, str_ndarray2list
except Exception:
    use_deploy = False


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


class TensorRTLLM(ITritonDeployable):
    """
    Exports nemo checkpoints to TensorRT-LLM and run fast inference.

    Example:
        from nemo.export import TensorRTLLM

        trt_llm_exporter = TensorRTLLM(model_dir="/path/for/model/files")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="llama",
            n_gpus=1,
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
    ):
        """
        Args:
            model_dir (str): path for storing the TensorRT-LLM model files.
            lora_ckpt_list (List[str]): lora checkpoint paths.
            load_model (bool): load TensorRT-LLM model if the engine files exist in the model_dir.
            use_python_runtime (bool): whether to use python or c++ runtime.
        """

        self.model_dir = model_dir
        self.lora_ckpt_list = lora_ckpt_list
        self.use_python_runtime = use_python_runtime
        self.model = None
        self.tokenizer = None
        self.n_gpus = None
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
        model_type: str,
        delete_existing_files: bool = True,
        n_gpus: int = 1,
        tensor_parallel_size: int = None,
        pipeline_parallel_size: int = None,
        max_input_token: int = 256,
        max_output_token: int = 256,
        max_batch_size: int = 8,
        max_prompt_embedding_table_size=None,
        use_parallel_embedding: bool = False,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        dtype: str = "bfloat16",
        load_model: bool = True,
        enable_multi_block_mode: bool = False,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        max_lora_rank: int = 64,
        max_num_tokens: int = None,
        opt_num_tokens: int = None,
        save_nemo_model_config: bool = False,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            nemo_checkpoint_path (str): path for the nemo checkpoint.
            model_type (str): type of the model. Currently, "llama", "gptnext", "falcon", and "starcoder" are supported.
            delete_existing_files (bool): if Truen, deletes all the files in model_dir.
            n_gpus (int): number of GPUs to use for inference.
            tensor_parallel_size (int): tensor parallelism.
            pipeline_parallel_size (int): pipeline parallelism.
            max_input_token (int): max input length.
            max_output_token (int): max output length.
            max_batch_size (int): max batch size.
            max_prompt_embedding_table_size (int): max prompt embedding size.
            use_parallel_embedding (bool): whether to use parallel embedding feature of TRT-LLM or not
            paged_kv_cache (bool): if True, uses kv cache feature of the TensorRT-LLM.
            remove_input_padding (bool): enables removing input padding or not.
            dtype (str): Floating point type for model weights (Supports BFloat16/Float16).
            load_model (bool): load TensorRT-LLM model after the export.
            enable_multi_block_mode (bool): enable faster decoding in multihead attention. Required for long context.
            use_lora_plugin (str): use dynamic lora or not.
            lora_target_modules (List[str]): list of the target lora modules.
            max_lora_rank (int): maximum lora rank.
            max_num_tokens (int):
            opt_num_tokens (int):
            save_nemo_model_config (bool):
        """

        if model_type not in self.get_supported_models_list:
            raise Exception(
                "Model {0} is not currently a supported model type. "
                "Supported model types are llama, gptnext, falcon, and starcoder".format(model_type)
            )

        if model_type == "gpt" or model_type == "starcoder":
            model_type = "gptnext"

        if model_type == "mixtral":
            model_type = "llama"

        if pipeline_parallel_size is None:
            tensor_parallel_size = n_gpus
            pipeline_parallel_size = 1
        elif tensor_parallel_size is None:
            tensor_parallel_size = 1
            pipeline_parallel_size = n_gpus

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

        if max_prompt_embedding_table_size is None:
            max_prompt_embedding_table_size = 0

        self.model = None

        if tensorrt_llm.mpi_rank() == 0:
            tmp_dir = tempfile.TemporaryDirectory()
            nemo_export_dir = Path(tmp_dir.name)

            if nemo_checkpoint_path.endswith("qnemo"):
                if os.path.isdir(nemo_checkpoint_path):
                    nemo_export_dir = nemo_checkpoint_path
                else:
                    unpack_tarball(nemo_checkpoint_path, tmp_dir.name)
                    nemo_checkpoint_path = tmp_dir.name
                self.tokenizer = get_nmt_tokenizer(nemo_checkpoint_path)

                qnemo_to_tensorrt_llm(
                    nemo_checkpoint_path=nemo_checkpoint_path,
                    engine_dir=self.model_dir,
                    max_input_len=max_input_token,
                    max_output_len=max_output_token,
                    max_batch_size=max_batch_size,
                    max_prompt_embedding_table_size=max_prompt_embedding_table_size,
                    lora_target_modules=lora_target_modules,
                )
            else:
                model, model_configs, self.tokenizer = load_nemo_model(nemo_checkpoint_path, nemo_export_dir)
                weights_dicts, model_configs = model_to_trtllm_ckpt(
                    model=model,
                    nemo_model_config=model_configs,
                    nemo_export_dir=nemo_export_dir,
                    decoder_type=model_type,
                    dtype=dtype,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    use_parallel_embedding=use_parallel_embedding,
                )

                for weight_dict, model_config in zip(weights_dicts, model_configs):
                    build_and_save_engine(
                        max_input_len=max_input_token,
                        max_output_len=max_output_token,
                        max_batch_size=max_batch_size,
                        model_config=model_config,
                        model_weights=weight_dict,
                        model_dir=self.model_dir,
                        model_type=model_type,
                        lora_ckpt_list=self.lora_ckpt_list,
                        use_lora_plugin=use_lora_plugin,
                        max_lora_rank=max_lora_rank,
                        lora_target_modules=lora_target_modules,
                        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
                        enable_multi_block_mode=enable_multi_block_mode,
                        paged_kv_cache=paged_kv_cache,
                        remove_input_padding=remove_input_padding,
                        max_num_tokens=max_num_tokens,
                        opt_num_tokens=opt_num_tokens,
                    )

            tokenizer_path = os.path.join(nemo_export_dir, "tokenizer.model")
            if os.path.exists(tokenizer_path):
                shutil.copy(tokenizer_path, self.model_dir)
            else:
                self.tokenizer.save_pretrained(os.path.join(self.model_dir, 'huggingface_tokenizer'))

            nemo_model_config = os.path.join(nemo_export_dir, "model_config.yaml")
            if os.path.exists(nemo_model_config):
                shutil.copy(nemo_model_config, self.model_dir)

            tmp_dir.cleanup()

        if tensorrt_llm.mpi_world_size() > 1:
            tensorrt_llm.mpi_barrier()

        if load_model:
            self._load()

    def forward(
        self,
        input_texts: List[str],
        max_output_token: int = 64,
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
        **sampling_kwargs,
    ):
        """
        Exports nemo checkpoints to TensorRT-LLM.

        Args:
            input_texts (List(str)): list of sentences.
            max_output_token (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            task_ids (List(str)): list of the task ids for the prompt tables.
            prompt_embeddings_table (List(float)): prompt embeddings table.
            prompt_embeddings_checkpoint_path (str): path for the nemo checkpoint for the prompt embedding table.
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
                    max_output_len=max_output_token,
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
                    **sampling_kwargs,
                )
            else:
                return generate_streaming(
                    input_texts=input_texts,
                    max_output_len=max_output_token,
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
        if self.ptuning_tables is not None:
            for i in range(len(self.ptuning_tables)):
                if self.ptuning_tables[i]["task_name"] == task_name:
                    self.ptuning_tables.pop(i)
                    with open(os.path.join(self.model_dir, 'prompt_tables.pkl'), 'wb') as f:
                        pickle.dump(self.ptuning_tables, f)
                    return
            self._prep_ptuning_table()

    @property
    def get_supported_models_list(self):
        # gpt and gptnext are the same. Keeping the gptnext due to backward compatibility.
        return ["gpt", "gptnext", "llama", "falcon", "starcoder", "mixtral", "gemma"]

    @property
    def get_hidden_size(self):
        if self.config is None:
            return None
        else:
            return self.config["pretrained_config"]["hidden_size"]

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_token", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="stop_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="bad_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="no_repeat_ngram_size", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="task_id", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="lora_uids", shape=(-1,), dtype=bytes, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (Tensor(name="outputs", shape=(-1,), dtype=bytes),)
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_output_token" in inputs:
                infer_input["max_output_token"] = inputs.pop("max_output_token")[0][0]
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")[0][0]
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")[0][0]
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")[0][0]
            if "random_seed" in inputs:
                infer_input["random_seed"] = inputs.pop("random_seed")[0][0]
            if "stop_words_list" in inputs:
                stop_words_list = str_ndarray2list(inputs.pop("stop_words_list"))
                infer_input["stop_words_list"] = [[stop_word] for stop_word in stop_words_list]
            if "bad_words_list" in inputs:
                bad_words_list = str_ndarray2list(inputs.pop("bad_words_list"))
                infer_input["bad_words_list"] = [[bad_word] for bad_word in bad_words_list]
            if "no_repeat_ngram_size" in inputs:
                infer_input["no_repeat_ngram_size"] = inputs.pop("no_repeat_ngram_size")[0][0]
            if "task_id" in inputs:
                task_id = np.char.decode(inputs.pop("task_id").astype("bytes"), encoding="utf-8")
                infer_input["task_ids"] = task_id[0]
            if "lora_uids" in inputs:
                lora_uids = np.char.decode(inputs.pop("lora_uids").astype("bytes"), encoding="utf-8")
                infer_input["lora_uids"] = lora_uids[0].tolist()

            output_texts = self.forward(**infer_input)
            output = cast_output(output_texts, np.bytes_)
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output = cast_output([err_msg], np.bytes_)

        return {"outputs": output}

    @batch
    def triton_infer_fn_streaming(self, **inputs: np.ndarray):
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_output_token" in inputs:
                infer_input["max_output_token"] = inputs.pop("max_output_token")[0][0]
            if "top_k" in inputs:
                infer_input["top_k"] = inputs.pop("top_k")[0][0]
            if "top_p" in inputs:
                infer_input["top_p"] = inputs.pop("top_p")[0][0]
            if "temperature" in inputs:
                infer_input["temperature"] = inputs.pop("temperature")[0][0]
            if "random_seed" in inputs:
                infer_input["random_seed"] = inputs.pop("random_seed")[0][0]
            if "stop_words_list" in inputs:
                stop_words_list = str_ndarray2list(inputs.pop("stop_words_list"))
                infer_input["stop_words_list"] = [[stop_word] for stop_word in stop_words_list]
            if "bad_words_list" in inputs:
                bad_words_list = str_ndarray2list(inputs.pop("bad_words_list"))
                infer_input["bad_words_list"] = [[bad_word] for bad_word in bad_words_list]
            if "no_repeat_ngram_size" in inputs:
                infer_input["no_repeat_ngram_size"] = inputs.pop("no_repeat_ngram_size")[0][0]
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
            if not is_nemo_file(prompt_embeddings_checkpoint_path):
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
        engine_dir = Path(self.model_dir)
        config_path = engine_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError("file: {0} could not be found.".format(config_path))

    def _load(self):
        self.model = None
        self.tokenizer = None
        self.n_gpus = None
        self.config = None
        self.ptuning_tables = []

        if Path(self.model_dir).exists():
            folders = os.listdir(self.model_dir)
            if len(folders) > 0:
                try:
                    self._load_config_file()
                    self.tokenizer = get_tokenzier(Path(os.path.join(self.model_dir)))
                    self.model = load(
                        tokenizer=self.tokenizer,
                        engine_dir=self.model_dir,
                        lora_ckpt_list=self.lora_ckpt_list,
                        use_python_runtime=self.use_python_runtime,
                    )
                    self._load_prompt_tables()
                except Exception as error:
                    raise Exception(
                        "Files in the TensorRT-LLM folder is corrupted and "
                        "model needs to be exported again. "
                        "Error message: " + repr(error)
                    ) from error
