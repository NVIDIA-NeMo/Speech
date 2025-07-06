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

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

LOGGER = logging.getLogger("NeMo")

triton_supported = True
try:
    from nemo.deploy import DeployPyTriton
    from nemo.deploy.nlp import NemoQueryLLM
except Exception as e:
    LOGGER.warning(f"Cannot import Triton, deployment will not be available. {type(e).__name__}: {e}")
    triton_supported = False

in_framework_supported = True
try:
    from megatron.core.inference.common_inference_params import CommonInferenceParams

    from nemo.deploy.nlp import NemoQueryLLMPyTorch
    from nemo.deploy.nlp.megatronllm_deployable import MegatronLLMDeploy, MegatronLLMDeployableNemo2
except Exception as e:
    LOGGER.warning(
        "Cannot import MegatronLLMDeploy* classes, or NemoQueryLLMPyTorch, or CommonInferenceParams, "
        f"in-framework inference will not be available. Reason: {type(e).__name__}: {e}"
    )
    in_framework_supported = False

trt_llm_supported = True
try:
    from nemo.export.tensorrt_llm import TensorRTLLM
except Exception as e:
    LOGGER.warning(f"Cannot import the TensorRTLLM exporter, it will not be available. {type(e).__name__}: {e}")
    trt_llm_supported = False

vllm_supported = True
try:
    from nemo.export.vllm_exporter import vLLMExporter
except Exception as e:
    LOGGER.warning(f"Cannot import the vLLM exporter, it will not be available. {type(e).__name__}: {e}")
    vllm_supported = False


class UsageError(Exception):
    pass


@dataclass
class FunctionalResult:
    regular_pass: Optional[bool] = None
    deployed_pass: Optional[bool] = None


@dataclass
class AccuracyResult:
    accuracy: float
    accuracy_relaxed: float
    deployed_accuracy: float
    deployed_accuracy_relaxed: float
    evaluation_time: float


def get_accuracy_with_lambada(model, nq, task_ids, lora_uids, test_data_path):
    # lambada dataset based accuracy test, which includes more than 5000 sentences.
    # Use generated last token with original text's last token for accuracy comparison.
    # If the generated last token start with the original token, trtllm_correct make an increment.
    # It generates a CSV file for text comparison detail.

    correct_answers = 0
    correct_answers_deployed = 0
    correct_answers_relaxed = 0
    correct_answers_deployed_relaxed = 0
    all_expected_outputs = []
    all_actual_outputs = []

    with open(test_data_path, 'r') as file:
        records = json.load(file)

        eval_start = time.monotonic()
        for record in records:
            prompt = record["text_before_last_word"]
            expected_output = record["last_word"].strip().lower()
            all_expected_outputs.append(expected_output)
            if model is not None:
                if in_framework_supported and isinstance(model, MegatronLLMDeployableNemo2):
                    model_output = model.generate(
                        prompts=[prompt],
                        inference_params=CommonInferenceParams(
                            temperature=0.1,
                            top_k=1,
                            top_p=0.0,
                            num_tokens_to_generate=1,
                            return_log_probs=False,
                        ),
                    )
                    model_output = model_output[0].generated_text  # Index [0] as a single prompt is used
                else:
                    model_output = model.forward(
                        input_texts=[prompt],
                        max_output_len=1,
                        top_k=1,
                        top_p=0.0,
                        temperature=0.1,
                        task_ids=task_ids,
                        lora_uids=lora_uids,
                    )
                    model_output = model_output[0][0].strip().lower()
                all_actual_outputs.append(model_output)

                if expected_output == model_output:
                    correct_answers += 1

                if (
                    expected_output == model_output
                    or model_output.startswith(expected_output)
                    or expected_output.startswith(model_output)
                ):
                    if len(model_output) == 1 and len(expected_output) > 1:
                        continue
                    correct_answers_relaxed += 1

            if nq is not None:
                if in_framework_supported and isinstance(nq, NemoQueryLLMPyTorch):
                    deployed_output = nq.query_llm(
                        prompts=[prompt],
                        max_length=1,
                        top_k=1,
                        top_p=0.0,
                        temperature=0.1,
                    )
                    # Accessing [0][0] of "text" is to get a raw string entry from a NumPy array
                    # for a single prompt (batch size = 1) and stripping prefix if needed:
                    deployed_output = deployed_output["choices"][0]["text"][0][0][0:].strip().lower()
                else:
                    deployed_output = nq.query_llm(
                        prompts=[prompt],
                        max_output_len=1,
                        top_k=1,
                        top_p=0.0,
                        temperature=0.1,
                        task_id=task_ids,
                    )
                    deployed_output = deployed_output[0][0].strip().lower()

                if expected_output == deployed_output:
                    correct_answers_deployed += 1

                if (
                    expected_output == deployed_output
                    or deployed_output.startswith(expected_output)
                    or expected_output.startswith(deployed_output)
                ):
                    if len(deployed_output) == 1 and len(expected_output) > 1:
                        continue
                    correct_answers_deployed_relaxed += 1
        eval_end = time.monotonic()

    return AccuracyResult(
        accuracy=correct_answers / len(all_expected_outputs),
        accuracy_relaxed=correct_answers_relaxed / len(all_expected_outputs),
        deployed_accuracy=correct_answers_deployed / len(all_expected_outputs),
        deployed_accuracy_relaxed=correct_answers_deployed_relaxed / len(all_expected_outputs),
        evaluation_time=eval_end - eval_start,
    )


# Tests if the model outputs contain the expected keywords.
def check_model_outputs(streaming: bool, model_outputs, expected_outputs: List[str]) -> bool:

    # In streaming mode, we get a list of lists of lists, and we only care about the last item in that list
    if streaming:
        if len(model_outputs) == 0:
            return False
        model_outputs = model_outputs[-1]

    # See if we have the right number of final answers.
    if len(model_outputs) != len(expected_outputs):
        return False

    # Check the presence of keywords in the final answers.
    for i in range(len(model_outputs)):
        if expected_outputs[i] not in model_outputs[i][0]:
            return False

    return True


def run_inference(
    model_name,
    model_type,
    prompts,
    expected_outputs,
    checkpoint_path,
    model_dir,
    use_vllm,
    use_huggingface,
    max_batch_size=8,
    use_embedding_sharing=False,
    max_input_len=128,
    max_output_len=128,
    max_num_tokens=None,
    use_parallel_embedding=False,
    ptuning=False,
    p_tuning_checkpoint=None,
    lora=False,
    lora_checkpoint=None,
    tp_size=1,
    pp_size=1,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    run_accuracy=False,
    debug=True,
    streaming=False,
    stop_words_list=None,
    test_cpp_runtime=False,
    test_deployment=False,
    test_data_path=None,
    save_engine=False,
    fp8_quantized=False,
    fp8_kvcache=False,
    trt_llm_export_kwargs=None,
    vllm_export_kwargs=None,
) -> Tuple[Optional[FunctionalResult], Optional[AccuracyResult]]:
    if trt_llm_export_kwargs is None:
        trt_llm_export_kwargs = {}

    if vllm_export_kwargs is None:
        vllm_export_kwargs = {}

    if Path(checkpoint_path).exists():
        if tp_size > torch.cuda.device_count():
            print(
                "Path: {0} and model: {1} with {2} tps won't be tested since available # of gpus = {3}".format(
                    checkpoint_path, model_name, tp_size, torch.cuda.device_count()
                )
            )
            return (None, None)

        Path(model_dir).mkdir(parents=True, exist_ok=True)

        if debug:
            print("")
            print("")
            print(
                "################################################## NEW TEST ##################################################"
            )
            print("")

            print("Path: {0} and model: {1} with {2} tps will be tested".format(checkpoint_path, model_name, tp_size))

        prompt_embeddings_checkpoint_path = None
        task_ids = None
        max_prompt_embedding_table_size = 0

        if ptuning:
            if Path(p_tuning_checkpoint).exists():
                prompt_embeddings_checkpoint_path = p_tuning_checkpoint
                max_prompt_embedding_table_size = 8192
                task_ids = ["0"]
                if debug:
                    print("---- PTuning enabled.")
            else:
                print("---- PTuning could not be enabled and skipping the test.")
                return (None, None)

        lora_ckpt_list = None
        lora_uids = None
        use_lora_plugin = None
        lora_target_modules = None

        if lora:
            if Path(lora_checkpoint).exists():
                lora_ckpt_list = [lora_checkpoint]
                lora_uids = ["0", "-1", "0"]
                use_lora_plugin = "bfloat16"
                lora_target_modules = ["attn_qkv"]
                if debug:
                    print("---- LoRA enabled.")
            else:
                print("---- LoRA could not be enabled and skipping the test.")
                return (None, None)

        if use_vllm:
            exporter = vLLMExporter()

            exporter.export(
                nemo_checkpoint=checkpoint_path,
                model_dir=model_dir,
                model_type=model_type,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                max_model_len=max_input_len + max_output_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                **vllm_export_kwargs,
            )
        else:
            exporter = TensorRTLLM(model_dir, lora_ckpt_list, load_model=False)
            if use_huggingface:
                exporter.export_hf_model(
                    hf_model_path=checkpoint_path,
                    max_batch_size=max_batch_size,
                    tensor_parallelism_size=tp_size,
                    max_input_len=max_input_len,
                    max_num_tokens=max_num_tokens,
                    model_type=model_type,
                )
            else:
                exporter.export(
                    nemo_checkpoint_path=checkpoint_path,
                    model_type=model_type,
                    tensor_parallelism_size=tp_size,
                    pipeline_parallelism_size=pp_size,
                    max_input_len=max_input_len,
                    max_seq_len=(max_input_len + max_output_len),
                    max_batch_size=max_batch_size,
                    use_parallel_embedding=use_parallel_embedding,
                    max_prompt_embedding_table_size=max_prompt_embedding_table_size,
                    use_lora_plugin=use_lora_plugin,
                    lora_target_modules=lora_target_modules,
                    max_num_tokens=max_num_tokens,
                    use_embedding_sharing=use_embedding_sharing,
                    fp8_quantized=fp8_quantized,
                    fp8_kvcache=fp8_kvcache,
                    **trt_llm_export_kwargs,
                )

        if ptuning:
            exporter.add_prompt_table(
                task_name="0",
                prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path,
            )

        output = exporter.forward(
            input_texts=prompts,
            max_output_len=max_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            task_ids=task_ids,
            lora_uids=lora_uids,
            streaming=streaming,
            stop_words_list=stop_words_list,
        )

        # Unwrap the generator if needed
        output = list(output)

        functional_result = FunctionalResult()

        # Check non-deployed funcitonal correctness
        if args.functional_test:
            functional_result.regular_pass = True
            if not check_model_outputs(streaming, output, expected_outputs):
                LOGGER.warning("Model outputs don't match the expected result.")
                functional_result.regular_pass = False

        output_cpp = ""
        if test_cpp_runtime and not use_lora_plugin and not ptuning and not use_vllm:
            # This may cause OOM for large models as it creates 2nd instance of a model
            exporter_cpp = TensorRTLLM(
                model_dir,
                load_model=True,
                use_python_runtime=False,
            )

            output_cpp = exporter_cpp.forward(
                input_texts=prompts,
                max_output_len=max_output_len,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

        nq = None
        nm = None
        output_deployed = ""
        if test_deployment:
            nm = DeployPyTriton(
                model=exporter,
                triton_model_name=model_name,
                http_port=8000,
            )
            nm.deploy()
            nm.run()
            nq = NemoQueryLLM(url="localhost:8000", model_name=model_name)

            output_deployed = nq.query_llm(
                prompts=prompts,
                max_output_len=max_output_len,
                top_k=1,
                top_p=0.0,
                temperature=1.0,
                lora_uids=lora_uids,
            )

            # Unwrap the generator if needed
            output_deployed = list(output_deployed)

            # Check deployed funcitonal correctness
            if args.functional_test:
                functional_result.deployed_pass = True
                if not check_model_outputs(streaming, output_deployed, expected_outputs):
                    LOGGER.warning("Deployed model outputs don't match the expected result.")
                    functional_result.deployed_pass = False

        if debug or functional_result.regular_pass == False or functional_result.deployed_pass == False:
            print("")
            print("--- Prompt: ", prompts)
            print("")
            print("--- Expected keywords: ", expected_outputs)
            print("")
            print("--- Output: ", output)
            print("")
            print("--- Output deployed: ", output_deployed)
            print("")
            print("")
            print("--- Output with C++ runtime: ", output_cpp)
            print("")

        accuracy_result = None
        if run_accuracy:
            print("Start model accuracy testing ...")
            accuracy_result = get_accuracy_with_lambada(exporter, nq, task_ids, lora_uids, test_data_path)

        if test_deployment:
            nm.stop()

        if not save_engine and model_dir:
            shutil.rmtree(model_dir)

        return (functional_result, accuracy_result)
    else:
        raise Exception("Checkpoint {0} could not be found.".format(checkpoint_path))


def run_in_framework_inference(
    model_name,
    prompts,
    checkpoint_path,
    num_gpus=1,
    max_output_len=128,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    run_accuracy=False,
    debug=True,
    test_data_path=None,
    enable_flash_decode=True,
    legacy_ckpt=False,
) -> Tuple[Optional[FunctionalResult], Optional[AccuracyResult]]:
    if Path(checkpoint_path).exists():
        if debug:
            print("")
            print("")
            print(
                "################################################## NEW TEST ##################################################"
            )
            print("")

            print("Path: {0} and model: {1} will be tested".format(checkpoint_path, model_name))

        deployed_model = MegatronLLMDeploy.get_deployable(
            checkpoint_path, num_gpus, enable_flash_decode=enable_flash_decode, legacy_ckpt=legacy_ckpt
        )

        nm = DeployPyTriton(
            model=deployed_model,
            triton_model_name=model_name,
            http_port=8000,
        )
        nm.deploy()
        nm.run()
        nq = NemoQueryLLMPyTorch(url="localhost:8000", model_name=model_name)

        output_deployed = nq.query_llm(
            prompts=prompts, top_k=top_k, top_p=top_p, temperature=temperature, max_length=max_output_len
        )
        output_deployed = output_deployed["choices"][0]["text"]

        # Unwrap the generator if needed
        output_deployed = list(output_deployed)
        print("\n --------- Output: ", output_deployed)

        accuracy_result = None
        if run_accuracy:
            print("Start model accuracy testing ...")
            # This script is not written with torch.distributed support in mind, so running non-deployed in-framework models on multiple devices will not work
            accuracy_result = get_accuracy_with_lambada(deployed_model, nq, None, None, test_data_path)

        nm.stop()

        return (None, accuracy_result)
    else:
        raise Exception("Checkpoint {0} could not be found.".format(checkpoint_path))


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--min_tps",
        type=int,
        default=1,
        required=True,
    )
    parser.add_argument(
        "--max_tps",
        type=int,
    )
    parser.add_argument(
        "--pps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/nemo_checkpoint/",
        required=False,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
    )
    parser.add_argument(
        "--use_parallel_embedding",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--p_tuning_checkpoint",
        type=str,
    )
    parser.add_argument(
        "--ptuning",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
    )
    parser.add_argument(
        "--lora",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--run_accuracy",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--streaming", default=False, action="store_true")
    parser.add_argument(
        "--test_cpp_runtime",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--test_deployment",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--functional_test",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_engine",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--use_vllm",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--use_huggingface",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--enable_flash_decode",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--in_framework",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--legacy_ckpt",
        type=str,
        default="False",
        help="Load checkpoint saved with TE < 1.14 (only for in-framework inference)",
    )
    parser.add_argument(
        "-gmu",
        '--gpu_memory_utilization',
        default=0.95,  # 0.95 is needed to run Mixtral-8x7B on 2x48GB GPUs
        type=float,
        help="GPU memory utilization percentage for vLLM.",
    )
    parser.add_argument(
        "-fp8",
        "--export_fp8_quantized",
        default="auto",
        type=str,
        help="Enables exporting to a FP8-quantized TRT LLM checkpoint",
    )
    parser.add_argument(
        "-kv_fp8",
        "--use_fp8_kv_cache",
        default="auto",
        type=str,
        help="Enables exporting with FP8-quantizatized KV-cache",
    )
    parser.add_argument(
        "--trt_llm_export_kwargs",
        default={},
        type=json.loads,
        help="Extra keyword arguments passed to TensorRTLLM.export",
    )
    parser.add_argument(
        "--vllm_export_kwargs",
        default={},
        type=json.loads,
        help="Extra keyword arguments passed to vLLMExporter.export",
    )

    args = parser.parse_args()

    def str_to_bool(name: str, s: str, optional: bool = False) -> Optional[bool]:
        s = s.lower()
        true_strings = ["true", "1"]
        false_strings = ["false", "0"]
        if s == '':
            return False
        if s in true_strings:
            return True
        if s in false_strings:
            return False
        if optional and s == 'auto':
            return None
        raise UsageError(f"Invalid boolean value for argument --{name}: '{s}'")

    args.model_type = None if str(args.model_type).lower() == "none" else args.model_type
    args.test_cpp_runtime = str_to_bool("test_cpp_runtime", args.test_cpp_runtime)
    args.test_deployment = str_to_bool("test_deployment", args.test_deployment)
    args.functional_test = str_to_bool("functional_test", args.functional_test)
    args.save_engine = str_to_bool("save_engine", args.save_engine)
    args.run_accuracy = str_to_bool("run_accuracy", args.run_accuracy)
    args.use_vllm = str_to_bool("use_vllm", args.use_vllm)
    args.use_huggingface = str_to_bool("use_huggingface", args.use_huggingface)
    args.enable_flash_decode = str_to_bool("enable_flash_decode", args.enable_flash_decode)
    args.lora = str_to_bool("lora", args.lora)
    args.ptuning = str_to_bool("ptuning", args.ptuning)
    args.use_parallel_embedding = str_to_bool("use_parallel_embedding", args.use_parallel_embedding)
    args.in_framework = str_to_bool("in_framework", args.in_framework)
    args.export_fp8_quantized = str_to_bool("export_fp8_quantized", args.export_fp8_quantized, optional=True)
    args.use_fp8_kv_cache = str_to_bool("use_fp8_kv_cache", args.use_fp8_kv_cache, optional=True)
    args.legacy_ckpt = str_to_bool("legacy_ckpt", args.legacy_ckpt)

    return args


def run_inference_tests(args):
    if not args.use_vllm and not args.in_framework and not trt_llm_supported:
        raise UsageError("TensorRT-LLM engine is not supported in this environment.")

    if args.use_vllm and not vllm_supported:
        raise UsageError("vLLM engine is not supported in this environment.")

    if args.in_framework and not in_framework_supported:
        raise UsageError("In-framework inference is not supported in this environment.")

    if args.use_vllm and (args.ptuning or args.lora):
        raise UsageError("The vLLM integration currently does not support P-tuning or LoRA.")

    if args.test_deployment and not triton_supported:
        raise UsageError("Deployment tests are not available because Triton is not supported in this environment.")

    if args.run_accuracy and args.test_data_path is None:
        raise UsageError("Accuracy testing requires the --test_data_path argument.")

    if args.max_tps is None:
        args.max_tps = args.min_tps

    if args.use_vllm and args.min_tps != args.max_tps:
        raise UsageError(
            "vLLM doesn't support changing tensor parallel group size without relaunching the process. "
            "Use the same value for --min_tps and --max_tps."
        )

    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    result_dic: Dict[int, Tuple[FunctionalResult, Optional[AccuracyResult]]] = {}

    if not args.in_framework and args.model_dir is None:
        raise Exception("When using custom checkpoints, --model_dir is required.")

    prompts = ["The capital of France is", "Largest animal in the sea is"]
    expected_outputs = ["Paris", "blue whale"]
    tps = args.min_tps

    while tps <= args.max_tps:
        if args.in_framework:
            result_dic[tps] = run_in_framework_inference(
                model_name=args.model_name,
                prompts=prompts,
                checkpoint_path=args.checkpoint_dir,
                num_gpus=tps,
                max_output_len=args.max_output_len,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                run_accuracy=args.run_accuracy,
                debug=args.debug,
                test_data_path=args.test_data_path,
                enable_flash_decode=args.enable_flash_decode,
                legacy_ckpt=args.legacy_ckpt,
            )
        else:
            result_dic[tps] = run_inference(
                model_name=args.model_name,
                model_type=args.model_type,
                prompts=prompts,
                expected_outputs=expected_outputs,
                checkpoint_path=args.checkpoint_dir,
                model_dir=args.model_dir,
                use_vllm=args.use_vllm,
                use_huggingface=args.use_huggingface,
                tp_size=tps,
                pp_size=args.pps,
                max_batch_size=args.max_batch_size,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
                max_num_tokens=args.max_num_tokens,
                use_parallel_embedding=args.use_parallel_embedding,
                ptuning=args.ptuning,
                p_tuning_checkpoint=args.p_tuning_checkpoint,
                lora=args.lora,
                lora_checkpoint=args.lora_checkpoint,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                run_accuracy=args.run_accuracy,
                debug=args.debug,
                streaming=args.streaming,
                test_deployment=args.test_deployment,
                test_cpp_runtime=args.test_cpp_runtime,
                test_data_path=args.test_data_path,
                save_engine=args.save_engine,
                fp8_quantized=args.export_fp8_quantized,
                fp8_kvcache=args.use_fp8_kv_cache,
                trt_llm_export_kwargs=args.trt_llm_export_kwargs,
                vllm_export_kwargs=args.vllm_export_kwargs,
            )

        tps = tps * 2

    functional_test_result = "PASS"
    accuracy_test_result = "PASS"
    print_separator = False
    print("============= Test Summary ============")
    # in-framework tests will only return deployed model accuracy results for tps > 1
    deployed_tests_only = args.in_framework and args.max_tps > 1
    for num_tps, results in result_dic.items():
        functional_result, accuracy_result = results

        if print_separator:
            print("---------------------------------------")
        print_separator = True

        def optional_bool_to_pass_fail(b: Optional[bool]):
            if b is None:
                return "N/A"
            return "PASS" if b else "FAIL"

        print(f"Tensor Parallelism:              {num_tps}")

        if args.functional_test and functional_result is not None:
            print(f"Functional Test:                 {optional_bool_to_pass_fail(functional_result.regular_pass)}")
            print(f"Deployed Functional Test:        {optional_bool_to_pass_fail(functional_result.deployed_pass)}")

            if functional_result.regular_pass == False:
                functional_test_result = "FAIL"
            if functional_result.deployed_pass == False:
                functional_test_result = "FAIL"

        if args.run_accuracy and accuracy_result is not None:
            print(f"Model Accuracy:                  {accuracy_result.accuracy:.4f}")
            print(f"Relaxed Model Accuracy:          {accuracy_result.accuracy_relaxed:.4f}")
            print(f"Deployed Model Accuracy:         {accuracy_result.deployed_accuracy:.4f}")
            print(f"Deployed Relaxed Model Accuracy: {accuracy_result.deployed_accuracy_relaxed:.4f}")
            print(f"Evaluation Time [s]:             {accuracy_result.evaluation_time:.2f}")
            if (deployed_tests_only and accuracy_result.deployed_accuracy_relaxed < args.accuracy_threshold) or (
                not deployed_tests_only and accuracy_result.accuracy_relaxed < args.accuracy_threshold
            ):
                accuracy_test_result = "FAIL"

    print("=======================================")
    if args.functional_test:
        print(f"Functional: {functional_test_result}")
    if args.run_accuracy:
        print(f"Acccuracy: {accuracy_test_result}")

    if functional_test_result == "FAIL":
        raise Exception("Functional test failed")

    if accuracy_test_result == "FAIL":
        raise Exception(f"Model accuracy is below {args.accuracy_threshold}")


if __name__ == '__main__':
    try:
        args = get_args()
        run_inference_tests(args)
    except UsageError as e:
        LOGGER.error(f"{e}")
        raise e
    except argparse.ArgumentError as e:
        LOGGER.error(f"{e}")
        raise e
