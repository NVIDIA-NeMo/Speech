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

import argparse
import logging
import sys

from nemo.export.tensorrt_llm import TensorRTLLM

LOGGER = logging.getLogger("NeMo")


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Exports nemo models stored in nemo checkpoints to TensorRT-LLM",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", required=True, type=str, help="Source .nemo file")
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=True,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "gemma"],
        help="Type of the model. gptnext, gpt, llama, falcon, and starcoder are only supported."
        " gptnext and gpt are the same and keeping it for backward compatibility",
    )
    parser.add_argument(
        "-mr", "--model_repository", required=True, default=None, type=str, help="Folder for the trt-llm model files"
    )
    parser.add_argument("-ng", "--num_gpus", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument("-tps", "--tensor_parallelism_size", default=1, type=int, help="Tensor parallelism size")
    parser.add_argument("-pps", "--pipeline_parallelism_size", default=1, type=int, help="Pipeline parallelism size")
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )
    parser.add_argument("-mil", "--max_input_len", default=256, type=int, help="Max input length of the model")
    parser.add_argument("-mol", "--max_output_len", default=256, type=int, help="Max output length of the model")
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument("-mnt", "--max_num_tokens", default=None, type=int, help="Max number of tokens")
    parser.add_argument("-ont", "--opt_num_tokens", default=None, type=int, help="Optimum number of tokens")
    parser.add_argument(
        "-mpet", "--max_prompt_embedding_table_size", default=None, type=int, help="Max prompt embedding table size"
    )
    parser.add_argument(
        "-npkc", "--no_paged_kv_cache", default=False, action='store_true', help="Enable paged kv cache."
    )
    parser.add_argument(
        "-drip",
        "--disable_remove_input_padding",
        default=False,
        action='store_true',
        help="Disables the remove input padding option.",
    )
    parser.add_argument(
        "-mbm",
        '--multi_block_mode',
        default=False,
        action='store_true',
        help='Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU.',
    )
    parser.add_argument(
        '--use_lora_plugin',
        nargs='?',
        const=None,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lora plugin which enables embedding sharing.",
    )
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help="Add lora in which modules. Only be activated when use_lora_plugin is enabled.",
    )
    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.',
    )
    parser.add_argument("-dm", "--debug_mode", default=False, action='store_true', help="Enable debug mode")

    args = parser.parse_args(argv)
    return args


def nemo_export_trt_llm(argv):
    args = get_args(argv)
    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if args.dtype != "bfloat16":
        LOGGER.error(
            "Only bf16 is currently supported for the optimized deployment with TensorRT-LLM. "
            "Support for the other precisions will be added in the coming releases."
        )
        return

    try:
        trt_llm_exporter = TensorRTLLM(model_dir=args.model_repository, load_model=False)

        LOGGER.info("Export to TensorRT-LLM function is called.")
        trt_llm_exporter.export(
            nemo_checkpoint_path=args.nemo_checkpoint,
            model_type=args.model_type,
            n_gpus=args.num_gpus,
            tensor_parallelism_size=args.tensor_parallelism_size,
            pipeline_parallelism_size=args.pipeline_parallelism_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_batch_size=args.max_batch_size,
            max_num_tokens=args.max_num_tokens,
            opt_num_tokens=args.opt_num_tokens,
            max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
            paged_kv_cache=(not args.no_paged_kv_cache),
            remove_input_padding=(not args.disable_remove_input_padding),
            dtype=args.dtype,
            enable_multi_block_mode=args.multi_block_mode,
            use_lora_plugin=args.use_lora_plugin,
            lora_target_modules=args.lora_target_modules,
            max_lora_rank=args.max_lora_rank,
        )

        LOGGER.info("Export is successful.")
    except Exception as error:
        LOGGER.error("Error message: " + str(error))
        raise error


if __name__ == '__main__':
    nemo_export_trt_llm(sys.argv[1:])
