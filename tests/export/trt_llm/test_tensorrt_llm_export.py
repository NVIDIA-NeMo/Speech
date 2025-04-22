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

import shutil

import pytest


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
@pytest.mark.parametrize("tensor_parallelism_size,pipeline_parallelism_size", [(2, 1), (1, 2)])
def test_nemo2_convert_to_safe_tensors(tensor_parallelism_size, pipeline_parallelism_size):
    """
    Test safe tensor exporter. This tests the whole nemo export until engine building.
    """
    from pathlib import Path

    from nemo.export.tensorrt_llm import TensorRTLLM

    trt_llm_exporter = TensorRTLLM(model_dir="/tmp/safe_tensor_test/")
    trt_llm_exporter.convert_to_safe_tensors(
        nemo_checkpoint_path="/home/TestData/llm/models/llama32_1b_nemo2",
        model_type="llama",
        delete_existing_files=True,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism_size=pipeline_parallelism_size,
        gpus_per_node=2,
        use_parallel_embedding=False,
        use_embedding_sharing=False,
        dtype="bfloat16",
    )

    assert Path("/tmp/safe_tensor_test/").exists(), "Safe tensors were not generated."
    assert Path("/tmp/safe_tensor_test/rank0.safetensors").exists(), "Safe tensors for rank0 were not generated."
    if pipeline_parallelism_size == 1 and tensor_parallelism_size == 2:
        assert Path("/tmp/safe_tensor_test/rank1.safetensors").exists(), "Safe tensors for rank1 were not generated."
    assert Path("/tmp/safe_tensor_test/config.json").exists(), "config.yaml was not generated."

    shutil.rmtree("/tmp/safe_tensor_test/")


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_nemo2_convert_to_export():
    """
    Test safe tensor exporter. This tests the whole nemo export until engine building.
    """
    from pathlib import Path

    from nemo.export.tensorrt_llm import TensorRTLLM

    trt_llm_exporter = TensorRTLLM(model_dir="/tmp/safe_tensor_test_2/")
    trt_llm_exporter.export(
        nemo_checkpoint_path="/home/TestData/llm/models/llama32_1b_nemo2",
        model_type="llama",
        delete_existing_files=True,
        tensor_parallelism_size=1,
        pipeline_parallelism_size=1,
        gpus_per_node=None,
        max_input_len=1024,
        max_output_len=256,
        max_batch_size=4,
        max_prompt_embedding_table_size=None,
        use_parallel_embedding=False,
        use_embedding_sharing=False,
        paged_kv_cache=True,
        remove_input_padding=True,
        paged_context_fmha=False,
        dtype=None,
        load_model=True,
        use_lora_plugin=None,
        lora_target_modules=None,
        max_lora_rank=64,
        max_num_tokens=None,
        opt_num_tokens=None,
        max_seq_len=512,
        multiple_profiles=False,
        gpt_attention_plugin="auto",
        gemm_plugin="auto",
        use_mcore_path=True,
        reduce_fusion=True,
        fp8_quantized=None,
        fp8_kvcache=None,
        gather_context_logits=True,
        gather_generation_logits=True,
        build_rank=None,
    )

    output = trt_llm_exporter.forward(
        input_texts=["Tell me the capitol of France "],
        max_output_len=16,
        top_k=1,
        top_p=0.0,
        temperature=0.1,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        task_ids=None,
        lora_uids=None,
        prompt_embeddings_table=None,
        prompt_embeddings_checkpoint_path=None,
        streaming=False,
        output_log_probs=False,
        output_context_logits=False,
        output_generation_logits=False,
    )

    print(output)

    assert Path("/tmp/safe_tensor_test_2/trtllm_engine/").exists(), "Safe tensors were not generated."
    assert Path(
        "/tmp/safe_tensor_test_2/trtllm_engine/rank0.engine"
    ).exists(), "Safe tensors for rank0 were not generated."
    assert Path("/tmp/safe_tensor_test_2/trtllm_engine/config.json").exists(), "config.yaml was not generated."

    shutil.rmtree("/tmp/safe_tensor_test_2/")
