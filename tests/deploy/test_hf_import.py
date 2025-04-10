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


import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo.deploy.nlp.hf_deployable import HuggingFaceLLMDeploy
from nemo.deploy.utils import broadcast_list


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_hf_generate():
    """Tests HF deployable class's generate function."""

    hf_deployable = HuggingFaceLLMDeploy(
        hf_model_id_path="/home/TestData/llm/models/llama3.2-1B-hf/",
        task="text-generation",
        trust_remote_code=True,
        device_map=None,
        tp_plan=None,
    )

    output = hf_deployable.generate(
        text_inputs=["What is the color of a banana? ", "Tell me a joke."],
        max_length=32,
        do_sample=True,
    )

    assert len(output) == 2, "Output should have to be a list."
    assert len(output[0]) > 0, "First list in the output should have more than 0 elements."
    assert len(output[1]) > 0, "Second list in the output should have more than 0 elements."


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
@pytest.mark.skip(reason="will be enabled later.")
def test_hf_multigpu_generate():
    """Tests HF deployable class's generate function with multiple GPUs."""

    mp.spawn(_run_generate, nprocs=2)


def _run_generate(rank):
    """Code to run generate in each rank."""

    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if rank == 0:
        os.environ['RANK'] = str(rank)
        dist.init_process_group("nccl", rank=rank, world_size=2)
        _hf_generate_ranks()
        dist.destroy_process_group()
    else:
        os.environ['RANK'] = str(rank)
        dist.init_process_group("nccl", rank=rank, world_size=2)
        _hf_generate_ranks()
        dist.destroy_process_group()


def _hf_generate_ranks():
    """Generate by Ranks"""

    torch.cuda.set_device(dist.get_rank())

    hf_deployable = HuggingFaceLLMDeploy(
        hf_model_id_path="/home/TestData/llm/models/llama3.2-1B-hf/",
        task="text-generation",
        trust_remote_code=True,
        device_map=None,
        tp_plan=None,
    )

    if dist.get_rank() == 0:
        temperature = 1.0
        top_k = 1
        top_p = 0.0
        num_tokens_to_generate = 32
        output_logits = False
        output_scores = False

        prompts = ["What is the color of a banana? ", "Tell me a joke."]

        dist.broadcast(torch.tensor([0], dtype=torch.long, device="cuda"), src=0)
        broadcast_list(prompts, src=0)
        broadcast_list(
            data=[
                temperature,
                top_k,
                top_p,
                num_tokens_to_generate,
                output_logits,
                output_scores,
            ],
            src=0,
        )

        output = hf_deployable.generate(
            text_inputs=prompts,
            max_length=num_tokens_to_generate,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            output_logits=output_logits,
            output_scores=output_scores,
        )
        dist.broadcast(torch.tensor([1], dtype=torch.long, device="cuda"), src=0)
    else:
        hf_deployable.generate_other_ranks()

    dist.barrier()

    if dist.get_rank() == 0:
        assert len(output) == 2, "Output should have to be a lists."
        assert len(output[0]) > 0, "First list in the output should have more than 0 elements."
        assert len(output[1]) > 0, "Second list in the output should have more than 0 elements."
