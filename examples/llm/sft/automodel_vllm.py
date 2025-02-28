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


from nemo.deploy import DeployPyTriton
from nemo.deploy.nlp import NemoQueryLLM

try:
    from nemo.export.vllm_hf_exporter import vLLMHFExporter
except Exception:
    raise Exception(
        "vLLM should be installed in the environment or import "
        "the vLLM environment in the NeMo FW container using "
        "source /opt/venv/bin/activate command"
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="Local path or model name on Hugging Face")
    parser.add_argument('--triton-model-name', required=True, type=str, help="Name for the service")
    args = parser.parse_args()

    exporter = vLLMHFExporter()
    exporter.export(model=args.model)

    nm = DeployPyTriton(
        model=exporter,
        triton_model_name=args.triton_model_name,
        triton_model_version=1,
        max_batch_size=64,
        http_port=8000,
        address="0.0.0.0",
    )

    nm.deploy()
    nm.run()

    nq = NemoQueryLLM(url="localhost:8000", model_name=args.triton_model_name)
    output_deployed = nq.query_llm(
        prompts=["How are you doing?"],
        max_output_len=128,
        top_k=1,
        top_p=0.2,
        temperature=1.0,
    )

    print("------------- Output: ", output_deployed)
    nm.stop()
