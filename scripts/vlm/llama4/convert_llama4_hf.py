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

from nemo.collections import llm, vlm
from nemo.collections.vlm.llama4.model.llama4_omni import Llama4ScoutExperts16Config

if __name__ == '__main__':
    # Specify the Hugging Face model ID
    hf_model_id = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    # Import the model and convert to NeMo 2.0 format
    llm.import_ckpt(
        model=vlm.Llama4OmniModel(Llama4ScoutExperts16Config()),  # Model configuration
        source=f"hf://{hf_model_id}",  # Hugging Face model source
    )
    # If you are only converting and use llm part of llama4 use below:
    # import_ckpt(
    #     model=llm.LlamaModel(llm.Llama4Experts16Config()),  # Model configuration
    #     source=f"hf://{hf_model_id}",  # Hugging Face model source
    # )
