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


from nemo.lightning.io.artifact import DirOrStringArtifact, FileArtifact
from nemo.lightning.io.mixin import track_io

# Registers all required classes with track_io functionality
try:
    # Track HF tokenizers
    from transformers import AutoTokenizer as HfAutoTokenizer
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

    for cls in [HfAutoTokenizer, LlamaTokenizer, LlamaTokenizerFast]:
        track_io(
            cls,
            artifacts=[
                FileArtifact(attr_name, required=False)
                for attr_name in ['vocab_file', 'merges_file', 'tokenizer_file', 'name_or_path']
            ],
        )

    from nemo.collections.common.tokenizers import AutoTokenizer

    track_io(
        AutoTokenizer,
        artifacts=[
            FileArtifact("vocab_file", required=False),
            FileArtifact("merges_file", required=False),
            DirOrStringArtifact("pretrained_model_name", required=False),
        ],
    )
except ImportError:
    # HF tokenizers are not available, no need to track them
    pass


try:
    from nemo.collections.common.tokenizers import SentencePieceTokenizer

    track_io(SentencePieceTokenizer, artifacts=[FileArtifact("model_path")])
except ImportError:
    # SentencePieceTokenizer is not available, no need to track it
    pass
