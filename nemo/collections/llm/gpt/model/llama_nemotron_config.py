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


# Config copy from https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/blob/main/config.json
LLAMA_33_NEMOTRON_SUPER_49B_HETEROGENEOUS_CONFIG = """
{
  "_name_or_path": "llama_nemotron_super",
  "architectures": [
    "DeciLMForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_decilm.DeciLMConfig",
    "AutoModelForCausalLM": "modeling_decilm.DeciLMForCausalLM"
  },
  "block_configs": [
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 3.28125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.3125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.5,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.5,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.5,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.5,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.5,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.5,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 8,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 5.25,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    }
  ],
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128008,
    128009
  ],
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": null,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "nemotron-nas",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": null,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.48.3",
  "use_cache": true,
  "vocab_size": 128256
}
"""

# Config copy from https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1/blob/main/config.json
LLAMA_31_NEMOTRON_ULTRA_253B_HETEROGENEOUS_CONFIG = """
{
  "_name_or_path": "llama_nemotron_ultra",
  "architectures": [
    "DeciLMForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_decilm.DeciLMConfig",
    "AutoModelForCausalLM": "modeling_decilm.DeciLMForCausalLM"
  },
  "block_configs": [
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.4875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 0.975,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.4625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.4625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.4625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.4625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.4625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.4625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 3.4125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 3.4125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 3.4125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.925,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.925,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 36.5625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 39.0,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 31.40625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 27.5625,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 1.95,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": null,
        "no_op": true,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": null,
        "no_op": true,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 3.4125,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 4.875,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    },
    {
      "attention": {
        "n_heads_in_group": 16,
        "no_op": false,
        "num_sink_tokens": null,
        "replace_with_linear": false,
        "sparsify": null,
        "unshifted_sink": false,
        "use_prefill_window_in_sink_attention": false,
        "window_length": null
      },
      "ffn": {
        "ffn_mult": 2.4375,
        "no_op": false,
        "replace_with_linear": false,
        "sparsify": null
      }
    }
  ],
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128008,
    128009
  ],
  "hidden_act": "silu",
  "hidden_size": 16384,
  "initializer_range": 0.02,
  "intermediate_size": null,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "nemotron-nas",
  "num_attention_heads": 128,
  "num_hidden_layers": 162,
  "num_key_value_heads": null,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 16.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.1",
  "use_cache": true,
  "vocab_size": 128256
}
"""
