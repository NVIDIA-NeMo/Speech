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

from nemo.collections.llm.gpt.model.ssm import (
    BaseMambaConfig1_3B,
    BaseMambaConfig2_7B,
    BaseMambaConfig130M,
    BaseMambaConfig370M,
    BaseMambaConfig780M,
    NemotronHConfig4B,
    NemotronHConfig8B,
    NemotronHConfig47B,
    NemotronHConfig56B,
    NVIDIAMambaConfig8B,
    NVIDIAMambaHybridConfig8B,
    SSMConfig,
    gpt_data_step,
    ssm_forward_step,
)


def test_ssm_config():
    config = SSMConfig()
    assert config.fp16_lm_cross_entropy is False
    assert config.parallel_output is True
    assert config.share_embeddings_and_output_weights is False
    assert config.num_layers == 2
    assert config.num_attention_heads == 1
    assert config.hybrid_attention_ratio == 0.0
    assert config.hybrid_mlp_ratio == 0.0
    assert config.hybrid_override_pattern is None
    assert config.post_process is True
    assert config.pre_process is True
    assert config.seq_length == 8192
    assert config.position_embedding_type == "none"
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 10000
    assert config.seq_len_interpolation_factor is None
    assert config.apply_rope_fusion is True
    assert config.make_vocab_size_divisible_by == 128
    assert config.gated_linear_unit is False
    assert config.normalization == "RMSNorm"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.layernorm_epsilon == 1e-5
    assert config.get_attention_mask_from_fusion is False
    assert config.forward_step_fn == ssm_forward_step
    assert config.data_step_fn == gpt_data_step


def test_base_mamba_config_130m():
    config = BaseMambaConfig130M()
    assert config.hybrid_override_pattern == "M" * 24
    assert config.num_layers == 24
    assert config.seq_length == 2048
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 768
    assert config.make_vocab_size_divisible_by == 16
    assert config.tokenizer_library == 'huggingface'
    assert config.tokenizer_name == "EleutherAI/gpt-neox-20b"
    assert config.mapping_type == "base"


def test_base_mamba_config_370m():
    config = BaseMambaConfig370M()
    assert config.hybrid_override_pattern == "M" * 48
    assert config.num_layers == 48
    assert config.seq_length == 2048
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 1024
    assert config.make_vocab_size_divisible_by == 16
    assert config.tokenizer_library == 'huggingface'
    assert config.tokenizer_name == "EleutherAI/gpt-neox-20b"
    assert config.mapping_type == "base"


def test_base_mamba_config_780m():
    config = BaseMambaConfig780M()
    assert config.hybrid_override_pattern == "M" * 48
    assert config.num_layers == 48
    assert config.seq_length == 2048
    assert config.hidden_size == 1536
    assert config.ffn_hidden_size == 1536
    assert config.make_vocab_size_divisible_by == 16
    assert config.tokenizer_library == 'huggingface'
    assert config.tokenizer_name == "EleutherAI/gpt-neox-20b"
    assert config.mapping_type == "base"


def test_base_mamba_config_1_3b():
    config = BaseMambaConfig1_3B()
    assert config.hybrid_override_pattern == "M" * 48
    assert config.num_layers == 48
    assert config.seq_length == 2048
    assert config.hidden_size == 2048
    assert config.ffn_hidden_size == 2048
    assert config.make_vocab_size_divisible_by == 16
    assert config.tokenizer_library == 'huggingface'
    assert config.tokenizer_name == "EleutherAI/gpt-neox-20b"
    assert config.mapping_type == "base"


def test_base_mamba_config_2_7b():
    config = BaseMambaConfig2_7B()
    assert config.hybrid_override_pattern == "M" * 64
    assert config.num_layers == 64
    assert config.seq_length == 2048
    assert config.hidden_size == 2560
    assert config.ffn_hidden_size == 2560
    assert config.make_vocab_size_divisible_by == 16
    assert config.tokenizer_library == 'huggingface'
    assert config.tokenizer_name == "EleutherAI/gpt-neox-20b"
    assert config.mapping_type == "base"


def test_nvidia_mamba_config_8b():
    config = NVIDIAMambaConfig8B()
    assert config.hybrid_override_pattern == "M" * 56
    assert config.num_layers == 56
    assert config.seq_length == 4096
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 4096
    assert config.make_vocab_size_divisible_by == 128
    assert config.tokenizer_library == 'megatron'
    assert config.tokenizer_name == "GPTSentencePieceTokenizer"
    assert config.mapping_type == "nvidia-pure"


def test_nvidia_mamba_hybrid_config_8b():
    config = NVIDIAMambaHybridConfig8B()
    assert config.hybrid_override_pattern == "M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"
    assert config.num_layers == 56
    assert config.seq_length == 4096
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 16384
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.make_vocab_size_divisible_by == 128
    assert config.tokenizer_library == 'megatron'
    assert config.tokenizer_name == "GPTSentencePieceTokenizer"
    assert config.mapping_type == "nvidia-hybrid"


def test_nemotronh_config_4b():
    config = NemotronHConfig4B()
    assert config.hybrid_override_pattern == "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    assert config.num_layers == 52
    assert config.seq_length == 8192
    assert config.hidden_size == 3072
    assert config.mamba_num_heads == 112
    assert config.kv_channels == 128
    assert config.mamba_num_groups == 8
    assert config.mamba_state_dim == 128
    assert config.mamba_head_dim == 64
    assert config.ffn_hidden_size == 12288
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.make_vocab_size_divisible_by == 128
    assert config.use_mamba_mem_eff_path is False
    assert config.tokenizer_library == 'tiktoken'
    assert config.tokenizer_name == "TiktokenTokenizer"
    assert config.mapping_type == "nvidia-hybrid-nemotronh"
    assert config.masked_softmax_fusion is True
    assert config.apply_query_key_layer_scaling is False
    assert config.persist_layer_norm is True
    assert config.attention_softmax_in_fp32 is False
    assert config.vocab_size == 131072
    assert config.first_last_layers_bf16 is True
    assert config.is_hybrid_model is True


def test_nemotronh_config_8b():
    config = NemotronHConfig8B()
    assert config.hybrid_override_pattern == "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    assert config.num_layers == 52
    assert config.seq_length == 8192
    assert config.hidden_size == 4096
    assert config.mamba_num_groups == 8
    assert config.mamba_state_dim == 128
    assert config.mamba_head_dim == 64
    assert config.ffn_hidden_size == 21504
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.make_vocab_size_divisible_by == 128
    assert config.tokenizer_library == 'tiktoken'
    assert config.tokenizer_name == "TiktokenTokenizer"
    assert config.mapping_type == "nvidia-hybrid-nemotronh"
    assert config.masked_softmax_fusion is True
    assert config.apply_query_key_layer_scaling is False
    assert config.persist_layer_norm is True
    assert config.attention_softmax_in_fp32 is False
    assert config.vocab_size == 131072
    assert config.first_last_layers_bf16 is True
    assert config.is_hybrid_model is True


def test_nemotronh_config_47b():
    config = NemotronHConfig47B()
    assert config.hybrid_override_pattern == (
        "M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M-"
    )
    assert config.num_layers == 98
    assert config.seq_length == 8192
    assert config.hidden_size == 8192
    assert config.mamba_num_groups == 8
    assert config.mamba_state_dim == 256
    assert config.mamba_head_dim == 64
    assert config.ffn_hidden_size == 30720
    assert config.num_attention_heads == 64
    assert config.num_query_groups == 8
    assert config.make_vocab_size_divisible_by == 128
    assert config.tokenizer_library == 'tiktoken'
    assert config.tokenizer_name == "TiktokenTokenizer"
    assert config.mapping_type == "nvidia-hybrid-nemotronh"
    assert config.masked_softmax_fusion is True
    assert config.apply_query_key_layer_scaling is False
    assert config.persist_layer_norm is True
    assert config.attention_softmax_in_fp32 is False
    assert config.vocab_size == 131072
    assert config.first_last_layers_bf16 is True
    assert config.is_hybrid_model is True


def test_nemotronh_config_56b():
    config = NemotronHConfig56B()
    assert config.hybrid_override_pattern == (
        "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-"
        "M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    )
    assert config.num_layers == 118
    assert config.seq_length == 8192
    assert config.hidden_size == 8192
    assert config.mamba_num_groups == 8
    assert config.mamba_state_dim == 256
    assert config.mamba_head_dim == 64
    assert config.ffn_hidden_size == 32768
    assert config.num_attention_heads == 64
    assert config.num_query_groups == 8
    assert config.make_vocab_size_divisible_by == 128
    assert config.tokenizer_library == 'tiktoken'
    assert config.tokenizer_name == "TiktokenTokenizer"
    assert config.mapping_type == "nvidia-hybrid-nemotronh"
    assert config.masked_softmax_fusion is True
    assert config.apply_query_key_layer_scaling is False
    assert config.persist_layer_norm is True
    assert config.attention_softmax_in_fp32 is False
    assert config.vocab_size == 131072
    assert config.first_last_layers_bf16 is True
    assert config.is_hybrid_model is True
