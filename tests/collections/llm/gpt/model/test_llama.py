import torch.nn.functional as F

from nemo.collections.llm.gpt.model.llama import (
    CodeLlamaConfig7B,
    CodeLlamaConfig13B,
    CodeLlamaConfig34B,
    CodeLlamaConfig70B,
    Llama2Config7B,
    Llama2Config13B,
    Llama2Config70B,
    Llama3Config,
    Llama3Config8B,
    Llama3Config70B,
    Llama31Config,
    Llama31Config8B,
    Llama31Config70B,
    Llama31Config405B,
    LlamaConfig,
)


def test_llama_config():
    config = LlamaConfig(num_attention_heads=32, num_layers=32, hidden_size=4096)
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.seq_length == 4096
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False


def test_llama3_config():
    config = Llama3Config(
        num_layers=80, hidden_size=1024, num_attention_heads=16, num_query_groups=4, ffn_hidden_size=4096
    )
    assert config.num_query_groups == 4
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.normalization == "RMSNorm"
    assert config.init_method_std == 0.01
    assert config.layernorm_epsilon == 1.0e-05
    assert config.add_bias_linear is False
    assert config.bias_activation_fusion is True
    assert config.masked_softmax_fusion is True
    assert config.persist_layer_norm is True
    assert config.bias_dropout_fusion is True
    assert config.apply_rope_fusion is True
    assert config.share_embeddings_and_output_weights is False
    assert config.position_embedding_type == "rope"
    assert config.rotary_percent == 1.0


# individual model config tests below...


def test_llama2_config_7b():
    config = Llama2Config7B()
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 32
    assert config.ffn_hidden_size == 11008
    assert config.normalization == "RMSNorm"


def test_llama2_config_13b():
    config = Llama2Config13B()
    assert config.num_layers == 40
    assert config.hidden_size == 5120
    assert config.num_attention_heads == 40
    assert config.num_query_groups == 40
    assert config.ffn_hidden_size == 13824


def test_llama2_config_70b():
    config = Llama2Config70B()
    assert config.num_layers == 80
    assert config.hidden_size == 8192
    assert config.num_attention_heads == 64
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 28672


def test_llama3_config_8b():
    config = Llama3Config8B()
    assert config.rotary_base == 500_000
    assert config.seq_length == 8192
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 14336
    assert config.num_attention_heads == 32


def test_llama3_config_70b():
    config = Llama3Config70B()
    assert config.rotary_base == 500_000
    assert config.seq_length == 8192
    assert config.num_layers == 80
    assert config.hidden_size == 8192
    assert config.ffn_hidden_size == 28672
    assert config.num_attention_heads == 64
    assert config.init_method_std == 0.008944
    assert config.make_vocab_size_divisible_by == 128


def test_llama31_config():
    config = Llama31Config(num_layers=32, num_attention_heads=32, hidden_size=4096)
    assert config.scale_factor == 8
    assert config.low_freq_factor == 1
    assert config.high_freq_factor == 4
    assert config.old_context_len == 8192
    assert config.init_method_std == 0.02


def test_llama31_config_8b():
    config = Llama31Config8B()
    assert config.rotary_base == 500_000
    assert config.seq_length == 131072
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 14336
    assert config.num_attention_heads == 32


def test_llama31_config_70b():
    config = Llama31Config70B()
    assert config.rotary_base == 500_000
    assert config.seq_length == 131072
    assert config.num_layers == 80
    assert config.hidden_size == 8192
    assert config.ffn_hidden_size == 28672
    assert config.num_attention_heads == 64
    assert config.make_vocab_size_divisible_by == 128


def test_llama31_config_405b():
    config = Llama31Config405B()
    assert config.rotary_base == 500_000
    assert config.seq_length == 131072
    assert config.num_layers == 126
    assert config.hidden_size == 16384
    assert config.ffn_hidden_size == 53248
    assert config.num_attention_heads == 128
    assert config.make_vocab_size_divisible_by == 128


def test_codellama_config_7b():
    config = CodeLlamaConfig7B()
    assert config.rotary_base == 1_000_000
    assert config.seq_length == 16384


def test_codellama_config_13b():
    config = CodeLlamaConfig13B()
    assert config.rotary_base == 1_000_000
    assert config.seq_length == 16384


def test_codellama_config_34b():
    config = CodeLlamaConfig34B()
    assert config.num_layers == 48
    assert config.hidden_size == 8192
    assert config.num_attention_heads == 64
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 22016
    assert config.rotary_base == 1_000_000
    assert config.seq_length == 16384


def test_codellama_config_70b():
    config = CodeLlamaConfig70B()
    assert config.seq_length == 4096
    assert config.num_layers == 80
    assert config.hidden_size == 8192
    assert config.ffn_hidden_size == 28672
    assert config.num_attention_heads == 64
