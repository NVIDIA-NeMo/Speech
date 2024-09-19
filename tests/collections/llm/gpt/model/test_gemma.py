from nemo.collections.llm.fn.activation import openai_gelu
from nemo.collections.llm.gpt.model.gemma import (
    CodeGemmaConfig2B,
    CodeGemmaConfig7B,
    GemmaConfig,
    GemmaConfig2B,
    GemmaConfig7B,
)


def test_gemma_config():
    config = GemmaConfig(num_layers=18)
    assert config.normalization == "RMSNorm"
    assert config.activation_func == openai_gelu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.seq_length == 8192
    assert config.kv_channels == 256
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is True
    assert config.layernorm_zero_centered_gamma is True


def test_gemma_config_2b():
    config = GemmaConfig2B()
    assert config.num_layers == 18
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 8
    assert config.num_query_groups == 1
    assert config.ffn_hidden_size == 16384


def test_gemma_config_7b():
    config = GemmaConfig7B()
    assert config.num_layers == 28
    assert config.hidden_size == 3072
    assert config.num_attention_heads == 16
    assert config.num_query_groups == 16
    assert config.ffn_hidden_size == 24576


def test_code_gemma_config_2b():
    config = CodeGemmaConfig2B()
    assert config.num_layers == 18
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 8
    assert config.num_query_groups == 1
    assert config.ffn_hidden_size == 16384


def test_code_gemma_config_7b():
    config = CodeGemmaConfig7B()
    assert config.num_layers == 28
    assert config.hidden_size == 3072
    assert config.num_attention_heads == 16
    assert config.num_query_groups == 16
    assert config.ffn_hidden_size == 24576
