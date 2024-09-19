from nemo.collections.llm.gpt.model.base import (
    GPTConfig5B,
    GPTConfig7B,
    GPTConfig20B,
    GPTConfig40B,
    GPTConfig126M,
    GPTConfig175B,
)


def test_gpt_config_126m():
    config = GPTConfig126M()
    assert config.seq_length == 2048
    assert config.num_layers == 12
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 12


def test_gpt_config_5b():
    config = GPTConfig5B()
    assert config.seq_length == 2048
    assert config.num_layers == 24
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 16384
    assert config.num_attention_heads == 32


def test_gpt_config_7b():
    config = GPTConfig7B()
    assert config.seq_length == 2048
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 10880
    assert config.num_attention_heads == 32


def test_gpt_config_20b():
    config = GPTConfig20B()
    assert config.seq_length == 2048
    assert config.num_layers == 44
    assert config.hidden_size == 6144
    assert config.ffn_hidden_size == 24576
    assert config.num_attention_heads == 48


def test_gpt_config_40b():
    config = GPTConfig40B()
    assert config.seq_length == 2048
    assert config.num_layers == 48
    assert config.hidden_size == 8192
    assert config.ffn_hidden_size == 32768
    assert config.num_attention_heads == 64


def test_gpt_config_175b():
    config = GPTConfig175B()
    assert config.seq_length == 2048
    assert config.num_layers == 96
    assert config.hidden_size == 12288
    assert config.ffn_hidden_size == 49152
    assert config.num_attention_heads == 96
