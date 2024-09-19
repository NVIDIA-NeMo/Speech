import torch.nn.functional as F

from nemo.collections.llm.gpt.model.baichuan import Baichuan2Config, Baichuan2Config7B


def test_baichuan2_config():
    config = Baichuan2Config(num_layers=32, hidden_size=4096, num_attention_heads=32)
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.add_bias_linear is False
    assert config.seq_length == 4096
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-6
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False


def test_baichuan2_config_7b():
    config = Baichuan2Config7B()
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 32
    assert config.ffn_hidden_size == 11008
