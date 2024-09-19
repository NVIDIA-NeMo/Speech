import torch.nn.functional as F

from nemo.collections.llm.gpt.model.starcoder import StarcoderConfig, StarcoderConfig15B


def test_starcoder_config():
    config = StarcoderConfig(num_layers=40, num_attention_heads=48, hidden_size=6144)
    assert config.normalization == "LayerNorm"
    assert config.activation_func == F.gelu
    assert config.add_bias_linear is True
    assert config.seq_length == 8192
    assert config.position_embedding_type == "learned_absolute"
    assert config.hidden_dropout == 0.2
    assert config.attention_dropout == 0.2
    assert config.init_method_std == 0.01
    assert config.layernorm_epsilon == 1e-5
    assert config.share_embeddings_and_output_weights is False
    assert config.kv_channels is 6144 // 48
    assert config.num_query_groups == 1
    assert config.attention_softmax_in_fp32 is True
    assert config.bias_activation_fusion is True
    assert config.bias_dropout_fusion is True


def test_starcoder_config_15b():
    config = StarcoderConfig15B()
    assert config.num_layers == 40
    assert config.hidden_size == 6144
    assert config.ffn_hidden_size == 24576
    assert config.num_attention_heads == 48
    assert config.init_method_std == 0.02
