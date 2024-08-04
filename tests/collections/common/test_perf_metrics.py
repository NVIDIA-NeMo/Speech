import pytest
import yaml

from nemo.collections.common.metrics.perf_metrics import FLOPsMeasurementCallback

LLAMA2_CFG_STR = """
    run:
        name: train_llama2_7b_tp1_pp1_FP8_1node_15steps
    trainer:
        num_nodes: 1
        devices: 8
        precision: bf16
    exp_manager:
        explicit_log_dir: "results/logs"
    model:
        micro_batch_size: 1
        global_batch_size: 128
        encoder_seq_length: 4096
        max_position_embeddings: 4096
        num_layers: 32
        hidden_size: 4096
        ffn_hidden_size: 11008
        num_attention_heads: 32
"""

NEMOTRON_CFG_STR = """
    run:
      name: train_nemotron_8b_tp2_pp1_FP8_8node_20steps
    trainer:
      num_nodes: 8
      devices: 8
      precision: bf16
    exp_manager:
      explicit_log_dir: null
    model:
      micro_batch_size: 4
      global_batch_size: 256
      encoder_seq_length: 4096
      max_position_embeddings: 4096
      num_layers: 32
      hidden_size: 4096
      ffn_hidden_size: 16384
      num_attention_heads: 32
      fp8: true
"""

UNSUPPORTED_MODEL_CFG_STR = """
    run:
        name: unsupported_model
    trainer:
        num_nodes: 1
        devices: 8
        precision: bf64
    exp_manager:
        explicit_log_dir: null
    model:
        micro_batch_size: 1
        global_batch_size: 1
        encoder_seq_length: 1
        max_position_embeddings: 1
        num_layers: 1
        hidden_size: 1
        ffn_hidden_size: 1
        num_attention_heads: 1
"""

NULL_MODEL_CFG_STR = """
    run:
        name: null
"""


@pytest.fixture
def model_config(cfg):
    return yaml.safe_load(cfg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "cfg, model_name, train_step_time, expected_value",
    [
        (LLAMA2_CFG_STR, None, 8, 377.53),
        (LLAMA2_CFG_STR, "llama2", 8, 377.53),
        (LLAMA2_CFG_STR, None, [8, 8, 8, 8], 377.53),
        (NEMOTRON_CFG_STR, None, 1.31, 642.73),
        (
            UNSUPPORTED_MODEL_CFG_STR,
            None,
            1,  # model_name in config is unsupported
            "Failed to extract valid model name from or missing FLOPs calculations for unsupported_model",
        ),
        (
            UNSUPPORTED_MODEL_CFG_STR,
            "unknown_model",
            1,  # overrided model name is unsupported
            "Failed to extract valid model name from or missing FLOPs calculations for unknown_model",
        ),
        (
            NULL_MODEL_CFG_STR,
            None,
            1,  # both- config and overrided model name are None
            "Failed to extract valid model name from or missing FLOPs calculations for None",
        ),
    ],
)
def test_eval_tflops_per_sec_per_gpu(model_config, model_name, train_step_time, expected_value):
    if isinstance(expected_value, (int, float)):
        flops_callback = FLOPsMeasurementCallback(model_config, model_name=model_name)
        tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time)
        assert tflops_per_sec_per_gpu == pytest.approx(expected_value, rel=1e-4)

        if model_name is None:
            # extract valid model name with delimiter='-'
            model_config["run"]["name"] = model_config["run"]["name"].replace("_", ".")
            flops_callback = FLOPsMeasurementCallback(model_config, model_name=model_name)
            tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time)
            assert tflops_per_sec_per_gpu == pytest.approx(expected_value, rel=1e-4)

            # # extract valid model name from a string
            model_config["run"]["name"] = model_config["run"]["name"].replace("_", "")
            flops_callback = FLOPsMeasurementCallback(model_config, model_name=model_name)
            tflops_per_sec_per_gpu = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time)
            assert tflops_per_sec_per_gpu == pytest.approx(expected_value, rel=1e-4)

    if isinstance(expected_value, str):
        flops_callback = FLOPsMeasurementCallback(model_config, model_name=model_name)
        with pytest.raises(KeyError, match=expected_value):
            _ = flops_callback.eval_tflops_per_sec_per_gpu(train_step_time)
