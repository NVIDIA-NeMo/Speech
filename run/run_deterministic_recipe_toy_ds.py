import argparse
import os

import nemo_run as run
import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.transformer.enums import AttnBackend

from nemo.collections import llm
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run deterministic recipe for toy dataset")
    parser.add_argument("--data-paths", type=str, nargs="+", required=True, help="Paths to data")
    parser.add_argument("--pretrained-checkpoint", type=str, required=True, help="Path to pretrained checkpoint")

    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=2)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=2)
    parser.add_argument("--expert-model-parallel-size", type=int, default=2)
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)

    parser.add_argument("--wandb-api-key", type=str, required=True, help="Wandb API key")
    parser.add_argument("--wandb-project", type=str, default="nemo2_mbridge_comparison_new")
    parser.add_argument(
        "--wandb-exp-name", type=str, default="dsv3_2layers_mbridge_final_bs16_tp2pp2ep2cp2sp_100steps"
    )
    parser.add_argument("--wandb-entity", type=str, default="dsv3_comparision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # start from default deepseek v3 recipe
    pretrain = llm.deepseek_v3.pretrain_recipe(
        # dir="/checkpoints/deepseek_v3",
        name="deepseek_v3",
        num_nodes=1,
        num_gpus_per_node=8,
        use_mtp=False,
    )

    # Load pretrained checkpoint
    pretrain.resume.restore_config = run.Config(
        RestoreConfig,
        path=args.pretrained_checkpoint,
        load_model_state=True,
        load_optim_state=False,
        load_artifacts=True,
    )

    # Model configuration relating to the toy deepseek v3 model
    # here we use the 2-layer toy model with reduced hidden size and experts
    pretrain.model.config.num_layers = 2
    pretrain.model.config.moe_layer_freq = [0, 1]
    pretrain.model.config.num_moe_experts = 16
    pretrain.model.config.hidden_size = 7168 // 2
    pretrain.model.config.ffn_hidden_size = 18432 // 2
    pretrain.trainer.strategy.num_layers_in_first_pipeline_stage = None
    pretrain.trainer.strategy.num_layers_in_last_pipeline_stage = None

    # Model Parallelism configuration
    pretrain.trainer.strategy.tensor_model_parallel_size = args.tensor_model_parallel_size
    pretrain.trainer.strategy.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    pretrain.trainer.strategy.expert_model_parallel_size = args.expert_model_parallel_size
    pretrain.trainer.strategy.context_parallel_size = 1  # CP fixed to 1 since unfused attention does not apply CP
    pretrain.trainer.strategy.sequence_parallel = True
    pretrain.trainer.strategy.pipeline_dtype = torch.bfloat16

    pretrain.model.config.recompute_granularity = None
    pretrain.model.config.recompute_modules = None

    pretrain.trainer.max_steps = args.train_iters
    pretrain.trainer.val_check_interval = args.eval_interval
    pretrain.trainer.log_every_n_steps = 1  # Log at every step

    # Deterministic mode configuration
    pretrain.model.config.deterministic_mode = True
    pretrain.model.config.cross_entropy_loss_fusion = False
    pretrain.model.config.attention_backend = AttnBackend.unfused
    pretrain.model.config.gradient_accumulation_fusion = False  # Match MBridge - use param.grad path

    pretrain.data = run.Config(
        PreTrainingDataModule,
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        split='9999,8,2',  # Align with Mbridge split ratio
        paths=args.data_paths,
    )

    pretrain.optim.lr_scheduler = None

    # Match MBRIDGE's settings
    pretrain.optim.config.weight_decay = 0.1  # Original matching value
    pretrain.model.config.make_vocab_size_divisible_by = 1280
    pretrain.model.config.vocab_size = 129280  # Default vocab size does not match the deepseek v3 model

    # Deterministic mode environment variables
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    pretrain.log.wandb = run.Config(
        WandbLogger,
        project=args.wandb_project,
        name=args.wandb_exp_name,
        entity=args.wandb_entity,
    )

    run.run(pretrain)
