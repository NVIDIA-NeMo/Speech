import nemo_run as run
import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.transformer.enums import AttnBackend

from nemo.collections import llm
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig

pretrain = llm.deepseek_v3.pretrain_recipe(
    dir="/checkpoints/deepseek_v3",
    name="deepseek_v3",
    num_nodes=1,
    num_gpus_per_node=8,
    use_mtp=False,
)

pretrain.resume.restore_config = run.Config(
    RestoreConfig,
    path="/aot/checkpoints/dsv3/nemo2-2l",
    load_model_state=True,
    load_optim_state=False,
    load_artifacts=True,
)

pretrain.model.config.num_layers = 2
pretrain.model.config.moe_layer_freq = [0, 1]
pretrain.model.config.num_moe_experts = 16
pretrain.trainer.strategy.num_layers_in_first_pipeline_stage = None
pretrain.trainer.strategy.num_layers_in_last_pipeline_stage = None


# IMPORTANT: Also set on trainer.strategy (this is what actually gets used)
pretrain.trainer.strategy.tensor_model_parallel_size = 2
pretrain.trainer.strategy.pipeline_model_parallel_size = 2
pretrain.trainer.strategy.expert_model_parallel_size = 2
pretrain.trainer.strategy.context_parallel_size = 2
pretrain.trainer.strategy.pipeline_dtype = torch.bfloat16


pretrain.model.config.recompute_granularity = None
pretrain.model.config.recompute_modules = None

pretrain.trainer.max_steps = 50
pretrain.trainer.val_check_interval = 50
pretrain.trainer.log_every_n_steps = 1  # Log at every step
pretrain.model.config.deterministic_mode = True
pretrain.model.config.cross_entropy_loss_fusion = False
pretrain.model.config.attention_backend = AttnBackend.auto

pretrain.data = run.Config(
    PreTrainingDataModule,
    seq_length=4096,
    global_batch_size=16,
    micro_batch_size=1,
    split='9999,8,2',
    paths=[
        '/lustre/fsw/coreai_dlalgo_llm/datasets/RedPajama2/kenlm_perp_head_gopher_linefilter_decompressed/bin_idx/nemo/head_01_text_document'
    ],
)

pretrain.optim.lr_scheduler = None
pretrain.model.config.make_vocab_size_divisible_by = 1280
pretrain.model.config.vocab_size = 129280


import os

os.environ["WANDB_API_KEY"] = "f37880d4fc7a812145caab826f6fd1bf2dbd169c"
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


pretrain.log.wandb = run.Config(
    WandbLogger, project="nemo2_mbridge_comparison", name="dsv3_2layers_nemo2_tp2pp2ep2cp2"
)
run.run(pretrain)
