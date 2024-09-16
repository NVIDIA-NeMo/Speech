import nemo_run as run
import torch

from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm import (
    GemmaConfig2B,
    GPTConfig126M,
    Llama3Config8B,
    MistralConfig7B,
    MixtralConfig8x3B,
    Nemotron4Config22B,
    PreTrainingDataModule,
)
from nemo.collections.llm.tools.auto_configurator import AutoConfigurator
from nemo.collections.llm.tools.auto_configurator.core.base_config import BaseConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def get_tokenizer() -> run.Config:
    return run.Config(AutoTokenizer, pretrained_model_name="GPT2BPETokenizer")


def get_data(seq_length, global_batch_size) -> run.Config[PreTrainingDataModule]:
    config = {
        "paths": "/",
        "seq_length": seq_length,
        "global_batch_size": global_batch_size,
        "num_workers": 2,
        "index_mapping_dir": None,
    }

    return run.Config(
        PreTrainingDataModule,
        **config,
        tokenizer=get_tokenizer(),
    )


def get_trainer(num_nodes) -> run.Config[nl.Trainer]:
    trainer_config = {
        "accelerator": "gpu",
        "enable_checkpointing": False,
        "use_distributed_sampler": False,
        "max_epochs": None,
        "log_every_n_steps": 1,
        "limit_val_batches": 1,
        "limit_test_batches": 1,
        "accumulate_grad_batches": 1,
        "num_nodes": num_nodes,
        "devices": 8,
        "max_steps": 50,
        "val_check_interval": 50,
    }

    strategy = run.Config(
        nl.MegatronStrategy,
        pipeline_dtype=torch.bfloat16,
    )

    return run.Config(
        nl.Trainer,
        **trainer_config,
        strategy=strategy,
        plugins=run.Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        callbacks=[run.Config(TimingCallback)],
    )


def get_optim() -> run.Config[OptimizerConfig]:
    optim_params = {
        "optimizer": "adam",
        "lr": 1e-4,
        "min_lr": 1e-5,
        "use_distributed_optimizer": True,
        "bf16": True,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "clip_grad": 1.0,
        "adam_eps": 1e-5,
    }

    optim_config = run.Config(
        OptimizerConfig,
        **optim_params,
    )

    sched = run.Config(
        CosineAnnealingScheduler,
        warmup_steps=10,
        constant_steps=0,
        min_lr=optim_config.min_lr,
    )

    return run.Config(
        MegatronOptimizerModule,
        config=optim_config,
        lr_scheduler=sched,
    )


def get_logger() -> run.Config[nl.NeMoLogger]:
    tb_logger = run.Config(TensorBoardLogger, save_dir="tb_logs")

    ckpt = run.Config(
        nl.ModelCheckpoint,
        monitor="reduced_train_loss",
        save_last=False,
        save_top_k=0,
    )

    return run.Config(
        nl.NeMoLogger,
        ckpt=ckpt,
        tensorboard=tb_logger,
        wandb=None,
        log_dir="/",
    )


class TestBaseConfigs:
    def test_gpt3_base_config(self):
        # GPT3 7B
        model_config = run.Config(GPTConfig126M)
        runner = AutoConfigurator(model=model_config, num_nodes=8, path_to_logs="/", data_paths="/")
        base_config = BaseConfig(runner)
        model_size = runner._get_model_size(model_config)
        model_type = runner._get_model_type(model_config)
        data_config = get_data(2048, 'auto')
        trainer_config = get_trainer(8)
        optim_config = get_optim()
        logger_config = get_logger()

        assert (
            base_config.model == model_config
        ), f"{model_config} is expected class object but got {base_config.model}"
        assert model_size == 0.126, f"0.126 is expected size for {model_config} but got {model_size}"
        assert model_type == "gpt3", f"gpt3 is expected model type for {model_config} but got {model_type}"
        assert (
            base_config.data == data_config
        ), f"f{data_config} is expected data config for {model_config} but got {base_config.data}"
        assert (
            base_config.trainer == trainer_config
        ), f"f{trainer_config} is expected trainer config for {model_config} but got {base_config.trainer}"
        assert (
            base_config.optim == optim_config
        ), f"f{optim_config} is expected trainer config for {model_config} but got {base_config.optim}"
        assert (
            base_config.log == logger_config
        ), f"f{logger_config} is expected trainer config for {model_config} but got {logger_config}"

    def test_llama_base_config(self):
        # Llama3 8B
        model_config = run.Config(Llama3Config8B)
        runner = AutoConfigurator(
            model=model_config,
            num_nodes=16,
            path_to_logs="/",
            data_paths="/",
            seq_length=8192,
            global_batch_size=2048,
        )
        base_config = BaseConfig(runner)
        model_size = runner._get_model_size(model_config)
        model_type = runner._get_model_type(model_config)
        data_config = get_data(8192, 2048)
        trainer_config = get_trainer(16)
        optim_config = get_optim()
        logger_config = get_logger()

        assert (
            base_config.model == model_config
        ), f"{model_config} is expected class object but got {base_config.model}"
        assert model_size == 8, f"8 is expected size for {model_config} but got {model_size}"
        assert model_type == "llama", f"llama is expected model type for {model_config} but got {model_type}"
        assert (
            base_config.data == data_config
        ), f"f{data_config} is expected data config for {model_config} but got {base_config.data}"
        assert (
            base_config.trainer == trainer_config
        ), f"f{trainer_config} is expected trainer config for {model_config} but got {base_config.trainer}"
        assert (
            base_config.optim == optim_config
        ), f"f{optim_config} is expected trainer config for {model_config} but got {base_config.optim}"
        assert (
            base_config.log == logger_config
        ), f"f{logger_config} is expected trainer config for {model_config} but got {logger_config}"

    def test_mistral_base_config(self):
        # Mistral 7B
        model_config = run.Config(MistralConfig7B)
        runner = AutoConfigurator(
            model=model_config,
            num_nodes=16,
            path_to_logs="/",
            data_paths="/",
            seq_length=32768,
            global_batch_size=2048,
        )
        base_config = BaseConfig(runner)
        model_size = runner._get_model_size(model_config)
        model_type = runner._get_model_type(model_config)
        data_config = get_data(32768, 2048)
        trainer_config = get_trainer(16)
        optim_config = get_optim()
        logger_config = get_logger()

        assert (
            base_config.model == model_config
        ), f"{model_config} is expected class object but got {base_config.model}"
        assert model_size == 7, f"7 is expected size for {model_config} but got {model_size}"
        assert model_type == "mistral", f"mistral is expected model type for {model_config} but got {model_type}"
        assert (
            base_config.data == data_config
        ), f"f{data_config} is expected data config for {model_config} but got {base_config.data}"
        assert (
            base_config.trainer == trainer_config
        ), f"f{trainer_config} is expected trainer config for {model_config} but got {base_config.trainer}"
        assert (
            base_config.optim == optim_config
        ), f"f{optim_config} is expected trainer config for {model_config} but got {base_config.optim}"
        assert (
            base_config.log == logger_config
        ), f"f{logger_config} is expected trainer config for {model_config} but got {logger_config}"

    def test_mixtral_base_config(self):
        # Mixtral 8x3B
        model_config = run.Config(MixtralConfig8x3B)
        runner = AutoConfigurator(
            model=model_config,
            num_nodes=16,
            path_to_logs="/",
            data_paths="/",
            seq_length=4096,
            global_batch_size=2048,
        )
        base_config = BaseConfig(runner)
        model_size = runner._get_model_size(model_config)
        model_type = runner._get_model_type(model_config)
        data_config = get_data(4096, 2048)
        trainer_config = get_trainer(16)
        optim_config = get_optim()
        logger_config = get_logger()

        assert (
            base_config.model == model_config
        ), f"{model_config} is expected class object but got {base_config.model}"
        assert model_size == 3, f"3 is expected size for {model_config} but got {model_size}"
        assert model_type == "mixtral", f"mixtral is expected model type for {model_config} but got {model_type}"
        assert (
            base_config.data == data_config
        ), f"f{data_config} is expected data config for {model_config} but got {base_config.data}"
        assert (
            base_config.trainer == trainer_config
        ), f"f{trainer_config} is expected trainer config for {model_config} but got {base_config.trainer}"
        assert (
            base_config.optim == optim_config
        ), f"f{optim_config} is expected trainer config for {model_config} but got {base_config.optim}"
        assert (
            base_config.log == logger_config
        ), f"f{logger_config} is expected trainer config for {model_config} but got {logger_config}"

    def test_gemma_base_config(self):
        # Gemma 2B
        model_config = run.Config(GemmaConfig2B)
        runner = AutoConfigurator(
            model=model_config,
            num_nodes=8,
            path_to_logs="/",
            data_paths="/",
            seq_length=4096,
            global_batch_size=1024,
        )
        base_config = BaseConfig(runner)
        model_size = runner._get_model_size(model_config)
        model_type = runner._get_model_type(model_config)
        data_config = get_data(4096, 1024)
        trainer_config = get_trainer(8)
        optim_config = get_optim()
        logger_config = get_logger()

        assert (
            base_config.model == model_config
        ), f"{model_config} is expected class object but got {base_config.model}"
        assert model_size == 2, f"2 is expected size for {model_config} but got {model_size}"
        assert model_type == "gemma", f"gemma is expected model type for {model_config} but got {model_type}"
        assert (
            base_config.data == data_config
        ), f"f{data_config} is expected data config for {model_config} but got {base_config.data}"
        assert (
            base_config.trainer == trainer_config
        ), f"f{trainer_config} is expected trainer config for {model_config} but got {base_config.trainer}"
        assert (
            base_config.optim == optim_config
        ), f"f{optim_config} is expected trainer config for {model_config} but got {base_config.optim}"
        assert (
            base_config.log == logger_config
        ), f"f{logger_config} is expected trainer config for {model_config} but got {logger_config}"

    def test_nemotron_base_config(self):
        # Nemotron 22B
        model_config = run.Config(Nemotron4Config22B)
        runner = AutoConfigurator(
            model=model_config,
            num_nodes=64,
            path_to_logs="/",
            data_paths="/",
            seq_length=4096,
            global_batch_size=2048,
        )
        base_config = BaseConfig(runner)
        model_size = runner._get_model_size(model_config)
        model_type = runner._get_model_type(model_config)
        data_config = get_data(4096, 2048)
        trainer_config = get_trainer(64)
        optim_config = get_optim()
        logger_config = get_logger()

        assert (
            base_config.model == model_config
        ), f"{model_config} is expected class object but got {base_config.model}"
        assert model_size == 22, f"22 is expected size for {model_config} but got {model_size}"
        assert model_type == "nemotron", f"nemotron is expected model type for {model_config} but got {model_type}"
        assert (
            base_config.data == data_config
        ), f"f{data_config} is expected data config for {model_config} but got {base_config.data}"
        assert (
            base_config.trainer == trainer_config
        ), f"f{trainer_config} is expected trainer config for {model_config} but got {base_config.trainer}"
        assert (
            base_config.optim == optim_config
        ), f"f{optim_config} is expected trainer config for {model_config} but got {base_config.optim}"
        assert (
            base_config.log == logger_config
        ), f"f{logger_config} is expected trainer config for {model_config} but got {logger_config}"
