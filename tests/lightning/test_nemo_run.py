from functools import partial

import pytest

BASE_CHECKPOINT_DIR = "/nemo_run/checkpoints"


@pytest.mark.parametrize(
    "module, recipe, name",
    [
        ("llama3_8b", "pretrain_recipe", "llama3_8b_pretrain"),
        ("llama3_8b", "finetune_recipe", "llama3_8b_finetune"),
        ("llama3_8b_16k", "pretrain_recipe", "llama3_8b_16k_pretrain"),
        ("llama3_8b_64k", "pretrain_recipe", "llama3_8b_64k_pretrain"),
        ("llama3_70b", "pretrain_recipe", "llama3_70b_pretrain"),
        ("llama3_70b", "finetune_recipe", "llama3_70b_finetune"),
        ("llama3_70b_16k", "pretrain_recipe", "llama3_70b_16k_pretrain"),
        ("llama3_70b_64k", "pretrain_recipe", "llama3_70b_64k_pretrain"),
        ("llama31_405b", "pretrain_recipe", "llama31_405b_pretrain"),
        ("mistral", "pretrain_recipe", "mistral_pretrain"),
        ("mistral", "finetune_recipe", "mistral_finetune"),
        ("mixtral_8x7b", "pretrain_recipe", "mixtral_8x7b_pretrain"),
        ("mixtral_8x7b", "finetune_recipe", "mixtral_8x7b_finetune"),
        ("mixtral_8x7b_16k", "pretrain_recipe", "mixtral_8x7b_16k_pretrain"),
        ("mixtral_8x7b_64k", "pretrain_recipe", "mixtral_8x7b_64k_pretrain"),
        ("mixtral_8x22b", "pretrain_recipe", "mixtral_8x22b_pretrain"),
        ("mixtral_8x22b", "finetune_recipe", "mixtral_8x22b_finetune"),
        ("nemotron3_4b", "pretrain_recipe", "nemotron3_4b_pretrain"),
        ("nemotron3_8b", "pretrain_recipe", "nemotron3_8b_pretrain"),
        ("nemotron3_8b", "finetune_recipe", "nemotron3_8b_finetune"),
        ("nemotron4_15b", "pretrain_recipe", "nemotron4_15b_pretrain"),
        ("nemotron4_15b_16k", "pretrain_recipe", "nemotron4_15b_16k_pretrain"),
        ("nemotron4_15b_64k", "pretrain_recipe", "nemotron4_15b_64k_pretrain"),
        ("nemotron4_22b", "pretrain_recipe", "nemotron4_22b_pretrain"),
        ("nemotron4_22b_16k", "pretrain_recipe", "nemotron4_22b_16k_pretrain"),
        ("nemotron4_22b_64k", "pretrain_recipe", "nemotron4_22b_64k_pretrain"),
        ("nemotron4_340b", "pretrain_recipe", "nemotron4_340b_pretrain"),
        ("nemotron4_340b", "finetune_recipe", "nemotron4_340b_finetune"),
    ],
)
def test_recipes_with_nemo_run(module, recipe, name, tmpdir, monkeypatch):
    monkeypatch.setenv("NEMORUN_HOME", str(tmpdir))
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    import nemo_run as run

    from nemo.collections import llm
    from nemo.collections.llm.recipes.log.default import wandb_logger
    from nemo.lightning.run import plugins

    recipe_config = getattr(getattr(llm, module), recipe)(
        name=name, dir=BASE_CHECKPOINT_DIR, num_nodes=1, num_gpus_per_node=8
    )
    run_plugins = [
        plugins.PreemptionPlugin(),
        plugins.WandbPlugin(name=name, logger_fn=partial(wandb_logger, entity="dummy", project="dummy")),
    ]
    validation_plugin = plugins.ConfigValidationPlugin(validate_wandb=True)
    run_plugins.append(validation_plugin)

    with run.Experiment(f"{name}-unit-test") as exp:
        exp.add(
            recipe_config,
            executor=run.SlurmExecutor(
                account="dummy",
                partition="dummy",
                nodes=recipe_config.trainer.num_nodes,
                ntasks_per_node=recipe_config.trainer.devices,
            ),
            name=name,
            plugins=run_plugins,
        )
        exp.dryrun()

    with pytest.raises(AssertionError):
        with run.Experiment(f"{name}-unit-test-fail-validate-nodes-and-devices") as exp:
            exp.add(
                recipe_config,
                executor=run.SlurmExecutor(
                    account="dummy",
                    partition="dummy",
                    nodes=recipe_config.trainer.num_nodes + 1,
                    ntasks_per_node=recipe_config.trainer.devices + 1,
                ),
                name=name,
                plugins=run_plugins,
            )
            exp.dryrun()

    with pytest.raises(AssertionError):
        cfg = recipe_config.clone()
        cfg.log.log_dir = "/temporary-does-not-exist"
        with run.Experiment(f"{name}-unit-test-fail-validate-checkpoint-dir") as exp:
            exp.add(
                cfg,
                executor=run.SlurmExecutor(
                    account="dummy",
                    partition="dummy",
                    nodes=cfg.trainer.num_nodes,
                    ntasks_per_node=cfg.trainer.devices,
                ),
                name=name,
                plugins=run_plugins,
            )
            exp.dryrun()

    run_plugins = [plugins.NsysPlugin(start_step=3, end_step=4)] + run_plugins
    with run.Experiment(f"{name}-nsys-unit-test") as exp:
        exp.add(
            recipe_config,
            executor=run.SlurmExecutor(
                account="dummy",
                partition="dummy",
                nodes=recipe_config.trainer.num_nodes,
                ntasks_per_node=recipe_config.trainer.devices,
            ),
            name=name,
            plugins=run_plugins,
        )
        exp.dryrun()
