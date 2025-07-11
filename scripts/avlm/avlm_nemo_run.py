# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import nemo_run as run

from nemo.collections import avlm


def configure_recipe(
    nodes: int = 1,
    gpus_per_node: int = 8,
    pretrain=False,
    language_model_from_pretrained=None,
    checkpoint_path=None,
    output_dir=None,
    freeze_modules=None,
):
    """Configure the recipe"""
    if pretrain:
        recipe = avlm.avlm_8b.pretrain_recipe(
            dir=output_dir,  # Path to store checkpoints
            name="avlm_pretrain",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            language_model_from_pretrained=language_model_from_pretrained,
            freeze_modules=freeze_modules,
        )
    else:
        recipe = avlm.avlm_8b.finetune_recipe(
            checkpoint_path=checkpoint_path,
            dir=output_dir,  # Path to store checkpoints
            name="avlm_finetune",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            freeze_modules=freeze_modules,
            # DEBUGGING
            peft_scheme="lora",
        )
    recipe.trainer.max_steps = 20
    recipe.trainer.val_check_interval = 20
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 8) -> run.LocalExecutor:
    # pylint: disable=C0115,C0116
    # Env vars for jobs are configured here
    env_vars = {}

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def run_pretraining(language_model_from_pretrained=None, checkpoint_path=None, output_dir=None, freeze_modules=None):
    # pylint: disable=C0115,C0116
    recipe = configure_recipe(
        pretrain=True,
        language_model_from_pretrained=language_model_from_pretrained,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        freeze_modules=freeze_modules,
    )
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


def run_finetuning(checkpoint_path=None, output_dir=None, freeze_modules=None):
    # pylint: disable=C0115,C0116
    recipe = configure_recipe(
        pretrain=False, checkpoint_path=checkpoint_path, output_dir=output_dir, freeze_modules=freeze_modules
    )
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script with two optional arguments.")
    parser.add_argument(
        "--training_mode",
        type=str,
        required=True,
        choices=["pretrain", "finetune"],
        help="Training mode - either 'pretrain' or 'finetune'",
    )
    parser.add_argument(
        "--language_model_from_pretrained",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained language model (optional).",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, required=False, help="Path to checkpoint (optional)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/checkpoints/avlm", help="Path to store checkpoints (optional)."
    )
    parser.add_argument("--unfreeze_language_model", action="store_true", help="Unfreeze language model (optional).")
    parser.add_argument("--unfreeze_vision_model", action="store_true", help="Unfreeze vision model (optional).")
    parser.add_argument("--unfreeze_audio_model", action="store_true", help="Unfreeze audio model (optional).")
    parser.add_argument(
        "--unfreeze_vision_projection", action="store_true", help="Unfreeze vision projection (optional)."
    )
    parser.add_argument(
        "--unfreeze_audio_projection", action="store_true", help="Unfreeze audio projection (optional)."
    )
    args = parser.parse_args()

    # run nemo_run
    freeze_modules = {
        "freeze_language_model": not args.unfreeze_language_model,
        "freeze_vision_model": not args.unfreeze_vision_model,
        "freeze_audio_model": not args.unfreeze_audio_model,
        "freeze_vision_projection": not args.unfreeze_vision_projection,
        "freeze_audio_projection": not args.unfreeze_audio_projection,
    }
    if args.training_mode == "pretrain":
        run_pretraining(
            language_model_from_pretrained=args.language_model_from_pretrained,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            freeze_modules=freeze_modules,
        )
    elif args.training_mode == "finetune":
        run_finetuning(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            freeze_modules=freeze_modules,
        )
