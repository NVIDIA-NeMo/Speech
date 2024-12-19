# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import modelopt.torch.prune as mtp
import torch.multiprocessing as mp
from datasets import load_dataset
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils.model_utils import load_config

mp.set_start_method("spawn", force=True)

"""
Nemo pruning example script.

Please consult examples/nlp/language_modeling/conf/megatron_gpt_prune.yaml config on available pruning arguments,
models supported as well as how to set up data and inference for calibration (with defaults recommended).

Example usage to prune width automatically:
```
python examples/nlp/language_modeling/megatron_gpt_prune.py \
    model.restore_from_path=llama3.1-8b.nemo \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=8 \
    trainer.num_nodes=1 \
    trainer.precision=bf16 \
    trainer.devices=8 \
    prune.ffn_hidden_size=9216 \
    prune.num_attention_heads=null \
    prune.num_query_groups=null \
    prune.hidden_size=3072 \
    export.save_path=llama3.1-8b-width-pruned.nemo
```

Example usage to prune depth automatically using cosine-similarity based importance metric:
```
python examples/nlp/language_modeling/megatron_gpt_prune.py \
    model.restore_from_path=llama3.1-8b.nemo \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=8 \
    trainer.num_nodes=1 \
    trainer.precision=bf16 \
    trainer.devices=8 \
    prune.num_layers=16 \
    export.save_path=llama3.1-8b-depth-pruned.nemo
```

Example usage to prune width and depth automatically:
```
python examples/nlp/language_modeling/megatron_gpt_prune.py \
    model.restore_from_path=llama3.1-8b.nemo \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=8 \
    trainer.num_nodes=1 \
    trainer.precision=bf16 \
    trainer.devices=8 \
    prune.ffn_hidden_size=9216 \
    prune.num_attention_heads=null \
    prune.num_query_groups=null \
    prune.hidden_size=3072 \
    prune.num_layers=16 \
    export.save_path=llama3.1-8b-width-and-depth-pruned.nemo
```

NOTE: for above usages, `model.tensor_model_parallel_size` and `inference.batch_size` must be 1
because of the current prune API limitation

Example usage to prune depth by dropping specific model layers (1-indexed):
```
python examples/nlp/language_modeling/megatron_gpt_prune.py \
    model.restore_from_path=llama3.1-8b.nemo \
    model.tensor_model_parallel_size=8 \
    model.pipeline_model_parallel_size=1 \
    trainer.num_nodes=1 \
    trainer.precision=bf16 \
    trainer.devices=8 \
    'prune.drop_layers=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]' \
    export.save_path=llama3.1-8b-pruned.nemo
```
"""


def get_calib_data_iter(data="wikitext", batch_size=1, calib_size=1024, max_sequence_length=512):
    """Get a data iterator for calibration."""
    if data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    else:
        # Assume a local JSON dataset with a column named "text"
        dataset = load_dataset("json", data_files=data, split="train")
        text_column = "text"
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


@hydra_runner(config_path="conf", config_name="megatron_gpt_prune")
def main(cfg) -> None:
    """Prune a model using modelopt."""
    # Overwrite model config with the one from the model checkpoint and apply pruning modifications
    model_cfg = load_config(cfg.model.restore_from_path)
    model_cfg.update(cfg.model)
    model_cfg.name = "modelopt"  # Use modelopt transformer spec for pruning

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    model = MegatronGPTModel.restore_from(
        restore_path=cfg.model.restore_from_path, override_config_path=model_cfg, trainer=trainer
    )

    def forward_loop(model):
        data_iter = get_calib_data_iter(
            cfg.prune.calib_dataset,
            cfg.inference.batch_size,
            cfg.prune.num_calib_size,
            cfg.inference.max_context_length,
        )
        dataloader = [data for data in data_iter]

        # NOTE: Alternatively you can also use `model.forward_bwd_step(data_iter, forward_only=True)`
        # if your model is setup for training.
        model.set_inference_config(OmegaConf.to_container(cfg.inference))
        for i, batch in enumerate(tqdm(dataloader, desc="Calibrating")):
            model.predict_step(batch, i)

    export_config = {
        k: cfg.prune.get(k)
        for k in [
            "ffn_hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "hidden_size",
            "num_layers",
        ]
        if cfg.prune.get(k) is not None
    }

    drop_layers = OmegaConf.to_object(cfg.prune.drop_layers)  # convert to native python list
    if drop_layers:
        assert (
            not export_config
        ), f"Cannot specify `prune.drop_layers` with other prune constraints. Recieved: {cfg.prune}"
        mtp.plugins.megatron.drop_mcore_gpt_layers(model.model, layers_to_drop=drop_layers)
        setattr(model.cfg, "num_layers", model.model.config.num_layers)
    else:
        assert (
            cfg.model.tensor_model_parallel_size == 1
        ), "Pruning currently only supports tensor_model_parallel_size=1"

        mtp.prune(
            model,
            mode="mcore_gpt_minitron",
            constraints={"export_config": export_config},
            dummy_input=None,  # Not used
            config={"forward_loop": forward_loop},
        )

    model.save_to(cfg.export.save_path)
    print(f"Pruned model saved to {cfg.export.save_path}")


if __name__ == "__main__":
    main()
