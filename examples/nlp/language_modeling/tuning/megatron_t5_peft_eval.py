# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


import asyncio
import threading
from functools import partial

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf


from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

mp.set_start_method("spawn", force=True)
"""
This is the script to run inference with a PEFT model or an SFT Model.

If you want to evaluate an SFT .nemo file:

python examples/nlp/language_modeling/tuning/megatron_t5_peft_eval.py \
	model.restore_from_path=<path_to_sft_nemo_file> \
	model.peft.restore_from_path=null \
	trainer.devices=1 model.data.test_ds.file_names=\[<path_to_test_jsonl_file1>, <path_to_test_jsonl_file2>] \
	model.data.test_ds.names=\['name_for_test_file1', 'name_for_test_file2'] \  # this is not the filename just some identifier
	model.data.test_ds.global_batch_size=4 \  # or some other value
	model.data.test_ds.micro_batch_size=4 \
	model.data.test_ds.tokens_to_generate=30 \
	inference.greedy=True \
	inference.outfile_path=\'<path_to_jsonl_output_file>'  

If you want to evaluate a PEFT Model, you should provide a base T5 model and a PEFT model .nemo file

python examples/nlp/language_modeling/tuning/megatron_t5_peft_eval.py \
	model.restore_from_path=<path_to_sft_nemo_file> \
	model.peft.restore_from_path=<path_to_peft_nemo_file> \ # this will be created if you use `megatron_t5_peft_tuning.py`
	trainer.devices=1 model.data.test_ds.file_names=\[<path_to_test_jsonl_file1>, <path_to_test_jsonl_file2>] \
	model.data.test_ds.names=\['name_for_test_file1', 'name_for_test_file2'] \  # this is not the filename just some identifier
	model.data.test_ds.global_batch_size=4 \  # or some other value
	model.data.test_ds.micro_batch_size=4 \
	model.data.test_ds.tokens_to_generate=30 \
	inference.greedy=True \
	inference.outfile_path=\'<path_to_jsonl_output_file>'  

"""


def use_inference_server(cfg, model, trainer):
    if not HAVE_MEGATRON_CORE:
        raise ValueError('Megatron-core needs to be installed to use this feature!')

    from nemo.collections.nlp.modules.common.megatron_web_server import get_chatbot_demo, get_demo

    trainer.test(model, dataloaders=None)

    if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
        if cfg.web_server:
            if cfg.chat:
                defaults = {
                    'user': cfg.chatbot_config.user,
                    'assistant': cfg.chatbot_config.assistant,
                    'system': cfg.chatbot_config.system,
                }
                web_ui = partial(
                    get_chatbot_demo,
                    defaults=defaults,
                    value=cfg.chatbot_config.value,
                    attributes=cfg.chatbot_config.attributes,
                )
            else:
                web_ui = get_demo
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=web_ui, daemon=True, args=(cfg.share, cfg.username, cfg.password, cfg.port, cfg.web_port, loop),
            )
            thread.start()
        server = MegatronServer(model.cuda())
        server.run("0.0.0.0", port=cfg.port)

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            generate(model.cuda())


@hydra_runner(config_path="conf", config_name="megatron_t5_peft_eval_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()

    model_cfg = MegatronT5SFTModel.merge_inference_cfg(cfg.model.peft.restore_from_path, cfg)
    model = MegatronT5SFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)

    model.load_adapters(cfg.model.peft.restore_from_path)

    model.freeze()
    logging.info(f"Freezing parameters for PEFT eval:\n{model.summarize()}")

    if not cfg.model.get('use_flash_attention', False):
        cfg.inference.compute_attention_mask = True

    if not cfg.server:
        trainer.test(model)
    else:
        use_inference_server(cfg, model, trainer)


if __name__ == "__main__":
    main()
