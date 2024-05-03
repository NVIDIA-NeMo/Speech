# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import json
import math
import os

import shortuuid
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.utils.get_rank import is_global_rank_zero

CFG_STRING = """
trainer:
  devices: 1
  num_nodes: 1
  accelerator: gpu
  logger: False # logger provided by exp_manager
  precision: bf16 # 16, 32, or bf16

inference:
  greedy: True # Whether or not to use sampling ; use greedy decoding otherwise
  top_k: 0  # The number of highest probability vocabulary tokens to keep for top-k-filtering.
  top_p: 0.9 # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
  temperature: 0.2 # sampling temperature
  add_BOS: True # add the bos token at the begining of the prompt
  tokens_to_generate: 64 # The minimum length of the sequence to be generated.
  all_probs: False  # whether return the log prob for all the tokens in vocab
  repetition_penalty: 1.2  # The parameter for repetition penalty. 1.0 means no penalty.
  min_tokens_to_generate: 0  # The minimum length of the sequence to be generated.
  compute_logprob: False  # a flag used to compute logprob of all the input text, a very special case of running inference, default False
  end_strings: ["<extra_id_1>","<extra_id_7>",]  # generation will stop when one of these tokens is generated
  media_base_path: /pwd/images
  insert_media_token: null # `left` or `right` or `null`

cluster_type: BCP
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
pipeline_model_parallel_split_rank: 0 # used for encoder and decoder model (0 for others)

neva_model_file: /pwd/nemo_experiments/nemo_llava.nemo #neva_22b_tp8_finetuned_v1.nemo neva_8b_tp4_finetuned_v1.nemo
base_model_file: null
checkpoint_dir: null #/pwd/nemo_multimodal/nemo_experiments/nemo_llava_finetune/checkpoints # checkpoint file dir. This is used to load the PTL checkpoint generated during the Kosmos training
checkpoint_name: null #megatron_clip--val_loss=0.41-step=13499-consumed_samples=431904.0.ckpt # PTL checkpoint file name, only used for PTL checkpoint loading
hparams_file: null #/pwd/nemo_multimodal/nemo_experiments/nemo_llava_finetune/version_0/hparams.yaml # model configuration file, only used for PTL checkpoint loading
"""


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    cfg = OmegaConf.create(CFG_STRING)
    cfg.neva_model_file = args.model_path
    cfg.base_model_file = args.model_base
    cfg.inference.media_base_path = args.image_folder
    cfg.tensor_model_parallel_size = args.tp
    cfg.pipeline_model_parallel_size = args.pp
    cfg.trainer.devices = args.tp * args.pp

    model, image_processor, _ = create_neva_model_and_processor(cfg)
    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }
    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
        "end_strings": cfg.inference.end_strings,
    }

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if is_global_rank_zero():
        ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions, disable=(not is_global_rank_zero()))):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            cur_prompt = qs = '<image>' + cur_prompt
            line['image'] = image_processor(os.path.join(cfg.inference.media_base_path, line['image']))

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        responses = model.generate(
            input_prompts=[dict(prompt=qs, image=line.get('image', None))],
            length_params=length_params,
            sampling_params=sampling_params,
            inference_config=cfg,
        )
        if responses is None:
            continue
        outputs = responses[0]["clean_response"]

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs

            responses = model.generate(
                input_prompts=[prompt + outputs_reasoning + ' ###\nANSWER:'],
                length_params=length_params,
                sampling_params=sampling_params,
                inference_config=cfg,
            )
            outputs = responses[0]["clean_response"]
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        if is_global_rank_zero():
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": args.model_path,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()
    if is_global_rank_zero():
        ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
