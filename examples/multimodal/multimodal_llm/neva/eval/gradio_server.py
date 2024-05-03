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

import base64
import io

import gradio as gr
import PIL.Image
from omegaconf import OmegaConf

from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor

CFG_STRING = """
trainer:
  devices: 1
  num_nodes: 1
  accelerator: gpu
  logger: False # logger provided by exp_manager
  precision: bf16 # 16, 32, or bf16

inference:
  greedy: False # Whether or not to use sampling ; use greedy decoding otherwise
  top_k: 0  # The number of highest probability vocabulary tokens to keep for top-k-filtering.
  top_p: 0.9 # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
  temperature: 0.2 # sampling temperature
  add_BOS: False # add the bos token at the begining of the prompt
  tokens_to_generate: 256 # The minimum length of the sequence to be generated.
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

cfg = OmegaConf.create(CFG_STRING)
cfg.neva_model_file = "/path/to/llava-v1.5-7b.nemo"
model, image_processor, _ = create_neva_model_and_processor(cfg)


def predict(prompt, image_base64=None):
    input_data = {"prompt": prompt}
    if image_base64 is not None:
        image_data = base64.b64decode(image_base64)
        # image = PIL.Image.fromarray(image)
        image = PIL.Image.open(io.BytesIO(image_data))
        input_data["image"] = image_processor(image)

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

    # Generate model responses
    responses = model.generate(
        input_prompts=[input_data],  # Adjust based on your model's requirements
        length_params=length_params,  # Define these parameters as in your original code
        sampling_params=sampling_params,  # Define these parameters as in your original code
        inference_config=cfg,
    )

    return responses[0]["clean_response"]


iface = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(), gr.Textbox()],
    outputs="text",
    title="Multimodal Model Inference",
    description="Enter a prompt and optionally upload an image for model inference.",
)

if __name__ == "__main__":
    iface.launch(server_port=8890, share=False)
