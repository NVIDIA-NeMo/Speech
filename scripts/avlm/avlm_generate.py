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

"""
Example:
  python scripts/avlm/avlm_generate.py \
    --local_model_path ${MODEL_PATH} \
    --image_path ${IMAGE_PATH} \
    --audio_path ${AUDIO_PATH} \
    --top_p 0.9 \
    --temperature 1.0 \
    --top_k 40 \
    --tokens_to_generate 100
"""

import argparse

import torch
from megatron.core.transformer.enums import AttnBackend

import nemo.lightning as nl
from nemo.collections import avlm, llm, vlm
from nemo.collections.avlm.data.energon import AVLMEnergonQASample, AVLMSampleConfig
from nemo.collections.avlm.data.energon.avlm_sample_config import AVLMSample
from nemo.collections.avlm.data.energon.avlm_task_encoder import AVLMSampleEncoderQA
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.utils import logging


def nucleus_sampling(logits, top_p=0.9, temperature=1.0, top_k=None):
    """Nucleus (top-p) sampling with temperature and top-k support."""
    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering if specified
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Create mask for tokens to keep
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')

    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def generate(model, sample_encoder, sample, tokens_to_generate=20, top_p=0.9, temperature=1.0, top_k=None):
    # pylint: disable=C0115,C0116

    # Encode samples
    encoded_sample = AVLMSample()
    encoded_sample = sample_encoder.encode(sample, encoded_sample)
    encoded_sample.tokens = torch.tensor(encoded_sample.tokens).unsqueeze(0).cuda()
    # Remove the last unneccesary space token (e.g., id - "29871" for "llava-hf/llava-1.5-7b-hf") from encoded_sample.tokens
    encoded_sample.tokens = encoded_sample.tokens[:, :-1]
    encoded_sample.images = torch.tensor(encoded_sample.images).cuda()
    encoded_sample.audios = torch.stack(encoded_sample.audios).cuda()
    position_ids = (
        torch.arange(encoded_sample.tokens.size(1), dtype=torch.long, device=encoded_sample.tokens.device)
        .unsqueeze(0)
        .expand_as(encoded_sample.tokens)
    ).cuda()

    from itertools import chain, groupby

    def mark_ignore_spans(tokens, values_list):
        return list(
            chain.from_iterable(
                [f"{len(list(g))} x ({k})"] if k in values_list else list(g) for k, g in groupby(tokens)
            )
        )

    print(
        f"encoded_sample.tokens[0]: {mark_ignore_spans(encoded_sample.tokens[0].tolist(), values_list=[-100, -200, -300, 0])}"
    )

    # Generate
    input_ids = encoded_sample.tokens
    generated_ids = input_ids.clone()
    for _ in range(tokens_to_generate):
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                images=encoded_sample.images,
                num_image_tiles=encoded_sample.num_image_tiles,
                audios=encoded_sample.audios,
                audio_lengths=encoded_sample.audio_lengths,
            )
            # Use nucleus sampling with temperature and top-k
            next_token_ids = nucleus_sampling(output[:, -1], top_p=top_p, temperature=temperature, top_k=top_k)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            print(f"next_token_ids {next_token_ids}")

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == sample_encoder.tokenizer.eos_token_id:
                print(f"breaking")
                break
    generated_ids[generated_ids < 0] = 0
    generated_texts = sample_encoder.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")


def main(args) -> None:
    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )
    fabric = trainer.to_fabric()

    # set tokenizer
    tokenizer = AutoTokenizer("llava-hf/llava-1.5-7b-hf")

    # Configure sample encoder
    avlm_sample_config = AVLMSampleConfig(
        audio_encoder_config={  # whisper audio encoder
            "model_type": "whisper",
            "window_stride": 0.01,
            "sample_rate": 16000,
            "fixed_max_audio_length": 29.9999 * 16000,
            "encoder_down_sampling": 2,
            "num_mel_bins": None,
            "patch_size": None,
            "time_stride": None,
            "frequency_stride": None,
            "max_spectrogram_length": None,
        },
        image_encoder_config={
            "model_type": "vit",
            "img_width": 336,
            "img_height": 336,
            "patch_size": 14,
            "projection_downsample_factor": None,
        },
    )
    avlm_sample_config.conversation_template_config.system = ''
    sample_encoder = AVLMSampleEncoderQA(
        tokenizer=tokenizer,
        audio_processor=None,
        image_processor=None,
        multimodal_sample_config=avlm_sample_config,
    )

    # Configure AVLM model
    language_transformer_config = llm.Llama2Config7B(
        seq_length=8192,
        attention_backend=AttnBackend.fused,
        # manually set vocab size to 32768. Originally the size is 32000, but with TP=8, it is padded to 32768.
        make_vocab_size_divisible_by=32768,
    )
    language_model_from_pretrained = None
    # vision config
    vision_transformer_config = vlm.HFCLIPVisionConfig(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
    )
    vision_model_from_pretrained = None
    vision_projection_config = vlm.MultimodalProjectorConfig(
        projector_type="mlp2x_gelu",
        input_size=vision_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=language_transformer_config.hidden_size,
    )
    # audio config
    audio_transformer_config = ASRModuleConfig(
        _target_="nemo.collections.speechlm.modules.asr_module.ASRModuleConfig",
        use_hf_auto_model=True,
        hf_trust_remote_code=False,
        hf_load_pretrained_weights=True,
        pretrained_model="openai/whisper-large-v3",
        hidden_size=1280,
        target_module="model.encoder",
    )
    audio_model_from_pretrained = None
    audio_projection_config = vlm.MultimodalProjectorConfig(
        projector_type="mlp2x_gelu",
        input_size=audio_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=language_transformer_config.hidden_size,
    )
    # AVLM model configuration
    avlm_config = avlm.AVLMConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        audio_transformer_config=audio_transformer_config,
        audio_projection_config=audio_projection_config,
        language_model_from_pretrained=language_model_from_pretrained,
        vision_model_from_pretrained=vision_model_from_pretrained,
        audio_model_from_pretrained=audio_model_from_pretrained,
        freeze_language_model=True,
        freeze_vision_model=True,
        freeze_vision_projection=True,
        freeze_audio_model=True,
        freeze_audio_projection=True,
    )
    model = avlm.AVLMModel(avlm_config, tokenizer=sample_encoder.tokenizer)

    # Load model from local path
    print("Loading checkpoint from: ", args.local_model_path)
    model = fabric.load_model(args.local_model_path, model)

    # Setup model for inference
    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)

    # Load and process the image and audio
    with open(args.image_path, 'rb') as file:
        image_bytes = file.read()
    with open(args.audio_path, 'rb') as file:
        audio_bytes = file.read()
    images = [{"media_type": "image", "media_value": image_bytes}]
    audios = [{"media_type": "audio", "media_value": audio_bytes}]

    conversations = [{"from": "human", "value": "<image><audio>"}, {"from": "gpt", "value": ""}]
    sample = AVLMEnergonQASample(
        __key__="dummy",
        __restore_key__="dummy",
        __subflavor__="dummy",
        __subflavors__="dummy",
        context=[conversations[0]["value"]],
        answers=[conversations[1]["value"]],
        audios=audios,
        videos=None,
        images=images,
    )

    # Run generation
    generate(
        model,
        sample_encoder,
        sample,
        top_p=args.top_p,
        temperature=args.temperature,
        top_k=args.top_k,
        tokens_to_generate=args.tokens_to_generate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVLM Pretraining Script")

    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        # pylint: disable=line-too-long
        default=None,
        help="Path to the audio to use for inference.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        # pylint: disable=line-too-long
        default=None,
        help="Path to the audio to use for inference.",
    )
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--tokens_to_generate", type=int, default=20, help="Number of tokens to generate")

    args = parser.parse_args()
    main(args)
