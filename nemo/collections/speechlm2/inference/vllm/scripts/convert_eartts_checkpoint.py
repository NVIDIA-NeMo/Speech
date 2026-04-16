# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import os
import json
import argparse

import torch
from omegaconf import OmegaConf, DictConfig
from safetensors.torch import save_file, load_file
from transformers import AutoConfig
from nemo.utils import logging

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def convert(outdir, config, model_path):
    os.makedirs(outdir, exist_ok=True)

    # load config
    with open(config, "r") as f:
        full_config = json.load(f)
    config_dict = full_config["model"]["speech_generation"]
    cfg = DictConfig(config_dict)
    # config modification that is needed to run inference
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.data.add_audio_prompt_after_description = True
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.model.subword_mask_exactly_as_eartts = False
    cfg.model.context_hidden_mask_exactly_as_eartts = False
    cfg.model.tts_config.disable_eos_prediction = True
    cfg.model.inference_force_speech_silence_on_eos = True
    cfg.model.use_word_sep_tokenizer = False
    cfg.model.num_delay_speech_tokens = 0
    cfg.data.source_sample_rate = 22050
    cfg.data.target_sample_rate = 22050
    cfg.model.pretrained_model = None

    # Resolve tokenizer name from the STT config (same tokenizer is shared by TTS).
    _pretrained_tokenizer_name = (
        full_config.get("model", {}).get("stt", {}).get("model", {}).get("pretrained_llm", None)
    )
    if _pretrained_tokenizer_name is None:
        raise ValueError(
            "Cannot determine tokenizer: 'pretrained_llm' not found in "
            "config.json -> model -> stt -> model. Check the checkpoint."
        )

    model = DuplexEARTTS(OmegaConf.to_container(cfg, resolve=True)).eval()
    # get subword encoder vocabs and config
    subword_id_to_char_ids = model.tts_model.embed_subword.subword_id_to_char_ids
    char_vocab = model.tts_model.embed_subword.char_vocab
    # create weights for the embedding layers that convert subword ids to char ids
    vocab_size = len(subword_id_to_char_ids)
    max_char_len = max(len(char_ids) for char_ids in subword_id_to_char_ids.values())
    hidden_size = cfg.model.tts_config.backbone_config.hidden_size

    # load checkpoint (support both safetensors and pytorch formats)
    weights = load_file(model_path)
    # select tts model weights, strip off one nested layer
    weights = {k[len("tts_model."):]: v for k, v in weights.items() if "tts_model." in k}

    # duplicate weights for rvq embeddings and embed code
    rvq_embs_weight = weights["tts_model.rvq_embs"].clone()  # 31 x codebook_size x latent_size
    rvq_embs_weight_pad = torch.nn.functional.pad(rvq_embs_weight, [0, 0, 0, 1])  # 31 x (codebook_size + 1) x latent_size
    embed_code_weight = weights["tts_model.embed_code.weight"].clone()  # latent_size x hidden_size

    # ======================
    # embedding module weights
    bos_emb = weights["tts_model.bos_emb"]
    null_emb = weights["tts_model.null_emb"]
    embed_subwords_weight = torch.zeros(
        (vocab_size, max_char_len), dtype=bos_emb.dtype, device=bos_emb.device
    )
    embed_subwords_mask_weight = torch.zeros(
        (vocab_size, max_char_len), dtype=bos_emb.dtype, device=bos_emb.device
    )
    for subword_id_str, char_ids_lst in subword_id_to_char_ids.items():
        subword_id = int(subword_id_str)
        char_ids = torch.tensor(
            char_ids_lst, dtype=bos_emb.dtype, device=bos_emb.device
        )
        embed_subwords_weight[subword_id, : len(char_ids)] = char_ids
        embed_subwords_mask_weight[subword_id, : len(char_ids)] = 1

    # create weights for the embedding model that runs outside of the eartts
    embedding_module_weights = {}
    embedding_module_weights["bos_emb"] = bos_emb
    embedding_module_weights["null_emb"] = null_emb

    # embedding transformer has a lot of weights
    for key, weight in weights.items():
        if "tts_model.embed_subword" in key:
            key = key[len("tts_model.") :]
            # bos_eos_emb and subword_flag_emb are moved outside embed_subword
            if key.startswith("embed_subword.bos_eos_emb.") or key.startswith("embed_subword.subword_flag_emb."):
                key = key[len("embed_subword."):]
            embedding_module_weights[key] = weight
    for key, weight in weights.items():
        if "tts_model.gated_fusion_audio_text" in key:
            key = key[len("tts_model.") :]
            embedding_module_weights[key] = weight
    if "tts_model.audio_prompt_projection_W" in weights:
        embedding_module_weights["audio_prompt_projection_W"] = weights["tts_model.audio_prompt_projection_W"]
    embedding_module_weights["embed_subword.embed_subwords.weight"] = (
        embed_subwords_weight
    )
    embedding_module_weights["embed_subword.embed_subwords_mask.weight"] = (
        embed_subwords_mask_weight
    )
    for i in range(rvq_embs_weight_pad.shape[0]):
        embedding_module_weights[f"rvq_embs.{i}.weight"] = rvq_embs_weight_pad[i]
    embedding_module_weights["embed_code.weight"] = embed_code_weight
    embedding_module_weights = {
        f"total_emb.{k}": v for k, v in embedding_module_weights.items()
    }

    # ======================
    # gemma backbone weights
    backbone_module_weights = {k[len("tts_model."):]: v for k, v in weights.items() if k.startswith("tts_model.backbone.")}
    backbone_module_weights["backbone.embed_tokens.weight"] = torch.randn(1, hidden_size, dtype=bos_emb.dtype, device=bos_emb.device)

    # ======================
    # sampler weights
    used_keys = ["rvq_embs", "embed_code", "mog_head"]
    sampler_weights = {k[len("tts_model."):]: v for k, v in weights.items() if any(k.startswith(f"tts_model.{key}") for key in used_keys)}
    sampler_weights = {"sampler." + k: v for k, v in sampler_weights.items()}

    # combine embedding module and backbone module weights
    weights = {**embedding_module_weights, **backbone_module_weights, **sampler_weights}
    weights = {"model." + k: v for k, v in weights.items()}

    # save weights
    safetensors_path = os.path.join(outdir, "model.safetensors")
    save_file(weights, safetensors_path)
    logging.info("Saved weights for vllm model")
    weight_map = {name: "model.safetensors" for name in weights.keys()}
    index = {
        "metadata": {
            "total_size": sum(w.numel() * w.element_size() for w in weights.values())
        },
        "weight_map": weight_map,
    }
    index_path = os.path.join(outdir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logging.info("Saved model index")

    # save config.json
    flat_config = {"architectures": ["EarTTSForCausalLM"], "model_type": "eartts"}
    # not using vocab size of the backbone model
    flat_config["vocab_size"] = 1

    # Parse backbone config exactly as NeMo does to get all defaults from transformers
    backbone_type = cfg.model.tts_config.get("backbone_type", None)
    backbone_config_dict = OmegaConf.to_container(
        cfg.model.tts_config.backbone_config, resolve=True
    ) if cfg.model.tts_config.get("backbone_config") else {}
    
    # Create AutoConfig the same way NeMo does - this fills in all defaults
    parsed_backbone_config = AutoConfig.for_model(backbone_type, **backbone_config_dict)
    
    # Store the backbone type for vllm to use
    flat_config["backbone_type"] = backbone_type
    
    # Forward all backbone configs from the parsed AutoConfig (includes defaults)
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rope_local_base_freq",
        "sliding_window",
        "layer_types",
    ]:
        if hasattr(parsed_backbone_config, key):
            value = getattr(parsed_backbone_config, key)
            # convert to list if it's a tuple or other iterable (except str)
            if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                value = list(value)
            flat_config[key] = value
    # forward overall configs
    for key in ["latent_size", "codebook_size", "num_quantizers", "exponent"]:
        flat_config[key] = cfg.model.tts_config[key]
    # forward mog head configs
    for key in ["num_layers", "low_rank", "num_predictions", "min_log_std", "eps"]:
        flat_config[f"mog_{key}"] = cfg.model.tts_config.mog_head_config[key]

    # forward inference configs (with name mapping for vLLM model)
    # num_iter is hardcoded to 8 in native model's _get_generation_config
    flat_config["num_iter"] = 8
    flat_config["noise_scale"] = cfg.model.get("inference_noise_scale", 0.8)
    flat_config["top_p_or_k"] = cfg.model.get("inference_top_p_or_k", 0.8)
    flat_config["guidance_scale"] = cfg.model.get("inference_guidance_scale", 0.5)

    # configuration of the embedding module
    flat_config["emb_backbone_config"] = OmegaConf.to_container(
        cfg.model.tts_config.cas_config.backbone_config, resolve=True
    )
    flat_config["emb_backbone_type"] = cfg.model.tts_config.cas_config.backbone_type
    flat_config["emb_vocab_size"] = vocab_size
    flat_config["emb_char_vocab_size"] = len(char_vocab)
    flat_config["max_char_len"] = max_char_len

    # configuration of flag embeddings
    flat_config["pretrained_tokenizer_name"] = _pretrained_tokenizer_name
    flat_config["use_subword_flag_emb"] = cfg.model.tts_config.use_subword_flag_emb
    flat_config["use_bos_eos_emb"] = cfg.model.tts_config.use_bos_eos_emb
    flat_config["use_gated_fusion_for_text_audio"] = cfg.model.tts_config.use_gated_fusion_for_text_audio
    flat_config["use_audio_prompt_frozen_projection"] = cfg.model.tts_config.use_audio_prompt_frozen_projection
    # hardcode enabling guidance so emb is created and application
    # of cfg is captured into a cuda graph
    flat_config["enable_guidance"] = True

    # configuring custom inputs/outputs
    flat_config["custom_input_specs"] = [
        {
            "name": "acoustic_tokens",
            "dim": flat_config["num_quantizers"],
            "dtype": "int32",
        },
        {"name": "text_tokens", "dtype": "int32"},
        {"name": "text_mask"},
        {"name": "bos_mask"},
        {"name": "speaker_latent", "dim": flat_config["hidden_size"]},
    ]
    flat_config["custom_outputs"] = ["acoustic_tokens"]

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(flat_config, f, indent=2)
    logging.info("Saved vllm config")

    # Extract and save pre-computed speaker latents (audio_prompt_latents.*)
    # from the NeMo checkpoint so they can be used at inference time.
    speaker_latents_dir = os.path.join(outdir, "speaker_latents")
    all_weights = load_file(model_path)
    found_latents = False
    for key, tensor in all_weights.items():
        if "audio_prompt_latents." in key:
            speaker_name = key.split("audio_prompt_latents.")[-1]
            os.makedirs(speaker_latents_dir, exist_ok=True)
            latent_path = os.path.join(speaker_latents_dir, f"{speaker_name}.pt")
            torch.save(tensor, latent_path)
            logging.info(f"Saved speaker latent '{speaker_name}' to {latent_path} (shape={tensor.shape})")
            found_latents = True
    if not found_latents:
        logging.warning(
            "No audio_prompt_latents found in checkpoint. "
            "speaker_name will not work unless latents are added."
        )


if __name__ == "__main__":
    args = parse_args()
    convert(args.outdir, args.config, args.model)
