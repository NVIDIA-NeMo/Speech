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

r"""
Conversion script to convert Huggingface Mixtral checkpoints into NeMo checkpoint.
  Example to run this conversion script:
    python3 convert_mixtral_hf_to_nemo.py \
     --input_name_or_path <path_to_mixtral_checkpoints_folder> \
     --output_path <path_to_output_nemo_file> \
     --precision=bf16
"""

import json
import os
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import megatron.core.parallel_state as parallel_state
import torch
import torch.nn
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging

torch.set_grad_enabled(False)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface Mixtral checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    valid_precision_values = [16, '16', 'bf16', '16-mixed', 'bf16-mixed']
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=valid_precision_values, help="Model precision"
    )
    parser.add_argument('--low-ram', action='store_true')
    parser.add_argument('--tmp-dir', default='/tmp/mixtral_ckpt_parts/')
    args = parser.parse_args()
    return args


def restore_model_from_checkpoint(cls, checkpoint, strict, **kwargs):
    try:
        if 'cfg' in kwargs:
            model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
        else:
            model = cls(cfg=checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY], **kwargs)
            for name, module in model.named_parameters():
                if name in checkpoint['state_dict']:
                    # cast to target precision and
                    module.data = checkpoint['state_dict'][name].to(dtype=module.data.dtype)
                    checkpoint['state_dict'].pop(name)
                else:
                    print(f"Unexpected key: {name} not in checkpoint but in model.")

            for name, buffer in model.named_buffers():
                if name in checkpoint['state_dict']:
                    buffer.data = checkpoint['state_dict'][name]
                    checkpoint['state_dict'].pop(name)

            if len(checkpoint['state_dict'].keys()) != 0:
                raise RuntimeError(
                    f"Additional keys: {checkpoint['state_dict'].keys()} in checkpoint but not in model."
                )

            # register the artifacts
            cfg = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
            assert os.path.exists(
                cfg.tokenizer.model
            ), f"Expected cfg.tokenizer.model {cfg.tokenizer.model} to be present"
            if cfg.tokenizer.model is not None:
                model.register_artifact("tokenizer.tokenizer_model", cfg.tokenizer.model)
            if cfg.tokenizer.vocab_file is not None:
                model.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file)
            if cfg.tokenizer.merge_file is not None:
                model.register_artifact("tokenizer.merge_file", cfg.tokenizer.merge_file)
    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return model


def load_config(mixtral_config, tokenizer_path):
    nemo_config = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), '../../examples/nlp/language_modeling/conf/megatron_llama_config.yaml')
    ).model
    nemo_config.encoder_seq_length = mixtral_config['max_position_embeddings']
    nemo_config.num_layers = int(mixtral_config['num_hidden_layers'])
    nemo_config.hidden_size = mixtral_config['hidden_size']
    nemo_config.ffn_hidden_size = mixtral_config['intermediate_size']
    nemo_config.num_attention_heads = mixtral_config['num_attention_heads']
    nemo_config.max_position_embeddings = mixtral_config['max_position_embeddings']
    nemo_config.init_method_std = mixtral_config['initializer_range']
    # RMSNorm's epsilon.
    nemo_config.layernorm_epsilon = mixtral_config['rms_norm_eps']
    nemo_config.normalization = 'rmsnorm'
    nemo_config.micro_batch_size = 1
    nemo_config.global_batch_size = 1
    nemo_config.expert_model_parallel_size = 1

    if 'num_key_value_heads' in mixtral_config:
        nemo_config.num_query_groups = mixtral_config['num_key_value_heads']

    nemo_config.num_moe_experts = int(mixtral_config['num_local_experts'])
    assert nemo_config.num_moe_experts > 0, "num_experts must be greater than zero."
    nemo_config.moe_router_topk = int(mixtral_config['num_experts_per_tok'])
    assert nemo_config.moe_router_topk > 0, "moe_router_topk must be greater than zero."
    nemo_config.moe_router_pre_softmax = True
    nemo_config.use_cpu_initialization = True
    # Mixtral uses SiLU, but it is the same as swish with beta = 1.
    nemo_config.activation = 'fast-swiglu'

    nemo_config.tokenizer.model = tokenizer_path
    # TODO(@akoumparouli): rope_scaling.
    nemo_config['rotary_base'] = mixtral_config['rope_theta']

    base = 128
    while mixtral_config['vocab_size'] % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def load_hf_model_args(in_dir):
    params_file = os.path.join(in_dir, 'config.json')
    assert os.path.exists(params_file)
    with open(params_file, 'r') as fp:
        model_args = json.load(fp)
    return model_args


def load_mixtral_ckpt(in_dir, load_model=True):
    model_args = load_hf_model_args(in_dir)
    ckpt = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(in_dir, torch_dtype='auto')
        ckpt = model.state_dict()

    tokenizer = AutoTokenizer.from_pretrained(in_dir)
    assert tokenizer.vocab_size == model_args['vocab_size']
    return model_args, ckpt, tokenizer


def parse_precision(precision):
    if precision in ["32", "16"]:
        return int(float(precision))
    elif precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return precision
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            return precision[2:]  # prune bf in string
    else:
        return precision


def make_trainer(args, nemo_config):
    model_args, ckpt, tokenizer = load_mixtral_ckpt(args.input_name_or_path, load_model=False)
    nemo_config = load_config(model_args, tokenizer.vocab_file)

    precision = parse_precision(args.precision)
    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=nemo_config.get('native_amp_init_scale', 2**32),
                growth_interval=nemo_config.get('native_amp_growth_interval', 1000),
                hysteresis=nemo_config.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if nemo_config.get('megatron_amp_O2', False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if precision == 32:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32  # fallback

    nemo_config.precision = precision
    print(f"nemo_config: {nemo_config}")

    trainer = Trainer(plugins=plugins, accelerator='cpu', strategy=NLPDDPStrategy())
    return trainer, dtype


def convert(args):
    logging.info(f"loading checkpoint {args.input_name_or_path}")

    model_args, ckpt, tokenizer = load_mixtral_ckpt(args.input_name_or_path)
    nemo_config = load_config(model_args, tokenizer.vocab_file)

    hidden_size = nemo_config.hidden_size
    head_num = nemo_config.num_attention_heads
    head_size = hidden_size // head_num
    num_layers = nemo_config.num_layers

    mcore_gpt = nemo_config.mcore_gpt

    assert mcore_gpt == nemo_config.get(
        'transformer_engine', False
    ), "mcore_gpt transformer_engine must be enabled (or disabled) together."

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    embed_weight = ckpt[f'model.embed_tokens.weight']
    if mcore_gpt:
        embed_weights_base_name = f'model.embedding.word_embeddings.weight'
    else:
        embed_weights_base_name = f'model.language_model.embedding.word_embeddings.weight'
    checkpoint['state_dict'][embed_weights_base_name] = embed_weight

    if nemo_config.num_query_groups is None or nemo_config.num_query_groups == head_num:
        num_query_groups = head_num
    else:
        num_query_groups = nemo_config.num_query_groups
        assert head_num % num_query_groups == 0, 'head_num must be divisible by num_query_groups'
    if mcore_gpt:
        assert nemo_config.activation.startswith('fast-'), 'mcore only supports fast version of gated linear unit.'

    yield checkpoint
    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    for l in range(int(num_layers)):
        print(f"converting layer {l}")
        old_tensor_shape = ckpt[f'model.layers.{l}.self_attn.q_proj.weight'].size()
        new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
        new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

        q = ckpt[f'model.layers.{l}.self_attn.q_proj.weight'].view(*new_q_tensor_shape)
        k = ckpt[f'model.layers.{l}.self_attn.k_proj.weight'].view(*new_kv_tensor_shape)
        v = ckpt[f'model.layers.{l}.self_attn.v_proj.weight'].view(*new_kv_tensor_shape)

        heads_per_group = head_num // num_query_groups
        qkv_weights_l = []
        for i in range(num_query_groups):
            qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
            qkv_weights_l.append(k[i : i + 1, :, :])
            qkv_weights_l.append(v[i : i + 1, :, :])
        qkv_weights = torch.cat(qkv_weights_l)
        qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
        if mcore_gpt:
            qkv_weights_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'
        else:
            qkv_weights_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
        checkpoint['state_dict'][qkv_weights_base_name] = qkv_weights

        # attention dense
        o_weight = ckpt[f'model.layers.{l}.self_attn.o_proj.weight']
        if mcore_gpt:
            o_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
        else:
            o_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
        checkpoint['state_dict'][o_weight_base_name] = o_weight

        # # MLP
        # Handle gate
        moe_gate = ckpt[f'model.layers.{l}.block_sparse_moe.gate.weight']
        if mcore_gpt:
            moe_gate_name = f'model.decoder.layers.{l}.mlp.router.weight'
        else:
            raise Exception("not implemented")
        checkpoint['state_dict'][moe_gate_name] = moe_gate
        # Handle experts
        for i in range(nemo_config.num_moe_experts):
            gate_proj = ckpt[f'model.layers.{l}.block_sparse_moe.experts.{i}.w1.weight']
            up_proj = ckpt[f'model.layers.{l}.block_sparse_moe.experts.{i}.w3.weight']
            if mcore_gpt:
                mlp_down_base_name = f'model.decoder.layers.{l}.mlp.experts.local_experts.{i}.linear_fc1.weight'
            else:
                raise Exception("not implemented")
            mlp_down_weight = torch.cat((gate_proj, up_proj), axis=0)
            checkpoint['state_dict'][mlp_down_base_name] = mlp_down_weight

            mlp_up_weight = ckpt[f'model.layers.{l}.block_sparse_moe.experts.{i}.w2.weight']
            if mcore_gpt:
                mlp_up_base_name = f'model.decoder.layers.{l}.mlp.experts.local_experts.{i}.linear_fc2.weight'
            else:
                raise Exception("not implemented")
            checkpoint['state_dict'][mlp_up_base_name] = mlp_up_weight

        # LayerNorm
        input_ln_weight = ckpt[f'model.layers.{l}.input_layernorm.weight']

        if mcore_gpt:
            input_ln_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
        else:
            input_ln_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
        checkpoint['state_dict'][input_ln_base_name] = input_ln_weight

        post_attn_ln_weight = ckpt[f'model.layers.{l}.post_attention_layernorm.weight']
        if mcore_gpt:
            # @akoumparouli: switch to the following once TE supports MoE.
            # post_attn_ln_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
            post_attn_ln_base_name = f'model.decoder.layers.{l}.pre_mlp_layernorm.weight'
        else:
            post_attn_ln_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
        checkpoint['state_dict'][post_attn_ln_base_name] = post_attn_ln_weight

        print(f"done layer {l}")

        yield checkpoint
        checkpoint = OrderedDict()
        checkpoint['state_dict'] = OrderedDict()

    final_ln_weight = ckpt[f'model.norm.weight']
    if mcore_gpt:
        final_ln_base_name = f'model.decoder.final_layernorm.weight'
    else:
        final_ln_base_name = f'model.language_model.encoder.final_layernorm.weight'
    checkpoint['state_dict'][final_ln_base_name] = final_ln_weight

    output_layer_weight = ckpt[f'lm_head.weight']
    if mcore_gpt:
        output_layer_base_name = f'model.output_layer.weight'
    else:
        output_layer_base_name = f'model.language_model.output_layer.weight'
    checkpoint['state_dict'][output_layer_base_name] = output_layer_weight

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config
    yield checkpoint
    del ckpt


def merge(a: dict, b: dict, path=[]):
    is_dict = lambda x: isinstance(x, OrderedDict) or isinstance(x, dict)
    for key in b:
        if key in a:
            if is_dict(a[key]) and is_dict(b[key]):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception('Value conflict: ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def init_spm(spm_model_cls):
    from google.protobuf.json_format import Parse, ParseDict

    src = {
        "trainerSpec": {
            "modelPrefix": "tok_v0",
            "modelType": "BPE",
            "vocabSize": 32000,
            "selfTestSampleSize": 0,
            "inputFormat": "text",
            "characterCoverage": 0.99995,
            "inputSentenceSize": "200000000",
            "seedSentencepieceSize": 1000000,
            "shrinkingFactor": 0.75,
            "numThreads": 80,
            "numSubIterations": 2,
            "maxSentenceLength": 4192,
            "shuffleInputSentence": True,
            "maxSentencepieceLength": 16,
            "splitByUnicodeScript": True,
            "splitByWhitespace": True,
            "splitByNumber": True,
            "treatWhitespaceAsSuffix": False,
            "splitDigits": True,
            "allowWhitespaceOnlyPieces": True,
            "vocabularyOutputPieceScore": True,
            "hardVocabLimit": True,
            "useAllVocab": False,
            "byteFallback": True,
            "requiredChars": "",
            "unkId": 0,
            "bosId": 1,
            "eosId": 2,
            "padId": -1,
            "unkSurface": " \u2047 ",
            "unkPiece": "<unk>",
            "bosPiece": "<s>",
            "eosPiece": "</s>",
            "padPiece": "<pad>",
            "trainExtremelyLargeCorpus": False,
            "enableDifferentialPrivacy": False,
            "differentialPrivacyNoiseLevel": 0.0,
            "differentialPrivacyClippingThreshold": "0",
            "pretokenizationDelimiter": "",
        },
        "normalizerSpec": {
            "name": "identity",
            "precompiledCharsmap": "",
            "addDummyPrefix": True,
            "removeExtraWhitespaces": False,
            "normalizationRuleTsv": "",
        },
    }
    return ParseDict(src, spm_model_cls.ModelProto())


def make_sentencepiece_tokenizer(hf_tok):
    import sys

    sys.path.insert(0, 'sentencepiece/python/src/sentencepiece/')
    try:
        import sentencepiece_model_pb2 as spm_model_cls  # import model # sentencepiece_model as model
    except ImportError:
        # If this fails, download sentencepiece and extract it here.
        print(
            "Sentencepiece was not found; run `(cd scripts/checkpoint_converters; git clone https://github.com/google/sentencepiece.git)` & retry"
        )
        quit()

    vocab = list(hf_tok.vocab.items())
    vocab.sort(key=lambda x: x[1])

    m = init_spm(spm_model_cls)
    prefix = 0
    found_boundary = False
    for token, i in vocab:
        new_token = spm_model_cls.ModelProto().SentencePiece()
        # print(token, len(token), type(token), i)
        new_token.piece = token
        if token == '<unk>':
            if not found_boundary:
                prefix += 1
            new_token.type = 2
            new_token.score = 0
        elif token in ['<s>', '</s>']:
            if not found_boundary:
                prefix += 1
            new_token.type = 3
            new_token.score = 0
        elif len(token) == 6 and token.startswith('<0x') and token[-1] == '>':
            if not found_boundary:
                prefix += 1
            new_token.type = 6
            new_token.score = 0
        elif set(token) == set(["▁"]):
            if token == '▁▁':
                found_boundary = True
            new_token.score = -1e09
        else:
            new_token.score = -float(i) + prefix
        m.pieces.append(new_token)

    output_path = 'new.model'
    with open(output_path, 'wb') as fp:
        fp.write(m.SerializeToString())
    return output_path


def save_to_nemo(args, checkpoint):

    logging.info(f"loading checkpoint {args.input_name_or_path}")
    model_args, ckpt, tokenizer = load_mixtral_ckpt(args.input_name_or_path, load_model=False)
    if tokenizer.vocab_file is None:
        tokenizer.vocab_file = make_sentencepiece_tokenizer(tokenizer)
    nemo_config = load_config(model_args, tokenizer.vocab_file)
    nemo_config.precision = parse_precision(args.precision)
    nemo_config.megatron_amp_O2 = True
    trainer, dtype = make_trainer(args, nemo_config)

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config
    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY].use_cpu_initialization = True
    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY].perform_initialization = False

    if nemo_config.get('megatron_amp_O2', False):
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            checkpoint['state_dict'][key.replace('model.', 'model.module.', 1)] = checkpoint['state_dict'].pop(key)

    model = restore_model_from_checkpoint(MegatronGPTModel, checkpoint, strict=False, trainer=trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    # disable cpu init
    model.cfg.use_cpu_initialization = False
    model.cfg.perform_initialization = True

    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    if args.low_ram:
        os.makedirs(args.tmp_dir, exist_ok=True)

    parallel_state.set_expert_model_parallel_world_size(1)
    checkpoint = OrderedDict()
    for i, ckpt_part in enumerate(convert(args)):
        if args.low_ram:
            torch.save(ckpt_part, f'{args.tmp_dir}/nemo_ckpt_part_{i}.pth')
        else:
            checkpoint = merge(checkpoint, ckpt_part)

    if args.low_ram:
        print("Loading partial checkpoints")
        for path in map(str, Path(args.tmp_dir).rglob("*.pth")):
            print(f"Loading checkpoint: {path}")
            checkpoint = merge(checkpoint, torch.load(path, mmap=True))

    save_to_nemo(args, checkpoint)
