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
import os
import tempfile
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig, OmegaConf
from torch.utils.checkpoint import checkpoint
from transformers import CLIPTextModel, CLIPTokenizer

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import CLIPModel
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel import Timestep
from nemo.collections.multimodal.modules.stable_diffusion.encoders.x_transformer import (
    TransformerWrapper,  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
)
from nemo.collections.multimodal.modules.stable_diffusion.encoders.x_transformer import Encoder
from nemo.collections.multimodal.parts.stable_diffusion.utils import (
    count_params,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    ParallelLinearAdapterConfig,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core import adapter_mixins
from nemo.utils import logging

try:
    from megatron.core import ModelParallelConfig, parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


class AbstractEncoder(nn.Module):
    def __init__(self, enable_lora_finetune=False, target_block=[], target_module=[]):
        super().__init__()
        self.TARGET_BLOCK = target_block
        self.TARGET_MODULE = target_module
        if enable_lora_finetune:
            self.lora_layers = []

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def _enable_lora(self, lora_model):
        for module_name, module in lora_model.named_modules():
            if module.__class__.__name__ in self.TARGET_BLOCK:
                tmp = {}
                for sub_name, sub_module in module.named_modules():
                    if sub_module.__class__.__name__ in self.TARGET_MODULE:
                        if hasattr(sub_module, "input_size") and hasattr(
                            sub_module, "output_size"
                        ):  # for megatron ParallelLinear
                            lora = LoraWrapper(sub_module, sub_module.input_size, sub_module.output_size)
                        else:  # for nn.Linear
                            lora = LoraWrapper(sub_module, sub_module.in_features, sub_module.out_features)
                        self.lora_layers.append(lora)
                        if sub_name not in tmp.keys():
                            tmp.update({sub_name: lora})
                        else:
                            print(f"Duplicate subnames are found in module {module_name}")
                for sub_name, lora_layer in tmp.items():
                    lora_name = f'{sub_name}_lora'
                    module.add_module(lora_name, lora_layer)


class AbstractEmbModel(nn.Module):
    def __init__(self, enable_lora_finetune=False, target_block=[], target_module=[]):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

        self.TARGET_BLOCK = target_block
        self.TARGET_MODULE = target_module
        if enable_lora_finetune:
            self.lora_layers = []

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def _enable_lora(self, lora_model):
        for module_name, module in lora_model.named_modules():
            if module.__class__.__name__ in self.TARGET_BLOCK:
                tmp = {}
                for sub_name, sub_module in module.named_modules():
                    if sub_module.__class__.__name__ in self.TARGET_MODULE:
                        if hasattr(sub_module, "input_size") and hasattr(
                            sub_module, "output_size"
                        ):  # for megatron ParallelLinear
                            lora = LoraWrapper(sub_module, sub_module.input_size, sub_module.output_size)
                        else:  # for nn.Linear
                            lora = LoraWrapper(sub_module, sub_module.in_features, sub_module.out_features)
                        self.lora_layers.append(lora)
                        if sub_name not in tmp.keys():
                            tmp.update({sub_name: lora})
                        else:
                            print(f"Duplicate subnames are found in module {module_name}")
                for sub_name, lora_layer in tmp.items():
                    lora_name = f'{sub_name}_lora'
                    module.add_module(lora_name, lora_layer)


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: List[ListConfig]):
        super().__init__()
        embedders = []

        for n, embconfig in enumerate(emb_models):
            embedder = embconfig['emb_model']
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)),
                            emb,
                        )
                        * emb
                    )
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size, max_seq_len=max_seq_len, attn_layers=Encoder(dim=n_embed, depth=n_layer)
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size=30522,
        max_seq_len=77,
        device="cuda",
        use_tokenizer=True,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self, n_stages=1, method='bilinear', multiplier=0.5, in_channels=3, out_channels=None, bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class LoraWrapper(nn.Module, adapter_mixins.AdapterModuleMixin):
    def __init__(self, target_module, in_features, out_features, lora_network_alpha=None):
        super().__init__()
        self.target_module = target_module
        self.set_accepted_adapter_types([ParallelLinearAdapterConfig._target_])
        self.lora_network_alpha = lora_network_alpha
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        org_results = self.target_forward(x)
        if self.is_adapter_available():
            lora_linear_adapter = self.get_adapter_module(AdapterName.PARALLEL_LINEAR_ADAPTER)
            lora_mixed_x = lora_linear_adapter(x)
            # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
            # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
            mixed_x = org_results[0] if isinstance(org_results, tuple) else org_results

            if self.lora_network_alpha:
                mixed_x = mixed_x + lora_mixed_x * (self.lora_network_alpha / lora_linear_adapter.dim)
            else:
                mixed_x = mixed_x + lora_mixed_x

            if isinstance(org_results, tuple):
                org_results = (mixed_x, *org_results[1:])
            else:
                org_results = mixed_x

        return org_results

    def add_adapter(self, name, cfg, **kwargs):
        self.lora_network_alpha = cfg.network_alpha
        kwargs = {}
        adapter_mixins.AdapterModuleMixin.add_adapter(self, name, cfg, **kwargs)
        self.target_forward = self.target_module.forward
        self.target_module.forward = self.forward
        del self.target_module


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        enable_lora_finetune=False,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):
        super().__init__(enable_lora_finetune, target_block=["CLIPAttention", "CLIPMLP"], target_module=["Linear"])
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()
        if enable_lora_finetune:
            self._enable_lora(self.transformer)
            print(f"CLIP transformer encoder add {len(self.lora_layers)} lora layers.")

        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=(self.layer == "hidden"))

        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]

        # Pad the seq length to multiple of 8
        seq_len = (z.shape[1] + 8 - 1) // 8 * 8
        z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        use_fp16=False,
        cache_dir=None,
    ):
        super().__init__()
        assert layer in self.LAYERS
        print(f"Downloading clip with", arch, version, cache_dir)
        self.device = device
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version, cache_dir=cache_dir,
        )
        del model.visual
        self.model = model

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        if isinstance(text, list) and isinstance(text[0], str):
            tokens = open_clip.tokenize(text)
        else:
            # tokenizer has been invoked before
            tokens = text
        z = self.encode_with_transformer(tokens.to(self.device, non_blocking=True))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenMegatronCLIPEmbedder(AbstractEmbModel):
    def __init__(
        self,
        restore_from_path,
        device="cuda",
        layer="last",
        freeze=True,
        cfg=None,
        always_return_pooled=False,
        enable_lora_finetune=False,
    ):
        super().__init__(
            enable_lora_finetune=enable_lora_finetune,
            target_block=["ParallelAttention", "ParallelMLP"],
            target_module=["ColumnParallelLinear", "RowParallelLinear"],
        )
        if restore_from_path is not None:
            cfg, state_dict = self.load_config_and_state_from_nemo(restore_from_path)
        elif cfg is not None:
            state_dict = None
        else:
            raise ValueError("Either restore_from_path or cfg should not be None")

        self.cfg = cfg
        self.build_tokenizer(cfg)
        self.load_model(cfg, state_dict)
        self.return_pooled = always_return_pooled

        self.device = device
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

        if enable_lora_finetune:
            self._enable_lora(self.model.language_model)
            print(f"Megatron CLIP encoder add {len(self.lora_layers)} lora layers.")

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def load_config_and_state_from_nemo(self, nemo_path):
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        save_restore_connector = NLPSaveRestoreConnector()
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                save_restore_connector._unpack_nemo_file(path2file=nemo_path, out_folder=tmpdir)

                # Change current working directory to
                os.chdir(tmpdir)
                config_yaml = os.path.join(tmpdir, save_restore_connector.model_config_yaml)
                cfg = OmegaConf.load(config_yaml)

                model_weights = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
                state_dict = save_restore_connector._load_state_dict_from_disk(
                    model_weights, map_location=map_location
                )
            finally:
                os.chdir(cwd)

        return cfg, state_dict

    def build_tokenizer(self, cfg):
        legacy = cfg.tokenizer.sentencepiece_legacy
        self.tokenizer = get_nmt_tokenizer(
            library=cfg.tokenizer.library,
            model_name=cfg.tokenizer.type,
            tokenizer_model=cfg.tokenizer.model,
            vocab_file=cfg.tokenizer.vocab_file,
            merges_file=cfg.tokenizer.merge_file,
            delimiter=cfg.tokenizer.get('delimiter', None),
            legacy=legacy,
        )

        _, self.text_transform = get_preprocess_fns(cfg, self.tokenizer, is_train=False,)
        self.max_length = cfg.text.get("max_position_embeddings")

    def load_model(self, cfg, state_dict):
        padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
        )
        model = CLIPModel(
            model_cfg=cfg,
            model_parallel_config=ModelParallelConfig(),
            padded_vocab_size=padded_vocab_size,
            pre_process=cfg.text.pre_process,
            post_process=cfg.text.post_process,
        )

        if state_dict is not None:
            clip_state_dict = {}
            for key, value in state_dict.items():
                key = key[6:]
                clip_state_dict[key] = value
            model.load_state_dict(clip_state_dict)

        del model.vision_encoder
        self.model = model.text_encoder

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        return after

    def forward(self, text):
        '''
        Get embeddings from input text
        '''
        texts = self.text_transform(text)
        z, z_pooled = self.encode_with_transformer(texts.to(self.device))
        # # Pad the seq length to multiple of 8
        seq_len = (z.shape[1] + 8 - 1) // 8 * 8
        z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
        if self.return_pooled:
            return z, z_pooled
        return z

    def encode_with_transformer(self, text):
        x = self.model.language_model.embedding.word_embeddings(text)
        x = x + self.model.language_model.embedding.position_embeddings
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = self.model.language_model.encoder.final_layernorm(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.return_pooled:
            pooled = self.pool(x, text)
            return x, pooled
        return x, None

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ (self.model.head.weight.T)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.language_model.encoder.layers):
            if i == len(self.model.language_model.encoder.layers) - self.layer_idx:
                break
            x = r(x, attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.projection_dim = 1280
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device("cpu"), pretrained=version,)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            z_layer = z[self.layer]
            # # Pad the seq length to multiple of 8
            seq_len = (z_layer.shape[1] + 8 - 1) // 8 * 8
            z_layer = torch.nn.functional.pad(z_layer, (0, 0, 0, seq_len - z_layer.shape[1]), value=0.0)
            return z_layer, z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim, device='cuda'):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim
        self.device = device

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        if self.device == 'cuda':
            return emb.to(torch.cuda.current_device())
        return emb


class PrecachedEmbModel(AbstractEmbModel):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def forward(self, *args):
        if self.device == 'cuda':
            return [arg.to(torch.cuda.current_device()) for arg in args]
        return list(args)


if __name__ == "__main__":
    from ldm.util import count_params

    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
