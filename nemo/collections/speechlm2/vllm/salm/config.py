# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Configuration for NeMo Speech LM (SALM) models in vLLM.

Provides ``NeMoSpeechLMConfig``, a HuggingFace-compatible config class
that wraps the LLM backbone's text config with NeMo-specific fields
(perception, audio_locator_tag, etc.).  The checkpoint's ``config.json``
determines which LLM backbone and encoder are used; hybrid (Mamba+MoE)
vs standard transformer backends are auto-detected from the backbone's
own ``architectures`` field.
"""

from transformers import AutoConfig, PretrainedConfig

_HYBRID_ARCHITECTURES = frozenset(
    {
        "NemotronHForCausalLM",
        "NemotronHybridForCausalLM",
    }
)

# The audio locator tag this plugin supports. Hardcoded because vLLM's
# class-level ``get_placeholder_str`` interface (used during chat-template
# prompt assembly) cannot read per-checkpoint config. ``audio_locator_tag``
# from ``config.json`` is validated against this constant at load time so
# any incompatible checkpoint fails fast with a clear error instead of
# silently rendering the wrong placeholder at request time.
_AUDIO_PLACEHOLDER = "<|audio|>"

# Historical serving-time headroom above the backbone vocabulary. vLLM builds
# the target and draft embedding tables at this padded size, and the weight
# loader zero-pads the smaller training tensors to match. ``prepare_for_vllm``
# validates the tokenizer's audio-token ID against this bound during export;
# the default export flow treats a validation failure as non-fatal and leaves
# an HF-only checkpoint.
_SPEECHLM_EMBED_EXTRA_ROWS = 10

_MTP_BLOCK_TYPE_TO_VLLM_SYMBOL = {
    "attention": "*",
    "moe": "E",
}


def _is_hybrid_backend(architectures: list[str]) -> bool:
    return bool(set(architectures) & _HYBRID_ARCHITECTURES)


def _mtp_pattern_from_backbone_config(text_config) -> str | None:
    """Return the vLLM Nemotron-H MTP pattern encoded by a backbone config.

    Nemotron 3.5 exports the physical MTP topology in one of two forms:
    ``mtp_hybrid_override_pattern`` (symbol string) or
    ``mtp_layers_block_type`` (list of block names). vLLM 0.23 supports only
    attention (``*``) and MoE (``E``) MTP sublayers, so unsupported topology
    must fail before model construction rather than load the wrong draft head.
    """
    pattern = getattr(text_config, "mtp_hybrid_override_pattern", None)
    if pattern:
        unsupported = sorted(set(pattern) - set(_MTP_BLOCK_TYPE_TO_VLLM_SYMBOL.values()))
        if unsupported:
            raise ValueError(
                f"vLLM Nemotron-H MTP does not support pattern symbols {unsupported!r} "
                f"in mtp_hybrid_override_pattern={pattern!r}; supported symbols are '*' and 'E'."
            )
        return pattern

    block_types = getattr(text_config, "mtp_layers_block_type", None)
    if not block_types:
        return None
    try:
        return "".join(_MTP_BLOCK_TYPE_TO_VLLM_SYMBOL[block_type] for block_type in block_types)
    except KeyError as error:
        raise ValueError(
            f"vLLM Nemotron-H MTP does not support block type {error.args[0]!r} in "
            f"mtp_layers_block_type={list(block_types)!r}; supported block types are "
            f"{sorted(_MTP_BLOCK_TYPE_TO_VLLM_SYMBOL)!r}."
        ) from error


def _resolve_speechlm_mtp_config(
    *,
    mtp: dict | None,
    compute_mtp: bool,
    text_config,
    num_nextn_predict_layers: int | None = None,
    use_repeated_layer: bool | None = None,
) -> dict | None:
    """Normalize the SpeechLM MTP contract consumed by the vLLM plugin.

    New exports carry an explicit root ``mtp`` dictionary. Older SpeechLM
    exports, including the first Nemotron 3.5 Lightning checkpoints, only
    carry ``compute_mtp`` at the root and keep MTP topology in the saved
    backbone config. Derive the missing dictionary for those checkpoints so
    they do not need to be re-exported.
    """
    explicit_mtp = dict(mtp) if isinstance(mtp, dict) else None
    if explicit_mtp is not None:
        enabled = bool(explicit_mtp.get("enabled", True))
    else:
        enabled = bool(compute_mtp)

    if not enabled:
        return None

    if num_nextn_predict_layers is None:
        if explicit_mtp is not None and "num_nextn_predict_layers" in explicit_mtp:
            num_nextn_predict_layers = explicit_mtp["num_nextn_predict_layers"]
        else:
            num_nextn_predict_layers = getattr(text_config, "num_nextn_predict_layers", 0)
    num_nextn_predict_layers = int(num_nextn_predict_layers or 0)
    if num_nextn_predict_layers <= 0:
        raise ValueError(
            "SpeechLM MTP is enabled but num_nextn_predict_layers is not positive in either "
            "the root mtp config or the backbone config."
        )

    explicit_pattern = explicit_mtp.get("hybrid_override_pattern") if explicit_mtp is not None else None
    backbone_pattern = _mtp_pattern_from_backbone_config(text_config)
    if explicit_pattern and backbone_pattern and explicit_pattern != backbone_pattern:
        raise ValueError(
            f"Root mtp.hybrid_override_pattern={explicit_pattern!r} disagrees with "
            f"backbone MTP topology {backbone_pattern!r}."
        )
    pattern = explicit_pattern or backbone_pattern
    if not pattern:
        raise ValueError(
            "SpeechLM MTP is enabled but neither mtp.hybrid_override_pattern nor the backbone's "
            "mtp_hybrid_override_pattern/mtp_layers_block_type declares the physical MTP topology."
        )
    # Validate explicit patterns as well as backbone-derived patterns.
    unsupported = sorted(set(pattern) - set(_MTP_BLOCK_TYPE_TO_VLLM_SYMBOL.values()))
    if unsupported:
        raise ValueError(
            f"vLLM Nemotron-H MTP does not support pattern symbols {unsupported!r} "
            f"in hybrid_override_pattern={pattern!r}; supported symbols are '*' and 'E'."
        )

    if use_repeated_layer is None:
        use_repeated_layer = bool(explicit_mtp.get("use_repeated_layer", False)) if explicit_mtp else False

    return {
        "enabled": True,
        "num_nextn_predict_layers": num_nextn_predict_layers,
        "use_repeated_layer": bool(use_repeated_layer),
        "hybrid_override_pattern": pattern,
    }


class NeMoSpeechLMConfig(PretrainedConfig):
    """HuggingFace config for NeMo Speech LM multimodal models.

    Wraps a pretrained LLM config (e.g. NemotronH, Qwen3) with
    additional fields for the speech perception module.  Hybrid vs
    standard transformer is auto-detected from ``pretrained_llm``.

    A single ``NeMoSpeechLMForConditionalGeneration`` model class handles
    both backbone families via composition (see ``backends.py``). vLLM's
    runtime ``model_config.is_hybrid`` property gates the hybrid KV cache
    path on ``text_config.layer_types``: for non-hybrid backbones we
    populate it with ``["attention"] * num_hidden_layers`` so vLLM treats
    the model as attention-only at runtime even though the model class
    declares ``IsHybrid`` (needed for the NemotronH path).
    """

    model_type = "nemo_speechlm"

    def __init__(
        self,
        perception: dict | None = None,
        pretrained_llm: str | None = None,
        llm_config: dict | None = None,
        pretrained_asr: str | None = None,
        audio_locator_tag: str | None = None,
        prompt_format: str | None = None,
        pretrained_weights: bool | None = None,
        lora: dict | None = None,
        encoder_chunk_size_seconds: float | None = None,
        pe_encoder_path: str | None = None,
        pe_encoder_config: dict | None = None,
        pe_encoder_overrides: dict | None = None,
        speaker_encoder: dict | None = None,
        **kwargs,
    ):
        required_fields = {
            "pretrained_llm": pretrained_llm,
            "pretrained_asr": pretrained_asr,
            "audio_locator_tag": audio_locator_tag,
            "prompt_format": prompt_format,
            "pretrained_weights": pretrained_weights,
        }
        is_default_init = (
            perception is None
            and lora is None
            and encoder_chunk_size_seconds is None
            and pe_encoder_path is None
            and pe_encoder_config is None
            and pe_encoder_overrides is None
            and speaker_encoder is None
            and llm_config is None
            and not kwargs
            and all(value is None for value in required_fields.values())
        )

        # Newer Transformers validates token ids in PretrainedConfig.__init__
        # and may call get_text_config() before this subclass finishes
        # initialization. Seed an inert text_config for that early base-class
        # path; real checkpoint loads replace it below after field validation.
        self.text_config = PretrainedConfig()
        self.is_hybrid = False
        self._pending_image_token_index = None

        super().__init__(**kwargs)

        if is_default_init:
            # HuggingFace may instantiate config classes with no arguments when
            # building a default config for serialization/comparison. Keep that
            # path inert; real checkpoint loads continue through validation below.
            self.perception = {}
            self.pretrained_llm = None
            self.llm_config = None
            self.pretrained_asr = None
            self.audio_locator_tag = None
            self.prompt_format = None
            self.pretrained_weights = None
            self.lora = None
            self.encoder_chunk_size_seconds = None
            self.pe_encoder_path = None
            self.pe_encoder_config = None
            self.pe_encoder_overrides = None
            self.speaker_encoder = None
            self.__dict__.pop("_pending_image_token_index", None)
            return

        for name, value in required_fields.items():
            if value is None or value == "":
                raise ValueError(f"NeMo SpeechLM config must declare {name}.")
        # The plugin's runtime path uses the hardcoded ``_AUDIO_PLACEHOLDER``
        # constant everywhere (vLLM's class-level ``get_placeholder_str`` can't
        # read per-checkpoint config). Reject mismatched checkpoints at load
        # time rather than silently rendering with the wrong token at request.
        if audio_locator_tag != _AUDIO_PLACEHOLDER:
            raise ValueError(
                f"vLLM SpeechLM plugin currently supports only "
                f"audio_locator_tag={_AUDIO_PLACEHOLDER!r}, but checkpoint "
                f"config declares {audio_locator_tag!r}. To serve checkpoints "
                f"with a different audio token, both _AUDIO_PLACEHOLDER and "
                f"the model class's get_placeholder_str (vLLM-mandated "
                f"class-level metadata) need to be updated together."
            )
        has_pe_encoder = pe_encoder_path not in (
            None,
            "",
            False,
        ) or pe_encoder_config not in (
            None,
            {},
            "",
            False,
        )
        if has_pe_encoder and speaker_encoder not in (None, {}, "", False):
            raise ValueError("ParallelExpertEncoder and speaker_encoder are mutually exclusive.")
        if pe_encoder_overrides not in (None, {}) and pe_encoder_path in (
            None,
            "",
            False,
        ):
            raise ValueError("pe_encoder_overrides requires pe_encoder_path.")
        self.perception = perception or {}
        self.pretrained_llm = pretrained_llm
        self.llm_config = llm_config
        self.pretrained_asr = pretrained_asr
        self.audio_locator_tag = audio_locator_tag
        self.prompt_format = prompt_format
        self.pretrained_weights = pretrained_weights
        self.lora = lora
        self.encoder_chunk_size_seconds = encoder_chunk_size_seconds

        if llm_config is None:
            self.text_config = AutoConfig.from_pretrained(pretrained_llm, trust_remote_code=True)
        else:
            if not isinstance(llm_config, dict):
                raise ValueError(f"NeMo SpeechLM llm_config must be a dict, got {type(llm_config).__name__}.")
            embedded_config = dict(llm_config)
            model_type = embedded_config.pop("model_type", None)
            if not model_type:
                raise ValueError("NeMo SpeechLM llm_config must declare model_type.")
            self.text_config = AutoConfig.for_model(model_type, **embedded_config)
        self.pe_encoder_path = pe_encoder_path
        self.pe_encoder_config = pe_encoder_config
        self.pe_encoder_overrides = pe_encoder_overrides
        self.speaker_encoder = speaker_encoder

        # Backward compatibility for early Nemotron 3.5 SpeechLM exports:
        # they carry ``compute_mtp`` at the root and the MTP topology only in
        # llm_backbone/config.json. Normalize that into the explicit contract
        # used by the vLLM speculative-config hook.
        self.mtp = _resolve_speechlm_mtp_config(
            mtp=self.__dict__.get("mtp"),
            compute_mtp=bool(self.__dict__.get("compute_mtp", False)),
            text_config=self.text_config,
        )

        raw_archs = getattr(self.text_config, "architectures", [])
        if len(raw_archs) != 1:
            raise ValueError(
                f"Expected exactly one architecture in the backbone config, "
                f"got {raw_archs!r}. NeMo SpeechLM checkpoints must target a "
                f"single backbone; a mixed list makes the hybrid-vs-standard "
                f"routing ambiguous."
            )
        self.is_hybrid = _is_hybrid_backend(raw_archs)

        if self.is_hybrid:
            # Normalize to vLLM's official NemotronH architecture name.
            self.text_config.architectures = ["NemotronHForCausalLM"]
            if not hasattr(self.text_config, "total_num_kv_heads") or self.text_config.total_num_kv_heads is None:
                if (
                    not hasattr(self.text_config, "num_key_value_heads")
                    or self.text_config.num_key_value_heads is None
                ):
                    raise ValueError("NemotronH config must define num_key_value_heads.")
                self.text_config.total_num_kv_heads = self.text_config.num_key_value_heads
            if not hasattr(self.text_config, "rms_norm_eps"):
                if not hasattr(self.text_config, "layer_norm_epsilon"):
                    raise ValueError("NemotronH config must define layer_norm_epsilon.")
                self.text_config.rms_norm_eps = self.text_config.layer_norm_epsilon
        else:
            # All-attention ``layer_types`` makes vLLM's runtime
            # ``ModelConfig.is_hybrid`` property return False for transformer
            # backbones.
            num_layers = getattr(self.text_config, "num_hidden_layers", 0) or 0
            if num_layers > 0:
                self.text_config.layer_types = ["attention"] * num_layers

        self.text_config.vocab_size += _SPEECHLM_EMBED_EXTRA_ROWS
        pending_image_token_index = self.__dict__.pop("_pending_image_token_index", None)
        if pending_image_token_index is not None and pending_image_token_index != self.image_token_index:
            raise ValueError(
                f"image_token_index={pending_image_token_index!r} does not match the backbone vocabulary "
                f"boundary {self.image_token_index}. Remove this legacy serialized field; SpeechLM derives "
                f"the vLLM compatibility value at runtime."
            )

    @property
    def llm_architectures(self) -> list[str]:
        """Return the LLM backbone architectures list."""
        return getattr(self.text_config, "architectures", None) or []

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.text_config

    @property
    def image_token_index(self) -> int | None:
        """Return the vocabulary-boundary value expected by vLLM's MTP proposer.

        vLLM calls this compatibility field ``image_token_index`` even for an
        audio multimodal target. Actual audio locations come from vLLM's
        placeholder ranges; no token-index field is serialized by SpeechLM.
        """
        vocab_size = getattr(self.text_config, "vocab_size", None)
        if vocab_size is None:
            return None
        return int(vocab_size) - _SPEECHLM_EMBED_EXTRA_ROWS

    @image_token_index.setter
    def image_token_index(self, value: int | None) -> None:
        """Accept vLLM's runtime target-to-draft copy without serializing it."""
        expected = self.image_token_index
        if expected is None:
            # Transformers applies unknown config kwargs in its base-class
            # constructor, before this wrapper has loaded the real backbone.
            # Defer validation and discard the temporary value afterwards so
            # it never becomes serialized state.
            self._pending_image_token_index = value
        elif value is not None and value != expected:
            raise ValueError(
                f"image_token_index={value!r} does not match the backbone vocabulary boundary {expected}."
            )

    @property
    def mtp_hybrid_override_pattern(self) -> str:
        """Hybrid layer pattern for MTP heads, consumed by NemotronHMultiTokenPredictor.

        Reads from the ``mtp.hybrid_override_pattern`` field in config.json.
        vLLM supports any sequence of ``"*"`` (attention) and ``"E"`` (MoE),
        with one physical MTP layer module instantiated per character. Other
        characters are rejected by vLLM during model construction.
        """
        mtp_cfg = self.__dict__.get("mtp") or {}
        return mtp_cfg.get("hybrid_override_pattern", "*") if isinstance(mtp_cfg, dict) else "*"

    _ATTR_ALIASES = {
        "rms_norm_eps": "layer_norm_epsilon",
        "layer_norm_eps": "layer_norm_epsilon",
    }

    def __getattr__(self, name):
        """Delegate unknown attribute lookups to the wrapped backbone config.

        Called only when the attribute is not found in the normal lookup chain
        (instance ``__dict__`` + class hierarchy). Short-circuits in two cases:

        * names starting with ``_`` (dunders and privates) -- pickling,
          copying, and reflection rely on the default ``AttributeError`` path;
        * plugin-specific fields (``perception``, ``pretrained_llm``, ...) --
          guards against infinite recursion if one of them is queried before
          ``__init__`` finishes, and prevents accidental delegation to a
          same-named attribute on ``text_config``.

        For everything else, translate aliases (``rms_norm_eps`` ->
        ``layer_norm_epsilon`` on hybrid backends) and delegate to
        ``self.text_config``.
        """
        if name.startswith("_") or name in (
            "perception",
            "pretrained_llm",
            "llm_config",
            "pretrained_asr",
            "audio_locator_tag",
            "image_token_index",
            "prompt_format",
            "pretrained_weights",
            "text_config",
            "lora",
            "is_hybrid",
            "encoder_chunk_size_seconds",
            "pe_encoder_path",
            "pe_encoder_config",
            "pe_encoder_overrides",
            "speaker_encoder",
        ):
            raise AttributeError(name)
        alias = self._ATTR_ALIASES.get(name, name) if self.is_hybrid else name
        try:
            return getattr(self.text_config, alias)
        except AttributeError:
            if alias != name:
                try:
                    return getattr(self.text_config, name)
                except AttributeError:
                    pass
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
