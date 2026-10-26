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
"""Derive a SALMAutomodel fine-tuning config from an exported checkpoint.

Hand-writing the `model:` section for a fine-tune is the main source of silent
breakage: any drift between the architecture the YAML describes and the one the
checkpoint was exported from produces parameters that never get restored. This
script instead copies the checkpoint's own `config.json` into the `model:`
section and changes only what a fine-tune must change:

* weights come from `init_from_checkpoint`, not from `pretrained_llm`/`pretrained_asr`
* `pretrained_llm` points at the checkpoint's bundled `llm_backbone` config
* the freeze policy, LoRA block, optimizer and LR schedule are recipe-owned
* single-GPU-friendly Automodel backends

Everything else — perception, `pe_encoder_config`, speculative-decoding heads, prompt format, packing —
is carried over verbatim so the fine-tuned model stays export- and
vLLM-compatible.
"""
import argparse
import json
from pathlib import Path

import yaml

# Keys that belong to the HuggingFace serialization, not to the NeMo model config.
_HF_ONLY_KEYS = {
    "architectures",
    "model_type",
    "llm_config",
    "dtype",
    "pad_token_id",
    "eos_token_id",
    "init_configure_model",
    "trust_remote_code",
}
# Recipe-owned keys: always replaced by this script's arguments.
_RECIPE_KEYS = {
    "pretrained_llm",
    "tokenizer_path",
    "pretrained_asr",
    "pretrained_weights",
    "init_from_checkpoint",
    "freeze_params",
    "prevent_freeze_params",
    "optimizer",
    "lr_scheduler",
    "automodel_backend",
    "lora",
    "train_gate",
    "moe_metrics",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="exported SALMAutomodel checkpoint directory")
    p.add_argument("--out", required=True, help="path of the YAML config to write")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--val-name", default="atc_dev")
    p.add_argument("--exp-dir", required=True)
    p.add_argument(
        "--prompt",
        default="Provide a verbatim transcript of the audio.",
        help="fixed prompt attached to every example. Pass 'manifest' to use each row's own "
        "`context` field instead (per-example prompts: QA, label lists, target language). "
        "A fixed prompt is attached with `tags`, which OVERWRITES any per-row context.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=None,
        help="multiply the LR for perception/encoder parameters by this factor. Residual errors in "
        "this domain are acoustic substitutions and the encoder is the acoustic module, so it "
        "often wants a larger step than LoRA adapters on a frozen LLM.",
    )
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--lora-dim", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj,in_proj,out_proj,up_proj,down_proj",
        help="empty string disables LoRA entirely",
    )
    p.add_argument(
        "--train-encoder",
        action="store_true",
        help="unfreeze the speech encoder (the ASR branch of a parallel-expert encoder, if the checkpoint has one)",
    )
    p.add_argument("--train-proj", action="store_true", help="unfreeze perception.proj")
    p.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "adafactor"],
        help="adafactor factors the second moment, cutting optimizer state from ~8 bytes/param "
        "to near zero. For a ~30B-parameter LLM in bf16, full SFT with AdamW needs ~400 GB of weights, "
        "gradients and optimizer state; Adafactor brings that to ~130 GB.",
    )
    p.add_argument(
        "--llm-lr-scale",
        type=float,
        default=None,
        help="LR multiplier for unfrozen full-rank LLM weights (--tune-mode llm-partial/full). "
        "Full-rank pretrained weights want a much smaller step than freshly initialized "
        "LoRA adapters; 0.1-0.3 is a reasonable starting range.",
    )
    p.add_argument(
        "--tune-mode",
        default="lora",
        choices=["lora", "llm-partial", "full"],
        help="how much of the LLM to train; see build_freeze_policy",
    )
    p.add_argument(
        "--unfreeze-llm-layers",
        type=int,
        default=8,
        help="with --tune-mode llm-partial, how many final LLM blocks to unfreeze",
    )
    p.add_argument(
        "--llm-num-layers", type=int, default=52, help="total number of LLM blocks in the checkpoint's backbone"
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--batch-tokens",
        type=int,
        default=None,
        # On a 256 GB GPU, 120000 reaches peak throughput (~179 GB peak memory).
        # Scale down on a smaller card or a small corpus:
        # peak_GB ~= 65 + 0.95 * batch_tokens/1000, and a dataset under ~20 h
        # needs a smaller budget to get a usable number of optimizer steps.
        help="tokens per batch (audio frames + text tokens) using multimodal sampling",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="DROPS examples longer than this many tokens (a filter, not a batch cap); unset by default",
    )
    # Augmentation. Off by default: it is a large behavioural change and should
    # be opted into. On a small corpus it is usually the difference between
    # memorizing and generalizing.
    p.add_argument(
        "--augment", action="store_true", help="enable speed perturbation plus the radio-channel augmentations below"
    )
    p.add_argument(
        "--noise-manifest",
        default=None,
        help="NeMo manifest of noise cuts for on-the-fly mixing (see extract_noise.py)",
    )
    p.add_argument("--noise-snr", default="5,20", help="min,max SNR in dB")
    p.add_argument("--noise-prob", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--limit-train-batches", type=int, default=250)
    p.add_argument(
        "--val-every-n-epochs",
        type=int,
        default=None,
        help="switch validation/checkpointing to epoch cadence; required when a data "
        "epoch yields fewer batches than --limit-train-batches",
    )
    p.add_argument("--limit-val-batches", type=int, default=25)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--save-top-k", type=int, default=3)
    p.add_argument("--activation-checkpointing", action="store_true", default=True)
    p.add_argument("--no-activation-checkpointing", dest="activation_checkpointing", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def encoder_patterns(model_cfg: dict) -> list:
    """Parameter-name patterns of the trainable speech encoder for this checkpoint.

    A parallel-expert (PE) encoder mounts an ASR branch and a diarization branch under ``perception.encoder``; only
    the ASR branch and its norm are trained (the diarization branch is a fixed speaker-activity feature extractor).
    A regular SALM has the whole encoder at ``perception.encoder`` (or ``perception.encoder_multilayer.encoder``
    behind a multi-layer connector).
    """
    if any(model_cfg.get(k) not in (None, "", False, {}) for k in ("pe_encoder_path", "pe_encoder_config")):
        return [r"^perception\.encoder\.asr_encoder\..+$", r"^perception\.encoder\.asr_norm\..+$"]
    return [r"^perception\.encoder\..+$", r"^perception\.encoder_multilayer\.encoder\..+$"]


def build_freeze_policy(args, model_cfg: dict):
    """Freeze everything, then re-open exactly the modules the recipe trains.

    ``freeze_params`` is a list of regexes matched against parameter names;
    ``prevent_freeze_params`` wins over it. LoRA parameters are re-opened
    automatically by ``ensure_lora_trainable``.

    ``--tune-mode`` selects how much of the model moves:

    ``lora``       LLM frozen, adapted only through LoRA. Cheapest, and the only
                   mode whose optimizer state fits comfortably beside a large LLM.
    ``llm-partial`` additionally unfreezes the LLM's final ``--unfreeze-llm-layers``
                   blocks. The last blocks carry most of the output-distribution
                   adaptation, so this buys much of full SFT's benefit at a
                   fraction of the optimizer memory.
    ``full``       unfreezes every LLM parameter. Included for completeness; on a
                   large LLM this needs optimizer state far beyond a single
                   GPU unless the optimizer shards or offloads. Perception is
                   still controlled only by ``--train-encoder``/``--train-proj``.

    Note the asymmetry: the perception encoder is *always* a candidate for full
    training (it is small), while a large LLM is not. That is a memory fact, not a
    modelling opinion.
    """
    freeze = [r"^llm\..+$", r"^perception\..+$"]
    prevent = []
    if args.train_encoder:
        prevent.extend(encoder_patterns(model_cfg))
    if args.train_proj:
        prevent.append(r"^perception\.proj\..+$")

    if args.tune_mode == "full":
        freeze = [r"^perception\..+$"]  # the whole LLM trains; perception follows its two flags
    elif args.tune_mode == "llm-partial":
        n = args.unfreeze_llm_layers
        total = args.llm_num_layers
        first = max(0, total - n)
        # Match the final n decoder blocks by index, plus the output norm/head.
        idx = "|".join(str(i) for i in range(first, total))
        prevent.append(rf"^llm\.model\.layers\.({idx})\..+$")
        prevent.append(r"^llm\.model\.norm_f\..+$")
        prevent.append(r"^llm\.lm_head\..+$")
    return freeze, prevent


def checkpoint_every_n_steps(args) -> int:
    """Step-based checkpoint cadence in optimizer steps, matching validation.

    ``val_check_interval`` (= ``limit_train_batches``) counts dataloader batches, while ``every_n_train_steps``
    counts optimizer steps; with gradient accumulation one optimizer step consumes ``accumulate_grad_batches``
    batches, so the two only line up if the interval is divisible by it.
    """
    acc = max(1, args.accumulate_grad_batches)
    if args.limit_train_batches % acc:
        raise SystemExit(
            f"--limit-train-batches ({args.limit_train_batches}) must be a multiple of --accumulate-grad-batches "
            f"({acc}) so that validation and checkpoints happen at the same optimizer step"
        )
    return args.limit_train_batches // acc


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())

    model = {k: v for k, v in cfg.items() if k not in _HF_ONLY_KEYS and k not in _RECIPE_KEYS}

    # The bundled ``llm_backbone`` directory holds only the LLM architecture config;
    # tokenizer.json and tokenizer_config.json live at the checkpoint root. NeMo's
    # from_pretrained path splits these two automatically, but a hand-built config
    # must do it explicitly or SALMAutomodel.__init__ fails to build the tokenizer.
    model["pretrained_llm"] = str(ckpt / "llm_backbone")
    model["tokenizer_path"] = str(ckpt)
    # Keep the original string rather than nulling it. Training never reads it
    # (``pretrained_weights: False`` plus ``init_from_checkpoint`` supply the
    # weights, and the encoder is built from ``pe_encoder_config``), but the path
    # is copied into the exported config, and vLLM's SpeechLM plugin *validates*
    # that ``pretrained_asr`` is non-null before it will build a ModelConfig:
    #     Value error, NeMo SpeechLM config must declare pretrained_asr
    # It never dereferences it — the base checkpoint ships a dead Lustre path and
    # serves fine — so preserving the value verbatim is what keeps an exported
    # fine-tune loadable.
    model["pretrained_asr"] = cfg.get("pretrained_asr")
    # False keeps NeMo from re-downloading and re-initializing the child modules;
    # init_from_checkpoint then restores the consolidated trained weights.
    model["pretrained_weights"] = False
    model["init_from_checkpoint"] = str(ckpt)
    model["torch_dtype"] = cfg.get("torch_dtype", "bfloat16")

    # Consumed by nemo.collections.speechlm2.parts.optim_setup.build_param_groups,
    # which puts matching parameters in their own optimizer group at
    # lr = optimizer.lr * multiplier. First match wins, so the LLM pattern is
    # listed before the encoder pattern only if both could match a name (they
    # cannot here, but order is part of the contract).
    multipliers = {}
    if args.llm_lr_scale:
        # Only non-LoRA LLM tensors: the adapters keep the base LR.
        multipliers[r"^llm\.(?!.*lora_).+$"] = args.llm_lr_scale
    if args.encoder_lr_scale:
        multipliers[r"^perception\..+$"] = args.encoder_lr_scale
    if multipliers:
        model["lr_multipliers"] = multipliers

    freeze, prevent = build_freeze_policy(args, cfg)
    model["freeze_params"] = freeze
    model["prevent_freeze_params"] = prevent

    # The router stays frozen during fine-tuning. Re-training it on a narrow
    # domain collapses expert utilization and is not recoverable from the
    # exported checkpoint.
    model["train_gate"] = False
    model["moe_metrics"] = {"enabled": True, "mode": "brief", "detailed_every_steps": None, "top_k_experts": 5}

    targets = [t for t in args.lora_targets.split(",") if t]
    if targets:
        model["lora"] = {
            "dim": args.lora_dim,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": targets,
        }

    # Single GPU: no NVLINK/NVSHMEM all-to-all, so the DeepEP dispatcher the
    # checkpoint was trained with must be replaced by the portable one.
    backend = dict(cfg.get("automodel_backend") or {})
    backend["dispatcher"] = "torch"
    backend.pop("dispatcher_num_sms", None)
    backend.pop("dispatcher_async_dispatch", None)
    model["automodel_backend"] = backend

    if args.optimizer == "adafactor":
        # No betas/fused/foreach: torch.optim.Adafactor does not accept them.
        model["optimizer"] = {
            "_target_": "torch.optim.Adafactor",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    else:
        model["optimizer"] = {
            "_target_": "torch.optim.AdamW",
            "lr": args.lr,
            "betas": [0.9, 0.98],
            "weight_decay": args.weight_decay,
            "foreach": False,
            "fused": True,
        }
    model["lr_scheduler"] = {
        "_target_": "nemo.core.optim.lr_scheduler.CosineAnnealing",
        "warmup_steps": args.warmup_steps,
        "min_lr": args.min_lr,
        "max_steps": args.max_steps,
    }

    audio_token_estimator = cfg.get("audio_token_estimator")

    def dataset_block(manifest, context):
        """Build an ``input_cfg`` list from a manifest spec.

        Accepts either a single path, or a weighted blend written as
        ``path:weight,path:weight,...``. Weights are Lhotse sampling
        probabilities, so a blend draws from each source independently of its
        size -- that is the point: it lets a 5.9 h target corpus dominate
        training while a 100 h replay corpus contributes a fixed small share.
        """
        specs = []
        for part in str(manifest).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part and not part.endswith(".json"):
                path, weight = part.rsplit(":", 1)
                specs.append((path, float(weight)))
            else:
                specs.append((part, None))

        entries = []
        for path, weight in specs:
            entry = {
                "type": "lhotse_as_conversation",
                "manifest_filepath": path,
                "audio_locator_tag": cfg["audio_locator_tag"],
            }
            if context != "manifest":
                entry["tags"] = {"context": context}
            if weight is not None:
                entry["weight"] = weight
            entries.append(entry)
        return entries

    train_ds = {
        "sample_rate": 16000,
        "prompt_format": cfg["prompt_format"],
        "token_equivalent_duration": 0.08,
        "audio_token_estimator": audio_token_estimator,
        "input_cfg": dataset_block(args.train_manifest, args.prompt),
        "seed": args.seed,
        "shuffle": True,
        "shard_seed": "randomized",
        "num_workers": args.num_workers,
        "shuffle_buffer_size": 10000,
    }
    if args.batch_tokens is not None:
        # `lhotse_as_conversation` yields NeMoMultimodalConversation objects, which
        # expose `total_length` (audio frames + text tokens) and have no `duration`.
        # Duration-based bucketing therefore fails with
        # "AttributeError: No such attribute: duration"; multimodal sampling is the
        # supported dynamic-batching path for this data type.
        train_ds["batch_size"] = None
        train_ds["use_multimodal_sampling"] = True
        train_ds["measure_total_length"] = True
        train_ds["batch_tokens"] = args.batch_tokens
        # Packed sequences make bucketing counterproductive. Bucketing exists to
        # stop short utterances being padded up to the longest in the batch, but
        # THD packing concatenates variable-length sequences with no padding at
        # all -- measured packing efficiency is ~0.95 without it. Keeping both
        # only narrows the length distribution each batch draws from, which
        # correlates the gradient and costs shuffle quality for no memory win.
        train_ds["use_bucketing"] = False
        # Required to make batch_tokens mean what it looks like it means. The
        # sampling constraint otherwise measures a batch as
        # batch_size * longest_example -- padded accounting -- so a batch of
        # mixed-length utterances is declared full long before its real token
        # count approaches batch_tokens. Measured on AMI: with this off and
        # batch_tokens=48000, actual batches held ~8.4k tokens. Packed
        # accounting sums the true example lengths, which is what the THD
        # attention path actually allocates.
        #
        # Exact packed sampling requires `audio_token_estimator` (set above from
        # the checkpoint's own preprocessor/subsampling config);
        # token_equivalent_duration alone is only an approximation and the
        # sampler refuses to use it here.
        train_ds["use_packed_sequence_sampling"] = True
        # NOTE: `max_tokens` is a *filter*, not a batching cap -- TokenCountFilter
        # silently drops every example longer than it. Leave it unset unless you
        # intend to discard long utterances, and if you do set it, log how many
        # examples it removes.
        if args.max_tokens is not None:
            train_ds["max_tokens"] = args.max_tokens
        # Leave pretokenize at its default (True). The dataloader warns that
        # tokenizing in the main process may slow training and suggests
        # pretokenize=false, but that advice does not apply to token-based
        # multimodal sampling: the sampler calls measure_formattable_length,
        # which reads example.context_ids, and those only exist once the example
        # has been tokenized. Turning it off fails at the first batch with
        # "AttributeError: No such attribute: context_ids".
    else:
        train_ds["batch_size"] = args.batch_size or 8

    if args.augment:
        # Only the *cut-level* augmentations are usable here. Lhotse attaches
        # augmentations at two different points, and the distinction decides
        # compatibility with this data type:
        #
        #   cut-level    (applied to the CutSet before the sampler):
        #                noise mixing, speed perturbation  -> work
        #   sampler-level (applied as .map() over sampled batches):
        #                lowpass, clipping, rir, compression -> do NOT work
        #
        # The sampler-level transforms expect raw `Cut` objects and read
        # `example.sampling_rate`, which `NeMoMultimodalConversation` does not
        # expose, so they die with "AttributeError: No such attribute:
        # sampling_rate" on the first batch. This is the same family of failure
        # as `duration` (bucketing) and `context_ids` (pretokenize=False): the
        # conversation type supports a narrower interface than a Cut, and the
        # error never says so.
        #
        # Band-limiting/clipping/codec simulation would suit a VHF radio domain
        # well; to use them the audio has to be corrupted offline instead.
        train_ds["perturb_speed"] = True
        if args.noise_manifest:
            lo, hi = (float(x) for x in args.noise_snr.split(","))
            train_ds["noise_path"] = {"manifest_filepath": args.noise_manifest}
            train_ds["noise_snr"] = [lo, hi]
            train_ds["noise_mix_prob"] = args.noise_prob

    validation_ds = {
        "prompt_format": cfg["prompt_format"],
        "token_equivalent_duration": 0.08,
        "audio_token_estimator": audio_token_estimator,
        "sample_rate": 16000,
        "batch_size": args.val_batch_size,
        "seed": args.seed,
        "shard_seed": "randomized",
        "num_workers": 2,
        "datasets": {args.val_name: {"input_cfg": dataset_block(args.val_manifest, args.prompt)}},
    }

    trainer = {
        "devices": 1,
        "accelerator": "gpu",
        "num_nodes": 1,
        "precision": "bf16-true",
        "logger": False,
        "enable_checkpointing": False,
        "use_distributed_sampler": False,
        "max_steps": args.max_steps,
        # `val_check_interval` counts batches *within* an epoch. Lhotse's
        # dataloader is finite, so if a data epoch yields fewer batches than this
        # value the counter never reaches it and **validation never runs** --
        # silently. `val_loss` then never enters callback_metrics, every
        # checkpoint logs "'val_loss' was not in top k", and `save_top_k` keeps
        # nothing but `-last`. Seen on a 5.9 h corpus: ~22 batches/epoch against
        # val_check_interval=100.
        #
        # For corpora large enough to exceed the interval, keep step-based
        # cadence. Otherwise drive validation and checkpointing off epoch
        # boundaries, where they coincide by construction.
        "limit_train_batches": args.limit_train_batches,
        **(
            {"val_check_interval": args.limit_train_batches}
            if args.val_every_n_epochs is None
            else {"check_val_every_n_epoch": args.val_every_n_epochs}
        ),
        "limit_val_batches": args.limit_val_batches,
        "log_every_n_steps": 10,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "strategy": {
            "_target_": "nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy",
            "dp_size": 1,
            "dp_replicate_size": 1,
            "tp_size": 1,
            "pp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
            "activation_checkpointing_llm": args.activation_checkpointing,
            "activation_checkpointing_perception": args.activation_checkpointing,
        },
    }

    exp_manager = {
        "exp_dir": None,
        "explicit_log_dir": args.exp_dir,
        "name": Path(args.exp_dir).name,
        "create_tensorboard_logger": True,
        "create_checkpoint_callback": True,
        "use_datetime_version": False,
        "resume_if_exists": True,
        "resume_ignore_no_checkpoint": True,
        "create_wandb_logger": False,
        "checkpoint_callback_params": {
            "filename": "{step}",
            # Rank on val_loss rather than val_acc. Neither is a reliable proxy
            # for decoded WER (val_loss can prefer a checkpoint that decodes
            # several points worse on dev), so the checkpoint still has to be
            # chosen by decoding. The reason to use
            # val_loss anyway is that val_acc is teacher-forced next-token
            # accuracy over a few batches: it quantizes coarsely and ties almost
            # immediately, and once it ties `save_top_k` degenerates into "keep
            # the most recent k", discarding the early checkpoints you would
            # want to compare against. val_loss keeps moving, so top-k retains a
            # spread of the run instead of only its most over-fit end.
            "monitor": "val_loss",
            "mode": "min",
            # exp_manager defaults every_n_epochs to 1; Lightning rejects the two
            # step/epoch cadences being set at once, so exactly one is set here.
            # Whichever cadence validation uses, checkpointing must match it --
            # a checkpoint taken at a step where validation has not just run has
            # no fresh `val_loss` to be ranked on.
            **(
                {"every_n_train_steps": checkpoint_every_n_steps(args), "every_n_epochs": 0}
                if args.val_every_n_epochs is None
                else {"every_n_train_steps": None, "every_n_epochs": args.val_every_n_epochs}
            ),
            "save_top_k": args.save_top_k,
            "always_save_nemo": False,
            "save_nemo_on_train_end": False,
        },
    }

    out = {
        "model": model,
        "trainer": trainer,
        "data": {"train_ds": train_ds, "validation_ds": validation_ds},
        "exp_manager": exp_manager,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fo:
        yaml.safe_dump(out, fo, sort_keys=False, default_flow_style=False, width=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
