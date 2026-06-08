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
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from transformers import GenerationConfig
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2.models import SALM, SALMWithAsrDecoder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


def _resolve_model_cls(pretrained_name: str, use_asr_decoder: bool, use_nemo_automodel: bool | None):
    """Pick model class. Auto-detects from config.json when use_nemo_automodel is None."""
    if use_asr_decoder:
        return SALMWithAsrDecoder
    if use_nemo_automodel is None:
        # Auto-detect: peek at config.json
        from transformers.utils import cached_file

        config_path = cached_file(
            pretrained_name,
            "config.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if config_path is not None:
            with open(config_path) as f:
                use_nemo_automodel = json.load(f).get("use_nemo_automodel", False)
        else:
            use_nemo_automodel = False
    if use_nemo_automodel:
        from nemo.collections.speechlm2.models import SALMAutomodel

        return SALMAutomodel
    return SALM


@dataclass
class SalmEvalConfig:
    pretrained_name: str
    inputs: str
    batch_size: int = 64
    max_new_tokens: int = 128
    output_manifest: Optional[str] = "generations.jsonl"
    verbose: bool = True
    use_normalizer: Optional[str] = "english"  # "english", "basic", or "none" / "None"
    device: str = "cuda"
    dtype: str = "bfloat16"
    extra_eos_tokens: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    enable_thinking: Optional[bool] = None
    use_asr_decoder: bool = False  # set this to True if using SALMWithAsrDecoder
    use_nemo_automodel: Optional[bool] = None  # None = auto-detect from config.json
    # Parallelism sizes for distributed inference (launch with torchrun)
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    # When True and the model has an MTP head, report the estimated average
    # acceptance length (how many speculative tokens would be accepted per step).
    report_mtp_acceptance: bool = False


@torch.no_grad()
def _compute_mtp_acceptance(model, answer_ids: torch.Tensor) -> Optional[dict]:
    """Estimate MTP acceptance length on a batch of generated sequences.

    Runs the LLM in teacher-forced mode on ``answer_ids``, activates the MTP
    head (which requires ``model.llm.training=True``), and computes per-depth
    prediction accuracy.  Returns ``None`` when the model has no MTP head,
    when distributed FSDP2 sharding is active (DTensor forward outside a
    collective is unsupported), or when the sequences are too short.

    Acceptance-length formula (geometric approximation, depth-independent):
        E[accept_len] ≈ Σ_{d=0}^{D-1}  Π_{k=0}^{d} accuracy_k

    where ``accuracy_k`` = fraction of positions where MTP depth k correctly
    predicted the actual next token.

    Args:
        model: SALMAutomodel with ``_mtp_enabled=True``.
        answer_ids: ``[B, T]`` tensor of generated token IDs (no prompt tokens).

    Returns:
        Dict with keys ``per_depth_accuracy`` (list[float]) and
        ``avg_acceptance_length`` (float), or ``None``.
    """
    if not getattr(model, '_mtp_enabled', False):
        return None
    if getattr(model, '_use_fsdp', False):
        # DTensor forward outside a collective context is unsupported.
        return None
    llm = model.llm
    if getattr(llm, 'mtp', None) is None:
        return None

    B, T = answer_ids.shape
    D = llm.mtp.num_depths
    # Need at least D+3 tokens: input slice [0..T-2] + D targets starting at t+2.
    if T < D + 3:
        return None

    # Input: positions 0..T-2 → teacher-forced to predict positions 1..T-1.
    input_ids = answer_ids[:, :-1]  # [B, T-1]

    # Temporarily switch LLM to train mode so the MTP module fires.
    # Most production models use dropout=0.0, so the forward is deterministic.
    was_training = llm.training
    llm.train()
    try:
        inputs_embeds = model._embed_tokens(input_ids)
        out = llm(inputs_embeds=inputs_embeds, input_ids=input_ids, return_dict=True)
    finally:
        if not was_training:
            llm.eval()

    mtp_h = getattr(out, 'mtp_per_depth_h', None)
    if not mtp_h:
        return None

    from nemo_automodel.components.loss.utils import _get_lm_head_module

    lm_head = _get_lm_head_module(llm)
    if lm_head is None:
        return None

    # MTP depth d, slot i → predicts answer_ids[:, i + d + 2].
    # Valid slots: 0 .. T - d - 3  (need target at i+d+2 < T).
    per_depth_accuracy: list[float] = []
    for d, h_d in enumerate(mtp_h):
        target_start = d + 2
        valid_len = T - target_start  # number of valid prediction positions
        if valid_len <= 0:
            break
        logits_d = lm_head(h_d[:, :valid_len])  # [B, valid_len, V]
        preds_d = logits_d.argmax(-1)  # [B, valid_len]
        targets = answer_ids[:, target_start : target_start + valid_len]  # [B, valid_len]
        accuracy = (preds_d == targets).float().mean().item()
        per_depth_accuracy.append(accuracy)

    if not per_depth_accuracy:
        return None

    # E[accept_len] = Σ_{d} Π_{k<=d} p_k  (geometric approximation)
    avg_accept_len = 0.0
    cumulative = 1.0
    for p in per_depth_accuracy:
        cumulative *= p
        avg_accept_len += cumulative

    return {'per_depth_accuracy': per_depth_accuracy, 'avg_acceptance_length': avg_accept_len}


@hydra_runner(config_name="SalmEvalConfig", schema=SalmEvalConfig)
def main(cfg: SalmEvalConfig):
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    is_distributed = any(s > 1 for s in [cfg.tp_size, cfg.ep_size, cfg.pp_size, cfg.cp_size])
    model_cls = _resolve_model_cls(cfg.pretrained_name, cfg.use_asr_decoder, cfg.use_nemo_automodel)

    if is_distributed and model_cls is SALM:
        raise RuntimeError(
            "Distributed inference requires SALMAutomodel. Set use_nemo_automodel=true or use a checkpoint "
            "exported from SALMAutomodel."
        )

    if is_distributed:
        from nemo.collections.speechlm2.parts.parallel import setup_distributed

        strategy = setup_distributed(
            tp_size=cfg.tp_size, ep_size=cfg.ep_size, pp_size=cfg.pp_size, cp_size=cfg.cp_size
        )
        model = model_cls.from_pretrained(
            cfg.pretrained_name,
            device_mesh=strategy.device_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            moe_mesh=strategy.moe_mesh,
            torch_dtype=cfg.dtype,
        )
    else:
        model = model_cls.from_pretrained(cfg.pretrained_name)
        model = model.to(getattr(torch, cfg.dtype)).to(cfg.device)
    model = model.eval()

    cuts = guess_parse_cutset(cfg.inputs).sort_by_duration()
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        # rank=0 world_size=1 hardcoded so lhotse doesn't accidentally auto-split batches in model parallel settings
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size, rank=0, world_size=1),
        num_workers=1,
        batch_size=None,
    )

    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(
        cfg.use_normalizer, lambda x: x
    )

    eos_tokens = [model.text_eos_id]
    if cfg.extra_eos_tokens is not None:
        for t in cfg.extra_eos_tokens:
            tid = model.tokenizer.token_to_id(t)
            assert tid is not None, f"Token '{t}' is not in the model's vocabulary."
            eos_tokens.append(tid)

    # Construct the prompt from ASR data of the form.
    # Optional system prompt goes first.
    prompt = []
    if cfg.system_prompt is not None:
        prompt.append({"role": "system", "content": cfg.system_prompt})
    # If no user prompt is provided, just use the audio placeholder.
    content = model.audio_locator_tag
    # Otherwise:
    # * if user prompt already has audio placeholder, add it as-is,
    # * if not, append audio placeholder at the end of user prompt
    if cfg.user_prompt is not None:
        content = cfg.user_prompt
        if model.audio_locator_tag not in content:
            content = f"{content} {model.audio_locator_tag}"
    prompt.append({"role": "user", "content": content})

    refs = []
    hyps = []
    input_durations = []
    infer_durations = []
    # MTP acceptance tracking: list of per-batch avg_acceptance_length values.
    mtp_accept_lengths: list[float] = []
    _report_mtp = cfg.report_mtp_acceptance and getattr(model, '_mtp_enabled', False)

    for batch_idx, batch in enumerate(dloader):
        ts = perf_counter()
        answer_ids = model.generate(
            prompts=[prompt] * len(batch["cuts"]),  # identical prompt for each example
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            generation_config=GenerationConfig(
                max_new_tokens=cfg.max_new_tokens,
                bos_token_id=model.text_bos_id,
                eos_token_id=eos_tokens,
                pad_token_id=model.text_pad_id,
            ),
            enable_thinking=cfg.enable_thinking,
        )
        answer_ids = answer_ids.cpu()
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_hyps = [
            normalizer(model.tokenizer.ids_to_text(parse_hyp(ans, eos_tokens)).strip()) for ans in answer_ids
        ]

        if _report_mtp:
            mtp_stats = _compute_mtp_acceptance(model, answer_ids.to(model.device))
            if mtp_stats is not None:
                mtp_accept_lengths.append(mtp_stats['avg_acceptance_length'])
                if cfg.verbose:
                    depth_acc = ", ".join(f"{p:.3f}" for p in mtp_stats['per_depth_accuracy'])
                    logging.info(
                        f"Batch {batch_idx}: MTP depth accuracies=[{depth_acc}] "
                        f"avg_accept_len={mtp_stats['avg_acceptance_length']:.3f}"
                    )

        if cfg.verbose:
            batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info(
                f"Batch {batch_idx}: WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] RTFx={batch_rtfx:.1f}"
            )

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    # RTFx is baseline inference speed (no MTP speculative decoding).
    logging.info(f"RTFx (baseline, no speculative decoding): {rtfx:.1f}")
    if mtp_accept_lengths:
        avg_accept = sum(mtp_accept_lengths) / len(mtp_accept_lengths)
        logging.info(f"MTP avg acceptance length: {avg_accept:.3f}")

    with _create_output_writer(cfg.output_manifest) as writer:
        for cut, ref, hyp in zip(cuts, refs, hyps):
            writer.write({"id": cut.id, "duration": cut.duration, "text": ref, "pred_text": hyp})


def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = torch.isin(answer, torch.tensor(eos_tokens)).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


class _NullWriter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def write(self, data):
        pass


def _create_output_writer(output_manifest: Optional[str]):
    if output_manifest is None or not is_global_rank_zero():
        return _NullWriter()
    return SequentialJsonlWriter(output_manifest)


if __name__ == '__main__':
    main()
