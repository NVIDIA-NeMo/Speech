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


class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


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
    use_asr_decoder: bool = False  # set this to True if using SALMWithAsrDecoder


@hydra_runner(config_name="SalmEvalConfig", schema=SalmEvalConfig)
def main(cfg: SalmEvalConfig):
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    if cfg.use_asr_decoder:
        model = SALMWithAsrDecoder.from_pretrained(cfg.pretrained_name)
    else:
        model = SALM.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    cuts = guess_parse_cutset(cfg.inputs).sort_by_duration()
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size),
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
        )
        answer_ids = answer_ids.cpu()
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_hyps = [
            normalizer(model.tokenizer.ids_to_text(parse_hyp(ans, eos_tokens)).strip()) for ans in answer_ids
        ]
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
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for cut, ref, hyp in zip(cuts, refs, hyps):
                writer.write({"id": cut.id, "duration": cut.duration, "text": ref, "pred_text": hyp})


def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = torch.isin(answer, torch.tensor(eos_tokens)).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


if __name__ == '__main__':
    main()
