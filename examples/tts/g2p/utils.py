# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
from nemo.collections.tts.g2p.models.t5 import T5G2PModel
from nemo.utils import logging


def get_model(cfg, trainer):
    """
    Get model instance

    Args:
        cfg: model's config file
        trainer: trainer
    Return:
        G2PModel instance
    """
    if "CTC" in cfg.name:
        model = CTCG2PModel(cfg=cfg.model, trainer=trainer)
    elif cfg.name == "T5G2P":
        model = T5G2PModel(cfg=cfg.model, trainer=trainer)
    else:
        raise ValueError(f"{cfg.name} is not supported. Choose from [G2P-Conformer-CTC, T5G2P]")
    return model


def get_metrics(manifest: str, pred_field="pred_text", phoneme_field="text", grapheme_field="text_graphemes"):
    """
    Calculates WER and PER metrics (for duplicated grapheme entries with multiple reference values,
        the best matching prediction will be used for evaluation.)

    Args:
        manifest: Path to .json manifest file
        pred_field: name of the field in the output_file to save predictions
        phoneme_field: name of the field in manifest_filepath for ground truth phonemes
        grapheme_field: name of the field in manifest_filepath for input grapheme text

    Returns: WER and PER values
    """
    all_preds = []
    all_references = []
    all_graphemes = {}
    with open(manifest, "r") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            all_preds.append(line[pred_field])
            all_references.append(line[phoneme_field])

            if line[grapheme_field] not in all_graphemes:
                all_graphemes[line[grapheme_field]] = []
            all_graphemes[line[grapheme_field]].append(i)

    # collect all examples with multiple phoneme options and same grapheme form, choose the one with min PER
    all_graphemes = {k: v for k, v in all_graphemes.items() if len(v) > 1}
    lines_to_drop = []
    for phon_amb_indices in all_graphemes.values():
        refs, preds = [], []
        for phon_amb_indices_ in phon_amb_indices:
            refs.append(all_references[phon_amb_indices_])
            preds.append(all_preds[phon_amb_indices_])
        pers = []
        for ref_, pred_ in zip(refs, preds):
            pers.append(word_error_rate(hypotheses=[pred_], references=[ref_], use_cer=True))

        min_idx = pers.index(min(pers))

        phon_amb_indices.pop(min_idx)
        lines_to_drop.extend(phon_amb_indices)

    # drop duplicated examples, only keep with min PER
    all_preds = [x for i, x in enumerate(all_preds) if i not in lines_to_drop]
    all_references = [x for i, x in enumerate(all_references) if i not in lines_to_drop]

    wer = word_error_rate(hypotheses=all_preds, references=all_references)
    per = word_error_rate(hypotheses=all_preds, references=all_references, use_cer=True)

    logging.info(f"{manifest}: PER: {per * 100:.2f}%, WER: {wer * 100:.2f}%, lines: {len(all_references)}")
    return wer, per
