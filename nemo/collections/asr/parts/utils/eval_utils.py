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

import json
import re
from typing import Optional, Tuple, Union

from torchmetrics.text import SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode

TEXT_METRICS_MAPPING = {
    'bleu': SacreBLEUScore,
    'rouge': ROUGEScore,
}

from omegaconf import DictConfig


def flatten_dict_config(config: DictConfig, parent_key='', sep='.', join='\n') -> str:
    """
    Flatten a DictConfig object into a string of parameter names and their values.

    Args:
        config (DictConfig): The input DictConfig object.
        parent_key (str): The parent key for nested configurations.
        sep (str): Separator between keys.

    Returns:
        str: Flattened string of parameter names and their values.
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten_dict_config(v, new_key, sep=sep, join=join).split(join))
        else:
            items.append(f"{new_key}={v}")
    return join.join(items)


def get_hydra_override_from_config(config: Optional[DictConfig] = None, exclude_keys: Optional[list] = None) -> str:
    """
    Flatten a DictConfig object into a string of hydra overrides for commandline, for example:
    >>> config = OmegaConf.create({"foo": {"bar": 1, "baz": 2}})
    >>> get_hydra_override_from_config(config)
    "++foo.bar=1 ++foo.baz=2"
    """
    if not config:
        return ""
    join = '\n'
    overrides = flatten_dict_config(config, join=join).split(join)
    if exclude_keys:
        overrides = [x for x in overrides if not any([y == x.split("=")[0] for y in exclude_keys])]
    param_str = " ".join([f"++{x}" for x in overrides])
    return param_str


def strip_spaces_before_punctuations(text: str) -> str:
    """
    Remove spaces before punctuations, e.g. "hello , world" -> "hello, world"
    """
    result = re.sub(r'(\w)\s+([.,;!?])', r'\1\2', text)
    return result


def remove_punctuations(text: str, punctuations: Optional[Union[list, str]] = None) -> str:
    """
    Remove punctuations from a string
    """
    if not punctuations:
        punctuations = [char for char in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~']

    for punctuation in punctuations:
        text = text.replace(punctuation, '')
    return text


def clean_label(_str: str, num_to_words: bool = True, langid="en") -> str:
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    """
    replace_with_space = [char for char in '/?*\",.:=?_{|}~¨«·»¡¿„…‧‹›≪≫!:;ː→']
    replace_with_blank = [char for char in '`¨´‘’“”`ʻ‘’“"‘”']
    replace_with_apos = [char for char in '‘’ʻ‘’‘']
    _str = _str.strip()
    _str = _str.lower()
    for i in replace_with_blank:
        _str = _str.replace(i, "")
    for i in replace_with_space:
        _str = _str.replace(i, " ")
    for i in replace_with_apos:
        _str = _str.replace(i, "'")
    if num_to_words:
        if langid == "en":
            _str = convert_num_to_words(_str, langid="en")
        else:
            logging.warning(
                "Currently support basic num_to_words in English only. Please use Text Normalization to convert other languages! Skipping!",
                mode=LogMode.ONCE,
            )

    ret = " ".join(_str.split())
    return ret


def convert_num_to_words(_str: str, langid: str = "en") -> str:
    """
    Convert digits to corresponding words. Note this is a naive approach and could be replaced with text normalization.
    """
    if langid == "en":
        num_to_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        _str = _str.strip()
        words = _str.split()
        out_str = ""
        num_word = []
        for word in words:
            if word.isdigit():
                num = int(word)
                while num:
                    digit = num % 10
                    digit_word = num_to_words[digit]
                    num_word.append(digit_word)
                    num = int(num / 10)
                    if not (num):
                        num_str = ""
                        num_word = num_word[::-1]
                        for ele in num_word:
                            num_str += ele + " "
                        out_str += num_str + " "
                        num_word.clear()
            else:
                out_str += word + " "
        out_str = out_str.strip()
    else:
        logging.warning(
            "Currently support basic num_to_words in English only. Please use Text Normalization to convert other languages!",
            mode=LogMode.ONCE,
        )
    return out_str


def cal_write_wer(
    pred_manifest: str = None,
    gt_text_attr_name: str = "text",
    pred_text_attr_name: str = "pred_text",
    clean_groundtruth_text: bool = False,
    langid: str = 'en',
    use_cer: bool = False,
    output_filename: str = None,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    punctuations: Optional[list] = None,
    strip_punc_space: bool = False,
) -> Tuple[str, dict, str]:
    """ 
    Calculate wer, inserion, deletion and substitution rate based on groundtruth text and pred_text_attr_name (pred_text) 
    We use WER in function name as a convention, but Error Rate (ER) currently support Word Error Rate (WER) and Character Error Rate (CER)
    """
    samples = []
    hyps = []
    refs = []
    eval_metric = "cer" if use_cer else "wer"

    with open(pred_manifest, 'r') as fp:
        for line in fp:
            sample = json.loads(line)

            if gt_text_attr_name not in sample:
                if "text" in sample:
                    gt_text_attr_name = "text"
                else:
                    logging.info(
                        f"ground-truth text attribute {gt_text_attr_name} is not present in manifest! Cannot calculate WER. Returning!"
                    )
                return None, None, eval_metric

            hyp = sample[pred_text_attr_name].strip()
            ref = sample[gt_text_attr_name].strip()

            if clean_groundtruth_text:
                ref = clean_label(ref, langid=langid)

            if ignore_punctuation:
                ref = remove_punctuations(ref, punctuations=punctuations)
                hyp = remove_punctuations(hyp, punctuations=punctuations)
            elif strip_punc_space:
                ref = strip_spaces_before_punctuations(ref)
                hyp = strip_spaces_before_punctuations(hyp)

            if ignore_capitalization:
                ref = ref.lower()
                hyp = hyp.lower()

            wer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
                hypotheses=[hyp], references=[ref], use_cer=use_cer
            )
            sample[eval_metric] = wer  # evaluatin metric, could be word error rate of character error rate
            sample['tokens'] = tokens  # number of word/characters/tokens
            sample['ins_rate'] = ins_rate  # insertion error rate
            sample['del_rate'] = del_rate  # deletion error rate
            sample['sub_rate'] = sub_rate  # substitution error rate

            samples.append(sample)
            hyps.append(hyp)
            refs.append(ref)

    total_wer, total_tokens, total_ins_rate, total_del_rate, total_sub_rate = word_error_rate_detail(
        hypotheses=hyps, references=refs, use_cer=use_cer
    )

    if not output_filename:
        output_manifest_w_wer = pred_manifest
    else:
        output_manifest_w_wer = output_filename

    with open(output_manifest_w_wer, 'w') as fout:
        for sample in samples:
            json.dump(sample, fout)
            fout.write('\n')
            fout.flush()

    total_res = {
        "samples": len(samples),
        "tokens": total_tokens,
        eval_metric: total_wer,
        "ins_rate": total_ins_rate,
        "del_rate": total_del_rate,
        "sub_rate": total_sub_rate,
    }
    return output_manifest_w_wer, total_res, eval_metric


def cal_write_text_metric(
    pred_manifest: str = None,
    gt_text_attr_name: str = "text",
    pred_text_attr_name: str = "pred_text",
    output_filename: str = None,
    ignore_capitalization: bool = False,
    ignore_punctuation: bool = False,
    punctuations: Optional[list] = None,
    metric: str = 'bleu',
    metric_args: Optional[dict] = None,
    strip_punc_space: bool = False,
):
    samples = []
    hyps = []
    refs = []

    if metric not in TEXT_METRICS_MAPPING:
        raise ValueError(f"metric {metric} is not supported! Please choose from {TEXT_METRICS_MAPPING.keys()}")

    metric_calculator = TEXT_METRICS_MAPPING[metric](**metric_args) if metric_args else TEXT_METRICS_MAPPING[metric]()
    with open(pred_manifest, 'r') as fp:
        for line in fp:
            sample = json.loads(line)

            if gt_text_attr_name not in sample:
                if "text" in sample:
                    gt_text_attr_name = "text"
                else:
                    logging.info(
                        f"ground-truth text attribute {gt_text_attr_name} is not present in manifest! Cannot calculate {metric}. Returning!"
                    )
                return None, None, metric

            hyp = sample[pred_text_attr_name].strip()
            ref = sample[gt_text_attr_name].strip()

            if ignore_punctuation:
                ref = remove_punctuations(ref, punctuations=punctuations)
                hyp = remove_punctuations(hyp, punctuations=punctuations)
            elif strip_punc_space:
                ref = strip_spaces_before_punctuations(ref)
                hyp = strip_spaces_before_punctuations(hyp)

            if ignore_capitalization:
                ref = ref.lower()
                hyp = hyp.lower()

            if metric == 'bleu':
                score = metric_calculator([hyp], [[ref]]).item()
            else:
                score = metric_calculator(hyp, ref).item()
            sample[metric] = score  # evaluatin metric, could be word error rate of character error rate

            samples.append(sample)
            hyps.append(hyp)
            refs.append(ref)

    if metric == 'bleu':
        refs = [[ref] for ref in refs]
    total_score = metric_calculator(hyps, refs).item()

    if not output_filename:
        output_manifest_w_wer = pred_manifest
    else:
        output_manifest_w_wer = output_filename

    with open(output_manifest_w_wer, 'w') as fout:
        for sample in samples:
            json.dump(sample, fout)
            fout.write('\n')
            fout.flush()

    total_res = {
        "samples": len(samples),
        metric: total_score,
    }
    return output_manifest_w_wer, total_res, metric
