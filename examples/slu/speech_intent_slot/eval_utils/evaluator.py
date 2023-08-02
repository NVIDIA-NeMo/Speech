# ! /usr/bin/python
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import ast
from typing import Dict, List, Tuple, Union

from .evaluation.metrics.metrics import ErrorMetric


def parse_semantics_str2dict(semantics_str: Union[List[str], str, Dict]) -> Tuple[Dict, bool]:
    """
    This function parse the input string to a valid python dictionary for later evaluation.
    Part of this function is adapted from
    https://github.com/speechbrain/speechbrain/blob/develop/recipes/SLURP/direct/train_with_wav2vec2.py#L110-L127
    """
    invalid = False
    if isinstance(semantics_str, dict):
        return semantics_str, invalid
    if isinstance(semantics_str, list):
        semantics_str = " ".join(semantics_str)

    try:
        if "|" in semantics_str:
            semantics_str = semantics_str.replace("|", ",")
        _dict = ast.literal_eval(semantics_str)
        if not isinstance(_dict, dict):
            _dict = {
                "scenario": "none",
                "action": "none",
                "entities": [],
            }
            invalid = True
    except Exception:  # need this if the output is not a valid dict
        _dict = {
            "scenario": "none",
            "action": "none",
            "entities": [],
        }
        invalid = True

    if "scenario" not in _dict or not isinstance(_dict["scenario"], str):
        _dict["scenario"] = "none"
        invalid = True
    if "action" not in _dict or not isinstance(_dict["action"], str):
        _dict["action"] = "none"
        invalid = True
    if "entities" not in _dict:
        _dict["entities"] = []
        invalid = True
    else:

        def _parse_entity(item: Dict):
            error = False
            for key in ["type", "filler"]:
                if key not in item or not isinstance(item[key], str):
                    item[key] = "none"
                    error = True
            return item, error

        for i, x in enumerate(_dict["entities"]):
            item, entity_error = _parse_entity(x)
            invalid = invalid or entity_error
            _dict["entities"][i] = item

    return _dict, invalid


class SLURPEvaluator:
    """
    Evaluator class for calculating SLURP metrics
    """

    def __init__(self, average_mode: str = 'micro') -> None:
        if average_mode not in ['micro', 'macro']:
            raise ValueError(f"Only supports 'micro' or 'macro' average, but got {average_mode} instead.")
        self.average_mode = average_mode
        self.scenario_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.action_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.intent_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.span_f1 = ErrorMetric.get_instance(metric="span_f1", average=average_mode)
        self.distance_metrics = {}
        for distance in ['word', 'char']:
            self.distance_metrics[distance] = ErrorMetric.get_instance(
                metric="span_distance_f1", average=average_mode, distance=distance
            )
        self.slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=average_mode)
        self.invalid = 0
        self.total = 0

    def reset(self):
        self.scenario_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.action_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.intent_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.span_f1 = ErrorMetric.get_instance(metric="span_f1", average=self.average_mode)
        self.distance_metrics = {}
        for distance in ['word', 'char']:
            self.distance_metrics[distance] = ErrorMetric.get_instance(
                metric="span_distance_f1", average=self.average_mode, distance=distance
            )
        self.slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=self.average_mode)
        self.invalid = 0
        self.total = 0

    def update(self, predictions: Union[List[str], str], groundtruth: Union[List[str], str]) -> None:
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(groundtruth, str):
            groundtruth = [groundtruth]

        for pred, truth in zip(predictions, groundtruth):
            pred, syntax_error = parse_semantics_str2dict(pred)
            truth, _ = parse_semantics_str2dict(truth)
            self.scenario_f1(truth["scenario"], pred["scenario"])
            self.action_f1(truth["action"], pred["action"])
            self.intent_f1(f"{truth['scenario']}_{truth['action']}", f"{pred['scenario']}_{pred['action']}")
            self.span_f1(truth["entities"], pred["entities"])
            for distance, metric in self.distance_metrics.items():
                metric(truth["entities"], pred["entities"])

            self.total += 1
            self.invalid += int(syntax_error)

    def compute(self, aggregate=True) -> Dict:
        scenario_results = self.scenario_f1.get_metric()
        action_results = self.action_f1.get_metric()
        intent_results = self.intent_f1.get_metric()
        entity_results = self.span_f1.get_metric()
        word_dist_results = self.distance_metrics['word'].get_metric()
        char_dist_results = self.distance_metrics['char'].get_metric()
        self.slu_f1(word_dist_results)
        self.slu_f1(char_dist_results)
        slurp_results = self.slu_f1.get_metric()

        if not aggregate:
            return {
                "scenario": scenario_results,
                "action": action_results,
                "intent": intent_results,
                "entity": entity_results,
                "word_dist": word_dist_results,
                "char_dist": char_dist_results,
                "slurp": slurp_results,
                "invalid": self.invalid,
                "total": self.total,
            }

        scores = dict()
        scores["invalid"] = self.invalid
        scores["total"] = self.total
        self.update_scores_dict(scenario_results, scores, "scenario")
        self.update_scores_dict(action_results, scores, "action")
        self.update_scores_dict(intent_results, scores, "intent")
        self.update_scores_dict(entity_results, scores, "entity")
        self.update_scores_dict(word_dist_results, scores, "word_dist")
        self.update_scores_dict(char_dist_results, scores, "char_dist")
        self.update_scores_dict(slurp_results, scores, "slurp")

        return scores

    def update_scores_dict(self, source: Dict, target: Dict, tag: str = '') -> Dict:
        scores = source['overall']
        p, r, f1 = scores[:3]
        target[f"{tag}_p"] = p
        target[f"{tag}_r"] = r
        target[f"{tag}_f1"] = f1
        return target
