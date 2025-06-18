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

import operator
from collections import defaultdict
from functools import partial
from typing import List, Optional

import regex as re
import torch
from lhotse import CutSet
from lhotse.cut import MixedCut
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch
from nemo.collections.asr.metrics.wer import WER
from nemo.core.classes import Serialization

__all__ = ['MultiTaskMetric']


# Helper functions for managing constraint criteria on metrics.
class ConstraintParser:
    """Boolean Parser class for constraint passing in config"""

    _primitives = None
    _booleans = None

    def parse_constraint(self, constraint: str):
        array = re.sub(r"([()])", r" \1 ", constraint).strip().split()  # Add space only for keywords.
        if not array:
            return self._no_constraint

        self._resolve_primitives(array)
        if len(array) == 1:
            return array[0]

        # Basic nested list parser. Starts from tail to aid readibility in subfunction.
        stack = []
        array = ["("] + array + [")"]
        while array:
            if (c := array.pop()) == "(":
                expr = []
                while stack:
                    if (e := stack.pop()) == ")":
                        if not (fnc := self._resolve_bools(expr)):
                            raise SyntaxError(f"Malformed subexpression find in constraint parsing: {fnc}")
                        stack.append(fnc)
                        break
                    expr.append(e)
            else:
                stack.append(c)
        if not (fnc := self._resolve_bools(stack)):
            raise SyntaxError(f"Parser cannot resolve constraint: {constraint}")
        return fnc

    @property
    def primitives(self):
        if self._primitives is None:
            self._primitives = {
                "==": operator.eq,
                "!=": operator.ne,
            }
        return self._primitives

    @property
    def booleans(self):
        if self._booleans is None:
            self._booleans = {
                "and": self._logical_and,
                "or": self._logical_or,
                "xor": self._logical_xor,
            }
        return self._booleans

    @staticmethod
    def _logical_not(expr, properties):
        if not expr:
            raise ValueError(f"Malformed subexpression find in 'not' constraint parsing: {expr}")
        return not expr(properties)

    @staticmethod
    def _logical_and(l_expr, r_expr, properties):
        if not (l_expr and r_expr):
            raise ValueError(f"Malformed subexpression find in 'and' constraint parsing: {l_expr} and {r_expr}")
        return l_expr(properties) and r_expr(properties)

    @staticmethod
    def _logical_or(l_expr, r_expr, properties):
        if not (l_expr and r_expr):
            raise ValueError(f"Malformed subexpression find in 'or' constraint parsing: {l_expr} or {r_expr}")
        return l_expr(properties) or r_expr(properties)

    @staticmethod
    def _logical_xor(l_expr, r_expr, properties):
        if not (l_expr and r_expr):
            raise ValueError(f"Malformed subexpression find in 'xor' constraint parsing: {l_expr} xor {r_expr}")
        return l_expr(properties) ^ r_expr(properties)

    @staticmethod
    def _no_constraint(properties):
        return True

    @staticmethod
    def _static_constraint(fnc, key, val, properties):
        return fnc(val, properties.get(key))

    @staticmethod
    def _compare_constraint(fnc, key1, key2, properties):
        return (
            (prop_val1 := properties.get(key1)) is not None
            and (prop_val2 := properties.get(key2)) is not None
            and fnc(prop_val1, prop_val2)
        )

    def _resolve_primitives(self, constraint):
        for idx, c in enumerate(constraint):
            for n, o in self.primitives.items():
                # Check if string is for value assertion or equivalency of values.
                entail, equal = fr'\.(\S+)\s*{n}\s*(\S+)', fr'\.(\S+)\s*{n}\s*\.(\S+)'
                match_entail, match_equal = re.match(entail, c), re.match(equal, c)
                if match_equal:
                    key1, key2 = match_equal.groups()
                    constraint[idx] = partial(self._compare_constraint, o, key1, key2)
                elif match_entail:
                    key1, val = match_entail.groups()
                    constraint[idx] = partial(self._static_constraint, o, key1, val)
                else:
                    pass

    def _resolve_bools(self, constraint: List[str]):
        idx = 0
        stack = []
        while idx < len(constraint):
            c = constraint[idx]
            if c == "not":
                c = partial(self._logical_not, constraint[idx + 1])
                idx += 1  # Skip so don't see the character again.
            stack.append(c)
            idx += 1

        constraint = stack
        for n, o in self.booleans.items():
            idx = 0
            stack = []
            while idx < len(constraint):
                c = constraint[idx]
                if c == n:
                    c = partial(o, stack.pop(), constraint[idx + 1])
                    idx += 1
                stack.append(c)
                idx += 1
            constraint = stack
        if len(constraint) > 1:  # More than one constraint, something went wrong.
            return None

        return constraint[0]


class MultiTaskMetric(Serialization):
    """
    Wrapper class for managing multiple metrics in multitask ASR/NLP models.

    This class enables conditional metric computation based on sample properties stored in Lhotse cuts.
    It's primarily designed for `EncDecMultiTaskModel` but can support any model with a prompt schema.

    Key Features:
        1. **Automatic Model Integration**: Instantiated metrics are automatically added as attributes
           to the parent model, enabling seamless integration with existing logging infrastructure.

        2. **Conditional Metric Updates**: Only samples meeting specific constraints are passed to
           each metric, avoiding inappropriate metric calculations (e.g., WER for translation tasks).

        3. **Flexible Constraint System**: Supports complex logical expressions for determining
           when metrics should be applied to samples.

        4. **Configuration Inheritance**: Global configuration parameters are automatically
           inherited by all metrics unless explicitly overridden.

    Args:
        model (nn.Module): Parent model that will receive metric instances as attributes.
                          Must have a `decoding` attribute for metrics that require decoding.
        cfg (DictConfig): Configuration dictionary containing metric definitions and constraints.

    Configuration Format:
        The configuration should follow this structure:

        ``'
        # Global parameters (inherited by all metrics unless overridden)
        log_predictions: true
        batch_dim_index: 0

        # Metric definitions
        metrics:
            wer:
                _target_: nemo.collections.asr.metrics.WER  # Metric class to instantiate
                constraint: ".task == transcribe"           # When to apply this metric
                use_cer: false                              # Metric-specific parameters
            bleu:
                _target_: nemo.collections.asr.metrics.BLEU
                constraint: ".task == translate"
                bleu_tokenizer: "13a"
                n_gram: 4

        ```

    Constraint Syntax:
        Constraints are evaluated against the `custom` dictionary of Lhotse cuts:

        - **Custom attribute Access**: `.task`, `.lang`, `.domain`
        - **Comparisons**: `==`, `!=`
        - **Logical Operations**: `and`, `or`, `not`, `xor`
        - **Property Comparisons**: `.source_lang == .target_lang`

        Examples:
        - `".task == transcribe"` - Apply to transcription tasks
        - `".task == translate and .source_lang != .target_lang"` - Cross-lingual translation
        - `"not .task == other"` - Apply to all tasks except 'other'
        - `".domain == medical or .domain == legal"` - Specific domains

    Usage Example:
        ```python
        # In model initialization
        if hasattr(cfg, 'multitask_metrics'):
            self.multitask_metrics = MultiTaskMetric(self, cfg.multitask_metrics)

        # During training/validation
        if hasattr(self, 'multitask_metrics'):
            metrics = self.multitask_metrics.eval(
                batch=batch,
                predictions=predictions,
                predictions_lengths=pred_lengths,
                predictions_mask=pred_mask,
                prefix="val",
                return_all_metrics=True
            )
            self.log_dict(metrics)
        ```

    Note:
        - Each metric receives the model's `decoding` instance for text decoding operations
        - Metrics are automatically instantiated for the parent model as attributes (e.g., `model.wer`, `model.bleu`)
        - Global configuration parameters are inherited unless explicitly overridden per metric
        - Metrics defined without 'constraint' keyword are called on every prediction sample
        - Empty batches (no samples matching constraints) are handled by child metrics.
    """

    def __init__(self, model: nn.Module, cfg: DictConfig):
        """
        Initialize MultiTaskMetric with model and configuration.

        Args:
            model (nn.Module): Parent model that will contain metric instances
            cfg (DictConfig): Configuration containing metric definitions
        """
        super().__init__()

        # Setup tracking dictionaries
        self._metric_dict, self._constr_dict = {}, {}
        cfg = OmegaConf.to_container(cfg)

        # Process each metric instance.
        parser = ConstraintParser()
        seen_types = set()
        for name, metric_cfg in cfg.pop("metrics").items():
            constraint = metric_cfg.pop(
                "constraint", ""
            )  # Empty string for no constraint value. Metric always calculated.

            # Inherit global configuration parameters
            for k, v in cfg.items():
                if k not in metric_cfg:  # do not override explicit metric values
                    metric_cfg[k] = v

            # Instantiates as instance of `model`. Avoids breaking behavior when other modules call specific metrics. (See `asr_model` for example.)
            metric_cfg["decoding"] = model.decoding  # For decoding reliant metrics like 'WER' or 'BLEU'
            metric = MultiTaskMetric.from_config_dict(metric_cfg)
            setattr(model, name, metric)

            # TODO: This is a from `asr_model` aggregation. To fix, update metric classes to support custom naming
            # and update `asr_model` `multi_{validation,test}_epoch_end` to support metric aggregation with custom names.
            metric_type = type(metric)
            if metric_type in seen_types:
                raise TypeError(
                    "MultiTaskMetric currently only supports one instance of each metric class. Please check your configs for duplicates values of `_target_` entry."
                )
            seen_types.add(metric_type)

            # Store metric and its constraint function
            self._metric_dict[name] = metric
            self._constr_dict[name] = parser.parse_constraint(constraint)

    # Performs full PyMetrics validation loop for all metrics.
    def eval(
        self,
        batch: PromptedAudioToTextMiniBatch,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        predictions_mask: torch.Tensor,
        return_all_metrics: Optional[bool] = False,
        prefix: Optional[str] = None,
    ):
        metric_dict = {}
        self.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_mask=predictions_mask,
            input_ids=getattr(batch, "prompt", None),  # Allows for CTC and RNN-T support.
            cuts=batch.cuts,
        )
        metric_dict.update(
            self.compute(
                prefix=f"{prefix}_" if prefix else "",
                return_all_metrics=return_all_metrics,
            )
        )
        self.reset()
        return metric_dict

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        predictions_mask: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        cuts: CutSet,
    ):

        # Update each metric with its filtered data
        cuts_split, idx_split = self._split_cuts(cuts)
        for name, metric in self._metric_dict.items():
            cuts_subset, indices = cuts_split[name], idx_split[name]
            # Update metric with filtered tensors
            metric.update(
                predictions=predictions[indices],
                predictions_lengths=predictions_lengths[indices],
                predictions_mask=predictions_mask[indices],
                targets=targets[indices],
                targets_lengths=targets_lengths[indices],
                input_ids=input_ids[indices],
                cuts=cuts_subset,
            )

    def compute(self, return_all_metrics=False, prefix=""):
        output_dict = {}

        for name, metric in self._metric_dict.items():
            # Handle WER metric's special return format
            # Custom name of metric used as suffix to allow custom naming.
            # TODO: Standardize WER to return dict like other metrics
            if type(metric) is WER:
                wer, wer_num, wer_denom = metric.compute()
                if return_all_metrics:
                    output_dict.update(
                        {
                            f"{prefix}wer": wer,
                            f"{prefix}wer_num": wer_num,
                            f"{prefix}wer_denom": wer_denom,
                        }
                    )
                else:
                    output_dict.update(
                        {
                            f"{prefix}wer": wer,
                        }
                    )
            else:
                # Standard metric compute (returns dict)
                output_dict.update(
                    metric.compute(
                        return_all_metrics=return_all_metrics,
                        prefix=prefix,
                    )
                )
        return output_dict

    def reset(self):
        {metric.reset() for name, metric in self._metric_dict.items()}

    def _split_cuts(self, cuts):
        """
        Split cuts based on metric constraints and return filtered subsets.

        This method evaluates each cut against all metric constraints and creates
        separate lists of cuts and indices for each metric.

        Args:
            cuts (CutSet): Input cuts containing sample metadata

        Returns:
            tuple: (cuts_splits, idx_splits) where:
                - cuts_splits (dict): Maps metric names to lists of matching cuts
                - idx_splits (dict): Maps metric names to lists of matching indices

        Note:
            - Handles both regular cuts and MixedCuts (uses first_non_padding_cut)
            - A single cut may match multiple metrics
            - Cuts not matching any constraints are ignored
        """
        cuts_splits, idx_splits = defaultdict(list), defaultdict(list)
        for idx, c in enumerate(cuts):
            c = c.first_non_padding_cut if isinstance(c, MixedCut) else c
            for metric, constr in self._constr_dict.items():
                if constr(c.custom):
                    cuts_splits[metric].append(c)
                    idx_splits[metric].append(idx)
        return cuts_splits, idx_splits
