# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

from abc import ABC, abstractmethod
import numpy as np
import torch
from rouge_score import rouge_scorer
from sacrebleu import BLEU


class ValMetric(ABC):
    @abstractmethod
    def get_score(self, ground_truth, predicted_text):
        pass


class AccuracyScore(ValMetric):
    def get_score(self, ground_truth, predicted_text):
        corrects = 0
        for (pred, label) in zip(predicted_text, ground_truth):
            if pred == label:
                corrects += 1

        val_acc = corrects / len(ground_truth)
        return {'accuracy': torch.tensor(val_acc)}


class BLEUScore(ValMetric):
    def __init__(self):
        self.scorer = BLEU()

    def get_score(self, ground_truth, predicted_text):
        return {
            'bleu_score': torch.tensor(self.scorer.corpus_score(predicted_text, [[i] for i in ground_truth],).score)
        }


class ROUGEScores(ValMetric):
    def __init__(self):
        self.rscorers = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)

    def get_score(self, ground_truth, predicted_text):
        all_rouges = []
        for i in range(len(ground_truth)):
            scores = self.rscorers.score(predicted_text[i], ground_truth[i])
            all_rouges.append(
                [
                    scores['rouge1'].fmeasure,
                    scores['rouge2'].fmeasure,
                    scores['rouge3'].fmeasure,
                    scores['rougeL'].fmeasure,
                ]
            )

        all_rouges = np.mean(np.array(all_rouges), axis=0).tolist()

        return {
            'rouge_1_score': torch.tensor(all_rouges[0]),
            'rouge_2_score': torch.tensor(all_rouges[1]),
            'rouge_3_score': torch.tensor(all_rouges[2]),
            'rouge_L_score': torch.tensor(all_rouges[3]),
        }
