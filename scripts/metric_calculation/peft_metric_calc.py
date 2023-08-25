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

import argparse
import json
import re
import string
from collections import Counter
from rouge_score import rouge_scorer


"""
This script can be used to calcualte exact match and F1 scores for many different tasks, not just squad. 

Example command for T5 Preds

    ```
    python squad_metric_calc.py \
        --ground-truth squad_test_gt.jsonl \
        --preds squad_preds_t5.txt
    ```

Example command for GPT Preds

    ```
    python squad_metric_calc.py \
        --ground-truth squad_test_gt.jsonl \
        --preds squad_preds_gpt.txt \
        --split-string "answer:"
    ```

    In this case, the prediction file will be split on "answer: " when looking for the LM's predicted answer. 

"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--ground-truth',
        type=str,
        help="ground truth .jsonl file made from /NeMo/scripts/dataset_processing/nlp/squad/prompt_learning_squad_preprocessing.py",
    )
    parser.add_argument(
        '--preds',
        type=str,
        help="Text file with test set prompts + model predictions. Prediction file can be made by running NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning_eval.py",
    )
    parser.add_argument(
        '--split-string',
        type=str,
        help="The text at the end of the prompt, write before the predicted answer. This will be used to find the model's predictions in pred files when the pred file containers both the prompt and prediction.",
        default=None,
    )  # If the pred file only has preditions, just pass none
    parser.add_argument(
        '--answer-field',
        type=str,
        help="The field in the json file that contains the ground truth tokens",
        default="answer",
    )

    args = parser.parse_args()

    ground_truth_file = args.ground_truth
    pred_file = args.preds
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    preds = open(pred_file, encoding="utf-8").readlines()
    ground_truth = open(ground_truth_file).readlines()
    f1 = exact_match = total = r_score = 0

    for i in range(len(preds)):
        truth = json.loads(ground_truth[i])
        pred_answer = json.loads(preds[i])

        # Need to separate out preditions from prompt, spliting on the provided "split string"
        if args.split_string is not None:
            pred_answer = pred_answer["sentence"].split(args.split_string)[-1].strip()

        true_answers = truth[args.answer_field]
        if not isinstance(true_answers, list):
            true_answers = [true_answers]

        r_scores = []
        for ta in true_answers:
            r_scores.append(scorer.score(ta, pred_answer)['rougeL'].fmeasure)
        r_score += max(r_scores)
        exact_match += metric_max_over_ground_truths(exact_match_score, pred_answer, true_answers)
        f1 += metric_max_over_ground_truths(f1_score, pred_answer, true_answers)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    r_score = 100 * (r_score / total)
    res = {'exact_match': exact_match, 'f1': f1, "rougeL": r_score, 'total': total}
    print('\t'.join([f"{k} {v:.3f}" for k, v in res.items()]))


if __name__ == "__main__":
    main()
