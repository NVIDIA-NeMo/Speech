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

"""
Simplified MCQ (Multiple Choice Question) evaluator for openbookqa and mmsu datasets.

This code is based on the VoiceBench evaluation system:
- Repository: https://github.com/MatthewCYM/VoiceBench
- Paper: "VoiceBench: Benchmarking LLM-Based Voice Assistants" (arXiv:2410.17196)
  https://arxiv.org/abs/2410.17196
- Original License: Apache-2.0

Modifications for NeMo:
- Extracted MCQ evaluation logic only (specifically the answer extraction templates)
- Simplified for openbookqa and mmsu datasets
- Adapted to NeMo's logging and data structures
- Removed dependencies on VoiceBench's evaluation framework
"""

import json
import os
import random
from typing import Dict, List, Optional

from nemo.utils import logging


def load_jsonl(filepath: str) -> List[dict]:
    """Load JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def load_manifest(dataset_name: str, manifest_dir: str) -> Dict[str, str]:
    """
    Load manifest file and return a dictionary mapping id to answer.

    Args:
        dataset_name: Name of the dataset (e.g., 'openbookqa', 'mmsu')
        manifest_dir: Directory containing manifest files

    Returns:
        Dictionary mapping sample id to answer (e.g., {'0': 'A', '1': 'C', ...})
    """
    manifest_path = os.path.join(manifest_dir, dataset_name, f'{dataset_name}_manifest.jsonl')

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest_data = load_jsonl(manifest_path)
    answer_dict = {}

    for item in manifest_data:
        item_id = str(item['id'])
        answer = item.get('answer', '')
        answer_dict[item_id] = answer

    logging.info(f"Loaded {len(answer_dict)} answers from manifest: {manifest_path}")
    return answer_dict


def extract_key_from_id(item_id: str) -> str:
    """
    Extract the key from a sample ID by removing path prefixes and _dup suffixes.

    Args:
        item_id: Original sample ID (e.g., 'path/to/123_dup0')

    Returns:
        Cleaned ID (e.g., '123')
    """
    item_id = str(item_id)
    # Remove path and extension
    item_id = item_id.split('/')[-1].split('.')[0]

    # Remove _dup* suffix
    if "_dup" in item_id:
        item_id = item_id.split("_dup")[0]

    return item_id


def extract_answer(response: str) -> Optional[str]:
    """
    Extract the answer choice (A, B, C, or D) from the response text.

    This function uses various patterns to identify the selected answer.
    If extraction fails, returns None.

    Args:
        response: The model's text response

    Returns:
        Extracted answer ('A', 'B', 'C', or 'D') or None if extraction fails
    """
    if not response or response.strip() == '':
        return None

    response = response.lower()

    # Remove common prefixes
    if response.startswith('<1>') or response.startswith('<2>') or response.startswith('<3>'):
        response = response[3:].strip()

    # Define templates to search for
    templates = [
        "答案是[CHOICE]",
        "答案是 [CHOICE]",
        "答案是选项[CHOICE]",
        "答案应该是[CHOICE]",
        "answer is: **[CHOICE]",
        'answer is **[CHOICE]',
        "the answer to the question is: **[CHOICE]",
        "the answer to the multiple-choice question is **[CHOICE]",
        "the answer is '[CHOICE]'",
        '[CHOICE] is the best answer',
        'the answer is [CHOICE]',
        'the correct answer is [CHOICE]',
        'would select [CHOICE]',
        'would choose [CHOICE]',
        'would select option [CHOICE]',
        'would choose option [CHOICE]',
        'is \"[CHOICE]\"',
        "is **[CHOICE].",
        "is: **[CHOICE]",
        "be **[CHOICE]",
        "is: \n\n**[CHOICE]",
        'would be [CHOICE]',
        'would be option [CHOICE]',
        'is [CHOICE],',
        "option [CHOICE].",
        "option [CHOICE]:",
        "is [CHOICE]:",
        "is [CHOICE].",
        "is: [CHOICE]",
        "is ([CHOICE])",
        ":\n[CHOICE].",
        ":\n[CHOICE])",
        ":\n\n[CHOICE]",
        '([CHOICE]) would be',
        'is ([CHOICE]).',
        "is [CHOICE])",
        '(option [CHOICE])',
        'answer is ([CHOICE])',
        "is: [CHOICE]",
        "is **[CHOICE]**",
        " [CHOICE].",
        " [CHOICE],",
        " [CHOICE]:",
        " [CHOICE])",
        "**[CHOICE].",
        "**[CHOICE])",
        "([CHOICE])",
        "\"[CHOICE]\"",
    ]

    # Try to match templates
    for template in templates:
        for choice in ['a', 'b', 'c', 'd']:
            if template.replace('[CHOICE]', choice) in response:
                return choice.upper()

    # Try exact match or match with punctuation
    for choice in ['a', 'b', 'c', 'd']:
        if response == choice:
            return choice.upper()
        for punc in ['.', ',', ':', ')']:
            if response.startswith(choice + punc):
                return choice.upper()

    # No match found
    return None


class MCQEvaluator:
    """
    Evaluator for Multiple Choice Question datasets.
    """

    def __init__(self, manifest_dir: str):
        """
        Initialize the MCQ evaluator.

        Args:
            manifest_dir: Directory containing manifest files (e.g., .../eval-intelligence-main/manifest_files)
        """
        self.manifest_dir = manifest_dir
        self.answer_dicts = {}

    def load_dataset_manifest(self, dataset_name: str):
        """Load manifest for a specific dataset if not already loaded."""
        if dataset_name not in self.answer_dicts:
            self.answer_dicts[dataset_name] = load_manifest(dataset_name, self.manifest_dir)

    def evaluate(self, dataset_name: str, predictions: List[dict]) -> Dict[str, float]:
        """
        Evaluate predictions for a MCQ dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'openbookqa', 'mmsu')
            predictions: List of prediction dictionaries with keys:
                - 'id': sample ID
                - 'pred_text': predicted text response
                - 'target_text': (optional) ground truth answer, if not empty, use it instead of manifest

        Returns:
            Dictionary with evaluation metrics:
                - 'acc': accuracy (0-1) - includes empty responses in denominator
                - 'num_samples': total number of samples
                - 'num_correct': number of correct predictions
                - 'num_empty': number of empty responses
                - 'num_failed_extraction': number of failed answer extractions
        """
        # Only load manifest if needed (we'll check per-sample if target_text is empty)
        manifest_loaded = False
        answer_dict = None

        num_correct = 0
        num_empty = 0
        num_failed_extraction = 0
        num_samples = len(predictions)
        num_skipped = 0  # Samples with no ground truth available

        for pred in predictions:
            pred_text = pred['pred_text'].strip()

            # Check if response is empty
            if not pred_text:
                num_empty += 1
                # Empty responses are counted as incorrect (score = 0)
                # Continue to next sample, but this counts in the denominator
                continue

            # Extract answer from prediction
            extracted_answer = extract_answer(pred_text)

            if extracted_answer is None:
                num_failed_extraction += 1
                # If extraction fails, randomly select an answer (standard practice)
                extracted_answer = random.choice(['A', 'B', 'C', 'D'])

            # Get ground truth answer
            # First, check if target_text is provided and not empty
            target_text = pred.get('target_text', '').strip()

            if target_text:
                # Use target_text as ground truth (no need to look up manifest)
                ground_truth = target_text
            else:
                # target_text is empty, need to look up manifest
                if not manifest_loaded:
                    # Load manifest only when needed
                    try:
                        self.load_dataset_manifest(dataset_name)
                        answer_dict = self.answer_dicts[dataset_name]
                        manifest_loaded = True
                    except Exception as e:
                        logging.error(f"Failed to load manifest for {dataset_name}: {e}")
                        answer_dict = {}

                # Look up answer from manifest
                sample_id = extract_key_from_id(pred['id'])

                if sample_id not in answer_dict:
                    logging.warning(f"Sample ID {sample_id} not found in manifest for {dataset_name}")
                    num_skipped += 1
                    continue

                ground_truth = answer_dict[sample_id]

            # Check if correct
            if extracted_answer == ground_truth:
                num_correct += 1

        # Compute metrics
        # acc_with_empty: all samples (including empty) in denominator
        # This matches the original VoiceBench behavior
        acc = num_correct / num_samples if num_samples > 0 else 0.0
        empty_rate = num_empty / num_samples if num_samples > 0 else 0.0
        fail_rate = num_failed_extraction / num_samples if num_samples > 0 else 0.0

        metrics = {
            'acc': acc,  # This is acc_with_empty
            'num_samples': num_samples,
            'num_correct': num_correct,
            'num_empty': num_empty,
            'num_failed_extraction': num_failed_extraction,
            'num_skipped': num_skipped,
            'empty_rate': empty_rate,
            'fail_rate': fail_rate,
        }

        logging.info(
            f"MCQ Evaluation for {dataset_name}: "
            f"Accuracy={acc*100:.2f}% ({num_correct}/{num_samples}), "
            f"Empty={num_empty} ({empty_rate*100:.1f}%), "
            f"Failed extraction={num_failed_extraction} ({fail_rate*100:.1f}%), "
            f"Skipped={num_skipped}"
        )

        return metrics
