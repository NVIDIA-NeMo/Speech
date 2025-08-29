"""Tests the round-trip conversion of LlavaNext models: HF → NeMo → HF."""

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

'''
python tests/collections/vlm/llava_next/test_hf_conversion.py
'''
import argparse
import os
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlavaNextForConditionalGeneration

from nemo.collections import llm, vlm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-path",
        type=str,
        default="llava-hf/llava-v1.6-vicuna-7b-hf",
        help="Path or HF model ID of the original HF model",
    )
    parser.add_argument("--model-type", type=str, default="LlavaNextModel", help="Name of the NeMo model class")
    parser.add_argument("--model-config", type=str, default="LlavaNextConfig7B", help="Name of the NeMo model config")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save intermediate and final outputs"
    )
    parser.add_argument("--compare-logits", action="store_true", help="Compare model logits on a test prompt")
    parser.add_argument(
        "--save-models", action="store_true", help="Save the intermediate NeMo model and re-exported HF model"
    )
    return parser


def compare_parameters(model1, model2, label1="Model1", label2="Model2"):
    """Compare parameters between two models and report differences."""
    print(f"\nComparing parameters between {label1} and {label2}...")
    matched = 0
    mismatched = 0
    missing = 0
    vocab_size_diff = 0  # Count parameters with vocabulary size differences separately

    # Create dictionaries of parameters for easier lookup
    params1 = {name: param for name, param in model1.named_parameters()}
    params2 = {name: param for name, param in model2.named_parameters()}

    all_keys = set(params1.keys()).union(set(params2.keys()))

    # Parameters to ignore shape differences due to vocabulary size padding
    vocab_size_params = ['lm_head.weight', 'model.language_model.embed_tokens.weight']

    # Print summary of parameter counts
    print(f"{label1} parameters: {len(params1)}")
    print(f"{label2} parameters: {len(params2)}")
    print(f"Total unique parameters: {len(all_keys)}")
    print(f"NOTE: Vocabulary size differences in {vocab_size_params} are expected and will be ignored")

    for key in sorted(all_keys):
        if key not in params1:
            print(f"Parameter {key} missing from {label1}")
            missing += 1
            continue

        if key not in params2:
            print(f"Parameter {key} missing from {label2}")
            missing += 1
            continue

        # Compare the parameters
        if params1[key].shape != params2[key].shape:
            if key in vocab_size_params:
                # Handle vocabulary size differences differently
                print(f"Expected vocabulary size difference in {key}: {params1[key].shape} vs {params2[key].shape}")
                vocab_size_diff += 1
            else:
                print(f"Shape mismatch for {key}: {params1[key].shape} vs {params2[key].shape}")
                mismatched += 1
            continue

        # Check if values are equal (with some tolerance for float precision)
        if not torch.allclose(params1[key], params2[key], rtol=1e-4, atol=1e-4):
            print(f"Value mismatch for {key}")
            mismatched += 1
        else:
            matched += 1

    results = {
        "matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "vocab_size_diff": vocab_size_diff,
        "total": len(all_keys),
    }

    print(f"\nComparison Results:")
    print(
        f"  Matched parameters: {results['matched']}/{results['total']} ({results['matched']/results['total']*100:.2f}%)"
    )
    print(f"  Expected vocabulary size differences: {results['vocab_size_diff']}")
    print(f"  Mismatched parameters (excluding vocab size): {results['mismatched']}")
    print(f"  Missing parameters: {results['missing']}")

    return results


def compare_model_outputs(original_model, exported_model, tokenizer):
    """Compare the outputs of the original and exported models on a simple prompt."""
    print("\nComparing model outputs on a test prompt...")

    # Define a test prompt with image placeholder
    test_prompt = "<image>\nWhat do you see in this image?"

    # Create a dummy image tensor
    dummy_image = torch.rand(3, 224, 224)

    # Tokenize the input
    inputs = tokenizer(text=test_prompt, return_tensors="pt")

    # Get original model output
    with torch.no_grad():
        original_outputs = original_model(**inputs, pixel_values=dummy_image.unsqueeze(0))

    # Get exported model output
    with torch.no_grad():
        exported_outputs = exported_model(**inputs, pixel_values=dummy_image.unsqueeze(0))

    # Compare logits
    logits_match = torch.allclose(original_outputs.logits, exported_outputs.logits, rtol=1e-4, atol=1e-4)

    print(f"Logits match: {logits_match}")

    if not logits_match:
        # Print some statistics on the differences
        diff = (original_outputs.logits - exported_outputs.logits).abs()
        print(f"  Max absolute difference: {diff.max().item()}")
        print(f"  Mean absolute difference: {diff.mean().item()}")

    return logits_match


def run_conversion_pipeline(args):
    """Run the full HF → NeMo → HF conversion pipeline and validate results."""
    # Setup paths
    temp_dir = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        nemo_output_path = os.path.join(args.output_dir, "nemo_model")
        hf_output_path = os.path.join(args.output_dir, "hf_model_reexported")
    else:
        # Use temporary directories if no output path specified
        temp_dir = tempfile.mkdtemp()
        nemo_output_path = os.path.join(temp_dir, "nemo_model")
        hf_output_path = os.path.join(temp_dir, "hf_model_reexported")

    print(f"Using paths:")
    print(f"  Original HF model: {args.hf_path}")
    print(f"  Intermediate NeMo model: {nemo_output_path}")
    print(f"  Re-exported HF model: {hf_output_path}")

    # STEP 1: Import HF model to NeMo format
    print(f"\n=== STEP 1: Importing HF model to NeMo format ===")

    # Setup NeMo model for conversion
    config = getattr(vlm, args.model_config)()
    model = getattr(vlm, args.model_type)(config=config)

    # Convert HF model to NeMo
    print(f"Converting model from {args.hf_path} to NeMo format at {nemo_output_path}")
    nemo_path = llm.import_ckpt(
        model=model,
        source=f"hf://{args.hf_path}",
        output_path=nemo_output_path,
        overwrite=True,
    )

    # STEP 2: Export NeMo model back to HF format
    print(f"\n=== STEP 2: Exporting NeMo model back to HF format ===")
    print(f"Exporting NeMo model from {nemo_path} to HF format at {hf_output_path}")

    llm.export_ckpt(
        path=Path(nemo_path),
        target='hf',
        output_path=Path(hf_output_path),
        overwrite=True,
    )

    # STEP 3: Validate the round-trip conversion
    print(f"\n=== STEP 3: Validating round-trip conversion ===")

    # Load the original HF model
    print(f"Loading original HF model from {args.hf_path}")
    original_hf = LlavaNextForConditionalGeneration.from_pretrained(args.hf_path, trust_remote_code=True)

    # Load the re-exported HF model
    print(f"Loading re-exported HF model from {hf_output_path}")
    reexported_hf = LlavaNextForConditionalGeneration.from_pretrained(hf_output_path, trust_remote_code=True)

    # Compare parameters
    results = compare_parameters(original_hf, reexported_hf, "Original HF", "Re-exported HF")

    # Optionally compare model outputs
    if args.compare_logits:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_path)
        logits_match = compare_model_outputs(original_hf, reexported_hf, tokenizer)
        print(f"Logits match: {logits_match}")

    # Clean up if not saving models
    if not args.save_models and not args.output_dir:
        print(f"\nCleaning up temporary directories...")
        import shutil

        if temp_dir:
            shutil.rmtree(temp_dir)

    # Final validation (modified to exclude vocab size differences)
    if results['mismatched'] == 0 and results['missing'] == 0:
        print(
            '\nRound-trip conversion successful! All weights matched (ignoring expected vocabulary size differences).'
        )
        return True
    else:
        print('\nRound-trip conversion had issues. See details above.')
        return False


if __name__ == '__main__':
    args = get_parser().parse_args()
    success = run_conversion_pipeline(args)
    if not success:
        exit(1)  # Signal failure to calling process
