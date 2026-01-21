"""
NeMo ASR Evaluation Script for HuggingFace Datasets

This script downloads HuggingFace audio datasets and evaluates NeMo ASR models.
Similar to the Whisper evaluation pipeline but adapted for NeMo models.

Usage:
    python run_nemo_eval.py \
        --model_path=/path/to/model.nemo \
        --dataset_path=hf-audio/esb-datasets-test-only-sorted \
        --dataset=librispeech \
        --split=test.clean \
        --batch_size=16 \
        --output_dir=./results \
        --eval_mode=standard
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import lightning.pytorch as pl
from datasets import load_dataset, Audio
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging


def load_data(args):
    """Load dataset from HuggingFace."""
    logging.info(f"Loading dataset: {args.dataset_path}/{args.dataset} split={args.split}")
    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )
    return dataset


def prepare_data(dataset):
    """Resample audio to 16kHz."""
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def evaluate_nemo_model(args):
    """Main evaluation function for NeMo ASR models."""
    
    # Load model
    logging.info(f"Loading NeMo model from: {args.model_path}")
    if args.model_path.endswith(".nemo"):
        asr_model = ASRModel.restore_from(restore_path=args.model_path, map_location=f"cuda:{args.cuda}")
    elif args.model_path.endswith(".ckpt"):
        asr_model = ASRModel.load_from_checkpoint(checkpoint_path=args.model_path, map_location=f"cuda:{args.cuda}")
    else:
        asr_model = ASRModel.from_pretrained(model_name=args.model_path, map_location=f"cuda:{args.cuda}")
    
    asr_model = asr_model.eval()
    device = next(asr_model.parameters()).device
    logging.info(f"Model loaded on device: {device}")
    
    # Load dataset
    dataset = load_data(args)
    
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        logging.info(f"Limiting evaluation to {args.max_eval_samples} samples")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    
    dataset = prepare_data(dataset)
    
    # Collect results
    all_predictions = []
    all_references = []
    all_audio_lengths = []
    all_transcription_times = []
    
    # Process in batches
    batch_audio = []
    batch_refs = []
    batch_durations = []
    processed_samples = 0
    
    logging.info("Starting evaluation...")
    
    for sample in tqdm(dataset, desc=f"Evaluating {args.dataset}"):
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        
        # Calculate audio duration
        duration = len(audio_array) / sampling_rate
        
        # Get reference text
        ref_text = sample.get("text", sample.get("sentence", ""))
        
        batch_audio.append(audio_array)
        batch_refs.append(ref_text)
        batch_durations.append(duration)
        
        # Process batch when full or at end of dataset
        if len(batch_audio) >= args.batch_size:
            # Transcribe batch
            start_time = time.time()
            
            with torch.amp.autocast('cuda', enabled=args.amp):
                with torch.no_grad():
                    predictions = asr_model.transcribe(
                        audio=batch_audio,
                        batch_size=args.batch_size,
                    )
            
            transcription_time = time.time() - start_time
            per_sample_time = transcription_time / len(batch_audio)
            
            # Handle predictions (may be tuple of (best, all) or list)
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # Take best hypotheses
            
            # Store results
            all_predictions.extend(predictions)
            all_references.extend(batch_refs)
            all_audio_lengths.extend(batch_durations)
            all_transcription_times.extend([per_sample_time] * len(batch_audio))
            
            processed_samples += len(batch_audio)
            
            # Clear batch
            batch_audio = []
            batch_refs = []
            batch_durations = []
    
    # Process remaining samples
    if len(batch_audio) > 0:
        start_time = time.time()
        
        with torch.amp.autocast('cuda', enabled=args.amp):
            with torch.no_grad():
                predictions = asr_model.transcribe(
                    audio=batch_audio,
                    batch_size=args.batch_size,
                )
        
        transcription_time = time.time() - start_time
        per_sample_time = transcription_time / len(batch_audio)
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        all_predictions.extend(predictions)
        all_references.extend(batch_refs)
        all_audio_lengths.extend(batch_durations)
        all_transcription_times.extend([per_sample_time] * len(batch_audio))
        
        processed_samples += len(batch_audio)
    
    logging.info(f"Processed {processed_samples} samples")
    
    # Calculate metrics
    wer = word_error_rate(hypotheses=all_predictions, references=all_references, use_cer=args.use_cer)
    metric_name = "CER" if args.use_cer else "WER"
    
    total_audio_duration = sum(all_audio_lengths)
    total_transcription_time = sum(all_transcription_times)
    rtfx = total_transcription_time / total_audio_duration if total_audio_duration > 0 else 0
    
    logging.info(f"{metric_name}: {wer * 100:.2f}%")
    logging.info(f"RTFx: {rtfx:.4f}")
    logging.info(f"Total audio duration: {total_audio_duration:.2f}s")
    logging.info(f"Total transcription time: {total_transcription_time:.2f}s")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model name for output file
    if args.model_path.endswith((".nemo", ".ckpt")):
        model_name = Path(args.model_path).stem
    else:
        model_name = args.model_path.replace("/", "_")
    
    output_file = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{args.split.replace('.', '_')}.json"
    )
    
    with open(output_file, "w") as f:
        for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
            record = {
                "prediction": pred,
                "reference": ref,
                "audio_length_s": all_audio_lengths[i],
                "transcription_time_s": all_transcription_times[i],
            }
            f.write(json.dumps(record) + '\n')
    
    logging.info(f"Results saved to: {output_file}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"summary_{args.dataset}_{args.split.replace('.', '_')}.json")
    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": processed_samples,
        metric_name.lower(): round(wer * 100, 2),
        "rtfx": round(rtfx, 4),
        "total_audio_duration_s": round(total_audio_duration, 2),
        "total_transcription_time_s": round(total_transcription_time, 2),
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Summary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeMo ASR models on HuggingFace datasets")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .nemo model or pretrained model name")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted",
                        help="HuggingFace dataset path")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., librispeech, tedlium)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (e.g., test, test.clean)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large datasets")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Maximum number of samples to evaluate (-1 for all)")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA device number")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--use_cer", action="store_true",
                        help="Use CER instead of WER")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.setLevel(logging.INFO)
    
    # Run evaluation
    evaluate_nemo_model(args)


if __name__ == "__main__":
    main()
