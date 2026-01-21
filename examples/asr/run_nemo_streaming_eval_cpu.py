"""
NeMo Streaming ASR Evaluation Script for HuggingFace Datasets (CPU-only)

This script downloads HuggingFace audio datasets and evaluates NeMo cache-aware streaming models
using CPU inference only (no CUDA/GPU required).

Usage:
    python run_nemo_streaming_eval_cpu.py \
        --model_path=/path/to/streaming_model.nemo \
        --dataset_path=hf-audio/esb-datasets-test-only-sorted \
        --dataset=librispeech \
        --split=test.clean \
        --batch_size=1 \
        --output_dir=./results_cpu
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

# Force CPU mode - disable CUDA before any other imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['USE_CUDA'] = '0'
# Mock megatron to avoid CUDA JIT compilation
import sys
from unittest.mock import MagicMock
sys.modules['megatron'] = MagicMock()
sys.modules['megatron.core'] = MagicMock()
sys.modules['megatron.core.inference'] = MagicMock()
sys.modules['megatron.core.inference.contexts'] = MagicMock()
sys.modules['megatron.core.inference.unified_memory'] = MagicMock()

import torch
# Keep mkldnn/oneDNN enabled for performance (Intel optimized kernels)
# torch.backends.mkldnn.enabled = False
import numpy as np
import lightning.pytorch as pl
from datasets import load_dataset, Audio
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging

# Import whisper normalizer for text normalization before WER calculation
try:
    from whisper_normalizer.english import EnglishTextNormalizer
    NORMALIZER_AVAILABLE = True
    logging.info("Using EnglishTextNormalizer from whisper-normalizer package")
except ImportError:
    try:
        # Add open_asr_leaderboard to path
        import sys
        normalizer_path = "/datadisks/disk8/jiafa/accuracy/open_asr_leaderboard"
        if normalizer_path not in sys.path:
            sys.path.insert(0, normalizer_path)
        from normalizer import EnglishTextNormalizer
        NORMALIZER_AVAILABLE = True
        logging.info("Using EnglishTextNormalizer from open_asr_leaderboard/normalizer")
    except ImportError:
        NORMALIZER_AVAILABLE = False
        logging.warning("whisper-normalizer not available. Install with: pip install whisper-normalizer")


def extract_transcriptions(transcribed_texts):
    """Extract text from transcription hypotheses."""
    if isinstance(transcribed_texts[0], Hypothesis):
        transcriptions = [transcribed_texts[i].text for i in range(len(transcribed_texts))]
    else:
        transcriptions = transcribed_texts
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    """Calculate tokens to drop after downsampling."""
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def perform_streaming(
    asr_model,
    streaming_buffer,
    compute_dtype: torch.dtype,
    compare_vs_offline=False,
    pad_and_drop_preencoded=False,
):
    """Perform streaming inference on buffered audio."""
    batch_size = len(streaming_buffer.streams_length)
    
    offline_time = 0.0
    if compare_vs_offline:
        offline_start = time.time()
        with torch.inference_mode():
            processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
            processed_signal = processed_signal.to(compute_dtype)
            with torch.no_grad():
                (
                    pred_out_offline,
                    transcribed_texts,
                    cache_last_channel_next,
                    cache_last_time_next,
                    cache_last_channel_len,
                    best_hyp,
                ) = asr_model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    return_transcription=True,
                )
        offline_time = time.time() - offline_start
        final_offline_tran = extract_transcriptions(transcribed_texts)
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    
    streaming_start = time.time()
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        with torch.inference_mode():
            chunk_audio = chunk_audio.to(compute_dtype)
            with torch.no_grad():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = asr_model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded),
                    return_transcription=True,
                )
    streaming_time = time.time() - streaming_start

    final_streaming_tran = extract_transcriptions(transcribed_texts)

    return final_streaming_tran, final_offline_tran, streaming_time, offline_time


def load_data(args):
    """Load dataset from HuggingFace."""
    logging.info(f"Loading dataset: {args.dataset_path}/{args.dataset} split={args.split}")
    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=args.dataset_streaming,
        token=True,
    )
    return dataset


def prepare_data(dataset):
    """Resample audio to 16kHz."""
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def evaluate_streaming_model(args):
    """Main evaluation function for NeMo streaming ASR models."""
    
    # CPU-only device
    device = torch.device("cpu")
    map_location = device
    
    # Load model
    logging.info(f"Loading NeMo streaming model from: {args.model_path}")
    if args.model_path.endswith(".nemo"):
        asr_model = ASRModel.restore_from(restore_path=args.model_path, map_location=map_location)
    elif args.model_path.endswith(".ckpt"):
        asr_model = ASRModel.load_from_checkpoint(checkpoint_path=args.model_path, map_location=map_location)
    else:
        asr_model = ASRModel.from_pretrained(model_name=args.model_path, map_location=map_location)
    
    asr_model = asr_model.eval()
    logging.info(f"Model loaded on device: {device}")
    
    # CPU uses float32 only (no mixed precision)
    compute_dtype = torch.float32
    asr_model.to(compute_dtype)
    
    # Determine chunk_size and shift_size
    if hasattr(asr_model.encoder, 'streaming_cfg'):
        chunk_size = (
            args.chunk_size
            if args.chunk_size >= 0
            else asr_model.encoder.streaming_cfg.last_channel_cache_size
        )
        shift_size = args.shift_size if args.shift_size >= 0 else chunk_size
        logging.info(f"Using chunk_size={chunk_size} and shift_size={shift_size}")
        
        # Update the model's streaming configuration with user-specified values
        # Only call setup_streaming_params if chunk_size is explicitly provided
        if args.chunk_size >= 0:
            asr_model.encoder.setup_streaming_params(
                chunk_size=chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
            )
    else:
        if args.chunk_size < 0:
            raise ValueError("chunk_size must be specified for models without streaming_cfg")
        chunk_size = args.chunk_size
        shift_size = args.shift_size if args.shift_size >= 0 else chunk_size
        
        asr_model.encoder.setup_streaming_params(
            chunk_size=chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
        )

    # Setup online normalization
    online_normalization = args.online_normalization
    if hasattr(asr_model.encoder, 'streaming_cfg'):
        if hasattr(asr_model.encoder.streaming_cfg, 'norm_before_caching'):
            if asr_model.encoder.streaming_cfg.norm_before_caching:
                online_normalization = True
    
    # Create streaming buffer
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
    )
    
    # Load dataset
    if args.manifest_file:
        # Load from pre-built manifest file (no HF download needed)
        import soundfile as sf
        logging.info(f"Loading data from manifest: {args.manifest_file}")
        manifest_entries = []
        with open(args.manifest_file, 'r') as f:
            for line in f:
                if line.strip():
                    manifest_entries.append(json.loads(line))
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            manifest_entries = manifest_entries[:args.max_eval_samples]
        logging.info(f"Loaded {len(manifest_entries)} samples from manifest")
    else:
        if args.dataset is None:
            raise ValueError("Either --manifest_file or --dataset must be provided")
        dataset = load_data(args)

        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            logging.info(f"Limiting evaluation to {args.max_eval_samples} samples")
            if args.dataset_streaming:
                dataset = dataset.take(args.max_eval_samples)
            else:
                dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

        dataset = prepare_data(dataset)
    
    # Collect results
    all_streaming_predictions = []
    all_offline_predictions = []
    all_references = []
    all_audio_lengths = []
    all_streaming_times = []
    all_offline_times = []
    
    processed_samples = 0
    batch_size = args.batch_size
    
    # Save audio temporarily for streaming buffer
    import tempfile
    if not args.manifest_file:
        import soundfile as sf
    temp_dir = tempfile.mkdtemp()
    
    logging.info("Starting streaming evaluation...")
    
    batch_files = []
    batch_refs = []
    batch_durations = []
    
    start_time = time.time()
    
    # Build sample iterator from manifest or HF dataset
    if args.manifest_file:
        def _manifest_iter():
            for entry in manifest_entries:
                yield entry
        sample_iter = _manifest_iter()
        total_samples = len(manifest_entries)
        eval_desc = f"Evaluating manifest"
    else:
        sample_iter = None  # will use dataset directly below
        if args.max_eval_samples > 0:
            total_samples = args.max_eval_samples
        elif not args.dataset_streaming:
            total_samples = len(dataset)
        else:
            total_samples = None
        eval_desc = f"Evaluating {args.dataset}"
    
    if args.manifest_file:
        for sample_idx, entry in enumerate(tqdm(sample_iter, total=total_samples, desc=eval_desc)):
            audio_path = entry["audio_filepath"]
            # Resolve relative paths against working directory (manifest paths are relative to cwd)
            if not os.path.isabs(audio_path):
                audio_path = os.path.abspath(audio_path)
            ref_text = entry.get("text", "")
            duration = entry.get("duration", 0.0)

            streaming_buffer.append_audio_file(audio_path, stream_id=-1)

            batch_files.append(audio_path)
            batch_refs.append(ref_text)
            batch_durations.append(duration)

            is_last_sample = (sample_idx + 1) >= total_samples

            if (sample_idx + 1) % batch_size == 0 or is_last_sample:
                with torch.no_grad():
                    streaming_tran, offline_tran, streaming_time, offline_time = perform_streaming(
                        asr_model=asr_model,
                        streaming_buffer=streaming_buffer,
                        compute_dtype=compute_dtype,
                        compare_vs_offline=args.compare_vs_offline,
                        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                    )

                per_sample_streaming_time = streaming_time / len(streaming_tran)
                per_sample_offline_time = offline_time / len(streaming_tran) if offline_time > 0 else 0

                all_streaming_predictions.extend(streaming_tran)
                if args.compare_vs_offline:
                    all_offline_predictions.extend(offline_tran)
                all_references.extend(batch_refs)
                all_audio_lengths.extend(batch_durations)
                all_streaming_times.extend([per_sample_streaming_time] * len(streaming_tran))
                if args.compare_vs_offline:
                    all_offline_times.extend([per_sample_offline_time] * len(streaming_tran))

                processed_samples += len(streaming_tran)
                streaming_buffer.reset_buffer()
                batch_files = []
                batch_refs = []
                batch_durations = []
    else:
        for sample_idx, sample in enumerate(tqdm(dataset, desc=eval_desc)):
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
        
            # Calculate audio duration
            duration = len(audio_array) / sampling_rate
        
            # Get reference text
            ref_text = sample.get("text", sample.get("sentence", ""))
        
            # Save audio to temporary file for streaming buffer
            temp_file = os.path.join(temp_dir, f"temp_{sample_idx}.wav")
            sf.write(temp_file, audio_array, sampling_rate)
        
            # Add to streaming buffer - each call with stream_id=-1 creates a new independent stream
            streaming_buffer.append_audio_file(temp_file, stream_id=-1)
        
            batch_files.append(temp_file)
            batch_refs.append(ref_text)
            batch_durations.append(duration)
        
            # Check if we should process the batch
            is_last_sample = False
            if args.max_eval_samples > 0:
                is_last_sample = (sample_idx + 1) >= args.max_eval_samples
            elif total_samples is not None:
                is_last_sample = sample_idx == total_samples - 1
        
            # Process batch when full or at end
            if (sample_idx + 1) % batch_size == 0 or is_last_sample:
                with torch.no_grad():
                    streaming_tran, offline_tran, streaming_time, offline_time = perform_streaming(
                        asr_model=asr_model,
                        streaming_buffer=streaming_buffer,
                        compute_dtype=compute_dtype,
                        compare_vs_offline=args.compare_vs_offline,
                        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                    )
            
                per_sample_streaming_time = streaming_time / len(streaming_tran)
                per_sample_offline_time = offline_time / len(streaming_tran) if offline_time > 0 else 0
            
                all_streaming_predictions.extend(streaming_tran)
                if args.compare_vs_offline:
                    all_offline_predictions.extend(offline_tran)
                all_references.extend(batch_refs)
                all_audio_lengths.extend(batch_durations)
                all_streaming_times.extend([per_sample_streaming_time] * len(streaming_tran))
                if args.compare_vs_offline:
                    all_offline_times.extend([per_sample_offline_time] * len(streaming_tran))
            
                processed_samples += len(streaming_tran)

                # Clean up temp files
                for f in batch_files:
                    os.remove(f)
            
                # Reset buffer AFTER processing batch (like reference implementation)
                streaming_buffer.reset_buffer()
                batch_files = []
                batch_refs = []
                batch_durations = []
        
            # Limit evaluation if specified
            if args.max_eval_samples > 0 and processed_samples >= args.max_eval_samples:
                break
    
    # Process any remaining samples in the buffer after the loop ends
    if len(batch_files) > 0:
        logging.info(f"Processing final batch of {len(batch_files)} remaining samples")
        with torch.no_grad():
            streaming_tran, offline_tran, streaming_time, offline_time = perform_streaming(
                asr_model=asr_model,
                streaming_buffer=streaming_buffer,
                compute_dtype=compute_dtype,
                compare_vs_offline=args.compare_vs_offline,
                pad_and_drop_preencoded=args.pad_and_drop_preencoded,
            )
        
        per_sample_streaming_time = streaming_time / len(streaming_tran)
        per_sample_offline_time = offline_time / len(streaming_tran) if offline_time > 0 else 0
        
        all_streaming_predictions.extend(streaming_tran)
        if args.compare_vs_offline:
            all_offline_predictions.extend(offline_tran)
        all_references.extend(batch_refs)
        all_audio_lengths.extend(batch_durations)
        all_streaming_times.extend([per_sample_streaming_time] * len(streaming_tran))
        if args.compare_vs_offline:
            all_offline_times.extend([per_sample_offline_time] * len(streaming_tran))
        
        processed_samples += len(streaming_tran)
        
        # Clean up temp files
        for f in batch_files:
            os.remove(f)
        
        streaming_buffer.reset_buffer()
    
    total_time = time.time() - start_time
    
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    logging.info(f"Processed {processed_samples} samples in {total_time:.2f}s")
    
    # Normalize predictions and references for WER calculation
    if NORMALIZER_AVAILABLE and args.normalize_text and not args.use_cer:
        logging.info("Normalizing text with EnglishTextNormalizer before WER calculation...")
        normalizer = EnglishTextNormalizer()
        all_streaming_predictions_normalized = [normalizer(text) for text in all_streaming_predictions]
        all_references_normalized = [normalizer(text) for text in all_references]
        if args.compare_vs_offline and all_offline_predictions:
            all_offline_predictions_normalized = [normalizer(text) for text in all_offline_predictions]
    else:
        if args.normalize_text and not NORMALIZER_AVAILABLE:
            logging.warning("Text normalization requested but whisper-normalizer not available. Install with: pip install whisper-normalizer")
        all_streaming_predictions_normalized = all_streaming_predictions
        all_references_normalized = all_references
        if args.compare_vs_offline and all_offline_predictions:
            all_offline_predictions_normalized = all_offline_predictions
    
    # Calculate metrics
    streaming_wer = word_error_rate(hypotheses=all_streaming_predictions_normalized, references=all_references_normalized, use_cer=args.use_cer)
    metric_name = "CER" if args.use_cer else "WER"
    
    total_audio_duration = sum(all_audio_lengths)
    total_streaming_time = sum(all_streaming_times)
    total_offline_time = sum(all_offline_times) if all_offline_times else 0
    
    # RTFx = Real-Time Factor = audio_duration / processing_time
    # RTFx > 1 means faster than real-time (good!)
    # RTFx < 1 means slower than real-time (bad)
    streaming_rtfx = total_audio_duration / total_streaming_time if total_streaming_time > 0 else 0
    offline_rtfx = total_audio_duration / total_offline_time if total_offline_time > 0 else 0
    
    logging.info(f"Streaming {metric_name}: {streaming_wer * 100:.2f}%")
    logging.info(f"Streaming RTFx: {streaming_rtfx:.4f}x")
    
    if args.compare_vs_offline and all_offline_predictions:
        offline_wer = word_error_rate(hypotheses=all_offline_predictions_normalized, references=all_references_normalized, use_cer=args.use_cer)
        logging.info(f"Offline {metric_name}: {offline_wer * 100:.2f}%")
        logging.info(f"Offline RTFx: {offline_rtfx:.4f}x")
    
    logging.info(f"Total audio duration: {total_audio_duration:.2f}s")
    logging.info(f"Total streaming time: {total_streaming_time:.2f}s")
    if total_offline_time > 0:
        logging.info(f"Total offline time: {total_offline_time:.2f}s")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model name for output file
    if args.model_path.endswith((".nemo", ".ckpt")):
        model_name = Path(args.model_path).stem
    else:
        model_name = args.model_path.replace("/", "_")
    
    output_file = os.path.join(
        args.output_dir,
        f"streaming_{model_name}_{args.dataset}_{args.split.replace('.', '_')}.json"
    )
    
    with open(output_file, "w") as f:
        for i, pred in enumerate(all_streaming_predictions):
            record = {
                "streaming_prediction": pred,
                "reference": all_references[i],
                "audio_length_s": all_audio_lengths[i],
                "transcription_time_s": all_streaming_times[i],
            }
            if args.compare_vs_offline and i < len(all_offline_predictions):
                record["offline_prediction"] = all_offline_predictions[i]
            f.write(json.dumps(record) + '\n')
    
    logging.info(f"Results saved to: {output_file}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f"streaming_summary_{args.dataset}_{args.split.replace('.', '_')}.json")
    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": processed_samples,
        f"streaming_{metric_name.lower()}": round(streaming_wer * 100, 2),
        "streaming_rtfx": round(streaming_rtfx, 4),
        "total_audio_duration_s": round(total_audio_duration, 2),
        "total_streaming_time_s": round(total_streaming_time, 2),
        "chunk_size": chunk_size,
        "shift_size": shift_size,
    }
    
    if args.compare_vs_offline and all_offline_predictions:
        summary[f"offline_{metric_name.lower()}"] = round(offline_wer * 100, 2)
        summary["offline_rtfx"] = round(offline_rtfx, 4)
        summary["total_offline_time_s"] = round(total_offline_time, 2)
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Summary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeMo streaming ASR models on HuggingFace datasets")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .nemo streaming model or pretrained model name")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted",
                        help="HuggingFace dataset path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., librispeech, tedlium)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (e.g., test, test.clean)")
    parser.add_argument("--manifest_file", type=str, default=None,
                        help="Path to NeMo manifest file (JSON lines with audio_filepath, text, duration). Skips HF download.")
    parser.add_argument("--dataset_streaming", action="store_true",
                        help="Use HuggingFace streaming mode for large datasets (loads data on-the-fly)")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Maximum number of samples to evaluate (-1 for all)")
    
    # Streaming arguments
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="Chunk size for streaming (-1 to use model default)")
    parser.add_argument("--shift_size", type=int, default=-1,
                        help="Shift size for streaming (-1 to use chunk_size)")
    parser.add_argument("--left_chunks", type=int, default=2,
                        help="Left chunks for streaming (default: 2)")
    parser.add_argument("--online_normalization", action="store_true",
                        help="Use online normalization for streaming")
    parser.add_argument("--pad_and_drop_preencoded", action="store_true",
                        help="Pad and drop pre-encoded for streaming")
    parser.add_argument("--compare_vs_offline", action="store_true",
                        help="Compare streaming vs offline mode")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    # --cpu flag accepted for compatibility but always runs on CPU
    parser.add_argument("--cpu", action="store_true", default=True,
                        help="Always enabled (CPU-only script)")
    parser.add_argument("--use_cer", action="store_true",
                        help="Use CER instead of WER")
    parser.add_argument("--normalize_text", action="store_true", default=True,
                        help="Use whisper-normalizer to normalize text before WER calculation (default: True)")
    parser.add_argument("--no_normalize_text", dest="normalize_text", action="store_false",
                        help="Disable text normalization before WER calculation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.setLevel(logging.INFO)
    
    # Run evaluation
    evaluate_streaming_model(args)


if __name__ == "__main__":
    main()
