#!/usr/bin/env python3
"""
Calculate average WER across all evaluated datasets for NeMo streaming models.
Similar to eval_utils.score_results() in open_asr_leaderboard.
"""

import argparse
import json
import glob
import os
from pathlib import Path


def calculate_average_wer(results_dir, model_name=None):
    """
    Calculate average WER across all dataset results.
    
    Args:
        results_dir: Directory containing streaming_summary_*.json files
        model_name: Optional model name to include in output
    """
    
    # Find all summary files
    summary_files = glob.glob(os.path.join(results_dir, "streaming_summary_*.json"))
    
    if not summary_files:
        print(f"No summary files found in {results_dir}")
        return
    
    print("=" * 70)
    print("NeMo Streaming ASR - Average WER Calculation")
    print("=" * 70)
    if model_name:
        print(f"Model: {model_name}")
    print(f"Results directory: {results_dir}")
    print("=" * 70)
    print()
    
    # Collect all results
    all_results = []
    total_samples = 0
    
    for summary_file in sorted(summary_files):
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        dataset_name = data.get('dataset', 'unknown')
        split = data.get('split', 'test')
        num_samples = data.get('num_samples', 0)
        streaming_wer = data.get('streaming_wer', None)
        streaming_cer = data.get('streaming_cer', None)
        offline_wer = data.get('offline_wer', None)
        offline_cer = data.get('offline_cer', None)
        streaming_rtfx = data.get('streaming_rtfx', None)
        offline_rtfx = data.get('offline_rtfx', None)
        
        if streaming_wer is not None:
            metric_name = "WER"
            metric_value = streaming_wer
        elif streaming_cer is not None:
            metric_name = "CER"
            metric_value = streaming_cer
        else:
            continue
        
        all_results.append({
            'dataset': dataset_name,
            'split': split,
            'samples': num_samples,
            'metric_name': metric_name,
            'streaming_metric': metric_value,
            'offline_metric': offline_wer if offline_wer else offline_cer,
            'streaming_rtfx': streaming_rtfx,
            'offline_rtfx': offline_rtfx,
        })
        
        total_samples += num_samples
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Print individual results
    print("Individual Dataset Results:")
    print("-" * 105)
    print(f"{'Dataset':<25} {'Samples':<10} {'Streaming':<12} {'Offline':<12} {'Stream RTFx':<12} {'Offline RTFx':<12}")
    print("-" * 105)
    
    for result in all_results:
        dataset_split = f"{result['dataset']}/{result['split']}"
        offline_str = f"{result['offline_metric']:.2f}" if result['offline_metric'] else "N/A"
        stream_rtfx_str = f"{result['streaming_rtfx']:.2f}x" if result['streaming_rtfx'] else "N/A"
        offline_rtfx_str = f"{result['offline_rtfx']:.2f}x" if result['offline_rtfx'] else "N/A"
        print(f"{dataset_split:<25} {result['samples']:<10} "
              f"{result['streaming_metric']:<12.2f} {offline_str:<12} {stream_rtfx_str:<12} {offline_rtfx_str:<12}")
    
    print("-" * 105)
    print()
    
    # Calculate average WER (simple average across datasets)
    avg_streaming_wer = sum(r['streaming_metric'] for r in all_results) / len(all_results)
    
    # Calculate weighted average by number of samples
    weighted_streaming_wer = sum(r['streaming_metric'] * r['samples'] for r in all_results) / total_samples
    
    # Calculate offline averages if available
    offline_results = [r for r in all_results if r['offline_metric'] is not None]
    if offline_results:
        avg_offline_wer = sum(r['offline_metric'] for r in offline_results) / len(offline_results)
        weighted_offline_wer = sum(r['offline_metric'] * r['samples'] for r in offline_results) / sum(r['samples'] for r in offline_results)
    else:
        avg_offline_wer = None
        weighted_offline_wer = None
    
    # Calculate average RTFx
    streaming_rtfx_values = [r['streaming_rtfx'] for r in all_results if r['streaming_rtfx'] is not None]
    avg_streaming_rtfx = sum(streaming_rtfx_values) / len(streaming_rtfx_values) if streaming_rtfx_values else None
    
    offline_rtfx_values = [r['offline_rtfx'] for r in all_results if r['offline_rtfx'] is not None]
    avg_offline_rtfx = sum(offline_rtfx_values) / len(offline_rtfx_values) if offline_rtfx_values else None
    
    # Print summary
    print("=" * 105)
    print("SUMMARY - STREAMING MODE")
    print("=" * 105)
    print(f"Total datasets evaluated: {len(all_results)}")
    print(f"Total samples: {total_samples:,}")
    print()
    print(f"Average Streaming {all_results[0]['metric_name']} (simple average):   {avg_streaming_wer:.2f}%")
    print(f"Average Streaming {all_results[0]['metric_name']} (weighted by samples): {weighted_streaming_wer:.2f}%")
    if avg_streaming_rtfx:
        print(f"Average Streaming RTFx: {avg_streaming_rtfx:.2f}x faster than real-time")
    print("=" * 105)
    
    if avg_offline_wer is not None:
        print()
        print("=" * 105)
        print("SUMMARY - OFFLINE MODE (for comparison)")
        print("=" * 105)
        print(f"Total datasets evaluated: {len(offline_results)}")
        print()
        print(f"Average Offline {all_results[0]['metric_name']} (simple average):   {avg_offline_wer:.2f}%")
        print(f"Average Offline {all_results[0]['metric_name']} (weighted by samples): {weighted_offline_wer:.2f}%")
        if avg_offline_rtfx:
            print(f"Average Offline RTFx: {avg_offline_rtfx:.2f}x faster than real-time")
        print()
        print(f"Difference (Streaming - Offline):")
        print(f"  {all_results[0]['metric_name']}: {avg_streaming_wer - avg_offline_wer:+.2f}%")
        if avg_streaming_rtfx and avg_offline_rtfx:
            print(f"  RTFx: {avg_streaming_rtfx - avg_offline_rtfx:+.2f}x")
        print("=" * 105)
    
    # Save aggregated results
    output_file = os.path.join(results_dir, "aggregated_results.json")
    aggregated = {
        "model": model_name or "unknown",
        "num_datasets": len(all_results),
        "total_samples": total_samples,
        "streaming": {
            "average_wer_simple": round(avg_streaming_wer, 2),
            "average_wer_weighted": round(weighted_streaming_wer, 2),
            "average_rtfx": round(avg_streaming_rtfx, 2) if avg_streaming_rtfx else None,
        },
        "offline": {
            "average_wer_simple": round(avg_offline_wer, 2) if avg_offline_wer else None,
            "average_wer_weighted": round(weighted_offline_wer, 2) if weighted_offline_wer else None,
            "average_rtfx": round(avg_offline_rtfx, 2) if avg_offline_rtfx else None,
        } if avg_offline_wer else None,
        "datasets": all_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print()
    print(f"Aggregated results saved to: {output_file}")
    print()
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Calculate average WER for NeMo streaming evaluation")
    parser.add_argument("results_dir", type=str, help="Directory containing streaming_summary_*.json files")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for display")
    
    args = parser.parse_args()
    
    calculate_average_wer(args.results_dir, args.model_name)


if __name__ == "__main__":
    main()
