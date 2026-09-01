# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Benchmark padded, serial-THD, and grouped-THD ASR encoder implementations.

This is an end-to-end implementation comparison, not a controlled change of tensor
layout alone. PEE reports its historical padded grouped path, a serial native-THD
oracle, and its production layer-synchronous grouped-THD path. All paths use
identical weights, inputs, valid-token output loss, and weighted MoE load-balancing
loss. Trial order is counterbalanced to reduce cache/order bias.

Example::

    env PYTHONPATH=. python scripts/speech_recognition/benchmark_packed_asr_encoders.py \
        --encoders transformer moe pee --phases inference training \
        --warmup 20 --iterations 100 --repeats 6 \
        --output packed_sequence_asr_encoders_benchmark_final.json
"""

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder
from nemo.collections.asr.modules.parallel_expert_encoder_ggemm import ParallelExpertEncoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)

    results = []
    equivalence = {}
    for encoder_name in args.encoders:
        implementations = _implementation_labels(encoder_name, args.pee_implementations)
        trials = {(phase, implementation): [] for phase in args.phases for implementation in implementations}
        for repeat in range(args.repeats):
            torch.manual_seed(args.seed)
            model, inputs, lengths, speaker_targets = _make_workload(encoder_name, device, args.dtype)
            if encoder_name == 'pee':
                model.sequence_packed_moe_mode = args.pee_sequence_packed_moe_mode
            if repeat == 0:
                equivalence[encoder_name] = _numerical_preflight(
                    encoder_name, model, inputs, lengths, speaker_targets, implementations
                )

            for phase_index, phase in enumerate(args.phases):
                order = _counterbalanced_order(implementations, repeat + phase_index)
                for order_index, implementation in enumerate(order):
                    trial = _benchmark_implementation(
                        encoder_name,
                        model,
                        inputs,
                        lengths,
                        speaker_targets,
                        implementation=implementation,
                        phase=phase,
                        repeat=repeat,
                        order_index=order_index,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        profile=args.profile,
                    )
                    trials[(phase, implementation)].append(trial)

            del model, inputs, lengths, speaker_targets
            torch.cuda.empty_cache()

        for phase in args.phases:
            for implementation in implementations:
                result = _aggregate_trials(trials[(phase, implementation)])
                results.append(result)
                print(
                    f"{encoder_name:11s} {phase:9s} {implementation:13s} "
                    f"{result['latency_ms']:9.3f} ms "
                    f"[{result['latency_q1_ms']:.3f}, {result['latency_q3_ms']:.3f}] IQR  "
                    f"{result['valid_input_frames_per_second']:12.0f} frame/s  "
                    f"{result['peak_memory_mib']:9.1f} MiB"
                )

    report = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
        "torch": torch.__version__,
        "dtype": str(args.dtype).removeprefix("torch."),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "pee_sequence_packed_moe_mode": args.pee_sequence_packed_moe_mode,
        "provenance": _source_provenance(),
        "comparison_scope": (
            "End-to-end legacy padded, serial native-THD, and grouped native-THD implementations; "
            "backend and routing differences are reported and results must not be attributed to layout alone. "
            "Legacy padded MoE auxiliary loss includes padded positions for backwards compatibility, while "
            "native packed MoE routing and auxiliary loss intentionally include valid tokens only."
        ),
        "numerical_preflight": equivalence,
        "results": results,
        "comparisons": _compare_implementations(results),
    }
    print(json.dumps(report["comparisons"], indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def _implementation_labels(encoder_name, pee_implementations):
    if encoder_name == 'pee':
        return tuple(pee_implementations)
    return ('legacy_bhsd', 'native_thd')


def _counterbalanced_order(implementations, index):
    orders = tuple(itertools.permutations(implementations))
    return orders[index % len(orders)]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoders", nargs="+", choices=("transformer", "moe", "pee"), default=("transformer", "moe", "pee")
    )
    parser.add_argument("--phases", nargs="+", choices=("inference", "training"), default=("inference", "training"))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--pee-implementations",
        nargs="+",
        choices=("legacy_bhsd", "serial_thd", "grouped_thd"),
        default=("legacy_bhsd", "serial_thd", "grouped_thd"),
        help="PEE implementations to compare; the default preserves the serial THD oracle.",
    )
    parser.add_argument(
        "--pee-sequence-packed-moe-mode",
        choices=("auto", "dense", "topk", "native"),
        default="auto",
        help="Grouped-THD PEE MoE policy; topk is the memory-first grouped-kernel ablation.",
    )
    parser.add_argument("--profile", action="store_true", help="Add NVTX ranges around every measured iteration.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --iterations/--repeats must be positive")
    args.dtype = getattr(torch, args.dtype)
    return args


def _make_workload(name, device, dtype):
    if name == "transformer":
        model = TransformerEncoder(
            feat_in=512,
            d_model=512,
            n_heads=8,
            n_layers=6,
            subsampling_factor=1,
            ff_expansion=4.0,
            self_attention_model="rope",
            qk_norm=True,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            sync_max_audio_length=False,
        )
        inputs = torch.randn(8, 1024, 512, device=device, dtype=dtype)
        lengths = torch.tensor([1024, 768, 512, 384, 256, 192, 128, 64], device=device)
        speaker_targets = None
    elif name == "moe":
        model = MoETransformerEncoder(
            feat_in=512,
            d_model=512,
            n_heads=8,
            n_layers=4,
            subsampling_factor=1,
            ff_expansion=4.0,
            self_attention_model="rope",
            qk_norm=True,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            moe_num_experts=8,
            moe_top_k=2,
            moe_load_balance_loss_weight=0.01,
            sync_max_audio_length=False,
        )
        inputs = torch.randn(8, 1024, 512, device=device, dtype=dtype)
        lengths = torch.tensor([1024, 768, 512, 384, 256, 192, 128, 64], device=device)
        speaker_targets = None
    else:
        model = _make_pee()
        inputs = torch.randn(4, 128, 2048, device=device, dtype=dtype)
        lengths = torch.tensor([2048, 1024, 512, 256], device=device)
        speaker_targets = torch.zeros(4, 256, 4, device=device, dtype=dtype)
    return model.to(device=device, dtype=dtype), inputs, lengths, speaker_targets


def _make_pee():
    speech = _expert_config(
        "nemo.collections.asr.modules.MoETransformerEncoder",
        512,
        8,
        moe_num_experts=8,
        moe_top_k=2,
        moe_load_balance_loss_weight=0.01,
    )
    sound = _expert_config("nemo.collections.asr.modules.TransformerEncoder", 512, 8)
    speaker = _expert_config("nemo.collections.asr.modules.TransformerEncoder", 256, 4)
    sortformer = DictConfig(
        {
            "_target_": "nemo.collections.asr.modules.sortformer_modules.SortformerModules",
            "num_spks": 4,
            "dropout_rate": 0.0,
            "fc_d_model": 256,
            "tf_d_model": 256,
            "subsampling_factor": 8,
            "spkcache_len": 16,
            "fifo_len": 0,
            "chunk_len": 500,
            "spkcache_update_period": 500,
            "chunk_left_context": 0,
            "chunk_right_context": 0,
            "spkcache_sil_frames_per_spk": 1,
        }
    )
    return ParallelExpertEncoder(
        speech_expert_cfg=speech,
        speaker_expert_cfg=speaker,
        sound_expert_cfg=sound,
        sortformer_modules_cfg=sortformer,
        asr_normalize_type="per_feature",
        always_run_diarization=False,
        online_inference_length=500,
        chunk_left_context=0,
        chunk_right_context=0,
        diar_spkcache_len=16,
        diar_spkcache_update_period=500,
        merge_sound_expert_to_asr=True,
    )


def _expert_config(target, d_model, n_heads, **extra):
    config = {
        "_target_": target,
        "feat_in": 128,
        "feat_out": -1,
        "n_layers": 4,
        "d_model": d_model,
        "n_heads": n_heads,
        "subsampling": "feature_stacking",
        "subsampling_factor": 8,
        "ff_expansion": 4.0,
        "self_attention_model": "rope",
        "pos_emb_max_len": 5000,
        "xscaling": False,
        "qkv_bias": False,
        "qk_norm": True,
        "pre_block_norm": True,
        "attn_mode": "full",
        "drop_rate": 0.0,
        "dropout_pre_encoder": 0.0,
        "dropout_emb": 0.0,
        "sync_max_audio_length": False,
    }
    config.update(extra)
    return DictConfig(config)


def _source_provenance():
    repo = Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(
            ("git", *args),
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    source_paths = (
        "nemo/collections/asr/modules/ggemm_transformer_encoder.py",
        "nemo/collections/asr/modules/moe_transformer_encoder.py",
        "nemo/collections/asr/modules/parallel_expert_encoder_ggemm.py",
        "nemo/collections/asr/modules/transformer_encoder.py",
        "nemo/collections/asr/parts/packed_sequence.py",
        "nemo/collections/speechlm2/models/salm_automodel.py",
        "nemo/collections/speechlm2/modules/perception.py",
        "nemo/collections/speechlm2/parts/cp_helpers.py",
        "nemo/collections/speechlm2/parts/encoder_chunking.py",
        "scripts/speech_recognition/benchmark_packed_asr_encoders.py",
    )
    source_hash = hashlib.sha256()
    for relative_path in source_paths:
        source_hash.update(relative_path.encode())
        source_hash.update((repo / relative_path).read_bytes())
    try:
        import importlib.metadata

        flash_attention_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        flash_attention_version = None
    status = git("status", "--porcelain", "--untracked-files=all")
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "git_sha": git("rev-parse", "HEAD").decode().strip(),
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "benchmark_source_sha256": source_hash.hexdigest(),
        "cuda": torch.version.cuda,
        "flash_attention": flash_attention_version,
    }


def _numerical_preflight(encoder_name, model, inputs, lengths, speaker_targets, implementations):
    model.eval()
    outputs = {}
    with torch.inference_mode():
        for implementation in implementations:
            outputs[implementation] = _valid_output(
                encoder_name, model, inputs, lengths, speaker_targets, implementation
            )
    reference_name = 'legacy_bhsd' if 'legacy_bhsd' in outputs else implementations[0]
    reference_output, reference_lengths = outputs[reference_name]
    comparisons = {}
    for implementation in implementations:
        output, output_lengths = outputs[implementation]
        lengths_identical = torch.equal(output_lengths, reference_lengths)
        if not lengths_identical:
            raise AssertionError(
                f"{encoder_name}/{implementation} output lengths differ: "
                f"legacy={reference_lengths.tolist()}, candidate={output_lengths.tolist()}"
            )
        torch.testing.assert_close(output, reference_output, rtol=3e-2, atol=3e-2)
        difference = (output.float() - reference_output.float()).abs()
        reference_norm = torch.linalg.vector_norm(reference_output.float())
        relative_l2_error = torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-12)
        comparisons[implementation] = {
            "valid_output_tokens": int(reference_output.shape[0]),
            "lengths_identical": lengths_identical,
            "max_abs_error": float(difference.max().item()) if difference.numel() else 0.0,
            "mean_abs_error": float(difference.mean().item()) if difference.numel() else 0.0,
            "relative_l2_error": float(relative_l2_error.item()) if difference.numel() else 0.0,
            "rtol": 0.03,
            "atol": 0.03,
            "passed": lengths_identical,
        }
    if 'serial_thd' in outputs and 'grouped_thd' in outputs:
        serial_output, serial_lengths = outputs['serial_thd']
        grouped_output, grouped_lengths = outputs['grouped_thd']
        assert torch.equal(grouped_lengths, serial_lengths)
        torch.testing.assert_close(grouped_output, serial_output, rtol=3e-2, atol=3e-2)
    return {"reference": reference_name, "implementations": comparisons}


def _benchmark_implementation(
    encoder_name,
    model,
    inputs,
    lengths,
    speaker_targets,
    *,
    implementation,
    phase,
    repeat,
    order_index,
    warmup,
    iterations,
    profile,
):
    training = phase == "training"
    layout = 'bhsd' if implementation == 'legacy_bhsd' else 'thd'
    model.train(training)
    _clear_runtime_caches(model, encoder_name)
    torch.cuda.empty_cache()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _run_iteration(encoder_name, model, inputs, lengths, speaker_targets, implementation, training)
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    _clear_moe_auxiliary_loss(encoder_name, model)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    elapsed = []
    for iteration in range(iterations):
        model.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        range_name = f"packed_asr/{encoder_name}/{phase}/{implementation}/repeat_{repeat}/iteration_{iteration}"
        context = torch.cuda.nvtx.range(range_name) if profile else nullcontext()
        with context:
            start.record()
            _run_iteration(encoder_name, model, inputs, lengths, speaker_targets, implementation, training)
            end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))

    peak_total = torch.cuda.max_memory_allocated()
    peak = peak_total - baseline
    return {
        "encoder": encoder_name,
        "phase": phase,
        "layout": layout,
        "implementation": implementation,
        "qkv_projection": (
            "grouped_bmm_or_baddbmm"
            if implementation == "grouped_thd"
            else "fused" if implementation in ("legacy_bhsd", "serial_thd") else "independent_slices"
        ),
        "repeat": repeat,
        "order_index": order_index,
        "latencies_ms": elapsed,
        "peak_memory_bytes": peak,
        "resident_memory_bytes": baseline,
        "peak_total_memory_bytes": peak_total,
        "backend": _implementation_backend(encoder_name, phase, implementation),
        "packed_execution_trace": (
            getattr(model.pee, '_last_sequence_packed_execution', None)
            if encoder_name == 'pee' and implementation == 'grouped_thd'
            else None
        ),
        "moe_grouped_backends": (
            _moe_grouped_backend_values(model, encoder_name) if implementation == 'grouped_thd' else []
        ),
        "runtime_attention_backends": (
            _attention_runtime_values(model, encoder_name, "_last_sequence_packed_backend") if layout == "thd" else []
        ),
        "runtime_attention_providers": (
            _attention_runtime_values(model, encoder_name, "_last_sequence_packed_provider") if layout == "thd" else []
        ),
    }


def _run_iteration(encoder_name, model, inputs, lengths, speaker_targets, implementation, training):
    context = nullcontext() if training else torch.inference_mode()
    with context:
        output, _ = _valid_output(encoder_name, model, inputs, lengths, speaker_targets, implementation)
        if training:
            loss = output.float().square().mean()
            auxiliary_loss = _moe_auxiliary_loss(encoder_name, model)
            if auxiliary_loss is not None:
                loss = loss + auxiliary_loss
            loss.backward()
            _clear_moe_auxiliary_loss(encoder_name, model)


def _valid_output(encoder_name, model, inputs, lengths, speaker_targets, implementation):
    if encoder_name == "pee":
        if implementation == 'serial_thd':
            packed = model._forward_sequence_packed(
                inputs,
                lengths,
                spk_targets=speaker_targets,
                grouped=False,
            )
            return packed.data, packed.lengths
        if implementation == 'grouped_thd':
            packed = model.forward_sequence_packed(inputs, lengths, spk_targets=speaker_targets)
            return packed.data, packed.lengths
        output, output_lengths = model(inputs, lengths, spk_targets=speaker_targets)
    elif implementation == 'native_thd':
        packed = model.forward_sequence_packed(inputs, lengths, bypass_pre_encode=True)
        return packed.data, packed.lengths
    else:
        output, output_lengths = model(inputs, lengths, bypass_pre_encode=True)

    valid = torch.arange(output.shape[-1], device=output.device)[None, :] < output_lengths[:, None]
    return output.transpose(1, 2)[valid], output_lengths


def _moe_encoder(encoder_name, model):
    if encoder_name == "moe":
        return model
    if encoder_name == "pee":
        return model.pee.experts["speech"]
    return None


def _moe_auxiliary_loss(encoder_name, model):
    encoder = _moe_encoder(encoder_name, model)
    return encoder.get_moe_auxiliary_loss() if encoder is not None else None


def _clear_moe_auxiliary_loss(encoder_name, model):
    encoder = _moe_encoder(encoder_name, model)
    if encoder is None:
        return
    for layer_index in encoder.moe_layer_indices:
        encoder.layers[layer_index].ffn._aux_loss = None


def _clear_runtime_caches(model, encoder_name):
    if encoder_name == 'pee':
        model.pee.clear_packed_weights()


def _implementation_backend(encoder_name, phase, implementation):
    if implementation == 'native_thd':
        return "independent_token_flat_encoder_with_varlen_attention"
    if implementation == 'serial_thd':
        return "serial_token_flat_pee_experts_with_varlen_attention"
    if implementation == 'grouped_thd':
        return "layer_synchronous_grouped_thd_attention_projections_and_grouped_ffn_moe"
    if encoder_name != "pee":
        return "padded_flex_attention"
    if phase == "training":
        return "independent_padded_experts_with_flex_attention_and_native_topk_moe"
    return "cross_expert_head_packed_sdpa_and_grouped_dense_moe"


def _moe_grouped_backend_values(model, encoder_name):
    encoder = _moe_encoder(encoder_name, model)
    if encoder is None:
        return []
    return sorted(
        {
            value
            for layer_index in encoder.moe_layer_indices
            if (value := getattr(encoder.layers[layer_index].ffn, '_last_grouped_backend', None)) is not None
        }
    )


def _attention_runtime_values(model, encoder_name, attribute):
    encoders = model.pee.experts.values() if encoder_name == "pee" else (model,)
    values = {
        value
        for encoder in encoders
        for layer in encoder.layers
        if (value := getattr(layer.attn, attribute, None)) is not None
    }
    return sorted(values)


def _aggregate_trials(trials):
    latencies = [latency for trial in trials for latency in trial["latencies_ms"]]
    peaks = [trial["peak_memory_bytes"] for trial in trials]
    resident = [trial["resident_memory_bytes"] for trial in trials]
    total_peaks = [trial["peak_total_memory_bytes"] for trial in trials]
    latency_ms = statistics.median(latencies)
    template = trials[0]
    return {
        "encoder": template["encoder"],
        "phase": template["phase"],
        "layout": template["layout"],
        "implementation": template["implementation"],
        "qkv_projection": template["qkv_projection"],
        "latency_ms": latency_ms,
        "latency_q1_ms": _percentile(latencies, 0.25),
        "latency_q3_ms": _percentile(latencies, 0.75),
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
        "repeat_medians_ms": [statistics.median(trial["latencies_ms"]) for trial in trials],
        "valid_input_frames_per_second": float(_valid_input_frames(template["encoder"])) * 1000.0 / latency_ms,
        "peak_memory_bytes": int(statistics.median(peaks)),
        "peak_memory_mib": statistics.median(peaks) / 2**20,
        "repeat_peak_memory_bytes": peaks,
        "resident_memory_bytes": int(statistics.median(resident)),
        "peak_total_memory_bytes": int(statistics.median(total_peaks)),
        "peak_total_memory_mib": statistics.median(total_peaks) / 2**20,
        "repeat_resident_memory_bytes": resident,
        "repeat_peak_total_memory_bytes": total_peaks,
        "orders": [trial["order_index"] for trial in trials],
        "trials": trials,
        "backend": template["backend"],
        "packed_execution_trace": template["packed_execution_trace"],
        "moe_grouped_backends": sorted({value for trial in trials for value in trial["moe_grouped_backends"]}),
        "runtime_attention_backends": sorted(
            {value for trial in trials for value in trial["runtime_attention_backends"]}
        ),
        "runtime_attention_providers": sorted(
            {value for trial in trials for value in trial["runtime_attention_providers"]}
        ),
    }


def _percentile(values, quantile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _valid_input_frames(encoder_name):
    return 3840 if encoder_name == "pee" else 3328


def _compare_implementations(results):
    grouped = {(item["encoder"], item["phase"], item["implementation"]): item for item in results}
    comparisons = []
    encoder_phases = sorted({(item['encoder'], item['phase']) for item in results})
    for encoder, phase in encoder_phases:
        legacy = grouped.get((encoder, phase, 'legacy_bhsd'))
        candidates = [item for item in results if item['encoder'] == encoder and item['phase'] == phase]
        if legacy is not None:
            for candidate in candidates:
                if candidate['implementation'] == 'legacy_bhsd':
                    continue
                comparisons.append(
                    {
                        "encoder": encoder,
                        "phase": phase,
                        "baseline": "legacy_bhsd",
                        "candidate": candidate['implementation'],
                        "speedup": legacy["latency_ms"] / candidate["latency_ms"],
                        "peak_memory_reduction": 1.0 - candidate["peak_memory_bytes"] / legacy["peak_memory_bytes"],
                        "peak_total_memory_reduction": (
                            1.0 - candidate["peak_total_memory_bytes"] / legacy["peak_total_memory_bytes"]
                        ),
                    }
                )
        serial = grouped.get((encoder, phase, 'serial_thd'))
        packed_grouped = grouped.get((encoder, phase, 'grouped_thd'))
        if serial is not None and packed_grouped is not None:
            comparisons.append(
                {
                    "encoder": encoder,
                    "phase": phase,
                    "baseline": "serial_thd",
                    "candidate": "grouped_thd",
                    "speedup": serial["latency_ms"] / packed_grouped["latency_ms"],
                    "peak_memory_reduction": 1.0 - packed_grouped["peak_memory_bytes"] / serial["peak_memory_bytes"],
                    "peak_total_memory_reduction": (
                        1.0 - packed_grouped["peak_total_memory_bytes"] / serial["peak_total_memory_bytes"]
                    ),
                }
            )
    return comparisons


if __name__ == "__main__":
    main()
