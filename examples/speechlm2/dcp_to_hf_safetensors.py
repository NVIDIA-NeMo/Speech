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
"""Consolidate a distributed (DCP) checkpoint straight to HuggingFace safetensors.

Why this exists: ``examples/speechlm2/to_hf.py`` rebuilds the full model before
loading, so exporting a 30B MoE needs the sharded backbone AND the expert-layout
conversion temporaries resident at once.  On a 2-GPU / 125 GB host that OOMs.

This script never constructs the model, and never holds the whole state dict:

  * names, shapes and dtypes come from the checkpoint's own metadata, so the
    complete safetensors header (including every byte offset) is computed before
    any tensor data is read;
  * tensors are then loaded in size-bounded batches and appended to the file in
    header order, so peak memory is one batch (default 4 GiB), not 60 GiB.

``safetensors.torch.save_file`` cannot be used here: it serializes the whole dict
into a second buffer, which is what pushed a 60 GiB state dict past this host's
125 GB (the writer was OOM-killed).  The byte layout produced below was checked
against ``safetensors.torch.save`` output for an identical tensor.

It is CPU-only: no CUDA, no process group, no device mesh -- ``dcp.load``
reassembles the shards in a single process.

Output matches what ``save_hf_checkpoint`` writes for the parts that matter to
``nemo.collections.speechlm2.parts.hf_hub``: one ``model.safetensors`` (the loader
hard-requires a single file) plus ``config.json``.  It does NOT write the
``llm_backbone/`` config or the vLLM extras.

Usage:
    python examples/speechlm2/dcp_to_hf_safetensors.py <ckpt_dir> <exp_config.yaml> <output_dir> [dtype] [batch_gib]
"""

import json
import struct
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf

MODEL_PREFIX = "state_dict."

# safetensors dtype names; only what these checkpoints actually contain.
ST_DTYPE = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    ckpt_dir, cfg_file, output_dir = sys.argv[1:4]
    dtype = getattr(torch, sys.argv[4] if len(sys.argv) > 4 else "bfloat16")
    batch_bytes = int(float(sys.argv[5]) * 2**30) if len(sys.argv) > 5 else 4 * 2**30

    meta = dcp.FileSystemReader(ckpt_dir).read_metadata().state_dict_metadata

    # Model weights only -- the checkpoint also holds optimizer_0.* (about 2/3 of
    # the keys), which has no place in an inference checkpoint.
    keys = [k for k in meta if k.startswith(MODEL_PREFIX) and hasattr(meta[k], "size")]
    shapes = {k: torch.Size(meta[k].size) for k in keys}
    total = sum(shapes[k].numel() for k in keys)
    print(f"model tensors: {len(keys)}  params: {total / 1e9:.2f}B  -> {total * dtype.itemsize / 2**30:.1f} GiB")

    # 1. Header: every offset is known up front from shapes + target dtype.
    header, offset = {}, 0
    for k in keys:
        n = shapes[k].numel() * dtype.itemsize
        header[k[len(MODEL_PREFIX) :]] = {
            "dtype": ST_DTYPE[dtype],
            "shape": list(shapes[k]),
            "data_offsets": [offset, offset + n],
        }
        offset += n
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)  # data section stays 8-byte aligned

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    st_file = out_path / "model.safetensors"

    # 2. Stream the data section in header order, one bounded batch at a time.
    written, batch, batch_size = 0, [], 0
    with open(st_file, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)

        def flush(batch):
            nonlocal written
            if not batch:
                return
            sd = {k: torch.empty(shapes[k], dtype=meta[k].properties.dtype) for k in batch}
            dcp.load(sd, checkpoint_id=ckpt_dir)
            for k in batch:  # header order preserved
                t = sd.pop(k)
                t = t.to(dtype) if t.dtype != dtype else t
                fh.write(t.contiguous().flatten().view(torch.uint8).numpy().tobytes())
                written += 1
                del t
            print(f"  {written}/{len(keys)} tensors written", flush=True)

        for k in keys:
            n = shapes[k].numel() * meta[k].properties.dtype.itemsize
            if batch and batch_size + n > batch_bytes:
                flush(batch)
                batch, batch_size = [], 0
            batch.append(k)
            batch_size += n
        flush(batch)

    size_gib = st_file.stat().st_size / 2**30
    print(f"wrote {st_file} ({size_gib:.1f} GiB)")

    # 3. config.json mirrors _hf_export_config(): the training model config plus the
    # dtype fields. pretrained_weights=False stops from_pretrained() from pulling
    # the HF backbone just to overwrite it with these weights. init_configure_model
    # and torch_dtype are set by the loader (hf_hub._distributed_from_pretrained).
    model_cfg = OmegaConf.to_container(OmegaConf.load(cfg_file), resolve=True)["model"]
    dtype_name = str(dtype).replace("torch.", "")
    model_cfg["dtype"] = dtype_name
    model_cfg["torch_dtype"] = dtype_name
    model_cfg["pretrained_weights"] = False
    # The exported safetensors already contains every trained tensor, so re-loading
    # the original backbones is pure waste -- and fatal on a single GPU: with
    # load_llm_weights=true, configure_model() materializes the full 62 GB Nemotron
    # backbone plus the merged->split expert conversion temporaries before these
    # weights are applied, which OOMs a 95 GiB card.
    # Note hf_hub sets cfg['pretrained_weights']=False for this purpose, but
    # streaming_stt_model_automodel passes cfg.load_llm_weights / load_asr_weights
    # to Automodel instead, so those are the fields that actually matter here.
    model_cfg["load_llm_weights"] = False
    # load_asr_weights stays True: the perception encoder/preprocessor config lives
    # in the ASR .nemo, not in this config.json, so setting it False fails with
    # "Missing key preprocessor" in AudioPerceptionModule. It is only a 0.6B model.
    (out_path / "config.json").write_text(json.dumps(model_cfg, indent=2))
    print(f"wrote {out_path / 'config.json'}")


if __name__ == "__main__":
    main()
