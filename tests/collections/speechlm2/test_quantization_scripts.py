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

"""Unit tests for the checkpoint-surgery parts of scripts/speechlm2/quantization (no GPU, vLLM or ModelOpt)."""

import errno
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "speechlm2" / "quantization"
FP8_MAX = 448.0


@pytest.fixture(autouse=True)
def cpu_default_device():
    # Other speechlm2 test modules set CUDA as the default device at import; safetensors needs CPU tensors.
    with torch.device("cpu"):
        yield


@pytest.fixture(scope="module")
def scripts():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield SimpleNamespace(
            utils=importlib.import_module("checkpoint_utils"),
            export=importlib.import_module("export_llm_to_hf"),
            assemble=importlib.import_module("assemble_checkpoint"),
            fp8=importlib.import_module("quantize_encoder_mtp_fp8"),
            calibrate=importlib.import_module("calibrate_activations"),
        )
    finally:
        sys.path.remove(str(SCRIPTS))


def run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, argv)])
    module.main()


def write_json(path, data):
    path.write_text(json.dumps(data))


def read_json(path):
    return json.loads(path.read_text())


def no_hard_links(monkeypatch):
    def link(source, destination):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", link)


def tensors_of(path):
    with safe_open(path, "pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}, f.metadata()


def make_source(root, *, ctc=True):
    """A tiny BF16 SpeechLM checkpoint: decoder, MTP head, perception, optional CTC head, configs and docs."""
    root.mkdir()
    write_json(
        root / "config.json",
        {"model_type": "nemo_speechlm", "architectures": ["NeMoSpeechLMForConditionalGeneration"]},
    )
    (root / "llm_backbone").mkdir()
    write_json(root / "llm_backbone" / "config.json", {"model_type": "nemotron_h"})
    for name in ("README.md", "SHA256SUMS", "tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text(name)
    tensors = {
        "llm.model.embed_tokens.weight": torch.randn(8, 4).bfloat16(),
        "llm.model.layers.0.mixer.in_proj.weight": torch.randn(6, 4).bfloat16(),
        "llm.model.layers.0.mixer.in_proj._extra_state": torch.zeros(0).bfloat16(),
        "llm.model.layers.0.mixer._fp32_params.A_log": torch.tensor([0.123456789, -5.4321098, 1.0]),
        "llm.model.layers.0.mixer._fp32_params.D": torch.tensor([1.000123, 2.0, 3.0]),
        "llm.lm_head.weight": torch.randn(8, 4).bfloat16(),
        "llm.mtp.layers.0.eh_proj.weight": torch.randn(4, 8).bfloat16(),
        "perception.encoder.asr_encoder.layers.0.attn.w_qkv.weight": torch.randn(12, 4).bfloat16(),
        "perception.proj.weight": torch.randn(4, 4).bfloat16(),
    }
    metadata = {"format": "pt"}
    if ctc:
        tensors["ctc_timestamp.head.weight"] = torch.randn(5, 4).bfloat16()
        metadata["ctc_timestamp_format"] = "v1"
    save_file(tensors, root / "model.safetensors", metadata=metadata)
    return root


def make_quantized_llm(root, quant_config, *, single_file=False):
    """A tiny ModelOpt LLM-only export with the given quantization config, sharded or in one unindexed file."""
    root.mkdir()
    write_json(root / "config.json", {"model_type": "nemotron_h", "quantization_config": quant_config})
    write_json(root / "hf_quant_config.json", {"producer": {"name": "modelopt"}, "quantization": quant_config})
    tensors = {
        "backbone.layers.0.mixer.in_proj.weight": torch.randn(6, 4).to(torch.float8_e4m3fn),
        # ModelOpt exports these unquantized parameters rounded to BF16.
        "backbone.layers.0.mixer.A_log": torch.tensor([0.123456789, -5.4321098, 1.0]).bfloat16(),
        "backbone.layers.0.mixer.D": torch.tensor([1.000123, 2.0, 3.0]).bfloat16(),
    }
    if single_file:
        save_file(tensors, root / "model.safetensors")
        return root
    save_file(tensors, root / "model-00001-of-00001.safetensors")
    weight_map = dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
    write_json(root / "model.safetensors.index.json", {"metadata": {"total_size": 24}, "weight_map": weight_map})
    return root


FP8_CONFIG = {"quant_method": "modelopt", "quant_algo": "FP8", "ignore": ["lm_head"]}
MIXED_CONFIG = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {"backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"}},
}


class TestCheckpointUtils:
    def test_copy_tensors_is_byte_identical_and_keeps_metadata(self, scripts, tmp_path):
        source = make_source(tmp_path / "src")
        entries = scripts.utils.find_tensors(source, ("ctc_timestamp.",))
        metadata = scripts.utils.file_metadata(entries, "ctc_timestamp_")
        size = scripts.utils.copy_tensors(entries, tmp_path / "ctc.safetensors", metadata)

        copied, copied_metadata = tensors_of(tmp_path / "ctc.safetensors")
        original, _ = tensors_of(source / "model.safetensors")
        assert size == 5 * 4 * 2
        assert torch.equal(copied["ctc_timestamp.head.weight"], original["ctc_timestamp.head.weight"])
        assert copied_metadata == {"format": "pt", "ctc_timestamp_format": "v1"}

    def test_weight_files_reads_the_index_before_the_single_file(self, scripts, tmp_path):
        source = make_source(tmp_path / "src")
        assert scripts.utils.weight_files(source) == ["model.safetensors"]
        llm = make_quantized_llm(tmp_path / "llm", FP8_CONFIG)
        assert scripts.utils.weight_files(llm) == ["model-00001-of-00001.safetensors"]

    def test_weight_map_of_an_unindexed_checkpoint_comes_from_its_header(self, scripts, tmp_path):
        llm = make_quantized_llm(tmp_path / "llm", FP8_CONFIG, single_file=True)
        assert scripts.utils.read_weight_map(llm) == {
            "backbone.layers.0.mixer.in_proj.weight": "model.safetensors",
            "backbone.layers.0.mixer.A_log": "model.safetensors",
            "backbone.layers.0.mixer.D": "model.safetensors",
        }

    @pytest.mark.parametrize("output", [".", "..", "child"])
    def test_outputs_that_overlap_an_input_are_refused(self, scripts, tmp_path, output):
        source = tmp_path / "work" / "src"
        source.mkdir(parents=True)
        with pytest.raises(ValueError, match="overlaps input"):
            scripts.utils.check_output_path(source / output, tmp_path / "other", source)
        scripts.utils.check_output_path(tmp_path / "work" / "out", source)

    def test_link_or_copy_copies_where_hard_links_fail(self, scripts, tmp_path, monkeypatch):
        (tmp_path / "a").write_text("weights")
        (tmp_path / "c").write_text("other")
        with pytest.raises(FileExistsError):
            scripts.utils.link_or_copy(tmp_path / "a", tmp_path / "c")
        no_hard_links(monkeypatch)
        scripts.utils.link_or_copy(tmp_path / "a", tmp_path / "b")
        assert (tmp_path / "b").read_text() == "weights"
        assert not (tmp_path / "a").samefile(tmp_path / "b")


class TestExportLlm:
    def test_nemo_names_become_nemotron_h_names(self, scripts):
        names = lambda name, tensor: dict(scripts.export.hf_names(name, tensor))  # noqa: E731
        assert list(names("llm.model.embed_tokens.weight", torch.zeros(1))) == ["backbone.embeddings.weight"]
        assert list(names("llm.model.norm.weight", torch.zeros(1))) == ["backbone.norm_f.weight"]
        assert list(names("llm.lm_head.weight", torch.zeros(1))) == ["lm_head.weight"]
        assert list(names("llm.model.layers.0.mixer._fp32_params.A_log", torch.zeros(1))) == [
            "backbone.layers.0.mixer.A_log"
        ]

    def test_stacked_experts_split_into_transposed_tensors(self, scripts):
        down = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        experts = dict(scripts.export.hf_names("llm.model.layers.1.mixer.experts.down_projs", down))
        assert sorted(experts) == [f"backbone.layers.1.mixer.experts.{i}.down_proj.weight" for i in range(2)]
        assert torch.equal(experts["backbone.layers.1.mixer.experts.1.down_proj.weight"], down[1].t())
        up = dict(scripts.export.hf_names("llm.model.layers.1.mixer.experts.gate_and_up_projs", down))
        assert "backbone.layers.1.mixer.experts.0.up_proj.weight" in up

    def test_mtp_head_and_bookkeeping_are_left_out(self, scripts, tmp_path):
        source = make_source(tmp_path / "src")
        names = [name for name, _ in scripts.export.hf_llm_weights(source)]
        assert sorted(names) == [
            "backbone.embeddings.weight",
            "backbone.layers.0.mixer.A_log",
            "backbone.layers.0.mixer.D",
            "backbone.layers.0.mixer.in_proj.weight",
            "lm_head.weight",
        ]

    @pytest.mark.parametrize("output", ["src", "."])
    def test_overwrite_never_deletes_the_checkpoint(self, scripts, tmp_path, monkeypatch, output):
        source = make_source(tmp_path / "src")
        with pytest.raises(ValueError, match="overlaps input"):
            run(scripts.export, monkeypatch, "--checkpoint", source, "--output", tmp_path / output, "--overwrite")
        assert (source / "model.safetensors").exists()


class TestAssemble:
    def test_fp8_assembly_copies_heads_and_excludes_them(self, scripts, tmp_path, monkeypatch):
        source = make_source(tmp_path / "src")
        llm = make_quantized_llm(tmp_path / "llm", FP8_CONFIG)
        out = tmp_path / "out"
        run(scripts.assemble, monkeypatch, "--source", source, "--quantized-llm", llm, "--output", out)

        assert not (out / "README.md").exists() and not (out / "SHA256SUMS").exists()
        assert (out / "tokenizer.json").read_text() == "tokenizer.json"
        assert (out / "llm_backbone" / "config.json").exists()
        original, _ = tensors_of(source / "model.safetensors")
        mtp, _ = tensors_of(out / "model-mtp-bf16.safetensors")
        ctc, ctc_metadata = tensors_of(out / "model-ctc-bf16.safetensors")
        assert torch.equal(mtp["llm.mtp.layers.0.eh_proj.weight"], original["llm.mtp.layers.0.eh_proj.weight"])
        assert ctc_metadata["ctc_timestamp_format"] == "v1"

        quant = read_json(out / "config.json")["quantization_config"]
        assert quant["ignore"] == ["lm_head", "mtp*", "ctc_timestamp*"]
        assert read_json(out / "hf_quant_config.json")["quantization"]["exclude_modules"] == [
            "mtp*",
            "ctc_timestamp*",
        ]
        weight_map = read_json(out / "model.safetensors.index.json")["weight_map"]
        assert weight_map["backbone.layers.0.mixer.in_proj.weight"] == "model-00001-of-00001.safetensors"
        assert weight_map["perception.proj.weight"] == "perception.safetensors"
        assert weight_map["ctc_timestamp.head.weight"] == "model-ctc-bf16.safetensors"
        assert "llm.model.embed_tokens.weight" not in weight_map

    def test_mixed_assembly_renames_per_layer_keys_and_adds_no_exclusions(self, scripts, tmp_path, monkeypatch):
        source = make_source(tmp_path / "src")
        llm = make_quantized_llm(tmp_path / "llm", MIXED_CONFIG)
        out = tmp_path / "out"
        run(scripts.assemble, monkeypatch, "--source", source, "--quantized-llm", llm, "--output", out)

        quant = read_json(out / "config.json")["quantization_config"]
        assert quant["quantized_layers"] == {"language_model.model.layers.0.mixer.in_proj": {"quant_algo": "FP8"}}
        assert "ignore" not in quant
        assert "exclude_modules" not in read_json(out / "hf_quant_config.json")["quantization"]

    def test_fp32_mamba_parameters_come_from_the_source(self, scripts, tmp_path, monkeypatch):
        source = make_source(tmp_path / "src")
        llm = make_quantized_llm(tmp_path / "llm", FP8_CONFIG)
        out = tmp_path / "out"
        run(scripts.assemble, monkeypatch, "--source", source, "--quantized-llm", llm, "--output", out, "--hardlink")

        decoder, _ = tensors_of(out / "model-00001-of-00001.safetensors")
        original, _ = tensors_of(source / "model.safetensors")
        assert decoder["backbone.layers.0.mixer.A_log"].dtype == torch.float32
        assert torch.equal(
            decoder["backbone.layers.0.mixer.A_log"], original["llm.model.layers.0.mixer._fp32_params.A_log"]
        )
        assert torch.equal(decoder["backbone.layers.0.mixer.D"], original["llm.model.layers.0.mixer._fp32_params.D"])
        assert decoder["backbone.layers.0.mixer.in_proj.weight"].dtype == torch.float8_e4m3fn
        exported, _ = tensors_of(llm / "model-00001-of-00001.safetensors")
        assert exported["backbone.layers.0.mixer.A_log"].dtype == torch.bfloat16  # the hard-linked export is untouched

    def test_skip_ctc_head(self, scripts, tmp_path, monkeypatch):
        source = make_source(tmp_path / "src")
        llm = make_quantized_llm(tmp_path / "llm", FP8_CONFIG)
        out = tmp_path / "out"
        run(
            scripts.assemble,
            monkeypatch,
            "--source",
            source,
            "--quantized-llm",
            llm,
            "--output",
            out,
            "--skip-ctc-head",
        )

        assert not (out / "model-ctc-bf16.safetensors").exists()
        assert read_json(out / "config.json")["quantization_config"]["ignore"] == ["lm_head", "mtp*"]

    def test_unindexed_single_file_export_assembles(self, scripts, tmp_path, monkeypatch):
        source = make_source(tmp_path / "src")
        llm = make_quantized_llm(tmp_path / "llm", FP8_CONFIG, single_file=True)
        out = tmp_path / "out"
        run(scripts.assemble, monkeypatch, "--source", source, "--quantized-llm", llm, "--output", out, "--hardlink")

        weight_map = read_json(out / "model.safetensors.index.json")["weight_map"]
        assert weight_map["backbone.layers.0.mixer.in_proj.weight"] == "model.safetensors"
        assert weight_map["perception.proj.weight"] == "perception.safetensors"
        decoder, _ = tensors_of(out / "model.safetensors")
        assert decoder["backbone.layers.0.mixer.A_log"].dtype == torch.float32

    def test_output_inside_the_source_is_refused(self, scripts, tmp_path, monkeypatch):
        source = make_source(tmp_path / "src")
        llm = make_quantized_llm(tmp_path / "llm", FP8_CONFIG)
        with pytest.raises(ValueError, match="overlaps input"):
            run(scripts.assemble, monkeypatch, "--source", source, "--quantized-llm", llm, "--output", source / "q")
        assert not (source / "q").exists()

    def test_refuses_names_older_modelopt_failed_to_resolve(self, scripts, tmp_path, monkeypatch):
        source = make_source(tmp_path / "src")
        llm = make_quantized_llm(tmp_path / "llm", {**FP8_CONFIG, "ignore": ["lm_head.\x00backbone.pt_name_sentinel"]})
        with pytest.raises(SystemExit, match="Model-Optimizer#2508"):
            run(
                scripts.assemble, monkeypatch, "--source", source, "--quantized-llm", llm, "--output", tmp_path / "out"
            )


def make_assembled(root, quant_config):
    """A tiny assembled checkpoint with a BF16 encoder (2 layers) and a BF16 MTP head."""
    root.mkdir()
    write_json(root / "config.json", {"quantization_config": json.loads(json.dumps(quant_config))})
    write_json(root / "hf_quant_config.json", {"quantization": json.loads(json.dumps(quant_config))})
    encoder = "perception.encoder.asr_encoder."
    perception = {
        encoder + "pre_encode.out.weight": torch.randn(4, 4).bfloat16(),
        "perception.proj.weight": torch.randn(4, 4).bfloat16(),
    }
    for layer in range(2):
        for target in ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3"):
            perception[f"{encoder}layers.{layer}.{target}.weight"] = torch.randn(8, 4).bfloat16()
    save_file(perception, root / "perception.safetensors", metadata={"format": "pt"})
    mtp = {
        "llm.mtp.layers.0.eh_proj.weight": torch.randn(4, 8).bfloat16(),
        "llm.mtp.layers.0.mixer.q_proj.weight": torch.randn(4, 4).bfloat16(),
        "llm.mtp.layers.0.mixer.o_proj.weight": torch.randn(4, 4).bfloat16(),
        "llm.mtp.layers.0.mixer.q_proj._extra_state": torch.zeros(0).bfloat16(),
        "llm.mtp.layers.1.mixer.experts.gate_and_up_projs": torch.randn(3, 4, 6).bfloat16(),
        "llm.mtp.layers.1.mixer.experts.down_projs": torch.randn(3, 6, 4).bfloat16(),
        "llm.mtp.layers.1.mixer.gate.weight": torch.randn(3, 4).bfloat16(),
        "llm.mtp.layers.1.norm.weight": torch.ones(4).bfloat16(),
    }
    save_file(mtp, root / "model-mtp-bf16.safetensors")
    weight_map = {
        **dict.fromkeys(perception, "perception.safetensors"),
        **dict.fromkeys(mtp, "model-mtp-bf16.safetensors"),
    }
    write_json(root / "model.safetensors.index.json", {"metadata": {}, "weight_map": weight_map})
    ranges = {
        "encoder": {
            f"layers.{layer}.{target}": 2.0
            for layer in range(2)
            for target in ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")
        },
        "mtp": {
            "model.layers.0.eh_proj": 4.0,
            "model.layers.0.mixer.qkv_proj": 3.0,
            "model.layers.0.mixer.o_proj": 1.5,
            "model.layers.1.mixer.experts.w13_input": 5.0,
            "model.layers.1.mixer.experts.w2_input": 50.0,
        },
    }
    write_json(root.parent / "ranges.json", ranges)
    return root


class TestEncoderMtpFp8:
    def test_fp8_config_writes_scales_and_moves_exclusions(self, scripts, tmp_path, monkeypatch):
        ckpt = make_assembled(
            tmp_path / "ckpt",
            {"quant_algo": "FP8", "ignore": ["lm_head", "mtp*"], "exclude_modules": ["lm_head", "mtp*"]},
        )
        out = tmp_path / "out"
        run(
            scripts.fp8,
            monkeypatch,
            "--checkpoint",
            ckpt,
            "--activation-ranges",
            tmp_path / "ranges.json",
            "--output",
            out,
        )

        perception, _ = tensors_of(out / "perception.safetensors")
        source, _ = tensors_of(ckpt / "perception.safetensors")
        name = "perception.encoder.asr_encoder.layers.1.ffn.net.0"
        assert perception[name + ".weight"].dtype == torch.float8_e4m3fn
        assert torch.equal(perception[name + ".weight_scale"], source[name + ".weight"].float().abs().max() / FP8_MAX)
        assert torch.allclose(perception[name + ".input_scale"], torch.tensor(2.0 * 1.5 / FP8_MAX))
        assert perception["perception.proj.weight"].dtype == torch.bfloat16

        mtp, _ = tensors_of(out / "model-mtp-fp8.safetensors")
        assert not (out / "model-mtp-bf16.safetensors").exists()
        assert "llm.mtp.layers.0.mixer.q_proj._extra_state" not in mtp
        assert mtp["llm.mtp.layers.0.mixer.q_proj.input_scale"] == pytest.approx(3.0 / FP8_MAX)
        assert mtp["llm.mtp.layers.1.mixer.experts.2.down_proj.weight"].shape == (4, 6)
        assert mtp["llm.mtp.layers.1.mixer.experts.2.down_proj.input_scale"] == pytest.approx(50.0 / FP8_MAX)
        assert mtp["llm.mtp.layers.1.mixer.gate.weight"].dtype == torch.bfloat16

        ignore = read_json(out / "config.json")["quantization_config"]["ignore"]
        assert "mtp*" not in ignore and "mtp.layers.1.mixer.gate" in ignore and "perception.proj" in ignore
        weight_map = read_json(out / "model.safetensors.index.json")["weight_map"]
        assert weight_map[name + ".input_scale"] == "perception.safetensors"

    def test_mixed_config_lists_fp8_rows(self, scripts, tmp_path, monkeypatch):
        ckpt = make_assembled(tmp_path / "ckpt", {"quant_algo": "MIXED_PRECISION", "quantized_layers": {}})
        out = tmp_path / "out"
        run(
            scripts.fp8,
            monkeypatch,
            "--checkpoint",
            ckpt,
            "--activation-ranges",
            tmp_path / "ranges.json",
            "--output",
            out,
        )

        rows = read_json(out / "config.json")["quantization_config"]["quantized_layers"]
        assert rows["perception.encoder.asr_encoder.layers.0.attn.w_qkv"] == {"quant_algo": "FP8"}
        assert rows["mtp.layers.1.mixer.experts"] == {"quant_algo": "FP8"}
        assert "perception.proj" not in rows and "mtp.layers.1.mixer.gate" not in rows

    def test_skip_mtp_keeps_the_bf16_head_excluded(self, scripts, tmp_path, monkeypatch):
        ckpt = make_assembled(
            tmp_path / "ckpt", {"quant_algo": "FP8", "ignore": ["mtp*"], "exclude_modules": ["mtp*"]}
        )
        out = tmp_path / "out"
        run(
            scripts.fp8,
            monkeypatch,
            "--checkpoint",
            ckpt,
            "--activation-ranges",
            tmp_path / "ranges.json",
            "--output",
            out,
            "--skip-mtp",
        )

        assert (out / "model-mtp-bf16.safetensors").exists() and not (out / "model-mtp-fp8.safetensors").exists()
        assert "mtp*" in read_json(out / "config.json")["quantization_config"]["ignore"]

    def test_skip_encoder_excludes_the_bf16_perception(self, scripts, tmp_path, monkeypatch):
        ckpt = make_assembled(
            tmp_path / "ckpt", {"quant_algo": "FP8", "ignore": ["mtp*"], "exclude_modules": ["mtp*"]}
        )
        out = tmp_path / "out"
        run(
            scripts.fp8,
            monkeypatch,
            "--checkpoint",
            ckpt,
            "--activation-ranges",
            tmp_path / "ranges.json",
            "--output",
            out,
            "--skip-encoder",
        )

        assert (out / "perception.safetensors").samefile(ckpt / "perception.safetensors")
        assert "perception*" in read_json(out / "config.json")["quantization_config"]["ignore"]
        assert "perception*" in read_json(out / "hf_quant_config.json")["quantization"]["exclude_modules"]

    def test_unchanged_files_are_copied_where_hard_links_fail(self, scripts, tmp_path, monkeypatch):
        ckpt = make_assembled(tmp_path / "ckpt", {"quant_algo": "FP8", "ignore": ["mtp*"]})
        (ckpt / "tokenizer.json").write_text("tokenizer")
        (ckpt / "llm_backbone").mkdir()
        (ckpt / "llm_backbone" / "config.json").write_text("{}")
        no_hard_links(monkeypatch)
        out = tmp_path / "out"
        run(
            scripts.fp8,
            monkeypatch,
            "--checkpoint",
            ckpt,
            "--activation-ranges",
            tmp_path / "ranges.json",
            "--output",
            out,
        )

        assert (out / "tokenizer.json").read_text() == "tokenizer"
        assert not (out / "tokenizer.json").samefile(ckpt / "tokenizer.json")
        assert (out / "llm_backbone" / "config.json").read_text() == "{}"

    def test_check_marks_perception_linears_as_the_plugin_builds_them(self, scripts):
        bf16 = {"perception.proj.weight": {"dtype": "BF16", "shape": [4, 4]}}
        listed = {"perception.proj": {"quant_algo": "FP8"}}
        check = lambda quant: scripts.fp8.check_output(bf16, {"quantization_config": quant})  # noqa: E731

        check({"quant_algo": "MIXED_PRECISION", "quantized_layers": listed, "ignore": ["perception.proj"]})
        check({"quant_algo": "FP8", "ignore": ["lm_head"]})  # no perception entry: the encoder predates FP8
        with pytest.raises(SystemExit, match="marks it FP8"):
            check({"quant_algo": "MIXED_PRECISION", "quantized_layers": listed})
        with pytest.raises(SystemExit, match="marks it FP8"):
            check({"quant_algo": "FP8", "ignore": ["perception.encoder.diarization_model*"]})

    def test_uncalibrated_encoder_layers_are_refused(self, scripts, tmp_path, monkeypatch):
        ckpt = make_assembled(tmp_path / "ckpt", {"quant_algo": "FP8", "ignore": ["mtp*"]})
        ranges = read_json(tmp_path / "ranges.json")
        del ranges["encoder"]["layers.1.ffn.net.3"]
        write_json(tmp_path / "ranges.json", ranges)
        with pytest.raises(SystemExit, match="differ from the calibrated ones"):
            run(
                scripts.fp8,
                monkeypatch,
                "--checkpoint",
                ckpt,
                "--activation-ranges",
                tmp_path / "ranges.json",
                "--output",
                tmp_path / "out",
            )


class TestCalibrateActivations:
    def test_expert_activations_follow_each_routed_expert(self, scripts):
        hidden_states = torch.randn(5, 4)
        topk_ids = torch.tensor([[0, 2], [2, 1], [0, 1], [2, 0], [1, 2]])
        w13 = torch.randn(3, 6, 4)

        activations = list(scripts.calibrate.expert_activations(hidden_states, topk_ids, w13))
        assert len(activations) == 3
        rows = torch.tensor([True, False, True, True, False])
        assert torch.equal(activations[0], torch.relu(hidden_states[rows] @ w13[0].t()).square())

    def test_ranges_inside_the_checkpoint_are_refused(self, scripts, tmp_path, monkeypatch):
        ckpt = make_assembled(tmp_path / "ckpt", {"quant_algo": "FP8", "ignore": ["mtp*"]})
        with pytest.raises(ValueError, match="overlaps input"):
            run(
                scripts.calibrate,
                monkeypatch,
                "--checkpoint",
                ckpt,
                "--manifest",
                tmp_path / "calib.jsonl",
                "--output",
                ckpt / "ranges.json",
            )

    def test_packed_expert_weights_are_refused(self, scripts):
        with pytest.raises(RuntimeError, match="unpacked"):
            next(scripts.calibrate.expert_activations(torch.randn(5, 4), torch.zeros(5, 2), torch.randn(3, 6, 2, 2)))
