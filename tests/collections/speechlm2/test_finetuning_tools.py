# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the SALM fine-tuning tools in examples/speechlm2/finetuning (CPU only)."""
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors.torch import save_file

FT = Path(__file__).resolve().parents[3] / "examples" / "speechlm2" / "finetuning"
sys.path.insert(0, str(FT))


def load(name):
    spec = importlib.util.spec_from_file_location(f"ft_{name}", FT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- training: torchrun, cadence, freeze policy


@pytest.mark.unit
def test_torchrun_uses_standalone_rendezvous():
    rt = load("run_training")
    cmd = rt.torchrun_cmd(Path("/tmp/cfg/my_ft.yaml"), ["trainer.max_steps=5"])
    assert "--standalone" in cmd
    assert not any(c.startswith("--master_port") for c in cmd)
    assert cmd[-1] == "trainer.max_steps=5"


@pytest.mark.unit
@pytest.mark.parametrize(
    "trainer,expected",
    [
        ({"val_check_interval": 100, "limit_train_batches": 100, "accumulate_grad_batches": 1}, 100),
        ({"val_check_interval": 100, "limit_train_batches": 100, "accumulate_grad_batches": 4}, 25),
        ({"val_check_interval": None, "limit_train_batches": 30, "check_val_every_n_epoch": 2}, 60),
        ({"limit_train_batches": 30, "check_val_every_n_epoch": 2, "accumulate_grad_batches": 3}, 20),
    ],
)
def test_validation_interval_counts_optimizer_steps(trainer, expected):
    assert load("run_training").validation_interval_steps(trainer) == expected


@pytest.mark.unit
def test_checkpoint_cadence_accounts_for_gradient_accumulation():
    mk = load("make_ft_config")
    assert mk.checkpoint_every_n_steps(Namespace(limit_train_batches=100, accumulate_grad_batches=4)) == 25
    assert mk.checkpoint_every_n_steps(Namespace(limit_train_batches=100, accumulate_grad_batches=1)) == 100
    with pytest.raises(SystemExit):
        mk.checkpoint_every_n_steps(Namespace(limit_train_batches=100, accumulate_grad_batches=3))


def _freeze_args(**kw):
    base = dict(train_encoder=False, train_proj=False, tune_mode="lora", unfreeze_llm_layers=8, llm_num_layers=52)
    base.update(kw)
    return Namespace(**base)


@pytest.mark.unit
def test_train_encoder_targets_plain_encoder_without_parallel_expert():
    mk = load("make_ft_config")
    _, prevent = mk.build_freeze_policy(_freeze_args(train_encoder=True), {})
    assert any(__import__("re").match(p, "perception.encoder.layers.0.weight") for p in prevent)
    assert not any("asr_encoder" in p for p in prevent)


@pytest.mark.unit
def test_train_encoder_targets_asr_branch_with_parallel_expert():
    mk = load("make_ft_config")
    _, prevent = mk.build_freeze_policy(_freeze_args(train_encoder=True), {"pe_encoder_config": {"x": 1}})
    import re

    assert any(re.match(p, "perception.encoder.asr_encoder.layers.0.weight") for p in prevent)
    assert not any(re.match(p, "perception.encoder.diar_encoder.layers.0.weight") for p in prevent)


@pytest.mark.unit
@pytest.mark.parametrize("train_encoder,train_proj", [(False, False), (True, False), (False, True), (True, True)])
def test_full_mode_trains_llm_and_keeps_perception_under_its_flags(train_encoder, train_proj):
    import re

    mk = load("make_ft_config")
    freeze, prevent = mk.build_freeze_policy(
        _freeze_args(tune_mode="full", train_encoder=train_encoder, train_proj=train_proj), {}
    )

    def trainable(name):
        frozen = any(re.match(p, name) for p in freeze)
        return not frozen or any(re.match(p, name) for p in prevent)

    assert trainable("llm.model.layers.0.mixer.weight")
    assert trainable("perception.encoder.layers.0.weight") == train_encoder
    assert trainable("perception.proj.weight") == train_proj
    assert not trainable("perception.preprocessor.featurizer.window")


@pytest.mark.unit
def test_smoke_requires_parallel_expert_only_when_declared():
    rt = load("run_training")
    assert rt.declares_parallel_expert({"pe_encoder_path": "/x.nemo"})
    assert rt.declares_parallel_expert({"pe_encoder_config": {"a": 1}})
    assert not rt.declares_parallel_expert({"pe_encoder_path": None, "pe_encoder_config": {}})


# ---------------------------------------------------------------- provenance fingerprints


def _fake_checkpoint(path: Path, value: float):
    path.mkdir(parents=True)
    save_file({"w": torch.full((4, 4), value)}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"a": 1}))
    return path


@pytest.mark.unit
def test_checkpoint_digest_detects_weight_changes(tmp_path, monkeypatch):
    fpm = load("fingerprint")
    a = _fake_checkpoint(tmp_path / "a", 1.0)
    b = _fake_checkpoint(tmp_path / "b", 2.0)
    assert fpm.checkpoint_digest(a) != fpm.checkpoint_digest(b)
    monkeypatch.setattr(fpm, "FULL_HASH_LIMIT", 16)  # force the sampled path
    assert fpm.file_digest(a / "model.safetensors").startswith("sampled:")
    assert fpm.checkpoint_digest(a) != fpm.checkpoint_digest(b)


@pytest.mark.unit
def test_check_or_write_refuses_mismatch(tmp_path):
    fpm = load("fingerprint")
    p = tmp_path / "exp" / "fingerprint.json"
    assert fpm.check_or_write(p, {"base": "x", "code": "1"}, "exp") is False
    assert fpm.check_or_write(p, {"base": "x", "code": "1"}, "exp") is True
    with pytest.raises(fpm.FingerprintMismatch, match="base"):
        fpm.check_or_write(p, {"base": "y", "code": "1"}, "exp")


@pytest.mark.unit
def test_training_refuses_to_resume_with_a_different_base(tmp_path, monkeypatch):
    rt = load("run_training")
    exp = tmp_path / "exp"
    exp.mkdir()
    cfg = {"model": {"init_from_checkpoint": "x"}, "trainer": {}}
    fps = iter([{"base_checkpoint": "A"}, {"base_checkpoint": "B"}])
    monkeypatch.setattr(rt, "training_fingerprint", lambda cfg, overrides: next(fps))
    rt.check_provenance(exp, cfg, [])  # writes the fingerprint
    with pytest.raises(SystemExit, match="base_checkpoint"):
        rt.check_provenance(exp, cfg, [])


@pytest.mark.unit
def test_training_refuses_checkpoints_of_unknown_provenance(tmp_path, monkeypatch):
    rt = load("run_training")
    (tmp_path / "exp" / "checkpoints" / "step=10.ckpt").mkdir(parents=True)
    monkeypatch.setattr(rt, "training_fingerprint", lambda cfg, overrides: {"a": 1})
    with pytest.raises(SystemExit, match="provenance is unknown"):
        rt.check_provenance(tmp_path / "exp", {"model": {}}, [])


# ---------------------------------------------------------------- export and averaging


@pytest.mark.unit
def test_export_rejects_partial_and_mismatched_outputs(tmp_path):
    ex = load("export_checkpoint")
    out = tmp_path / "export"
    assert ex.check_existing(out, {"step": 1}) is False
    out.mkdir()
    with pytest.raises(SystemExit, match="not a complete export"):
        ex.check_existing(out, {"step": 1})
    save_file({"w": torch.zeros(1)}, str(out / "model.safetensors"))
    (out / "provenance.json").write_text(json.dumps({"step": 1}))
    assert ex.check_existing(out, {"step": 1}) is True
    with pytest.raises(ex.FingerprintMismatch):
        ex.check_existing(out, {"step": 2})


@pytest.mark.unit
def test_export_detects_lora_tensors(tmp_path):
    ex = load("export_checkpoint")
    plain = tmp_path / "plain.safetensors"
    save_file({"llm.layers.0.q_proj.weight": torch.zeros(2, 2)}, str(plain))
    lora = tmp_path / "lora.safetensors"
    save_file(
        {
            "llm.layers.0.q_proj.weight": torch.zeros(2, 2),
            "llm.layers.0.q_proj.lora_A.weight": torch.zeros(1, 2),
            "llm.layers.0.q_proj.lora_B.weight": torch.ones(2, 1),
        },
        str(lora),
    )
    assert ex.lora_keys(plain) == []
    assert sorted(ex.lora_keys(lora)) == ["llm.layers.0.q_proj.lora_A.weight", "llm.layers.0.q_proj.lora_B.weight"]


@pytest.mark.unit
def test_average_checkpoints_is_atomic_and_provenance_checked(tmp_path):
    a = _fake_checkpoint(tmp_path / "a", 1.0)
    b = _fake_checkpoint(tmp_path / "b", 3.0)
    c = _fake_checkpoint(tmp_path / "c", 5.0)
    dst = tmp_path / "avg"
    run = lambda *x: subprocess.run(  # noqa: E731
        [sys.executable, str(FT / "average_checkpoints.py"), *map(str, x)], capture_output=True, text=True
    )
    r = run("--src", a, b, "--dst", dst)
    assert r.returncode == 0, r.stderr
    from safetensors.torch import load_file

    assert torch.allclose(load_file(str(dst / "model.safetensors"))["w"], torch.full((4, 4), 2.0))
    assert not (tmp_path / "avg.unfinished").exists()
    assert run("--src", a, b, "--dst", dst).returncode != 0  # existing destination without --reuse
    assert run("--src", a, b, "--dst", dst, "--reuse").returncode == 0
    assert run("--src", a, c, "--dst", dst, "--reuse").returncode != 0  # different sources
    (tmp_path / "avg2.unfinished").mkdir()
    assert run("--src", a, b, "--dst", tmp_path / "avg2").returncode != 0  # interrupted earlier run


# ---------------------------------------------------------------- evaluation: exact label scores, metrics


class _Tok:
    """Whitespace tokenizer: one token per character-group, enough to test span arithmetic."""

    def encode(self, text, add_special_tokens=False):
        return text.replace("<|im_end|>", " <|im_end|> ").split()


@pytest.mark.unit
def test_label_spans_cover_label_and_end_token():
    ev = load("vllm_task_eval")
    text, k = ev.label_logprob_spans(_Tok(), "prompt words", " spoof")
    assert text == "prompt words spoof<|im_end|>"
    assert k == 2  # "spoof" + "<|im_end|>"


@pytest.mark.unit
def test_sum_label_logprob_is_exact_and_fails_on_missing_token():
    ev = load("vllm_task_eval")
    lp = lambda v: Namespace(logprob=v)  # noqa: E731
    token_ids = [5, 6, 7]
    prompt_logprobs = [None, {6: lp(-1.0)}, {7: lp(-0.5), 9: lp(-0.1)}]
    assert ev.sum_label_logprob(prompt_logprobs, token_ids, 2) == pytest.approx(-1.5)
    with pytest.raises(RuntimeError):
        ev.sum_label_logprob([None, {6: lp(-1.0)}, {9: lp(-0.1)}], token_ids, 2)


@pytest.mark.unit
def test_eer_rejects_degenerate_inputs():
    tm = load("task_metrics")
    assert tm.eer([0.9, 0.1, 0.8, 0.2], [True, False, True, False]) == 0.0
    with pytest.raises(ValueError, match="both"):
        tm.eer([0.1, 0.2], [True, True])
    with pytest.raises(ValueError, match="at least one"):
        tm.eer([], [])
    with pytest.raises(ValueError, match="label_score"):
        tm.score_eer([{"text": "spoof", "pred_text": "spoof"}, {"text": "bonafide", "pred_text": "spoof"}])


@pytest.mark.unit
def test_metrics_reject_empty_inputs():
    tm = load("task_metrics")
    with pytest.raises(ValueError, match="no rows"):
        tm.score("cls", [])
    with pytest.raises(ValueError, match="at least two"):
        tm.score_pcc([{"scores": {}, "pred_text": ""}])


@pytest.mark.unit
def test_wer_and_mer_are_corpus_level_edit_rates():
    tm = load("task_metrics")
    rows = [
        {"text": "the cat sat", "pred_text": "the cat sat down"},  # 1 insertion
        {"text": "on the  mat", "pred_text": "on a mat"},  # 1 substitution; extra spaces don't count
        {"text": "", "pred_text": "ignored"},  # empty references are skipped
    ]
    wer = tm.score_wer(rows, normalizer="none")
    stats = wer["by_normalizer"]["none"]
    assert wer["score"] == pytest.approx(2 / 6)
    assert (stats["ins"], stats["sub"], stats["del"]) == (pytest.approx(1 / 6), pytest.approx(1 / 6), 0)
    assert stats["cer"] == pytest.approx((5 + 3) / (11 + 10))  # " down", "the" -> "a"
    mer = tm.score_mer([{"text": "我想 go home", "pred_text": "我要 go home"}])  # CJK scored per character
    assert mer["score"] == pytest.approx(1 / 4)


@pytest.mark.unit
def test_dependency_check_runs_before_decoding(monkeypatch):
    tm = load("task_metrics")
    monkeypatch.setitem(tm.TASK_DEPENDENCIES, "asr", ["definitely_not_installed_module_xyz"])
    with pytest.raises(SystemExit, match="definitely_not_installed_module_xyz"):
        tm.check_dependencies("asr")
    tm.check_dependencies("cls")  # no extra dependencies


# ---------------------------------------------------------------- TTA, ROVER, manifests, prompt search


@pytest.mark.unit
def test_tta_reads_only_the_manifest_segment(tmp_path):
    tta = load("make_tta")
    sr = 16000
    x = np.arange(3 * sr, dtype=np.float32) / (3 * sr)
    wav = tmp_path / "a.wav"
    sf.write(str(wav), x, sr, subtype="FLOAT")
    seg, got_sr = tta.read_segment({"audio_filepath": str(wav), "offset": 1.0, "duration": 0.5})
    assert got_sr == sr and len(seg) == sr // 2
    np.testing.assert_allclose(seg, x[sr : sr + sr // 2], atol=1e-6)
    full, _ = tta.read_segment({"audio_filepath": str(wav)})
    assert len(full) == len(x)
    with pytest.raises(ValueError, match="outside the file"):
        tta.read_segment({"audio_filepath": str(wav), "offset": 2.9, "duration": 0.5})


@pytest.mark.unit
def test_rover_votes_on_insertions():
    rv = load("rover_combine")
    assert rv.rover([["a", "c"], ["a", "b", "c"], ["a", "b", "c"]]) == ["a", "b", "c"]  # majority insertion wins
    assert rv.rover([["a", "c"], ["a", "b", "c"], ["a", "c"]]) == ["a", "c"]  # minority insertion is dropped
    assert rv.rover([["x", "b"], ["y", "b"], ["z", "b"]]) == ["x", "b"]  # ties keep the base system
    assert rv.rover([["b"], ["a", "b"], ["a", "b"]]) == ["a", "b"]  # insertion before the first word


@pytest.mark.unit
def test_shuffle_manifest_replaces_atomically(tmp_path):
    p = tmp_path / "train.json"
    rows = [json.dumps({"text": str(i % 3)}) for i in range(50)]
    p.write_text("\n".join(rows) + "\n")
    r = subprocess.run([sys.executable, str(FT / "shuffle_manifest.py"), str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(p.read_text().splitlines()) == sorted(rows)
    assert not list(tmp_path.glob(".shuffle.*"))


@pytest.mark.unit
@pytest.mark.parametrize("bad", [["--count", "0"], ["--per-class", "-1"], ["--fraction", "0"], ["--fraction", "1.5"]])
def test_subset_manifest_rejects_invalid_sizes(tmp_path, bad):
    p = tmp_path / "train.json"
    p.write_text(json.dumps({"text": "a"}) + "\n")
    r = subprocess.run(
        [sys.executable, str(FT / "subset_manifest.py"), "--in", str(p), "--out", str(tmp_path / "o.json"), *bad],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and "must be" in r.stderr


@pytest.mark.unit
def test_prompt_search_runs_without_a_shell(monkeypatch):
    ps = load("prompt_search")
    calls = []
    monkeypatch.setattr(ps.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    ps.run(["python", "x.py", "--prompt", "it's a; rm -rf /"])
    argv, kw = calls[0]
    assert isinstance(argv, list) and argv[-1] == "it's a; rm -rf /"
    assert not kw.get("shell")
