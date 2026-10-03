# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Run a SALM fine-tune with the safety checks that catch this stack's silent failures.

    python run_training.py --config my_ft.yaml [--smoke] [--keep-steps 240 420] [hydra overrides ...]

Stages:
  --preflight   check_checkpoint_coverage.py: every checkpoint tensor must load into the configured model
                (a mismatched encoder config otherwise loads silently with random weights).
  --smoke       20-step run first; fails unless the restore log is clean
                (``N / M layers are restored (N exact, 0 partial, 0 skipped``), no checkpoint tensor was
                dropped, the loss is finite, and -- if the checkpoint declares one -- the parallel-expert
                encoder was mounted.
  full run      ``torchrun --standalone examples/speechlm2/salm_train.py`` with a watchdog: if no
                ``val_loss`` has been logged by 1.6x the validation interval (in optimizer steps, i.e.
                accounting for ``accumulate_grad_batches``), validation is not running
                (``limit_train_batches`` >= batches per epoch) and the run is stopped.
  provenance    ``<exp_dir>/fingerprint.json`` records the base checkpoint, manifests, resolved config,
                overrides and code revision. Training resumes from an existing experiment directory only
                if they match; otherwise it refuses (use a new ``exp_dir``).
  --keep-steps  delete every saved checkpoint whose step is not listed, as soon as it is complete
                (checkpoints are ~2 bytes/parameter, tens of GB for a large model; configs use ``save_top_k: -1``).

The full training log goes to ``<exp_dir>/train.log``; progress is printed every few minutes.
A file lock (``$SALM_FT_WORK/.train.lock``) makes concurrent runs queue for the GPU.
"""
import argparse
import fcntl
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A Jupyter kernel exports MPLBACKEND=module://matplotlib_inline...; child processes that import matplotlib (NeMo does,
# via torchmetrics) fail on it when that backend is not installed in their environment.
os.environ.pop("MPLBACKEND", None)

NEMO_ROOT = Path(__file__).resolve().parents[3]


def events(exp):
    f = glob.glob(f"{exp}/events.out*")
    if not f:
        return {}
    from tensorboard.backend.event_processing import event_accumulator as E

    ea = E.EventAccumulator(f[0], size_guidance={"scalars": 0})
    ea.Reload()
    return {t: ea.Scalars(t) for t in ("loss", "val_loss") if t in ea.Tags()["scalars"]}


def torchrun_cmd(cfg, overrides):
    # --standalone picks a free rendezvous port, so independent one-GPU jobs on the same node don't collide
    # (a fixed --master_port fails with EADDRINUSE).
    return [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "examples/speechlm2/salm_train.py",
        f"--config-path={cfg.parent.absolute()}",
        f"--config-name={cfg.stem}",
        *overrides,
    ]


def torchrun(cfg, exp, overrides, log):
    with open(log, "w") as fo:
        return subprocess.Popen(torchrun_cmd(cfg, overrides), cwd=NEMO_ROOT, stdout=fo, stderr=subprocess.STDOUT)


def validation_interval_steps(trainer: dict) -> int:
    """Validation interval in optimizer steps (what the loss/val_loss logs and the checkpoint cadence count).

    Lightning's ``val_check_interval`` and ``limit_train_batches`` count dataloader batches; with gradient
    accumulation one optimizer step consumes ``accumulate_grad_batches`` of them.
    """
    micro = trainer.get("val_check_interval") or trainer["limit_train_batches"] * trainer.get(
        "check_val_every_n_epoch", 1
    )
    return max(1, math.ceil(micro / max(1, int(trainer.get("accumulate_grad_batches") or 1))))


def declares_parallel_expert(model_cfg: dict) -> bool:
    return any(model_cfg.get(k) not in (None, "", False, {}) for k in ("pe_encoder_path", "pe_encoder_config"))


def _manifests(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _manifests(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _manifests(v)
    elif isinstance(obj, str) and obj.endswith((".json", ".jsonl")) and os.path.isfile(obj):
        yield obj


def training_fingerprint(cfg: dict, overrides: list) -> dict:
    from fingerprint import checkpoint_digest, code_revision, config_digest, file_digest

    return {
        "base_checkpoint": checkpoint_digest(cfg["model"]["init_from_checkpoint"]),
        "manifests": {m: file_digest(m) for m in sorted(set(_manifests(cfg.get("data", {}))))},
        "config": config_digest({k: v for k, v in cfg.items() if k != "exp_manager"}),
        "overrides": sorted(overrides),
        "code": code_revision(),
    }


def check_provenance(exp: Path, cfg: dict, overrides: list):
    """Refuse to resume an experiment directory that was produced from different inputs."""
    from fingerprint import FingerprintMismatch, check_or_write

    fp = exp / "fingerprint.json"
    if not fp.exists() and any((exp / "checkpoints").glob("*.ckpt")):
        sys.exit(
            f"{exp} has checkpoints but no fingerprint.json, so its provenance is unknown. Use a new exp_dir or "
            f"delete {exp}."
        )
    try:
        check_or_write(fp, training_fingerprint(cfg, overrides), f"experiment {exp}")
    except FingerprintMismatch as e:
        sys.exit(str(e))


def keeper(ckdir, keep, stop):
    """Delete complete checkpoints whose step is not in `keep`, and `step=N-last.ckpt` duplicates of `step=N.ckpt`
    (each is a full ~66 GB copy). Runs until `stop` is set, then makes one final pass."""
    while True:
        done = stop.is_set()
        for c in glob.glob(f"{ckdir}/step=*.ckpt"):
            m = re.match(r"step=(\d+)(-last)?\.ckpt$", os.path.basename(c))
            if not m or glob.glob(f"{ckdir}/step={m.group(1)}*unfinished"):
                continue
            duplicate = m.group(2) and os.path.exists(f"{ckdir}/step={m.group(1)}.ckpt")
            if int(m.group(1)) not in keep or duplicate:
                shutil.rmtree(c, ignore_errors=True)
        if done:
            return
        stop.wait(20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--preflight", action="store_true", help="check_checkpoint_coverage.py before anything else")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--keep-steps", nargs="*", type=int, default=None)
    # Hydra overrides (key=value) may follow --keep-steps, whose nargs would otherwise swallow them.
    overrides = [x for x in sys.argv[1:] if "=" in x and not x.startswith("-")]
    a = p.parse_args([x for x in sys.argv[1:] if x not in overrides])
    cfg = yaml.safe_load(open(a.config))
    exp = Path(cfg["exp_manager"]["explicit_log_dir"])
    exp.mkdir(parents=True, exist_ok=True)
    t = cfg["trainer"]
    interval = validation_interval_steps(t)
    check_provenance(exp, cfg, overrides)

    lock_path = Path(os.environ.get("SALM_FT_WORK", "salm_ft_work")).absolute() / ".train.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    print(f"[train] waiting for the GPU training lock {lock_path} ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)

    if a.preflight:
        ck = cfg["model"]["init_from_checkpoint"]
        r = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "check_checkpoint_coverage.py"),
                "--checkpoint",
                ck,
                "--config",
                str(a.config),
            ],
            cwd=NEMO_ROOT,
            capture_output=True,
            text=True,
        )
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith(
                ("checkpoint tensors", "model parameters", "matched", "new LoRA", "OK:", "ERROR", "  ")
            ):
                print(f"[preflight] {line}")
        if r.returncode != 0:
            sys.exit("checkpoint coverage preflight failed: the configured model does not match the checkpoint")

    if a.smoke:
        log = exp.parent / f"{exp.name}.smoke.log"
        vci = ["trainer.val_check_interval=10"] if "val_check_interval" in t else []
        pr = torchrun(
            a.config,
            exp,
            [
                "trainer.max_steps=20",
                "trainer.limit_train_batches=20",
                "trainer.limit_val_batches=2",
                *vci,
                f"exp_manager.explicit_log_dir={exp}.smoke",
                "exp_manager.create_checkpoint_callback=false",
            ],
            log,
        )
        rc = pr.wait()
        text = log.read_text(errors="ignore")
        restored = re.findall(r"(\d+) / (\d+) layers are restored \((\d+) exact, (\d+) partial, (\d+) skipped", text)
        checks = {
            "smoke run exited cleanly": rc == 0,
            "parallel-expert encoder mounted (if declared)": (
                "Mounted ParallelExpertEncoder" in text or not declares_parallel_expert(cfg["model"])
            ),
            "restore log clean (0 partial, 0 skipped)": bool(restored)
            and all(r[3] == "0" and r[4] == "0" for r in restored),
            "no checkpoint tensors dropped": not re.search(r"were dropped|no matching parameter", text),
            "finite loss": not re.search(r"loss[^a-z]*nan", text, re.I),
        }
        for k, v in checks.items():
            print(f"[smoke] {'OK  ' if v else 'FAIL'} {k}")
        if restored:
            print(f"[smoke] {restored[0][0]} / {restored[0][1]} layers restored")
        shutil.rmtree(f"{exp}.smoke", ignore_errors=True)
        if not all(checks.values()):
            sys.exit(f"smoke run failed; see {log}")

    log = exp / "train.log"
    pr = torchrun(a.config, exp, overrides, log)
    stop = threading.Event()
    keep_thread = None
    if a.keep_steps is not None:
        keep_thread = threading.Thread(target=keeper, args=(exp / "checkpoints", set(a.keep_steps), stop), daemon=True)
        keep_thread.start()
    t0, last_print = time.time(), 0.0
    while pr.poll() is None:
        time.sleep(60)
        ev = events(exp)
        step = ev["loss"][-1].step + 1 if "loss" in ev else 0
        if "val_loss" not in ev and step > math.ceil(interval * 1.6):
            pr.terminate()
            sys.exit(
                f"no val_loss by step {step} (validation interval {interval}): validation is not running. "
                f"Lower limit_train_batches below the real batches per epoch (epoch_batches.py)."
            )
        if time.time() - last_print > 300:
            vl = (
                f"  val_loss {ev['val_loss'][-1].value:.4f} @ {ev['val_loss'][-1].step + 1}"
                if "val_loss" in ev
                else ""
            )
            print(
                (
                    f"[train] {(time.time() - t0) / 60:5.1f} min  step {step}  loss {ev['loss'][-1].value:.4f}{vl}"
                    if "loss" in ev
                    else f"[train] {(time.time() - t0) / 60:5.1f} min  starting ..."
                ),
                flush=True,
            )
            last_print = time.time()
    stop.set()
    if keep_thread is not None:
        keep_thread.join()  # final pass: prune checkpoints written in the last minute
    if pr.returncode != 0:
        sys.exit(f"training failed (rc={pr.returncode}); see {log}")
    ev = events(exp)
    print("[train] val_loss by step: " + ", ".join(f"{e.step + 1}: {e.value:.4f}" for e in ev.get("val_loss", [])))
    print("[train] checkpoints:", sorted(os.listdir(exp / "checkpoints")))


if __name__ == "__main__":
    main()
