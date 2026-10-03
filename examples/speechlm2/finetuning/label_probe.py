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
"""Linear probe on a frozen SALM's label log-probabilities (GTZAN).

Weight fine-tuning on GTZAN's 443 training clips erased genre knowledge the prompted model
already had. This keeps the model frozen and instead fits a multinomial logistic
regression on its per-label log-probabilities -- full clip + mean over 10 s windows, for
one or more prompts -- using only the *training* clips. Dev selects the regularisation;
test is scored once with the chosen setting.

Usage:
  genre_calib.py --train base_train.jsonl [p1_train.jsonl ...] --dev base_dev.jsonl [p1_dev.jsonl ...]
                 [--test base_test.jsonl [p1_test.jsonl ...] --l2 10 --out preds.jsonl]
Each list must be given in the same prompt order.
"""
import argparse
import collections
import json

import numpy as np


POOL_MAX = False


def log_softmax(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def features(paths, windows=True):
    per_src = []
    for p in paths:
        by = collections.defaultdict(dict)
        for line in open(p):
            r = json.loads(line)
            by[r["id"]][r["window"]] = r
        per_src.append(by)
    ids = sorted(per_src[0])
    labels = sorted(per_src[0][ids[0]]["full"]["scores"])
    X, y = [], []
    for i in ids:
        f = []
        for by in per_src:
            w = by[i]
            f.append([w["full"]["scores"][label] for label in labels])
            if windows:
                ws = [v for n, v in w.items() if n != "full"] or [w["full"]]
                wl = [log_softmax(np.array([[v["scores"][label] for label in labels]]))[0] for v in ws]
                f.append(np.mean(wl, 0))
                if POOL_MAX:
                    f.append(np.max(wl, 0))
        X.append(np.concatenate([log_softmax(np.array([b]))[0] for b in f]))
        y.append(labels.index(per_src[0][i]["full"]["text"]))
    return np.array(X), np.array(y), labels, ids


def fit(X, y, k, l2):
    """Multinomial logistic regression, L-BFGS, L2 on weights (not bias)."""
    from scipy.optimize import minimize

    d = X.shape[1]

    def f(theta):
        W, b = theta[: d * k].reshape(d, k), theta[d * k :]
        Z = X @ W + b
        Z = Z - Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        loss = (lse - Z[np.arange(len(y)), y]).mean() + 0.5 * l2 * (W**2).sum() / len(y)
        P = np.exp(Z - lse[:, None])
        P[np.arange(len(y)), y] -= 1
        P /= len(y)
        return loss, np.concatenate([(X.T @ P + l2 * W / len(y)).ravel(), P.sum(0)])

    r = minimize(f, np.zeros(d * k + k), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x[: d * k].reshape(d, k), r.x[d * k :]


def cv_acc(X, y, k, l2, folds=5):
    idx = np.random.default_rng(0).permutation(len(y))
    acc = []
    for i in range(folds):
        te = idx[i::folds]
        tr = np.setdiff1d(idx, te)
        W, b = fit(X[tr], y[tr], k, l2)
        acc.append(((X[te] @ W + b).argmax(1) == y[te]).mean())
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--test", nargs="+")
    ap.add_argument("--l2", type=float, default=None, help="fixed regularisation (else sweep on dev)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pool-max", action="store_true", help="add max-over-windows features")
    ap.add_argument(
        "--fit-on-dev-too",
        action="store_true",
        help="after selecting l2 by train-CV, refit on train+dev before scoring test",
    )
    a = ap.parse_args()
    global POOL_MAX
    POOL_MAX = a.pool_max
    Xt, yt, labels, _ = features(a.train)
    Xd, yd, _, _ = features(a.dev)
    k = len(labels)
    print(f"raw zero-shot (first source, full clip) dev acc {(Xd[:, :k].argmax(1) == yd).mean():.4f}")
    grid = [a.l2] if a.l2 is not None else [0.1, 0.3, 1, 3, 10, 30, 100]
    res = {}
    for l2 in grid:
        cv = cv_acc(Xt, yt, k, l2)
        W, b = fit(Xt, yt, k, l2)
        res[l2] = (cv, W, b)
        print(f"l2={l2:<6} train-CV {cv:.4f}  dev {((Xd @ W + b).argmax(1) == yd).mean():.4f}")
    # regularisation is chosen by 5-fold CV on the 443 training clips; dev stays a held-out check
    best = max(res, key=lambda z: res[z][0])
    _, W, b = res[best]
    print(f"selected l2={best} by train-CV ({res[best][0]:.4f}); dev {((Xd @ W + b).argmax(1) == yd).mean():.4f}")
    if a.test:
        Xs, ys, _, ids = features(a.test)
        _, W, b = res[best]
        if a.fit_on_dev_too:
            W, b = fit(np.vstack([Xt, Xd]), np.concatenate([yt, yd]), k, best)
        pred = (Xs @ W + b).argmax(1)
        print(f"TEST acc {(pred == ys).mean():.4f}  (raw zero-shot {(Xs[:, :k].argmax(1) == ys).mean():.4f})")
        if a.out:
            with open(a.out, "w") as f:
                for i, p, g in zip(ids, pred, ys):
                    f.write(json.dumps({"id": i, "text": labels[g], "pred_text": labels[p]}) + "\n")


if __name__ == "__main__":
    main()
