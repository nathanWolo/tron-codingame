#!/usr/bin/env python3
"""Fit logistic models on features.py output and compare feature sets.

usage: fit.py FEAT.npz
"""
import sys

import numpy as np

CURRENT = {"terr": 50, "edges": 12, "mob": 6, "front": 4}


def fit_logit(X, y, l2=1e-3, iters=60):
    """Newton's method logistic regression (with intercept)."""
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(k + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + l2 * w
        s = p * (1 - p)
        Hm = (Xb * s[:, None]).T @ Xb + l2 * np.eye(k + 1)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def loglik(X, y, w):
    Xb = np.hstack([X, np.ones((len(y), 1))])
    z = Xb @ w
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def acc(X, y, w):
    Xb = np.hstack([X, np.ones((len(y), 1))])
    return float(np.mean(((Xb @ w) > 0) == (y > 0.5)))


def main():
    d = np.load(sys.argv[1])
    X, y, names = d["X"], d["y"], list(d["names"])
    n = len(y)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    def cols(sel):
        return [names.index(s) for s in sel]

    # scale features for conditioning
    scale = X.std(axis=0) + 1e-9

    def run(sel, label):
        c = cols(sel)
        w = fit_logit(Xtr[:, c] / scale[c], ytr)
        ll = loglik(Xte[:, c] / scale[c], yte, w)
        a = acc(Xte[:, c] / scale[c], yte, w)
        # express weights per raw unit, relative to terr
        raw = w[:-1] / scale[c]
        rel = raw / raw[sel.index("terr")] * 50 if "terr" in sel else raw
        print(f"{label:40s} test loglik {ll:.4f}  acc {a:.3f}")
        print("    weights (terr=50 scale):", ", ".join(f"{s}={r:.2f}" for s, r in zip(sel, rel)))
        return ll

    base = ["terr", "edges", "mob", "front", "center"]
    run(["terr"], "terr only")
    run(base, "current 5 features (refit)")
    # current fixed weights as a single score
    c = cols(base)
    score = X[:, c] @ np.array([50, 12, 6, 4, 6.0])
    w = fit_logit(score[tr, None] / score.std(), ytr)
    print(f"{'current eval as-is (1 slope)':40s} test loglik {loglik(score[te, None] / score.std(), yte, w):.4f}  acc {acc(score[te, None] / score.std(), yte, w):.3f}")
    for extra in ["ties", "near", "far", "teeth", "cb", "reach", "tomove", "corridor", "wall_adj", "terr_x_phase"]:
        run(base + [extra], f"5 + {extra}")
    run(base + ["near", "far", "teeth", "cb", "tomove", "corridor", "wall_adj"], "5 + all structural")
    run(names, "all features")


if __name__ == "__main__":
    main()
