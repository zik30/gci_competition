"""Sprint 3 Task B1: fixed-formula rank-average blend (R2-compliant). Run from the project root:

    .venv/bin/python src/blend.py --exp E21 --lgb E19 --xgb E20 --hypothesis "..."

blend = w * rank(lgb) + (1 - w) * rank(xgb), ranks = rankdata(p) / len(p), computed separately for OOF and
test. w (LightGBM weight) is chosen on a coarse grid 0.0, 0.1, ..., 1.0 by OOF AUC. The blend is compared
with the LightGBM component (the best single model) using the original rule incl. CONFIRM; for CONFIRM both
components are re-run on the alternative fold sets (seeds 1, 2, 3) and blended with the same w.
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import SEED, compare  # noqa: E402
from run_experiment3 import ALT_FOLD_SEEDS, alt_folds, load_config, run_config  # noqa: E402
from utils import log_experiment, save_artifacts, seed_everything  # noqa: E402

GRID = [round(0.1 * i, 1) for i in range(11)]


def rank01(p):
    return rankdata(p) / len(p)


def blend(p_lgb, p_xgb, w):
    return w * rank01(p_lgb) + (1 - w) * rank01(p_xgb)


def fold_aucs(y, p, folds):
    return [roc_auc_score(y[folds == k], p[folds == k]) for k in range(folds.max() + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--lgb", required=True)
    ap.add_argument("--xgb", required=True)
    ap.add_argument("--hypothesis", required=True)
    a = ap.parse_args()
    seed_everything(SEED)
    t0 = time.time()
    y = pd.read_csv("input/train.csv", usecols=["TARGET"])["TARGET"].values
    folds = np.load("outputs/folds.npy")
    o_l, o_x = np.load(f"outputs/oof/{a.lgb}_oof.npy"), np.load(f"outputs/oof/{a.xgb}_oof.npy")
    t_l, t_x = np.load(f"outputs/test_preds/{a.lgb}_test.npy"), np.load(f"outputs/test_preds/{a.xgb}_test.npy")

    rho_oof = spearmanr(o_l, o_x).correlation
    grid = {w: roc_auc_score(y, blend(o_l, o_x, w)) for w in GRID}
    w = max(grid, key=lambda k: (grid[k], k))          # ties -> larger LightGBM weight
    print(f"Spearman(LGB, XGB) OOF = {rho_oof:.4f}")
    print("OOF AUC by w:", {k: round(v, 5) for k, v in grid.items()}, "-> w =", w)
    oof_b, test_b = blend(o_l, o_x, w), blend(t_l, t_x, w)
    fa_b, fa_l = fold_aucs(y, oof_b, folds), fold_aucs(y, o_l, folds)
    verdict = compare(fa_b, fa_l)
    notes = [f"w_lgb={w}", f"spearman_oof={rho_oof:.4f}", "grid " + ", ".join(f"{k}:{v:.5f}" for k, v in grid.items())]
    if verdict.startswith("CONFIRM"):
        train, test = pd.read_csv("input/train.csv"), pd.read_csv("input/test.csv")
        ys = train["TARGET"]
        cl, cx = load_config(a.lgb), load_config(a.xgb)
        sets = {"main": (float(np.mean(fa_l)), float(np.mean(fa_b)))}
        for s in ALT_FOLD_SEEDS:
            f = alt_folds(y, s)
            ol, _, al, _, _ = run_config(cl, train, test, ys, f, quiet=True)
            ox, _, _, _, _ = run_config(cx, train, test, ys, f, quiet=True)
            sets[f"seed{s}"] = (float(np.mean(al)), float(np.mean(fold_aucs(y, blend(ol, ox, w), f))))
            print(f"  fold seed {s}: lgb {sets[f'seed{s}'][0]:.5f} | blend {sets[f'seed{s}'][1]:.5f}")
        p20, c20 = np.mean([v[0] for v in sets.values()]), np.mean([v[1] for v in sets.values()])
        wins = sum(v[1] > v[0] for v in sets.values())
        verdict = "KEEP (CONFIRMED)" if (c20 > p20 and wins >= 3) else "DISCARD (CONFIRM failed)"
        notes.append("alt folds: " + "; ".join(f"{k} {v[0]:.5f}->{v[1]:.5f}" for k, v in sets.items())
                     + f"; 20-fold {p20:.5f}->{c20:.5f}; set wins {wins}/4")
    d = float(np.mean(fa_b) - np.mean(fa_l))
    wins5 = int((np.asarray(fa_b) > np.asarray(fa_l)).sum())
    save_artifacts(a.exp, oof_b, test_b, {"model": "blend", "components": [a.lgb, a.xgb], "weight_lgb": w,
                                          "formula": "w*rank(lgb)+(1-w)*rank(xgb), rank=rankdata/len"},
                   [a.lgb, a.xgb], [], out_dir="outputs")
    log_experiment({
        "exp_id": a.exp, "date": dt.date.today().isoformat(), "parent_id": a.lgb, "hypothesis": a.hypothesis,
        "change": f"rank-average blend of {a.lgb} (w={w}) and {a.xgb} (1-w)", "model": "blend", "n_features": "",
        **{f"fold_{k+1}": round(v, 5) for k, v in enumerate(fa_b)},
        "mean_auc": round(float(np.mean(fa_b)), 5), "std_auc": round(float(np.std(fa_b)), 5),
        "oof_auc": round(roc_auc_score(y, oof_b), 5), "delta_vs_parent": f"{d:+.5f}", "folds_improved": f"{wins5}/5",
        "verdict": verdict, "public_lb": "", "notes": "; ".join(notes + [f"{time.time()-t0:.0f}s"]),
    })
    print(f"SUMMARY {a.exp} | parent {a.lgb} | Δmean {d:+.5f} | {wins5}/5 | {verdict} | mean {np.mean(fa_b):.5f} | "
          f"folds {np.round(fa_b, 5).tolist()} | w {w} | rho {rho_oof:.4f}")


if __name__ == "__main__":
    main()
