"""Sprint-2 experiment runner (Task 2.2). Run from the project root, e.g.

    .venv/bin/python src/run_experiment.py --exp E3 --parent E2 --flags days_employed_fix \
        --hypothesis "Replacing DAYS_EMPLOYED=365243 with NaN + flag improves AUC"

Exactly ONE thing may differ from the parent: one feature flag OR the model. Results are judged with
compare_sprint2() (Sprint-2 decision rule), artifacts saved, the log appended.
"""
import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import SEED, compare_sprint2, run_cv_lgb  # noqa: E402
from features import FEATURE_FLAGS_DEFAULT, make_features  # noqa: E402
from utils import log_experiment, save_artifacts, seed_everything  # noqa: E402

# Fixed LightGBM params for the whole sprint (docs/sprints/sprint-2.md §2) — do not change in Sprint 2
BASE_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=1.0,
)
LOG = Path("experiments/experiment_log.csv")


def load_parent(parent):
    meta = json.load(open(f"outputs/params/{parent}.json"))
    p = meta["params"]
    flags = {**FEATURE_FLAGS_DEFAULT, **p.get("flags", {})}
    model = p.get("model", "rf")             # E0.json predates the "model" key -> RF baseline
    # Parent fold AUCs recomputed at full precision from its saved OOF predictions (the log is rounded
    # to 5 decimals, which would make identical models look like tiny "wins").
    y = pd.read_csv("input/train.csv", usecols=["TARGET"])["TARGET"].values
    folds_idx = np.load("outputs/folds.npy")
    oof = np.load(f"outputs/oof/{parent}_oof.npy")
    folds = [roc_auc_score(y[folds_idx == k], oof[folds_idx == k]) for k in range(5)]
    return flags, model, folds


def low_importance_list(parent, X_tr):
    """E10: zero-total-gain features in the parent's importance file + near-constant columns."""
    imp = pd.read_csv(f"outputs/importance/{parent}.csv", index_col=0)["gain_mean"]
    drop = set(imp[imp <= 0].index)
    for c in X_tr.columns:
        top = X_tr[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top >= 0.999:                       # e.g. FLAG_MOBIL
            drop.add(c)
    return sorted(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--flags", default=None, help="the ONE feature flag to switch on")
    ap.add_argument("--model", default=None, choices=["lgb"], help="model change (E1 only)")
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--change", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--analysis", action="store_true", help="analysis experiment: verdict ANALYSIS")
    a = ap.parse_args()
    seed_everything(SEED)
    t0 = time.time()

    with open(LOG, newline="") as f:
        if any(r["exp_id"] == a.exp for r in csv.DictReader(f)):
            sys.exit(f"{a.exp} already in the log - experiments are never overwritten")

    p_flags, p_model, p_folds = load_parent(a.parent)
    flags, model = dict(p_flags), a.model or p_model
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    y = train["TARGET"]

    if a.flags:
        assert a.flags in FEATURE_FLAGS_DEFAULT, f"unknown flag {a.flags}"
        assert not flags[a.flags], f"{a.flags} already on in {a.parent}"
        if a.flags == "drop_low_importance":
            X_par, _, _, _ = make_features(train, test, p_flags)
            flags[a.flags] = low_importance_list(a.parent, X_par)
            print(f"drop list ({len(flags[a.flags])}): {flags[a.flags]}")
        else:
            flags[a.flags] = True
    n_changes = sum(flags[k] != p_flags[k] for k in flags) + int(model != p_model)
    if n_changes != 1:
        sys.exit(f"REFUSED: {n_changes} things differ from {a.parent} (exactly 1 allowed)")
    assert model == "lgb", "Sprint 2 experiments are LightGBM"

    X_tr, X_te, names, cats = make_features(train, test, flags)
    folds = np.load("outputs/folds.npy")
    assert len(folds) == len(X_tr)
    imps = []
    oof, test_pred, fold_aucs, best_iters = run_cv_lgb(
        X_tr, y, X_te, BASE_PARAMS, folds, seed=SEED, cat_features=cats or "auto", importance_out=imps)
    elapsed = time.time() - t0

    # Sanity guards (sprint-2.md §4)
    guards = []
    if max(fold_aucs) > 0.80:
        guards.append("fold AUC > 0.80 (leakage?)")
    if max(best_iters) >= 10000 or min(best_iters) < 50:
        guards.append(f"best_iteration out of range {best_iters}")
    if elapsed > 300:
        guards.append(f"run time {elapsed:.0f}s > 300s")

    removal = a.flags in ("drop_low_importance", "drop_gender")
    verdict, d_mean, wins = compare_sprint2(fold_aucs, p_folds, removal=removal)
    if a.analysis:
        verdict = f"ANALYSIS (rule: {verdict})"

    imp = pd.concat(imps, axis=1)
    Path("outputs/importance").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gain_mean": imp.mean(axis=1), "gain_std": imp.std(axis=1)}).sort_values(
        "gain_mean", ascending=False).to_csv(f"outputs/importance/{a.exp}.csv")
    params = {"model": model, "lgb_params": BASE_PARAMS, "flags": flags}
    save_artifacts(a.exp, oof, test_pred, params, names, best_iters, out_dir="outputs")
    changed = a.flags or f"model {p_model} -> {model}"
    log_experiment({
        "exp_id": a.exp, "date": dt.date.today().isoformat(), "parent_id": a.parent,
        "hypothesis": a.hypothesis, "change": a.change or changed, "model": model, "n_features": len(names),
        **{f"fold_{k+1}": round(v, 5) for k, v in enumerate(fold_aucs)},
        "mean_auc": round(float(np.mean(fold_aucs)), 5), "std_auc": round(float(np.std(fold_aucs)), 5),
        "oof_auc": round(roc_auc_score(y, oof), 5), "delta_vs_parent": f"{d_mean:+.5f}",
        "folds_improved": f"{wins}/5", "verdict": verdict, "public_lb": "",
        "notes": "; ".join(filter(None, [a.notes, f"best_iters={best_iters}", f"{elapsed:.0f}s"] + guards)),
    }, log_path=LOG)
    print(f"SUMMARY {a.exp} | parent {a.parent} | Δmean {d_mean:+.4f} | {wins}/5 | {verdict} | "
          f"mean {np.mean(fold_aucs):.4f} | folds {np.round(fold_aucs, 5).tolist()} | {len(names)} feats | "
          f"iters {best_iters} | {elapsed:.0f}s")
    if guards:
        print("SANITY GUARD TRIGGERED:", guards)
        sys.exit(2)


if __name__ == "__main__":
    main()
