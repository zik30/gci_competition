"""Sprint 3 Task T1: Optuna tuning of LightGBM on the E10 feature set. Run from the project root:

    .venv/bin/python src/tune_lgb.py [--trials 60]

Objective = mean fold AUC over outputs/folds.npy via run_cv_lgb (never a single split), lr fixed at 0.05.
TPESampler(seed=42), sequential trials (n_jobs=1) -> the study is reproducible. No pruning.
Same search code is inlined in the notebook behind RUN_OPTUNA = False.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import SEED, run_cv_lgb  # noqa: E402
from features import make_features  # noqa: E402
from utils import seed_everything  # noqa: E402

FIXED = dict(objective="binary", learning_rate=0.05, bagging_freq=1)


def lgb_search_space(trial):
    # Search space from docs/sprints/sprint-3.md §3
    return {
        "num_leaves":        trial.suggest_int("num_leaves", 8, 128, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
        "cat_smooth":        trial.suggest_float("cat_smooth", 1, 100, log=True),
        "cat_l2":            trial.suggest_float("cat_l2", 1, 100, log=True),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 20, 500, log=True),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--base-exp", default="E10")
    a = ap.parse_args()
    seed_everything(SEED)
    flags = json.load(open(f"outputs/params/{a.base_exp}.json"))["params"]["flags"]
    train, test = pd.read_csv("input/train.csv"), pd.read_csv("input/test.csv")
    X, Xt, names, cats = make_features(train, test, flags)
    y = train["TARGET"]
    folds = np.load("outputs/folds.npy")
    out = Path("outputs/optuna"); out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        params = {**FIXED, **lgb_search_space(trial)}
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, fold_aucs, iters = run_cv_lgb(X, y, Xt.iloc[:1], params, folds, seed=SEED, cat_features=cats)
        trial.set_user_attr("fold_aucs", [round(v, 6) for v in fold_aucs])
        trial.set_user_attr("best_iters", iters)
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        print(f"trial {trial.number:3d} | mean {np.mean(fold_aucs):.5f} | iters {iters} | {time.time()-t0:.0f}s", flush=True)
        return float(np.mean(fold_aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=a.trials, n_jobs=1)

    study.trials_dataframe().to_csv(out / "lgb_study.csv", index=False)
    best = {**FIXED, **study.best_params}
    json.dump({"best_value": study.best_value, "best_trial": study.best_trial.number, "params": best,
               "fold_aucs": study.best_trial.user_attrs["fold_aucs"]}, open(out / "lgb_best_params.json", "w"), indent=2)
    imp = optuna.importance.get_param_importances(study)
    pd.Series(imp, name="importance").to_csv(out / "lgb_param_importance.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals = [t.value for t in study.trials]
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.scatter(range(len(vals)), vals, s=18, color="#86b6ef", label="trial")
    ax.plot(range(len(vals)), np.maximum.accumulate(vals), color="#2a78d6", lw=2, label="best so far")
    ax.axhline(0.75790, color="#898781", lw=1, ls="--"); ax.text(0, 0.75790, " E10 (BASE_PARAMS) 0.75790", va="bottom", fontsize=8, color="#52514e")
    ax.set_xlabel("trial"); ax.set_ylabel("mean 5-fold CV AUC")
    lo = np.percentile(vals, 10); ax.set_ylim(min(lo, 0.7570), max(vals) + 0.0008)
    ax.grid(axis="y", color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("T1 Optuna history (LightGBM, lr 0.05, 60 trials)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig("reports/figures/T1_optuna_history.png", dpi=120)
    print("BEST", study.best_value, best)
    print("IMPORTANCE", {k: round(v, 3) for k, v in imp.items()})


if __name__ == "__main__":
    main()
