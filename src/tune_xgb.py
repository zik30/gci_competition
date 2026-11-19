"""Sprint 3 Task X1: brief Optuna tuning of XGBoost on the E10 feature set. Run from the project root:

    .venv/bin/python src/tune_xgb.py [--trials 30]

Objective = mean fold AUC over outputs/folds.npy via run_cv_xgb; learning_rate fixed at 0.05 during the
search (refit at the final LightGBM learning rate afterwards). TPESampler(seed=42), sequential trials.
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
from cv import SEED, run_cv_xgb  # noqa: E402
from features import make_features  # noqa: E402
from utils import seed_everything  # noqa: E402

FIXED = dict(learning_rate=0.05)


def xgb_search_space(trial):
    # Search space from docs/sprints/sprint-3.md §4
    return {
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_child_weight":  trial.suggest_float("min_child_weight", 1, 100, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
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
        params = {**FIXED, **xgb_search_space(trial)}
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, fold_aucs, iters = run_cv_xgb(X, y, Xt.iloc[:1], params, folds, seed=SEED)
        trial.set_user_attr("fold_aucs", [round(v, 6) for v in fold_aucs])
        trial.set_user_attr("best_iters", iters)
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        print(f"trial {trial.number:3d} | mean {np.mean(fold_aucs):.5f} | iters {iters} | {time.time()-t0:.0f}s", flush=True)
        return float(np.mean(fold_aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=a.trials, n_jobs=1)
    study.trials_dataframe().to_csv(out / "xgb_study.csv", index=False)
    best = {**FIXED, **study.best_params}
    json.dump({"best_value": study.best_value, "best_trial": study.best_trial.number, "params": best,
               "fold_aucs": study.best_trial.user_attrs["fold_aucs"]}, open(out / "xgb_best_params.json", "w"), indent=2)
    imp = optuna.importance.get_param_importances(study)
    pd.Series(imp, name="importance").to_csv(out / "xgb_param_importance.csv")
    print("BEST", study.best_value, best)
    print("IMPORTANCE", {k: round(v, 3) for k, v in imp.items()})


if __name__ == "__main__":
    main()
