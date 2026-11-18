"""Development helpers: seeding, experiment log, artifact saving, submission checks.

The same code is inlined in notebooks/main.ipynb (section 5.2); the final
notebook does not import from src/ (rule R3).
"""
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

LOG_COLS = ["exp_id", "date", "parent_id", "hypothesis", "change", "model", "n_features",
            "fold_1", "fold_2", "fold_3", "fold_4", "fold_5", "mean_auc", "std_auc", "oof_auc",
            "delta_vs_parent", "folds_improved", "verdict", "public_lb", "notes"]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_experiment(row: dict, log_path="experiments/experiment_log.csv"):
    """Append one row to the experiment log (schema: validation_setup.md §4).

    Append-only and idempotent: if `exp_id` is already in the log, the row is NOT
    written again (re-running the notebook must not duplicate or overwrite history).
    Returns True if a row was written.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    if not new:
        with open(log_path, newline="") as f:
            if any(r["exp_id"] == row["exp_id"] for r in csv.DictReader(f)):
                print(f"[log] {row['exp_id']} already logged - not appended again")
                return False
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in LOG_COLS})
    print(f"[log] {row['exp_id']} appended to {log_path}")
    return True


def save_artifacts(exp_id, oof, test_pred, params, features, best_iters, out_dir="outputs"):
    """Write OOF + test predictions (.npy) and params/features/best iterations (.json)."""
    out = Path(out_dir)
    for d in ["oof", "test_preds", "params"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    np.save(out / "oof" / f"{exp_id}_oof.npy", np.asarray(oof))
    np.save(out / "test_preds" / f"{exp_id}_test.npy", np.asarray(test_pred))
    meta = {
        "exp_id": exp_id,
        "params": params,
        "features": list(features),
        "best_iters": [None if b is None else int(b) for b in best_iters],
    }
    with open(out / "params" / f"{exp_id}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def check_submission(path, sample_sub_path="input/sample_submission.csv"):
    """Submission sanity checks (validation_setup.md §6). Raises AssertionError on failure."""
    sub = pd.read_csv(path)
    ss = pd.read_csv(sample_sub_path)
    assert list(sub.columns) == ["SK_ID_CURR", "TARGET"]
    assert len(sub) == len(ss) and (sub.SK_ID_CURR.values == ss.SK_ID_CURR.values).all()
    assert sub.TARGET.between(0, 1).all() and sub.TARGET.notna().all()
    print(f"[check] {path}: OK ({len(sub)} rows, TARGET in [{sub.TARGET.min():.4f}, {sub.TARGET.max():.4f}])")
    return True
