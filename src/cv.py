"""Frozen validation framework (docs/validation-setup.md §1-§3).

The same code is inlined in notebooks/main.ipynb (section 5.2).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_JOBS = 4  # fixed thread count for every model (reproducibility)


def make_folds(y, n_splits=5, seed=SEED, path="outputs/folds.npy"):
    """Fold id (0..n_splits-1) for each training row; saved to `path` (None = don't save)."""
    folds = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[va] = k
    assert (folds >= 0).all()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, folds)
    return folds


def load_or_make_folds(y, path="outputs/folds.npy", seed=SEED):
    """Load saved folds; create them once if missing. Verifies they match a fresh regeneration."""
    path = Path(path)
    fresh = make_folds(y, seed=seed, path=None)
    if path.exists():
        folds = np.load(path)
        assert len(folds) == len(y), "folds.npy does not match the training data length"
        assert (folds == fresh).all(), "saved folds differ from make_folds(seed) - investigate"
    else:
        folds = make_folds(y, seed=seed, path=path)
    return folds


def run_cv_lgb(X, y, X_test, params, folds, seed=SEED, cat_features="auto", importance_out=None):
    """Returns oof (np.ndarray), test_pred (np.ndarray), fold_aucs (list), best_iters (list).

    Sprint 2 (approved Q3): validation data passed as eval_X/eval_y (LightGBM >= 4.6 API) instead of the
    deprecated eval_set; fold logic, early stopping and outputs are unchanged.
    importance_out: optional list; if given, one pd.Series of gain importance per fold is appended.
    """
    import lightgbm as lgb

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs, best_iters = [], []
    n_splits = folds.max() + 1
    for k in range(n_splits):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        model = lgb.LGBMClassifier(
            **params,
            n_estimators=10000,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_X=(X.iloc[va],),
            eval_y=(y.iloc[va],),
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / n_splits
        fold_aucs.append(roc_auc_score(y.iloc[va], oof[va]))
        best_iters.append(model.best_iteration_)
        if importance_out is not None:
            importance_out.append(pd.Series(
                model.booster_.feature_importance(importance_type="gain"),
                index=model.booster_.feature_name()))
    print(f"Fold AUCs: {np.round(fold_aucs, 5)}")
    print(f"Mean: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f} | OOF: {roc_auc_score(y, oof):.5f}")
    return oof, test_pred, fold_aucs, best_iters


def run_cv_sklearn(model, X, y, X_test, folds):
    """Same fold handling and outputs as run_cv_lgb, for any sklearn classifier (no early stopping).

    `model` is an unfitted estimator; it is cloned for every fold (fit uses its n_jobs, predict uses
    n_jobs=1 for bit-exact reproducibility). best_iters is a list of None.
    """
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs, best_iters = [], []
    n_splits = folds.max() + 1
    for k in range(n_splits):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        # Predict single-threaded: sklearn forests sum per-tree probabilities across threads in a
        # non-deterministic order, which changes the last float bit (and the CSV hash) between runs.
        if "n_jobs" in m.get_params():
            m.set_params(n_jobs=1)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test_pred += m.predict_proba(X_test)[:, 1] / n_splits
        fold_aucs.append(roc_auc_score(y.iloc[va], oof[va]))
        best_iters.append(None)
    print(f"Fold AUCs: {np.round(fold_aucs, 5)}")
    print(f"Mean: {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f} | OOF: {roc_auc_score(y, oof):.5f}")
    return oof, test_pred, fold_aucs, best_iters


def compare(new_folds, best_folds, min_wins=4, tiny=0.0005):
    new_folds, best_folds = np.asarray(new_folds), np.asarray(best_folds)
    delta = new_folds - best_folds
    wins = int((delta > 0).sum())
    mean_delta = delta.mean()
    if mean_delta > 0 and wins >= min_wins:
        verdict = "KEEP" if mean_delta >= tiny else "CONFIRM (tiny gain → re-run with seeds 1, 2, 3)"
    else:
        verdict = "DISCARD"
    print(f"Δmean={mean_delta:+.5f} | wins={wins}/{len(delta)} | per-fold Δ={np.round(delta, 5)} → {verdict}")
    return verdict


def compare_sprint2(new_folds, best_folds, removal=False, min_wins=4, tiny=0.0005):
    """Sprint-2 override of validation-setup §3 (approved by the user for time reasons):
    - CONFIRM (0 < Δmean < tiny with >= 4/5 wins) is replaced by DISCARD.
    - removal=True (experiment only REMOVES features): KEEP if Δmean >= -0.0002 and at most 2 folds
      got worse by more than 0.0005 (simpler model, same score).
    """
    new_folds, best_folds = np.asarray(new_folds), np.asarray(best_folds)
    delta = new_folds - best_folds
    wins = int((delta > 0).sum())
    mean_delta = float(delta.mean())
    if mean_delta >= tiny and wins >= min_wins:
        verdict = "KEEP"
    elif removal and mean_delta >= -0.0002 and int((delta < -0.0005).sum()) <= 2:
        verdict = "KEEP"
    else:
        verdict = "DISCARD"
    print(f"Δmean={mean_delta:+.5f} | wins={wins}/{len(delta)} | per-fold Δ={np.round(delta, 5)} → {verdict}")
    return verdict, mean_delta, wins
