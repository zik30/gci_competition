"""Task 0.4 tests for the validation framework. Run from the project root:

    .venv/bin/python src/test_framework.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import SEED, compare, load_or_make_folds, make_folds, run_cv_lgb, run_cv_sklearn  # noqa: E402

train = pd.read_csv("input/train.csv", usecols=["TARGET", "EXT_SOURCE_2", "AMT_INCOME_TOTAL", "AMT_CREDIT"])
y = train["TARGET"]

print("== Test 1: folds.npy length, values, per-fold target rate")
folds = load_or_make_folds(y.values, path="outputs/folds.npy")
saved = np.load("outputs/folds.npy")
assert len(saved) == len(train), (len(saved), len(train))
assert set(np.unique(saved)) == {0, 1, 2, 3, 4}
overall = y.mean()
for k in range(5):
    rate = y[saved == k].mean()
    diff_pp = (rate - overall) * 100
    print(f"fold {k}: n={int((saved == k).sum())}, target rate={rate*100:.3f}% (Δ={diff_pp:+.3f} pp)")
    assert abs(diff_pp) <= 0.2
print(f"overall target rate={overall*100:.3f}%  -> PASS")

print("\n== Test 2: make_folds() twice gives identical arrays")
a = make_folds(y.values, path=None)
b = make_folds(y.values, path=None)
assert (a == b).all() and (a == saved).all()
print("identical: True -> PASS")

print("\n== Test 3: run_cv_lgb / run_cv_sklearn on a tiny random subset")
rng = np.random.RandomState(SEED)
idx = rng.choice(len(train), 3000, replace=False)
Xs = train.drop(columns="TARGET").iloc[idx].reset_index(drop=True)
ys = y.iloc[idx].reset_index(drop=True)
Xt = Xs.iloc[:500]
fs = make_folds(ys.values, path=None)
oof, tp, aucs, iters = run_cv_lgb(Xs, ys, Xt, {"learning_rate": 0.1, "num_leaves": 8}, fs)
assert oof.shape == (len(Xs),) and tp.shape == (len(Xt),) and len(aucs) == 5 and len(iters) == 5
assert np.isfinite(oof).all() and ((tp >= 0) & (tp <= 1)).all()
print(f"lgb shapes: oof={oof.shape}, test={tp.shape}, folds={len(aucs)}, best_iters={iters} -> PASS")
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
oof2, tp2, aucs2, _ = run_cv_sklearn(
    RandomForestClassifier(n_estimators=20, max_depth=5, random_state=SEED, n_jobs=4),
    Xs.fillna(-1), ys, Xt.fillna(-1), fs)
assert oof2.shape == (len(Xs),) and tp2.shape == (len(Xt),) and len(aucs2) == 5
print(f"sklearn shapes: oof={oof2.shape}, test={tp2.shape}, folds={len(aucs2)} -> PASS")

print("\n== Test 4: compare() decision rule")
base = [0.70, 0.70, 0.70, 0.70, 0.70]
assert compare([0.71] * 5, base) == "KEEP"
assert compare([0.7001] * 5, base).startswith("CONFIRM")
assert compare([0.75, 0.75, 0.69, 0.69, 0.69], base) == "DISCARD"
assert compare([0.69] * 5, base) == "DISCARD"
print("-> PASS")
print("\nALL TESTS PASSED")
