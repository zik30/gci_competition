"""Builds notebooks/main.ipynb from docs/tutorial.ipynb, src/cv.py, src/utils.py, src/evidence.py,
reports/market_analysis.md and REFERENCES.md. Run from the project root:  .venv/bin/python src/build_notebook.py"""
import json
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path.cwd()
SCR = Path(__file__).resolve().parent  # src/ (evidence.py lives here)
tut = json.load(open(ROOT / "docs/tutorial.ipynb"))
T = lambda i: "".join(tut["cells"][i]["source"])  # noqa: E731

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

# ---------------------------------------------------------------- 0
md("""
# 0. Home Credit Default Risk — GCI Global 2026 Competition

**Objective:** predict the probability that a loan applicant will have payment difficulties (`TARGET = 1`).
**Metric:** ROC AUC. **Submission:** `SK_ID_CURR,TARGET`.

| Item | Value |
| ---- | ----- |
| Final model | _placeholder — to be filled in the final sprint (after tuning)_ |
| Current best experiment | **E10** — LightGBM (`BASE_PARAMS`), 32 raw features + 5 credit ratios − `FLAG_MOBIL` (36 features) |
| 5-fold CV AUC of E10 (mean ± std / OOF) | **0.75790 ± 0.00313 / 0.75784** (tutorial baseline E0: 0.67020 ± 0.00190) |
| Public LB AUC | _placeholder_ |

**Reproducibility:** one global `SEED = 42`, fixed `n_jobs`, frozen 5-fold split saved in `outputs/folds.npy`.
*Restart & Run All* regenerates the submission CSV bit-for-bit (verified by MD5).
This notebook is self-contained: it does not import from `src/`.

**Structure:** Market analysis (§3.0) → EDA (§3) → Machine learning (§4–§6) → Model evaluation & analysis (§7) → Business proposal (§8) → References (§9) → Experiment log (Appendix A).

> Rule compliance: only `input/train.csv` and `input/test.csv` are used (no external data);
> every test prediction comes from `predict_proba` of trained models (no hand-labeling);
> all borrowed code is cited with `# Source:` comments and in §9.
""")

# ---------------------------------------------------------------- 1
md("## 1.Setup")
md("""
### 1.1 Import Libraries
Basic libraries, library versions and the global seed (rule R3).
""")
code('''
# Source: GCI tutorial.ipynb §1.1 — adapted: added version printout and global SEED (rule R3)
import os, random, sys, platform, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
import sklearn, lightgbm as lgb
warnings.filterwarnings("ignore")

SEED = 42
N_JOBS = 4  # fixed thread count for every model (multi-threading must not change results)
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
seed_everything()

print("Python", sys.version.split()[0], "|", platform.platform())
for m in [np, pd, sklearn, lgb, matplotlib, sns]:
    print(f"{m.__name__:>12}: {m.__version__}")
''')
md("""
### 1.2 Connect with Google Drive
Works on Google Colab (mounts Drive) and locally (project root or `notebooks/`).
On Colab, change the `os.chdir` path to the folder that contains `input/`.
""")
code('''
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    os.chdir("/content/drive/MyDrive/gci_global/Competition")  # TODO(user): confirm path
current_dir = Path(os.getcwd())
if current_dir.name == "notebooks":          # running locally from notebooks/
    current_dir = current_dir.parent
os.chdir(current_dir)                        # all relative paths below are relative to the project root
train_file      = current_dir / "input" / "train.csv"
test_file       = current_dir / "input" / "test.csv"
sample_sub_file = current_dir / "input" / "sample_submission.csv"
OUT = current_dir / "outputs"
for d in ["oof", "test_preds", "params", "submissions"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)
(current_dir / "experiments").mkdir(exist_ok=True)
''')
md(T(10))
code(T(11))  # tutorial cell verbatim (harmless: cwd is already the project root)

# ---------------------------------------------------------------- 2
md("## 2.Loading the Data")
md(T(13))
code('''
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
sample_sub = pd.read_csv(sample_sub_file)

# Full copies (all 32 features) for EDA / market-analysis checks; `train`/`test` are reduced in §2.2
train_full = train.copy()
test_full = test.copy()
''')
md(T(15))
code(T(16))
code(T(17))
md("""
**Data check (Sprint 1, Task 0.2):** test has the same columns as train minus `TARGET`; IDs are unique and
disjoint; `sample_submission` IDs equal the test IDs in the same order.
""")
code('''
assert list(test.columns) == [c for c in train.columns if c != "TARGET"]
assert train.SK_ID_CURR.is_unique and test.SK_ID_CURR.is_unique
assert len(set(train.SK_ID_CURR) & set(test.SK_ID_CURR)) == 0
assert (sample_sub.SK_ID_CURR.values == test.SK_ID_CURR.values).all()
print(f"All data checks passed. Target rate: {train.TARGET.mean():.5f} ({train.TARGET.sum()} positives of {len(train)})")
''')
md(T(19))
code(T(20))
md(T(21))
code(T(22))
code(T(23))

# ---------------------------------------------------------------- 3
md("## 3.Visualizing and Understanding the Data")
md(T(26))
md("### 3.0 Market Analysis\n\n*Full text also in `reports/market_analysis.md`.*")
ma = (ROOT / "reports/market_analysis.md").read_text()
ma = re.sub(r"^# .*\n", "", ma, count=1)
for part in re.split(r"\n(?=## )", ma):
    part = part.strip()
    if part:
        part = re.sub(r"^## ", "#### 3.0.", part, count=1, flags=re.M)
        md(part)
md("#### 3.0.9 Initial evidence for the hypotheses (Task 1.2 — preview only, full EDA in Sprint 2)")
ev_src = (SCR / "evidence.py").read_text().split('if __name__ == "__main__":')[0]
ev_src = ev_src.replace("import numpy as np, pandas as pd\nfrom sklearn.metrics import roc_auc_score\n",
                        "from sklearn.metrics import roc_auc_score\n")
code(ev_src.strip() + '''

evidence = hypothesis_evidence(train_full)
pd.set_option("display.max_colwidth", None)
pd.DataFrame({"hypothesis": list(evidence), "initial evidence (train)": list(evidence.values())})
''')
md(T(27))
code(T(28))
code(T(29))
md(T(30))
md(T(31))
EDA_MD = (SCR / "eda_text.md").read_text().split("\n<!-- cell -->\n")
eda_src = (SCR / "eda.py").read_text()
eda_src = eda_src.split('FIG_DIR = Path("reports/figures")', 1)[1]
code("# EDA helpers (same code as src/eda.py). Works on copies; never modifies train_full / test_full.\n"
     "from sklearn.metrics import roc_auc_score\nfrom sklearn.model_selection import StratifiedKFold\n"
     'FIG_DIR = Path("reports/figures")' + eda_src.rstrip())
md(EDA_MD[0])
code("a1_rate = a1_target_balance(train_full); plt.show()\nprint(f'default rate: {a1_rate:.4%}')")
md(EDA_MD[1])
code("a2_tab = a2_missing(train_full); plt.show()\na2_tab")
md(EDA_MD[2])
code("a3_tab, a3_overlap = a3_days_employed(train_full); plt.show()\ndisplay(a3_tab); a3_overlap")
md(EDA_MD[3])
code("a4_tab = a4_deciles(train_full); plt.show()\na4_tab")
md(EDA_MD[4])
code("a5 = a5_categoricals(train_full); plt.show()\npd.DataFrame({c: s.head(8).to_dict() for c, s in a5.items()}).T")
md(EDA_MD[5])
code("a6_pairs = a6_correlation(train_full); plt.show()\na6_pairs")
md(EDA_MD[6])
code("Xr_tr, Xr_te = raw_matrices(train_full, test_full)\n"
     "a7_auc, a7_top = a7_drift(Xr_tr, Xr_te, n_jobs=N_JOBS, seed=SEED); plt.show()\n"
     "print(f'Adversarial validation AUC (train=0 / test=1): {a7_auc:.4f}')\na7_top")
md(EDA_MD[7])
code("a8 = a8_fairness(train_full)\nfor k, v in a8.items():\n    print(k); display(v)")
md(EDA_MD[8])
code("assert train_full.shape == (171202, 34) and test_full.shape == (61500, 33)  # EDA did not modify the data")

# ---------------------------------------------------------------- 4
md(T(65))
md("""
### 4.1 Baseline preprocessing (E0 — tutorial)
Baseline E0 uses **exactly the tutorial's preprocessing** of its 5 features. The only code change is
technical: with pandas ≥ 3 (Copy-on-Write), `df[col].replace(..., inplace=True)` / `.fillna(..., inplace=True)`
silently do nothing, so they are rewritten as assignments with identical meaning. The encoders are fitted
on the full train set exactly like the tutorial (no `TARGET` is used, so no leakage — approved by the user
for E0).
""")
md(T(66))
code('''
# Source: GCI tutorial.ipynb §4 — adapted: inplace replace → assignment (pandas>=3 Copy-on-Write)
# Numerization of NAME_CONTRACT_TYPE（Label Encoding）
train["NAME_CONTRACT_TYPE"] = train["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}).astype(int)
test["NAME_CONTRACT_TYPE"] = test["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}).astype(int)

train.head(5)
''')
md(T(68))
code("# Source: GCI tutorial.ipynb §4 — unchanged\n" + T(69) + '''

# Sanity check: every test category was seen in train (otherwise .map would create NaN)
assert test["ORGANIZATION_TYPE"].notna().all()''')
md(T(70))
code('''
# Source: GCI tutorial.ipynb §4 — adapted: inplace fillna → assignment (pandas>=3 Copy-on-Write)
# Complete missing values of EXT_SOURCE_2 with the average
ext2_train_mean = train["EXT_SOURCE_2"].mean()
train["EXT_SOURCE_2"] = train["EXT_SOURCE_2"].fillna(ext2_train_mean)
test["EXT_SOURCE_2"] = test["EXT_SOURCE_2"].fillna(ext2_train_mean) # Use average of train data to fill test data
assert train["EXT_SOURCE_2"].notna().all() and test["EXT_SOURCE_2"].notna().all()

train.isnull().sum()
''')
md(T(72))
code("# Source: GCI tutorial.ipynb §4 — unchanged\n" + T(73))
md(T(74))
code("# Source: GCI tutorial.ipynb §4 — unchanged\n" + T(75))
code("# Source: GCI tutorial.ipynb §4 — adapted: test dummies re-indexed to train's columns (safety)\n" + T(76).replace(
    "# Remove original OWN_CAR_AGE",
    "# Make sure train and test have identical dummy columns in the same order\n"
    "ohe_cols = list(train_car_age_ohe.columns)\n"
    "assert set(test_car_age_ohe.columns) <= set(ohe_cols), \"test has a car-age group unseen in train\"\n"
    "test = test.drop(columns=list(test_car_age_ohe.columns)).join(test_car_age_ohe.reindex(columns=ohe_cols, fill_value=False))\n\n"
    "# Remove original OWN_CAR_AGE"))
md("""
### 4.2 Feature pipeline for the LightGBM experiments (`make_features`, Sprint 2)
One function with on/off flags, used by every experiment E1–E13 (same code as `src/features.py`).
It works on the full raw data (`train_full` / `test_full`), never uses `TARGET`, and returns sorted feature
names. `FEATURE_FLAGS` below are the flags of the **current best experiment (E10)**; see Appendix A for the
experiments that were tried and discarded.
""")
fe_src = (ROOT / "src/features.py").read_text()
fe_src = fe_src.split("import pandas as pd\n", 1)[1]
code("# Source: own code (Sprint 2). Ratio / missing-value / EXT_SOURCE ideas are standard domain features, see §9.\n"
     + fe_src.strip())
code('''
# Flags of the current best experiment E10 (literal copy of outputs/params/E10.json)
FEATURE_FLAGS = {
    **FEATURE_FLAGS_DEFAULT,
    "raw_all": True,                       # E2  KEEP
    "credit_ratios": True,                 # E4  KEEP
    "drop_low_importance": ["FLAG_MOBIL"], # E10 KEEP (zero gain / constant)
}
BEST_EXP = "E10"
X_lgb, X_test_lgb, LGB_FEATURES, LGB_CATS = make_features(train_full, test_full, FEATURE_FLAGS)
print(f"{len(LGB_FEATURES)} features, {len(LGB_CATS)} categorical: {LGB_CATS}")
X_lgb.head(3)
''')

# ---------------------------------------------------------------- 5
md(T(78))
md(T(79) + "\n- `StratifiedKFold`: the frozen 5-fold split used for every experiment (see `docs/validation-setup.md`).")
code('''
# Importing libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import csv, json, datetime as dt
''')
md("""
### 5.2 Preparing the Data
Split the data into explanatory and target variables, standardize as in the tutorial, then set up the
frozen validation framework (folds, CV functions, keep/discard rule, experiment log, artifact saving).
""")
code('''
# Source: GCI tutorial.ipynb §5.2 — adapted: keep DataFrames (column names) instead of .values
# Split the data into explanatory and target variables
X = train.drop("TARGET", axis=1)
y = train["TARGET"]
X_test = test[X.columns]
assert list(X_test.columns) == list(X.columns)
FEATURES = list(X.columns)
print(f"{len(FEATURES)} model features: {FEATURES}")

# Standardization (fitted on the full train set, exactly as in the tutorial)
sc = StandardScaler()
sc.fit(X)
X_std = pd.DataFrame(sc.transform(X), columns=FEATURES)
X_test_std = pd.DataFrame(sc.transform(X_test), columns=FEATURES)
''')
md("#### Validation framework (frozen — `docs/validation-setup.md`; same code as `src/cv.py`, `src/utils.py`)")
cv_src = (ROOT / "src/cv.py").read_text()
cv_src = cv_src.split("N_JOBS = 4")[1].split("\n", 1)[1]  # drop module header / imports / constants
cv_src = cv_src.replace("    import lightgbm as lgb\n\n", "")
code("# Source: docs/validation-setup.md §1-§3 (project validation spec) — adapted: load_or_make_folds and\n"
     "# run_cv_sklearn added (same fold handling/outputs as run_cv_lgb)\n" + cv_src.strip())
ut_src = (ROOT / "src/utils.py").read_text()
ut_src = "LOG_COLS" + ut_src.split("LOG_COLS", 1)[1]
ut_src = ut_src.split("def seed_everything")[0] + "def log_experiment" + ut_src.split("def log_experiment", 1)[1]
code("# Source: docs/validation-setup.md §4, §6 — adapted: log is append-only AND idempotent (no duplicate exp_id)\n" + ut_src.strip())
code('''
folds = load_or_make_folds(y.values, path="outputs/folds.npy")
for k in range(5):
    print(f"fold {k}: n={(folds == k).sum()}, target rate={y[folds == k].mean():.5f}")
''')
md("""### 5.3 Training the Model
**Experiment E0 — baseline (harness verification).** The tutorial's final model (Random Forest,
`max_depth=10`) on the tutorial's 5 preprocessed features, evaluated with the frozen 5-fold CV instead of a
single 70/30 holdout. Only change vs the tutorial: `random_state = SEED (42)` instead of `0` (rule R3) and a
fixed `n_jobs`. The test prediction is the average of the 5 fold models' `predict_proba`.
""")
code('''
# Source: GCI tutorial.ipynb §5.3 (RandomForestClassifier(max_depth=10)) — adapted: random_state=SEED, n_jobs fixed,
# evaluated with the frozen 5-fold CV
from sklearn.ensemble import RandomForestClassifier

E0_PARAMS = dict(max_depth=10, random_state=SEED, n_jobs=N_JOBS)
rf = RandomForestClassifier(**E0_PARAMS)
e0_oof, e0_test_pred, e0_fold_aucs, e0_best_iters = run_cv_sklearn(rf, X_std, y, X_test_std, folds)
''')
code('''
log_experiment({
    "exp_id": "E0", "date": "2026-11-18", "parent_id": "-",
    "hypothesis": "Baseline: tutorial 5-feature model",
    "change": "Tutorial RF(max_depth=10) + tutorial preprocessing, evaluated in frozen 5-fold CV (random_state=42)",
    "model": "rf", "n_features": len(FEATURES),
    **{f"fold_{k+1}": round(a, 5) for k, a in enumerate(e0_fold_aucs)},
    "mean_auc": round(float(np.mean(e0_fold_aucs)), 5), "std_auc": round(float(np.std(e0_fold_aucs)), 5),
    "oof_auc": round(roc_auc_score(y, e0_oof), 5),
    "delta_vs_parent": "-", "folds_improved": "-", "verdict": "BASELINE", "public_lb": "",
    "notes": "Harness verification; RF has no early stopping so best_iters are None",
}, log_path="experiments/experiment_log.csv")
save_artifacts("E0", e0_oof, e0_test_pred, E0_PARAMS, FEATURES, e0_best_iters, out_dir="outputs")
''')
md("""
*Debug only (not logged):* the tutorial's 70/30 holdout (with `SEED`), to sanity-check the
preprocessing was ported faithfully. A single split is never used to choose between experiments.
""")
code('''
# Source: GCI tutorial.ipynb §5.3 — adapted: random_state=SEED (rule R3); debug check only
X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.3, stratify=y, random_state=SEED)
rf_holdout = RandomForestClassifier(random_state=SEED, max_depth=10, n_jobs=N_JOBS).fit(X_train, y_train)
print(f"Tutorial-style holdout  Train: {roc_auc_score(y_train, rf_holdout.predict_proba(X_train)[:, 1]):.5f} "
      f"| Valid: {roc_auc_score(y_valid, rf_holdout.predict_proba(X_valid)[:, 1]):.5f}")
''')
md("""
#### Best model — E10: LightGBM on the Sprint-2 feature set
LightGBM with the fixed Sprint-2 parameters (`BASE_PARAMS`, no tuning yet), early stopping on each validation
fold (200 rounds), same frozen folds. The test prediction is the average of the 5 fold models'
`predict_proba`. The run is checked against the stored E10 artifacts from the experiment runner.
""")
code('''
# Fixed LightGBM params for Sprint 2 (docs/sprints/sprint-2.md §2) — tuning happens in Sprint 3
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
best_imps = []
best_oof, best_test_pred, best_fold_aucs, best_iters = run_cv_lgb(
    X_lgb, y, X_test_lgb, BASE_PARAMS, folds, seed=SEED, cat_features=LGB_CATS, importance_out=best_imps)
print("best iterations:", best_iters)
''')
code('''
log_experiment({
    "exp_id": BEST_EXP, "date": "2026-11-19", "parent_id": "E4",
    "hypothesis": "Fewer noisy features (zero gain / near-constant), same or better AUC",
    "change": "drop_low_importance", "model": "lgb", "n_features": len(LGB_FEATURES),
    **{f"fold_{k+1}": round(a, 5) for k, a in enumerate(best_fold_aucs)},
    "mean_auc": round(float(np.mean(best_fold_aucs)), 5), "std_auc": round(float(np.std(best_fold_aucs)), 5),
    "oof_auc": round(roc_auc_score(y, best_oof), 5), "verdict": "KEEP",
}, log_path="experiments/experiment_log.csv")   # idempotent: skipped if E10 is already logged
best_importance = pd.concat(best_imps, axis=1)
best_importance = pd.DataFrame({"gain_mean": best_importance.mean(axis=1), "gain_std": best_importance.std(axis=1)}
                               ).sort_values("gain_mean", ascending=False)
stored = OUT / "test_preds" / f"{BEST_EXP}_test.npy"
if stored.exists():   # consistency with the experiment runner (src/run_experiment.py)
    assert np.array_equal(np.load(stored), best_test_pred), "notebook result differs from the logged E10 run"
    assert np.array_equal(np.load(OUT / "oof" / f"{BEST_EXP}_oof.npy"), best_oof)
    print(f"{BEST_EXP}: notebook predictions are bit-identical to the stored experiment artifacts")
else:
    save_artifacts(BEST_EXP, best_oof, best_test_pred, {"model": "lgb", "lgb_params": BASE_PARAMS,
                   "flags": FEATURE_FLAGS}, LGB_FEATURES, best_iters, out_dir="outputs")
''')
md("""
### 5.4 (Optional) Ensemble Learning
*Placeholder — planned for Sprint 3 (seed averaging, XGBoost, OOF-based rank-average blend).*
""")

# ---------------------------------------------------------------- 6
md(T(95))
md("""
### 6.1 Predicting on the test data
The prediction is the average of the 5 fold models' `predict_proba` on the test data from the current best
experiment (**E10**, LightGBM). No prediction is modified after `predict_proba`. The E0 baseline prediction
(`e0_test_pred`) is kept only for comparison.
""")
code('''
# Make predictions for the test data
# Average of the 5 fold models of the current best experiment (E10)
pred = best_test_pred
''')
md(T(98))
code(T(99))
code(T(100))
md(T(101))
md(T(102))
s103 = T(103)
assert s103.count('"tutorial.ipynb"') == 1 and s103.count('"tutorial.zip"') == 1
code(s103.replace('notebook_filename = "tutorial.ipynb"', 'notebook_filename = "main.ipynb"')
         .replace('zip_filename = "tutorial.zip"', 'zip_filename = "main.zip"'))
md("""
### 6.4 Archive the submission and run sanity checks (added; does not change 6.2)
Copies `output/submission.csv` (written by §6.2) to `outputs/submissions/<BEST_EXP>_submission.csv` and runs the
submission sanity checks from `docs/validation-setup.md` §6.
""")
code('''
import shutil, hashlib
CURRENT_EXP = BEST_EXP
archived = OUT / "submissions" / f"{CURRENT_EXP}_submission.csv"
shutil.copyfile(output_file, archived)
check_submission(archived, sample_sub_path=sample_sub_file)
print("MD5:", hashlib.md5(archived.read_bytes()).hexdigest())
''')

# ---------------------------------------------------------------- 7-9
md("""
## 7. Model Evaluation & Analysis
*Full evaluation (SHAP via LightGBM `pred_contrib=True`, error analysis, approval-rate curves) in Sprint 4.*

#### 7.1 Preview: top 20 features of E10 by mean gain over the 5 folds
""")
code('''
fig, ax = plt.subplots(figsize=(7, 6))
top20 = best_importance.head(20)[::-1]
ax.barh(top20.index, top20["gain_mean"], xerr=top20["gain_std"], color="#4C72B0")
ax.set_xlabel("mean gain (± std over folds)"); ax.set_title(f"{BEST_EXP}: top 20 features by gain")
fig.savefig("reports/figures/E10_top20_gain.png", dpi=120, bbox_inches="tight"); plt.show()
best_importance.head(20).round(1)
''')
md("""
## 8. Business Proposal
*Placeholder — approval-rate vs default-rate table at several cut-offs (final model vs E0 baseline),
automation bands, risk-based terms, fairness discussion (later sprints).*
""")
refs = (ROOT / "REFERENCES.md").read_text()
refs = re.sub(r"^# References\n", "", refs)
md("## 9. References\n" + refs)
md("## Appendix A. Experiment log\nRendered from `experiments/experiment_log.csv`.")
code('''
pd.read_csv("experiments/experiment_log.csv")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
nbf.validate(nb)
nbf.write(nb, ROOT / "notebooks/main.ipynb")
print(f"wrote notebooks/main.ipynb with {len(cells)} cells")
