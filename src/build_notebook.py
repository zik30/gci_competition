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
| Final model | **E21** — rank-average blend: 0.7 × tuned LightGBM (5-seed average, E19) + 0.3 × tuned XGBoost (E20); 36 features (32 raw + 5 credit ratios − `FLAG_MOBIL`) |
| 5-fold CV AUC of E21 (mean ± std / OOF) | **0.76342 ± 0.00315 / 0.76339** (Sprint-2 best E10: 0.75790; tutorial baseline E0: 0.67020) |
| Public LB AUC | E10: 0.7638 (E21: not submitted yet) |

**Reproducibility:** one global `SEED = 42`, fixed `n_jobs`, frozen 5-fold split saved in `outputs/folds.npy`.
*Restart & Run All* regenerates the submission CSV bit-for-bit (verified by MD5).
This notebook is self-contained: it does not import from `src/`.

**Structure:** Market analysis (§3.0) → EDA (§3) → Machine learning (§4–§6) → Model evaluation & analysis (§7) → Business proposal (§8) → References (§9) → Experiment log (Appendix A) → Optuna search (Appendix B).

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
import sklearn, lightgbm as lgb, xgboost as xgb, scipy
warnings.filterwarnings("ignore")

SEED = 42
N_JOBS = 4  # fixed thread count for every model (multi-threading must not change results)
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
seed_everything()

print("Python", sys.version.split()[0], "|", platform.platform())
for m in [np, pd, sklearn, lgb, xgb, scipy, matplotlib, sns]:
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
names. `FEATURE_FLAGS` below are the flags of E10 (the best Sprint-2 feature set). The final Sprint-3 models use
the same feature set, because the Sprint-3 feature re-tests (E17, E18) were discarded. Appendix A lists the
experiments that were tried and discarded.
""")
fe_src = (ROOT / "src/features.py").read_text()
fe_src = fe_src.split("import pandas as pd\n", 1)[1]
code("# Source: own code (Sprint 2). Ratio / missing-value / EXT_SOURCE ideas are standard domain features, see §9.\n"
     + fe_src.strip())
code('''
# Flags of E10 = feature set of the final models E19/E20/E21 (literal copy of outputs/params/E10.json)
FEATURE_FLAGS = {
    **FEATURE_FLAGS_DEFAULT,
    "raw_all": True,                       # E2  KEEP
    "credit_ratios": True,                 # E4  KEEP
    "drop_low_importance": ["FLAG_MOBIL"], # E10 KEEP (zero gain / constant)
}
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
code("# Source: docs/validation-setup.md §1-§3 (project validation spec) — adapted: load_or_make_folds,\n"
     "# run_cv_sklearn and run_cv_xgb (Sprint 3) added (same fold handling/outputs as run_cv_lgb)\n" + cv_src.strip())
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
#### Hyperparameter search (Sprint 3, T1 / X1) — Optuna, switched off by default
The tuned parameters below were found with Optuna (`TPESampler(seed=42)`, sequential trials, no pruning).
The objective is the **mean 5-fold CV AUC on the frozen folds** (never a single split), with the learning rate
fixed at 0.05. LightGBM: 60 trials (T1); XGBoost: 30 trials (X1). The search takes about 2 hours on 4 CPU
threads, so it is behind `RUN_OPTUNA = False`. The results are copied into the notebook as literals in the
next cells. Setting `RUN_OPTUNA = True` re-runs the search and rewrites `outputs/optuna/` (same code as
`src/tune_lgb.py` and `src/tune_xgb.py`). History plots and parameter importance are in Appendix B.

**Tuning-overfit check (E14).** Tuning on the main folds could overfit those folds. E14 (tuned vs Sprint-2
params) was therefore re-run on 3 alternative fold sets (`StratifiedKFold` seeds 1, 2, 3). The gain on the main
folds is +0.00450, and on the alternative folds it is +0.00393 / +0.00376 / +0.00475 (about 92% of the main
gain), so the gain is real.
""")
code('''
# Source: Optuna API — TPESampler / create_study / get_param_importances (see §9); search spaces from the
# project plan docs/sprints/sprint-3.md §3-§4. Same code as src/tune_lgb.py, src/tune_xgb.py.
RUN_OPTUNA = False

def lgb_search_space(trial):
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

def xgb_search_space(trial):
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

def run_optuna(kind, n_trials):
    import optuna, contextlib, io
    fixed = (dict(objective="binary", learning_rate=0.05, bagging_freq=1) if kind == "lgb"
             else dict(learning_rate=0.05))
    space = lgb_search_space if kind == "lgb" else xgb_search_space
    def objective(trial):
        params = {**fixed, **space(trial)}
        with contextlib.redirect_stdout(io.StringIO()):
            if kind == "lgb":
                _, _, fa, _ = run_cv_lgb(X_lgb, y, X_test_lgb.iloc[:1], params, folds, seed=SEED, cat_features=LGB_CATS)
            else:
                _, _, fa, _ = run_cv_xgb(X_lgb, y, X_test_lgb.iloc[:1], params, folds, seed=SEED)
        trial.set_user_attr("fold_aucs", [round(v, 6) for v in fa])
        return float(np.mean(fa))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    (OUT / "optuna").mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(OUT / "optuna" / f"{kind}_study.csv", index=False)
    pd.Series(optuna.importance.get_param_importances(study), name="importance").to_csv(
        OUT / "optuna" / f"{kind}_param_importance.csv")
    json.dump({"best_value": study.best_value, "best_trial": study.best_trial.number,
               "params": {**fixed, **study.best_params}, "fold_aucs": study.best_trial.user_attrs["fold_aucs"]},
              open(OUT / "optuna" / f"{kind}_best_params.json", "w"), indent=2)
    return study

if RUN_OPTUNA:
    lgb_study = run_optuna("lgb", 60)
    xgb_study = run_optuna("xgb", 30)
    print("LightGBM best:", lgb_study.best_value, lgb_study.best_params)
    print("XGBoost  best:", xgb_study.best_value, xgb_study.best_params)
else:
    print("RUN_OPTUNA = False -> using the literal tuned parameters below")
''')
md("""
#### Final model, part 1 — E19: tuned LightGBM, average of 5 seeds
LightGBM with the tuned parameters (T1 / E14). The learning rate stays at 0.05: 0.02 (E15) and 0.01 (E16) did
not improve CV. The model is trained 5 times with seeds 42–46 on the same frozen folds, and the OOF and test
predictions are averaged (S1 / E19: +0.00069, 5/5 folds). Each run uses early stopping on its validation fold
(200 rounds), and its test prediction is the average of its 5 fold models' `predict_proba`. Features: the Sprint-2
set from §4.2 (re-adding EXT_SOURCE aggregates (E17) or missing/bureau features (E18) did not help).
""")
code('''
# Tuned LightGBM params: literal copy of outputs/optuna/lgb_best_params.json (T1, trial 25, CV 0.76240)
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    bagging_freq=1,
    num_leaves=9,
    max_depth=4,
    min_child_samples=63,
    feature_fraction=0.33659756616656034,
    bagging_fraction=0.8531556342271209,
    reg_alpha=0.7533466964187072,
    reg_lambda=7.219902584162699,
    min_split_gain=0.6750913711780251,
    cat_smooth=56.94958502272585,
    cat_l2=76.69763415418393,
    min_data_per_group=278,
    max_cat_threshold=25,
)
LGB_SEEDS = [42, 43, 44, 45, 46]
lgb_imps, _oofs, _tests, lgb_iters = [], [], [], []
for s in LGB_SEEDS:
    print(f"--- LightGBM seed {s}")
    o, t, _, it = run_cv_lgb(X_lgb, y, X_test_lgb, LGB_PARAMS, folds, seed=s, cat_features=LGB_CATS,
                             importance_out=lgb_imps)
    _oofs.append(o); _tests.append(t); lgb_iters.append(it)
lgb_oof, lgb_test = np.mean(_oofs, axis=0), np.mean(_tests, axis=0)   # fixed formula: plain average
lgb_fold_aucs = [roc_auc_score(y[folds == k], lgb_oof[folds == k]) for k in range(5)]
print(f"E19 (5-seed average) fold AUCs: {np.round(lgb_fold_aucs, 5)} | mean {np.mean(lgb_fold_aucs):.5f} "
      f"| OOF {roc_auc_score(y, lgb_oof):.5f}")
''')
md("""
#### Final model, part 2 — E20: tuned XGBoost (blend component)
XGBoost (`hist` trees, native categorical handling) on the same features and folds, with the tuned parameters
from X1. It was tuned at learning rate 0.05, which is also the final LightGBM learning rate, so no refit was
needed. Early stopping is on each validation fold (200 rounds), and `predict_proba` uses the best iteration.
On its own it is slightly weaker than E19 (0.76206 vs 0.76308). It is used only as the second model in the blend.
""")
code('''
# Tuned XGBoost params: literal copy of outputs/optuna/xgb_best_params.json (X1, CV 0.76206)
XGB_PARAMS = dict(
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=30.14958744826777,
    subsample=0.6620910504918238,
    colsample_bytree=0.6874890830880247,
    reg_alpha=11.520010553131216,
    reg_lambda=0.004388272705102041,
    gamma=1.2995702542443917,
    max_cat_threshold=23,
)
xgb_oof, xgb_test, xgb_fold_aucs, xgb_iters = run_cv_xgb(X_lgb, y, X_test_lgb, XGB_PARAMS, folds, seed=SEED)
''')
md("""
### 5.4 (Optional) Ensemble Learning
**B1 / E21: rank-average blend of LightGBM (E19) and XGBoost (E20), using a fixed formula (rule R2).**
Each model's predictions are converted to ranks scaled to (0, 1] (`rankdata(p) / len(p)`), separately for OOF
and test, and then combined as `blend = w · rank_lgb + (1 − w) · rank_xgb`. The LightGBM weight `w` was chosen
on a coarse grid 0.0, 0.1, …, 1.0 by OOF AUC (the grid is coarse to limit overfitting to the OOF predictions),
which gave **w = 0.7**. It is used here as a literal. The two models are highly correlated (Spearman 0.985 on
OOF), so the gain is small: +0.00033 on 5/5 folds. The gain held on the 3 alternative fold sets, so the blend
was kept after the CONFIRM step (4/4 fold sets won).

*Note:* the blend is a **rank score**. It is fine for AUC, which only uses the order of the predictions, but it
is not a calibrated probability of default. For the business tables in §8 (expected default rates at
cut-offs) we use the LightGBM probabilities `lgb_oof` / `lgb_test`.
""")
code('''
# Source: scipy.stats.rankdata (see §9); blend formula and weight from B1 (src/blend.py)
from scipy.stats import rankdata, spearmanr
W_LGB = 0.7

def rank01(p):
    return rankdata(p) / len(p)

blend_oof = W_LGB * rank01(lgb_oof) + (1 - W_LGB) * rank01(xgb_oof)
blend_test = W_LGB * rank01(lgb_test) + (1 - W_LGB) * rank01(xgb_test)
blend_fold_aucs = [roc_auc_score(y[folds == k], blend_oof[folds == k]) for k in range(5)]
print(f"Spearman(LightGBM, XGBoost) on OOF: {spearmanr(lgb_oof, xgb_oof).correlation:.4f}")
print(pd.DataFrame({"E19 LightGBM": lgb_fold_aucs, "E20 XGBoost": xgb_fold_aucs, "E21 blend": blend_fold_aucs},
                   index=[f"fold {k+1}" for k in range(5)]).T.assign(mean=lambda d: d.mean(axis=1)).round(5))
''')
code('''
BEST_EXP = "E21"
final_runs = {"E19": (lgb_oof, lgb_test), "E20": (xgb_oof, xgb_test), "E21": (blend_oof, blend_test)}
for exp, (o, t) in final_runs.items():   # consistency with the experiment runner (src/run_experiment3.py, src/blend.py)
    stored = OUT / "test_preds" / f"{exp}_test.npy"
    if stored.exists():
        assert np.array_equal(np.load(stored), t), f"notebook test prediction differs from the logged {exp} run"
        assert np.array_equal(np.load(OUT / "oof" / f"{exp}_oof.npy"), o), f"notebook OOF differs from {exp}"
        print(f"{exp}: notebook predictions are bit-identical to the stored experiment artifacts")
    else:
        print(f"{exp}: no stored artifacts (fresh environment) — see Appendix A for the logged results")
best_importance = pd.concat(lgb_imps, axis=1)   # 5 seeds x 5 folds of the final LightGBM
best_importance = pd.DataFrame({"gain_mean": best_importance.mean(axis=1), "gain_std": best_importance.std(axis=1)}
                               ).sort_values("gain_mean", ascending=False)
''')

# ---------------------------------------------------------------- 6
md(T(95))
md("""
### 6.1 Predicting on the test data
The prediction comes from the final model **E21**: the fixed-formula rank-average blend (§5.4,
`w = 0.7`) of the tuned 5-seed LightGBM (E19) and the tuned XGBoost (E20). Each component is the average of
its fold models' `predict_proba`. No prediction is changed by hand (rule R2). The E0 baseline prediction
(`e0_test_pred`) is kept only for comparison.
""")
code('''
# Make predictions for the test data
# Final model E21: 0.7 * rank(LightGBM E19) + 0.3 * rank(XGBoost E20)
pred = blend_test
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

#### 7.1 Preview: top 20 features of the final LightGBM (E19) by mean gain over 5 seeds x 5 folds
""")
code('''
fig, ax = plt.subplots(figsize=(7, 6))
top20 = best_importance.head(20)[::-1]
ax.barh(top20.index, top20["gain_mean"], xerr=top20["gain_std"], color="#4C72B0")
ax.set_xlabel("mean gain (± std over 25 models)"); ax.set_title("E19 LightGBM: top 20 features by gain")
fig.savefig("reports/figures/E19_top20_gain.png", dpi=120, bbox_inches="tight"); plt.show()
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
md("""
## Appendix B. Optuna search (T1 LightGBM, X1 XGBoost)
Trial history and parameter importance (fANOVA, `optuna.importance.get_param_importances`). The plots are drawn
from the saved studies in `outputs/optuna/`, which `RUN_OPTUNA = True` in §5.3 regenerates. The dashed line is
the untuned reference: E10 (Sprint-2 `BASE_PARAMS`, CV 0.75790).
""")
code('''
study_files = {"T1 LightGBM (60 trials)": OUT / "optuna" / "lgb_study.csv",
               "X1 XGBoost (30 trials)": OUT / "optuna" / "xgb_study.csv"}
if all(f.exists() for f in study_files.values()):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (name, f) in zip(axes, study_files.items()):
        v = pd.read_csv(f)["value"].values
        ax.scatter(range(len(v)), v, s=16, color="#86b6ef", label="trial")
        ax.plot(range(len(v)), np.maximum.accumulate(v), color="#2a78d6", lw=2, label="best so far")
        ax.axhline(0.75790, color="#898781", lw=1, ls="--")
        ax.set_title(name, loc="left"); ax.set_xlabel("trial")
        ax.grid(axis="y", color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("mean 5-fold CV AUC"); axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_ylim(np.percentile(pd.read_csv(study_files["X1 XGBoost (30 trials)"])["value"], 5) - 0.001, 0.7635)
    fig.tight_layout(); fig.savefig("reports/figures/optuna_history_sprint3.png", dpi=120); plt.show()
    display(pd.concat({k: pd.read_csv(OUT / "optuna" / f"{k}_param_importance.csv", index_col=0)["importance"]
                       for k in ["lgb", "xgb"]}, axis=1).round(3).sort_values("lgb", ascending=False))
else:
    print("No saved Optuna studies in outputs/optuna/ — set RUN_OPTUNA = True in §5.3 to create them")
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
