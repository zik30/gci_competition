"""Feature pipeline for all Sprint-2 experiments (Task 2.1). Inlined into notebooks/main.ipynb §4.

One function, `make_features(train, test, flags)`, switched by FEATURE_FLAGS. It never uses TARGET:
unsupervised steps (category levels, group means) are computed on concat(train, test), which is
allowed by rule R6 / user decision Q6 because no target information is involved.
"""
import numpy as np
import pandas as pd

FEATURE_FLAGS_DEFAULT = {
    "raw_all": False,              # E2
    "days_employed_fix": False,    # E3
    "credit_ratios": False,        # E4
    "ext_source_aggs": False,      # E5
    "time_ratios": False,          # E6
    "missing_bureau": False,       # E7
    "per_person": False,           # E8
    "group_stats": False,          # E9
    "drop_low_importance": False,  # E10 (False, or a literal list of feature names to drop)
    "drop_gender": False,          # E11 (analysis)
    "ext_source_weighted": False,  # E12 (optional)
    "count_encoding": False,       # E13 (optional)
}

ID_COL, TARGET_COL = "SK_ID_CURR", "TARGET"
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
BUREAU_COLS = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON",
               "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
DAYS_EMPLOYED_PLACEHOLDER = 365243


def _div(a, b):
    """Safe division: x / 0 -> NaN (never inf)."""
    return a / b.replace(0, np.nan)


def _tutorial_e0_features(train, test):
    """Tutorial preprocessing of its 5 features (E0/E1 base). Mirrors notebook §4 of tutorial.ipynb.

    Source: GCI tutorial.ipynb §4 — adapted: written as one function, assignment instead of inplace.
    Count encoding / mean imputation use train only, exactly like the tutorial.
    """
    use = ["NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "EXT_SOURCE_2", "OWN_CAR_AGE", "ORGANIZATION_TYPE"]
    tr, te = train[use].copy(), test[use].copy()
    for d in (tr, te):
        d["NAME_CONTRACT_TYPE"] = d["NAME_CONTRACT_TYPE"].replace({"Cash loans": 0, "Revolving loans": 1}).astype(int)
    ce = tr["ORGANIZATION_TYPE"].value_counts()
    tr["ORGANIZATION_TYPE"] = tr["ORGANIZATION_TYPE"].map(ce)
    te["ORGANIZATION_TYPE"] = te["ORGANIZATION_TYPE"].map(ce)
    m = tr["EXT_SOURCE_2"].mean()
    tr["EXT_SOURCE_2"] = tr["EXT_SOURCE_2"].fillna(m)
    te["EXT_SOURCE_2"] = te["EXT_SOURCE_2"].fillna(m)
    out = []
    for d in (tr, te):
        d.loc[d["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan
        d["OWN_CAR_AGE"] = d["OWN_CAR_AGE"] // 10
        out.append(pd.get_dummies(d["OWN_CAR_AGE"]).add_prefix("OWN_CAR_AGE_"))
    cols = list(out[0].columns)
    assert set(out[1].columns) <= set(cols)
    tr = tr.drop(columns="OWN_CAR_AGE").join(out[0].astype(int))
    te = te.drop(columns="OWN_CAR_AGE").join(out[1].reindex(columns=cols, fill_value=False).astype(int))
    return tr, te


def make_features(train: pd.DataFrame, test: pd.DataFrame, flags: dict):
    """Returns X_train, X_test, feature_names (sorted, deterministic), categorical_feature_names.

    Applied to concat(train, test) for unsupervised steps only. Never uses TARGET.
    """
    flags = {**FEATURE_FLAGS_DEFAULT, **(flags or {})}
    unknown = set(flags) - set(FEATURE_FLAGS_DEFAULT)
    assert not unknown, f"unknown flags: {unknown}"
    n_tr = len(train)
    raw = pd.concat([train.drop(columns=[TARGET_COL], errors="ignore"), test], axis=0, ignore_index=True)
    assert (raw[ID_COL].values[:n_tr] == train[ID_COL].values).all()
    assert (raw[ID_COL].values[n_tr:] == test[ID_COL].values).all()
    raw_cols = [c for c in raw.columns if c != ID_COL]
    R = raw.copy()                       # working copy for engineered features
    feats = {}                           # name -> Series (engineered)

    # E7 / H-M7 + A2: missingness count is taken on the ORIGINAL raw values (before any fix)
    if flags["missing_bureau"]:
        feats["N_MISSING"] = raw[raw_cols].isna().sum(axis=1).astype(float)
        feats["BUREAU_REQ_MISSING"] = raw[BUREAU_COLS].isna().all(axis=1).astype(int)
        feats["BUREAU_REQ_TOTAL"] = raw[BUREAU_COLS].sum(axis=1, min_count=1)

    # E3 / H-M3 + A3: DAYS_EMPLOYED = 365243 is a placeholder (mostly pensioners) -> NaN + flag
    if flags["days_employed_fix"]:
        anom = R["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER
        R["DAYS_EMPLOYED"] = R["DAYS_EMPLOYED"].where(~anom, np.nan)
        feats["DAYS_EMPLOYED_ANOM"] = anom.astype(int)

    # Base feature set
    if flags["raw_all"]:
        # E2 / H-M3, H-M4: all 32 raw features; strings -> category with levels fixed from the sorted
        # union of train+test (identical codes every run); no imputation / scaling (LightGBM handles NaN)
        base = R[raw_cols].copy()
        for c in base.columns:
            if not pd.api.types.is_numeric_dtype(base[c]):
                levels = sorted(base[c].dropna().unique().tolist())
                base[c] = pd.Categorical(base[c], categories=levels)
    else:
        # E0 / E1: tutorial's 5 features with tutorial preprocessing (10 columns)
        tr0, te0 = _tutorial_e0_features(train, test)
        base = pd.concat([tr0, te0], axis=0, ignore_index=True)

    # E4 / H-M1, H-M2, H-M5: domain ratios
    if flags["credit_ratios"]:
        feats["CREDIT_ANNUITY_RATIO"] = _div(R["AMT_CREDIT"], R["AMT_ANNUITY"])
        feats["ANNUITY_INCOME_RATIO"] = _div(R["AMT_ANNUITY"], R["AMT_INCOME_TOTAL"])
        feats["CREDIT_INCOME_RATIO"] = _div(R["AMT_CREDIT"], R["AMT_INCOME_TOTAL"])
        feats["CREDIT_GOODS_RATIO"] = _div(R["AMT_CREDIT"], R["AMT_GOODS_PRICE"])
        feats["GOODS_MINUS_CREDIT"] = R["AMT_GOODS_PRICE"] - R["AMT_CREDIT"]

    # E5 / H-M4: aggregates of the external scores (NaN-aware; product is NaN if any score is NaN)
    if flags["ext_source_aggs"]:
        E = R[EXT_COLS]
        feats["EXT_SOURCE_MEAN"] = E.mean(axis=1)
        feats["EXT_SOURCE_MIN"] = E.min(axis=1)
        feats["EXT_SOURCE_MAX"] = E.max(axis=1)
        feats["EXT_SOURCE_STD"] = E.std(axis=1, ddof=0)
        feats["EXT_SOURCE_PROD"] = E.prod(axis=1, skipna=False, min_count=3)
        feats["EXT_SOURCE_NAN_COUNT"] = E.isna().sum(axis=1).astype(float)

    # E6 / H-M3, H-M6: life-stage stability (all DAYS_* are negative, ratios are positive)
    if flags["time_ratios"]:
        age_years = -R["DAYS_BIRTH"] / 365.25
        feats["EMPLOYED_BIRTH_RATIO"] = _div(R["DAYS_EMPLOYED"], R["DAYS_BIRTH"])
        feats["ID_PUBLISH_BIRTH_RATIO"] = _div(R["DAYS_ID_PUBLISH"], R["DAYS_BIRTH"])
        feats["REGISTRATION_BIRTH_RATIO"] = _div(R["DAYS_REGISTRATION"], R["DAYS_BIRTH"])
        feats["PHONE_CHANGE_BIRTH_RATIO"] = _div(R["DAYS_LAST_PHONE_CHANGE"], R["DAYS_BIRTH"])
        feats["CAR_AGE_BIRTH_RATIO"] = _div(R["OWN_CAR_AGE"], age_years)
        # implied term in months = credit / annuity (annuity assumed monthly)
        feats["AGE_AT_LOAN_END"] = age_years + _div(R["AMT_CREDIT"], R["AMT_ANNUITY"]) / 12

    # E8: disposable income per head
    if flags["per_person"]:
        feats["INCOME_PER_PERSON"] = _div(R["AMT_INCOME_TOTAL"], R["CNT_FAM_MEMBERS"])
        feats["ANNUITY_PER_PERSON"] = _div(R["AMT_ANNUITY"], R["CNT_FAM_MEMBERS"])
        feats["CHILDREN_RATIO"] = _div(R["CNT_CHILDREN"], R["CNT_FAM_MEMBERS"])

    # E9: position relative to peer group. Group means are computed on train+test together; this is
    # an unsupervised statistic that never touches TARGET (allowed by R6 / Q6).
    if flags["group_stats"]:
        ext_f = feats["EXT_SOURCE_MEAN"] if flags["ext_source_aggs"] else R["EXT_SOURCE_2"]
        cred_f = feats["CREDIT_ANNUITY_RATIO"] if flags["credit_ratios"] else R["AMT_CREDIT"]
        cols = {"AMT_INCOME_TOTAL": R["AMT_INCOME_TOTAL"],
                ("EXT_SOURCE_MEAN" if flags["ext_source_aggs"] else "EXT_SOURCE_2"): ext_f,
                ("CREDIT_ANNUITY_RATIO" if flags["credit_ratios"] else "AMT_CREDIT"): cred_f}
        for g in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE"]:
            key = R[g].fillna("__NA__")
            for fname, f in cols.items():
                gm = f.groupby(key).transform("mean")
                feats[f"{fname}_MINUS_{g}_MEAN"] = f - gm
                feats[f"{fname}_DIV_{g}_MEAN"] = _div(f, gm)

    # E12 (optional) / H-M4: weighted external score 2*E1 + 1*E2 + 3*E3, normalised by the weights of the
    # non-missing scores (fixed formula on features; weights chosen from domain/univariate AUC, not predictions)
    if flags["ext_source_weighted"]:
        w = pd.Series({"EXT_SOURCE_1": 2.0, "EXT_SOURCE_2": 1.0, "EXT_SOURCE_3": 3.0})
        E = R[EXT_COLS]
        feats["EXT_SOURCE_WEIGHTED"] = _div((E * w).sum(axis=1, min_count=1), E.notna().mul(w).sum(axis=1))

    # E13 (optional): count encoding of the two high-cardinality categoricals, in addition to category dtype.
    # Counts are computed on train+test (unsupervised, no TARGET: allowed by R6 / Q6).
    if flags["count_encoding"]:
        for c in ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]:
            feats[f"{c}_COUNT"] = R[c].map(R[c].value_counts()).astype(float)

    X = pd.concat([base, pd.DataFrame(feats)], axis=1) if feats else base
    assert not X.columns.duplicated().any(), X.columns[X.columns.duplicated()].tolist()

    # E10: drop low-importance features (literal list, stored in the experiment's flags)
    if flags["drop_low_importance"]:
        X = X.drop(columns=[c for c in flags["drop_low_importance"] if c in X.columns])
    # E11 (analysis): exclude the sensitive attribute
    if flags["drop_gender"] and "CODE_GENDER" in X.columns:
        X = X.drop(columns="CODE_GENDER")

    assert ID_COL not in X.columns and TARGET_COL not in X.columns
    names = sorted(X.columns)
    X = X[names]
    cats = sorted(c for c in names if isinstance(X[c].dtype, pd.CategoricalDtype))
    num = X[[c for c in names if c not in cats]]
    vals = num.to_numpy(dtype=float)
    assert not np.isinf(vals).any(), "inf in numeric features"
    X_tr = X.iloc[:n_tr].reset_index(drop=True)
    X_te = X.iloc[n_tr:].reset_index(drop=True)
    assert list(X_tr.columns) == list(X_te.columns) and (X_tr.dtypes == X_te.dtypes).all()
    return X_tr, X_te, names, cats
