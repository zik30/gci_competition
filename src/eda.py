"""Sprint-2 EDA (notebook §3.2, analyses A1-A8). Every function works on copies and never modifies its
inputs. Figures are saved to reports/figures/ (PNG, 120 dpi). Inlined into notebooks/main.ipynb.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FIG_DIR = Path("reports/figures")
EDA_SEED = 42
CAT_COLS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=120, bbox_inches="tight")


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (vectorised)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def rate_table(y, groups):
    """Default rate, count and Wilson CI per group (NaN group kept as 'missing')."""
    g = pd.Series(groups).astype(object).where(pd.Series(groups).notna(), "missing")
    t = pd.DataFrame({"y": np.asarray(y), "g": g.values}).groupby("g", sort=False)["y"].agg(["sum", "count"])
    t["rate"] = t["sum"] / t["count"]
    t["ci_lo"], t["ci_hi"] = wilson_ci(t["sum"], t["count"])
    return t


def a1_target_balance(train):
    rate = train["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(4, 3))
    train["TARGET"].value_counts().sort_index().plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title(f"A1 Target balance (default rate {rate:.2%})")
    ax.set_xlabel("TARGET"); ax.set_ylabel("rows")
    _save(fig, "A1_target_balance.png")
    return rate


def a2_missing(train):
    d = train.drop(columns=["SK_ID_CURR", "TARGET"])
    miss = d.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    rows = []
    for c in miss.index:
        m = d[c].isna()
        rows.append({"column": c, "missing_%": 100 * m.mean(),
                     "default_rate_missing_%": 100 * train.loc[m, "TARGET"].mean(),
                     "default_rate_present_%": 100 * train.loc[~m, "TARGET"].mean()})
    tab = pd.DataFrame(rows).set_index("column").round(2)
    fig, ax = plt.subplots(figsize=(8, 4))
    (100 * miss).plot.bar(ax=ax, color="#4C72B0")
    ax.set_ylabel("% missing (train)"); ax.set_title("A2 Missing values per column")
    _save(fig, "A2_missing.png")
    return tab


def a3_days_employed(train):
    anom = train["DAYS_EMPLOYED"] == 365243
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(train["DAYS_EMPLOYED"], bins=80, color="#4C72B0")
    axes[0].set_title("A3 DAYS_EMPLOYED (raw) — spike at 365243")
    axes[1].hist(-train.loc[~anom, "DAYS_EMPLOYED"] / 365.25, bins=60, color="#55A868")
    axes[1].set_title("Employment years (placeholder removed)")
    _save(fig, "A3_days_employed.png")
    t = pd.DataFrame({"rows": [int(anom.sum()), int((~anom).sum())],
                      "default_rate_%": [100 * train.loc[anom, "TARGET"].mean(),
                                         100 * train.loc[~anom, "TARGET"].mean()]},
                     index=["DAYS_EMPLOYED = 365243", "other"]).round(2)
    overlap = train.loc[anom, "NAME_INCOME_TYPE"].value_counts()
    return t, overlap


def a4_deciles(train):
    d = train.copy()
    emp = d["DAYS_EMPLOYED"].where(d["DAYS_EMPLOYED"] != 365243)
    feats = {
        "EXT_SOURCE_1": d["EXT_SOURCE_1"], "EXT_SOURCE_2": d["EXT_SOURCE_2"], "EXT_SOURCE_3": d["EXT_SOURCE_3"],
        "age_years": -d["DAYS_BIRTH"] / 365.25, "employment_years": -emp / 365.25,
        "annuity/income (H-M1)": d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"],
        "credit/annuity (H-M2)": d["AMT_CREDIT"] / d["AMT_ANNUITY"],
        "credit/goods (H-M5)": d["AMT_CREDIT"] / d["AMT_GOODS_PRICE"],
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    summary = {}
    for ax, (name, s) in zip(axes.ravel(), feats.items()):
        s = s.replace([np.inf, -np.inf], np.nan)
        m = s.notna()
        dec = pd.qcut(s[m].rank(method="first"), 10, labels=False) + 1
        t = rate_table(d.loc[m, "TARGET"], dec).sort_index()
        ax.plot(t.index, 100 * t["rate"], marker="o", color="#C44E52")
        ax.fill_between(t.index, 100 * t["ci_lo"], 100 * t["ci_hi"], alpha=0.25, color="#C44E52")
        ax.set_title(name); ax.set_xlabel("decile (low → high)"); ax.set_xticks(range(1, 11))
        summary[name] = (100 * t["rate"]).round(1).tolist()
    axes[0, 0].set_ylabel("default rate %"); axes[1, 0].set_ylabel("default rate %")
    fig.suptitle("A4 Default rate by decile (95% Wilson CI)")
    fig.tight_layout()
    _save(fig, "A4_deciles.png")
    return pd.DataFrame(summary, index=[f"D{i}" for i in range(1, 11)]).T


def a5_categoricals(train):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    out = {}
    for ax, c in zip(axes.ravel(), CAT_COLS):
        t = rate_table(train["TARGET"], train[c]).sort_values("count", ascending=False).head(15)
        ax2 = ax.twinx()
        ax.bar(range(len(t)), t["count"], color="#B0C4DE")
        ax2.errorbar(range(len(t)), 100 * t["rate"], yerr=[100 * (t["rate"] - t["ci_lo"]), 100 * (t["ci_hi"] - t["rate"])],
                     fmt="o", color="#C44E52")
        ax.set_xticks(range(len(t))); ax.set_xticklabels([str(i)[:22] for i in t.index], rotation=60, ha="right", fontsize=8)
        ax.set_title(c); ax.set_ylabel("count"); ax2.set_ylabel("default rate %")
        out[c] = (100 * t["rate"]).round(1)
    fig.suptitle("A5 Count (bars) and default rate with 95% CI (dots), top 15 categories")
    fig.tight_layout()
    _save(fig, "A5_categoricals.png")
    return out


def a6_correlation(train):
    num = train.drop(columns=["SK_ID_CURR", "TARGET"]).select_dtypes("number")
    num = num.loc[:, num.nunique() > 1]
    rho = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(rho, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title("A6 Spearman correlation (numeric features)")
    _save(fig, "A6_spearman.png")
    pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(rho.columns) for b in rho.columns[i + 1:]
             if abs(rho.loc[a, b]) > 0.8]
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman_rho"]).round(3)


# Strongest raw features of the LightGBM models (gain ranking of E4/E10, restricted to raw columns)
A7_TOP_FEATURES = ["EXT_SOURCE_2", "EXT_SOURCE_3", "ORGANIZATION_TYPE", "EXT_SOURCE_1",
                   "DAYS_BIRTH", "OCCUPATION_TYPE", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH"]


def raw_matrices(train, test):
    """All 32 raw features; strings -> category with levels from the sorted train+test union."""
    cols = [c for c in test.columns if c != "SK_ID_CURR"]
    both = pd.concat([train[cols], test[cols]], axis=0, ignore_index=True)
    for c in cols:
        if not pd.api.types.is_numeric_dtype(both[c]):
            both[c] = pd.Categorical(both[c], categories=sorted(both[c].dropna().unique().tolist()))
    return both.iloc[:len(train)].reset_index(drop=True), both.iloc[len(train):].reset_index(drop=True)


def a7_drift(X_tr, X_te, top_features=A7_TOP_FEATURES, n_jobs=4, seed=EDA_SEED):
    """Overlaid KDEs for the top features + adversarial validation (train=0 / test=1).

    X_tr / X_te: feature matrices with the same columns (no ID, no TARGET).
    Returns (out-of-fold AUC, top-10 features by gain share).
    """
    import lightgbm as lgb
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), top_features[:8]):
        if isinstance(X_tr[c].dtype, pd.CategoricalDtype):
            a = X_tr[c].value_counts(normalize=True); b = X_te[c].value_counts(normalize=True)
            top = a.index[:10]
            ax.bar(np.arange(len(top)) - 0.2, a[top], 0.4, label="train"); ax.bar(np.arange(len(top)) + 0.2, b.reindex(top), 0.4, label="test")
            ax.set_xticks(range(len(top))); ax.set_xticklabels([str(t)[:12] for t in top], rotation=60, fontsize=7)
        else:
            sns.kdeplot(X_tr[c].dropna(), ax=ax, label="train"); sns.kdeplot(X_te[c].dropna(), ax=ax, label="test")
        ax.set_title(c); ax.legend(fontsize=7)
    fig.suptitle("A7 Train vs test distributions (top 8 features of the best model)")
    fig.tight_layout()
    _save(fig, "A7_train_vs_test.png")

    X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    z = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))].astype(int)
    oof = np.zeros(len(X)); gains = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(z)), z):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed,
                               deterministic=True, force_row_wise=True, n_jobs=n_jobs, verbose=-1)
        m.fit(X.iloc[tr], z[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        gains.append(pd.Series(m.booster_.feature_importance("gain"), index=m.booster_.feature_name()))
    auc = roc_auc_score(z, oof)
    top10 = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False).head(10)
    return auc, (100 * top10 / top10.sum()).round(1).rename("gain_share_%_of_top10")


def a8_fairness(train):
    age = pd.cut(-train["DAYS_BIRTH"] / 365.25, [0, 25, 35, 45, 55, 100], labels=["<25", "25-34", "35-44", "45-54", "55+"])
    out = {}
    for name, g in [("CODE_GENDER", train["CODE_GENDER"]), ("age band", age), ("NAME_FAMILY_STATUS", train["NAME_FAMILY_STATUS"])]:
        t = rate_table(train["TARGET"], g)
        t = t.assign(**{"share_%": 100 * t["count"] / t["count"].sum(), "default_rate_%": 100 * t["rate"],
                        "ci_95_%": [f"{100*a:.1f}–{100*b:.1f}" for a, b in zip(t["ci_lo"], t["ci_hi"])]})
        order = list(g.cat.categories) if isinstance(g.dtype, pd.CategoricalDtype) else sorted(t.index, key=str)
        out[name] = t[["count", "share_%", "default_rate_%", "ci_95_%"]].round(2).reindex([o for o in order if o in t.index])
    return out
