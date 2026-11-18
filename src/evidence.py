# Task 1.2 - one-number previews of the market-analysis hypotheses (train only, univariate)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def hypothesis_evidence(df):
    d = df.copy()
    y = d["TARGET"]
    def q_rates(s, q=5):
        s = s.replace([np.inf, -np.inf], np.nan)
        m = s.notna()
        b = pd.qcut(s[m].rank(method="first"), q, labels=[f"Q{i}" for i in range(1, q + 1)])
        return y[m].groupby(b, observed=True).mean()
    def q1q5(s):
        r = q_rates(s)
        return f"Q1 {r.iloc[0]*100:.1f}% → Q5 {r.iloc[-1]*100:.1f}%"
    def uni_auc(s):
        m = s.notna()
        a = roc_auc_score(y[m], s[m])
        return max(a, 1 - a)
    emp = d["DAYS_EMPLOYED"].where(d["DAYS_EMPLOYED"] != 365243)
    age_years = -d["DAYS_BIRTH"] / 365.25
    ev = {}
    ev["H-M1"] = "annuity/income " + q1q5(d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"])
    ev["H-M2"] = "credit/annuity " + q1q5(d["AMT_CREDIT"] / d["AMT_ANNUITY"])
    ev["H-M3"] = ("age (youngest→oldest) " + q1q5(age_years) +
                  "; employment length (shortest→longest, 365243 excluded) " + q1q5(-emp))
    num = d.drop(columns=["SK_ID_CURR", "TARGET"]).select_dtypes("number")
    aucs = {c: uni_auc(num[c]) for c in num.columns if num[c].nunique() > 1}
    top = sorted(aucs.items(), key=lambda kv: -kv[1])[:4]
    ev["H-M4"] = ("EXT_SOURCE_2 " + q1q5(d["EXT_SOURCE_2"]) + "; top univariate AUCs: " +
                  ", ".join(f"{c} {a:.3f}" for c, a in top))
    over = d["AMT_CREDIT"] > d["AMT_GOODS_PRICE"]
    known = d["AMT_GOODS_PRICE"].notna()
    ev["H-M5"] = (f"credit > goods price {y[over & known].mean()*100:.1f}% (n={int((over & known).sum())}) "
                  f"vs credit ≤ goods price {y[~over & known].mean()*100:.1f}%")
    ev["H-M6"] = ("abs(DAYS_ID_PUBLISH) (most recent→oldest) " + q1q5(d["DAYS_ID_PUBLISH"].abs()) +
                  "; abs(DAYS_LAST_PHONE_CHANGE) " + q1q5(d["DAYS_LAST_PHONE_CHANGE"].abs()))
    yr = d["AMT_REQ_CREDIT_BUREAU_YEAR"]
    grp = pd.cut(yr, [-0.5, 0.5, 2.5, np.inf], labels=["0", "1-2", "3+"])
    r = y.groupby(grp, observed=True).mean()
    ev["H-M7"] = ("bureau enquiries in last year: " + ", ".join(f"{k}: {v*100:.1f}%" for k, v in r.items()) +
                  f", missing: {y[yr.isna()].mean()*100:.1f}%")
    r = y.groupby(d["REGION_RATING_CLIENT"]).mean()
    ev["H-M8"] = "rating " + ", ".join(f"{k}: {v*100:.1f}%" for k, v in r.items())
    return ev

if __name__ == "__main__":
    import sys
    ev = hypothesis_evidence(pd.read_csv(sys.argv[1]))
    for k, v in ev.items(): print(k, "|", v)
