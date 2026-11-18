# Market Analysis: Consumer Credit Risk for Under-Banked Borrowers

> Scope: general domain knowledge only. No external datasets, statistics or
> articles were used. The only numbers below come from `input/train.csv`. Any
> number that is an assumption is marked *illustrative*.

## 1. Industry overview

Consumer finance companies such as Home Credit lend to people with **little or
no formal credit history**: young adults, workers in the informal economy,
first-time borrowers and households in emerging markets. Traditional
scorecards depend heavily on a credit-bureau file. "Thin-file" and "no-file"
applicants are therefore often rejected, or offered poor terms, even though
many of them would repay reliably. A lender that can tell the reliable ones
apart from the rest using **application-time information** gains both a
commercial edge and a social one, because it widens access to credit.

Main products in this segment:

| Product | How it works | Link to this dataset |
| ------- | ------------ | -------------------- |
| **Cash loan** | Fixed amount, repaid in fixed periodic installments (annuity) | `NAME_CONTRACT_TYPE = Cash loans`, `AMT_CREDIT`, `AMT_ANNUITY` |
| **Revolving loan** | Credit line or card with a reusable limit | `NAME_CONTRACT_TYPE = Revolving loans` |
| **Point-of-sale (consumer) loan** | Financing for a specific purchase (electronics, furniture) arranged at the store | `AMT_GOODS_PRICE` (the price of the financed goods) |

## 2. Business problem

Every approval decision can be wrong in one of two ways:

| Decision error | Business cost |
| -------------- | ------------- |
| **Approve a client who defaults** (false negative) | Loss of principal, collection costs, higher provisioning and capital requirements |
| **Reject a client who would have repaid** (false positive) | Lost interest income and customer lifetime value; a good client is excluded from credit |

The two costs are asymmetric and they change over time with funding costs,
competition and regulation. So the lender does not need a fixed yes/no
classifier. It needs a **risk ranking**: a score that orders applicants from
safest to riskiest. Management then sets an approval cut-off that matches its
current risk appetite (for example, "approve the safest 80%") and can move that
cut-off without retraining the model.

## 3. Why ROC AUC is the right metric

- **It measures ranking, not a threshold.** AUC is the probability that a
  randomly chosen defaulter gets a higher score than a randomly chosen
  non-defaulter. That is exactly the property a risk ranking needs, because the
  cut-off is chosen later as a business decision.
- **It is robust to class imbalance.** About 8.1% of training clients have
  `TARGET = 1`. A model that predicts "no default" for everyone reaches about
  92% accuracy and is worthless. AUC gives that model 0.5.
- **It is the industry standard in disguise.** Credit-risk teams report the
  **Gini coefficient**, where Gini = 2·AUC − 1. For example, an AUC of 0.76
  corresponds to a Gini of 0.52.
- **Limitation:** AUC ignores calibration. If the scores are later used as
  probability-of-default estimates (for pricing or provisioning), they must be
  calibrated separately. This is noted for the Business Proposal.

## 4. Stakeholders

| Stakeholder | Main interest | What they need from the model |
| ----------- | ------------- | ----------------------------- |
| Risk management | Lower default rate at a given approval rate | High, stable AUC; monitoring of drift |
| Sales / product | High approval rates, fast decisions | A score good enough to automate most decisions |
| Loan officers | Clear reasons for borderline decisions | Explanations (feature importance, SHAP) for individual cases |
| Regulators | Fair, explainable, non-discriminatory lending | Documented features; careful handling of `CODE_GENDER`, age and family status |
| Clients | Access to credit on fair terms; quick answers | Fewer wrongful rejections; transparent criteria |

## 5. Value of a better model

1. **Better trade-off curve.** A higher AUC moves the whole approval-rate vs
   default-rate curve. At the same approval rate the lender has fewer defaults,
   or at the same default rate it can approve more clients, which means more
   revenue and more inclusion.
2. **Automation.** Applicants with clearly low or clearly high scores can be
   decided automatically. Only the uncertain middle band goes to manual review,
   which cuts cost and decision time.
3. **Risk-based terms.** Instead of a hard yes/no, risky-but-not-bad applicants
   can be offered a smaller amount, a shorter term or a larger down payment.
4. *Illustrative example (assumed numbers, not from data):* take a portfolio of
   100,000 approved loans with an average exposure of 10,000 and a 50% loss
   given default. Each percentage point of default rate removed at the same
   approval rate then saves about 100,000 × 1% × 10,000 × 50% = **5 million**
   currency units.

## 6. Project success criteria

| Level | Criterion |
| ----- | --------- |
| Technical | 5-fold CV AUC clearly above the tutorial baseline (E0) and stable across folds (low std); leaderboard AUC consistent with CV |
| Business | The Business Proposal shows the default rate among approved clients at several approval cut-offs (e.g. 70/80/90%) for the final model vs the baseline |
| Academic | Clear hypothesis → experiment → analysis → revision trail in `experiments/experiment_log.csv`; fully reproducible notebook (identical CSV hash on re-run); all sources cited in `REFERENCES.md` |

## 7. Data limitations and ethical considerations

- **Application-time data only.** The data has no repayment history, previous
  applications or bureau records, so the achievable AUC is capped. A realistic
  range for strong models on these 32 columns is about 0.75–0.77.
- **Black-box external scores.** `EXT_SOURCE_1/2/3` are likely to be the
  strongest predictors, but their construction is unknown, so they cannot be
  explained to clients or regulators in detail. They may also carry biases
  from their own sources.
- **Missing values carry information.** `OWN_CAR_AGE` and `EXT_SOURCE_1` are
  mostly missing, and `DAYS_EMPLOYED = 365243` is a placeholder (typically
  pensioners or unemployed). Whether a value is missing can itself be
  predictive, and it must be handled the same way in train and test.
- **Fairness.** `CODE_GENDER`, age (`DAYS_BIRTH`), `NAME_FAMILY_STATUS` and
  `CNT_CHILDREN` relate to protected or sensitive characteristics. Using them
  may raise the score, but it can produce discriminatory decisions and may be
  prohibited in some jurisdictions. The Business Proposal will report model
  performance with and without these features, and the default/approval rates
  by group.
- **Population drift.** A model trained on past applicants may degrade as the
  economy, the products or the marketing channels change. A production model
  would need monitoring (for example, a population stability index) and
  periodic retraining.

## 8. Hypotheses from domain knowledge (tested in Sprint 2 EDA)

Each hypothesis is a testable statement about the default rate (`TARGET`
mean). The *Initial evidence* column is a one-number preview computed from
`train.csv` (Task 1.2). Quintiles Q1–Q5 run from the lowest to the highest
feature value. Overall default rate: **8.07%**.

| ID | Hypothesis (testable) | Feature(s) | Initial evidence (train) |
| -- | --------------------- | ---------- | ------------------------ |
| H-M1 | A higher payment burden (`AMT_ANNUITY / AMT_INCOME_TOTAL`) means a higher default rate | engineered ratio | annuity/income Q1 7.2% → Q5 8.4% — *supports (weak)* |
| H-M2 | A longer implied loan term (`AMT_CREDIT / AMT_ANNUITY`) means higher risk | engineered ratio | credit/annuity Q1 8.0% → Q5 6.4% — ***contradicts** at Q5 (check non-monotonicity in Sprint 2)* |
| H-M3 | Younger clients, and clients with shorter employment, have higher default rates | `DAYS_BIRTH`, `DAYS_EMPLOYED` | age (youngest→oldest) Q1 11.4% → Q5 5.2%; employment length (shortest→longest, 365243 excluded) Q1 11.0% → Q5 5.2% — *supports* |
| H-M4 | Low external scores (`EXT_SOURCE_*`) are the strongest single predictors | `EXT_SOURCE_1/2/3` | EXT_SOURCE_2 Q1 15.3% → Q5 3.5%; top univariate AUCs: EXT_SOURCE_3 0.680, EXT_SOURCE_1 0.665, EXT_SOURCE_2 0.658, DAYS_BIRTH 0.584 — *supports (univariate AUC on non-missing rows)* |
| H-M5 | Financing more than the goods price (`AMT_CREDIT > AMT_GOODS_PRICE`) signals higher risk | engineered flag | credit > goods price 8.8% (n=111021) vs credit ≤ goods price 6.8% — *supports* |
| H-M6 | Clients with recent document or phone changes (small `abs(DAYS_ID_PUBLISH)`, `abs(DAYS_LAST_PHONE_CHANGE)`) are riskier | `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE` | abs(DAYS_ID_PUBLISH) (most recent→oldest) Q1 10.1% → Q5 6.0%; abs(DAYS_LAST_PHONE_CHANGE) Q1 9.3% → Q5 5.7% — *supports* |
| H-M7 | Many recent credit-bureau enquiries signal credit hunger and therefore higher risk | `AMT_REQ_CREDIT_BUREAU_*` | bureau enquiries in last year: 0: 7.1%, 1-2: 7.7%, 3+: 8.2%, missing: 10.4% — *supports (weak); missingness itself is risky* |
| H-M8 | Clients in region rating 3 default more often than those in rating 1 | `REGION_RATING_CLIENT` | rating 1: 4.7%, 2: 7.9%, 3: 11.0% — *supports* |

These previews are univariate and do not establish causation. Each hypothesis
will be confirmed or rejected in the Sprint 2 EDA. The ones that hold become
candidate features for experiments E1, E2, ….
