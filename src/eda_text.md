### 3.2 Visualization and analysis of each feature (Sprint 2 EDA, A1–A8)
All figures are saved to `reports/figures/` (PNG, 120 dpi). Default rates come with 95% Wilson confidence
intervals. EDA findings are only ever turned into **features or preprocessing** applied identically to train
and test, never into rules that set predictions (R2).

> Process note: in Sprint 2 the feature experiments E1–E13 were run in the same session as this EDA, driven by
> the Sprint-1 hypotheses and their initial evidence (§3.0.9). The summary table at the end of this section
> therefore confirms after the fact which findings explain which experiment results.

#### A1 Target balance
<!-- cell -->
**A1.** 8.07% of training clients defaulted (13,821 of 171,202). The data is imbalanced, so we use ROC AUC
and stratified folds. Resampling is not needed, because AUC only depends on the ranking.

#### A2 Missing values, and default rate when missing vs present
<!-- cell -->
**A2.** `EXT_SOURCE_1` (69.5%), `OWN_CAR_AGE` (66.0%) and `EXT_SOURCE_3` (31.9%) are mostly or often missing.
The four `AMT_REQ_CREDIT_BUREAU_*` columns are missing together (13.5%), and those clients default at
**10.4% vs 7.7%**, so missingness is informative (H-M7, input for E7). LightGBM routes NaN to its own branch,
so this signal is already usable without extra features. That is consistent with E7 being discarded
(+0.0004, below the 0.0005 threshold).

#### A3 `DAYS_EMPLOYED` placeholder 365243
<!-- cell -->
**A3.** 30,898 training rows (18.0%) have the placeholder `DAYS_EMPLOYED = 365243`. Of these, **30,888 are
pensioners and 10 unemployed**, so the value means "not employed". These clients default less (5.4% vs
8.7%). One tree split isolates the value, so replacing it with NaN plus a flag (E3) adds nothing
(Δ −0.0005, discarded).

#### A4 Default rate by decile for the key numeric features (tests H-M1 … H-M5)
<!-- cell -->
**A4.**
- **H-M4 confirmed.** All three `EXT_SOURCE` scores are steep and monotonic: decile 1 is 17–20%, decile 10 is 2.4–3.1%.
- **H-M3 confirmed.** Age falls from 11.6% to 4.8%. Employment length is flat at about 11% for the shortest
  three deciles and then falls to 4.7%.
- **H-M1 is weak.** Annuity/income rises only from 6.9% to 8.9%.
- **H-M2 is rejected as a monotonic statement.** Credit/annuity (the implied term) is strongly *non-monotonic*:
  4.7%, 11.3%, …, 6.2%, **13.6%**, 5.0%. Risk depends on specific term buckets, which are probably product types.
  A linear score cannot use this, but trees can, and this is part of E4's +0.0078 gain.
- **H-M5 confirmed.** Credit/goods is flat up to decile 6 and then rises to 12.6% at decile 10 (financing well
  above the goods price).

#### A5 Categorical features: count and default rate per category
<!-- cell -->
**A5.** Categories carry large differences:
- Income type: unemployed 20%, pensioner 5.4%.
- Education: lower secondary 11.1%, higher education 5.4%.
- Housing: rented 12.9%, with parents 11.8%.
- Occupation: low-skill laborers 16.2%, accountants 4.6%.
- Organization: construction 10.9%, `XNA` 5.4% (= pensioners, see A3).

This supports keeping all 8 string columns as LightGBM `category` features (E2).

#### A6 Spearman correlation of numeric features
<!-- cell -->
**A6.** Four pairs have |ρ| > 0.8:
- `AMT_CREDIT`–`AMT_GOODS_PRICE` 0.985
- `AMT_CREDIT`–`AMT_ANNUITY` 0.83
- `AMT_ANNUITY`–`AMT_GOODS_PRICE` 0.83
- `CNT_CHILDREN`–`CNT_FAM_MEMBERS` 0.81

The loan amounts are largely redundant with each other. What is unique is their *ratios*, which is why the
ratio features (E4) helped. Collinearity is not a problem for tree models.

#### A7 Train vs test distributions and adversarial validation
<!-- cell -->
**A7.** The train and test distributions of the strongest features overlap almost perfectly. A LightGBM
classifier that tries to tell train rows from test rows reaches an **AUC of about 0.51**, so there is no
meaningful drift. No feature pushes it above the 0.6 warning level; `ORGANIZATION_TYPE` is the largest but
still weak contributor. CV scores should therefore transfer to the leaderboard.

#### A8 Fairness view: default rate by gender, age band and family status
<!-- cell -->
**A8.**
- Gender: men default more often than women (10.1% vs 7.0%).
- Age: the rate falls from 12.1% (< 25) to 5.2% (55+).
- Family status: single and civil-marriage clients 9.9%, married 7.6%, widows 5.8%.

Sensitive attributes are therefore predictive. That makes them a fairness risk, not just a modelling choice.
Analysis experiment **E11** shows that removing `CODE_GENDER` costs **−0.0017 AUC** (0/5 folds). This trade-off
is discussed in §8.

#### EDA summary: finding → hypothesis → experiment

| Finding | Hypothesis | Experiment | Result |
| ------- | ---------- | ---------- | ------ |
| EXT_SOURCE_1/2/3 are the strongest, monotonic predictors (A4); E0 used only EXT_SOURCE_2 | H-M4 | E2 (all raw features) | **KEEP**, +0.0754 |
| Age and employment length are monotonic risk drivers (A4) | H-M3 | E2 | **KEEP** (included in E2) |
| 8 categorical columns with large rate differences (A5) | — | E2 (category dtype) | **KEEP** (included in E2) |
| `DAYS_EMPLOYED = 365243` means pensioner / not employed (A3) | H-M3 | E3 (NaN + flag) | DISCARD, −0.0005 (one split isolates it) |
| Credit/annuity strongly non-monotonic; credit/goods rises at the top (A4); amounts are collinear (A6) | H-M1, H-M2, H-M5 | E4 (credit ratios) | **KEEP**, +0.0078 |
| External scores are complementary (A4) | H-M4 | E5 (aggregates), E12 (weighted mean) | DISCARD, +0.0002 / −0.0016 |
| Life-stage ratios (age vs employment / ID / phone) | H-M3, H-M6 | E6 | DISCARD, +0.0000 |
| Missing bureau enquiries → 10.4% vs 7.7% default (A2) | H-M7 | E7 | DISCARD, +0.0004 (NaN already handled natively) |
| Family size relates to disposable income | — | E8 (per person) | DISCARD, −0.0008 |
| Peer-group position (organization / occupation / education) | — | E9 (group stats), E13 (count encoding) | DISCARD, −0.0005 / −0.0004 |
| `FLAG_MOBIL` constant, zero gain | — | E10 (drop) | **KEEP** (same AUC, simpler) |
| Gender is predictive but sensitive (A8) | fairness | E11 (drop `CODE_GENDER`) | ANALYSIS: −0.0017 AUC |
| No train/test drift (A7, AUC ≈ 0.51) | — | — | CV is trustworthy for the leaderboard |
