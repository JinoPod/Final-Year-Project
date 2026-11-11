# Results for the Report

Every number your report needs, in one place. Generated 2026-11-12 from:

- `financial-fraud-detection/models/results/metrics.json`
- `financial-fraud-detection/models/results/stability.json`
- `financial-fraud-detection/models/visualizations/report/report_cases.json`

If you retrain, the source files change and these tables need updating.

The dashboard shows only XGBoost. The baseline models (Logistic Regression, Random Forest, LightGBM) are kept here to justify choosing XGBoost.

---

## 1. Experimental setup (Chapter 3: Methodology)

| Item | Value |
|---|---|
| Dataset | European Credit Card Fraud (ULB / Kaggle), 284,807 transactions, 492 fraud (0.17%) |
| Cleaning | 1,081 exact duplicate rows removed, leaving 283,726 transactions (473 fraud) |
| Features | `Time`, `Amount` (standardised), `V1`–`V28` (PCA components, already anonymised by the dataset publisher) |
| Split | Stratified 80/20, random_state 42 |
| Training set | 226,980 transactions (378 fraud) |
| Test set | 56,746 transactions (95 fraud). Used once, for final evaluation only. |
| Scaling | StandardScaler fitted on the training set only |
| Imbalance handling | SMOTE, applied only to the training folds inside cross-validation, never to validation or test data |
| Tuning | Optuna (TPE sampler, seed 42), 40 trials, 3-fold stratified cross-validation, objective = F1 |
| Decision threshold | The value that maximises F1 on out-of-fold predictions from the training data |
| Explainability | SHAP TreeExplainer (exact TreeSHAP, log-odds output); LIME used for comparison |
| Software | Python 3.11, scikit-learn 1.9.0, XGBoost 3.2.0, LightGBM 4.7.0, SHAP 0.51.0, Optuna 5.0.0 |

**Final XGBoost settings:**

| Setting | Value |
|---|---|
| n_estimators | 250 |
| max_depth | 8 |
| learning_rate | 0.1288 |
| subsample | 0.852 |
| colsample_bytree | 0.582 |
| min_child_weight | 1 |
| gamma | 0.722 |
| SMOTE ratio | 0.1 (fraud oversampled to 10% of the legitimate count) |
| Threshold | 0.8141 |
| Cross-validation F1 | 0.870 |
| Cross-validation ROC-AUC | 0.980 |

All baselines went through the same pipeline: same split, same SMOTE-in-CV, and each had its own threshold tuned the same way. None used class weights, so imbalance is corrected only once, by SMOTE. The baselines used default hyperparameters (Logistic Regression max_iter 1000, Random Forest 200 trees); only XGBoost was tuned. If you mention this, describe it as "a fair, untuned comparison."

---

## 2. Main results on the held-out test set (Chapter 4)

| Model | Role | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **Primary** | 0.814 | **0.960** | **0.758** | **0.847** | **0.978** | 0.803 |
| Random Forest | Baseline | 0.530 | 0.947 | 0.758 | 0.842 | 0.961 | 0.826 |
| LightGBM | Baseline | 0.717 | 0.889 | 0.758 | 0.818 | 0.972 | 0.802 |
| Logistic Regression | Baseline | 0.979 | 0.826 | 0.747 | 0.785 | 0.967 | 0.691 |

Accuracy is 0.999 or higher for every model. **Do not use accuracy to compare models.** A model that labels everything "legitimate" would also score 99.8%. Say this explicitly in the report, because it justifies why you focus on precision, recall and F1.

### Confusion matrices (test set: 56,651 legitimate, 95 fraud)

| Model | Frauds caught (TP) | Missed frauds (FN) | False alarms (FP) | Correctly passed (TN) |
|---|---|---|---|---|
| **XGBoost** | 72 | 23 | **3** | 56,648 |
| Random Forest | 72 | 23 | 4 | 56,647 |
| LightGBM | 72 | 23 | 9 | 56,642 |
| Logistic Regression | 71 | 24 | 15 | 56,636 |

**In plain terms:** XGBoost caught 72 of the 95 frauds. It raised 75 alerts, and only 3 of those were false. That is about 1 false alarm per 19,000 legitimate transactions.

### Effect of threshold tuning (F1 at the default 0.5 vs the tuned threshold)

| Model | F1 at 0.5 | F1 at tuned threshold |
|---|---|---|
| XGBoost | 0.827 | 0.847 |
| Random Forest | 0.842 | 0.842 |
| LightGBM | 0.796 | 0.818 |
| Logistic Regression | 0.547 | 0.785 |

Tuning the threshold matters most for Logistic Regression. For XGBoost it adds about 2 F1 points.

---

## 3. Stability: 5 independent splits (Chapter 4: robustness)

The whole process was repeated on 5 new random splits (seeds 101–105): new 80/20 split, fresh Optuna tuning (20 trials each) and fresh threshold. Values are mean ± standard deviation, with the range in brackets.

| Model | Precision | Recall | F1 | ROC-AUC | False alarms per split | Missed frauds per split |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.922 ± 0.023** [0.884–0.941] | 0.760 ± 0.055 [0.674–0.800] | **0.832 ± 0.031** [0.785–0.859] | 0.970 ± 0.010 | **6.2** [4–10] | 22.8 [19–31] |
| Random Forest | 0.873 ± 0.025 [0.843–0.893] | 0.781 ± 0.036 [0.737–0.832] | 0.824 ± 0.023 [0.787–0.840] | 0.971 ± 0.011 | 10.8 [9–14] | 20.8 [16–25] |
| LightGBM | 0.893 ± 0.056 [0.813–0.971] | 0.758 ± 0.040 [0.695–0.800] | 0.818 ± 0.020 [0.796–0.849] | 0.962 ± 0.010 | 9.0 [2–17] | 23.0 [19–29] |
| Logistic Regression | 0.784 ± 0.038 [0.733–0.827] | 0.724 ± 0.054 [0.663–0.811] | 0.752 ± 0.038 [0.696–0.798] | 0.971 ± 0.008 | 19.0 [14–23] | 26.2 [18–32] |

### Head-to-head F1 (same split, XGBoost vs each baseline)

| Comparison | XGBoost wins | Average F1 advantage |
|---|---|---|
| vs Random Forest | 4 of 5 splits | +0.008 |
| vs LightGBM | 3 of 5 splits | +0.014 |
| vs Logistic Regression | 5 of 5 splits | +0.080 |

### 95% bootstrap confidence intervals on the main test set (1,000 resamples)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | 0.908–1.000 | 0.672–0.843 | 0.788–0.900 | 0.960–0.992 |
| Random Forest | 0.893–0.988 | 0.673–0.842 | 0.777–0.897 | 0.931–0.985 |
| LightGBM | 0.813–0.951 | 0.670–0.839 | 0.753–0.874 | 0.949–0.990 |
| Logistic Regression | 0.744–0.899 | 0.657–0.828 | 0.715–0.842 | 0.944–0.987 |

---

## 4. What the results mean (for the discussion section)

1. **XGBoost is the best overall choice, mainly because of precision.** It has the highest precision on the main test set (0.960) and across splits (0.922 ± 0.023), with the fewest false alarms (6.2 per split vs 10.8 for Random Forest). For an analyst, fewer false alarms means less wasted review time.
2. **Be honest that the F1 lead over Random Forest is small.** It is +0.008 on average, and the confidence intervals overlap heavily. A fair claim is "XGBoost matched or slightly exceeded the strongest baseline while producing fewer false alarms." Avoid saying it "clearly outperformed" Random Forest.
3. **The headline F1 to report is about 0.83 ± 0.03.** That is the average over 5 splits. The 0.847 on the main split is a little above average. Report both and explain the difference.
4. **Recall is the weak point and the most variable metric (0.67–0.80).** With only about 95 frauds per test set, each fraud is worth about 1 percentage point of recall. About 1 in 4 frauds is missed at the chosen threshold.
5. **ROC-AUC barely separates the models (all about 0.96–0.98).** On data this imbalanced, ROC-AUC is dominated by the many easy legitimate transactions. Precision, recall and the precision–recall curve are more informative.
6. **The threshold is sensitive.** It varied from 0.81 to 0.99 across the 5 splits. Put this under limitations. It also suggests a practical feature: let the bank choose the threshold according to how many false alarms it can afford to review.
7. **Logistic Regression is clearly weaker** (F1 about 0.75, about 3× more false alarms than XGBoost). This shows the non-linear tree models capture fraud patterns that a linear model cannot.

---

## 5. Explainability results (SHAP, objective ii)

### Global importance: mean |SHAP| over the whole test set

| Rank | Feature | Mean \|SHAP\| (all transactions) | Mean \|SHAP\| (fraud only) |
|---|---|---|---|
| 1 | V14 | 1.344 | 3.250 |
| 2 | V4 | 1.220 | 0.849 |
| 3 | V12 | 0.610 | 1.384 |
| 4 | V8 | 0.543 | 0.503 |
| 5 | V10 | 0.532 | 1.788 |
| 6 | V17 | 0.502 | 1.548 |
| 7 | V3 | 0.487 | 0.579 |
| 8 | V1 | 0.468 | 0.208 |
| 9 | V11 | 0.452 | 0.387 |
| 10 | V18 | 0.410 | 0.259 |

From the beeswarm plot: **low** values of V14, V12, V10 and V17 push a transaction towards fraud, and **high** values of V4 do the same. Among fraud cases, V14, V10, V17 and V12 dominate. These are the strongest fraud signals.

**Limitation to state:** V1–V28 are PCA components anonymised by the dataset publisher for confidentiality. SHAP shows *which* components drive a decision, but their business meaning (location, merchant type and so on) is not available. A bank deploying this system on its own raw features would get directly meaningful explanations.

### Local explanations: three example cases

All three come from the held-out test set and were chosen by a fixed rule (written in `report_cases.json`) so they are reproducible.

| Case | Figure | Score (threshold 0.814) | Top reasons |
|---|---|---|---|
| Caught fraud (typical) | `waterfall_true_positive.png` | 1.000 | V14 = −6.68, V10 = −3.62, V17 = −8.76, V12 = −5.58, V4 = 6.61 (all raise risk) |
| Worst false alarm | `waterfall_false_positive.png` | 1.000 | V14 = −7.11, V10 = −5.62, V17 = −12.13, V12 = −7.10 (raise risk) |
| Nearest missed fraud | `waterfall_false_negative.png` | 0.795 | V4 = 2.68 (raises), V24 = −0.59 (lowers), V17, V19, V18 (raise) |

**Discussion points:**

- **Worst false alarm.** Its SHAP pattern is almost identical to the typical caught fraud (same features, same direction, even more extreme values). SHAP shows the model behaved consistently, and that this transaction genuinely looks like fraud. It may be an unreported fraud or a labelling error in the dataset. This is a good example of explanations helping an analyst judge an alert, not just accept or reject it.
- **Nearest missed fraud.** It scored 0.795, just under the 0.814 threshold. It was missed because it lacks the strong V14/V12/V10 signals, and V24 pulled the score down. This shows why borderline cases should be flagged for manual review, which is a planned dashboard feature.

---

## 6. Figures available

| Figure | File (under `financial-fraud-detection/models/visualizations/`) | Suggested use |
|---|---|---|
| SHAP global summary (beeswarm) | `report/shap_global_beeswarm.png` | Chapter 4, global explainability |
| SHAP global importance (bar) | `report/shap_global_importance.png` | Chapter 4, global explainability |
| Waterfall: caught fraud | `report/waterfall_true_positive.png` | Chapter 4, local explanation |
| Waterfall: false alarm | `report/waterfall_false_positive.png` | Chapter 4, error analysis |
| Waterfall: missed fraud | `report/waterfall_false_negative.png` | Chapter 4, error analysis |
| XGBoost confusion matrix | `XGBoost_confusion_matrix.png` | Chapter 4, main result |
| Baseline confusion matrices | `Random Forest_…`, `LightGBM_…`, `Logistic Regression_confusion_matrix.png` | Appendix |
| Model comparison bar chart | `model_comparison.png` | Chapter 4, model comparison |
| ROC and precision–recall curves | `roc_pr_curves.png` | Chapter 4 (the precision–recall panel is the more useful one) |
| Stability box plot | `stability_boxplot.png` | Chapter 4, robustness |

---

## 7. Changes from the interim report to mention

- **Duplicates removed.** 1,081 duplicate rows were removed before splitting, so the same transaction cannot appear in both training and test data.
- **SMOTE moved inside cross-validation.** It now runs only on training folds, avoiding optimistic validation scores.
- **Threshold tuned on training data.** The decision threshold is now chosen on out-of-fold training predictions instead of the default 0.5, and the test set is used once.
- **Stability check added.** It shows the results hold across different splits.
- **Real-time streaming remains out of scope**, as in the interim report (Scope vii). It goes under future work.

---

## 8. Explanation layer for analysts (design contribution)

The dashboard translates the model's output into language an analyst can act on. The code is in `financial-fraud-detection/src/explanations.py`.

**Risk levels** replace the raw yes/no decision:

| Level | Rule | Suggested action |
|---|---|---|
| 🔴 High risk | score ≥ alert threshold (0.814) | Hold the transaction and contact the cardholder |
| 🟡 Needs review | 0.5 ≤ score < 0.814 | Review manually before approving |
| 🟢 Low risk | score < 0.5 | No action |

- **Why 0.5 marks the review band.** 0.5 is the standard default decision point of a probabilistic classifier. It was fixed in advance, not tuned on the test set.
- **Test-set result.** The review band contains 9 transactions, 2 of them fraud. In other words, an analyst checking 7 extra legitimate transactions recovers 2 of the 23 missed frauds.
- **Limitation.** The other 21 missed frauds score far lower (19 score below 0.1). The model sees them as normal, so a review band cannot rescue them.
- **Borderline warning.** Scores within ±0.05 of the alert threshold carry a note that the decision is close to the line.

**Plain-language reasons.** Each of the top 5 SHAP factors becomes a sentence such as:

> **V14 = −6.68** is extremely low: lower than 99.9% of normal transactions (normal range −1.42 to 1.39; frauds typically have about −6.75). This strongly raises the fraud score.

How each part of the sentence is produced:

- **The comparisons** use the training set only: percentiles of the 226,602 legitimate training transactions, and the median of the 378 training frauds.
- **The strength words** come from |SHAP| in log-odds: ≥ 2 is "strongly", ≥ 0.5 is "moderately", otherwise "slightly".
- **The reasons are phrased as comparisons with normal transactions** because V1–V28 are anonymised, so business meanings are not available.

**Other aids:**

- a simple red/blue bar chart of the top reasons, with the full SHAP waterfall available on request;
- a "How to read this chart" note on every chart;
- a plain-terms summary of model performance, e.g. "96 out of every 100 alerts are real fraud" and "catches about 3 in 4 frauds";
- an integrity-check message in the Audit Log.

**Suggested evaluation for the report.** Ask 3–5 people (ideally non-technical) to explain why a transaction was flagged:

1. once using only the SHAP waterfall;
2. once using the new verdict and reasons view.

Record how long they take and whether their explanation is correct, plus a short rating of clarity from 1 to 5. Even a small test like this gives evidence that the explanation layer improves interpretability.
