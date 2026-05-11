# Sepsis Survival Minimal Clinical Analysis

This repository studies sepsis mortality prediction under an intentionally hard setting: only three demographic / administrative variables are available (`age`, `sex`, `episode_number`). The project has evolved from a week-1 EDA baseline into a week-2 analysis of theoretical limits, calibration, and cross-cohort robustness.

The current conclusion is not that a new model has solved the problem. It is the opposite: the experiments show that the main bottleneck is the information content of the feature space, while calibration is the most meaningful practical improvement.

**Authors:** Le Ngoc Anh Thu, Le Duc Minh, Nguyen Quang Minh  
**Institution:** Faculty of DSAI, College of Technology, National Economics University (NEU)

## 1. Repository Layout

- `data/raw/`: original primary, study, and Korean validation CSV cohorts.
- `data/result/`: week-1 baseline artifacts such as `baseline_results.csv`, `experiment_log_week1.json`, and `oof_probabilities_lr.csv`.
- `docs/`: narrative summaries and reports, including:
  - `data_processing_strategy.md`
  - `strategy_results_summary.md`
  - `week_2_analysis_report.md`
  - `khung_report.md`
- `notebooks/EDA/`: exploratory notebooks for week 1 and week 2 analysis.
- `notebooks/baseline_model/`: baseline modeling notebook.
- `notebooks/feature_engineering/`: engineered-feature notebook.
- `notebooks/models_strategies/`: `sepsis_strategy_training.py` and archived strategy outputs.
- `requirements.txt`: Python dependencies used in the project.

## 2. Dataset and Evaluation Setup

### Dataset

- Source: UCI Minimal Clinical Records of Sepsis Survival.
- Raw target in the CSV files: `hospital_outcome_1alive_0dead`.
- Core features:
  - `age_years`
  - `sex_0male_1female`
  - `episode_number`

### Cohorts

| Cohort | Samples | Mortality Rate | Notes |
|---|---:|---:|---|
| Primary | 110,204 | 0.0735 | Main training cohort |
| Study | 19,051 | 0.1893 | External test cohort |
| Validation (Korea) | 137 | 0.1752 | External validation cohort |

### Modeling Decisions Actually Used

- In the strategy runs, the target is flipped to `dead=1` so PR-AUC and MCC reflect mortality prediction rather than the majority survival class.
- Imbalance is handled with class weighting and weighted boosting, not SMOTE.
- Thresholds are tuned on training folds for MCC instead of being fixed at `0.5`.
- Main evaluation metrics are AUROC, PR-AUC, MCC, Brier Score, and Expected Calibration Error (ECE).

## 3. Experiments Completed

### 3.1 Week 1: EDA and Naive Baseline

Completed work:

- Descriptive statistics by outcome.
- Class imbalance analysis.
- Correlation analysis.
- Outlier inspection.
- Majority-class dummy baseline.
- Stratified CV setup and out-of-fold logistic probabilities.

Important result:

- The dummy classifier reaches **92.65% accuracy** simply by predicting the majority class, but **MCC = 0.0000** and **AUROC = 0.5000**.
- This is the core **accuracy illusion** of the dataset.

Saved outputs:

- `data/result/baseline_results.csv`
- `data/result/oof_probabilities_lr.csv`
- `data/result/experiment_log_week1.json`

### 3.2 Week 2: Statistical Signal and Theoretical Limit Analysis

Completed work:

- Chi-square test for `sex` vs outcome.
- Kruskal-Wallis tests for `age` and `episode`.
- Effect-size analysis with Cohen's d.
- Empirical Bayes error estimation from 1-NN.
- Reliability / miscalibration analysis from logistic OOF probabilities.

Key findings:

| Quantity | Value | Interpretation |
|---|---:|---|
| Chi-square (`sex` vs outcome) | 43.0373 | Statistically significant, but weak alone |
| Kruskal-Wallis (`age`) | 3766.4964 | Age carries the strongest usable signal |
| Kruskal-Wallis (`episode`) | 18.4735 | Episode has signal, but much weaker than age |
| Cohen's d for `age` | 0.6612 | Moderate-to-large effect size |
| 1-NN empirical error | 12.27% | Used to estimate the theoretical lower bound |
| Bayes Error Bound | ~6.57% | Very close to the mortality baseline |
| Baseline mortality rate | 7.35% | Shows how narrow the achievable margin really is |

Interpretation:

- `age` is the strongest single predictor among the three raw variables.
- The estimated Bayes error bound is so close to the baseline mortality rate that the problem is constrained by a **feature bottleneck**, not by a lack of model complexity.

### 3.3 Strategy Runs: Feature Engineering, Calibration, and Cross-Cohort Testing

Implemented in `notebooks/models_strategies/sepsis_strategy_training.py`:

- Raw logistic regression with class balancing.
- Engineered logistic regression.
- Post-hoc calibrated logistic regression.
- Random Forest with balanced subsampling.
- XGBoost with `scale_pos_weight`.

Feature engineering actually used:

- Polynomial terms: `age_squared`, `episode_squared`
- Interaction terms: `age_x_sex`, `age_x_episode`, `sex_x_episode`, `age_x_sex_x_episode`
- Recurrent episode flag: `is_recurrent`
- One-hot age bins
- Fold-safe subgroup target encoding by `age_bin x sex`

Evaluation design:

- Internal CV on the primary cohort.
- Train on full primary cohort, then test on the study cohort.
- Train on full primary cohort, then test on the Korean validation cohort.

## 4. Main Results

### 4.1 Internal CV on the Primary Cohort

| Model | AUROC | PR-AUC | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Dummy most frequent | 0.5000 | 0.0735 | 0.0000 | 0.0735 | 0.0735 |
| LR raw balanced | 0.7056 | 0.1365 | 0.1575 | 0.2246 | 0.3656 |
| LR engineered balanced | 0.7066 | 0.1377 | 0.1570 | 0.2251 | 0.3636 |
| LR engineered calibrated | 0.7067 | 0.1377 | 0.1565 | 0.0657 | 0.0006 |
| RF engineered balanced | 0.7050 | 0.1362 | 0.1551 | 0.2226 | 0.3583 |
| XGB engineered weighted | 0.7061 | 0.1374 | 0.1558 | 0.2250 | 0.3621 |

### 4.2 External Test on the Study Cohort

| Model | AUROC | PR-AUC | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| LR engineered calibrated | 0.5894 | 0.2305 | 0.1018 | 0.1598 | 0.0934 |
| RF engineered balanced | 0.5945 | 0.2367 | 0.1134 | 0.2767 | 0.3359 |
| XGB engineered weighted | 0.5931 | 0.2347 | 0.1140 | 0.2780 | 0.3377 |

### 4.3 External Test on the Korean Validation Cohort

| Model | AUROC | PR-AUC | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| LR engineered calibrated | 0.5721 | 0.2183 | 0.1372 | 0.1579 | 0.1195 |
| RF engineered balanced | 0.5857 | 0.2207 | 0.0748 | 0.2009 | 0.2057 |
| XGB engineered weighted | 0.5813 | 0.2186 | 0.0676 | 0.2020 | 0.2065 |

## 5. What These Results Mean

### 5.1 Accuracy Is a Misleading Success Metric Here

- The dummy model gets 92.65% accuracy while having zero useful discrimination.
- For this dataset, headline accuracy mainly tracks class imbalance, not predictive value.

### 5.2 The Feature Bottleneck Is Real

- Theoretical analysis puts the Bayes floor around **6.57%**, very near the **7.35%** mortality baseline.
- That gap is too small to expect clinically meaningful gains from algorithmic complexity alone.

### 5.3 Calibration Is the Biggest Practical Improvement

- Calibrated logistic regression keeps AUROC and PR-AUC almost unchanged.
- But it improves probability quality dramatically:
  - **Brier: 0.2246 -> 0.0657** on primary CV
  - **ECE: 0.3636 -> 0.0006** on primary CV
- This means the main gain is not better ranking, but more trustworthy risk probabilities.

### 5.4 More Complex Models Do Not Break the Ceiling

- On primary CV, XGB (`0.7061`) is essentially tied with calibrated logistic (`0.7067`) in AUROC.
- On the study cohort, RF and XGB are only marginally better in discrimination than calibrated logistic.
- Those gains are too small to claim a meaningful modeling breakthrough.

### 5.5 Cross-Cohort Degradation Remains the Main Deployment Risk

- Internal AUROC is around `0.706`, but drops to roughly `0.589-0.594` on the study cohort.
- It drops again to roughly `0.572-0.586` on the Korean validation cohort.
- Calibration also degrades under shift, even though calibrated logistic remains much better behaved than the raw balanced models.

## 6. Current Bottom Line

- The current experiments support a clear conclusion: **the limit is in the information, not the algorithm**.
- If the objective is a clinically trustworthy probability estimate from this minimal feature set, the most defensible current model is **calibrated logistic regression**.
- If the objective is a robust mortality classifier with materially stronger discrimination, the project will need **richer clinical variables**, not just more complex models.

## 7. Recommended Next Steps

- Add physiologic and laboratory features to test whether the feature bottleneck can be broken.
- Explore domain-adaptive or cohort-specific recalibration for external deployment.
- Add conformal prediction or another uncertainty layer on top of calibrated logistic regression.
