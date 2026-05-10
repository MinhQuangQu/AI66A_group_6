# Strategy Review and Training Run

## Paper Review -> Modeling Decisions

1. Chicco & Jurman (2020): confirms strong cross-cohort degradation, so the run includes primary->study and primary->validation evaluation instead of only internal CV.
2. Carchiolo & Malgeri (2024): their SMOTE-driven accuracy is likely inflated for this low-information setting, so no oversampling is used here.
3. He & Garcia (2009): motivates cost-sensitive learning and minority-aware metrics, so the script uses class weighting, MCC, PR-AUC, Brier Score, and ECE.
4. Niculescu-Mizil & Caruana (2005): motivates post-hoc calibration, so a calibrated logistic model is included alongside the raw logistic baseline.
5. Vovk et al. (2005): motivates uncertainty-aware deployment; conformal prediction is left as the next step after identifying the strongest calibrated model.
6. Cover & Hart (1967): motivates interpreting improvements conservatively because the feature space is likely close to its Bayes-limit ceiling.

## Data Processing Implemented

- Target flipped to `dead=1` so minority-class metrics reflect mortality prediction rather than survival prediction.
- Feature engineering includes polynomial terms, pairwise and three-way interactions, recurrent-episode flag, one-hot age bins, and subgroup mortality target encoding by age bin x sex.
- Class imbalance is handled with `class_weight='balanced'`, `class_weight='balanced_subsample'`, and `scale_pos_weight` when XGBoost is available.
- Thresholds are tuned for MCC on training folds rather than fixed at 0.5.

## Dataset Summary

- Primary cohort: 110204 rows, mortality 0.0735
- Study cohort: 19051 rows, mortality 0.1893
- Validation cohort: 137 rows, mortality 0.1752
- XGBoost available: True

## Cross-Validation Summary

| model                  |   auroc_mean |   auroc_std |   pr_auc_mean |   pr_auc_std |   mcc_mean |   mcc_std |   brier_mean |   brier_std |   ece_mean |   ece_std |   global_threshold |
|:-----------------------|-------------:|------------:|--------------:|-------------:|-----------:|----------:|-------------:|------------:|-----------:|----------:|-------------------:|
| RF_engineered_balanced |        0.705 |       0.001 |        0.1362 |        0.002 |     0.1551 |    0.0001 |       0.2226 |      0.0004 |     0.3583 |    0.0002 |               0.45 |

## Cross-Cohort Summary

|   auroc |   pr_auc |    mcc |   brier |    ece |   threshold | model                  | cohort     |   prevalence |   n_samples |
|--------:|---------:|-------:|--------:|-------:|------------:|:-----------------------|:-----------|-------------:|------------:|
|  0.5945 |   0.2367 | 0.1134 |  0.2767 | 0.3359 |        0.45 | RF_engineered_balanced | study      |       0.1893 |       19051 |
|  0.5857 |   0.2207 | 0.0748 |  0.2009 | 0.2057 |        0.45 | RF_engineered_balanced | validation |       0.1752 |         137 |

## Interpretation

- If calibrated logistic improves Brier/ECE without materially improving AUROC or PR-AUC, that supports the paper-driven hypothesis that calibration helps trustworthiness more than discrimination.
- If tree models only provide small gains over logistic, that supports the feature-bottleneck argument in the strategy document.
- If study/validation scores collapse relative to internal CV, that confirms cohort shift remains the main deployment risk.