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

| model                    |   auroc_mean |   auroc_std |   pr_auc_mean |   pr_auc_std |   mcc_mean |   mcc_std |   brier_mean |   brier_std |   ece_mean |   ece_std |   global_threshold |
|:-------------------------|-------------:|------------:|--------------:|-------------:|-----------:|----------:|-------------:|------------:|-----------:|----------:|-------------------:|
| LR_engineered_calibrated |       0.7068 |      0.004  |        0.1381 |       0.0019 |     0.1572 |    0.0046 |       0.0657 |      0.0001 |     0.0013 |    0.0006 |              0.085 |
| LR_engineered_balanced   |       0.7068 |      0.0041 |        0.138  |       0.0019 |     0.1573 |    0.0051 |       0.2253 |      0.0005 |     0.3639 |    0.0002 |              0.515 |
| LR_raw_balanced          |       0.7055 |      0.0044 |        0.1368 |       0.0023 |     0.1569 |    0.0055 |       0.2246 |      0.0006 |     0.3656 |    0.0004 |              0.48  |
| Dummy_most_frequent      |       0.5    |      0      |        0.0735 |       0      |     0      |    0      |       0.0735 |      0      |     0.0735 |    0      |              0.05  |

## Cross-Cohort Summary

|   auroc |   pr_auc |    mcc |   brier |    ece |   threshold | model                    | cohort     |   prevalence |   n_samples |
|--------:|---------:|-------:|--------:|-------:|------------:|:-------------------------|:-----------|-------------:|------------:|
|  0.5896 |   0.2307 | 0.1046 |  0.2795 | 0.3386 |       0.515 | LR_engineered_balanced   | study      |       0.1893 |       19051 |
|  0.5894 |   0.2305 | 0.1018 |  0.1598 | 0.0934 |       0.085 | LR_engineered_calibrated | study      |       0.1893 |       19051 |
|  0.5872 |   0.2284 | 0.1069 |  0.2794 | 0.3377 |       0.48  | LR_raw_balanced          | study      |       0.1893 |       19051 |
|  0.5    |   0.1893 | 0      |  0.1893 | 0.1893 |       0.05  | Dummy_most_frequent      | study      |       0.1893 |       19051 |
|  0.5721 |   0.2183 | 0.1372 |  0.1579 | 0.1195 |       0.085 | LR_engineered_calibrated | validation |       0.1752 |         137 |
|  0.5725 |   0.2178 | 0.1353 |  0.2045 | 0.2085 |       0.515 | LR_engineered_balanced   | validation |       0.1752 |         137 |
|  0.5496 |   0.2087 | 0.11   |  0.2059 | 0.2144 |       0.48  | LR_raw_balanced          | validation |       0.1752 |         137 |
|  0.5    |   0.1752 | 0      |  0.1752 | 0.1752 |       0.05  | Dummy_most_frequent      | validation |       0.1752 |         137 |

## Interpretation

- If calibrated logistic improves Brier/ECE without materially improving AUROC or PR-AUC, that supports the paper-driven hypothesis that calibration helps trustworthiness more than discrimination.
- If tree models only provide small gains over logistic, that supports the feature-bottleneck argument in the strategy document.
- If study/validation scores collapse relative to internal CV, that confirms cohort shift remains the main deployment risk.