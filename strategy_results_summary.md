# Strategy Review and Run Summary

## 1. Paper Review -> Practical Decisions

1. **Chicco & Jurman (2020)**
   - Key takeaway: with only `age`, `sex`, and `episode_number`, internal discrimination is limited and cross-cohort generalization degrades sharply.
   - Decision used here: evaluate not only internal CV on the primary cohort, but also train-on-primary/test-on-study and train-on-primary/test-on-validation.

2. **Carchiolo & Malgeri (2024)**
   - Key takeaway: reported accuracy gains after SMOTE are likely misleading for a low-feature, high-overlap, imbalanced medical dataset.
   - Decision used here: no SMOTE or oversampling. The runs use cost-sensitive learning only.

3. **He & Garcia (2009)**
   - Key takeaway: class-weighting and imbalance-aware evaluation are usually more reliable than naive resampling when feature information is weak.
   - Decision used here: `class_weight='balanced'`, `class_weight='balanced_subsample'`, `scale_pos_weight`, and metrics focused on the death class: PR-AUC, MCC, Brier, and ECE.

4. **Niculescu-Mizil & Caruana (2005)**
   - Key takeaway: calibration quality matters, and uncalibrated cost-sensitive models can become over-confident.
   - Decision used here: compare raw logistic regression against calibrated logistic regression.

5. **Vovk, Gammerman & Shafer (2005)**
   - Key takeaway: uncertainty-aware prediction is more clinically useful than a brittle hard label.
   - Decision used here: not fully implemented yet, but the calibrated logistic model is the correct base candidate for a later conformal layer.

6. **Cover & Hart (1967)**
   - Key takeaway: if the Bayes ceiling is close to the base mortality rate, more complex models will not create large gains.
   - Decision used here: interpret any gain conservatively and focus on whether calibration improves trustworthiness, not whether model complexity inflates headline accuracy.

## 2. Data Processing Actually Implemented

- Flipped the target from `alive=1` to `dead=1`, so minority-class metrics reflect mortality prediction rather than survival prediction.
- Built engineered features from the three raw columns:
  - polynomial terms: `age_squared`, `episode_squared`
  - interactions: `age_x_sex`, `age_x_episode`, `sex_x_episode`, `age_x_sex_x_episode`
  - recurrent flag: `is_recurrent`
  - domain bins: one-hot age bins `0_18`, `19_40`, `41_60`, `61_80`, `81_plus`
  - subgroup target encoding: mortality rate by `age_bin x sex`, learned only on training folds
- Handled imbalance with class weighting and weighted boosting, not oversampling.
- Tuned the classification threshold for MCC instead of fixing it at `0.5`.
- Evaluated on minority-aware metrics: AUROC, PR-AUC, MCC, Brier Score, and ECE.

## 3. Quick Run Results

Configuration used for the completed comparable runs:

- Primary cohort CV: `2-fold`, quick mode
- External validation: train on full primary cohort, test on study and Korea validation cohorts

### 3.1 Internal CV on Primary Cohort

| Model | AUROC | PR-AUC | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Dummy most frequent | 0.5000 | 0.0735 | 0.0000 | 0.0735 | 0.0735 |
| LR raw balanced | 0.7056 | 0.1365 | 0.1575 | 0.2246 | 0.3656 |
| LR engineered balanced | 0.7066 | 0.1377 | 0.1570 | 0.2251 | 0.3636 |
| LR engineered calibrated | 0.7067 | 0.1377 | 0.1565 | 0.0657 | 0.0006 |
| RF engineered balanced | 0.7050 | 0.1362 | 0.1551 | 0.2226 | 0.3583 |
| XGB engineered weighted | 0.7061 | 0.1374 | 0.1558 | 0.2250 | 0.3621 |

### 3.2 Cross-Cohort Test on Study Cohort

| Model | AUROC | PR-AUC | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Dummy most frequent | 0.5000 | 0.1893 | 0.0000 | 0.1893 | 0.1893 |
| LR raw balanced | 0.5872 | 0.2284 | 0.1031 | 0.2794 | 0.3377 |
| LR engineered balanced | 0.5896 | 0.2307 | 0.1014 | 0.2795 | 0.3386 |
| LR engineered calibrated | 0.5894 | 0.2305 | 0.1018 | 0.1598 | 0.0934 |
| RF engineered balanced | 0.5945 | 0.2367 | 0.1134 | 0.2767 | 0.3359 |
| XGB engineered weighted | 0.5931 | 0.2347 | 0.1140 | 0.2780 | 0.3377 |

### 3.3 Cross-Cohort Test on Validation Cohort

| Model | AUROC | PR-AUC | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Dummy most frequent | 0.5000 | 0.1752 | 0.0000 | 0.1752 | 0.1752 |
| LR raw balanced | 0.5496 | 0.2087 | 0.1353 | 0.2059 | 0.2144 |
| LR engineered balanced | 0.5725 | 0.2178 | 0.1372 | 0.2045 | 0.2085 |
| LR engineered calibrated | 0.5721 | 0.2183 | 0.1372 | 0.1579 | 0.1195 |
| RF engineered balanced | 0.5857 | 0.2207 | 0.0748 | 0.2009 | 0.2057 |
| XGB engineered weighted | 0.5813 | 0.2186 | 0.0676 | 0.2020 | 0.2065 |

## 4. What These Results Mean

- The paper-based hypothesis is supported: model discrimination improves only modestly above the trivial baseline and then plateaus.
- Feature engineering helps only a little:
  - raw balanced logistic to engineered balanced logistic gains about `+0.0011` AUROC and `+0.0012` PR-AUC in internal CV.
  - that is directionally positive, but not clinically transformative.
- Calibration is the biggest practical gain:
  - calibrated logistic keeps AUROC and PR-AUC roughly unchanged,
  - but Brier drops from about `0.225` to `0.0657`, and ECE drops from about `0.364` to `0.0006`.
  - that means calibration materially improves probability quality, even though class separation remains weak.
- Stronger tree models do not break the bottleneck:
  - RF and XGB are extremely close to logistic in internal CV.
  - on study cohort they are slightly better in AUROC/PR-AUC/MCC, but the gain is still small.
- Cross-cohort degradation remains real:
  - internal AUROC is about `0.706`, but drops to about `0.589-0.594` on study and `0.572-0.586` on validation.
  - this is consistent with the original paper's warning about cohort shift.

## 5. Bottom Line

These results are **not bad engineering-wise**, but they are **not truly promising as a clinical prediction system** with only 3 demographic variables.

- If the question is "Can better preprocessing and better models squeeze out some signal?" then the answer is **yes, a little**.
- If the question is "Can this feature set support a robust mortality predictor?" then the answer is **still no**.
- The most defensible model from the current runs is **calibrated logistic regression** if you care about probability quality.
- The best discrimination in these quick runs comes from **RF/XGB on the study cohort**, but the gains over logistic are too small to claim a meaningful breakthrough.

## 6. Files Produced

- `sepsis_strategy_training.py`
- `strategy_outputs_lr_quick_2fold/`
- `strategy_outputs_rf_quick/`
- `strategy_outputs_xgb_quick/`
