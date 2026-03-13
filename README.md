# Sepsis Survival Minimal Clinical Analysis

## 1. Project Overview
This repository contains the research and implementation for the Sepsis Survival Minimal Clinical Analysis project. Sepsis is a critical condition caused by an overreactive immune response to infection, leading to rapid multiple organ failure and septic shock if not treated promptly. Clinically, every hour of delayed antibiotic treatment reduces the survival probability by 7-8%. This project investigates the mathematical and architectural limits of machine learning models applied to minimal, highly imbalanced clinical records for early sepsis prognosis.

**Authors:** Le Ngoc Anh Thu, Le Duc Minh, Nguyen Quang Minh 
**Institution:** Faculty of DSAI, College of Technology, National Economics University (NEU) 

## 2. Dataset Specifications
The study utilizes the Minimal Clinical Records of Sepsis Survival dataset from the UCI Machine Learning Repository, constructed from hospital cohorts in Norway and South Korea.

- **Feature Space (3D):**
  - `age`: Continuous variable representing age in years.
  - `sex`: Binary indicator where 0 is male and 1 is female.
  - `episode_number`: Discrete count of prior sepsis episodes.
- **Target Variable:**
  - `hospital_outcome_1alive_0dead`: Binary clinical outcome where 1 indicates alive and 0 indicates dead. 
- **Systemic Data Challenges:**
  - **Severe Class Imbalance:** The mortality rate (minority class) is approximately 14.42%.
  - **High Feature Overlap:** Strong density overlap between classes across features (e.g., Hellinger distance overlap).
  - **Distribution Shift:** High vulnerability to cross-cohort degradation due to medical data heterogeneity.

---

## 3. Week 1 Report: Literature Review & Problem Formulation

### 3.1. Critical Analysis of Prior Works
An evaluation of the original paper and external benchmarks reveals significant methodological gaps in the existing literature:
- **Chicco & Jurman (2020):** Highlighted severe distribution shifts where models collapsed on the minority class during cross-cohort validation, with the study cohort ROC-AUC dropping to 0.568.
- **Carchiolo & Malgeri (2024):** Claimed 0.982 accuracy using SMOTE and 10 classifiers. However, this demonstrates a critical "Accuracy Illusion", as evaluating heavily imbalanced medical data strictly via Accuracy and AUC mathematically rewards majority-class memorization.
- **MetaPerceptron & Kaggle Benchmarks:** Approaches utilizing complex architectures (e.g., Swarm-Evolutionary MLPs, XGBoost with Focal Loss) failed to produce clinically viable results, capping at a PR-AUC of roughly 0.14 and F1 scores around 0.55. This proves that architectural capacity is not the bottleneck.

### 3.2. Root Cause Analysis: The Mathematical Limits
We hypothesize that the poor predictive performance stems from fundamental theoretical constraints of the dataset rather than model inadequacy:
- **The Convergence Ceiling:** Complex models simply amplify overfitting noise, as evidenced by the universally low evaluation metrics on the minority class.
- **The 3D Feature Bottleneck:**
  - **Low Mutual Information:** The limited 3-dimensional input space lacks sufficient information entropy to separate the survival classes.
  - **High Bayes Error Rate:** Visualizations and statistical densities confirm that the classes are mathematically inseparable in this specific feature space.
- **Improper Labeling Setup:** Failing to flip labels (i.e., treating `Dead` as the positive minority class) results in distorted evaluations of recall and precision.

### 3.3. Proposed Theoretical Framework & Next Steps
To rigorously address these bottlenecks, the next phases of the project will transition toward theoretical bounding and robust systemic evaluation:

**1. Theoretical Proofs of Data Constraints:**
- **Bayes Error Bound Estimation:** Derive the theoretical lower bound on classification error based on the joint probability distribution of the 3D features.
- **Mutual Information Analysis:** Quantify the maximum achievable predictive power by calculating the mutual information between the features and the mortality target.
- **Sensitivity & Overlap Analysis:** Utilize dimensionality reduction (PCA/t-SNE) and ablation studies to mathematically prove class non-separability.

**2. Methodological Reforms:**
- **Metric Reform:** Discard Accuracy in favor of strictly evaluating MCC, PR-AUC, Expected Calibration Error (ECE), and Brier Score to ensure rigorous uncertainty measurement.
- **Cost-Sensitive Learning:** Implement asymmetric empirical risk minimization where the loss penalty for False Negatives strictly outweighs False Positives.
- **Uncertainty Quantification:** Deploy methods like Conformal Prediction or Monte Carlo Dropout to generate statistically valid prediction intervals, flagging uncertain cases for physician review.
- **Feature Enrichment & Fairness Audit:** Ensure demographic parity across sub-populations and investigate the necessity of integrating external clinical biomarkers to break the current entropy limit.
