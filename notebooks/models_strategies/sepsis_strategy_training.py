from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


ROOT = Path(__file__).resolve().parent
PRIMARY_PATH = ROOT / "s41598-020-73558-3_sepsis_survival_primary_cohort.csv"
STUDY_PATH = ROOT / "s41598-020-73558-3_sepsis_survival_study_cohort.csv"
VALIDATION_PATH = ROOT / "s41598-020-73558-3_sepsis_survival_validation_cohort.csv"

FEATURES = ["age_years", "sex_0male_1female", "episode_number"]
TARGET = "hospital_outcome_1alive_0dead"


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        include_engineered: bool = True,
        include_age_bins: bool = True,
        include_target_encoding: bool = True,
        age_bin_edges: Tuple[float, ...] = (-np.inf, 18, 40, 60, 80, np.inf),
        age_bin_labels: Tuple[str, ...] = ("0_18", "19_40", "41_60", "61_80", "81_plus"),
    ):
        self.include_engineered = include_engineered
        self.include_age_bins = include_age_bins
        self.include_target_encoding = include_target_encoding
        self.age_bin_edges = age_bin_edges
        self.age_bin_labels = age_bin_labels

    def fit(self, X: pd.DataFrame, y: Iterable[int] | None = None):
        X_df = self._ensure_frame(X)
        age_bins = self._make_age_bins(X_df["age_years"])

        self.global_dead_rate_ = float(np.mean(y)) if y is not None else 0.0
        self.group_dead_rate_ = {}
        if y is not None and self.include_target_encoding:
            rate_frame = pd.DataFrame(
                {
                    "age_bin": age_bins,
                    "sex_0male_1female": X_df["sex_0male_1female"].astype(int),
                    "dead": np.asarray(y, dtype=int),
                }
            )
            grouped = rate_frame.groupby(["age_bin", "sex_0male_1female"], observed=True)["dead"].mean()
            self.group_dead_rate_ = grouped.to_dict()

        feature_frame = self._build_feature_frame(X_df)
        self.feature_names_ = feature_frame.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = self._ensure_frame(X)
        feature_frame = self._build_feature_frame(X_df)
        for column in self.feature_names_:
            if column not in feature_frame:
                feature_frame[column] = 0.0
        return feature_frame[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)

    def _ensure_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, FEATURES].copy()
        return pd.DataFrame(X, columns=FEATURES)

    def _make_age_bins(self, age_series: pd.Series) -> pd.Series:
        return pd.cut(
            age_series.astype(float),
            bins=list(self.age_bin_edges),
            labels=list(self.age_bin_labels),
            include_lowest=True,
            right=True,
        )

    def _lookup_dead_rate(self, age_bin: object, sex_value: float) -> float:
        if pd.isna(age_bin):
            return self.global_dead_rate_
        key = (str(age_bin), int(sex_value))
        return float(self.group_dead_rate_.get(key, self.global_dead_rate_))

    def _build_feature_frame(self, X_df: pd.DataFrame) -> pd.DataFrame:
        age = X_df["age_years"].astype(float)
        sex = X_df["sex_0male_1female"].astype(float)
        episode = X_df["episode_number"].astype(float)
        age_bins = self._make_age_bins(age)

        features = pd.DataFrame(index=X_df.index)
        features["age_years"] = age
        features["sex_0male_1female"] = sex
        features["episode_number"] = episode

        if self.include_engineered:
            features["age_squared"] = age.pow(2)
            features["episode_squared"] = episode.pow(2)
            features["age_x_sex"] = age * sex
            features["age_x_episode"] = age * episode
            features["sex_x_episode"] = sex * episode
            features["age_x_sex_x_episode"] = age * sex * episode
            features["is_recurrent"] = (episode > 1).astype(float)

        if self.include_age_bins:
            for label in self.age_bin_labels:
                features[f"age_bin_{label}"] = (age_bins == label).astype(float)

        if self.include_target_encoding:
            features["age_sex_dead_rate"] = [
                self._lookup_dead_rate(age_bin, sex_value)
                for age_bin, sex_value in zip(age_bins.tolist(), sex.tolist())
            ]

        return features


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += abs(confidence - accuracy) * float(mask.mean())
    return ece


def select_mcc_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [matthews_corrcoef(y_true, (y_prob >= threshold).astype(int)) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index])


def positive_proba(estimator, X: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Estimator must expose binary predict_proba outputs.")
    return probabilities[:, 1]


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


def load_cohort(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    X = frame.loc[:, FEATURES].copy()
    y_dead = 1 - frame[TARGET].astype(int).to_numpy()
    return X, y_dead


def build_estimators(seed: int, class_ratio: float, quick: bool) -> Dict[str, object]:
    raw_feature_block = ClinicalFeatureEngineer(
        include_engineered=False,
        include_age_bins=False,
        include_target_encoding=False,
    )
    engineered_feature_block = ClinicalFeatureEngineer()

    logistic_kwargs = {
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": seed,
    }

    rf_estimators = 120 if quick else 240
    rf_depth = 6 if quick else 8

    estimators: Dict[str, object] = {
        "Dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=seed),
        "LR_raw_balanced": Pipeline(
            [
                ("features", raw_feature_block),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**logistic_kwargs)),
            ]
        ),
        "LR_engineered_balanced": Pipeline(
            [
                ("features", engineered_feature_block),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**logistic_kwargs)),
            ]
        ),
        "LR_engineered_calibrated": CalibratedClassifierCV(
            estimator=Pipeline(
                [
                    ("features", engineered_feature_block),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(**logistic_kwargs)),
                ]
            ),
            cv=3,
            method="sigmoid" if quick else "isotonic",
        ),
        "RF_engineered_balanced": Pipeline(
            [
                ("features", engineered_feature_block),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=rf_estimators,
                        max_depth=rf_depth,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGBOOST:
        estimators["XGB_engineered_weighted"] = Pipeline(
            [
                ("features", engineered_feature_block),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        scale_pos_weight=class_ratio,
                        n_estimators=120 if quick else 240,
                        learning_rate=0.05,
                        max_depth=3,
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return estimators


def cross_validate_model(name: str, estimator, X: pd.DataFrame, y: np.ndarray, cv) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray, float]:
    fold_rows = []
    oof_sum = np.zeros(len(X), dtype=float)
    oof_count = np.zeros(len(X), dtype=int)

    for fold_index, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        fitted = clone(estimator)
        fitted.fit(X_train, y_train)

        train_prob = positive_proba(fitted, X_train)
        threshold = select_mcc_threshold(y_train, train_prob)

        val_prob = positive_proba(fitted, X_val)
        fold_metrics = evaluate_probs(y_val, val_prob, threshold)
        fold_metrics.update(
            {
                "model": name,
                "fold": fold_index,
                "val_prevalence": float(y_val.mean()),
            }
        )
        fold_rows.append(fold_metrics)

        oof_sum[val_idx] += val_prob
        oof_count[val_idx] += 1

    oof_prob = oof_sum / np.maximum(oof_count, 1)
    global_threshold = select_mcc_threshold(y, oof_prob)
    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "model": name,
        "auroc_mean": float(fold_df["auroc"].mean()),
        "auroc_std": float(fold_df["auroc"].std(ddof=0)),
        "pr_auc_mean": float(fold_df["pr_auc"].mean()),
        "pr_auc_std": float(fold_df["pr_auc"].std(ddof=0)),
        "mcc_mean": float(fold_df["mcc"].mean()),
        "mcc_std": float(fold_df["mcc"].std(ddof=0)),
        "brier_mean": float(fold_df["brier"].mean()),
        "brier_std": float(fold_df["brier"].std(ddof=0)),
        "ece_mean": float(fold_df["ece"].mean()),
        "ece_std": float(fold_df["ece"].std(ddof=0)),
        "global_threshold": float(global_threshold),
    }
    return fold_df, summary, oof_prob, global_threshold


def evaluate_external_cohort(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float,
    cohort_name: str,
) -> Dict[str, float]:
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    test_prob = positive_proba(fitted, X_test)
    metrics = evaluate_probs(y_test, test_prob, threshold)
    metrics.update(
        {
            "model": name,
            "cohort": cohort_name,
            "prevalence": float(y_test.mean()),
            "n_samples": int(len(y_test)),
        }
    )
    return metrics


def write_markdown_report(
    output_path: Path,
    dataset_summary: Dict[str, object],
    cv_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
    xgboost_available: bool,
) -> None:
    lines = [
        "# Strategy Review and Training Run",
        "",
        "## Paper Review -> Modeling Decisions",
        "",
        "1. Chicco & Jurman (2020): confirms strong cross-cohort degradation, so the run includes primary->study and primary->validation evaluation instead of only internal CV.",
        "2. Carchiolo & Malgeri (2024): their SMOTE-driven accuracy is likely inflated for this low-information setting, so no oversampling is used here.",
        "3. He & Garcia (2009): motivates cost-sensitive learning and minority-aware metrics, so the script uses class weighting, MCC, PR-AUC, Brier Score, and ECE.",
        "4. Niculescu-Mizil & Caruana (2005): motivates post-hoc calibration, so a calibrated logistic model is included alongside the raw logistic baseline.",
        "5. Vovk et al. (2005): motivates uncertainty-aware deployment; conformal prediction is left as the next step after identifying the strongest calibrated model.",
        "6. Cover & Hart (1967): motivates interpreting improvements conservatively because the feature space is likely close to its Bayes-limit ceiling.",
        "",
        "## Data Processing Implemented",
        "",
        "- Target flipped to `dead=1` so minority-class metrics reflect mortality prediction rather than survival prediction.",
        "- Feature engineering includes polynomial terms, pairwise and three-way interactions, recurrent-episode flag, one-hot age bins, and subgroup mortality target encoding by age bin x sex.",
        "- Class imbalance is handled with `class_weight='balanced'`, `class_weight='balanced_subsample'`, and `scale_pos_weight` when XGBoost is available.",
        "- Thresholds are tuned for MCC on training folds rather than fixed at 0.5.",
        "",
        "## Dataset Summary",
        "",
        f"- Primary cohort: {dataset_summary['primary_n']} rows, mortality {dataset_summary['primary_mortality']:.4f}",
        f"- Study cohort: {dataset_summary['study_n']} rows, mortality {dataset_summary['study_mortality']:.4f}",
        f"- Validation cohort: {dataset_summary['validation_n']} rows, mortality {dataset_summary['validation_mortality']:.4f}",
        f"- XGBoost available: {xgboost_available}",
        "",
        "## Cross-Validation Summary",
        "",
        cv_summary.to_markdown(index=False),
        "",
        "## Cross-Cohort Summary",
        "",
        cross_summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- If calibrated logistic improves Brier/ECE without materially improving AUROC or PR-AUC, that supports the paper-driven hypothesis that calibration helps trustworthiness more than discrimination.",
        "- If tree models only provide small gains over logistic, that supports the feature-bottleneck argument in the strategy document.",
        "- If study/validation scores collapse relative to internal CV, that confirms cohort shift remains the main deployment risk.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sepsis survival models following the strategy document.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names to run. Use 'all' to run every configured model.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "strategy_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    X_primary, y_primary = load_cohort(PRIMARY_PATH)
    X_study, y_study = load_cohort(STUDY_PATH)
    X_validation, y_validation = load_cohort(VALIDATION_PATH)

    class_ratio = float((len(y_primary) - y_primary.sum()) / y_primary.sum())
    estimators = build_estimators(seed=args.seed, class_ratio=class_ratio, quick=args.quick)
    if args.models.lower() != "all":
        requested = {name.strip() for name in args.models.split(",") if name.strip()}
        estimators = {name: estimator for name, estimator in estimators.items() if name in requested}
        missing = requested.difference(estimators)
        if missing:
            raise ValueError(f"Unknown model names requested: {sorted(missing)}")

    print(
        f"Running {len(estimators)} model(s) | quick={args.quick} | "
        f"splits={args.n_splits} | repeats={args.n_repeats}",
        flush=True,
    )

    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )

    fold_tables = []
    summary_rows = []
    thresholds = {}

    for model_name, estimator in estimators.items():
        print(f"Starting: {model_name}", flush=True)
        fold_df, summary, _, threshold = cross_validate_model(model_name, estimator, X_primary, y_primary, cv)
        fold_tables.append(fold_df)
        summary_rows.append(summary)
        thresholds[model_name] = threshold
        print(
            f"[{model_name}] CV AUROC={summary['auroc_mean']:.4f} | "
            f"PR-AUC={summary['pr_auc_mean']:.4f} | MCC={summary['mcc_mean']:.4f} | "
            f"Brier={summary['brier_mean']:.4f} | ECE={summary['ece_mean']:.4f}",
            flush=True,
        )

    cv_folds = pd.concat(fold_tables, ignore_index=True)
    cv_summary = pd.DataFrame(summary_rows).sort_values(
        by=["pr_auc_mean", "mcc_mean", "brier_mean"],
        ascending=[False, False, True],
    )

    cross_rows = []
    for model_name, estimator in estimators.items():
        threshold = thresholds[model_name]
        cross_rows.append(
            evaluate_external_cohort(
                model_name,
                estimator,
                X_primary,
                y_primary,
                X_study,
                y_study,
                threshold,
                cohort_name="study",
            )
        )
        cross_rows.append(
            evaluate_external_cohort(
                model_name,
                estimator,
                X_primary,
                y_primary,
                X_validation,
                y_validation,
                threshold,
                cohort_name="validation",
            )
        )

    cross_summary = pd.DataFrame(cross_rows).sort_values(by=["cohort", "pr_auc"], ascending=[True, False])

    dataset_summary = {
        "primary_n": int(len(y_primary)),
        "study_n": int(len(y_study)),
        "validation_n": int(len(y_validation)),
        "primary_mortality": float(y_primary.mean()),
        "study_mortality": float(y_study.mean()),
        "validation_mortality": float(y_validation.mean()),
        "class_ratio": class_ratio,
        "n_splits": int(args.n_splits),
        "n_repeats": int(args.n_repeats),
        "quick": bool(args.quick),
        "xgboost_available": bool(HAS_XGBOOST),
    }

    cv_folds.to_csv(output_dir / "strategy_cv_fold_results.csv", index=False)
    cv_summary.to_csv(output_dir / "strategy_cv_summary.csv", index=False)
    cross_summary.to_csv(output_dir / "strategy_cross_cohort_results.csv", index=False)
    (output_dir / "strategy_run_metadata.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "strategy_run_report.md",
        dataset_summary=dataset_summary,
        cv_summary=cv_summary.round(4),
        cross_summary=cross_summary.round(4),
        xgboost_available=HAS_XGBOOST,
    )

    print("\nSaved outputs:", flush=True)
    print(f"- {output_dir / 'strategy_cv_fold_results.csv'}", flush=True)
    print(f"- {output_dir / 'strategy_cv_summary.csv'}", flush=True)
    print(f"- {output_dir / 'strategy_cross_cohort_results.csv'}", flush=True)
    print(f"- {output_dir / 'strategy_run_report.md'}", flush=True)


if __name__ == "__main__":
    main()