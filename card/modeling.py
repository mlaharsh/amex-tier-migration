"""Model training, evaluation, selection, and serialization for card tier migration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from card.config import (
    FEATURES,
    ID_COLUMN,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    RISK_BANDS,
    TEST_SIZE,
    TEST_PREDICTIONS_PATH,
    TIERS,
    TARGET,
    UPSELL_LADDER,
    DOWNSHELL_LADDER,
    TIER_ANNUAL_FEE,
    INTERCHANGE_RATE,
    INTEREST_RATE,
)
from card.data import load_portfolio
from card.features import build_preprocessor, derive_engineered_features, prepare_inference_data


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def compute_annual_margin(row: pd.Series, tier: str) -> float:
    """Compute expected annual margin for a customer at a given tier.

    Margin = (annual fee + interchange + interest) - (rewards cost + service cost + expected loss)
    """

    annual_fee = TIER_ANNUAL_FEE.get(tier, 0.0)
    annual_spend = float(row["annual_spend_usd"])
    statement_balance = float(row["statement_balance_usd"])
    delinquency = float(row["delinquency_rate"])

    # Revenue
    interchange = annual_spend * INTERCHANGE_RATE
    interest = statement_balance * INTEREST_RATE * 0.5  # ~half the balance revolves
    revenue = annual_fee + interchange + interest

    # Costs scale with tier richness
    rewards_cost = float(row["rewards_cost_usd"])
    service_cost = float(row["service_cost_usd"])
    expected_loss = statement_balance * delinquency

    margin = revenue - rewards_cost - service_cost - expected_loss
    return float(margin)


def recommend_tier(
    row: pd.Series,
    churn_probability: float,
    current_tier: str,
    margin_multiplier_for_risk: float = 1.0,
) -> dict[str, Any]:
    """Recommend the optimal card tier for a customer.

    Evaluates margin at each tier, adjusts for churn risk, and returns the
    tier with the highest risk-adjusted margin plus a recommended action.
    """

    tier_margins = {}
    for tier in TIERS:
        base_margin = compute_annual_margin(row, tier)
        # Risk adjustment: higher churn probability reduces the value of premium tiers
        risk_adjustment = 1.0 - (churn_probability * 0.5)
        tier_margins[tier] = base_margin * risk_adjustment

    optimal_tier = max(tier_margins, key=tier_margins.get)
    current_margin = tier_margins.get(current_tier, 0.0)
    optimal_margin = tier_margins[optimal_tier]
    margin_lift = optimal_margin - current_margin

    # Determine recommended action
    if current_tier == optimal_tier:
        if churn_probability >= 0.5:
            action = "Retention offer: waive fee or bonus points to prevent downgrade/churn"
            action_type = "retention"
        else:
            action = "Maintain current tier: already at optimal margin tier"
            action_type = "maintain"
    elif optimal_tier in UPSELL_LADDER.values() and _is_upsell(current_tier, optimal_tier):
        if margin_lift >= 50 and churn_probability < 0.4:
            action = f"Upsell offer: migrate from {current_tier} to {optimal_tier} with targeted bonus"
            action_type = "upsell"
        else:
            action = "Monitor: upside exists but risk or margin lift is marginal"
            action_type = "monitor"
    elif _is_downshell(current_tier, optimal_tier):
        if churn_probability >= 0.5:
            action = "Downsell warning: offer lower-fee tier before customer churns"
            action_type = "downsell"
        else:
            action = "Evaluate: lower tier may improve economics"
            action_type = "evaluate"
    else:
        action = "Maintain current tier"
        action_type = "maintain"

    # Migration priority score (0-100)
    priority = min(
        100,
        max(
            0,
            int(
                (margin_lift / 500) * 40
                + churn_probability * 30
                + (1 - digital_engagement(row)) * 20
                + (float(row["complaint_flag"]) == 1 if "complaint_flag" in row else 0) * 10
            ),
        ),
    )

    return {
        "current_tier": current_tier,
        "optimal_tier": optimal_tier,
        "current_margin": round(current_margin, 2),
        "optimal_margin": round(optimal_margin, 2),
        "expected_margin_lift": round(margin_lift, 2),
        "churn_probability": round(churn_probability, 4),
        "tier_margins": {t: round(m, 2) for t, m in tier_margins.items()},
        "recommended_action": action,
        "action_type": action_type,
        "migration_priority": priority,
    }


def _is_upsell(current: str, target: str) -> bool:
    return TIERS.index(target) > TIERS.index(current)


def _is_downshell(current: str, target: str) -> bool:
    return TIERS.index(target) < TIERS.index(current)


def digital_engagement(row: pd.Series) -> float:
    return float(row.get("digital_engagement_score", 0.5))


def train_migration_model(
    cv_splits: int = 5,
    no_figures: bool = False,
) -> dict[str, Any]:
    """Train the card tier migration model and serialize artifacts."""

    df = load_portfolio()
    y = df[TARGET].astype(int)
    X_df = prepare_inference_data(df)
    # Keep identifier for test predictions output
    X_df[ID_COLUMN] = df[ID_COLUMN].values

    X = X_df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor()
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=None,
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=None,
        return_train_score=False,
    )

    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Test predictions
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    # Threshold tuning: maximize F1 on out-of-fold train predictions
    from sklearn.model_selection import cross_val_predict

    oof_probs = cross_val_predict(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 161):
        preds = (oof_probs >= threshold).astype(int)
        f1 = f1_score(y_train, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    test_preds_at_threshold = (test_probabilities >= best_threshold).astype(int)

    metrics = {
        "threshold": best_threshold,
        "accuracy": float(accuracy_score(y_test, test_preds_at_threshold)),
        "precision": float(precision_score(y_test, test_preds_at_threshold, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds_at_threshold, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds_at_threshold, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "pr_auc": float(average_precision_score(y_test, test_probabilities)),
        "brier": float(brier_score_loss(y_test, test_probabilities)),
        "log_loss": float(log_loss(y_test, test_probabilities)),
        "cv_roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(cv_results["test_roc_auc"].std()),
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_precision_mean": float(cv_results["test_precision"].mean()),
        "cv_recall_mean": float(cv_results["test_recall"].mean()),
        "cv_f1_mean": float(cv_results["test_f1"].mean()),
    }

    # Confusion matrix at threshold
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, test_preds_at_threshold)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    # Feature importance (logistic regression coefficients)
    feature_names = list(
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    coefficients = np.asarray(pipeline.named_steps["classifier"].coef_).reshape(-1)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient", key=abs, ascending=False)
    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()

    # Dataset summary
    dataset_summary = {
        "customers": int(len(df)),
        "churners": int(y.sum()),
        "retained": int((y == 0).sum()),
        "churn_rate": float(y.mean()),
        "avg_tenure_months": float(df["tenure_months"].mean()),
        "avg_annual_spend": float(df["annual_spend_usd"].mean()),
        "avg_annual_fee": float(df["annual_fee_usd"].mean()),
        "tier_distribution": df["card_tier"].value_counts().to_dict(),
    }

    bundle = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "selected_model": "logistic_regression",
        "selected_model_display": "Logistic Regression",
        "features": FEATURES,
        "risk_bands": RISK_BANDS,
        "tiers": TIERS,
        "tier_annual_fee": TIER_ANNUAL_FEE,
        "baselines": {
            col: (
                float(X_df[col].median())
                if X_df[col].dtype in (np.float64, np.int64)
                else str(X_df[col].mode().iloc[0])
            )
            for col in FEATURES
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": __import__("sys").version,
        "metrics": metrics,
        "dataset": dataset_summary,
        "feature_importance": importance_df,
    }

    # Serialization
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metrics_path = MODEL_PATH.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k != "confusion_matrix"
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    importance_path = MODEL_PATH.parent / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    # Save test predictions for threshold replay
    test_predictions_df = pd.DataFrame(
        {
            "customerID": X_test["customerID"] if "customerID" in X_test.columns else range(len(X_test)),
            "churn_probability": test_probabilities,
            "churn_prediction": test_preds_at_threshold,
            "actual_churn": y_test.values,
            "tier": X_test["card_tier"].values if "card_tier" in X_test.columns else ["Unknown"] * len(y_test),
        }
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    print(f"Model trained. Holdout ROC-AUC: {metrics['roc_auc']:.4f}, Threshold: {best_threshold:.3f}")
    print(f"Artifacts saved to {MODEL_PATH}")

    return bundle


def load_model_bundle() -> dict[str, Any]:
    """Load the serialized model bundle."""
    return joblib.load(MODEL_PATH)
