"""Export the fitted card tier migration model for the static browser scorer.

Copies the exact fitted preprocessing parameters and logistic-regression
coefficients into a JavaScript data file so the public scorer can reproduce
the Python pipeline client-side without retraining or calling a server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from card.config import (
    FEATURES,
    ROOT,
    TEST_PREDICTIONS_PATH,
    TIER_ANNUAL_FEE,
    TIERS,
    INTERCHANGE_RATE,
    INTEREST_RATE,
    UPSELL_LADDER,
    DOWNSHELL_LADDER,
)
from card.data import load_portfolio
from card.features import prepare_inference_data
from card.modeling import load_model_bundle

BROWSER_DATA_PATH = ROOT / "docs" / "scorer_data.js"
BROWSER_DATA_PREFIX = "window.CARD_MODEL = "
BROWSER_DATA_SUFFIX = ";\n"

# Example customers used as demos and validation cases.
EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "gold-upsell",
        "label": "Gold card candidate for Platinum upsell",
        "raw": {
            "card_tier": "Gold",
            "tenure_months": 38,
            "annual_fee_usd": 250.0,
            "monthly_spend_usd": 4200.0,
            "annual_spend_usd": 50400.0,
            "rewards_cost_usd": 1612.8,
            "service_cost_usd": 95.0,
            "statement_balance_usd": 18000.0,
            "utilization_pct": 0.42,
            "delinquency_rate": 0.015,
            "credit_score": 740,
            "income_usd": 140000,
            "age": 42,
            "gender": "M",
            "region": "West",
            "acquisition_channel": "Online",
            "has_premium_rewards": "Yes",
            "travel_spend_pct": 0.22,
            "dining_spend_pct": 0.15,
            "grocery_spend_pct": 0.08,
            "digital_engagement_score": 0.82,
            "support_contacts_last_12m": 1,
            "complaint_flag": "No",
        },
    },
    {
        "id": "platinum-retention",
        "label": "Platinum customer at churn risk",
        "raw": {
            "card_tier": "Platinum",
            "tenure_months": 72,
            "annual_fee_usd": 695.0,
            "monthly_spend_usd": 7800.0,
            "annual_spend_usd": 93600.0,
            "rewards_cost_usd": 3556.8,
            "service_cost_usd": 210.0,
            "statement_balance_usd": 32000.0,
            "utilization_pct": 0.78,
            "delinquency_rate": 0.06,
            "credit_score": 700,
            "income_usd": 240000,
            "age": 51,
            "gender": "F",
            "region": "Northeast",
            "acquisition_channel": "Branch",
            "has_premium_rewards": "Yes",
            "travel_spend_pct": 0.18,
            "dining_spend_pct": 0.12,
            "grocery_spend_pct": 0.06,
            "digital_engagement_score": 0.45,
            "support_contacts_last_12m": 6,
            "complaint_flag": "Yes",
        },
    },
    {
        "id": "nofee-upsell",
        "label": "No-Fee customer with high spend",
        "raw": {
            "card_tier": "No-Fee",
            "tenure_months": 18,
            "annual_fee_usd": 0.0,
            "monthly_spend_usd": 8500.0,
            "annual_spend_usd": 102000.0,
            "rewards_cost_usd": 1224.0,
            "service_cost_usd": 28.0,
            "statement_balance_usd": 9000.0,
            "utilization_pct": 0.35,
            "delinquency_rate": 0.008,
            "credit_score": 760,
            "income_usd": 180000,
            "age": 36,
            "gender": "Non-binary",
            "region": "Southwest",
            "acquisition_channel": "Referral",
            "has_premium_rewards": "No",
            "travel_spend_pct": 0.25,
            "dining_spend_pct": 0.10,
            "grocery_spend_pct": 0.05,
            "digital_engagement_score": 0.78,
            "support_contacts_last_12m": 0,
            "complaint_flag": "No",
        },
    },
]


def _validate_bundle(bundle: dict[str, Any]) -> None:
    required = {"pipeline", "threshold", "baselines", "risk_bands", "features", "tiers"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {missing}")
    pipeline = bundle["pipeline"]
    try:
        pipeline.named_steps["preprocessor"]
        pipeline.named_steps["classifier"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("Model bundle does not contain expected pipeline steps.") from exc


def build_browser_artifact(
    bundle: dict[str, Any] | None = None,
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable payload embedded in scorer_data.js."""

    bundle = load_model_bundle() if bundle is None else bundle
    _validate_bundle(bundle)
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    coefficients = np.asarray(classifier.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(classifier.intercept_).reshape(-1)[0])
    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Encoded feature count does not match the classifier coefficient count."
        )
    coefficient_map = dict(zip(feature_names, coefficients, strict=True))

    categorical_transformer = preprocessor.named_transformers_["categorical"]
    onehot = categorical_transformer.named_steps["onehot"]
    numeric_transformer = preprocessor.named_transformers_["numeric"]
    imputer = numeric_transformer.named_steps["imputer"]
    scaler = numeric_transformer.named_steps["scaler"]

    from card.config import CATEGORICAL_FEATURES as _CAT_FEATURES

    categorical_features = list(_CAT_FEATURES)
    categorical: list[dict[str, Any]] = []
    for feature, categories in zip(
        categorical_features,
        onehot.categories_,
        strict=True,
    ):
        feature = str(feature)
        categorical.append(
            {
                "name": feature,
                "categories": [str(value) for value in categories],
                "coefficients": {
                    str(category): float(
                        coefficient_map.get(f"{feature}_{category}", 0.0)
                    )
                    for category in categories
                },
            }
        )

    from card.config import NUMERIC_FEATURES as _NUM_FEATURES

    numeric: list[dict[str, Any]] = []
    for name, median, mean, scale in zip(
        _NUM_FEATURES,
        imputer.statistics_,
        scaler.mean_,
        scaler.scale_,
        strict=True,
    ):
        name = str(name)
        numeric.append(
            {
                "name": name,
                "median": float(median),
                "mean": float(mean),
                "scale": float(scale),
                "coefficient": float(coefficient_map.get(name, 0.0)),
            }
        )

    if predictions is None:
        predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    required_prediction_columns = {"churn_probability", "actual_churn"}
    missing_predictions = sorted(required_prediction_columns - set(predictions.columns))
    if missing_predictions:
        raise ValueError(f"Test predictions are missing columns: {missing_predictions}")
    if len(predictions) == 0:
        raise ValueError("Test predictions are empty.")

    cleaned_examples = prepare_inference_data(
        pd.DataFrame(
            [
                {"customerID": f"demo-{index + 1}", **example["raw"]}
                for index, example in enumerate(EXAMPLES)
            ]
        )
    )
    examples: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for index, example in enumerate(EXAMPLES):
        features = {
            feature: cleaned_examples.iloc[index][feature]
            for feature in FEATURES
        }
        cleaned_features = {
            feature: (
                float(features[feature])
                if feature in _NUM_FEATURES
                else str(features[feature])
            )
            for feature in FEATURES
        }
        expected_probability = round(
            float(
                pipeline.predict_proba(
                    pd.DataFrame([cleaned_features], columns=FEATURES)
                )[0, 1]
            ),
            12,
        )
        examples.append(
            {
                "id": example["id"],
                "label": example["label"],
                "raw": example["raw"],
            }
        )
        validation.append(
            {
                "id": example["id"],
                "label": example["label"],
                "features": cleaned_features,
                "expectedProbability": expected_probability,
            }
        )

    holdout = {
        "n": int(len(predictions)),
        "churners": int(predictions["actual_churn"].sum()),
        "threshold": float(bundle["threshold"]),
        "probabilities": [
            round(float(value), 4) for value in predictions["churn_probability"].tolist()
        ],
        "labels": "".join(
            str(int(value)) for value in predictions["actual_churn"].tolist()
        ),
        "tiers": [
            str(value) for value in predictions.get("tier", ["Unknown"] * len(predictions)).tolist()
        ],
    }

    metrics = bundle.get("metrics", {})
    dataset = bundle.get("dataset", {})
    metadata = {
        "title": "Amex Card Tier Migration & Margin Optimization",
        "selectedModel": str(bundle.get("selected_model_display", "Logistic Regression")),
        "selectedModelKey": str(bundle.get("selected_model", "logistic_regression")),
        "threshold": float(bundle["threshold"]),
        "trainedAt": str(bundle.get("trained_at", "")),
        "sklearnVersion": str(bundle.get("sklearn_version", "")),
        "pythonVersion": str(bundle.get("python_version", "")),
        "tiers": [str(t) for t in TIERS],
        "tierAnnualFee": {str(t): float(f) for t, f in TIER_ANNUAL_FEE.items()},
        "interchangeRate": float(INTERCHANGE_RATE),
        "interestRate": float(INTEREST_RATE),
        "upsellLadder": {str(k): str(v) for k, v in UPSELL_LADDER.items()},
        "downshellLadder": {str(k): str(v) for k, v in DOWNSHELL_LADDER.items()},
        "riskBands": [
            {
                "name": str(band["name"]),
                "min": float(band["min"]),
                "max": float(band["max"]),
                "color": str(band["color"]),
            }
            for band in bundle["risk_bands"]
        ],
        "metrics": {
            "rocAuc": float(metrics.get("roc_auc", 0.0)),
            "prAuc": float(metrics.get("pr_auc", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "brier": float(metrics.get("brier", 0.0)),
        },
        "dataset": {
            "customers": int(dataset.get("customers", 0)),
            "churners": int(dataset.get("churners", 0)),
            "churnRate": float(dataset.get("churn_rate", 0.0)),
        },
        "source": "models/card_tier_model.joblib",
        "note": (
            "Coefficients and preprocessing parameters are exported from the fitted "
            "scikit-learn pipeline. Holdout threshold metrics replay the saved "
            "out-of-sample evaluation predictions."
        ),
    }

    return {
        "metadata": metadata,
        "numeric": numeric,
        "categorical": categorical,
        "intercept": intercept,
        "baselines": {
            key: float(value)
            if key in _NUM_FEATURES
            else str(value)
            for key, value in bundle["baselines"].items()
            if key in FEATURES
        },
        "examples": examples,
        "validation": validation,
        "holdout": holdout,
    }


def render_browser_data(artifact: dict[str, Any]) -> str:
    """Serialize the artifact as a classic script (works from file://)."""

    payload = json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False)
    return f"{BROWSER_DATA_PREFIX}{payload}{BROWSER_DATA_SUFFIX}"


def write_browser_data(
    path: Path = BROWSER_DATA_PATH,
    bundle: dict[str, Any] | None = None,
) -> Path:
    """Write the generated browser data file and return its path."""

    artifact = build_browser_artifact(bundle=bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_browser_data(artifact), encoding="utf-8", newline="\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if docs/scorer_data.js is out of date instead of rewriting it.",
    )
    args = parser.parse_args()

    if args.check:
        expected = render_browser_data(build_browser_artifact())
        current = BROWSER_DATA_PATH.read_text(encoding="utf-8")
        if current != expected:
            raise SystemExit(
                "docs/scorer_data.js is out of date. "
                "Run `python -m card.browser_export` and commit the result."
            )
        print(f"Browser artifact is current: {BROWSER_DATA_PATH}")
        return

    path = write_browser_data()
    print(f"Wrote browser scorer artifact: {path}")


if __name__ == "__main__":
    main()
