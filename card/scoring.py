"""Scoring logic for the card tier migration model.

This module mirrors the logic implemented in `docs/scorer.js` so the
client-side scorer reproduces the Python pipeline exactly.
"""

from __future__ import annotations

import math
from typing import Any

from card.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RISK_BANDS,
    TIERS,
    UPSELL_LADDER,
    DOWNSHELL_LADDER,
)


def sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-logit)
        return 1 / (1 + z)
    z = math.exp(logit)
    return z / (1 + z)


def derive_features(raw: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Derive engineered features from raw inputs (mirrors JS deriveFeatures)."""

    tenure = float(raw.get("tenure_months", 0) or 0)
    monthly_spend = float(raw.get("monthly_spend_usd", 0) or 0)
    annual_fee = float(raw.get("annual_fee_usd", 0) or 0)
    annual_spend = float(raw.get("annual_spend_usd", 0) or 0)
    statement_balance = float(raw.get("statement_balance_usd", 0) or 0)
    credit_limit = statement_balance / max(raw.get("utilization_pct", 0.5) or 0.5, 0.01)

    spend_to_fee_ratio = annual_spend / annual_fee if annual_fee > 0 else 0.0
    spend_per_tenure = annual_spend / tenure if tenure > 0 else 0.0

    from card.config import INTERCHANGE_RATE, INTEREST_RATE

    revenue = (
        annual_fee
        + annual_spend * INTERCHANGE_RATE
        + statement_balance * INTEREST_RATE * 0.5
    )
    margin_proxy = (
        revenue
        - float(raw.get("rewards_cost_usd", 0) or 0)
        - float(raw.get("service_cost_usd", 0) or 0)
    )

    is_premium_tier = 1 if raw.get("card_tier") in ("Platinum", "Centurion") else 0
    high_value_segment = 1 if annual_spend >= 60000 else 0

    features = {
        "tenure_months": tenure,
        "annual_fee_usd": annual_fee,
        "monthly_spend_usd": monthly_spend,
        "annual_spend_usd": annual_spend,
        "rewards_cost_usd": float(raw.get("rewards_cost_usd", 0) or 0),
        "service_cost_usd": float(raw.get("service_cost_usd", 0) or 0),
        "statement_balance_usd": statement_balance,
        "utilization_pct": float(raw.get("utilization_pct", 0) or 0),
        "delinquency_rate": float(raw.get("delinquency_rate", 0) or 0),
        "credit_score": float(raw.get("credit_score", 700) or 700),
        "income_usd": float(raw.get("income_usd", 70000) or 70000),
        "age": float(raw.get("age", 40) or 40),
        "gender": str(raw.get("gender", "M")),
        "region": str(raw.get("region", "West")),
        "acquisition_channel": str(raw.get("acquisition_channel", "Online")),
        "has_premium_rewards": str(raw.get("has_premium_rewards", "No")),
        "travel_spend_pct": float(raw.get("travel_spend_pct", 0.1) or 0.1),
        "dining_spend_pct": float(raw.get("dining_spend_pct", 0.1) or 0.1),
        "grocery_spend_pct": float(raw.get("grocery_spend_pct", 0.05) or 0.05),
        "digital_engagement_score": float(raw.get("digital_engagement_score", 0.5) or 0.5),
        "support_contacts_last_12m": float(raw.get("support_contacts_last_12m", 0) or 0),
        "complaint_flag": str(raw.get("complaint_flag", "No")),
        "spend_to_fee_ratio": spend_to_fee_ratio,
        "spend_per_tenure": spend_per_tenure,
        "margin_proxy": margin_proxy,
        "is_premium_tier": float(is_premium_tier),
        "high_value_segment": float(high_value_segment),
    }
    return features


def score_features(features: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    """Score a feature vector using exported coefficients (mirrors JS scoreFeatures)."""

    logit = artifact["intercept"]

    numeric_by_name = {f["name"]: f for f in artifact["numeric"]}
    for feature in artifact["numeric"]:
        name = feature["name"]
        value = features.get(name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = feature["median"]
        if not math.isfinite(value):
            value = feature["median"]
        logit += ((value - feature["mean"]) / feature["scale"]) * feature["coefficient"]

    categorical_by_name = {f["name"]: f for f in artifact["categorical"]}
    for feature in artifact["categorical"]:
        name = feature["name"]
        value = str(features.get(name, ""))
        if value in feature["coefficients"]:
            logit += feature["coefficients"][value]
        # Unknown levels are ignored (matches OneHotEncoder handle_unknown="ignore")

    raw_probability = sigmoid(logit)
    probability = round(raw_probability, 4)

    # Risk band
    band = artifact["metadata"]["riskBands"][-1]
    for b in artifact["metadata"]["riskBands"]:
        if b["min"] <= raw_probability < b["max"]:
            band = b
            break

    flagged = raw_probability >= artifact["metadata"]["threshold"]

    # Margin economics per tier
    from card.modeling import compute_annual_margin, recommend_tier

    # Build a pseudo-row for margin calc
    row = {
        "annual_spend_usd": features["annual_spend_usd"],
        "statement_balance_usd": features["statement_balance_usd"],
        "delinquency_rate": features["delinquency_rate"],
    }
    tier_margins = {
        tier: compute_annual_margin(pd.Series(row), tier) for tier in TIERS
    }
    current_tier = features.get("card_tier", "Gold")
    optimal_tier = max(tier_margins, key=tier_margins.get)
    margin_lift = tier_margins[optimal_tier] - tier_margins.get(current_tier, 0)

    recommendation = {
        "current_tier": current_tier,
        "optimal_tier": optimal_tier,
        "current_margin": round(tier_margins.get(current_tier, 0), 2),
        "optimal_margin": round(tier_margins[optimal_tier], 2),
        "expected_margin_lift": round(margin_lift, 2),
        "churn_probability": probability,
        "tier_margins": {t: round(m, 2) for t, m in tier_margins.items()},
    }

    return {
        "logit": logit,
        "rawProbability": raw_probability,
        "probability": probability,
        "riskTier": band["name"],
        "riskColor": band["color"],
        "flagged": flagged,
        "recommendation": recommendation,
    }


def score_raw(raw: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    features = derive_features(raw, artifact)
    return {"features": features, **score_features(features, artifact)}


def threshold_metrics(artifact: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Replay holdout predictions at a given threshold."""

    holdout = artifact["holdout"]
    tp = fp = fn = tn = 0

    for i in range(len(holdout["probabilities"])):
        prob = holdout["probabilities"][i]
        actual = holdout["labels"][i] == "1"
        contacted = prob >= threshold
        if contacted and actual:
            tp += 1
        elif contacted and not actual:
            fp += 1
        elif not contacted and actual:
            fn += 1
        else:
            tn += 1

    contacts = tp + fp
    precision = tp / contacts if contacts > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "threshold": threshold,
        "contacts": contacts,
        "alertRate": contacts / max(1, holdout["n"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def tradeoff_curve(artifact: dict[str, Any], minimum: float, maximum: float, steps: int) -> list:
    return [
        threshold_metrics(artifact, minimum + (maximum - minimum) * i / max(1, steps - 1))
        for i in range(steps)
    ]


import pandas as pd  # noqa: E402
