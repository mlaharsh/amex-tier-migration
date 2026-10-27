"""Synthetic Amex card portfolio data generation and loading.

This module creates a realistic synthetic portfolio of premium credit card
customers with tier, spend, cost, risk, and demographic attributes. It is used
for demonstration and development; in production, replace this with the real
customer data warehouse extract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from card.config import (
    DATA_PATH,
    FEATURES,
    ID_COLUMN,
    RAW_FEATURE_COLUMNS,
    RANDOM_STATE,
    TIERS,
    TIER_ANNUAL_FEE,
)


def generate_portfolio(
    n_customers: int = 25000,
    random_state: int = RANDOM_STATE,
    path: str | None = None,
) -> pd.DataFrame:
    """Generate a synthetic premium card portfolio.

    The data mimics an Amex-style portfolio with:
    - Tier distribution skewed toward mid-premium (Gold/Platinum)
    - Spend and fee economics tied to tier
    - Risk attributes (delinquency, utilization) correlated with tier
    - A target churn flag for model training
    """

    rng = np.random.default_rng(random_state)

    # Tier assignment: realistic mix (few Centurion, many Gold/Platinum)
    tier_probs = np.array([0.08, 0.17, 0.30, 0.35, 0.10])
    tiers = rng.choice(TIERS, size=n_customers, p=tier_probs)

    # Demographics
    age = rng.normal(45, 12, n_customers).clip(22, 85).astype(int)
    income = np.where(
        tiers == "Centurion",
        rng.normal(750000, 200000, n_customers).clip(200000, 3000000),
        np.where(
            tiers == "Platinum",
            rng.normal(220000, 80000, n_customers).clip(90000, 600000),
            np.where(
                tiers == "Gold",
                rng.normal(110000, 40000, n_customers).clip(50000, 250000),
                np.where(
                    tiers == "Silver",
                    rng.normal(65000, 25000, n_customers).clip(30000, 150000),
                    rng.normal(45000, 15000, n_customers).clip(20000, 90000),
                ),
            ),
        ),
    ).round(-2)

    credit_score = np.where(
        tiers == "Centurion",
        rng.normal(790, 30, n_customers).clip(720, 850),
        np.where(
            tiers == "Platinum",
            rng.normal(750, 40, n_customers).clip(680, 820),
            np.where(
                tiers == "Gold",
                rng.normal(710, 50, n_customers).clip(640, 780),
                np.where(
                    tiers == "Silver",
                    rng.normal(670, 60, n_customers).clip(600, 740),
                    rng.normal(630, 70, n_customers).clip(550, 720),
                ),
            ),
        ),
    ).round(0)

    # Tenure correlated with tier premiumness
    tenure_base = {
        "Centurion": 84,
        "Platinum": 60,
        "Gold": 42,
        "Silver": 24,
        "No-Fee": 12,
    }
    tenure = np.array(
        [rng.poisson(tenure_base[t] / 2) + 1 for t in tiers]
    ).clip(1, 180)

    # Monthly spend correlated with income and tier
    spend_multiplier = {
        "Centurion": 0.18,
        "Platinum": 0.14,
        "Gold": 0.10,
        "Silver": 0.07,
        "No-Fee": 0.04,
    }
    monthly_spend = (
        income
        * np.array([spend_multiplier[t] for t in tiers])
        * rng.lognormal(0, 0.4, n_customers)
    ).round(2)
    annual_spend = (monthly_spend * 12).round(2)

    # Annual fee by tier
    annual_fee = np.array([TIER_ANNUAL_FEE[t] for t in tiers], dtype=float)

    # Rewards cost: higher for premium tiers and higher spend
    rewards_rate = {
        "Centurion": 0.045,
        "Platinum": 0.038,
        "Gold": 0.032,
        "Silver": 0.022,
        "No-Fee": 0.012,
    }
    rewards_cost = (
        annual_spend
        * np.array([rewards_rate[t] for t in tiers])
        * rng.lognormal(0, 0.25, n_customers)
    ).round(2)

    # Service cost: premium tiers cost more to serve
    service_cost = np.where(
        tiers == "Centurion",
        rng.normal(400, 100, n_customers).clip(200, 800),
        np.where(
            tiers == "Platinum",
            rng.normal(180, 60, n_customers).clip(80, 400),
            np.where(
                tiers == "Gold",
                rng.normal(80, 30, n_customers).clip(30, 200),
                np.where(
                    tiers == "Silver",
                    rng.normal(40, 15, n_customers).clip(15, 100),
                    rng.normal(20, 10, n_customers).clip(5, 60),
                ),
            ),
        ),
    ).round(2)

    # Statement balance and utilization
    statement_balance = (annual_spend * rng.uniform(0.15, 0.45, n_customers)).round(2)
    credit_limit = (income * rng.uniform(0.15, 0.45, n_customers)).round(2)
    utilization_pct = (statement_balance / credit_limit).clip(0, 1)

    # Delinquency rate: higher for lower tiers and lower credit scores
    delinquency_rate = (
        0.02
        + (850 - credit_score) / 850 * 0.12
        + utilization_pct * 0.08
        - np.array([{"Centurion": 0.01, "Platinum": 0.005, "Gold": 0.0, "Silver": 0.005, "No-Fee": 0.01}[t] for t in tiers])
    ).clip(0, 0.30)

    # Categorical attributes
    gender = rng.choice(["M", "F", "Non-binary"], size=n_customers, p=[0.48, 0.49, 0.03])
    region = rng.choice(
        ["Northeast", "Southeast", "Midwest", "West", "Southwest"],
        size=n_customers,
        p=[0.22, 0.20, 0.18, 0.25, 0.15],
    )
    acquisition_channel = rng.choice(
        ["Online", "Direct Mail", "Branch", "Partner", "Referral"],
        size=n_customers,
        p=[0.35, 0.20, 0.15, 0.15, 0.15],
    )
    has_premium_rewards = rng.choice(
        ["Yes", "No"], size=n_customers, p=[0.65, 0.35]
    )

    # Spend mix
    travel_spend_pct = rng.beta(2, 5, n_customers)
    dining_spend_pct = rng.beta(2, 6, n_customers)
    grocery_spend_pct = rng.beta(3, 5, n_customers)

    # Engagement and support
    digital_engagement_score = rng.beta(3, 2, n_customers)
    support_contacts = rng.poisson(2, n_customers).clip(0, 20)
    complaint_flag = rng.choice(["Yes", "No"], size=n_customers, p=[0.08, 0.92])

    # Churn target: driven by delinquency, low engagement, complaints, fee sensitivity
    churn_logit = (
        -3.0
        + delinquency_rate * 8
        + (1 - digital_engagement_score) * 2
        + (complaint_flag == "Yes") * 1.5
        + (utilization_pct > 0.8) * 1.2
        + (tiers == "No-Fee") * 0.5
        - (tiers == "Centurion") * 1.0
        + rng.normal(0, 0.7, n_customers)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (rng.random(n_customers) < churn_prob).astype(int)

    # Build DataFrame
    df = pd.DataFrame(
        {
            ID_COLUMN: [f"AMEX-{i:07d}" for i in range(n_customers)],
            "card_tier": tiers,
            "tenure_months": tenure,
            "annual_fee_usd": annual_fee,
            "monthly_spend_usd": monthly_spend,
            "annual_spend_usd": annual_spend,
            "rewards_cost_usd": rewards_cost,
            "service_cost_usd": service_cost,
            "statement_balance_usd": statement_balance,
            "utilization_pct": utilization_pct,
            "delinquency_rate": delinquency_rate,
            "credit_score": credit_score,
            "income_usd": income,
            "age": age,
            "gender": gender,
            "region": region,
            "acquisition_channel": acquisition_channel,
            "has_premium_rewards": has_premium_rewards,
            "travel_spend_pct": travel_spend_pct,
            "dining_spend_pct": dining_spend_pct,
            "grocery_spend_pct": grocery_spend_pct,
            "digital_engagement_score": digital_engagement_score,
            "support_contacts_last_12m": support_contacts,
            "complaint_flag": complaint_flag,
            "Churn": churn,
        }
    )

    # Ensure required raw columns exist
    missing = set(RAW_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Generated data missing columns: {missing}")

    if path is None:
        path = DATA_PATH
    path = str(path)
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_portfolio(path: str | None = None) -> pd.DataFrame:
    """Load the card portfolio CSV."""
    import os

    if path is None:
        path = DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Portfolio data not found at {path}. Run `python -m card.train --generate-data` first."
        )
    return pd.read_csv(path)
