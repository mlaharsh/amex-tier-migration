"""Configuration for Amex Premium Card Tier Migration & Margin Optimization."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = ROOT / "data" / "raw" / "card_portfolio.csv"
MODEL_PATH = ROOT / "models" / "card_tier_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"
TEST_PREDICTIONS_PATH = ROOT / "reports" / "evaluation" / "test_predictions.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "reports" / "figures"

# Card tiers (ordered from entry to premium)
TIERS = ["No-Fee", "Silver", "Gold", "Platinum", "Centurion"]
TIER_DISPLAY = {
    "No-Fee": "No-Fee Card",
    "Silver": "Silver Card",
    "Gold": "Gold Card",
    "Platinum": "Platinum Card",
    "Centurion": "Centurion Card",
}

# Raw feature columns (the schema the scorer/batch UI accepts)
RAW_FEATURE_COLUMNS = [
    "customerID",
    "card_tier",
    "tenure_months",
    "annual_fee_usd",
    "monthly_spend_usd",
    "annual_spend_usd",
    "rewards_cost_usd",
    "service_cost_usd",
    "statement_balance_usd",
    "utilization_pct",
    "delinquency_rate",
    "credit_score",
    "income_usd",
    "age",
    "gender",
    "region",
    "acquisition_channel",
    "has_premium_rewards",
    "travel_spend_pct",
    "dining_spend_pct",
    "grocery_spend_pct",
    "digital_engagement_score",
    "support_contacts_last_12m",
    "complaint_flag",
    "Churn",
]

# Engineered feature columns used by the model
FEATURES = [
    "card_tier",
    "tenure_months",
    "annual_fee_usd",
    "monthly_spend_usd",
    "annual_spend_usd",
    "rewards_cost_usd",
    "service_cost_usd",
    "statement_balance_usd",
    "utilization_pct",
    "delinquency_rate",
    "credit_score",
    "income_usd",
    "age",
    "gender",
    "region",
    "acquisition_channel",
    "has_premium_rewards",
    "travel_spend_pct",
    "dining_spend_pct",
    "grocery_spend_pct",
    "digital_engagement_score",
    "support_contacts_last_12m",
    "complaint_flag",
    "spend_to_fee_ratio",
    "spend_per_tenure",
    "margin_proxy",
    "is_premium_tier",
    "high_value_segment",
]

# Categorical / numeric split for preprocessing
CATEGORICAL_FEATURES = [
    "card_tier",
    "gender",
    "region",
    "acquisition_channel",
    "has_premium_rewards",
    "complaint_flag",
]
NUMERIC_FEATURES = [
    "tenure_months",
    "annual_fee_usd",
    "monthly_spend_usd",
    "annual_spend_usd",
    "rewards_cost_usd",
    "service_cost_usd",
    "statement_balance_usd",
    "utilization_pct",
    "delinquency_rate",
    "credit_score",
    "income_usd",
    "age",
    "travel_spend_pct",
    "dining_spend_pct",
    "grocery_spend_pct",
    "spend_to_fee_ratio",
    "spend_per_tenure",
    "margin_proxy",
    "digital_engagement_score",
    "support_contacts_last_12m",
    "is_premium_tier",
    "high_value_segment",
]

TARGET = "Churn"
ID_COLUMN = "customerID"

# Margin economics (annual, per customer)
# Revenue components: annual fee + interchange (~1.8% of annual spend) + interest (~12% on revolving balance)
# Cost components: rewards cost + service cost + expected loss (delinquency * balance)
INTERCHANGE_RATE = 0.018
INTEREST_RATE = 0.12
BASE_SERVICE_COST = 25.0
PREMIUM_SERVICE_COST = 150.0

# Risk thresholds
DELINQUENCY_HIGH = 0.05
UTILIZATION_HIGH = 0.70

# Migration policy thresholds
UPSELL_PROPENSITY_THRESHOLD = 0.60
DOWNSHELL_RISK_THRESHOLD = 0.30
MARGIN_LIFT_MINIMUM = 50.0  # minimum annual margin lift to recommend migration

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Risk bands for migration propensity
RISK_BANDS = [
    {"name": "Low", "min": 0.0, "max": 0.30, "color": "#2A9D8F"},
    {"name": "Medium", "min": 0.30, "max": 0.55, "color": "#E9C46A"},
    {"name": "High", "min": 0.55, "max": 0.75, "color": "#F4A261"},
    {"name": "Critical", "min": 0.75, "max": 1.01, "color": "#E76F51"},
]

# Palette for charts
PALETTE = {
    "ink": "#0B1F33",
    "ink-soft": "#14384B",
    "teal": "#0E7C86",
    "teal-deep": "#0A6570",
    "mint": "#2A9D8F",
    "mint-light": "#8ED7D2",
    "coral": "#E76F51",
    "amber": "#E9C46A",
    "orange": "#F4A261",
    "cream": "#F7F5F0",
    "cream-deep": "#EFEBE2",
    "paper": "#FFFFFF",
    "muted": "#61717B",
    "muted-dark": "#4B5E69",
    "line": "#E1E6E9",
    "line-strong": "#D2DADE",
}

FEATURE_LABELS = {
    "tenure_months": "Tenure (months)",
    "annual_fee_usd": "Annual Fee (USD)",
    "monthly_spend_usd": "Monthly Spend (USD)",
    "annual_spend_usd": "Annual Spend (USD)",
    "rewards_cost_usd": "Rewards Cost (USD)",
    "service_cost_usd": "Service Cost (USD)",
    "statement_balance_usd": "Statement Balance (USD)",
    "utilization_pct": "Utilization (%)",
    "delinquency_rate": "Delinquency Rate",
    "credit_score": "Credit Score",
    "income_usd": "Income (USD)",
    "age": "Age",
    "gender": "Gender",
    "region": "Region",
    "acquisition_channel": "Acquisition Channel",
    "has_premium_rewards": "Has Premium Rewards",
    "travel_spend_pct": "Travel Spend (%)",
    "dining_spend_pct": "Dining Spend (%)",
    "grocery_spend_pct": "Grocery Spend (%)",
    "digital_engagement_score": "Digital Engagement",
    "support_contacts_last_12m": "Support Contacts (12m)",
    "complaint_flag": "Complaint Flag",
    "spend_to_fee_ratio": "Spend-to-Fee Ratio",
    "spend_per_tenure": "Spend per Tenure",
    "margin_proxy": "Margin Proxy",
    "is_premium_tier": "Is Premium Tier",
    "high_value_segment": "High-Value Segment",
    "card_tier": "Card Tier",
}

# Migration tier ladder (which tier to upsell to from current)
UPSELL_LADDER = {
    "No-Fee": "Silver",
    "Silver": "Gold",
    "Gold": "Platinum",
    "Platinum": "Centurion",
    "Centurion": "Centurion",  # already at top
}

DOWNSHELL_LADDER = {
    "Centurion": "Platinum",
    "Platinum": "Gold",
    "Gold": "Silver",
    "Silver": "No-Fee",
    "No-Fee": "No-Fee",
}

# Annual fees by tier
TIER_ANNUAL_FEE = {
    "No-Fee": 0.0,
    "Silver": 95.0,
    "Gold": 250.0,
    "Platinum": 695.0,
    "Centurion": 5000.0,
}
