"""Feature engineering for the card tier migration model."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from card.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the scikit-learn preprocessing pipeline.

    Numeric features: median impute + standard scale.
    Categorical features: one-hot encode with unknown category handling.
    Derived features (spend_to_fee_ratio, spend_per_tenure, margin_proxy,
    is_premium_tier, high_value_segment) are computed upstream in `prepare_inference_data`.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def derive_engineered_features(df) -> dict:
    """Compute engineered features from raw columns.

    Returns a dict keyed by feature name for a single-row DataFrame or Series.
    """

    annual_fee = df["annual_fee_usd"]
    monthly_spend = df["monthly_spend_usd"]
    annual_spend = df["annual_spend_usd"]
    tenure = df["tenure_months"]

    spend_to_fee_ratio = annual_spend / annual_fee.replace(0, np.nan)
    spend_to_fee_ratio = spend_to_fee_ratio.fillna(0).clip(0, 10000)

    spend_per_tenure = annual_spend / tenure.replace(0, np.nan)
    spend_per_tenure = spend_per_tenure.fillna(0).clip(0, 100000)

    # Margin proxy: annual fee + interchange + interest - rewards - service
    from card.config import INTERCHANGE_RATE, INTEREST_RATE, PREMIUM_SERVICE_COST, BASE_SERVICE_COST

    service_cost = df["service_cost_usd"]
    revenue = (
        annual_fee
        + annual_spend * INTERCHANGE_RATE
        + df["statement_balance_usd"] * INTEREST_RATE * 0.5
    )
    margin_proxy = revenue - df["rewards_cost_usd"] - service_cost

    is_premium_tier = df["card_tier"].isin(["Platinum", "Centurion"]).astype(int)
    high_value_segment = (annual_spend >= 60000).astype(int)

    return {
        "spend_to_fee_ratio": spend_to_fee_ratio,
        "spend_per_tenure": spend_per_tenure,
        "margin_proxy": margin_proxy,
        "is_premium_tier": is_premium_tier,
        "high_value_segment": high_value_segment,
    }


def prepare_inference_data(df):
    """Add engineered features to a raw dataframe for model input."""

    import numpy as np

    df = df.copy()
    engineered = derive_engineered_features(df)
    for name, values in engineered.items():
        df[name] = values
    return df[FEATURES]
