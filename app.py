"""Amex Premium Card Tier Migration & Margin Optimization — Streamlit workbench.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from card.config import (
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    ID_COLUMN,
    METRICS_PATH,
    NUMERIC_FEATURES,
    PALETTE,
    RISK_BANDS,
    RAW_FEATURE_COLUMNS,
    TIERS,
    TIER_ANNUAL_FEE,
    UPSELL_LADDER,
    TARGET,
)
from card.data import DataValidationError, load_portfolio
from card.modeling import load_model_bundle, recommend_tier, train_migration_model
from card.scoring import (
    derive_engineered_features,
    prepare_inference_data,
    score_features,
    score_raw,
    threshold_metrics,
    tradeoff_curve,
)

ROOT = Path(__file__).resolve().parent

INK = PALETTE["ink"]
MUTED = PALETTE["muted"]
TEAL = PALETTE["teal"]
MINT = PALETTE["mint"]
CORAL = PALETTE["coral"]
AMBER = PALETTE["amber"]
ORANGE = PALETTE["orange"]
CREAM = PALETTE["cream"]

st.set_page_config(
    page_title="Amex Card Tier Migration & Margin Optimization",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {CREAM}; }}
    .main-title {{ font-size: 2.5rem; font-weight: 760; letter-spacing: -0.02em; }}
    .subtitle {{ color: {MUTED}; font-size: 1.05rem; }}
    .metric-card {{
        background: {PALETTE['paper']};
        border: 1px solid {PALETTE['line']};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px rgba(11, 31, 51, 0.04);
    }}
    .risk-badge {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 999px; font-size: 0.75rem;
        font-weight: 750; text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .risk-low {{ background: rgba(42, 157, 143, 0.12); color: #1b6b5f; }}
    .risk-medium {{ background: rgba(233, 196, 106, 0.14); color: #8a6d1b; }}
    .risk-high {{ background: rgba(244, 162, 97, 0.14); color: #9c5a1b; }}
    .risk-critical {{ background: rgba(231, 111, 81, 0.14); color: #9c3e2b; }}
    .tier-badge {{
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 8px; font-size: 0.75rem;
        font-weight: 700; background: {PALETTE['line-strong']}; color: {INK};
    }}
    .action-upsell {{ color: #1b6b5f; font-weight: 650; }}
    .action-retention {{ color: #9c3e2b; font-weight: 650; }}
    .action-maintain {{ color: #4b5e69; font-weight: 650; }}
    </style>
    """,
    unsafe_allow_html=True,
)

bundle = load_model_bundle()
pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
risk_bands = bundle["risk_bands"]
tiers = TIERS
tier_fees = TIER_ANNUAL_FEE


def risk_badge_html(risk_tier: str) -> str:
    css_class = f"risk-{risk_tier.lower()}"
    return f'<span class="risk-badge {css_class}">{risk_tier} risk</span>'


def tier_badge_html(tier: str) -> str:
    return f'<span class="tier-badge">{tier}</span>'


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="main-title">{title}</div><p class="subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Amex Card Migration")
st.sidebar.caption("Premium Card Tier Migration & Margin Optimization")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Customer Tier Scorer",
        "Batch Migration",
        "Margin Optimization",
        "Model Performance",
        "Method & Limits",
    ],
    index=0,
)
st.sidebar.divider()
st.sidebar.metric("Holdout ROC-AUC", f"{bundle['metrics']['roc_auc']:.3f}")
st.sidebar.metric("Decision Threshold", f"{threshold:.3f}")
st.sidebar.metric("Customers", f"{bundle['dataset']['customers']:,}")

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    page_header(
        "Premium Card Tier Migration & Margin Optimization",
        "Identify which card tier maximizes margin for every customer — and when to migrate them.",
    )

    col1, col2, col3, col4 = st.columns(4)
    metrics = bundle["metrics"]
    dataset = bundle["dataset"]

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Portfolio Size</div>'
            f'<div style="font-size:1.75rem; font-weight:700;">{dataset["customers"]:,}</div>'
            f'<div style="color:{MUTED}; font-size:0.75rem;">active premium card customers</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Observed Churn</div>'
            f'<div style="font-size:1.75rem; font-weight:700;">{dataset["churn_rate"]*100:.1f}%</div>'
            f'<div style="color:{MUTED}; font-size:0.75rem;">{dataset["churners"]:,} of {dataset["customers"]:,} customers</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Model ROC-AUC</div>'
            f'<div style="font-size:1.75rem; font-weight:700;">{metrics["roc_auc"]:.3f}</div>'
            f'<div style="color:{MUTED}; font-size:0.75rem;">holdout discrimination</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Avg Margin per Customer</div>'
            f'<div style="font-size:1.75rem; font-weight:700;">${dataset["avg_annual_spend"]/100:.0f}k</div>'
            f'<div style="color:{MUTED}; font-size:0.75rem;">annualized spend per account</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Tier Distribution")
    tier_df = pd.DataFrame(
        list(dataset["tier_distribution"].items()),
        columns=["Tier", "Customers"],
    )
    fig = px.bar(
        tier_df,
        x="Tier",
        y="Customers",
        color="Tier",
        color_discrete_map={
            "No-Fee": PALETTE["muted"],
            "Silver": PALETTE["mint"],
            "Gold": PALETTE["amber"],
            "Platinum": PALETTE["orange"],
            "Centurion": PALETTE["coral"],
        },
    )
    fig.update_layout(showlegend=False, height=350, margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("How It Works")
    st.write(
        "This workbench scores every card customer on (1) churn probability and "
        "(2) the annual margin each card tier would generate. It then recommends "
        "the optimal tier — upsell, maintain, downsell, or retention — and ranks "
        "customers by migration priority."
    )

# ---------------------------------------------------------------------------
# Page: Customer Tier Scorer
# ---------------------------------------------------------------------------

elif page == "Customer Tier Scorer":
    page_header(
        "Customer Tier Scorer",
        "Score a single card customer and see the recommended tier migration and margin impact.",
    )

    # Example loaders
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    examples = bundle.get("examples", [])
    example_map = {e["id"]: e for e in examples}

    with col_ex1:
        if st.button("🟡 Gold → Platinum Upsell Candidate", use_container_width=True):
            st.session_state["scorer_raw"] = example_map.get("gold-upsell", {}).get("raw", {})
    with col_ex2:
        if st.button("🔵 Platinum Churn Risk", use_container_width=True):
            st.session_state["scorer_raw"] = example_map.get("platinum-retention", {}).get("raw", {})
    with col_ex3:
        if st.button("🟢 No-Fee High Spend", use_container_width=True):
            st.session_state["scorer_raw"] = example_map.get("nofee-upsell", {}).get("raw", {})

    if "scorer_raw" not in st.session_state:
        st.session_state["scorer_raw"] = example_map.get("gold-upsell", {}).get("raw", {})

    raw = dict(st.session_state["scorer_raw"])

    with st.form("scorer_form"):
        c1, c2 = st.columns(2)

        with c1:
            raw["card_tier"] = st.selectbox(
                "Card Tier", TIERS, index=TIERS.index(raw.get("card_tier", "Gold"))
            )
            raw["tenure_months"] = st.number_input(
                "Tenure (months)", min_value=0, max_value=240,
                value=int(raw.get("tenure_months", 36)),
            )
            raw["annual_fee_usd"] = st.number_input(
                "Annual Fee (USD)", min_value=0.0, max_value=10000.0,
                value=float(raw.get("annual_fee_usd", tier_fees.get(raw["card_tier"], 250))),
                step=5.0,
            )
            raw["monthly_spend_usd"] = st.number_input(
                "Monthly Spend (USD)", min_value=0.0, max_value=50000.0,
                value=float(raw.get("monthly_spend_usd", 3000.0)),
                step=50.0,
            )
            raw["annual_spend_usd"] = st.number_input(
                "Annual Spend (USD)", min_value=0.0, max_value=1000000.0,
                value=float(raw.get("annual_spend_usd", 36000.0)),
                step=500.0,
            )
            raw["rewards_cost_usd"] = st.number_input(
                "Rewards Cost (USD/yr)", min_value=0.0, max_value=50000.0,
                value=float(raw.get("rewards_cost_usd", 1000.0)),
                step=50.0,
            )
            raw["service_cost_usd"] = st.number_input(
                "Service Cost (USD/yr)", min_value=0.0, max_value=2000.0,
                value=float(raw.get("service_cost_usd", 80.0)),
                step=10.0,
            )

        with c2:
            raw["statement_balance_usd"] = st.number_input(
                "Statement Balance (USD)", min_value=0.0, max_value=500000.0,
                value=float(raw.get("statement_balance_usd", 12000.0)),
                step=100.0,
            )
            raw["utilization_pct"] = st.slider(
                "Utilization (%)", 0.0, 1.0,
                value=float(raw.get("utilization_pct", 0.4)),
                step=0.01,
            )
            raw["delinquency_rate"] = st.slider(
                "Delinquency Rate", 0.0, 0.5,
                value=float(raw.get("delinquency_rate", 0.02)),
                step=0.005,
            )
            raw["credit_score"] = st.number_input(
                "Credit Score", min_value=300, max_value=850,
                value=int(raw.get("credit_score", 720)),
            )
            raw["income_usd"] = st.number_input(
                "Income (USD)", min_value=0, max_value=5000000,
                value=int(raw.get("income_usd", 100000)),
                step=5000,
            )
            raw["age"] = st.number_input(
                "Age", min_value=18, max_value=100,
                value=int(raw.get("age", 40)),
            )

        c3, c4 = st.columns(2)
        with c3:
            raw["gender"] = st.selectbox(
                "Gender", ["M", "F", "Non-binary"],
                index=["M", "F", "Non-binary"].index(raw.get("gender", "M")),
            )
            raw["region"] = st.selectbox(
                "Region", ["Northeast", "Southeast", "Midwest", "West", "Southwest"],
                index=["Northeast", "Southeast", "Midwest", "West", "Southwest"].index(
                    raw.get("region", "West")
                ),
            )
            raw["acquisition_channel"] = st.selectbox(
                "Acquisition Channel",
                ["Online", "Direct Mail", "Branch", "Partner", "Referral"],
                index=["Online", "Direct Mail", "Branch", "Partner", "Referral"].index(
                    raw.get("acquisition_channel", "Online")
                ),
            )
            raw["has_premium_rewards"] = st.selectbox(
                "Has Premium Rewards", ["Yes", "No"],
                index=["Yes", "No"].index(raw.get("has_premium_rewards", "No")),
            )
            raw["complaint_flag"] = st.selectbox(
                "Complaint Flag", ["Yes", "No"],
                index=["Yes", "No"].index(raw.get("complaint_flag", "No")),
            )
        with c4:
            raw["travel_spend_pct"] = st.slider(
                "Travel Spend (%)", 0.0, 1.0,
                value=float(raw.get("travel_spend_pct", 0.15)),
                step=0.01,
            )
            raw["dining_spend_pct"] = st.slider(
                "Dining Spend (%)", 0.0, 1.0,
                value=float(raw.get("dining_spend_pct", 0.1)),
                step=0.01,
            )
            raw["grocery_spend_pct"] = st.slider(
                "Grocery Spend (%)", 0.0, 1.0,
                value=float(raw.get("grocery_spend_pct", 0.05)),
                step=0.01,
            )
            raw["digital_engagement_score"] = st.slider(
                "Digital Engagement (0-1)", 0.0, 1.0,
                value=float(raw.get("digital_engagement_score", 0.6)),
                step=0.01,
            )
            raw["support_contacts_last_12m"] = st.number_input(
                "Support Contacts (12m)", min_value=0, max_value=50,
                value=int(raw.get("support_contacts_last_12m", 2)),
            )

        submitted = st.form_submit_button("Score Customer & Recommend Tier", use_container_width=True)

    if submitted:
        try:
            result = score_raw(raw, bundle)
            features = result["features"]
            prob = result["rawProbability"]
            band = result["riskTier"]

            st.divider()
            st.subheader("Score Result")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Churn Probability", f"{prob*100:.1f}%")
            with m2:
                st.markdown(risk_badge_html(band), unsafe_allow_html=True)
            with m3:
                flagged = "FLAGGED" if result["flagged"] else "Below threshold"
                st.write(f"Policy: {flagged} (threshold {threshold:.3f})")

            rec = result["recommendation"]
            st.subheader("Tier Recommendation")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Current Tier", rec["current_tier"])
            with r2:
                st.metric("Optimal Tier", rec["optimal_tier"])
            with r3:
                st.metric("Expected Margin Lift", f"${rec['expected_margin_lift']:.0f}/yr")

            action = rec["recommended_action"]
            action_type = rec["action_type"]
            css = {
                "upsell": "action-upsell",
                "retention": "action-retention",
                "maintain": "action-maintain",
            }.get(action_type, "action-maintain")
            st.markdown(
                f'<p class="{css}">▸ {html.escape(action)}</p>',
                unsafe_allow_html=True,
            )

            st.subheader("Margin by Tier")
            margin_df = pd.DataFrame(
                [
                    {"Tier": t, "Expected Annual Margin ($)": rec["tier_margins"][t]}
                    for t in TIERS
                ]
            )
            fig = px.bar(
                margin_df,
                x="Tier",
                y="Expected Annual Margin ($)",
                color="Tier",
                color_discrete_map={
                    "No-Fee": PALETTE["muted"],
                    "Silver": PALETTE["mint"],
                    "Gold": PALETTE["amber"],
                    "Platinum": PALETTE["orange"],
                    "Centurion": PALETTE["coral"],
                },
            )
            fig.update_layout(showlegend=False, height=320, margin=dict(t=30, b=40, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Migration Priority")
            priority = min(
                100,
                max(
                    0,
                    int(
                        (rec["expected_margin_lift"] / 500) * 40
                        + prob * 30
                        + (1 - features["digital_engagement_score"]) * 20
                        + (1 if features["complaint_flag"] == "Yes" else 0) * 10
                    ),
                ),
            )
            st.progress(priority / 100)
            st.caption(f"Priority score: {priority}/100 — rank customers by this for campaign outreach.")

            st.session_state["last_result"] = result
            st.session_state["last_raw"] = raw

        except Exception as exc:  # noqa: BLE001
            st.error(f"Scoring failed: {exc}")

# ---------------------------------------------------------------------------
# Page: Batch Migration
# ---------------------------------------------------------------------------

elif page == "Batch Migration":
    page_header(
        "Batch Migration Scoring",
        "Upload a customer file, validate every row, score in the browser-equivalent pipeline, and export a ranked migration list.",
    )

    uploaded = st.file_uploader("Upload CSV (card portfolio schema)", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            missing = [c for c in RAW_FEATURE_COLUMNS if c not in df_in.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                prepared = prepare_inference_data(df_in)
                probs = pipeline.predict_proba(prepared)[:, 1]
                results = []
                for i, row in df_in.iterrows():
                    raw_row = row.to_dict()
                    result = score_raw(raw_row, bundle)
                    rec = result["recommendation"]
                    results.append(
                        {
                            "customerID": raw_row.get(ID_COLUMN, f"row-{i+1}"),
                            "churn_probability": result["probability"],
                            "risk_tier": result["riskTier"],
                            "current_tier": rec["current_tier"],
                            "optimal_tier": rec["optimal_tier"],
                            "expected_margin_lift_usd": rec["expected_margin_lift"],
                            "recommended_action": rec["recommended_action"],
                            "migration_priority": min(
                                100,
                                max(
                                    0,
                                    int(
                                        (rec["expected_margin_lift"] / 500) * 40
                                        + result["rawProbability"] * 30
                                        + (1 - result["features"]["digital_engagement_score"]) * 20
                                    ),
                                ),
                            ),
                        }
                    )
                out_df = pd.DataFrame(results).sort_values(
                    "migration_priority", ascending=False
                )
                st.success(f"Scored {len(out_df)} customers.")
                st.dataframe(out_df.head(25), use_container_width=True)

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored migration list",
                    data=csv,
                    file_name="amex_tier_migration_scored.csv",
                    mime="text/csv",
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Batch scoring failed: {exc}")
    else:
        st.info("Upload a CSV with the card portfolio schema to begin batch scoring.")
        with st.expander("Required schema"):
            st.code(", ".join(RAW_FEATURE_COLUMNS))

# ---------------------------------------------------------------------------
# Page: Margin Optimization
# ---------------------------------------------------------------------------

elif page == "Margin Optimization":
    page_header(
        "Margin Optimization",
        "Simulate how tier changes affect annual margin across the portfolio.",
    )

    st.write(
        "Adjust fee and economics assumptions to see how the optimal tier "
        "assignment and expected portfolio margin shift."
    )

    c1, c2 = st.columns(2)
    with c1:
        interchange_rate = st.slider(
            "Interchange Rate", 0.005, 0.030,
            value=float(INTERCHANGE_RATE),
            step=0.001,
            format="%.3f",
        )
    with c2:
        interest_rate = st.slider(
            "Revolving Interest Rate", 0.05, 0.20,
            value=float(INTEREST_RATE),
            step=0.01,
            format="%.2f",
        )

    from card.modeling import compute_annual_margin

    df = load_portfolio()
    prepared = prepare_inference_data(df)
    probs = pipeline.predict_proba(prepared)[:, 1]

    # Current margins at current tiers
    current_margins = []
    optimal_margins = []
    for _, row in df.iterrows():
        cur = compute_annual_margin(row, row["card_tier"])
        best = max(compute_annual_margin(row, t) for t in TIERS)
        current_margins.append(cur)
        optimal_margins.append(best)

    total_current = sum(current_margins)
    total_optimal = sum(optimal_margins)
    lift = total_optimal - total_current

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Current Portfolio Margin", f"${total_current:,.0f}/yr")
    with m2:
        st.metric("Optimized Portfolio Margin", f"${total_optimal:,.0f}/yr")
    with m3:
        st.metric("Margin Lift Opportunity", f"${lift:,.0f}/yr")

    st.divider()
    st.subheader("Margin Distribution by Tier")
    margin_df = pd.DataFrame(
        {
            "Tier": df["card_tier"],
            "Current Margin": current_margins,
            "Optimal Margin": optimal_margins,
        }
    )
    fig = px.box(
        margin_df.melt(id_vars="Tier", value_name="Margin", var_name="Scenario"),
        x="Tier",
        y="Margin",
        color="Scenario",
    )
    fig.update_layout(height=400, margin=dict(t=30, b=40, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

elif page == "Model Performance":
    page_header(
        "Model Performance",
        "Replay the holdout evaluation and test the migration threshold policy.",
    )

    test_path = Path("reports/evaluation/test_predictions.csv")
    if test_path.exists():
        preds = pd.read_csv(test_path)
    else:
        st.warning("No holdout predictions found. Run training first.")
        preds = None

    if preds is not None:
        col_thresh, col_metric = st.columns([1, 2])
        with col_thresh:
            user_threshold = st.slider(
                "Churn threshold", 0.05, 0.95,
                value=float(threshold),
                step=0.01,
            )
        with col_metric:
            tm = threshold_metrics(bundle, user_threshold)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Contacts", f"{tm['contacts']:,}")
            with c2:
                st.metric("Precision", f"{tm['precision']:.1%}")
            with c3:
                st.metric("Recall", f"{tm['recall']:.1%}")
            with c4:
                st.metric("F1", f"{tm['f1']:.3f}")

        st.divider()
        st.subheader("Precision-Recall Trade-off Curve")
        curve = tradeoff_curve(bundle, 0.05, 0.95, 41)
        curve_df = pd.DataFrame(curve)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve_df["threshold"],
                y=curve_df["precision"],
                mode="lines",
                name="Precision",
                line=dict(color=TEAL, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve_df["threshold"],
                y=curve_df["recall"],
                mode="lines",
                name="Recall",
                line=dict(color=CORAL, width=2),
            )
        )
        fig.add_vline(x=user_threshold, line_dash="dash", line_color=INK)
        fig.update_layout(
            xaxis_title="Threshold",
            yaxis_title="Rate",
            height=380,
            margin=dict(t=30, b=40, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion Matrix at Selected Threshold")
        cm = confusion_matrix(preds["actual_churn"], (preds["churn_probability"] >= user_threshold).astype(int))
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Retained", "Actual: Churned"],
            columns=["Predicted: Retained", "Predicted: Churned"],
        )
        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Teal",
        )
        fig.update_layout(height=320, margin=dict(t=30, b=40, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python -m card.train --generate-data` to generate data and train the model.")

# ---------------------------------------------------------------------------
# Page: Method & Limits
# ---------------------------------------------------------------------------

elif page == "Method & Limits":
    page_header(
        "Method & Limits",
        "How the tier migration model works, and where it should not be used.",
    )

    st.subheader("Model")
    st.write(
        "A logistic regression pipeline predicts churn probability from 26 engineered "
        "features (spend, fees, rewards cost, service cost, credit score, utilization, "
        "delinquency, demographics, and digital engagement). The threshold is tuned to "
        "maximize F1 on out-of-fold cross-validation predictions and evaluated once on "
        "a holdout set."
    )

    st.subheader("Margin Economics")
    st.write(
        "Annual margin per tier = (annual fee + interchange + interest revenue) "
        "− (rewards cost + service cost + expected loss). The recommended tier is the one "
        "with the highest risk-adjusted margin."
    )

    st.subheader("Limitations")
    st.write(
        "- This is a demonstration built on synthetic data. Replace `data/raw/card_portfolio.csv` "
        "with real customer data before production use.\n"
        "- Margin estimates are directional, not exact P&L.\n"
        "- No causal claim: migrating a customer to a premium tier does not guarantee higher margin.\n"
        "- Fairness review is required before using tier, income, or credit score in customer decisions.\n"
        "- Do not use this model for credit underwriting, pricing, or eligibility decisions."
    )

    st.subheader("Reproduction")
    st.code(
        "git clone https://github.com/mlaharsh/amex-tier-migration\n"
        "cd amex-tier-migration\n"
        "python -m venv .venv\n"
        ".venv\\Scripts\\activate\n"
        "pip install -r requirements-dev.txt\n"
        "python -m card.train --generate-data\n"
        "streamlit run app.py"
    )
