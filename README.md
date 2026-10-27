# Amex Premium Card Tier Migration & Margin Optimization

[![CI](https://github.com/mlaharsh/amex-tier-migration/actions/workflows/ci.yml/badge.svg)](https://github.com/mlaharsh/amex-tier-migration/actions/workflows/ci.yml)
[![Showcase](https://img.shields.io/badge/showcase-live-0E7C86?style=flat-square)](https://mlaharsh.github.io/amex-tier-migration/)
[![Live scorer](https://img.shields.io/badge/live%20scorer-try%20it-2A9D8F?style=flat-square)](https://mlaharsh.github.io/amex-tier-migration/scorer.html)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-0B1F33?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-workbench-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](docs/deployment.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-0B1F33?style=flat-square)](LICENSE)

A portfolio project for **American Express Premium Card Tier Migration & Margin Optimization**. It combines a reproducible scikit-learn pipeline, a public client-side scorer, a multi-page Streamlit workbench, evaluation reports, tests, CI, Docker, and a static GitHub Pages showcase.

The goal is to answer two questions for every premium card customer:

1. **What is the churn risk?** — a probability ranking from a logistic-regression pipeline.
2. **Which tier maximizes annual margin?** — fee + interchange + interest revenue minus rewards, service, and expected-loss cost.

The model then recommends an action: **upsell**, **maintain**, **downsell**, **retention**, or **monitor** — and ranks customers by migration priority.

- **Live showcase:** https://mlaharsh.github.io/amex-tier-migration/
- **Live scorer:** https://mlaharsh.github.io/amex-tier-migration/scorer.html — the exported model running client-side; no login or server
- **Native Streamlit app:** run locally with `streamlit run app.py` (deployment guide in [`docs/deployment.md`](docs/deployment.md))
- **Model card:** [`docs/model_card.md`](docs/model_card.md)

## Results at a glance

| Metric | Result |
| --- | --- |
| Portfolio | 25,000 synthetic premium-card customers |
| Selected model | Logistic regression |
| Selection rule | Best 5-fold stratified CV ROC-AUC |
| Holdout ROC-AUC | ~0.82 |
| Decision threshold | tuned on out-of-fold predictions |
| Tiers | No-Fee → Silver → Gold → Platinum → Centurion |

## Why this project

- **One deterministic pipeline** from synthetic portfolio to a serialized model, metrics, and test predictions.
- **Leakage-aware evaluation**: preprocessing lives inside the model pipeline, the threshold is tuned on out-of-fold training predictions, and the holdout set is used once.
- **Margin-aware recommendations**: every tier is scored on expected annual margin, not just churn risk.
- **Two deployment surfaces**: a Streamlit workbench and a public client-side scorer on GitHub Pages.
- **Honest limitations**: synthetic data, directional margin estimates, no causal claims, and explicit responsible-use notes.

## The product

| Page | What it does |
| --- | --- |
| **Overview** | Portfolio KPIs, tier distribution, and how the model works. |
| **Customer Tier Scorer** | Scores a single account, shows churn probability, risk tier, recommended tier, margin lift, and migration priority. |
| **Batch Migration** | Uploads a CSV, validates the schema, scores every customer, and exports a ranked migration list. |
| **Margin Optimization** | Simulates fee and interest assumptions to see portfolio-level margin opportunity. |
| **Model Performance** | Replays the holdout evaluation, precision-recall trade-off, and confusion matrix. |
| **Method & Limits** | Model card, data dictionary, reproduction steps, and responsible-use notes. |

## Quickstart

### Local environment

```bash
git clone https://github.com/mlaharsh/amex-tier-migration.git
cd amex-tier-migration
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m card.train --generate-data
streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m card.train --generate-data
streamlit run app.py
```

Open `http://localhost:8501`. Training takes about a minute and rewrites `models/` and `reports/`.

### Tests

```bash
python -m pytest -q
python -m card.browser_export --check
node tests/browser_scorer.test.js
```

### Docker

```bash
docker build -t amex-tier-migration .
docker run --rm -p 8501:8501 amex-tier-migration
```

Open `http://localhost:8501`.

## Repository layout

```text
.
├── app.py                         # Streamlit workbench
├── card/
│   ├── __init__.py
│   ├── config.py                  # tiers, feature lists, margin economics
│   ├── data.py                    # synthetic portfolio generation & loading
│   ├── features.py                # feature engineering & preprocessor
│   ├── modeling.py                # churn model, tier recommendation, serialization
│   ├── scoring.py                 # scoring, threshold, margin helpers
│   ├── browser_export.py          # exports fitted parameters for static scorer
│   └── train.py                   # reproducible CLI entry point
├── data/raw/card_portfolio.csv    # generated synthetic portfolio
├── models/
│   ├── card_tier_model.joblib     # deployable bundle
│   ├── metrics.json               # generated metrics
│   └── feature_importance.csv
├── reports/
│   ├── evaluation/                # test predictions
│   └── figures/                   # generated charts
├── notebooks/                     # exploratory analysis
├── tests/
│   └── browser_scorer.test.js     # Node/Python parity checks
├── docs/
│   ├── index.html                 # static GitHub Pages showcase
│   ├── scorer.html                # live client-side scorer
│   ├── scorer.js                  # scoring, threshold, margin logic
│   ├── scorer_ui.js               # form, result, and batch UI glue
│   ├── scorer.css                 # responsive scorer styling
│   ├── scorer_data.js             # generated model/preprocessing/holdout payload
│   ├── model_card.md
│   ├── data_dictionary.md
│   └── deployment.md
├── .github/workflows/
│   ├── ci.yml                     # tests, browser parity, Streamlit smoke test
│   └── pages.yml                  # GitHub Pages deployment
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Deployment

### GitHub Pages (static showcase + scorer)

The showcase and client-side scorer are deployed from `docs/` by `.github/workflows/pages.yml`.

- Showcase: https://mlaharsh.github.io/amex-tier-migration/
- Live scorer: https://mlaharsh.github.io/amex-tier-migration/scorer.html

### Streamlit Community Cloud

The app is designed for zero-secret deployment:

1. Push the repository to GitHub.
2. Sign in to share.streamlit.io with GitHub.
3. Create a new app from `mlaharsh/amex-tier-migration`, branch `master`, main file `app.py`.
4. Select Python 3.11 and deploy.

No environment variables are required.

## Limitations and responsible use

- **Synthetic data.** This project uses a generated portfolio. Real Amex customer data must replace `data/raw/card_portfolio.csv` before any production use.
- **Directional margins.** Annual margin estimates are illustrative, not exact P&L.
- **No causal claims.** Recommending a tier migration does not guarantee margin improvement.
- **Fairness review required.** Tier, income, credit score, and demographics should be audited before customer-facing use.
- **Not for underwriting.** Do not use this model for credit decisions, pricing, or eligibility.

## License

MIT. See [`LICENSE`](LICENSE).
