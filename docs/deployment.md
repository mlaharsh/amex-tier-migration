# Deployment guide

This project has two complementary deployment surfaces:

1. **GitHub Pages** — a static showcase and a client-side card tier migration scorer built from `docs/`. They are public, fast, and require no server.
2. **Streamlit** — the full native workbench (`app.py`). It can run locally, in Docker, or on Streamlit Community Cloud.

The GitHub Pages showcase itself is documentation and a visual tour. The **live scorer** at `/scorer.html` is a real model endpoint implemented in JavaScript: it loads the coefficients exported from the fitted pipeline and reproduces the Python preprocessing exactly. The Streamlit app remains the full reference implementation and loads the serialized model bundle without retraining.

## 1. GitHub Pages (static showcase + live scorer)

### One-time repository setup

1. Open **GitHub → repository → Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Ensure the default branch is `master` (this repository keeps `master`).
4. Run the **Deploy showcase** workflow from the Actions tab, or push a change under `docs/`.

The workflow is [`.github/workflows/pages.yml`](../.github/workflows/pages.yml). It:

- uploads the `docs/` folder with `actions/upload-pages-artifact`;
- deploys it with `actions/deploy-pages`;
- publishes the site at `https://mlaharsh.github.io/amex-tier-migration/`.

### Verify

```bash
gh run list --workflow pages.yml --limit 5
gh run watch <run-id>
curl -I https://mlaharsh.github.io/amex-tier-migration/
curl -I https://mlaharsh.github.io/amex-tier-migration/scorer.html
```

Expected: HTTP 200 for both and a page titled **Amex Card Tier Migration & Margin Optimization** for the showcase.

### Browser scorer (live)

`docs/scorer.html` is the public interactive surface. It has no backend, no login, and no secrets:

- `docs/scorer_data.js` contains the exported intercept, one-hot coefficients, imputer medians, scaler parameters, tier economics, validation cases, and the saved holdout arrays.
- `docs/scorer.js` applies the same derived features (spend-to-fee ratio, spend per tenure, margin proxy, premium tier flag), median imputation, standard scaling, and unknown-category handling as the Python pipeline.
- `docs/scorer_ui.js` renders the single-customer form, threshold lab, and CSV batch scorer.
- The threshold lab replays the saved out-of-sample holdout predictions. It does **not** refit or tune on the holdout set.

Regenerate the artifact after any model or data change, then commit the result:

```bash
python -m card.browser_export
python -m card.browser_export --check   # fails if docs/scorer_data.js is stale
node tests/browser_scorer.test.js        # JavaScript/Python parity checks
```

The export is deterministic. `docs/scorer_data.js` must stay in sync with `models/card_tier_model.joblib` and `reports/evaluation/test_predictions.csv`; CI enforces this.

This scorer is a portfolio demo, not a production decision system. It runs on synthetic Amex-style data and should not receive real customer records.

### Update the showcase

Edit `docs/index.html` and the images under `docs/assets/`. Push to `master`. The workflow path filter only runs when `docs/**` or the workflow file changes.

### Optional CLI setup

If Pages is not enabled yet, an authenticated owner can enable it through the API:

```bash
gh api repos/mlaharsh/amex-tier-migration/pages \
  -X POST \
  -f build_type=workflow
```

Then run the Pages workflow. If Pages already exists, use `-X PUT` instead of `-X POST`.

## 2. Streamlit Community Cloud

The app is designed to deploy without secrets or a database. The model, metrics, and evaluation CSVs are committed to the repository.

### Step by step

1. Push the repository to GitHub (branch `master`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Select **New app**.
4. Choose:
   - **Repository:** `mlaharsh/amex-tier-migration`
   - **Branch:** `master`
   - **Main file path:** `app.py`
   - **Python version:** 3.11 (recommended; 3.12 is also tested)
5. Deploy.

The app installs `requirements.txt`, loads `models/card_tier_model.joblib`, and starts on the Streamlit domain. No environment variables are required.

### Important status note

The repository includes a working local app and a verified CI smoke test. A public Streamlit Community Cloud URL is **not asserted in the README** unless a live link is added. If you deploy it, add the URL to the README and `docs/index.html` so the claim stays honest.

### If deployment fails

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Model artifact not found` | `models/card_tier_model.joblib` is missing from the branch | Run `python -m card.train --generate-data` locally and commit `models/` |
| `joblib` unpickle warning / error | scikit-learn version mismatch | Use the version range in `requirements.txt`; retrain if necessary |
| App starts but data pages fail | `reports/evaluation/test_predictions.csv` missing | Re-run training and commit `reports/evaluation/` |
| Build times out during install | heavy dependency resolution | Pin the versions from the tested environment or use Docker |

## 3. Docker

Build and run:

```bash
docker build -t amex-tier-migration .
docker run --rm -p 8501:8501 amex-tier-migration
```

Open http://localhost:8501.

The image uses Python 3.11 slim, installs `libgomp1` for scikit-learn, and includes a health check against `/_stcore/health`.

## 4. Local deployment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m card.train --generate-data
streamlit run app.py
```

The app runs at http://localhost:8501.

## 5. CI

`.github/workflows/ci.yml` runs on pushes to `master`, pull requests, and manual dispatch:

- **test** — Python 3.11 and 3.12, compile check, pytest;
- **browser-scorer** — freshness check for `docs/scorer_data.js`, JavaScript syntax checks, and Node/Python parity tests;
- **train** — data generation, model training, and artifact verification;
- **app-smoke** — starts Streamlit and checks the health endpoint.

## 6. Operational checklist

- [ ] `python -m pytest -q` passes.
- [ ] `python -m card.browser_export --check` passes and `node tests/browser_scorer.test.js` passes.
- [ ] `python -m card.train --generate-data` completes and regenerates artifacts.
- [ ] `streamlit run app.py` loads all six pages.
- [ ] Batch upload and template download work in both the Streamlit app and the browser scorer.
- [ ] GitHub Actions CI is green.
- [ ] Pages workflow is green and both the showcase and `/scorer.html` return HTTP 200.
- [ ] Record the model version, threshold, and date with every exported campaign list.

## 7. Security and privacy

- The app requires no credentials and stores no customer data.
- Uploaded batch files are processed in memory by the Streamlit session; they are not written to disk by this project.
- The browser scorer reads CSV files locally in the page and never uploads them; the download is generated as a client-side blob.
- Do not upload real customer data to a public demo app. Use anonymised extracts.
- Review the model card before using scores in a customer-facing process.
