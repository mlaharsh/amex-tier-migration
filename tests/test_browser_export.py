"""Regression tests for the card tier migration browser-scorer export.

The GitHub Pages scorer is only honest if it keeps using the *same* fitted
parameters as the Python pipeline. These tests pin the export to the committed
model bundle and evaluation artifacts, and fail when ``docs/scorer_data.js`` is
stale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from card.browser_export import (
    BROWSER_DATA_PATH,
    EXAMPLES,
    build_browser_artifact,
    render_browser_data,
)
from card.config import (
    CATEGORICAL_FEATURES as _CAT_FEATURES,
    FEATURES,
    NUMERIC_FEATURES as _NUM_FEATURES,
    TEST_PREDICTIONS_PATH,
)


def test_exported_parameters_match_the_fitted_pipeline(bundle: dict) -> None:
    artifact = build_browser_artifact(bundle=bundle)
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    assert artifact["metadata"]["threshold"] == bundle["threshold"]
    assert artifact["metadata"]["selectedModel"] == bundle["selected_model_display"]
    assert artifact["intercept"] == float(classifier.intercept_[0])
    assert len(artifact["numeric"]) + len(artifact["categorical"]) == len(FEATURES)

    numeric_transformer = preprocessor.named_transformers_["numeric"]
    imputer = numeric_transformer.named_steps["imputer"]
    scaler = numeric_transformer.named_steps["scaler"]
    numeric_by_name = {feature["name"]: feature for feature in artifact["numeric"]}
    assert list(numeric_by_name) == list(_NUM_FEATURES)
    for index, name in enumerate(numeric_transformer.feature_names_in_):
        exported = numeric_by_name[name]
        assert exported["median"] == float(imputer.statistics_[index])
        assert exported["mean"] == float(scaler.mean_[index])
        assert exported["scale"] == float(scaler.scale_[index])

    categorical_transformer = preprocessor.named_transformers_["categorical"]
    onehot = categorical_transformer.named_steps["onehot"]
    from card.config import CATEGORICAL_FEATURES as _CAT_FEATURES
    categorical_by_name = {
        feature["name"]: feature for feature in artifact["categorical"]
    }
    coefficient_map = dict(
        zip(
            (str(name) for name in preprocessor.get_feature_names_out()),
            np.asarray(classifier.coef_, dtype=float).reshape(-1),
            strict=True,
        )
    )
    for feature, categories in zip(
        list(_CAT_FEATURES),
        onehot.categories_,
        strict=True,
    ):
        exported = categorical_by_name[str(feature)]
        assert set(exported["coefficients"]) == {str(value) for value in categories}
        for category in categories:
            assert (
                exported["coefficients"][str(category)]
                == coefficient_map[f"{feature}_{category}"]
            )


def test_validation_cases_reproduce_python_probabilities(bundle: dict) -> None:
    artifact = build_browser_artifact(bundle=bundle)
    pipeline = bundle["pipeline"]
    assert len(artifact["validation"]) == len(EXAMPLES)

    for validation in artifact["validation"]:
        frame = pd.DataFrame([validation["features"]], columns=FEATURES)
        expected = float(pipeline.predict_proba(frame)[0, 1])
        assert validation["expectedProbability"] == pytest.approx(expected, abs=1e-12)
        assert 0.0 <= validation["expectedProbability"] <= 1.0


def test_holdout_replay_matches_saved_evaluation() -> None:
    artifact = build_browser_artifact()
    holdout = artifact["holdout"]
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)

    assert holdout["n"] == len(predictions)
    assert holdout["churners"] == int(predictions["actual_churn"].sum())
    assert len(holdout["probabilities"]) == holdout["n"]
    assert len(holdout["labels"]) == holdout["n"]
    assert holdout["probabilities"] == [
        round(float(value), 4) for value in predictions["churn_probability"]
    ]
    assert holdout["labels"] == "".join(
        str(int(value)) for value in predictions["actual_churn"]
    )


def test_committed_browser_data_is_current() -> None:
    expected = render_browser_data(build_browser_artifact())
    assert BROWSER_DATA_PATH.exists(), "run `python -m card.browser_export`"
    assert BROWSER_DATA_PATH.read_text(encoding="utf-8") == expected, (
        "docs/scorer_data.js is stale. Run `python -m card.browser_export` "
        "and commit the regenerated file."
    )
